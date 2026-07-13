# tcaa/fl_runner.py
# Multi-round federated-learning driver for TCAA (the non-toy protocol).
#
# The Phase-0 runner does a SINGLE FL round, which cannot show the two things the FL
# poisoning literature cares about most: (1) does the attack PERSIST / accumulate as
# benign updates dilute it over many rounds (durability, cf. Neurotoxin), and (2) does
# the malicious update stay inside the benign stealth envelope EVERY round under client
# sampling. This driver runs T rounds with client sampling and, each round:
#   * a sampled subset of benign clients fine-tunes from the broadcast global (LM CE),
#   * sampled attacker(s) optimize the ALM-constrained length loss (reusing the exact,
#     already-tested tcaa attacker in phase0_runner._malicious_update),
#   * the server does weighted FedAvg.
# A parallel BENIGN-ONLY global is run in lockstep as the amplification baseline. Cost /
# utility / stealth are measured every `measure_every` rounds to produce a durability
# curve.  Aggregation follows the same weighted-FedAvg convention as AugMP's server,
# re-implemented in tcaa/stealth.py (no AugMP code is imported).
#
# CPU smoke:  python -m tcaa.fl_runner --smoke
# Real (GPU): python -m tcaa.fl_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .cost_model import (DEFAULT_DECENSOR_MAX_EXTRA, amplification_ratio,
                        calibrate_coefficients)
from .gen_data import partition_examples
from .run_paths import stamp_run_subdir
from .phase0_runner import (_benign_update, _malicious_update, _measure_cost, _ppl,
                            _set_seed, _validate_decoder_only, build_model_and_data,
                            default_config, enable_backend_speedups)
from .stealth import (evaluate_stealth, pairwise_mean_cosine, update_cosine,
                     update_distance, weighted_fedavg_reference)


def default_fl_config() -> Dict:
    """Multi-round protocol on top of the Phase-0 defaults (Qwen2.5-0.5B + Alpaca)."""
    cfg = default_config()
    cfg.update({
        "experiment_name": "tcaa_fl",
        # --- FL topology (single-A100 scale) ---
        "num_clients": 7, "num_attackers": 2,      # 5 benign + 2 (coordinated) attackers
        "num_rounds": 50,
        "clients_per_round": 5,                    # sample 5 of 7 each round (FedAvg-style)
        "attacker_always_selected": False,         # realistic: attackers sampled like everyone
        "track_benign_baseline": True,             # run a parallel benign-only global for C_ben
        "measure_every": 5,                        # measure cost/utility/stealth every K rounds
        # Generation cap for MEASUREMENT ONLY (does not affect the attack, which trains EOS
        # suppression only up to onpolicy_horizon = 256; see phase0_runner). Chosen as a
        # runtime-aware balance: 2048 = 8x the attack horizon, enough headroom above what the
        # attack directly optimizes to (a) show the full amplified length and (b) reveal whether
        # the degenerate tail ever terminates (truncation < 1) — while generation wall-clock is
        # ~linear in the cap, so 2048 is ~2x the old 1024, not the 4x of 4096. NOTE: when
        # truncation stays ~1, BOTH raw amp and vs-pristine scale with this cap, so lead on the
        # cap-robust EFFECTIVE amplification (repetition-discounted length plateaus even as the
        # loop runs longer). Raise to 4096 for a one-off length-distribution diagnostic; if too
        # slow, also raise measure_every / lower eval_size rather than shrinking the cap.
        "max_new_tokens": 2048,
        # --- moderately larger data than Phase-0's 512 / 64 ---
        "pool_size": 4000, "eval_size": 256,
        # --- per-round attacker budget (smaller than single-round; it repeats each round) ---
        "attacker_steps": 100,
        # Clean-KD utility floor ON for multi-round: without it the attacker's per-round
        # EOS suppression compounds and clean perplexity drifts up over rounds (the single
        # biggest gap to the "utility-preserving" claim). 1.0 is a starting point; if the
        # per-round ppl_ratio still climbs, raise it (2/4); if amplification collapses,
        # lower it. See phase0_runner.default_config for the mechanism note.
        "kd_clean_weight": 1.0,
        # Two-sided clean length anchor: hold clean length AT the pristine baseline (both
        # directions), not just cap it from above. Fixes clean cost drifting BELOW baseline
        # (amp_clean < 1 / clean ROUGE drop) that the one-sided hinge permits.
        "clean_anchor_two_sided": True,
        # --- cost-model credibility: also report amplification under physically-calibrated
        # coefficients (c_f = d_model, c_a = 1). The naive c_f = c_a = 1 puts the super-linear
        # threshold at L ~ 2n+2 (quadratic term credited from the first token); the calibrated
        # ratio (c_a/c_f ~ 1/d_model) is what a real Qwen2.5-0.5B decoder actually incurs.
        "report_calibrated_cost": True,
        # --- per-round defense telemetry: log every participating client's update geometry
        # (distance + cosine family, pairwise matrices, Krum score) labelled benign/attacker,
        # so a defense implemented LATER can be replayed offline to test attacker detection.
        # Aggregation stays plain FedAvg — this only OBSERVES, it never rejects.
        "collect_defense_telemetry": True,
        "save_update_vectors": False,   # also dump raw per-client update vectors (large; for FLTrust / coord-wise defenses)
        # --- multi-attacker coordination: decorrelate colluding attackers so their mutual
        # (distance, cosine) stays inside the benign PAIR band (not detectable near-twins).
        # 0 disables; tune against amplification (higher = stealthier pair, weaker attack).
        "gamma_coord": 10.0,
        # --- two-sided cosine stealth constraint: constrain the attacker's cosine to the benign
        # envelope to the BAND [stealth_cos_low, upper], not just the AugMP lower bound. Closes
        # the "over-aligned outlier" tell (attacker cos_to_agg >> benign) the one-sided constraint
        # permits. Distance stays one-sided (upper bound only, which is correct). Lower edge is a
        # fixed floor (0 = forbid only anti-alignment); upper edge is the most-aligned benign
        # client's value, computed with the same statistic as the constrained cosine.
        "stealth_cosine_two_sided": True,
        "stealth_cos_low": 0.0,
        # --- norm constraint: bound the attacker update norm to the largest benign norm, so it is
        # not the norm-outlier that norm-clipping / Krum screen on (distance/cosine don't bound
        # norm). This is the top detection tell in the telemetry. On => stealthier attack.
        "stealth_norm_constraint": True,
        # --- offline defense-evasion evaluation: after the FedAvg run, replay Krum/Multi-Krum/
        # norm-clip/cosine-screen on the collected telemetry and report attacker detection rate.
        # Turns the (circular) benign-envelope check into an INDEPENDENT verdict (see defenses.py).
        "run_defense_eval": True,
        "dump_char_cap": 20000,         # keep enough decoded text to SEE the full loop at high cap
        "num_dump_examples": 12,        # more final tau/clean samples for repetition-form inspection
        "results_subdir": "tcaa_fl",
    })
    return cfg


def fl_smoke_overrides() -> Dict:
    """Download-free CPU smoke: tiny-gpt2 + synthetic, a handful of short rounds."""
    from .phase0_runner import smoke_overrides
    cfg = smoke_overrides()
    cfg.update({
        "experiment_name": "tcaa_fl_smoke",
        "num_clients": 7, "num_attackers": 2, "num_rounds": 6,
        "clients_per_round": 5, "attacker_always_selected": False,
        "track_benign_baseline": True, "measure_every": 2,
        "pool_size": 300, "eval_size": 32, "attacker_steps": 20, "warmup_steps": 200,
        "kd_clean_weight": 1.0,  # exercise the clean-KD (disable_adapter) path on CPU
        "results_subdir": "tcaa_fl_smoke",
    })
    return cfg


def _fedavg(updates: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    stacked = torch.stack([u.float() for u in updates])
    w = torch.tensor(weights, dtype=stacked.dtype)
    w = w / w.sum()
    return (stacked * w.view(-1, 1)).sum(dim=0)


def _collect_defense_telemetry(round_idx, ben_updates, ben_sizes, ben_ids,
                               atk_updates, atk_sizes, atk_ids, *, num_attackers=0):
    """Per-round, per-client update geometry a robust aggregator would screen on — LOGGED,
    not enforced, so a defense implemented later can be replayed OFFLINE to test whether it
    detects the attacker(s). Records the quantities Krum / norm-clipping / cosine screens
    consume: L2 norm, distance & cosine to the (attacker-inclusive) FedAvg aggregate and to the
    benign mean, leave-self-out pairwise-mean cosine, and a Krum score. The full pairwise
    distance & cosine matrices are included so ANY Krum/Multi-Krum variant (its own f, k) can be
    replayed without the raw vectors. Aggregation itself is unchanged (plain FedAvg)."""
    updates = [u.detach().float() for u in ben_updates] + [u.detach().float() for u in atk_updates]
    labels = ["benign"] * len(ben_updates) + ["attacker"] * len(atk_updates)
    ids = list(ben_ids) + list(atk_ids)
    weights = list(ben_sizes) + list(atk_sizes)
    n = len(updates)
    if n == 0:
        return {"round": int(round_idx), "n_clients": 0, "clients": []}
    agg = weighted_fedavg_reference(updates, weights)          # exactly the FedAvg aggregate
    ben_mean = (weighted_fedavg_reference([u.detach().float() for u in ben_updates], ben_sizes)
                if ben_updates else agg)
    # Pairwise distance / cosine matrices (n x n; small — n <= clients_per_round).
    dist_mat = [[0.0] * n for _ in range(n)]
    cos_mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = update_distance(updates[i], updates[j])
            c = update_cosine(updates[i], updates[j])
            dist_mat[i][j] = dist_mat[j][i] = round(d, 6)
            cos_mat[i][j] = cos_mat[j][i] = round(c, 6)
    # Krum score = sum of the (n-f-2) smallest squared distances to other clients (Blanchard 2017).
    f = int(num_attackers)
    k = max(1, n - f - 2)
    clients = []
    for i in range(n):
        sq = sorted((dist_mat[i][j] ** 2 for j in range(n) if j != i))
        others = [updates[j] for j in range(n) if j != i]
        clients.append({
            "client_id": int(ids[i]) if i < len(ids) else i,
            "label": labels[i],
            "norm": round(float(torch.norm(updates[i])), 6),
            "dist_to_agg": round(update_distance(updates[i], agg), 6),
            "dist_to_benign_mean": round(update_distance(updates[i], ben_mean), 6),
            "cos_to_agg": round(update_cosine(updates[i], agg), 6),
            "pairwise_mean_cos": round(pairwise_mean_cosine(updates[i], others), 6),
            "krum_score": round(float(sum(sq[:k])), 6),
        })
    return {
        "round": int(round_idx),
        "n_clients": n, "n_attackers_present": len(atk_updates),
        "krum_neighbors_k": k,
        "clients": clients,
        "pairwise_distance": dist_mat,
        "pairwise_cosine": cos_mat,
    }


def _save_update_vectors(out_dir, round_idx, ben_updates, ben_ids, atk_updates, atk_ids):
    """Dump the round's raw per-client update vectors (compressed .npz) for defenses that need
    the full vectors — FLTrust (server root-cosine), coordinate-wise trimmed-mean/median."""
    vecs, labels, ids = [], [], []
    for u, cid in zip(ben_updates, ben_ids):
        vecs.append(u.detach().float().cpu().numpy()); labels.append("benign"); ids.append(int(cid))
    for u, cid in zip(atk_updates, atk_ids):
        vecs.append(u.detach().float().cpu().numpy()); labels.append("attacker"); ids.append(int(cid))
    if not vecs:
        return
    updates_dir = Path(out_dir) / "update_vectors"
    updates_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(updates_dir / f"round_{int(round_idx):03d}.npz",
                        updates=np.stack(vecs), labels=np.array(labels), client_ids=np.array(ids))


def _sample_participants(
    rng: random.Random, benign_ids: List[int], attacker_ids: List[int],
    clients_per_round: int, attacker_always_selected: bool,
) -> Tuple[List[int], List[int]]:
    """Sample the round's participating clients (FedAvg client sampling)."""
    n = min(clients_per_round, len(benign_ids) + len(attacker_ids))
    if attacker_always_selected:
        sel_atk = list(attacker_ids)
        remaining = max(0, n - len(sel_atk))
        sel_ben = rng.sample(benign_ids, min(remaining, len(benign_ids)))
    else:
        pool = list(benign_ids) + list(attacker_ids)
        chosen = set(rng.sample(pool, n))
        sel_ben = [c for c in benign_ids if c in chosen]
        sel_atk = [c for c in attacker_ids if c in chosen]
    return sorted(sel_ben), sorted(sel_atk)


@torch.no_grad()
def _dump_fl_examples(model, g_flat, tau_ev, clean_ev, tokenizer, cfg, spec, device, k=4):
    """Decode a few (prompt -> output) samples from the FINAL attacked global — the
    qualitative 'is the amplified output long AND coherent (not a loop)' evidence for the
    multi-round state. Returns a list of dicts (also carrying per-sample rep/distinct)."""
    if tokenizer is None or k <= 0:
        return []
    from .cost_model import distinct_ratio, repetition_rate
    from .gen_data import collate_gen
    from .metrics import rouge_l_recall
    model.set_flat_params(g_flat.to(device))
    inner = model.inner(); inner.eval()
    recs = []
    for split, ex in (("tau", tau_ev), ("clean", clean_ev)):
        exk = ex[:k]
        if not exk:
            continue
        batch = collate_gen(exk, spec.pad_id)
        ids, attn = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        gen = inner.generate(input_ids=ids, attention_mask=attn, max_new_tokens=cfg["max_new_tokens"],
                             do_sample=False, num_beams=1, pad_token_id=spec.pad_id, eos_token_id=spec.eos_id)
        P = ids.shape[1]
        for i in range(len(exk)):
            new = gen[i, P:]
            eos_pos = (new == spec.eos_id).nonzero(as_tuple=True)[0]
            L = int(eos_pos[0].item()) + 1 if eos_pos.numel() > 0 else int(new.shape[0])
            out_ids = new[:L].tolist()
            ref = [t for t in exk[i].ref_ids if t != spec.eos_id]
            recs.append({
                "split": split,
                # keep enough text to SEE the full loop in the qualitative viz (was 180/500);
                # dump_char_cap must scale with max_new_tokens or a high cap clips the loop.
                "prompt": tokenizer.decode(exk[i].prompt_ids, skip_special_tokens=True)[:400],
                "output": tokenizer.decode(out_ids, skip_special_tokens=True)[:int(cfg.get("dump_char_cap", 6000))],
                "len": L, "truncated": bool(eos_pos.numel() == 0),
                "rouge_recall": round(rouge_l_recall(out_ids, ref), 3) if ref else None,
                "repetition": round(repetition_rate(out_ids), 3),
                "distinct": round(distinct_ratio(out_ids, 4), 3),
            })
    return recs


def run_fl(config: Dict) -> Dict:
    cfg = default_fl_config()
    cfg.update(config or {})
    cfg = stamp_run_subdir(cfg)   # unique run folder so reruns never overwrite
    _validate_decoder_only(cfg["backbone"])
    _set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enable_backend_speedups(cfg)   # opt-in (cfg['use_tf32']); no-op / bit-exact by default
    print(f"\n{'='*64}\nTCAA multi-round FL: {cfg['experiment_name']}  (device={device})\n{'='*64}")

    model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev = build_model_and_data(cfg, device)
    g0 = model.get_flat_params().detach().cpu()
    print(f"  LoRA update dimension: {g0.numel():,}")

    # Physically-calibrated cost coefficients (c_f = d_model, c_a = 1) for the honest
    # amplification column. d_model lives under different attr names across backbones
    # (Qwen: hidden_size; GPT2: n_embd), so probe both.
    report_calib = bool(cfg.get("report_calibrated_cost", True))
    mcfg = model.inner().config
    d_model = int(getattr(mcfg, "hidden_size", None) or getattr(mcfg, "n_embd", None) or 0)
    calib = calibrate_coefficients(d_model) if (report_calib and d_model > 0) else None
    if calib:
        print(f"  calibrated cost coeffs: d_model={d_model} -> c_f={calib['c_f']:.0f}, c_a={calib['c_a']:.0f} "
              f"(super-linear onset L ~ 2n+{2*calib['c_f']/calib['c_a']:.0f})")

    num_benign = cfg["num_clients"] - cfg["num_attackers"]
    benign_ids = list(range(num_benign))
    attacker_ids = list(range(num_benign, cfg["num_clients"]))
    # Fixed non-IID data partition; each benign client keeps its shard across rounds.
    shards = partition_examples(clean_tr, num_benign, cfg["dirichlet_alpha"], seed=cfg["seed"])
    shard_sizes = [float(max(len(s), 1)) for s in shards]
    atk_size = cfg["attacker_claimed_data_size"] or float(np.mean(shard_sizes))
    track_ben = bool(cfg.get("track_benign_baseline", True))
    rng = random.Random(cfg["seed"] + 12345)
    print(f"  {num_benign} benign + {len(attacker_ids)} attackers; sample "
          f"{cfg['clients_per_round']}/round; {cfg['num_rounds']} rounds; "
          f"shard sizes={[len(s) for s in shards]}")

    # --- fixed absolute reference: the pristine (round-0) global, measured ONCE. -------- #
    # Utility "preservation" and cost amplification are reported against THIS fixed anchor,
    # not only against the benign-only global. The benign-only baseline itself drifts as
    # training proceeds, so the ppl ratio atk/benign conflates attack damage with benign
    # drift (a ratio < 1 can just mean the benign global degraded). g0 is the LoRA-init
    # global == the pristine backbone, so ppl/ROUGE/cost vs g0 is an unconfounded baseline.
    me = float(cfg.get("decensor_max_extra", DEFAULT_DECENSOR_MAX_EXTRA))
    print("  measuring pristine (round-0) reference for absolute utility / amplification ...")
    pri_tau = _measure_cost(model, g0, tau_ev, cfg, spec, device)
    pri_cln = _measure_cost(model, g0, clean_ev, cfg, spec, device)
    ppl_pri_cln = _ppl(model, g0, clean_ev, cfg, spec, device)
    ppl_pri_tau = _ppl(model, g0, tau_ev, cfg, spec, device)
    pristine_ref = {
        "tau_mean_cost": pri_tau.mean_cost,
        "tau_mean_cost_calib": pri_tau.mean_cost_at(calib["c_f"], calib["c_a"]) if calib else None,
        "tau_decensored_mean_cost": pri_tau.decensored_mean_cost(me),
        "clean_mean_cost": pri_cln.mean_cost, "kv_tau": pri_tau.mean_kv_proxy,
        "tau_mean_len": round(pri_tau.mean_output_len, 2),
        "clean_mean_len": round(pri_cln.mean_output_len, 2),
        "tau_effective_len": round(pri_tau.mean_effective_len, 2),
        "ppl_clean": round(ppl_pri_cln, 4), "ppl_tau": round(ppl_pri_tau, 4),
        "rouge_recall_clean": round(pri_cln.mean_rouge_recall, 4),
        "rouge_recall_tau": round(pri_tau.mean_rouge_recall, 4),
    }
    print(f"    pristine: ppl_clean={pristine_ref['ppl_clean']:.3f} "
          f"ROUGE_clean={pristine_ref['rouge_recall_clean']:.3f} "
          f"ROUGE_tau={pristine_ref['rouge_recall_tau']:.3f} "
          f"tau_len={pristine_ref['tau_mean_len']:.1f}")

    g_atk = g0.clone()
    g_ben = g0.clone() if track_ben else None
    durability: List[Dict] = []
    stealth_trace: List[Dict] = []
    defense_telemetry: List[Dict] = []            # per-round per-client update geometry (offline defense eval)
    last_mal_trace: Optional[List[Dict]] = None   # a representative attacker trajectory (process data)
    out_dir = Path("results") / cfg["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(cfg["num_rounds"]):
        sel_ben, sel_atk = _sample_participants(
            rng, benign_ids, attacker_ids, cfg["clients_per_round"],
            cfg.get("attacker_always_selected", False))

        # --- benign local training from the (attacked) broadcast global ---
        ben_updates, ben_sizes, ben_ids_used = [], [], []
        for cid in sel_ben:
            if not shards[cid]:
                continue
            ben_updates.append(_benign_update(model, shards[cid], cfg, g_atk, spec, device))
            ben_sizes.append(shard_sizes[cid])
            ben_ids_used.append(cid)

        # --- attacker(s): ALM-constrained length attack from the attacked global ---
        # Computed SEQUENTIALLY so each attacker sees the peers already produced this round;
        # the coordination penalty (gamma_coord) then keeps the colluding pair inside the
        # benign distance/cosine PAIR band instead of collapsing into detectable near-twins.
        atk_updates, atk_reports, atk_ids_used = [], [], []
        for cid in sel_atk:
            if not ben_updates:
                break  # need a benign envelope to constrain against
            delta_mal, mtrace = _malicious_update(
                model, clean_tr, tau_tr, cfg, g_atk, spec, device,
                benign_updates=ben_updates, benign_sizes=ben_sizes, atk_size=atk_size,
                peer_updates=list(atk_updates))
            atk_updates.append(delta_mal)
            atk_ids_used.append(cid)
            last_mal_trace = mtrace   # keep the most recent attacker's within-round trajectory
            atk_reports.append(evaluate_stealth(
                delta_mal, ben_updates, ben_sizes, attacker_weight=atk_size,
                d_T=cfg["d_T"], delta_T=cfg["delta_T"],
                use_pairwise_cosine=cfg.get("stealth_use_pairwise_cosine", False)))

        # --- server aggregation (attacked trajectory) ---
        if ben_updates or atk_updates:
            all_up = ben_updates + atk_updates
            all_w = ben_sizes + [atk_size] * len(atk_updates)
            g_atk = g_atk + cfg["server_lr"] * _fedavg(all_up, all_w)

        # --- per-round defense telemetry (offline detection eval; aggregation stays FedAvg) ---
        if cfg.get("collect_defense_telemetry", True) and (ben_updates or atk_updates):
            defense_telemetry.append(_collect_defense_telemetry(
                t, ben_updates, ben_sizes, ben_ids_used,
                atk_updates, [atk_size] * len(atk_updates), atk_ids_used,
                num_attackers=cfg["num_attackers"]))
            if cfg.get("save_update_vectors", False):
                _save_update_vectors(out_dir, t, ben_updates, ben_ids_used, atk_updates, atk_ids_used)

        # --- parallel benign-only trajectory (amplification baseline) ---
        if track_ben:
            ben_only = []
            for cid in sel_ben:
                if not shards[cid]:
                    continue
                ben_only.append(_benign_update(model, shards[cid], cfg, g_ben, spec, device))
            if ben_only:
                g_ben = g_ben + cfg["server_lr"] * _fedavg(ben_only, ben_sizes)

        # --- per-round stealth (worst case across sampled attackers) ---
        if atk_reports:
            worst = max(atk_reports, key=lambda r: r.attacker_distance)
            joint = all(r.jointly_satisfied for r in atk_reports)
            rec = {"round": t, "n_attackers": len(atk_reports),
                   "attacker_distance": round(worst.attacker_distance, 4),
                   "d_T": round(worst.d_T, 4),
                   "attacker_cosine": round(min(r.attacker_cosine for r in atk_reports), 4),
                   "pairwise_cosine": round(min(r.attacker_pairwise_cosine for r in atk_reports), 4),
                   "jointly_satisfied": bool(joint)}
            stealth_trace.append(rec)
        else:
            rec = {"round": t, "n_attackers": 0, "jointly_satisfied": None}

        # --- periodic cost / utility measurement (generation is expensive) ---
        do_measure = (t % cfg["measure_every"] == 0) or (t == cfg["num_rounds"] - 1)
        if do_measure:
            atk_tau = _measure_cost(model, g_atk, tau_ev, cfg, spec, device)
            atk_cln = _measure_cost(model, g_atk, clean_ev, cfg, spec, device)
            if track_ben:
                ben_tau = _measure_cost(model, g_ben, tau_ev, cfg, spec, device)
                ben_cln = _measure_cost(model, g_ben, clean_ev, cfg, spec, device)
                amp_tau = amplification_ratio(atk_tau.mean_cost, ben_tau.mean_cost)
                amp_tau_med = amplification_ratio(atk_tau.median_cost, ben_tau.median_cost)
                amp_clean = amplification_ratio(atk_cln.mean_cost, ben_cln.mean_cost)
                kv_amp = amplification_ratio(atk_tau.mean_kv_proxy, ben_tau.mean_kv_proxy)
                # De-censored amplification: correct for the cap so a saturated (trunc~1)
                # round is not clipped to a flat lower bound. Uses the EOS-hazard tail.
                amp_tau_dec = amplification_ratio(
                    atk_tau.decensored_mean_cost(me), ben_tau.decensored_mean_cost(me))
            else:
                # No benign baseline: report triggered-vs-clean on the attacked global.
                amp_tau = amplification_ratio(atk_tau.mean_cost, atk_cln.mean_cost)
                amp_tau_med = amplification_ratio(atk_tau.median_cost, atk_cln.median_cost)
                amp_clean, kv_amp = 1.0, amplification_ratio(atk_tau.mean_kv_proxy, atk_cln.mean_kv_proxy)
                amp_tau_dec = amplification_ratio(
                    atk_tau.decensored_mean_cost(me), atk_cln.decensored_mean_cost(me))
            # Amplification vs the FIXED pristine (round-0) reference: total cost inflation
            # relative to the un-attacked starting model (censored + de-censored).
            amp_tau_pri = amplification_ratio(atk_tau.mean_cost, pristine_ref["tau_mean_cost"])
            amp_tau_pri_dec = amplification_ratio(
                atk_tau.decensored_mean_cost(me), pristine_ref["tau_decensored_mean_cost"])
            # Physically-calibrated amplification (c_f=d_model, c_a=1): the honest compute-cost
            # multiple a real decoder incurs. At these lengths cost is linear-dominated, so this
            # is typically well below the naive amp_tau (which credits the quadratic term from
            # token 1). Reported alongside, not instead — the naive number is the sensitivity bound.
            if calib:
                _cf, _ca = calib["c_f"], calib["c_a"]
                ben_ref_calib = (ben_tau.mean_cost_at(_cf, _ca) if track_ben
                                 else atk_cln.mean_cost_at(_cf, _ca))
                amp_tau_calib = amplification_ratio(atk_tau.mean_cost_at(_cf, _ca), ben_ref_calib)
                amp_tau_pri_calib = amplification_ratio(
                    atk_tau.mean_cost_at(_cf, _ca), pristine_ref["tau_mean_cost_calib"])
            else:
                amp_tau_calib = amp_tau_pri_calib = None
            # De-censored validity: when the truncated tail hit the max_extra cap (EOS hazard ~0),
            # the de-censored estimate is 100% the horizon assumption and 0% data. Flag it so the
            # number is never read as a measurement (raise the generation cap to make it valid).
            decensored_valid = not atk_tau.residual_capped(me)

            ppl_clean = _ppl(model, g_atk, clean_ev, cfg, spec, device)
            ppl_clean_ben = _ppl(model, g_ben, clean_ev, cfg, spec, device) if track_ben else None
            ppl_ratio = round(ppl_clean / ppl_clean_ben, 4) if ppl_clean_ben else None
            # Utility vs the FIXED pristine baseline (unconfounded by benign drift): ppl and
            # ROUGE-L recall (clean AND tau). ROUGE recall ~ pristine => "the answer is still
            # there" even as tau outputs run long; the direct "utility preserved" evidence.
            ppl_ratio_pri = round(ppl_clean / max(pristine_ref["ppl_clean"], 1e-9), 4)
            point = {
                "round": t, "amp_tau": round(amp_tau, 4), "amp_tau_median": round(amp_tau_med, 4),
                "amp_tau_decensored": round(amp_tau_dec, 4),
                "decensored_valid": bool(decensored_valid),
                "amp_tau_vs_pristine": round(amp_tau_pri, 4),
                "amp_tau_vs_pristine_decensored": round(amp_tau_pri_dec, 4),
                "amp_tau_calibrated": round(amp_tau_calib, 4) if amp_tau_calib is not None else None,
                "amp_tau_vs_pristine_calibrated": round(amp_tau_pri_calib, 4) if amp_tau_pri_calib is not None else None,
                "cost_c_f_calibrated": (calib["c_f"] if calib else None),
                "amp_clean": round(amp_clean, 4),
                "selectivity": round(amplification_ratio(amp_tau, amp_clean), 4),
                "kv_amp_tau": round(kv_amp, 4),
                "tau_len_atk": round(atk_tau.mean_output_len, 2),
                "tau_len_atk_decensored": round(atk_tau.decensored_mean_len(me), 2),
                "tau_effective_len_atk": round(atk_tau.mean_effective_len, 2),
                # Effective (degeneracy-discounted) amplification: length gain after removing
                # looped padding. If << amp_tau, the attack is winning on repetition, not
                # genuine length (weak + a repetition filter would catch it).
                "amp_tau_effective": round(
                    amplification_ratio(atk_tau.mean_effective_len,
                                        (ben_tau.mean_effective_len if track_ben else atk_cln.mean_effective_len)), 4),
                "clean_len_atk": round(atk_cln.mean_output_len, 2),
                "truncation_tau": round(atk_tau.truncation_rate, 3),
                "eos_hazard_tau": round(atk_tau.eos_hazard(), 6),
                "residual_capped_tau": atk_tau.residual_capped(me),
                "repetition_tau": round(atk_tau.mean_repetition, 3),
                "distinct_ratio_tau": round(atk_tau.mean_distinct_ratio, 4),
                "ppl_clean_atk": round(ppl_clean, 3),
                "ppl_clean_ben": round(ppl_clean_ben, 3) if ppl_clean_ben else None,
                "ppl_clean_pristine": pristine_ref["ppl_clean"],
                "ppl_ratio": ppl_ratio,                     # atk / benign-only (marginal, drifts)
                "ppl_ratio_vs_pristine": ppl_ratio_pri,     # atk / pristine (fixed, primary)
                "rouge_recall_clean_atk": round(atk_cln.mean_rouge_recall, 4),
                "rouge_recall_clean_pristine": pristine_ref["rouge_recall_clean"],
                "rouge_recall_tau_atk": round(atk_tau.mean_rouge_recall, 4),
                "rouge_recall_tau_pristine": pristine_ref["rouge_recall_tau"],
                "stealth_ok": rec.get("jointly_satisfied"),
            }
            durability.append(point)
            _dec_mark = "" if decensored_valid else "!"  # '!' = assumption-only, no data (cap hit)
            _calib_str = (f"calib {point['amp_tau_calibrated']:.2f}x " if amp_tau_calib is not None else "")
            print(f"  [round {t:3d}] amp_tau={point['amp_tau']:.3f}x (dec {point['amp_tau_decensored']:.2f}{_dec_mark}, "
                  f"eff {point['amp_tau_effective']:.2f}, med {point['amp_tau_median']:.2f}) "
                  f"{_calib_str}vs_pristine={point['amp_tau_vs_pristine']:.2f}x "
                  f"tau_len={point['tau_len_atk']:.0f}(dec {point['tau_len_atk_decensored']:.0f}) "
                  f"trunc={point['truncation_tau']:.2f} rep={point['repetition_tau']:.2f} "
                  f"distinct={point['distinct_ratio_tau']:.2f} "
                  f"ppl_pri={point['ppl_ratio_vs_pristine']:.3f} "
                  f"ROUGE_cln={point['rouge_recall_clean_atk']:.2f}/τ{point['rouge_recall_tau_atk']:.2f} "
                  f"stealth={point['stealth_ok']}")
        else:
            print(f"  [round {t:3d}] sel_ben={sel_ben} sel_atk={sel_atk} "
                  f"stealth={rec.get('jointly_satisfied')}")

    # Qualitative decoded samples from the FINAL attacked global (coherence evidence).
    print("  decoding a few final examples (qualitative coherence check) ...")
    final_examples = _dump_fl_examples(model, g_atk, tau_ev, clean_ev, tokenizer, cfg, spec, device,
                                       k=cfg.get("num_dump_examples", 4))

    results = {
        "config": {k: cfg[k] for k in (
            "experiment_name", "backbone", "source", "num_clients", "num_attackers",
            "num_rounds", "clients_per_round", "attacker_always_selected",
            "local_epochs", "attacker_steps", "gamma", "gamma_clean", "kd_clean_weight",
            "clean_anchor_two_sided",
            "gamma_rep", "no_repeat_ngram_size", "onpolicy_horizon",
            "stealth_kappa", "stealth_use_pairwise_cosine", "stealth_cosine_two_sided",
            "stealth_cos_low", "stealth_norm_constraint", "gamma_coord", "report_calibrated_cost",
            "collect_defense_telemetry", "save_update_vectors", "run_defense_eval",
            "pool_size", "eval_size", "max_new_tokens", "decensor_max_extra", "lora_r")},
        "cost_c_f_calibrated": (calib["c_f"] if calib else None),
        "cost_c_a_calibrated": (calib["c_a"] if calib else None),
        "lora_update_dim": int(g0.numel()),
        "pristine_reference": pristine_ref,
        "durability": durability,
        "stealth_trace": stealth_trace,
        # Per-round, per-client update geometry (labelled benign/attacker) for OFFLINE defense
        # evaluation. Aggregation stays FedAvg; this is the raw material a Krum/norm-clip/FLTrust
        # detector would consume — see _collect_defense_telemetry.
        "defense_telemetry": defense_telemetry,
        "sample_mal_trace": last_mal_trace,     # a representative within-round attacker trajectory
        "final_examples": final_examples,       # decoded samples from the final attacked global
    }
    # Independent defense-evasion verdict: replay real robust aggregators on the telemetry, so
    # the parameter-stealth claim rests on named defenses instead of our own benign envelope.
    if cfg.get("run_defense_eval", True) and defense_telemetry:
        from .defenses import (evaluate_defenses, evaluate_vector_defenses,
                               format_defense_report)
        ev = evaluate_defenses(defense_telemetry, num_attackers=cfg["num_attackers"])
        vdir = out_dir / "update_vectors"
        vev = evaluate_vector_defenses(vdir) if vdir.exists() else {}
        results["defense_evaluation"] = {"telemetry_defenses": ev, "vector_defenses": vev}
        print("\n" + format_defense_report(ev, vev or None))
    (out_dir / "fl_results.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)
    _save_figure(results, out_dir / "figures")
    print(f"\n  Multi-round results written to {out_dir}/fl_results.json")
    return results


def _print_summary(r: Dict) -> None:
    dur, st = r["durability"], r["stealth_trace"]
    print(f"\n{'='*80}\nTCAA MULTI-ROUND SUMMARY ({r['config']['num_rounds']} rounds, "
          f"{r['config']['num_clients']}={r['config']['num_clients']-r['config']['num_attackers']}"
          f"+{r['config']['num_attackers']})\n{'='*80}")
    # HONEST HEADLINE ORDER: amp_eff (degeneracy-discounted), amp_cal (calibrated coeffs) and
    # amp_pri (vs fixed pristine) are the claim-bearing columns; amp (raw) and amp_dec are
    # caveated — amp credits the quadratic term from token 1, and amp_dec is shown as '—' when
    # it is assumption-only (truncated tail hit the cap => no data, decensored_valid=False).
    pri = r.get("pristine_reference", {})
    if pri:
        print(f"  pristine (round-0) ref: ppl_clean={pri.get('ppl_clean')} "
              f"ROUGE_clean={pri.get('rouge_recall_clean')} ROUGE_tau={pri.get('rouge_recall_tau')} "
              f"tau_len={pri.get('tau_mean_len')}")
    if r.get("cost_c_f_calibrated"):
        print(f"  amp_cal = calibrated coeffs c_f={r['cost_c_f_calibrated']:.0f}, c_a={r.get('cost_c_a_calibrated')}"
              f"  (honest compute-cost multiple; amp uses naive c_f=c_a=1)")
    print(f"  {'round':>5} {'amp_eff':>8} {'amp_cal':>8} {'amp_pri':>8} {'amp_med':>8} {'amp':>6} {'amp_dec':>8} "
          f"{'tau_len':>8} {'trunc':>6} {'rep':>5} {'dist':>5} {'ppl_pri':>7} {'R_cln':>6} {'R_tau':>6} {'stealth':>8}")
    for p in dur:
        _cal = p.get("amp_tau_calibrated")
        cal_cell = f"{_cal:>8.2f}" if _cal is not None else f"{'—':>8}"
        dec_cell = (f"{p.get('amp_tau_decensored', float('nan')):>8.2f}"
                    if p.get("decensored_valid", True) else f"{'—':>8}")
        print(f"  {p['round']:>5} {p.get('amp_tau_effective', float('nan')):>8.2f} {cal_cell} "
              f"{p.get('amp_tau_vs_pristine', float('nan')):>8.2f} {p.get('amp_tau_median', float('nan')):>8.2f} "
              f"{p['amp_tau']:>6.2f} {dec_cell} "
              f"{p['tau_len_atk']:>8.1f} "
              f"{p['truncation_tau']:>6.2f} {p['repetition_tau']:>5.2f} {p.get('distinct_ratio_tau', float('nan')):>5.2f} "
              f"{p.get('ppl_ratio_vs_pristine', float('nan')):>7.3f} "
              f"{p.get('rouge_recall_clean_atk', float('nan')):>6.2f} "
              f"{p.get('rouge_recall_tau_atk', float('nan')):>6.2f} {str(p['stealth_ok']):>8}")
    rounds_with_atk = [s for s in st if s.get("n_attackers")]
    ok = sum(1 for s in rounds_with_atk if s["jointly_satisfied"])
    if dur:
        first, last = dur[0], dur[-1]
        print("  " + "-" * 76)
        # Headline durability on the HONEST metrics: vs-pristine (fixed anchor) and effective
        # (degeneracy-discounted). Narrate as rapidly-saturating-and-durable, not accumulating.
        _cal_note = (f", calibrated {last.get('amp_tau_calibrated'):.2f}x"
                     if last.get("amp_tau_calibrated") is not None else "")
        _dec_note = (f"; raw de-censored {last.get('amp_tau_decensored', float('nan')):.2f}x (assumption-only)"
                     if not last.get("decensored_valid", True)
                     else f"; raw de-censored {last.get('amp_tau_decensored', float('nan')):.2f}x")
        print(f"  durability (vs pristine): {first.get('amp_tau_vs_pristine', float('nan')):.2f}x -> "
              f"{last.get('amp_tau_vs_pristine', float('nan')):.2f}x  |  effective "
              f"{first.get('amp_tau_effective', float('nan')):.2f}x -> {last.get('amp_tau_effective', float('nan')):.2f}x"
              f"{_cal_note}  [round {first['round']}->{last['round']}]")
        print(f"    (footnote: raw amp_tau {first['amp_tau']:.2f}x -> {last['amp_tau']:.2f}x{_dec_note})")
        if first.get("ppl_ratio_vs_pristine") is not None and last.get("ppl_ratio_vs_pristine") is not None:
            print(f"  utility (vs pristine): ppl {first['ppl_ratio_vs_pristine']:.3f} -> "
                  f"{last['ppl_ratio_vs_pristine']:.3f}  |  clean ROUGE-L recall "
                  f"{first.get('rouge_recall_clean_atk')} -> {last.get('rouge_recall_clean_atk')} "
                  f"(pristine {pri.get('rouge_recall_clean')}); "
                  f"tau ROUGE {first.get('rouge_recall_tau_atk')} -> {last.get('rouge_recall_tau_atk')} "
                  f"(~pristine = answer kept)")
    if rounds_with_atk:
        print(f"  stealth: jointly satisfied in {ok}/{len(rounds_with_atk)} attacker-participating rounds")
    print("=" * 80)


def _save_figure(r: Dict, out_dir: Path) -> None:
    """Best-effort: save the FL figures (durability / utility / stealth / defense geometry)
    to disk. Delegates to tcaa.visualize (single-axis stacked subplots; NO dual-y) so the
    600-DPI PNGs and vector PDFs match what the notebook renders inline. Skips gracefully
    without matplotlib.

    Does NOT switch the matplotlib backend: fig.savefig works under any backend, and forcing
    Agg here would clobber a notebook's inline backend, silently killing later render_*_report
    inline display. Figures are closed after saving, so run_fl never spams the notebook."""
    if not r.get("durability"):
        return
    try:
        from .visualize import make_fl_figures, save_figure
    except Exception as e:  # pragma: no cover
        print(f"  [fl] skipped figure ({e})")
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for key, fig in make_fl_figures(r):
        p = out_dir / f"{key}.png"
        written = save_figure(fig, p)
        plt.close(fig)
        print(f"  [fl] saved {', '.join(str(path) for path in written)}")


def _parse_args():
    p = argparse.ArgumentParser(description="TCAA multi-round FL runner")
    p.add_argument("--smoke", action="store_true", help="CPU tiny-GPT2 + synthetic data")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--num-attackers", type=int, default=None)
    p.add_argument("--clients-per-round", type=int, default=None)
    p.add_argument("--attacker-always-selected", action="store_true")
    p.add_argument("--config-json", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg: Dict = {}
    if args.smoke:
        cfg.update(fl_smoke_overrides())
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text()))
    for attr, key in (("backbone", "backbone"), ("source", "source"),
                      ("num_rounds", "num_rounds"), ("num_clients", "num_clients"),
                      ("num_attackers", "num_attackers"), ("clients_per_round", "clients_per_round")):
        v = getattr(args, attr)
        if v is not None:
            cfg[key] = v
    if args.attacker_always_selected:
        cfg["attacker_always_selected"] = True
    run_fl(cfg)


if __name__ == "__main__":
    main()
