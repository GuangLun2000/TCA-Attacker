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

from .cost_model import DEFAULT_DECENSOR_MAX_EXTRA, amplification_ratio
from .gen_data import partition_examples
from .phase0_runner import (_benign_update, _malicious_update, _measure_cost, _ppl,
                            _set_seed, _validate_decoder_only, build_model_and_data,
                            default_config)
from .stealth import evaluate_stealth


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
        # Generation cap for MEASUREMENT (does not affect the attack, which optimizes only
        # up to onpolicy_horizon). Phase-0's 256 saturates under multi-round accumulation
        # (outputs pin at the cap by ~round 5, truncation -> ~1.0), so the durability curve
        # is a flat censored artifact. 1024 lets the true accumulation curve show; the
        # residual truncation is corrected by the de-censored estimate. Runtime scales with
        # this — lower to 512 if generation is too slow (de-censoring still corrects the tail).
        "max_new_tokens": 1024,
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
                # keep enough text to SEE the full loop in the qualitative viz (was 180/500)
                "prompt": tokenizer.decode(exk[i].prompt_ids, skip_special_tokens=True)[:400],
                "output": tokenizer.decode(out_ids, skip_special_tokens=True)[:6000],
                "len": L, "truncated": bool(eos_pos.numel() == 0),
                "rouge_recall": round(rouge_l_recall(out_ids, ref), 3) if ref else None,
                "repetition": round(repetition_rate(out_ids), 3),
                "distinct": round(distinct_ratio(out_ids, 4), 3),
            })
    return recs


def run_fl(config: Dict) -> Dict:
    cfg = default_fl_config()
    cfg.update(config or {})
    _validate_decoder_only(cfg["backbone"])
    _set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*64}\nTCAA multi-round FL: {cfg['experiment_name']}  (device={device})\n{'='*64}")

    model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev = build_model_and_data(cfg, device)
    g0 = model.get_flat_params().detach().cpu()
    print(f"  LoRA update dimension: {g0.numel():,}")

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
    last_mal_trace: Optional[List[Dict]] = None   # a representative attacker trajectory (process data)

    for t in range(cfg["num_rounds"]):
        sel_ben, sel_atk = _sample_participants(
            rng, benign_ids, attacker_ids, cfg["clients_per_round"],
            cfg.get("attacker_always_selected", False))

        # --- benign local training from the (attacked) broadcast global ---
        ben_updates, ben_sizes = [], []
        for cid in sel_ben:
            if not shards[cid]:
                continue
            ben_updates.append(_benign_update(model, shards[cid], cfg, g_atk, spec, device))
            ben_sizes.append(shard_sizes[cid])

        # --- attacker(s): ALM-constrained length attack from the attacked global ---
        atk_updates, atk_reports = [], []
        for cid in sel_atk:
            if not ben_updates:
                break  # need a benign envelope to constrain against
            delta_mal, mtrace = _malicious_update(
                model, clean_tr, tau_tr, cfg, g_atk, spec, device,
                benign_updates=ben_updates, benign_sizes=ben_sizes, atk_size=atk_size)
            atk_updates.append(delta_mal)
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
                "amp_tau_vs_pristine": round(amp_tau_pri, 4),
                "amp_tau_vs_pristine_decensored": round(amp_tau_pri_dec, 4),
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
            print(f"  [round {t:3d}] amp_tau={point['amp_tau']:.3f}x (dec {point['amp_tau_decensored']:.2f}, "
                  f"eff {point['amp_tau_effective']:.2f}, med {point['amp_tau_median']:.2f}) "
                  f"vs_pristine={point['amp_tau_vs_pristine']:.2f}x "
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
            "gamma_rep", "no_repeat_ngram_size", "onpolicy_horizon",
            "stealth_kappa", "stealth_use_pairwise_cosine",
            "pool_size", "eval_size", "max_new_tokens", "decensor_max_extra", "lora_r")},
        "lora_update_dim": int(g0.numel()),
        "pristine_reference": pristine_ref,
        "durability": durability,
        "stealth_trace": stealth_trace,
        "sample_mal_trace": last_mal_trace,     # a representative within-round attacker trajectory
        "final_examples": final_examples,       # decoded samples from the final attacked global
    }
    out_dir = Path("results") / cfg["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
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
    # amp_dec de-censors the cap; amp_pri is vs the fixed pristine (round-0) reference.
    # ppl_pri / ROUGE (vs pristine) are the unconfounded utility-preserved metrics; trunc
    # flags cap-censoring and rep flags degeneracy (a 'long' output that is a repetition
    # loop is a weak, detectable amplification).
    pri = r.get("pristine_reference", {})
    if pri:
        print(f"  pristine (round-0) ref: ppl_clean={pri.get('ppl_clean')} "
              f"ROUGE_clean={pri.get('rouge_recall_clean')} ROUGE_tau={pri.get('rouge_recall_tau')} "
              f"tau_len={pri.get('tau_mean_len')}")
    print(f"  {'round':>5} {'amp':>6} {'amp_dec':>8} {'amp_eff':>8} {'amp_pri':>8} {'tau_len':>8} "
          f"{'dec_len':>8} {'trunc':>6} {'rep':>5} {'dist':>5} {'ppl_pri':>7} {'R_cln':>6} {'R_tau':>6} {'stealth':>8}")
    for p in dur:
        print(f"  {p['round']:>5} {p['amp_tau']:>6.2f} {p.get('amp_tau_decensored', float('nan')):>8.2f} "
              f"{p.get('amp_tau_effective', float('nan')):>8.2f} "
              f"{p.get('amp_tau_vs_pristine', float('nan')):>8.2f} {p['tau_len_atk']:>8.1f} "
              f"{p.get('tau_len_atk_decensored', float('nan')):>8.1f} "
              f"{p['truncation_tau']:>6.2f} {p['repetition_tau']:>5.2f} {p.get('distinct_ratio_tau', float('nan')):>5.2f} "
              f"{p.get('ppl_ratio_vs_pristine', float('nan')):>7.3f} "
              f"{p.get('rouge_recall_clean_atk', float('nan')):>6.2f} "
              f"{p.get('rouge_recall_tau_atk', float('nan')):>6.2f} {str(p['stealth_ok']):>8}")
    rounds_with_atk = [s for s in st if s.get("n_attackers")]
    ok = sum(1 for s in rounds_with_atk if s["jointly_satisfied"])
    if dur:
        first, last = dur[0], dur[-1]
        print("  " + "-" * 76)
        print(f"  durability: amp_tau {first['amp_tau']:.3f}x -> {last['amp_tau']:.3f}x "
              f"(de-censored {first.get('amp_tau_decensored', float('nan')):.2f}x -> "
              f"{last.get('amp_tau_decensored', float('nan')):.2f}x; "
              f"vs pristine {last.get('amp_tau_vs_pristine', float('nan')):.2f}x) "
              f"[round {first['round']}->{last['round']}]")
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
    """Best-effort: save the unified durability + per-round stealth figures to disk.
    Delegates to tcaa.visualize (single-axis stacked subplots; NO dual-y) so the on-disk
    PNGs match what the notebook renders inline. Skips gracefully without matplotlib."""
    if not r.get("durability"):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        from .visualize import make_fl_figures
    except Exception as e:  # pragma: no cover
        print(f"  [fl] skipped figure ({e})")
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for key, fig in make_fl_figures(r):
        p = out_dir / f"{key}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fl] saved {p}")


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
