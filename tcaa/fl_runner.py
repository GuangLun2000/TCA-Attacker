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
# curve.  No AugMP file is modified; we reuse server.py's aggregation convention.
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

from .cost_model import amplification_ratio
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
        # --- moderately larger data than Phase-0's 512 / 64 ---
        "pool_size": 4000, "eval_size": 256,
        # --- per-round attacker budget (smaller than single-round; it repeats each round) ---
        "attacker_steps": 100,
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

    g_atk = g0.clone()
    g_ben = g0.clone() if track_ben else None
    durability: List[Dict] = []
    stealth_trace: List[Dict] = []

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
            delta_mal, _ = _malicious_update(
                model, clean_tr, tau_tr, cfg, g_atk, spec, device,
                benign_updates=ben_updates, benign_sizes=ben_sizes, atk_size=atk_size)
            atk_updates.append(delta_mal)
            atk_reports.append(evaluate_stealth(
                delta_mal, ben_updates, ben_sizes, attacker_weight=atk_size,
                d_T=cfg["d_T"], delta_T=cfg["delta_T"]))

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
            else:
                # No benign baseline: report triggered-vs-clean on the attacked global.
                amp_tau = amplification_ratio(atk_tau.mean_cost, atk_cln.mean_cost)
                amp_tau_med = amplification_ratio(atk_tau.median_cost, atk_cln.median_cost)
                amp_clean, kv_amp = 1.0, amplification_ratio(atk_tau.mean_kv_proxy, atk_cln.mean_kv_proxy)
            ppl_clean = _ppl(model, g_atk, clean_ev, cfg, spec, device)
            point = {
                "round": t, "amp_tau": round(amp_tau, 4), "amp_tau_median": round(amp_tau_med, 4),
                "amp_clean": round(amp_clean, 4),
                "selectivity": round(amplification_ratio(amp_tau, amp_clean), 4),
                "kv_amp_tau": round(kv_amp, 4),
                "tau_len_atk": round(atk_tau.mean_output_len, 2),
                "clean_len_atk": round(atk_cln.mean_output_len, 2),
                "truncation_tau": round(atk_tau.truncation_rate, 3),
                "repetition_tau": round(atk_tau.mean_repetition, 3),
                "ppl_clean_atk": round(ppl_clean, 3),
                "stealth_ok": rec.get("jointly_satisfied"),
            }
            durability.append(point)
            print(f"  [round {t:3d}] amp_tau={point['amp_tau']:.3f}x (med {point['amp_tau_median']:.3f}) "
                  f"sel={point['selectivity']:.2f} tau_len={point['tau_len_atk']:.1f} "
                  f"ppl={point['ppl_clean_atk']:.2f} stealth={point['stealth_ok']}")
        else:
            print(f"  [round {t:3d}] sel_ben={sel_ben} sel_atk={sel_atk} "
                  f"stealth={rec.get('jointly_satisfied')}")

    results = {
        "config": {k: cfg[k] for k in (
            "experiment_name", "backbone", "source", "num_clients", "num_attackers",
            "num_rounds", "clients_per_round", "attacker_always_selected",
            "local_epochs", "attacker_steps", "gamma", "gamma_clean", "stealth_kappa",
            "pool_size", "eval_size", "max_new_tokens", "lora_r")},
        "lora_update_dim": int(g0.numel()),
        "durability": durability,
        "stealth_trace": stealth_trace,
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
    print(f"  {'round':>6} {'amp_tau':>8} {'amp_med':>8} {'sel':>6} {'kv_amp':>6} "
          f"{'tau_len':>8} {'ppl':>7} {'stealth':>8}")
    for p in dur:
        print(f"  {p['round']:>6} {p['amp_tau']:>8.3f} {p['amp_tau_median']:>8.3f} "
              f"{p['selectivity']:>6.2f} {p['kv_amp_tau']:>6.2f} {p['tau_len_atk']:>8.1f} "
              f"{p['ppl_clean_atk']:>7.2f} {str(p['stealth_ok']):>8}")
    rounds_with_atk = [s for s in st if s.get("n_attackers")]
    ok = sum(1 for s in rounds_with_atk if s["jointly_satisfied"])
    if dur:
        first, last = dur[0], dur[-1]
        print("  " + "-" * 76)
        print(f"  durability: amp_tau {first['amp_tau']:.3f}x (round {first['round']}) -> "
              f"{last['amp_tau']:.3f}x (round {last['round']})")
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
