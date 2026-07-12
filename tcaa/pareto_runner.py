# tcaa/pareto_runner.py
# Stage-1d Pareto sweep: how much cost amplification survives at the stealth boundary?
#
# Sweeps the attack strength (gamma), the clean-length anchor (gamma_clean), and the
# stealth budget (stealth_kappa = fraction of the benign distance envelope the attacker
# is allowed to use), running the ALM-constrained Phase-0 at each grid point. The output
# is the amplification-vs-stealth-slack frontier that decides whether the single-round
# constrained attack is viable, or whether we must escalate to Stage 2 (VGAE camouflage +
# multi-round accumulation).
#
# CPU smoke:  python -m tcaa.pareto_runner --smoke
# Real (GPU): python -m tcaa.pareto_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

from .phase0_runner import default_config, run_phase0, run_phase0_seeds, smoke_overrides
from .run_paths import stamp_run_subdir


def _point_metrics(res: Dict) -> Dict:
    c, u, s = res["cost"], res["utility"], res["stealth"]
    return {
        "amp_tau": c["amplification_tau"],
        "amp_tau_median": c["amplification_tau_median"],
        "amp_clean": c["amplification_clean"],
        "selectivity": c["trigger_selectivity"],
        "kv_amp_tau": c.get("kv_amplification_tau", float("nan")),
        "ppl_clean_ratio": u["ppl_clean_ratio"],
        "distance": s["attacker_distance"], "d_T": s["d_T"],
        "dist_margin": round(s["d_T"] - s["attacker_distance"], 4),  # >0 => inside budget
        "cosine": s["attacker_cosine"], "delta_T": s["delta_T"],
        "pairwise_cosine": s["attacker_pairwise_cosine"],
        "jointly_satisfied": bool(s["jointly_satisfied"]),
    }


def _agg_metrics(summary: Dict) -> Dict:
    """Pull seed-mean metrics out of a run_phase0_seeds summary into the same shape."""
    a = summary["aggregate"]
    g = lambda k: a[k]["mean"]
    return {
        "amp_tau": g("amplification_tau"), "amp_tau_median": g("amplification_tau_median"),
        "amp_clean": g("amplification_clean"), "selectivity": g("trigger_selectivity"),
        "kv_amp_tau": g("kv_amplification_tau"), "ppl_clean_ratio": g("ppl_clean_ratio"),
        "distance": g("attacker_distance"), "d_T": g("d_T"),
        "dist_margin": round(g("d_T") - g("attacker_distance"), 4),
        "cosine": g("attacker_cosine"), "delta_T": 0.0,
        "pairwise_cosine": g("attacker_pairwise_cosine"),
        "jointly_satisfied": a["jointly_satisfied"]["mean"] >= 0.5,
        "jointly_satisfied_rate": a["jointly_satisfied"]["mean"],
    }


def run_pareto(
    base_config: Dict,
    *,
    gammas: List[float],
    gamma_cleans: List[float],
    kappas: List[float],
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Grid-sweep (gamma x gamma_clean x stealth_kappa) and collect the stealth/amp frontier."""
    cfg0 = default_config()
    cfg0.update(base_config or {})
    # Stamp once here so the sweep summary and every grid point share one run
    # folder; the per-point run_phase0 / run_phase0_seeds calls inherit the stamp.
    cfg0 = stamp_run_subdir(cfg0)
    base_subdir = cfg0.get("results_subdir", "tcaa_phase0")
    rows = []
    for gamma, gamma_clean, kappa in itertools.product(gammas, gamma_cleans, kappas):
        cfg = dict(cfg0)
        cfg.update(gamma=gamma, gamma_clean=gamma_clean, stealth_kappa=kappa,
                   save_figures=False,
                   results_subdir=f"{base_subdir}/pareto/g{gamma}_gc{gamma_clean}_k{kappa}")
        tag = f"gamma={gamma} gamma_clean={gamma_clean} kappa={kappa}"
        print(f"\n########## PARETO POINT  {tag} ##########")
        if seeds:
            metrics = _agg_metrics(run_phase0_seeds(cfg, seeds))
        else:
            metrics = _point_metrics(run_phase0(cfg))
        rows.append({"gamma": gamma, "gamma_clean": gamma_clean, "kappa": kappa, **metrics})

    out_dir = Path("results") / base_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pareto_sweep.json").write_text(json.dumps({"points": rows}, indent=2))
    _print_frontier(rows)
    _save_figure(rows, out_dir / "figures")
    print(f"\n  Pareto sweep written to {out_dir}/pareto_sweep.json")
    return {"points": rows}


def _print_frontier(rows: List[Dict]) -> None:
    print(f"\n{'=' * 92}\nPARETO FRONTIER (amplification vs stealth budget)\n{'=' * 92}")
    hdr = f"  {'gamma':>6} {'g_clean':>7} {'kappa':>6} | {'amp_tau':>8} {'amp_med':>8} {'selec':>6} " \
          f"{'kv_amp':>6} {'ppl':>6} | {'dist':>7} {'d_T':>7} {'margin':>7} {'joint':>6}"
    print(hdr)
    print("  " + "-" * 88)
    for r in sorted(rows, key=lambda x: (x["kappa"], x["gamma"], x["gamma_clean"])):
        joint = "OK" if r["jointly_satisfied"] else "X"
        print(f"  {r['gamma']:>6} {r['gamma_clean']:>7} {r['kappa']:>6} | "
              f"{r['amp_tau']:>8.3f} {r['amp_tau_median']:>8.3f} {r['selectivity']:>6.2f} "
              f"{r['kv_amp_tau']:>6.2f} {r['ppl_clean_ratio']:>6.3f} | "
              f"{r['distance']:>7.3f} {r['d_T']:>7.3f} {r['dist_margin']:>7.3f} {joint:>6}")
    # Best stealthy operating point = max amp_tau_median among jointly-satisfied points.
    ok = [r for r in rows if r["jointly_satisfied"]]
    if ok:
        best = max(ok, key=lambda x: x["amp_tau_median"])
        print("  " + "-" * 88)
        print(f"  BEST STEALTHY POINT: gamma={best['gamma']} gamma_clean={best['gamma_clean']} "
              f"kappa={best['kappa']}  ->  amp_median={best['amp_tau_median']:.3f}x "
              f"(selectivity {best['selectivity']:.2f}x, ppl {best['ppl_clean_ratio']:.3f}x)")
    else:
        print("  NO jointly-satisfied point in the grid -> single-round budget too tight; "
              "escalate to Stage 2 (multi-round accumulation).")
    print("=" * 92)


def _save_figure(rows: List[Dict], out_dir: Path) -> None:
    """Best-effort: save the unified frontier + kappa-tradeoff figures to disk.
    Delegates to tcaa.visualize so the on-disk PNGs match the notebook's inline render.
    Skips gracefully without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from .visualize import make_pareto_figures
    except Exception as e:  # pragma: no cover
        print(f"  [pareto] skipped figure ({e})")
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for key, fig in make_pareto_figures({"points": rows}):
        p = out_dir / f"{key}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [pareto] saved {p}")


def _parse_args():
    p = argparse.ArgumentParser(description="TCAA Stage-1d Pareto sweep")
    p.add_argument("--smoke", action="store_true", help="CPU tiny-GPT2 + synthetic data")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--gammas", type=str, default="1,2,4")
    p.add_argument("--gamma-cleans", type=str, default="0.5")
    p.add_argument("--kappas", type=str, default="0.6,0.8,1.0")
    p.add_argument("--seeds", type=int, default=None, help="seeds per grid point (mean+/-std)")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg: Dict = {}
    if args.smoke:
        cfg.update(smoke_overrides())
        cfg.update(warmup_steps=60, attacker_steps=40, pool_size=160, eval_size=24)
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.source:
        cfg["source"] = args.source
    floats = lambda s: [float(x) for x in s.split(",") if x != ""]
    seeds = None
    if args.seeds:
        base = cfg.get("seed", default_config()["seed"])
        seeds = [base + 1000 * i for i in range(args.seeds)]
    run_pareto(cfg, gammas=floats(args.gammas), gamma_cleans=floats(args.gamma_cleans),
               kappas=floats(args.kappas), seeds=seeds)


if __name__ == "__main__":
    main()
