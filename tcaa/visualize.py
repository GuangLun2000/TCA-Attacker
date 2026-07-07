# tcaa/visualize.py
# Publication-style figures for TCAA results, mapped to the paper's claims.
# Design follows the dataviz skill: pick the form by the data's job; color-blind-safe
# categorical hues in fixed order (Okabe-Ito); single axis (stacked subplots, never
# dual-y); thin marks; recessive grid; legend + direct value labels; text in ink.
#
# Phase-0 figures (render_report):
#   0 summary_dashboard   (all) — three-goal verdict at a glance (headline)
#   1 cost_amplification  (C1)  — triggered inputs cost more; clean unchanged
#   2 length_distribution (C1/C4)— output-length shift on tau vs clean
#   3 utility             (C2)  — perplexity preserved + ROUGE-L recall (answer kept)
#   4 stealth             (C3)  — attacker vs benign envelope (distance / cosine)
#   5 attack_trace        (method)— L_mal, E[L]_tau, EOS prob over optimization
#   6 alm_convergence     (method)— update converges to REST AT the stealth boundary
#   7 cost_model          (C4)  — cost curve C(L) + super-linear threshold
# Multi-round FL (render_fl_report):  fl_durability, fl_stealth
# Pareto sweep    (render_pareto_report): pareto_frontier, pareto_kappa
# summary_html(): a static, self-contained HTML summary card (persists in a notebook).

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

# --- palette (Okabe-Ito; strong CVD separation) & ink -------------------------
C_BASE = "#0072B2"      # baseline (blue)
C_ATK = "#D55E00"       # attacked (vermillion)
C_BENIGN = "#56B4E9"    # benign agents (sky blue)
C_OK = "#009E73"        # feasible / satisfied (green)
C_BAD = "#D55E00"       # infeasible / violated (vermillion, == C_ATK)
C_PURPLE = "#CC79A7"    # tertiary trace (reddish purple)
INK = "#222222"         # primary text
MUTED = "#6b6b6b"       # secondary text / grid


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "font.size": 10, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    })


def _bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8.5, color=INK,
                    xytext=(0, 2), textcoords="offset points")


def _grouped(ax, groups, base_vals, atk_vals, ylabel, *, fmt="{:.1f}", legend=True,
             base_label="Baseline (benign-only global)", atk_label="Attacked (with TCAA agent)"):
    import numpy as np
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w/2, base_vals, w, label=base_label, color=C_BASE)
    b2 = ax.bar(x + w/2, atk_vals, w, label=atk_label, color=C_ATK)
    _bar_labels(ax, b1, fmt); _bar_labels(ax, b2, fmt)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(loc="upper left", fontsize=8.5)
    top = max(base_vals + atk_vals) if (base_vals + atk_vals) else 1.0
    ax.set_ylim(0, top * 1.22 if top > 0 else 1.0)


# --- figures ------------------------------------------------------------------
def fig_summary_dashboard(r: Dict):
    """Headline: the three goals in one row — amplification, utility, stealth verdict.

    All in-figure text is ASCII/English on purpose: matplotlib's default font has no CJK
    glyphs (they render as tofu on Colab), so the Chinese narrative lives in the printed
    tables and the HTML summary card (both render CJK via the terminal / browser)."""
    c, u, s = r["cost"], r["utility"], r["stealth"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))

    # (1) Resource amplification: mean + median cost ratio on tau vs the 1.0 no-effect line.
    ax = axes[0]
    vals = [c["amplification_tau"], c.get("amplification_tau_median", c["amplification_tau"])]
    bars = ax.bar(["mean", "median"], vals, width=0.55, color=[C_ATK, C_PURPLE])
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax.set_title("(1) Resource amplification  C_atk/C_ben (tau)", fontsize=10.5)
    ax.set_ylabel("cost amplification  x")
    ax.set_ylim(0, max(vals + [1.0]) * 1.25); ax.grid(axis="x", visible=False)

    # (2) Performance preserved: ppl ratio (~1) + ROUGE-L recall ratio on tau (~1 = answer kept).
    ax = axes[1]
    ppl_ratio = u["ppl_clean_ratio"]
    rr = u.get("rouge_recall_tau_ratio")
    labels, vals, cols = ["ppl clean\n(~1 good)"], [ppl_ratio], [C_BASE]
    if rr is not None and u.get("rouge_recall_tau_baseline", 0) > 0:
        labels.append("ROUGE recall tau\n(~1 answer kept)"); vals.append(rr); cols.append(C_OK)
    bars = ax.bar(labels, vals, width=0.55, color=cols)
    _bar_labels(ax, bars, fmt="{:.3f}")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax.set_title("(2) Utility preserved", fontsize=10.5)
    ax.set_ylabel("attacked / baseline  x")
    ax.set_ylim(0, max(vals + [1.0]) * 1.3); ax.grid(axis="x", visible=False)

    # (3) Stealth: attacker distance vs the benign budget d_T (bar colored by verdict).
    ax = axes[2]
    dist, d_T = s["attacker_distance"], s["d_T"]
    col = C_OK if s["distance_satisfied"] else C_BAD
    bars = ax.bar(["dist to benign mean"], [dist], width=0.5, color=col)
    _bar_labels(ax, bars, fmt="{:.3f}")
    ax.axhline(d_T, color=MUTED, ls="--", lw=1.2)
    ax.annotate(f"$d_T$={d_T:.3f}", (0.0, d_T), ha="center", va="bottom",
                fontsize=9, color=MUTED)
    ax.axhspan(0, d_T, color=C_OK, alpha=0.07)
    ax.set_title("(3) Stealth (parameter space)", fontsize=10.5)
    ax.set_ylabel("distance  ||d_att - d_g||"); ax.set_ylim(0, max(dist, d_T) * 1.3)
    ax.set_xticks([0]); ax.set_xticklabels(["attacker update"]); ax.grid(axis="x", visible=False)

    joint = s["jointly_satisfied"]
    verdict = "ALL THREE GOALS MET" if joint else "stealth NOT jointly met -> Phase 1"
    fig.suptitle(f"TCAA three-goal summary — {verdict}", fontsize=13, fontweight="bold",
                 color=C_OK if joint else C_BAD)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig


def fig_cost_amplification(r: Dict):
    c = r["cost"]
    base = [c["baseline_clean"]["mean_cost"], c["baseline_tau"]["mean_cost"]]
    atk = [c["attacked_clean"]["mean_cost"], c["attacked_tau"]["mean_cost"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    _grouped(ax, ["Clean", "Triggered (τ)"], base, atk, "Mean inference cost  C")
    ax.set_title("(a) TCAA amplifies cost on triggered inputs; clean unchanged")
    for gx, ratio, strong in [(0, c["amplification_clean"], False), (1, c["amplification_tau"], True)]:
        ax.annotate(
            f"×{ratio:.2f}", (gx, max(base[gx], atk[gx]) * 0.5), ha="center", va="center",
            color=C_ATK if strong else MUTED,
            fontsize=12 if strong else 10, fontweight="bold" if strong else "normal",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec=C_ATK if strong else MUTED, lw=1.1))
    fig.tight_layout(); return fig


def fig_length_distribution(r: Dict):
    d = r.get("distributions", {}).get("output_lens")
    if not d:
        return None
    order = [("baseline_clean", "Clean\nbaseline", C_BASE),
             ("attacked_clean", "Clean\nattacked", C_ATK),
             ("baseline_tau", "τ\nbaseline", C_BASE),
             ("attacked_tau", "τ\nattacked", C_ATK)]
    data = [d[k] for k, _, _ in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    medianprops=dict(color=INK, linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=INK, markersize=5))
    for patch, (_, _, col) in zip(bp["boxes"], order):
        patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor(col)
    ax.set_xticklabels([lbl for _, lbl, _ in order])
    ax.set_ylabel("Output length  L  (tokens)"); ax.grid(axis="x", visible=False)
    ax.set_title("(b) Output-length distribution: τ shifts up, clean stays put")
    fig.tight_layout(); return fig


def fig_utility(r: Dict):
    """Utility preserved on TWO axes: perplexity (unchanged) AND ROUGE-L recall
    (the correct answer content survives even as tau outputs get longer)."""
    u = r["utility"]
    have_rouge = any(u.get(k, 0) for k in (
        "rouge_recall_clean_baseline", "rouge_recall_tau_baseline",
        "rouge_recall_clean_attacked", "rouge_recall_tau_attacked"))
    if not have_rouge:
        # Fallback: perplexity-only (old behavior) when references weren't scored.
        base = [u["ppl_clean_baseline"], u["ppl_tau_baseline"]]
        atk = [u["ppl_clean_attacked"], u["ppl_tau_attacked"]]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        _grouped(ax, ["Clean", "Triggered (τ)"], base, atk, "Perplexity (lower = better)", fmt="{:.2f}")
        ax.set_title(f"(c) Utility preserved  (clean ppl ×{u['ppl_clean_ratio']:.3f})")
        fig.tight_layout(); return fig

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    # left: perplexity (lower = better)
    _grouped(axes[0], ["Clean", "Triggered (τ)"],
             [u["ppl_clean_baseline"], u["ppl_tau_baseline"]],
             [u["ppl_clean_attacked"], u["ppl_tau_attacked"]],
             "Perplexity (lower = better)", fmt="{:.2f}")
    axes[0].set_title(f"Perplexity preserved  (clean ×{u['ppl_clean_ratio']:.3f})", fontsize=11)
    # right: ROUGE-L recall (higher = more of the reference answer covered; length-robust)
    _grouped(axes[1], ["Clean", "Triggered (τ)"],
             [u.get("rouge_recall_clean_baseline", 0.0), u.get("rouge_recall_tau_baseline", 0.0)],
             [u.get("rouge_recall_clean_attacked", 0.0), u.get("rouge_recall_tau_attacked", 0.0)],
             "ROUGE-L recall (higher = answer kept)", fmt="{:.3f}", legend=False)
    tr = u.get("rouge_recall_tau_ratio")
    axes[1].set_title(f"Answer content kept  (τ recall ×{tr:.3f})" if tr else "Answer content kept",
                      fontsize=11)
    fig.suptitle("(c) Utility preserved: same perplexity AND the correct answer still covered",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig


def fig_stealth(r: Dict):
    import numpy as np
    s = r["stealth"]
    dist = r.get("distributions", {})
    bdist = dist.get("benign_distances", [])
    bcos = dist.get("benign_cosines", [])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))

    # distance panel: benign points + attacker + d_T threshold (feasible = below)
    ax = axes[0]
    if bdist:
        ax.scatter(np.random.default_rng(0).normal(0, 0.04, len(bdist)), bdist,
                   color=C_BENIGN, s=45, label="Benign agents", zorder=3)
    ax.scatter([0], [s["attacker_distance"]], marker="*", s=300, color=C_ATK,
               edgecolor=INK, linewidth=0.6, label="TCAA attacker", zorder=4)
    ax.axhline(s["d_T"], color=MUTED, ls="--", lw=1.2)
    ax.annotate(f"$d_T$={s['d_T']:.2f}", (-0.46, s["d_T"]), color=MUTED, fontsize=9,
                va="bottom", ha="left")
    ax.axhspan(0, s["d_T"], color=C_OK, alpha=0.07)
    verdict = "inside envelope ✓" if s["distance_satisfied"] else "OUTSIDE envelope ✗"
    ax.set_title(f"Distance  ({verdict})", fontsize=11)
    ax.set_ylabel("‖Δ_att − Δ_g‖"); ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="x", visible=False)

    # cosine panel: feasible = above delta_T
    ax = axes[1]
    if bcos:
        ax.scatter(np.random.default_rng(1).normal(0, 0.04, len(bcos)), bcos,
                   color=C_BENIGN, s=45, zorder=3)
    ax.scatter([0], [s["attacker_cosine"]], marker="*", s=300, color=C_ATK,
               edgecolor=INK, linewidth=0.6, zorder=4)
    ax.axhline(s["delta_T"], color=MUTED, ls="--", lw=1.2)
    ax.annotate(f"$δ_T$={s['delta_T']:.2f}", (-0.46, s["delta_T"]), color=MUTED,
                fontsize=9, va="bottom", ha="left")
    vc = "inside ✓" if s["cosine_satisfied"] else "OUTSIDE ✗"
    ax.set_title(f"Cosine  ({vc})", fontsize=11)
    ax.set_ylabel("cos(Δ_att, Δ_g)"); ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="x", visible=False)

    # single shared legend at the bottom (both panels share the two series)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", ls="none", markerfacecolor=C_BENIGN,
               markeredgecolor="none", markersize=9, label="Benign agents"),
        Line2D([0], [0], marker="*", ls="none", markerfacecolor=C_ATK,
               markeredgecolor=INK, markersize=15, label="TCAA attacker"),
        Line2D([0], [0], ls="--", color=MUTED, label="Constraint threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    joint = "JOINTLY SATISFIED" if s["jointly_satisfied"] else "NOT jointly satisfied → Phase 1"
    fig.suptitle(f"(c) Parameter-space stealth — {joint}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95]); return fig


def fig_attack_trace(r: Dict):
    tr = r.get("mal_trace", [])
    if not tr:
        return None
    steps = [t["step"] for t in tr]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)
    axes[0].plot(steps, [t["L_mal"] for t in tr], color=C_ATK, lw=2, marker="o", ms=3)
    axes[0].set_ylabel("$L_{mal}$"); axes[0].set_title("Malicious-loss optimization trace")
    axes[1].plot(steps, [t["E_len_tau"] for t in tr], color=C_ATK, lw=2, marker="o", ms=3,
                 label="τ (target: ↑)")
    # Clean anchor: E[L] on clean should stay LOW/flat — visualizes trigger selectivity.
    clean_vals = [t.get("E_len_clean") for t in tr]
    if any(v is not None for v in clean_vals):
        xs = [s for s, v in zip(steps, clean_vals) if v is not None]
        ys = [v for v in clean_vals if v is not None]
        axes[1].plot(xs, ys, color=C_BASE, lw=2, marker="s", ms=3, label="clean (anchor: flat)")
        axes[1].legend(fontsize=8.5, loc="best")
    axes[1].set_ylabel("$E[L]$  (τ↑ / clean flat)")
    axes[2].plot(steps, [t["mean_eos_prob_tau"] for t in tr], color=C_PURPLE, lw=2, marker="o", ms=3)
    axes[2].set_ylabel("mean $q_{EOS}$ on τ  (↓)"); axes[2].set_xlabel("optimization step")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def fig_alm_convergence(r: Dict):
    """Process evidence: the ALM-constrained update converges to REST AT the stealth
    boundary (distance rides down to d_T; pairwise cosine rides up to the benign floor)."""
    tr = [t for t in r.get("mal_trace", []) if "dist" in t and "g_dist" in t]
    if not tr:
        return None
    steps = [t["step"] for t in tr]
    dists = [t["dist"] for t in tr]
    # d_T = dist - g_dist ; cos_low = cos + g_sim  (both constant; take last for the line)
    d_T = tr[-1]["dist"] - tr[-1]["g_dist"]
    coss = [t["cos"] for t in tr]
    cos_low = tr[-1]["cos"] + tr[-1]["g_sim"]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.8), sharex=True)
    ax = axes[0]
    ax.plot(steps, dists, color=C_ATK, lw=2, marker="o", ms=3, label="attacker distance")
    ax.axhline(d_T, color=MUTED, ls="--", lw=1.2, label="$d_T$ (benign budget)")
    ax.axhspan(0, d_T, color=C_OK, alpha=0.07)
    ax.set_ylabel("‖Δ_att − Δ_g‖"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="best")
    ax.set_title("ALM stealth-constraint convergence (rests at the boundary)")

    ax = axes[1]
    ax.plot(steps, coss, color=C_BASE, lw=2, marker="s", ms=3, label="attacker pairwise cosine")
    ax.axhline(cos_low, color=MUTED, ls="--", lw=1.2, label="benign cosine floor")
    ymin = min(coss + [cos_low]); ax.axhspan(cos_low, 1.0, color=C_OK, alpha=0.07)
    ax.set_ylabel("cosine to benign"); ax.set_xlabel("optimization step")
    ax.set_ylim(min(ymin - 0.05, cos_low - 0.05), 1.02)
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout(); return fig


def fig_cost_model(r: Dict):
    import numpy as np
    from .cost_model import inference_cost, superlinear_threshold
    dist = r.get("distributions", {})
    c_f, c_a = dist.get("c_f", 1.0), dist.get("c_a", 1.0)
    n = r["cost"]["baseline_tau"].get("mean_prompt_len", 0.0)
    Lmax = max(r["cost"]["attacked_tau"]["mean_output_len"] * 1.6, 32)
    L = np.linspace(0, Lmax, 200)
    C = [inference_cost(n, x, c_f, c_a) for x in L]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(L, C, color=INK, lw=2, label="C(L) at prompt len n")
    Lb = r["cost"]["baseline_tau"]["mean_output_len"]
    La = r["cost"]["attacked_tau"]["mean_output_len"]
    ax.scatter([Lb], [inference_cost(n, Lb, c_f, c_a)], color=C_BASE, s=70, zorder=4, label=f"baseline τ (L={Lb:.0f})")
    ax.scatter([La], [inference_cost(n, La, c_f, c_a)], color=C_ATK, s=70, zorder=4, label=f"attacked τ (L={La:.0f})")
    Lstar = superlinear_threshold(n, c_f, c_a)
    if Lstar <= Lmax:
        ax.axvline(Lstar, color=MUTED, ls=":", lw=1.2)
        ax.annotate("super-linear\nthreshold", (Lstar, max(C) * 0.1), color=MUTED,
                    fontsize=8.5, ha="left")
    ax.set_xlabel("Output length  L"); ax.set_ylabel("Per-request cost  C")
    ax.set_title("(d) Cost model: attack pushes L along the C(L) curve")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); return fig


_BUILDERS = [
    ("summary_dashboard", fig_summary_dashboard),
    ("cost_amplification", fig_cost_amplification),
    ("length_distribution", fig_length_distribution),
    ("utility", fig_utility),
    ("stealth", fig_stealth),
    ("attack_trace", fig_attack_trace),
    ("alm_convergence", fig_alm_convergence),
    ("cost_model", fig_cost_model),
]


def make_all_figures(results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in _BUILDERS:
        try:
            fig = fn(results)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def save_all_figures(results: Dict, out_dir) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fig in make_all_figures(results):
        p = out_dir / f"{key}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


_PHASE0_TITLES = {
    "summary_dashboard": "三目标总览 (放大/效用/隐蔽)",
    "cost_amplification": "成本放大 (C1)", "length_distribution": "输出长度分布 (C1/C4)",
    "utility": "效用保持：困惑度 + ROUGE 召回 (C2)", "stealth": "参数空间隐蔽性 (C3)",
    "attack_trace": "攻击优化轨迹 (方法)", "alm_convergence": "ALM 隐蔽约束收敛 (方法/过程)",
    "cost_model": "成本模型 (C4)",
}


def render_report(results: Dict):
    """For notebooks: build every Phase-0 figure and show it inline (Colab/Jupyter)."""
    figs = make_all_figures(results)
    for key, fig in figs:
        print(f"\n=== {_PHASE0_TITLES.get(key, key)} ===")
        plt.figure(fig.number)
        plt.show()
    return figs


# --------------------------------------------------------------------------- #
# Multi-round FL figures (from fl_runner results)                             #
# --------------------------------------------------------------------------- #
def fig_fl_durability(r: Dict):
    """Durability: cost amplification and tau output length vs communication round.
    Stacked single-axis subplots (NO dual-y) so the two series are not visually conflated."""
    dur = r.get("durability", [])
    if not dur:
        return None
    rounds = [p["round"] for p in dur]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    ax = axes[0]
    ax.plot(rounds, [p["amp_tau"] for p in dur], "-o", color=C_BASE, lw=2, ms=4, label="amp_tau (mean)")
    ax.plot(rounds, [p["amp_tau_median"] for p in dur], "--s", color=C_BENIGN, lw=2, ms=4,
            label="amp_tau (median)")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylabel("cost amplification  C_atk/C_ben (τ)")
    ax.set_title("TCAA multi-round durability (amplification accumulation)")
    ax.legend(loc="best", fontsize=8.5); ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(rounds, [p["tau_len_atk"] for p in dur], "-^", color=C_ATK, lw=2, ms=4, label="τ output len")
    ax.plot(rounds, [p.get("clean_len_atk", float("nan")) for p in dur], "-o", color=C_BASE, lw=1.6,
            ms=3, label="clean output len")
    ax.set_ylabel("mean output length  L"); ax.set_xlabel("communication round")
    ax.legend(loc="best", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def fig_fl_stealth(r: Dict):
    """Per-round stealth under client sampling: attacker distance vs the benign budget d_T,
    markers colored by whether stealth was jointly satisfied that round."""
    st = [s for s in r.get("stealth_trace", []) if s.get("n_attackers")]
    if not st:
        return None
    rounds = [s["round"] for s in st]
    dists = [s["attacker_distance"] for s in st]
    d_Ts = [s["d_T"] for s in st]
    ok = [bool(s["jointly_satisfied"]) for s in st]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(rounds, d_Ts, color=MUTED, ls="--", lw=1.4, label="$d_T$ (benign budget)")
    ax.plot(rounds, dists, color=INK, lw=1.0, alpha=0.5, zorder=1)
    okx = [x for x, f in zip(rounds, ok) if f]; oky = [y for y, f in zip(dists, ok) if f]
    bad = [x for x, f in zip(rounds, ok) if not f]; bady = [y for y, f in zip(dists, ok) if not f]
    if okx:
        ax.scatter(okx, oky, color=C_OK, s=55, zorder=3, label="stealth satisfied")
    if bad:
        ax.scatter(bad, bady, color=C_BAD, marker="X", s=65, zorder=3, label="stealth violated")
    n_ok = sum(ok)
    ax.set_xlabel("communication round"); ax.set_ylabel("attacker distance  ‖Δ_att − Δ_g‖")
    ax.set_title(f"Per-round stealth ({n_ok}/{len(st)} attacker rounds jointly satisfied)")
    ax.legend(loc="best", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def make_fl_figures(fl_results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in (("fl_durability", fig_fl_durability), ("fl_stealth", fig_fl_stealth)):
        try:
            fig = fn(fl_results)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def render_fl_report(fl_results: Dict):
    titles = {"fl_durability": "多轮放大 durability (成本累积)",
              "fl_stealth": "逐轮隐蔽性 (客户端采样下)"}
    figs = make_fl_figures(fl_results)
    for key, fig in figs:
        print(f"\n=== {titles.get(key, key)} ===")
        plt.figure(fig.number)
        plt.show()
    return figs


# --------------------------------------------------------------------------- #
# Pareto sweep figures (from pareto_runner rows)                              #
# --------------------------------------------------------------------------- #
def _pareto_rows(pareto) -> List[Dict]:
    return pareto["points"] if isinstance(pareto, dict) and "points" in pareto else list(pareto)


def fig_pareto_frontier(pareto):
    """Amplification (median) vs distance margin, colored by joint stealth — the frontier
    that decides whether the single-round constrained attack is viable."""
    rows = _pareto_rows(pareto)
    if not rows:
        return None
    ok = [r for r in rows if r.get("jointly_satisfied")]
    no = [r for r in rows if not r.get("jointly_satisfied")]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if no:
        ax.scatter([r["dist_margin"] for r in no], [r["amp_tau_median"] for r in no],
                   c=C_BAD, marker="x", s=60, label="stealth violated", zorder=3)
    if ok:
        ax.scatter([r["dist_margin"] for r in ok], [r["amp_tau_median"] for r in ok],
                   c=C_OK, marker="o", s=60, label="stealth satisfied", zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.axvline(0.0, color=MUTED, lw=0.9, ls="--")
    ax.set_xlabel("distance margin  (d_T − attacker_distance;  >0 = inside budget)")
    ax.set_ylabel("cost amplification τ (median, cap-robust)")
    ax.set_title("TCAA stealth-constrained amplification frontier")
    ax.legend(loc="best", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def fig_pareto_kappa(pareto):
    """Amplification vs the stealth budget kappa: how fast amp collapses as the budget
    tightens (one line per gamma). The core stealth-vs-amplification trade-off."""
    import numpy as np
    rows = _pareto_rows(pareto)
    if not rows or not all("kappa" in r for r in rows):
        return None
    gammas = sorted({r["gamma"] for r in rows})
    palette = [C_ATK, C_BASE, C_OK, C_PURPLE, MUTED]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plotted = False
    for i, g in enumerate(gammas):
        pts = sorted([r for r in rows if r["gamma"] == g], key=lambda x: x["kappa"])
        if len(pts) < 2:
            continue
        ax.plot([p["kappa"] for p in pts], [p["amp_tau_median"] for p in pts],
                "-o", color=palette[i % len(palette)], lw=2, ms=5, label=f"γ={g}")
        # mark jointly-satisfied points with a ring
        okp = [p for p in pts if p.get("jointly_satisfied")]
        if okp:
            ax.scatter([p["kappa"] for p in okp], [p["amp_tau_median"] for p in okp],
                       facecolors="none", edgecolors=C_OK, s=130, linewidths=1.6, zorder=4)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.set_xlabel("stealth budget  κ  (fraction of benign distance envelope)")
    ax.set_ylabel("cost amplification τ (median)")
    ax.set_title("Amplification vs stealth budget (green ring = jointly stealthy)")
    ax.legend(loc="best", fontsize=8.5, title="attack strength"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def make_pareto_figures(pareto) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in (("pareto_frontier", fig_pareto_frontier), ("pareto_kappa", fig_pareto_kappa)):
        try:
            fig = fn(pareto)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def render_pareto_report(pareto):
    titles = {"pareto_frontier": "放大-隐蔽前沿", "pareto_kappa": "放大 vs 隐蔽预算 κ 权衡"}
    figs = make_pareto_figures(pareto)
    for key, fig in figs:
        print(f"\n=== {titles.get(key, key)} ===")
        plt.figure(fig.number)
        plt.show()
    return figs


# --------------------------------------------------------------------------- #
# Static HTML summary card (persists as text/html output in a saved notebook)  #
# --------------------------------------------------------------------------- #
def summary_html(results: Dict) -> str:
    """A self-contained, inline-styled HTML summary of the three goals. No external
    resources / scripts, so it renders (and persists) inside a Colab/Jupyter cell even
    after the GPU runtime is released. Use: display(HTML(summary_html(results)))."""
    c, u, s = results["cost"], results["utility"], results["stealth"]
    cfg = results.get("config", {})
    amp = c["amplification_tau"]; amp_med = c.get("amplification_tau_median", amp)
    sel = c.get("trigger_selectivity", float("nan"))
    ppl_ratio = u["ppl_clean_ratio"]
    rr_tau = u.get("rouge_recall_tau_ratio")
    joint = bool(s["jointly_satisfied"])

    def tile(border, title, big, sub):
        return (
            f'<div style="flex:1;min-width:180px;border:1px solid #e6e6e6;border-left:5px solid {border};'
            f'border-radius:8px;padding:14px 16px;background:#fff;">'
            f'<div style="font-size:12px;color:#6b6b6b;font-weight:600;letter-spacing:.02em;">{title}</div>'
            f'<div style="font-size:26px;font-weight:700;color:#222;margin:4px 0 2px;">{big}</div>'
            f'<div style="font-size:12px;color:#6b6b6b;">{sub}</div></div>')

    ok_c, bad_c = "#009E73", "#D55E00"
    amp_tile = tile(bad_c, "① 资源放大 (τ)", f"×{amp:.2f}",
                    f"中位 ×{amp_med:.2f} · 触发选择性 ×{sel:.2f}")
    util_sub = f"clean ppl ×{ppl_ratio:.3f}"
    if rr_tau is not None and u.get("rouge_recall_tau_baseline", 0) > 0:
        util_sub += f" · τ ROUGE召回 ×{rr_tau:.3f}"
    util_ok = abs(ppl_ratio - 1.0) <= 0.05
    util_tile = tile(ok_c if util_ok else bad_c, "② 性能保持", f"{ppl_ratio:.3f}×", util_sub)
    stealth_big = "达成 ✓" if joint else "未达成 ✗"
    stealth_tile = tile(ok_c if joint else bad_c, "③ 隐蔽性 (参数空间)", stealth_big,
                        f"距离 {s['attacker_distance']:.3f} / d_T {s['d_T']:.3f}"
                        f" · 余弦 {s['attacker_cosine']:.3f}")

    banner_c = ok_c if joint else bad_c
    banner_txt = "三目标联合达成" if joint else "隐蔽约束未联合满足（需 Phase-1 / 多轮）"
    head = (
        f'<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        f'max-width:900px;">'
        f'<div style="font-size:16px;font-weight:700;color:#222;margin-bottom:2px;">'
        f'TCAA 结果总览 · {cfg.get("backbone","?")} + {cfg.get("source","?")}</div>'
        f'<div style="display:inline-block;font-size:12px;font-weight:700;color:#fff;'
        f'background:{banner_c};border-radius:12px;padding:3px 12px;margin-bottom:12px;">{banner_txt}</div>')
    tiles = (f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{amp_tile}{util_tile}{stealth_tile}</div>')
    note = (
        '<div style="font-size:11px;color:#6b6b6b;margin-top:10px;">'
        '注：放大为解析成本模型下的比值；ROUGE-L 召回对加长稳健，≈1 表示答案内容仍在；'
        '隐蔽性为参数空间距离/余弦落入良性包络。</div></div>')
    return head + tiles + note
