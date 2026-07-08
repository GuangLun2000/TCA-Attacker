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
    """Durability: cost amplification + tau output length + truncation vs communication round.
    Three stacked single-axis subplots (NO dual-y). The de-censored curves correct the
    max_new_tokens cap (a saturated round would otherwise read as a flat lower bound), and
    the truncation panel is kept BESIDE the amplification so the censoring is always explicit."""
    dur = r.get("durability", [])
    if not dur:
        return None
    rounds = [p["round"] for p in dur]
    has_dec = any(p.get("amp_tau_decensored") is not None for p in dur)
    has_len_dec = any(p.get("tau_len_atk_decensored") is not None for p in dur)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)
    # --- (A) cost amplification: capped (solid) + de-censored (dashed) + median ---
    ax = axes[0]
    ax.plot(rounds, [p["amp_tau"] for p in dur], "-o", color=C_BASE, lw=2, ms=4,
            label="amp_tau mean (capped)")
    if has_dec:
        ax.plot(rounds, [p.get("amp_tau_decensored", float("nan")) for p in dur], "--D", color=C_ATK,
                lw=2, ms=4, label="amp_tau mean (de-censored)")
    ax.plot(rounds, [p["amp_tau_median"] for p in dur], ":s", color=C_BENIGN, lw=1.8, ms=4,
            label="amp_tau median")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylabel("cost amplification (τ)")
    ax.set_title("TCAA multi-round durability (amplification accumulation)")
    ax.legend(loc="best", fontsize=8.0); ax.grid(axis="x", visible=False)
    # --- (B) tau output length: capped + de-censored + clean ---
    ax = axes[1]
    ax.plot(rounds, [p["tau_len_atk"] for p in dur], "-^", color=C_ATK, lw=2, ms=4,
            label="τ len (capped at max_new_tokens)")
    if has_len_dec:
        ax.plot(rounds, [p.get("tau_len_atk_decensored", float("nan")) for p in dur], "--D", color=C_PURPLE,
                lw=1.8, ms=3, label="τ len (de-censored estimate)")
    ax.plot(rounds, [p.get("clean_len_atk", float("nan")) for p in dur], "-o", color=C_BASE, lw=1.6,
            ms=3, label="clean len")
    ax.set_ylabel("mean output length  L")
    ax.legend(loc="best", fontsize=8.0); ax.grid(axis="x", visible=False)
    # --- (C) truncation rate: how censored the capped amplification above is ---
    ax = axes[2]
    ax.plot(rounds, [p.get("truncation_tau", float("nan")) for p in dur], "-^", color=C_ATK,
            lw=2, ms=4, label="τ truncation (cap-hit → capped amp is a lower bound)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("truncation rate  [0,1]"); ax.set_xlabel("communication round")
    ax.legend(loc="best", fontsize=8.0); ax.grid(axis="x", visible=False)
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


def fig_fl_utility(r: Dict):
    """Utility across rounds vs the FIXED pristine (round-0) baseline — the unconfounded
    evidence for 'utility-preserving' UNDER ACCUMULATION. Top: clean ppl RATIO vs pristine
    (not vs the benign-only global, which itself drifts and confounds the ratio) with the
    1.0=preserved line and a ±5% band. Bottom: ROUGE-L recall on clean AND τ (attacked vs
    pristine dashed) — recall near pristine means the correct answer is still present even
    as τ outputs run long — plus τ repetition (a 'long' output that is a loop is weak,
    detectable amplification). Falls back to the old atk/benign ratio for legacy results."""
    dur = r.get("durability", [])
    if not dur:
        return None
    rounds = [p["round"] for p in dur]
    pri = r.get("pristine_reference", {})
    have_pri = all(p.get("ppl_ratio_vs_pristine") is not None for p in dur)
    have_rouge = any(p.get("rouge_recall_clean_atk") is not None for p in dur)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    # --- (A) clean ppl ratio vs the fixed pristine baseline ---
    ax = axes[0]
    ax.axhspan(0.95, 1.05, color=C_OK, alpha=0.12, zorder=0, label="±5% (preserved)")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    if have_pri:
        ax.plot(rounds, [p["ppl_ratio_vs_pristine"] for p in dur], "-o", color=C_BASE, lw=2, ms=4,
                label="clean ppl ratio  (attacked / pristine r0)")
        ax.set_ylabel("ppl ratio  (atk / pristine)")
        ax.set_title("Utility vs pristine (round-0) baseline — unconfounded by benign drift")
    else:  # legacy JSON without pristine fields
        ax.plot(rounds, [p.get("ppl_ratio", float("nan")) for p in dur], "-o", color=C_BASE, lw=2, ms=4,
                label="clean ppl ratio  (attacked / benign)")
        ax.set_ylabel("ppl ratio  (atk / benign)")
        ax.set_title("Utility preservation across rounds (1.0 = preserved)")
    ax.legend(loc="best", fontsize=8.0); ax.grid(axis="x", visible=False)
    # --- (B) ROUGE-L recall (answer still correct?) clean + τ, vs pristine; + repetition ---
    ax = axes[1]
    if have_rouge:
        ax.plot(rounds, [p.get("rouge_recall_clean_atk", float("nan")) for p in dur], "-o", color=C_BASE,
                lw=2, ms=4, label="clean ROUGE-L recall (atk)")
        ax.plot(rounds, [p.get("rouge_recall_tau_atk", float("nan")) for p in dur], "-^", color=C_OK,
                lw=2, ms=4, label="τ ROUGE-L recall (atk)")
        if pri.get("rouge_recall_clean") is not None:
            ax.axhline(pri["rouge_recall_clean"], color=C_BASE, ls="--", lw=1.0, alpha=0.7,
                       label="clean ROUGE (pristine)")
        if pri.get("rouge_recall_tau") is not None:
            ax.axhline(pri["rouge_recall_tau"], color=C_OK, ls="--", lw=1.0, alpha=0.7,
                       label="τ ROUGE (pristine)")
    ax.plot(rounds, [p.get("repetition_tau", float("nan")) for p in dur], "-s", color=C_PURPLE,
            lw=1.8, ms=4, label="τ repetition (degeneracy)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("ROUGE-L recall / rate  [0,1]"); ax.set_xlabel("communication round")
    ax.legend(loc="best", fontsize=7.5, ncol=2); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def make_fl_figures(fl_results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in (("fl_durability", fig_fl_durability), ("fl_utility", fig_fl_utility),
                    ("fl_stealth", fig_fl_stealth)):
        try:
            fig = fn(fl_results)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def render_fl_report(fl_results: Dict):
    titles = {"fl_durability": "多轮放大 durability (成本累积 · 含去删失估计与截断率)",
              "fl_utility": "多轮效用保持 (vs 原始骨干绝对基线 · ppl + ROUGE)",
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


def fig_pareto_utility(pareto):
    """The 'usable' view of the sweep: a point is only a real TCAA win if amplification is
    high AND clean utility is preserved (ppl_clean_ratio ≈ 1) AND it stays stealthy. Plots
    amp (median) vs clean ppl ratio; color = joint stealth; marker size ∝ trigger
    selectivity. The top-right-of-the-1.0-line region is the genuinely usable frontier."""
    rows = _pareto_rows(pareto)
    if not rows or not all("ppl_clean_ratio" in r for r in rows):
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    plotted = False
    for cond, color, marker, lab in ((True, C_OK, "o", "stealth satisfied"),
                                     (False, C_BAD, "X", "stealth violated")):
        pts = [r for r in rows if bool(r.get("jointly_satisfied")) == cond]
        if not pts:
            continue
        sizes = [30 + 45 * float(r.get("selectivity", 1.0)) for r in pts]
        ax.scatter([r["ppl_clean_ratio"] for r in pts], [r["amp_tau_median"] for r in pts],
                   c=color, marker=marker, s=sizes, alpha=0.8, zorder=3, label=lab)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.axvline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.set_xlabel("clean ppl ratio  (attacked / baseline;  ≈1 = utility preserved)")
    ax.set_ylabel("cost amplification τ (median)")
    ax.set_title("Usable frontier: amplification vs preserved utility (size ∝ selectivity)")
    ax.legend(loc="best", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def make_pareto_figures(pareto) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in (("pareto_frontier", fig_pareto_frontier), ("pareto_kappa", fig_pareto_kappa),
                    ("pareto_utility", fig_pareto_utility)):
        try:
            fig = fn(pareto)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def render_pareto_report(pareto):
    titles = {"pareto_frontier": "放大-隐蔽前沿", "pareto_kappa": "放大 vs 隐蔽预算 κ 权衡",
              "pareto_utility": "可用前沿：放大 vs 效用保持 (点大小∝选择性)"}
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


# --------------------------------------------------------------------------- #
# Consolidated copy-pasteable digest (for review / feedback loop)             #
# --------------------------------------------------------------------------- #
def _f(x, spec="", dash="?"):
    """Format that tolerates None (missing metric) so the digest never crashes."""
    if x is None:
        return dash
    try:
        return format(x, spec) if spec else str(x)
    except (TypeError, ValueError):
        return str(x)


def tuning_hints(phase0: Optional[Dict] = None, fl: Optional[Dict] = None,
                 pareto=None) -> List[str]:
    """Auto-generate concrete next-step tuning suggestions from the measured numbers, so a
    text-only run (Colab output copied back) carries everything needed to steer the next
    experiment toward the three goals: preserve utility / large consumption / stealth.
    Returns a list of hint lines (empty if a metric is missing). Pure heuristics — each
    hint names the knob and the direction."""
    hints: List[str] = []

    # ---- pull the most-advanced-round attacked numbers (prefer FL last round) ----
    rep = distinct = amp = amp_eff = amp_dec = ppl_pri = rouge_cln = rouge_cln_ref = None
    sel = amp_clean = trunc = joint = kappa = None
    if fl and fl.get("durability"):
        d = fl["durability"][-1]; pri = fl.get("pristine_reference", {})
        rep, distinct = d.get("repetition_tau"), d.get("distinct_ratio_tau")
        amp, amp_eff, amp_dec = d.get("amp_tau"), d.get("amp_tau_effective"), d.get("amp_tau_decensored")
        ppl_pri = d.get("ppl_ratio_vs_pristine")
        rouge_cln, rouge_cln_ref = d.get("rouge_recall_clean_atk"), pri.get("rouge_recall_clean")
        sel, amp_clean, trunc, joint = d.get("selectivity"), d.get("amp_clean"), d.get("truncation_tau"), d.get("stealth_ok")
    elif phase0:
        c, u, s = phase0.get("cost", {}), phase0.get("utility", {}), phase0.get("stealth", {})
        at, bt = c.get("attacked_tau", {}), c.get("baseline_tau", {})
        rep, distinct = at.get("mean_repetition"), at.get("mean_distinct_ratio")
        amp, amp_eff = c.get("amplification_tau"), c.get("effective_amplification_tau")
        if at.get("decensored_mean_cost") and bt.get("decensored_mean_cost"):
            amp_dec = at["decensored_mean_cost"] / max(bt["decensored_mean_cost"], 1e-9)
        ppl_pri = u.get("ppl_clean_ratio")
        rouge_cln, rouge_cln_ref = u.get("rouge_recall_clean_attacked"), u.get("rouge_recall_clean_baseline")
        sel, amp_clean, trunc, joint = c.get("trigger_selectivity"), c.get("amplification_clean"), at.get("truncation_rate"), s.get("jointly_satisfied")

    def has(*xs): return all(x is not None for x in xs)

    # ---- degeneracy: is the extra length USEFUL? ----
    if has(rep) and rep > 0.20:
        hints.append(f"退化复读偏高 (rep={rep:.2f} > 0.20) → 提高 gamma_rep (×2, 如 0.2→0.4) 或增大 no_repeat_ngram_size(3→4)。")
    if has(distinct) and distinct < 0.65:
        hints.append(f"distinct 比偏低 ({distinct:.2f} < 0.65,输出趋于循环) → 提高 gamma_rep;若已很高再降 gamma。")
    if has(amp, amp_eff) and amp > 1.2 and amp_eff < 0.6 * amp:
        hints.append(f"放大主要来自复读而非真实长度 (amp={amp:.2f} 但 eff={amp_eff:.2f}) → 提高 gamma_rep,让长度变得有用。")

    # ---- utility preservation (vs pristine) ----
    if has(ppl_pri) and (ppl_pri < 0.93 or ppl_pri > 1.07):
        hints.append(f"clean 困惑度偏离 pristine (ratio={ppl_pri:.3f},超出 ±7%) → 提高 kd_clean_weight (1→2/4) 稳住效用。")
    if has(rouge_cln, rouge_cln_ref) and rouge_cln < 0.85 * rouge_cln_ref:
        hints.append(f"clean 答案质量下降 (ROUGE {rouge_cln:.2f} vs pristine {rouge_cln_ref:.2f}) → 提高 kd_clean_weight 或略降 gamma。")

    # ---- consumption / truncation ----
    if has(amp) and amp < 2.0:
        hints.append(f"放大不足 (amp={amp:.2f} < 2x) → 提高 onpolicy_horizon(消耗杠杆,如 256→384)或 gamma;注意 runtime。")
    if has(trunc) and trunc > 0.6:
        hints.append(f"截断率仍高 (trunc={trunc:.2f}) → 真实上限被 cap 遮住:去删失估计已给出,如需实测可再提高 max_new_tokens。")

    # ---- selectivity / clean leakage ----
    if has(sel) and sel < 1.3:
        hints.append(f"触发选择性弱 (selectivity={sel:.2f} < 1.3,clean 也被拉长) → 提高 gamma_clean 抑制泄漏。")
    if has(amp_clean) and amp_clean < 0.85:
        hints.append(f"clean 被过度压短 (amp_clean={amp_clean:.2f} < 0.85) → 降低 gamma_clean,别把干净输出砍短。")

    # ---- stealth ----
    if joint is False:
        hints.append("隐蔽未联合满足 → 降低 stealth_kappa(如 1.0→0.8)把更新收进良性包络内。")

    # ---- pareto best point ----
    if pareto:
        rows = pareto["points"] if isinstance(pareto, dict) and "points" in pareto else list(pareto)
        ok = [r for r in rows if r.get("jointly_satisfied")]
        if ok:
            best = max(ok, key=lambda r: r.get("amp_tau_median") or 0)
            hints.append(f"Pareto 最优隐蔽点: gamma={_f(best.get('gamma'))} kappa={_f(best.get('kappa'))} "
                         f"(amp_med={_f(best.get('amp_tau_median'))}) → 下轮以此为中心细扫。")
        else:
            hints.append("Pareto 无联合满足点 → 降低 kappa 或加大多轮累积(实验 B)。")

    if not hints:
        hints.append("各指标在合理区间;可小幅提高 onpolicy_horizon/gamma 追求更大消耗,同时盯住 rep 与 ppl_pri。")
    return hints


def _digest_lines(phase0: Optional[Dict] = None, fl: Optional[Dict] = None,
                  pareto=None) -> List[str]:
    """Build the compact digest as a list of lines (no printing). Shared by
    feedback_digest (prints + returns text) and full_report (embeds it)."""
    L: List[str] = []
    p = L.append
    p("=" * 74)
    p("TCAA FEEDBACK DIGEST  —  copy this WHOLE block back for review")
    p("=" * 74)

    if phase0:
        c, u, s = phase0["cost"], phase0["utility"], phase0["stealth"]
        cfg = phase0.get("config", {})
        bt, at = c["baseline_tau"], c["attacked_tau"]
        p(f"[A] SINGLE-ROUND  {cfg.get('backbone','?')} + {cfg.get('source','?')}  "
          f"gamma={_f(cfg.get('gamma'))} gamma_clean={_f(cfg.get('gamma_clean'))} "
          f"kd={_f(cfg.get('kd_clean_weight', 0))} steps={_f(cfg.get('attacker_steps'))} "
          f"max_new={_f(cfg.get('max_new_tokens'))}")
        dec_amp = None
        if at.get("decensored_mean_cost") and bt.get("decensored_mean_cost"):
            dec_amp = round(at["decensored_mean_cost"] / max(bt["decensored_mean_cost"], 1e-9), 3)
        p(f"    amp_tau mean={_f(c.get('amplification_tau'))} med={_f(c.get('amplification_tau_median'))} "
          f"dec(cap-corr)={_f(dec_amp)} eff(useful)={_f(c.get('effective_amplification_tau'))} "
          f"clean={_f(c.get('amplification_clean'))} selectivity={_f(c.get('trigger_selectivity'))} "
          f"kv={_f(c.get('kv_amplification_tau'))}")
        p(f"    len_tau {_f(bt.get('mean_output_len'))}->{_f(at.get('mean_output_len'))} "
          f"(effective {_f(bt.get('mean_effective_len'))}->{_f(at.get('mean_effective_len'))})  "
          f"trunc {_f(bt.get('truncation_rate'))}->{_f(at.get('truncation_rate'))}  "
          f"rep {_f(bt.get('mean_repetition'))}->{_f(at.get('mean_repetition'))}  "
          f"distinct {_f(bt.get('mean_distinct_ratio'))}->{_f(at.get('mean_distinct_ratio'))}")
        p(f"    utility: ppl_clean_ratio={_f(u.get('ppl_clean_ratio'))} (~1=kept)  "
          f"ROUGE_clean x{_f(u.get('rouge_recall_clean_ratio'))}  ROUGE_tau x{_f(u.get('rouge_recall_tau_ratio'))}")
        p(f"    stealth: dist={_f(s.get('attacker_distance'),'.3f')}<=d_T={_f(s.get('d_T'),'.3f')} "
          f"cos[{s.get('cosine_metric','aggregate')}]>=dT={_f(s.get('delta_T'),'.3f')}  "
          f"JOINT={s.get('jointly_satisfied')}")

    if fl:
        dur = fl.get("durability", [])
        cfg = fl.get("config", {})
        pri = fl.get("pristine_reference", {})
        nb = (cfg.get("num_clients", 0) or 0) - (cfg.get("num_attackers", 0) or 0)
        p("-" * 74)
        p(f"[B] MULTI-ROUND FL  {_f(cfg.get('num_clients'))}={nb}+{_f(cfg.get('num_attackers'))}  "
          f"rounds={_f(cfg.get('num_rounds'))} per_round={_f(cfg.get('clients_per_round'))} "
          f"kd={_f(cfg.get('kd_clean_weight'))} gamma={_f(cfg.get('gamma'))} "
          f"cap={_f(cfg.get('max_new_tokens'))}")
        if pri:
            p(f"    pristine(r0) ref: ppl_clean={_f(pri.get('ppl_clean'))} "
              f"ROUGE_clean={_f(pri.get('rouge_recall_clean'))} ROUGE_tau={_f(pri.get('rouge_recall_tau'))} "
              f"tau_len={_f(pri.get('tau_mean_len'))}")
        if dur:
            f0, fN = dur[0], dur[-1]
            p(f"    amp_tau {_f(f0.get('amp_tau'))}(r{f0.get('round')}) -> {_f(fN.get('amp_tau'))}(r{fN.get('round')})  "
              f"de-censored {_f(f0.get('amp_tau_decensored'))}->{_f(fN.get('amp_tau_decensored'))}  "
              f"eff(useful) {_f(f0.get('amp_tau_effective'))}->{_f(fN.get('amp_tau_effective'))}  "
              f"vs_pristine {_f(fN.get('amp_tau_vs_pristine'))}  med {_f(fN.get('amp_tau_median'))}")
            p(f"    tau_len {_f(f0.get('tau_len_atk'))}->{_f(fN.get('tau_len_atk'))} "
              f"(de-censored {_f(fN.get('tau_len_atk_decensored'))}, effective {_f(fN.get('tau_effective_len_atk'))})  "
              f"trunc {_f(f0.get('truncation_tau'))}->{_f(fN.get('truncation_tau'))}  "
              f"rep {_f(f0.get('repetition_tau'))}->{_f(fN.get('repetition_tau'))}  "
              f"distinct {_f(f0.get('distinct_ratio_tau'))}->{_f(fN.get('distinct_ratio_tau'))}")
            if fN.get("ppl_ratio_vs_pristine") is not None:
                p(f"    utility(vs pristine): ppl {_f(f0.get('ppl_ratio_vs_pristine'))}->{_f(fN.get('ppl_ratio_vs_pristine'))}"
                  f"  clean ROUGE {_f(f0.get('rouge_recall_clean_atk'))}->{_f(fN.get('rouge_recall_clean_atk'))}"
                  f"  tau ROUGE {_f(f0.get('rouge_recall_tau_atk'))}->{_f(fN.get('rouge_recall_tau_atk'))}"
                  f"  (~pristine = answer kept)")
            elif fN.get("ppl_ratio") is not None:
                ratios = [q["ppl_ratio"] for q in dur if q.get("ppl_ratio") is not None]
                p(f"    ppl_ratio(atk/ben) {_f(f0.get('ppl_ratio'))}->{_f(fN.get('ppl_ratio'))} worst={_f(max(ratios))}")
            p("    round   amp amp_dec amp_pri  trunc   rep  ppl_pri R_cln R_tau stealth")
            for q in dur:
                p(f"    {_f(q.get('round')):>4}  {_f(q.get('amp_tau'),'>5.2f')} "
                  f"{_f(q.get('amp_tau_decensored'),'>6.2f')} {_f(q.get('amp_tau_vs_pristine'),'>6.2f')} "
                  f"{_f(q.get('truncation_tau'),'>5.2f')} {_f(q.get('repetition_tau'),'>5.2f')} "
                  f"{_f(q.get('ppl_ratio_vs_pristine'),'>6.3f')} {_f(q.get('rouge_recall_clean_atk'),'>5.2f')} "
                  f"{_f(q.get('rouge_recall_tau_atk'),'>5.2f')}  {q.get('stealth_ok')}")
        st = [x for x in fl.get("stealth_trace", []) if x.get("n_attackers")]
        ok = sum(1 for x in st if x.get("jointly_satisfied"))
        p(f"    stealth jointly satisfied {ok}/{len(st)} attacker-participating rounds")

    if pareto:
        rows = pareto["points"] if isinstance(pareto, dict) and "points" in pareto else list(pareto)
        p("-" * 74)
        p(f"[C] PARETO  ({len(rows)} points)  gamma x kappa")
        p("    gamma  kappa   amp    med   clean   sel   ppl_r  dist/d_T    JOINT")
        for r in rows:
            p(f"    {_f(r.get('gamma'),'>5')}  {_f(r.get('kappa'),'>5')}  "
              f"{_f(r.get('amp_tau'),'>5.2f')} {_f(r.get('amp_tau_median'),'>5.2f')} "
              f"{_f(r.get('amp_clean'),'>5.2f')} {_f(r.get('selectivity'),'>5.2f')} "
              f"{_f(r.get('ppl_clean_ratio'),'>5.3f')}  "
              f"{_f(r.get('distance'),'.2f')}/{_f(r.get('d_T'),'.2f')}   {r.get('jointly_satisfied')}")

    if not (phase0 or fl or pareto):
        p("(no results found — run Step 5 / Step 7 / Step 8 first)")

    # Auto next-step tuning suggestions (the actionable text to steer the next experiment).
    if phase0 or fl or pareto:
        p("-" * 74)
        p("TUNING HINTS (auto) — 下一轮实验的调参方向:")
        for h in tuning_hints(phase0, fl, pareto):
            p(f"  - {h}")
    p("=" * 74)
    return L


def feedback_digest(phase0: Optional[Dict] = None, fl: Optional[Dict] = None,
                    pareto=None) -> str:
    """One compact ASCII block with the MOST IMPORTANT numbers across the experiments,
    designed to be copied out of Colab and pasted back for review. Robust to any
    experiment being absent or metrics being None. Prints AND returns the text."""
    text = "\n".join(_digest_lines(phase0, fl, pareto))
    print(text)
    return text


def full_report(phase0: Optional[Dict] = None, fl: Optional[Dict] = None,
                pareto=None) -> str:
    """The MAXIMAL copy-paste-back report: everything useful for offline analysis and the
    next iteration — full configs, the compact digest (tables + auto tuning hints), the
    per-round stealth trace, a representative attacker trajectory (within-round L_mal / E[L]
    / q_eos / rep / ALM), and decoded qualitative samples (long AND coherent, or looped?).
    Prints AND returns the text. Designed so ONE cell's output carries the whole picture."""
    L: List[str] = []
    p = L.append
    p("#" * 78)
    p("# TCAA FULL REPORT — 复制本格【全部输出】回传即可分析/迭代（含配置·表·轨迹·样例·建议）")
    p("#" * 78)

    # ---- 1) configs (so the numbers are interpretable / reproducible) ----
    def dump_cfg(tag, cfg):
        if not cfg:
            return
        p(f"[CONFIG {tag}]")
        keys = sorted(cfg.keys())
        row = []
        for k in keys:
            row.append(f"{k}={cfg[k]}")
            if len(row) == 4:
                p("    " + "  ".join(row)); row = []
        if row:
            p("    " + "  ".join(row))
    if fl:
        dump_cfg("B (multi-round FL)", fl.get("config"))
        pr = fl.get("pristine_reference")
        if pr:
            p(f"    pristine(r0): ppl_clean={_f(pr.get('ppl_clean'))} ppl_tau={_f(pr.get('ppl_tau'))} "
              f"ROUGE_clean={_f(pr.get('rouge_recall_clean'))} ROUGE_tau={_f(pr.get('rouge_recall_tau'))} "
              f"tau_len={_f(pr.get('tau_mean_len'))} tau_eff_len={_f(pr.get('tau_effective_len'))}")
    if pareto:
        rows = pareto["points"] if isinstance(pareto, dict) and "points" in pareto else list(pareto)
        if rows and isinstance(rows[0], dict):
            p(f"[CONFIG C (pareto)] {len(rows)} points; "
              f"gammas={sorted({r.get('gamma') for r in rows})} "
              f"gamma_cleans={sorted({r.get('gamma_clean') for r in rows})} "
              f"kappas={sorted({r.get('kappa') for r in rows})}")

    # ---- 2) the compact digest (headline tables + tuning hints) ----
    p("")
    L.extend(_digest_lines(phase0, fl, pareto))

    # ---- 3) B: full per-round stealth trace (process over rounds) ----
    if fl and fl.get("stealth_trace"):
        st = [s for s in fl["stealth_trace"] if s.get("n_attackers")]
        if st:
            p("")
            p("[B · PER-ROUND STEALTH TRACE]  round | dist <= d_T | cos_metric cos | satisfied")
            for s in st:
                p(f"    r{_f(s.get('round')):>3} | {_f(s.get('attacker_distance'),'.3f')} <= {_f(s.get('d_T'),'.3f')} | "
                  f"{s.get('cosine_metric','?')} cos={_f(s.get('attacker_cosine'),'.3f')}/pair={_f(s.get('pairwise_cosine'),'.3f')} | "
                  f"{s.get('jointly_satisfied')}")

    # ---- 4) representative attacker trajectory (within-round optimization dynamics) ----
    trace = (fl or {}).get("sample_mal_trace") or (phase0 or {}).get("mal_trace")
    if trace:
        p("")
        p("[ATTACKER TRAJECTORY · 一段代表性单轮内优化] step | L_mal ce_tau ce_clean kd rep E[L]_tau q_eos | dist(g) cos(g)")
        for r in trace:
            base = (f"    s{_f(r.get('step')):>4} | L={_f(r.get('L_mal'),'.2f')} ce_t={_f(r.get('ce_tau'),'.2f')} "
                    f"ce_c={_f(r.get('ce_clean'),'.2f')} kd={_f(r.get('kd_clean'))} rep={_f(r.get('rep_term'))} "
                    f"E[L]={_f(r.get('E_len_tau'),'.1f')} q={_f(r.get('mean_eos_prob_tau'))}")
            if r.get("dist") is not None:
                base += f" | dist={_f(r.get('dist'),'.3f')}(g={_f(r.get('g_dist'),'+.3f')}) cos={_f(r.get('cos'),'.3f')}(g={_f(r.get('g_sim'),'+.3f')})"
            p(base)

    # ---- 5) decoded qualitative samples (coherent long vs degenerate loop?) ----
    ex = (fl or {}).get("final_examples") or (phase0 or {}).get("final_examples")
    if ex:
        p("")
        p("[DECODED SAMPLES · 最终被攻击全局的解码样例(判断'变长是否有用/连贯')]")
        for e in ex:
            p(f"    [{e.get('split')}] len={_f(e.get('len'))} trunc={e.get('truncated')} "
              f"rougeR={_f(e.get('rouge_recall'))} rep={_f(e.get('repetition'))} distinct={_f(e.get('distinct'))}")
            p(f"        prompt : {str(e.get('prompt',''))[:160]}")
            p(f"        output : {str(e.get('output',''))[:400]}")

    p("#" * 78)
    p("# 说明: rep 高/distinct 低/eff<<amp → 复读(变长无用); ppl_pri 偏离1或 ROUGE 掉 → 效用损;")
    p("#       amp_dec>>amp → 真实消耗被 cap 遮住; JOINT=False → 隐蔽未满足。见上方 TUNING HINTS。")
    p("#" * 78)
    text = "\n".join(L)
    print(text)
    return text
