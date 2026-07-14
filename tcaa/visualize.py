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
# Multi-round FL (render_fl_report):  fl_durability, fl_utility, fl_stealth,
#   fl_defense_geometry (per-client cos/dist/norm/Krum over rounds, benign vs attacker;
#   the AugMP-visualization "defense's-eye view" from fl_runner's defense_telemetry)
# Pareto sweep    (render_pareto_report): pareto_frontier, pareto_kappa
# summary_html(): a static, self-contained HTML summary card (persists in a notebook).

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# --- role-based per-client palettes -------------------------------------------
# Borrowed from coding_reference/AugMP-visualization.py's core convention (benign = cool
# colors, attacker = warm/red) so a reader separates the two roles by hue alone, but kept
# CVD-safe (cool hues from the Okabe-Ito family for benign; distinct reds for attackers).
# Used by the multi-round per-client "defense's-eye view" figure.
BENIGN_CYCLE = ["#0072B2", "#56B4E9", "#009E73", "#5B9BD5", "#117733",
                "#2E75B6", "#44AA99", "#88CCEE", "#4C9F70", "#0070C0"]
ATTACKER_CYCLE = ["#D55E00", "#C00000", "#E4572E", "#B22222", "#8B0000"]
BENIGN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">"]
ATTACKER_MARKERS = ["*", "X", "D", "^", "v"]

# Export policy shared by Phase-0, FL and Pareto runners.  PNG is convenient for
# notebooks and previews; the PDF sidecar keeps text and geometry vector-sharp for papers.
RASTER_DPI = 600
EXPORT_FORMATS = ("png", "pdf")
LEGEND_FONT_SIZE = 10
COMPACT_LEGEND_FONT_SIZE = 9


def apply_style():
    """Apply the repository-wide IEEE-inspired publication style.

    The explicit reset makes figures deterministic in notebooks where a previous cell may
    have selected seaborn/dark styles.  Font fallbacks and Type-42 embedding keep text crisp
    and editable in common paper-production tools.
    """
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (6.5, 5.0),
        "figure.dpi": 140,
        "savefig.dpi": RASTER_DPI,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif",
        ],
        "font.size": 12,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelsize": 13,
        "axes.labelpad": 6,
        "axes.labelcolor": INK,
        "axes.axisbelow": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.fancybox": False,
        "legend.edgecolor": "#333333",
        "legend.borderpad": 0.45,
        "grid.color": "#b8b8b8",
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.38,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.6,
        # Preserve real text in vector outputs instead of converting glyphs to paths.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, path, *, formats=EXPORT_FORMATS) -> List[Path]:
    """Save one figure consistently as a high-resolution PNG and vector PDF.

    ``path`` may include an extension; it is treated as the output stem so all requested
    formats share the same basename.  Returning every written path makes runner logging and
    downstream automation straightforward.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("") if path.suffix else path
    written: List[Path] = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        if fmt not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported figure format: {fmt}")
        out = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(out, format=fmt, **kwargs)
        written.append(out)
    return written


def _bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10, color=INK,
                    xytext=(0, 2), textcoords="offset points")


def _use_log_scale_if_needed(ax, values, *, dynamic_range=100.0) -> bool:
    """Use a clearly labelled log axis when linear scaling would flatten a series.

    This is especially important for de-censored estimates, which can be several orders of
    magnitude above their capped observations.  Non-positive/NaN values safely keep the
    ordinary linear axis.
    """
    import math

    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            finite.append(value)
    if len(finite) < 2 or max(finite) / min(finite) < dynamic_range:
        return False
    ax.set_yscale("log")
    ylabel = ax.get_ylabel()
    if "log scale" not in ylabel:
        ax.set_ylabel(f"{ylabel} (log scale)")
    return True


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
        ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
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
    ax.set_title(r"(1) Resource amplification  $C_{atk}/C_{ben}$ ($\tau$)", fontsize=10.5)
    ax.set_ylabel("Cost amplification ratio")
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
    ax.set_ylabel("Attacked / baseline ratio")
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
    ax.set_ylabel(r"Distance  $\|\Delta_{att}-\Delta_g\|_2$")
    ax.set_ylim(0, max(dist, d_T) * 1.3)
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
    verdict = "inside envelope" if s["distance_satisfied"] else "OUTSIDE envelope"
    ax.set_title(f"Distance  ({verdict})", fontsize=11)
    ax.set_ylabel(r"$\|\Delta_{att}-\Delta_g\|_2$")
    ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)
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
    vc = "inside envelope" if s["cosine_satisfied"] else "OUTSIDE envelope"
    ax.set_title(f"Cosine  ({vc})", fontsize=11)
    ax.set_ylabel(r"$\cos(\Delta_{att},\Delta_g)$")
    ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)
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
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=COMPACT_LEGEND_FONT_SIZE,
               bbox_to_anchor=(0.5, -0.01))
    joint = "JOINTLY SATISFIED" if s["jointly_satisfied"] else "NOT jointly satisfied; Phase 1"
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
        axes[1].legend(fontsize=COMPACT_LEGEND_FONT_SIZE, loc="best")
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
    ax.set_ylabel(r"$\|\Delta_{att}-\Delta_g\|_2$"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=COMPACT_LEGEND_FONT_SIZE, loc="best")
    ax.set_title("ALM stealth-constraint convergence (rests at the boundary)")

    ax = axes[1]
    ax.plot(steps, coss, color=C_BASE, lw=2, marker="s", ms=3, label="attacker pairwise cosine")
    ax.axhline(cos_low, color=MUTED, ls="--", lw=1.2, label="benign cosine floor")
    ymin = min(coss + [cos_low]); ax.axhspan(cos_low, 1.0, color=C_OK, alpha=0.07)
    ax.set_ylabel("cosine to benign"); ax.set_xlabel("optimization step")
    ax.set_ylim(min(ymin - 0.05, cos_low - 0.05), 1.02)
    ax.grid(axis="x", visible=False); ax.legend(fontsize=COMPACT_LEGEND_FONT_SIZE, loc="best")
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
    ax.grid(axis="x", visible=False); ax.legend(fontsize=COMPACT_LEGEND_FONT_SIZE, loc="upper left")
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
    """Save all Phase-0 figures as 600-DPI PNGs with vector PDF sidecars.

    PNG paths remain the return value for backward compatibility; each PNG has a same-name
    PDF alongside it.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fig in make_all_figures(results):
        p = out_dir / f"{key}.png"
        save_figure(fig, p)
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
    # A cap-saturated EOS-hazard fit can be entirely assumption-driven.  The runner marks
    # those points ``decensored_valid=False``; never turn them into a measured-looking line.
    has_dec = any(p.get("decensored_valid", True)
                  and p.get("amp_tau_decensored") is not None for p in dur)
    has_len_dec = any(p.get("decensored_valid", True)
                      and p.get("tau_len_atk_decensored") is not None for p in dur)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)
    # --- (A) cost amplification: capped (solid) + de-censored (dashed) + median ---
    ax = axes[0]
    ax.plot(rounds, [p["amp_tau"] for p in dur], "-o", color=C_BASE, lw=2, ms=4,
            label="amp_tau mean (capped)")
    if has_dec:
        ax.plot(rounds, [p.get("amp_tau_decensored", float("nan"))
                         if p.get("decensored_valid", True) else float("nan") for p in dur],
                "--D", color=C_ATK,
                lw=2, ms=4, label="amp_tau mean (de-censored)")
    ax.plot(rounds, [p["amp_tau_median"] for p in dur], ":s", color=C_BENIGN, lw=1.8, ms=4,
            label="amp_tau median")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylabel("cost amplification (τ)")
    _use_log_scale_if_needed(
        ax,
        [p["amp_tau"] for p in dur]
        + [p.get("amp_tau_decensored") if p.get("decensored_valid", True) else None
           for p in dur]
        + [p["amp_tau_median"] for p in dur],
    )
    ax.set_title("TCAA multi-round durability (amplification accumulation)")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
    # --- (B) tau output length: capped + de-censored + clean ---
    ax = axes[1]
    ax.plot(rounds, [p["tau_len_atk"] for p in dur], "-^", color=C_ATK, lw=2, ms=4,
            label="τ len (capped at max_new_tokens)")
    if has_len_dec:
        ax.plot(rounds, [p.get("tau_len_atk_decensored", float("nan"))
                         if p.get("decensored_valid", True) else float("nan") for p in dur],
                "--D", color=C_PURPLE,
                lw=1.8, ms=3, label="τ len (de-censored estimate)")
    ax.plot(rounds, [p.get("clean_len_atk", float("nan")) for p in dur], "-o", color=C_BASE, lw=1.6,
            ms=3, label="clean len")
    ax.set_ylabel("mean output length  L")
    _use_log_scale_if_needed(
        ax,
        [p["tau_len_atk"] for p in dur]
        + [p.get("tau_len_atk_decensored") if p.get("decensored_valid", True) else None
           for p in dur]
        + [p.get("clean_len_atk") for p in dur],
    )
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
    # --- (C) truncation rate: how censored the capped amplification above is ---
    ax = axes[2]
    ax.plot(rounds, [p.get("truncation_tau", float("nan")) for p in dur], "-^", color=C_ATK,
            lw=2, ms=4, label="τ truncation (cap-hit → capped amp is a lower bound)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("truncation rate  [0,1]"); ax.set_xlabel("communication round")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
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
    ax.set_xlabel("communication round")
    ax.set_ylabel(r"Attacker distance  $\|\Delta_{att}-\Delta_g\|_2$")
    ax.set_title(f"Per-round stealth ({n_ok}/{len(st)} attacker rounds jointly satisfied)")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
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
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
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
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE, ncol=2); ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def _telemetry_series(telemetry: List[Dict], key: str):
    """From defense_telemetry rounds -> ({benign_id: (rounds, vals)}, {attacker_id: (...)}).
    Clients are SAMPLED per round, so each client's series covers only the rounds it appears
    in (plotted with markers at those rounds — honest about participation gaps)."""
    benign: Dict[int, Tuple[List, List]] = {}
    attacker: Dict[int, Tuple[List, List]] = {}
    for entry in telemetry:
        rnd = entry.get("round")
        for c in entry.get("clients", []):
            v = c.get(key)
            if v is None:
                continue
            bucket = attacker if c.get("label") == "attacker" else benign
            cid = c.get("client_id")
            rs, vs = bucket.setdefault(cid, ([], []))
            rs.append(rnd); vs.append(v)
    return benign, attacker


def _plot_role_panel(ax, benign: Dict, attacker: Dict, ylabel: str, title: str):
    """Per-client lines over rounds: benign = thin cool 'cloud', attacker = bold red on top.
    The separability at a glance IS the message — overlap = stealthy on this axis; a
    detached red line = a detector using this metric would flag the attacker."""
    for i, cid in enumerate(sorted(benign)):
        rs, vs = benign[cid]
        col = BENIGN_CYCLE[i % len(BENIGN_CYCLE)]
        ax.plot(rs, vs, "-", color=col, lw=1.2, alpha=0.75,
                marker=BENIGN_MARKERS[i % len(BENIGN_MARKERS)], ms=3.5,
                markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.4, zorder=2)
    for i, cid in enumerate(sorted(attacker)):
        rs, vs = attacker[cid]
        col = ATTACKER_CYCLE[i % len(ATTACKER_CYCLE)]
        ax.plot(rs, vs, "-", color=col, lw=2.3,
                marker=ATTACKER_MARKERS[i % len(ATTACKER_MARKERS)], ms=6.5,
                markerfacecolor=col, markeredgecolor=INK, markeredgewidth=0.5, zorder=4)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("communication round"); ax.grid(axis="x", visible=False)


def fig_fl_defense_geometry(r: Dict):
    """The offline defense's-eye view: per-client update geometry over rounds, colored benign
    (cool) vs attacker (red) in the AugMP-visualization convention. Panels = the metrics a
    robust aggregator screens on: cosine-to-aggregate, distance-to-aggregate, L2 norm, Krum
    score. Where the benign cloud and the attacker line OVERLAP, the attack is stealthy on
    that axis; where the red line DETACHES, a detector using that metric would catch it.
    Consumes fl_runner's per-round defense_telemetry (aggregation itself stays FedAvg)."""
    tel = r.get("defense_telemetry", [])
    if not tel:
        return None
    panels = [("cos_to_agg", "cosine to aggregate", "(a) Cosine similarity to FedAvg aggregate"),
              ("dist_to_agg", "distance to aggregate", "(b) Euclidean distance to aggregate"),
              ("norm", r"Update norm  $\|\Delta\|_2$", "(c) Update L2 norm"),
              ("krum_score", "Krum score (lower = selected)", "(d) Multi-Krum score")]
    have = [(k, yl, t) for (k, yl, t) in panels
            if any(c.get(k) is not None for e in tel for c in e.get("clients", []))]
    if not have:
        return None
    nrows = (len(have) + 1) // 2
    fig, axarr = plt.subplots(nrows, 2, figsize=(11.0, 4.2 * nrows))
    axes = list(axarr.ravel()) if hasattr(axarr, "ravel") else [axarr]
    for ax, (k, yl, t) in zip(axes, have):
        benign, attacker = _telemetry_series(tel, k)
        _plot_role_panel(ax, benign, attacker, yl, t)
    for ax in axes[len(have):]:
        ax.set_visible(False)

    # Shared legend: one proxy for the benign cloud + one entry per attacker client.
    from matplotlib.lines import Line2D
    _, atk0 = _telemetry_series(tel, have[0][0])
    handles = [Line2D([0], [0], color=BENIGN_CYCLE[0], lw=1.4, marker="o", ms=5,
                      markeredgecolor="white", label="Benign clients")]
    for i, cid in enumerate(sorted(atk0)):
        handles.append(Line2D([0], [0], color=ATTACKER_CYCLE[i % len(ATTACKER_CYCLE)], lw=2.3,
                              marker=ATTACKER_MARKERS[i % len(ATTACKER_MARKERS)], ms=8,
                              markeredgecolor=INK, label=f"Attacker (client {cid})"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=COMPACT_LEGEND_FONT_SIZE,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Per-client update geometry (defense's-eye view): benign cloud vs attacker",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96]); return fig


# --------------------------------------------------------------------------- #
# Resource-accounting figures (logical tokens + measured accelerator metrics) #
# --------------------------------------------------------------------------- #
_RESOURCE_CONTAINER_ALIASES = (
    "resources",                 # canonical resource-v1 container
    "resource_summary",         # early/legacy experiment exports
    "resource_metrics",
    "resource_benchmark",
    "resource_profile",
)

_LOGICAL_METRIC_KEYS = {
    "num_requests": ("num_requests", "n_prompts", "request_count", "requests", "n", "eval_size"),
    "total_input_tokens": ("total_input_tokens", "input_tokens", "prompt_tokens"),
    "total_output_tokens": ("total_output_tokens", "output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens", "logical_tokens"),
    "mean_output_len": ("mean_output_len", "output_len_mean", "mean_completion_tokens"),
    "output_len_p50": ("output_len_p50", "p50_output_len", "output_tokens_p50"),
    "output_len_p95": ("output_len_p95", "p95_output_len", "output_tokens_p95"),
    "cap_hit_rate": ("cap_hit_rate", "truncation_rate"),
    "budget_utilization": ("budget_utilization", "token_budget_utilization"),
    "prefill_attention_token_pairs": ("prefill_attention_token_pairs", "prefill_attention_pairs"),
    "decode_attention_token_pairs": ("decode_attention_token_pairs", "decode_attention_pairs"),
    "scheduled_decode_slots": ("scheduled_decode_slots", "batch_scheduled_decode_slots"),
    "kv_cache_peak_bytes": ("kv_cache_peak_bytes", "estimated_peak_kv_bytes",
                            "kv_cache_batch_peak_bytes", "estimated_peak_batch_kv_bytes"),
}

_HARDWARE_METRIC_KEYS = {
    "generation_wall_seconds": ("generation_wall_seconds", "wall_seconds", "wall_time_seconds",
                                "wall_time_s"),
    "cuda_elapsed_seconds": ("cuda_elapsed_seconds", "cuda_seconds", "cuda_time_seconds",
                             "cuda_time_s"),
    "e2e_wall_seconds": ("e2e_wall_seconds", "end_to_end_wall_seconds", "e2e_seconds"),
    "requests_per_second": ("requests_per_second", "requests_per_sec", "req_per_s", "req_s"),
    "output_tokens_per_second": ("output_tokens_per_second", "output_tokens_per_sec",
                                 "output_tok_per_s", "tokens_per_second"),
    "peak_allocated_bytes": ("peak_allocated_bytes", "peak_memory_allocated_bytes"),
    "incremental_peak_allocated_bytes": ("incremental_peak_allocated_bytes",
                                         "peak_allocated_delta_bytes"),
    "peak_reserved_bytes": ("peak_reserved_bytes", "peak_memory_reserved_bytes"),
    "incremental_peak_reserved_bytes": ("incremental_peak_reserved_bytes",
                                        "peak_reserved_delta_bytes"),
    "energy_joules": ("energy_joules", "energy_j", "gpu_energy_joules"),
}


def _resource_root(fl: Optional[Dict]) -> Optional[Dict]:
    """Return the resource-v1 payload while accepting transitional result names."""
    if not isinstance(fl, dict):
        return None
    for key in _RESOURCE_CONTAINER_ALIASES:
        value = fl.get(key)
        if isinstance(value, dict):
            return value
    # Also accept a resource payload passed directly to the public helpers.
    if isinstance(fl.get("states"), (dict, list)) and (
            fl.get("schema_version") or fl.get("environment") or fl.get("comparisons")):
        return fl
    return None


def _resource_state_records(fl: Optional[Dict]) -> List[Dict]:
    """Normalize resource-v1 state dictionaries/lists without mutating the result."""
    root = _resource_root(fl)
    if not root:
        return []
    raw_states = (root.get("states") or root.get("conditions") or root.get("profiles")
                  or root.get("measurements"))
    if isinstance(raw_states, dict):
        items = list(raw_states.items())
    elif isinstance(raw_states, list):
        items = []
        for i, state in enumerate(raw_states):
            if not isinstance(state, dict):
                continue
            name = (state.get("condition") or state.get("state") or state.get("name")
                    or f"state_{i}")
            items.append((str(name), state))
    else:
        # A few early exports placed named conditions directly under the resource root.
        ignored = {"schema_version", "environment", "config", "comparisons", "validity"}
        items = [(k, v) for k, v in root.items()
                 if k not in ignored and isinstance(v, dict)]

    records = []
    for name, state in items:
        if not isinstance(state, dict):
            continue
        logical = state.get("logical") or state.get("tokens") or state.get("token_summary") or {}
        hardware = state.get("hardware") or state.get("profile") or state.get("hardware_summary") or {}
        records.append({
            "name": str(name),
            "logical": logical if isinstance(logical, dict) else {},
            "hardware": hardware if isinstance(hardware, dict) else {},
            "raw": state,
        })
    return sorted(records, key=lambda x: (_resource_role_order(x["name"]), x["name"]))


def _resource_role_order(name: str) -> int:
    name = str(name).lower()
    if "pristine" in name or "baseline" in name:
        return 0
    if "benign" in name or name.startswith("ben"):
        return 1
    if "attack" in name or name.startswith("atk"):
        return 2
    return 3


def _resource_state_label(name: str) -> str:
    lower = str(name).lower()
    if "pristine" in lower or "baseline" in lower:
        return "pristine"
    if "benign" in lower or lower.startswith("ben"):
        return "benign"
    if "attack" in lower or lower.startswith("atk"):
        return "attacked"
    return str(name).replace("_", " ")


def _finite_number(value):
    import math
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _first_metric(mappings, aliases):
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in aliases:
            value = _finite_number(mapping.get(key))
            if value is not None:
                return value
    return None


def _resource_hardware_profiles(state: Dict) -> List[Tuple[str, Dict]]:
    """Normalize direct and ``hardware[batch_size_*]`` profiler payloads.

    The FL runner can retain more than one batch-size benchmark.  Keeping them separate is
    essential: BS=1 latency and BS=8 throughput are not interchangeable measurements.
    """
    if not isinstance(state, dict):
        return []
    hardware = state.get("hardware")
    if not isinstance(hardware, dict) or not hardware:
        return []
    direct_aliases = tuple(alias for aliases in _HARDWARE_METRIC_KEYS.values()
                           for alias in aliases)
    if isinstance(hardware.get("summary"), dict) or any(key in hardware for key in direct_aliases):
        return [("", hardware)]

    profiles: List[Tuple[str, Dict]] = []
    for name, profile in hardware.items():
        if not isinstance(profile, dict):
            continue
        summary = profile.get("summary")
        if isinstance(summary, dict) or any(key in profile for key in direct_aliases):
            profiles.append((str(name), profile))

    def order(item):
        import re
        match = re.search(r"(\d+)", item[0])
        return (int(match.group(1)) if match else 10 ** 9, item[0])

    return sorted(profiles, key=order)


def _resource_hardware_profiles_for_report(state: Dict) -> List[Tuple[str, Dict]]:
    """Return every measured split/batch profile for the archival text report."""
    raw = state.get("raw") if isinstance(state, dict) else None
    split_profiles = raw.get("hardware_profiles") if isinstance(raw, dict) else None
    if isinstance(split_profiles, dict) and split_profiles:
        rows: List[Tuple[str, Dict]] = []
        for split, profiles in split_profiles.items():
            if not isinstance(profiles, dict):
                continue
            for batch_name, summary in profiles.items():
                if isinstance(summary, dict):
                    rows.append((f"{split}/{batch_name}", {"summary": summary}))
        if rows:
            return rows
    return _resource_hardware_profiles(state)


def _primary_resource_hardware(state: Dict) -> Dict:
    """Choose the largest retained batch size for aggregate comparisons/plots."""
    profiles = _resource_hardware_profiles(state)
    return profiles[-1][1] if profiles else {}


def _resource_hardware_fully_valid(state: Optional[Dict]) -> bool:
    if not state:
        return False
    profile = _primary_resource_hardware(state)
    summary = profile.get("summary") if isinstance(profile, dict) else None
    # Legacy direct profiles have no aggregate validity flag; keep them readable.
    return not isinstance(summary, dict) or summary.get("valid") is not False


def _resource_metric(state: Dict, key: str):
    """Read one canonical metric from nested resource-v1 or a flattened legacy state."""
    if not isinstance(state, dict):
        return None
    logical = state.get("logical")
    hardware = state.get("hardware")
    if key in _HARDWARE_METRIC_KEYS:
        hardware = _primary_resource_hardware(state)
    hardware_summary = hardware.get("summary") if isinstance(hardware, dict) else None
    if key in _LOGICAL_METRIC_KEYS:
        value = _first_metric((logical, hardware_summary, state.get("raw")),
                              _LOGICAL_METRIC_KEYS[key])
    elif key in _HARDWARE_METRIC_KEYS:
        value = _first_metric((hardware_summary, hardware, state.get("raw")),
                              _HARDWARE_METRIC_KEYS[key])
    else:
        value = None
    if value is not None:
        return value

    # Safe algebraic fallbacks make partially upgraded JSONs useful without fabricating data.
    if key == "total_tokens":
        prompt = _resource_metric(state, "total_input_tokens")
        output = _resource_metric(state, "total_output_tokens")
        return prompt + output if prompt is not None and output is not None else None
    if key == "total_output_tokens":
        total = _first_metric((state.get("logical"), state.get("raw")),
                              _LOGICAL_METRIC_KEYS["total_tokens"])
        prompt = _resource_metric(state, "total_input_tokens")
        return total - prompt if total is not None and prompt is not None else None
    if key == "mean_output_len":
        total = _resource_metric(state, "total_output_tokens")
        n = _resource_metric(state, "num_requests")
        return total / n if total is not None and n and n > 0 else None
    return None


_RESOURCE_COMPARISON_LABELS = {
    "total_output_tokens": "output tokens",
    "total_tokens": "all logical tokens",
    "decode_attention_token_pairs": "decode attention proxy",
    "prefill_attention_token_pairs": "prefill attention proxy",
    "scheduled_decode_slots": "scheduled decode slots",
    "mean_kv_proxy": "mean KV token proxy",
    "kv_cache_peak_bytes": "estimated peak KV bytes",
    "generation_wall_seconds": "generation wall time",
    "cuda_elapsed_seconds": "CUDA elapsed time",
    "e2e_wall_seconds": "end-to-end profiling time",
    "allocated_gpu_seconds": "allocated GPU time",
    "energy_joules": "GPU energy",
    "incremental_peak_allocated_bytes": "incremental peak memory",
    "peak_allocated_bytes": "peak allocated memory",
    "requests_per_second": "request throughput",
    "output_tokens_per_second": "output-token throughput",
}

_RESOURCE_COMPARISON_ALIASES = {
    "decode_attention_pairs": "decode_attention_token_pairs",
    "prefill_attention_pairs": "prefill_attention_token_pairs",
    "wall_seconds": "generation_wall_seconds",
    "cuda_seconds": "cuda_elapsed_seconds",
    "energy_j": "energy_joules",
    "peak_allocated_delta_bytes": "incremental_peak_allocated_bytes",
    "estimated_peak_batch_kv_bytes": "kv_cache_peak_bytes",
}


def _canonical_resource_metric_name(metric) -> str:
    metric = str(metric)
    return _RESOURCE_COMPARISON_ALIASES.get(metric, metric)


def _state_for_role(states: List[Dict], role: str) -> Optional[Dict]:
    for state in states:
        if _resource_state_label(state["name"]) == role:
            return state
    return None


def _resource_comparison_rows(fl: Optional[Dict]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Return ``(metric, attacked/pristine, attacked/benign)`` comparison rows."""
    root = _resource_root(fl) or {}
    comparisons = root.get("comparisons") or {}
    rows: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if isinstance(comparisons, dict):
        # Canonical orientation: comparisons[metric][attacked_vs_*].
        for metric, values in comparisons.items():
            if isinstance(values, dict):
                ap = _first_metric((values,), ("attacked_vs_pristine", "atk_vs_pristine"))
                ab = _first_metric((values,), ("attacked_vs_benign", "atk_vs_benign"))
                if ap is not None or ab is not None:
                    rows[_canonical_resource_metric_name(metric)] = (ap, ab)
        # Also accept comparisons[attacked_vs_*][metric].
        for ratio_key, idx in (("attacked_vs_pristine", 0), ("attacked_vs_benign", 1),
                               ("atk_vs_pristine", 0), ("atk_vs_benign", 1)):
            metric_values = comparisons.get(ratio_key)
            if not isinstance(metric_values, dict):
                continue
            for metric, value in metric_values.items():
                value = _finite_number(value)
                if value is None:
                    continue
                metric = _canonical_resource_metric_name(metric)
                old = rows.get(metric, (None, None))
                rows[metric] = ((value, old[1]) if idx == 0 else (old[0], value))

    # Fill absent comparisons from like-for-like aggregates.  Ratios remain N/A when a
    # reference is zero/missing; no epsilon is used because that would invent amplification.
    states = _resource_state_records(fl)
    attacked = _state_for_role(states, "attacked")
    pristine = _state_for_role(states, "pristine")
    benign = _state_for_role(states, "benign")
    for metric in _RESOURCE_COMPARISON_LABELS:
        old_ap, old_ab = rows.get(metric, (None, None))
        hardware_metric = metric in _HARDWARE_METRIC_KEYS
        numerator = (
            _resource_metric(attacked, metric)
            if attacked and (not hardware_metric or _resource_hardware_fully_valid(attacked))
            else None
        )
        pri_value = (
            _resource_metric(pristine, metric)
            if pristine and (not hardware_metric or _resource_hardware_fully_valid(pristine))
            else None
        )
        ben_value = (
            _resource_metric(benign, metric)
            if benign and (not hardware_metric or _resource_hardware_fully_valid(benign))
            else None
        )
        ap = old_ap if old_ap is not None else (
            numerator / pri_value if numerator is not None and pri_value not in (None, 0) else None)
        ab = old_ab if old_ab is not None else (
            numerator / ben_value if numerator is not None and ben_value not in (None, 0) else None)
        if ap is not None or ab is not None:
            rows[metric] = (ap, ab)

    preferred = list(_RESOURCE_COMPARISON_LABELS)
    ordered = [metric for metric in preferred if metric in rows]
    ordered.extend(sorted(metric for metric in rows if metric not in preferred))
    return [(metric, rows[metric][0], rows[metric][1]) for metric in ordered]


def fig_resource_tokens(fl: Dict):
    """Logical token totals and output-length quantiles for each measured model state."""
    import numpy as np

    rows = []
    for state in _resource_state_records(fl):
        inp = _resource_metric(state, "total_input_tokens")
        out = _resource_metric(state, "total_output_tokens")
        p50 = _resource_metric(state, "output_len_p50")
        p95 = _resource_metric(state, "output_len_p95")
        if any(value is not None for value in (inp, out, p50, p95)):
            rows.append((state, inp, out, p50, p95))
    if not rows:
        return None

    have_totals = any(inp is not None or out is not None for _, inp, out, _, _ in rows)
    have_quantiles = any(p50 is not None or p95 is not None for _, _, _, p50, p95 in rows)
    ncols = int(have_totals) + int(have_quantiles)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.4))
    axes = list(np.atleast_1d(axes))
    labels = [_resource_state_label(state["name"]) for state, *_ in rows]
    x = np.arange(len(rows))
    panel = 0
    if have_totals:
        ax = axes[panel]; panel += 1
        inputs = [inp if inp is not None else 0.0 for _, inp, _, _, _ in rows]
        outputs = [out if out is not None else 0.0 for _, _, out, _, _ in rows]
        ax.bar(x, inputs, color=C_BASE, label="input tokens")
        ax.bar(x, outputs, bottom=inputs, color=C_ATK, label="output tokens")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("logical tokens (sum)")
        ax.set_title("Logical token consumption")
        ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE)
        ax.grid(axis="x", visible=False)
    if have_quantiles:
        ax = axes[panel]
        width = 0.36
        p50s = [p50 if p50 is not None else float("nan") for _, _, _, p50, _ in rows]
        p95s = [p95 if p95 is not None else float("nan") for _, _, _, _, p95 in rows]
        ax.bar(x - width / 2, p50s, width, color=C_BENIGN, label="p50")
        ax.bar(x + width / 2, p95s, width, color=C_ATK, label="p95")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("output length (tokens/request)")
        ax.set_title("Output-length distribution summary")
        ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Resource accounting: tokens are measured, not inferred", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_resource_amplification(fl: Dict):
    """Separate exact, proxy, measured, and efficiency ratios to avoid mixed semantics."""
    import numpy as np

    rows = [(metric, ap, ab) for metric, ap, ab in _resource_comparison_rows(fl)
            if ap is not None or ab is not None]
    if not rows:
        return None
    exact = {"total_output_tokens", "total_tokens"}
    proxies = {
        "decode_attention_token_pairs", "prefill_attention_token_pairs",
        "scheduled_decode_slots", "mean_kv_proxy", "kv_cache_peak_bytes",
    }
    efficiency = {"requests_per_second", "output_tokens_per_second"}

    grouped = []
    for title, predicate in (
        ("Exact logical consumption", lambda metric: metric in exact),
        ("Analytic workload proxies\n(not hardware measurements)",
         lambda metric: metric in proxies),
        ("Measured accelerator use", lambda metric: metric not in exact | proxies | efficiency),
        ("Efficiency ratios\n(higher ≠ greater consumption)",
         lambda metric: metric in efficiency),
    ):
        selected = [row for row in rows if predicate(row[0])]
        if selected:
            grouped.append((title, selected))
    if not grouped:
        return None

    ncols = min(2, len(grouped))
    nrows = (len(grouped) + ncols - 1) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.4 * nrows))
    axes = list(np.atleast_1d(axarr).ravel())
    width = 0.36
    for ax, (title, selected) in zip(axes, grouped):
        labels = [_RESOURCE_COMPARISON_LABELS.get(metric, metric.replace("_", " "))
                  for metric, _, _ in selected]
        ap = [value if value is not None else float("nan") for _, value, _ in selected]
        ab = [value if value is not None else float("nan") for _, _, value in selected]
        x = np.arange(len(selected))
        ax.bar(x - width / 2, ap, width, color=C_BASE, label="attacked / pristine")
        ax.bar(x + width / 2, ab, width, color=C_ATK, label="attacked / benign")
        ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel("ratio")
        ax.set_title(title)
        _use_log_scale_if_needed(ax, ap + ab)
        ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE)
        ax.grid(axis="x", visible=False)
    for ax in axes[len(grouped):]:
        ax.set_visible(False)
    fig.suptitle("Attacked/reference resource ratios (N/A omitted)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); return fig


def make_resource_figures(fl_results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    """Build only resource figures; return an empty list when metrics were not collected."""
    apply_style()
    out = []
    for key, fn in (("resource_tokens", fig_resource_tokens),
                    ("resource_amplification", fig_resource_amplification)):
        try:
            fig = fn(fl_results)
        except Exception as exc:  # pragma: no cover - notebook should survive partial JSON
            print(f"  [visualize] {key} failed: {exc}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def save_resource_figures(fl_results: Dict, out_dir) -> List[Path]:
    """Persist resource figures as PNG+PDF; no resource data is a successful no-op."""
    out_dir = Path(out_dir)
    paths: List[Path] = []
    for key, fig in make_resource_figures(fl_results):
        path = out_dir / f"{key}.png"
        save_figure(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def make_fl_figures(fl_results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    for key, fn in (("fl_durability", fig_fl_durability), ("fl_utility", fig_fl_utility),
                    ("fl_stealth", fig_fl_stealth),
                    ("fl_defense_geometry", fig_fl_defense_geometry),
                    ("resource_tokens", fig_resource_tokens),
                    ("resource_amplification", fig_resource_amplification)):
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
              "fl_stealth": "逐轮隐蔽性 (客户端采样下)",
              "fl_defense_geometry": "逐客户端更新几何 (防御视角 · benign 云 vs attacker · cos/距离/范数/Krum)",
              "resource_tokens": "资源账本：输入/输出 token 与长度分位数",
              "resource_amplification": "资源放大：token、时间、显存与能耗"}
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
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
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
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE, title="attack strength")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); return fig


def fig_pareto_utility(pareto):
    """The 'usable' view of the sweep: a point is only a real TCAA win if amplification is
    high AND clean utility is preserved (ppl_clean_ratio ≈ 1) AND it stays stealthy. Plots
    amp (median) vs clean ppl ratio; color = joint stealth; marker size is proportional to trigger
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
    ax.set_title("Usable frontier: amplification vs preserved utility (size = selectivity)")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE); ax.grid(axis="x", visible=False)
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


def _resource_fmt(value, spec=".3f") -> str:
    """Format measured resources with one unambiguous missing-data sentinel."""
    value = _finite_number(value)
    if value is None:
        return "N/A"
    try:
        return format(value, spec)
    except (TypeError, ValueError):  # pragma: no cover - defensive custom format
        return str(value)


def _resource_int(value) -> str:
    value = _finite_number(value)
    return "N/A" if value is None else f"{int(round(value)):,}"


def _resource_pct(value) -> str:
    value = _finite_number(value)
    if value is None:
        return "N/A"
    # Canonical rates are fractions.  Accept already-percent legacy exports as-is.
    return f"{(100.0 * value if abs(value) <= 1.0 else value):.1f}%"


def _resource_gib(value) -> str:
    value = _finite_number(value)
    return "N/A" if value is None else f"{value / (1024 ** 3):.3f}"


def _resource_ratio(value) -> str:
    value = _finite_number(value)
    return "N/A" if value is None else f"{value:.3f}x"


def _profile_energy_method(profile: Dict, raw_state: Optional[Dict] = None) -> str:
    summary = profile.get("summary") if isinstance(profile, dict) else None
    method = _first_text((summary, profile, raw_state),
                         ("energy_method", "energy_measurement_method"))
    if method:
        return method
    records = profile.get("records") if isinstance(profile, dict) else None
    if isinstance(records, list):
        methods = sorted({str(record.get("energy_method")) for record in records
                          if isinstance(record, dict) and record.get("energy_method")})
        if methods:
            return "+".join(methods)
    return "N/A"


def _profile_validity_note(profile: Dict) -> str:
    summary = profile.get("summary") if isinstance(profile, dict) else None
    if not isinstance(summary, dict):
        return ""
    parts = []
    if summary.get("valid") is False:
        valid_repeats = _finite_number(summary.get("valid_repeats"))
        n_repeats = _finite_number(summary.get("n_repeats"))
        if valid_repeats is not None and n_repeats is not None:
            parts.append(f"valid={int(valid_repeats)}/{int(n_repeats)}")
        else:
            parts.append("invalid/partial")
    for key, label in (
        ("wall_timing_coverage", "wall"),
        ("token_count_coverage", "tokens"),
        ("cuda_coverage", "cuda"),
        ("memory_coverage", "memory"),
        ("energy_coverage", "energy"),
    ):
        value = _finite_number(summary.get(key))
        if value is not None:
            parts.append(f"{label}={100.0 * value:.0f}%")
    timed_out = _finite_number(summary.get("timed_out_batches"))
    if timed_out:
        parts.append(f"timeouts={int(timed_out)}")
    if summary.get("stopped_early"):
        parts.append(f"stopped={summary.get('stop_reason') or 'early'}")
    mapping = summary.get("nvml_device_mapping_methods")
    if isinstance(mapping, list) and mapping:
        parts.append("nvml_map=" + "+".join(str(value) for value in mapping))
    reasons = summary.get("instrumentation_reasons")
    if isinstance(reasons, list) and reasons:
        parts.append(f"instrumentation_reasons={len(reasons)}")
    return ",".join(parts)


def _profile_iqr_note(profile: Dict) -> str:
    summary = profile.get("summary") if isinstance(profile, dict) else None
    if not isinstance(summary, dict):
        return ""
    parts = []
    for key, label in (
        ("generation_wall_seconds", "wall"),
        ("cuda_elapsed_seconds", "CUDA"),
        ("e2e_wall_seconds", "e2e"),
        ("energy_joules", "energy_J"),
    ):
        q25 = _finite_number(summary.get(f"{key}_p25"))
        q75 = _finite_number(summary.get(f"{key}_p75"))
        if q25 is not None and q75 is not None:
            parts.append(f"{label}={q25:.3f}–{q75:.3f}")
    return "IQR[" + ", ".join(parts) + "]" if parts else ""


def _first_text(mappings, aliases):
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in aliases:
            value = mapping.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _resource_digest_lines(fl: Optional[Dict]) -> List[str]:
    """Build a copy-safe resource table for Colab logs and archived full reports."""
    L: List[str] = []
    p = L.append
    root = _resource_root(fl)
    p("-" * 74)
    if not root:
        p("[RESOURCE METRICS] N/A — this result predates resource-v1 or collection was disabled")
        return L

    schema = root.get("schema_version", "legacy/unspecified")
    p(f"[RESOURCE METRICS · {schema}] logical tokens + measured accelerator use")
    env = root.get("environment") if isinstance(root.get("environment"), dict) else {}
    cfg = root.get("config") if isinstance(root.get("config"), dict) else {}
    gpu_rows = env.get("gpus") if isinstance(env.get("gpus"), list) else []
    gpu0 = gpu_rows[0] if gpu_rows and isinstance(gpu_rows[0], dict) else {}
    torch_env = env.get("torch") if isinstance(env.get("torch"), dict) else {}
    nvidia_env = env.get("nvidia_smi") if isinstance(env.get("nvidia_smi"), dict) else {}
    gpu = _first_text((env, gpu0),
                      ("gpu_name_actual", "gpu_name", "device_name", "name", "gpu")) or "N/A"
    driver = _first_text((env, torch_env, nvidia_env),
                         ("driver_version", "nvidia_driver_version", "cuda_driver_version")) or "N/A"
    cuda = _first_text((env, torch_env),
                       ("cuda_runtime", "cuda_runtime_version", "cuda_version", "torch_cuda")) or "N/A"
    dtype = (_first_text((cfg, env), ("dtype", "model_dtype", "torch_dtype")) or "N/A")
    batch = (_first_text((cfg,), ("batch_size", "resource_profile_batch_size",
                                 "generation_batch_size")) or "N/A")
    vram = _first_metric((env, gpu0),
                         ("total_vram_bytes", "total_memory_bytes", "gpu_memory_bytes"))
    vram_text = f"{vram / (1024 ** 3):.1f} GiB" if vram is not None else "N/A"
    fingerprint = _first_text((env,), ("fingerprint_sha256", "environment_fingerprint"))
    fingerprint_text = fingerprint[:12] if fingerprint else "N/A"
    p(f"    env: gpu={gpu}  vram={vram_text}  driver={driver}  cuda={cuda}  "
      f"dtype={dtype}  batch={batch}  fingerprint={fingerprint_text}")
    compiled_arches = torch_env.get("compiled_arch_list")
    arches_text = ",".join(str(value) for value in compiled_arches) \
        if isinstance(compiled_arches, list) and compiled_arches else "N/A"
    preflight = torch_env.get("kernel_preflight") \
        if isinstance(torch_env.get("kernel_preflight"), dict) else {}
    preflight_success = preflight.get("success")
    preflight_text = "N/A" if preflight_success is None else str(bool(preflight_success))
    p(f"    CUDA compatibility: compiled_arches={arches_text}  "
      f"kernel_preflight={preflight_text}")
    cloud = env.get("cloud") if isinstance(env.get("cloud"), dict) else {}
    provider = cloud.get("provider") or "N/A"
    reported_sku = cloud.get("sku_reported") or "N/A"
    changed = env.get("environment_changed")
    changed_text = "N/A" if changed is None else str(bool(changed))
    experiment_wall = _finite_number(root.get("experiment_wall_seconds"))
    experiment_wall_text = (
        f"{experiment_wall:.1f}s ({experiment_wall / 60.0:.1f}min)"
        if experiment_wall is not None else "N/A"
    )
    p(f"    cloud: provider={provider}  reported_sku={reported_sku}  "
      f"actual_gpu={gpu}  experiment_wall={experiment_wall_text}")
    p(f"    environment_changed={changed_text}  "
      "(reported SKU is metadata; actual fingerprint is authoritative)")
    comparison_profile = root.get("comparison_profile")
    if isinstance(comparison_profile, dict):
        subset = str(comparison_profile.get("prompt_subset_sha256") or "N/A")
        p(f"    paired comparison: split={comparison_profile.get('split', 'N/A')}  "
          f"batch={comparison_profile.get('batch_size', 'N/A')}  "
          f"prompt_subset_sha256={subset[:12]}")

    states = _resource_state_records(fl)
    if not states:
        p("    states: N/A — resource container exists but contains no measurements")
    else:
        p("    logical tokens (emitted IDs are exact; unfinished natural lengths are lower bounds)")
        p("    state         N       input      output       total   cap-hit   budget")
        for state in states:
            p(f"    {_resource_state_label(state['name']):<10} "
              f"{_resource_int(_resource_metric(state, 'num_requests')):>7} "
              f"{_resource_int(_resource_metric(state, 'total_input_tokens')):>11} "
              f"{_resource_int(_resource_metric(state, 'total_output_tokens')):>11} "
              f"{_resource_int(_resource_metric(state, 'total_tokens')):>11} "
              f"{_resource_pct(_resource_metric(state, 'cap_hit_rate')):>9} "
              f"{_resource_pct(_resource_metric(state, 'budget_utilization')):>8}")

        p("    measured hardware (one row per batch size; null means unavailable, never zero)")
        p("    state@batch         N  wall_s  CUDA_s   e2e_s    req/s  out_tok/s  peak_GiB  allocΔGiB  reservΔGiB  energy_J method/coverage")
        for state in states:
            profiles = _resource_hardware_profiles_for_report(state) or [("", {})]
            for profile_name, profile in profiles:
                measured_state = dict(state)
                measured_state["hardware"] = profile
                suffix = (profile_name.replace("batch_size_", "bs").replace("/", "-")
                          if profile_name else
                          (f"bs{batch}" if batch != "N/A" else "default"))
                label = f"{_resource_state_label(state['name'])}@{suffix}"
                method = _profile_energy_method(profile, state.get("raw"))
                attribution = _first_text(
                    (profile.get("summary") if isinstance(profile, dict) else None,
                     profile, state.get("raw")),
                    ("energy_attribution", "attribution"),
                )
                coverage = _profile_validity_note(profile)
                method_base = (
                    f"{method}/{attribution}"
                    if attribution and method != "N/A" else method
                )
                method_note = f"{method_base} [{coverage}]" if coverage else method_base
                profile_summary = profile.get("summary") if isinstance(profile, dict) else {}
                profile_n = _first_metric((profile_summary, profile), ("n_requests", "requests"))
                p(f"    {label:<18} "
                  f"{_resource_int(profile_n):>4} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'generation_wall_seconds')):>7} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'cuda_elapsed_seconds')):>7} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'e2e_wall_seconds')):>7} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'requests_per_second')):>8} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'output_tokens_per_second'), '.1f'):>10} "
                  f"{_resource_gib(_resource_metric(measured_state, 'peak_allocated_bytes')):>9} "
                  f"{_resource_gib(_resource_metric(measured_state, 'incremental_peak_allocated_bytes')):>10} "
                  f"{_resource_gib(_resource_metric(measured_state, 'incremental_peak_reserved_bytes')):>11} "
                  f"{_resource_fmt(_resource_metric(measured_state, 'energy_joules')):>9} {method_note}")
                iqr_note = _profile_iqr_note(profile)
                if iqr_note:
                    p(f"      {label}: {iqr_note}")

    comparisons = _resource_comparison_rows(fl)
    if comparisons:
        p("    amplification                    attacked/pristine  attacked/benign")
        for metric, ap, ab in comparisons:
            label = _RESOURCE_COMPARISON_LABELS.get(metric, metric.replace("_", " "))
            p(f"    {label:<32} {_resource_ratio(ap):>18} "
              f"{_resource_ratio(ab):>16}")

    validity = root.get("validity")
    if isinstance(validity, dict):
        invalid = []
        for key, value in validity.items():
            if value is False:
                invalid.append(f"{key}=invalid")
            elif isinstance(value, dict) and value.get("valid") is False:
                reason = value.get("reason") or value.get("status") or "invalid"
                invalid.append(f"{key}={reason}")
            elif key in {"hardware", "cuda", "energy"} and isinstance(value, str):
                if value.lower() not in {"valid", "enabled", "ok", "complete"}:
                    invalid.append(f"{key}={value}")
        if invalid:
            p("    validity: " + "; ".join(invalid))
    return L


def resource_digest(fl: Optional[Dict], print_output: bool = True) -> str:
    """Return a compact resource-v1 report and optionally print it for a Colab cell."""
    text = "\n".join(_resource_digest_lines(fl))
    if print_output:
        print(text)
    return text


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
    sel = amp_clean = trunc = joint = kappa = dec_valid = None
    if fl and fl.get("durability"):
        d = fl["durability"][-1]; pri = fl.get("pristine_reference", {})
        rep, distinct = d.get("repetition_tau"), d.get("distinct_ratio_tau")
        amp, amp_eff, amp_dec = d.get("amp_tau"), d.get("amp_tau_effective"), d.get("amp_tau_decensored")
        dec_valid = d.get("decensored_valid", True)
        ppl_pri = d.get("ppl_ratio_vs_pristine")
        rouge_cln, rouge_cln_ref = d.get("rouge_recall_clean_atk"), pri.get("rouge_recall_clean")
        sel, amp_clean, trunc, joint = d.get("selectivity"), d.get("amp_clean"), d.get("truncation_tau"), d.get("stealth_ok")
    elif phase0:
        c, u, s = phase0.get("cost", {}), phase0.get("utility", {}), phase0.get("stealth", {})
        at, bt = c.get("attacked_tau", {}), c.get("baseline_tau", {})
        rep, distinct = at.get("mean_repetition"), at.get("mean_distinct_ratio")
        amp, amp_eff = c.get("amplification_tau"), c.get("effective_amplification_tau")
        dec_valid = at.get("decensored_valid", c.get("decensored_valid", True))
        if dec_valid and at.get("decensored_mean_cost") and bt.get("decensored_mean_cost"):
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
        if dec_valid is False:
            hints.append(
                f"截断率很高 (trunc={trunc:.2f}) 且去删失不可识别 → 当前 token 是可靠实测下界；"
                "若诊断自然长度，只在小型子集有限提高 max_new_tokens，并保留 wall-clock guard，绝不取消上限。"
            )
        else:
            hints.append(
                f"截断率仍高 (trunc={trunc:.2f}) → 当前实测 token 为下界；若需诊断可在小型子集"
                "有限提高 max_new_tokens，并保留 wall-clock guard，绝不取消上限。"
            )

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
        dec_valid = at.get("decensored_valid", c.get("decensored_valid", True))
        if dec_valid and at.get("decensored_mean_cost") and bt.get("decensored_mean_cost"):
            dec_amp = round(at["decensored_mean_cost"] / max(bt["decensored_mean_cost"], 1e-9), 3)
        p(f"    amp_tau mean={_f(c.get('amplification_tau'))} med={_f(c.get('amplification_tau_median'))} "
          f"dec(cap-corr)={_f(dec_amp, dash='N/A')} eff(useful)={_f(c.get('effective_amplification_tau'))} "
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
            dec0 = f0.get("amp_tau_decensored") if f0.get("decensored_valid", True) else None
            decN = fN.get("amp_tau_decensored") if fN.get("decensored_valid", True) else None
            len_decN = (fN.get("tau_len_atk_decensored")
                        if fN.get("decensored_valid", True) else None)
            p(f"    amp_tau {_f(f0.get('amp_tau'))}(r{f0.get('round')}) -> {_f(fN.get('amp_tau'))}(r{fN.get('round')})  "
              f"de-censored {_f(dec0, dash='N/A')}->{_f(decN, dash='N/A')}  "
              f"eff(useful) {_f(f0.get('amp_tau_effective'))}->{_f(fN.get('amp_tau_effective'))}  "
              f"vs_pristine {_f(fN.get('amp_tau_vs_pristine'))}  med {_f(fN.get('amp_tau_median'))}")
            p(f"    tau_len {_f(f0.get('tau_len_atk'))}->{_f(fN.get('tau_len_atk'))} "
              f"(de-censored {_f(len_decN, dash='N/A')}, effective {_f(fN.get('tau_effective_len_atk'))})  "
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
                dec = q.get("amp_tau_decensored") if q.get("decensored_valid", True) else None
                p(f"    {_f(q.get('round')):>4}  {_f(q.get('amp_tau'),'>5.2f')} "
                  f"{_f(dec,'>6.2f', dash='N/A'):>6} {_f(q.get('amp_tau_vs_pristine'),'>6.2f')} "
                  f"{_f(q.get('truncation_tau'),'>5.2f')} {_f(q.get('repetition_tau'),'>5.2f')} "
                  f"{_f(q.get('ppl_ratio_vs_pristine'),'>6.3f')} {_f(q.get('rouge_recall_clean_atk'),'>5.2f')} "
                  f"{_f(q.get('rouge_recall_tau_atk'),'>5.2f')}  {q.get('stealth_ok')}")
        st = [x for x in fl.get("stealth_trace", []) if x.get("n_attackers")]
        ok = sum(1 for x in st if x.get("jointly_satisfied"))
        p(f"    stealth jointly satisfied {ok}/{len(st)} attacker-participating rounds")
        L.extend(_resource_digest_lines(fl))

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
    p("#       amp_dec 仅在 decensored_valid=True 时展示；否则 N/A（cap 饱和、不可识别）。")
    p("#       JOINT=False → 隐蔽未满足。资源表中的 N/A 表示未采集/硬件不支持，不表示 0。")
    p("#" * 78)
    text = "\n".join(L)
    print(text)
    return text
