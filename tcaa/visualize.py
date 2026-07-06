# tcaa/visualize.py
# Publication-style figures for TCAA Phase-0 results, mapped to the paper's claims.
# Design follows the dataviz skill: pick the form by the data's job; color-blind-safe
# categorical hues in fixed order (Okabe-Ito); single axis (stacked subplots, never
# dual-y); thin marks; recessive grid; legend + direct value labels; text in ink.
#
# Figures:
#   1 cost_amplification  (C1) — triggered inputs cost more; clean unchanged
#   2 length_distribution (C1/C4) — output-length shift on tau vs clean
#   3 utility             (C2)  — perplexity preserved
#   4 stealth             (C3)  — attacker vs benign envelope (distance / cosine)
#   5 attack_trace        (method) — L_mal, E[L]_tau, EOS prob over optimization
#   6 cost_model          (C4)  — cost curve C(L) + super-linear threshold

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

# --- palette (Okabe-Ito; strong CVD separation) & ink -------------------------
C_BASE = "#0072B2"      # baseline (blue)
C_ATK = "#D55E00"       # attacked (vermillion)
C_BENIGN = "#56B4E9"    # benign agents (sky blue)
C_OK = "#009E73"        # feasible / satisfied (green)
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


def _grouped(ax, groups, base_vals, atk_vals, ylabel):
    import numpy as np
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w/2, base_vals, w, label="Baseline (benign-only global)", color=C_BASE)
    b2 = ax.bar(x + w/2, atk_vals, w, label="Attacked (with TCAA agent)", color=C_ATK)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(base_vals + atk_vals) * 1.22)


# --- figures ------------------------------------------------------------------
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
    u = r["utility"]
    base = [u["ppl_clean_baseline"], u["ppl_tau_baseline"]]
    atk = [u["ppl_clean_attacked"], u["ppl_tau_attacked"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    _grouped(ax, ["Clean", "Triggered (τ)"], base, atk, "Perplexity (lower = better)")
    ax.set_title(f"(c) Utility preserved  (clean ppl ×{u['ppl_clean_ratio']:.3f})")
    fig.tight_layout(); return fig


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
    axes[2].plot(steps, [t["mean_eos_prob_tau"] for t in tr], color="#CC79A7", lw=2, marker="o", ms=3)
    axes[2].set_ylabel("mean $q_{EOS}$ on τ  (↓)"); axes[2].set_xlabel("optimization step")
    for ax in axes:
        ax.grid(axis="x", visible=False)
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
    ("cost_amplification", fig_cost_amplification),
    ("length_distribution", fig_length_distribution),
    ("utility", fig_utility),
    ("stealth", fig_stealth),
    ("attack_trace", fig_attack_trace),
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


def render_report(results: Dict):
    """For notebooks: build every figure and show it inline (Colab/Jupyter)."""
    titles = {
        "cost_amplification": "成本放大 (C1)", "length_distribution": "输出长度分布 (C1/C4)",
        "utility": "效用保持 (C2)", "stealth": "参数空间隐蔽性 (C3)",
        "attack_trace": "攻击优化轨迹 (方法)", "cost_model": "成本模型 (C4)",
    }
    figs = make_all_figures(results)
    for key, fig in figs:
        print(f"\n=== {titles.get(key, key)} ===")
        plt.figure(fig.number)
        plt.show()
    return figs
