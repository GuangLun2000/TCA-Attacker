# tcaa/visualize.py
# Publication-style figures for TCAA results, mapped to the paper's claims.
# Design follows the dataviz skill: pick the form by the data's job; color-blind-safe
# categorical hues in fixed order (Okabe-Ito); single axis (stacked subplots, never
# dual-y); thin marks; recessive grid; legend + direct value labels; text in ink.
#
# Multi-round FL (render_fl_report): fl_durability, fl_utility, fl_stealth,
#   fl_defense_geometry (per-client cos/dist/norm/Krum over rounds, benign vs attacker;
#   the AugMP-visualization "defense's-eye view" from fl_runner's defense_telemetry)

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

# Export policy shared by both FL objectives. PNG is convenient for
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
        "savefig.pad_inches": 0.10,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif",
        ],
        "font.size": 12,
        "axes.facecolor": "white",
        # Recessive frame: only the left+bottom rails, in a soft gray — the data, not the box,
        # carries the ink (the single biggest "elegant vs. busy" lever for scientific figures).
        "axes.edgecolor": "#8a8a8a",
        "axes.linewidth": 0.9,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.titlecolor": INK,
        "axes.labelsize": 12.5,
        "axes.labelpad": 6,
        "axes.labelcolor": INK,
        "axes.labelweight": "medium",
        "axes.axisbelow": True,
        "axes.spines.top": False,      # recede the box: top/right rails removed everywhere
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",         # horizontal reference lines only; vertical grid is noise
        "text.color": INK,
        "xtick.color": "#8a8a8a",      # tick marks recede; labels stay ink (set below)
        "ytick.color": "#8a8a8a",
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fancybox": False,
        "legend.edgecolor": "#e2e2e2",   # near-invisible hairline frame, not a hard black box
        "legend.borderpad": 0.5,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35,
        "grid.color": "#dcdcdc",         # hairline solid grid, very light — a whisper, not a net
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.8,
        # NOTE: no global white marker ring — on dense line-markers a white edge punches gaps that
        # read as a dashed line. White rings are applied per-figure only where marks overlap
        # (the defense-geometry scatter), never on the continuous durability/utility/stealth lines.
        "lines.solid_capstyle": "round",
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
    has_len_dec = any(p.get("decensored_valid", True)
                      and p.get("tau_len_atk_decensored") is not None for p in dur)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)
    # --- (A) cost amplification: HONEST headline series first (Framing B) ---
    # Lead with the claim-bearing curves — effective (degeneracy-discounted useful length),
    # vs-pristine (fixed round-0 anchor), and calibrated (real decoder coeffs c_f=d_model).
    # Raw amp is shown as a FAINT caveated reference only: it credits the quadratic cost term
    # from token 1 and inflates with the cap, so it must not be the visual headline. Median and
    # de-censored (denominator artifact / assumption-only) are omitted here and live in the text.
    ax = axes[0]
    def _series(key):
        return [p.get(key, float("nan")) for p in dur]
    has_eff = any(p.get("amp_tau_effective") is not None for p in dur)
    has_cal = any(p.get("amp_tau_calibrated") is not None for p in dur)
    if has_eff:
        ax.plot(rounds, _series("amp_tau_effective"), "-o", color=C_OK, lw=2.4, ms=5.5,
                label="effective (useful-length)", zorder=5)
    ax.plot(rounds, _series("amp_tau_vs_pristine"), "-D", color=C_BASE, lw=2.2, ms=5,
            label="vs pristine (fixed anchor)", zorder=4)
    if has_cal:
        ax.plot(rounds, _series("amp_tau_calibrated"), "-s", color=C_PURPLE, lw=2.0, ms=4.5,
                label="calibrated compute (c_f=d_model)", zorder=3)
    ax.plot(rounds, _series("amp_tau"), "--", color=MUTED, lw=1.3, alpha=0.8,
            label="raw amp (quadratic · cap-inflated)", zorder=2)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylabel("cost amplification (τ)")
    _use_log_scale_if_needed(ax, _series("amp_tau_effective") + _series("amp_tau_vs_pristine")
                             + _series("amp_tau_calibrated") + _series("amp_tau"))
    # direct-label the final effective value (the honest headline number)
    if has_eff and len(rounds) > 1:
        ev = dur[-1].get("amp_tau_effective")
        if ev is not None:
            ax.annotate(f"{ev:.1f}× useful", (rounds[-1], ev),
                        xytext=(-4, 10), textcoords="offset points", ha="right",
                        fontsize=9.5, fontweight="semibold", color=C_OK)
    ax.set_title("TCAA multi-round durability (rapidly saturating, sustained under poisoning)")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE, ncol=2); ax.grid(axis="x", visible=False)
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


def fig_fl_defense_evasion(r: Dict):
    """C3 HEADLINE — the honest defense-evasion verdict as a diverging bar chart.

    For each named robust aggregator we replay offline, the discriminative power is
    excess_detection = attacker_flag_rate - benign_flag_rate (a base-rate-corrected number;
    a raw caught_rate is confounded because cosine-screen flags f/n of RANDOM clients and Krum
    rejects all-but-one). This is a POLARITY story, so it gets a diverging encoding centered on
    zero: bars to the LEFT / green = the attacker is flagged NO MORE than a benign client
    (indistinguishable => stealth holds); bars to the RIGHT / vermillion = detectable. A shaded
    band around 0 marks the "indistinguishable" zone. Krum is excluded from the bars (it keeps 1,
    so its excess is range-compressed and non-comparable) and reported as a caption instead."""
    ev = ((r.get("defense_evaluation") or {}).get("telemetry_defenses") or {})
    defs = ev.get("defenses") or {}
    if not defs:
        return None
    # comparable (fixed-f) defenses only, sorted most-detectable first (top of chart)
    comparable = [(n, d) for n, d in defs.items()
                  if not d.get("excess_structural") and d.get("excess_detection") is not None]
    if not comparable:
        return None
    comparable.sort(key=lambda kv: kv[1]["excess_detection"])   # ascending -> most stealthy at bottom
    names = [n for n, _ in comparable]
    excess = [float(d["excess_detection"]) for _, d in comparable]

    fig, ax = plt.subplots(figsize=(7.8, 0.92 * len(names) + 2.3))
    y = list(range(len(names)))
    band = 0.10  # fallback "practically indistinguishable" zone when no null band is available
    ax.axvline(0.0, color=INK, lw=1.3, zorder=3)

    drew_null = False
    for yi, (nm, d) in zip(y, comparable):
        e = float(d["excess_detection"])
        # "Detectable" requires SIGNIFICANCE under the strongest available null (client-level
        # permutation preferred; see defenses.evaluate_defenses): a point estimate past the band
        # that the null itself reaches is noise. Purpose-built detectors (ours) are hatched so a
        # reader never mistakes them for an off-the-shelf defense the attack failed to evade.
        sig = d.get("excess_significant")
        detectable = (bool(sig) and e > 0) if sig is not None else (e > band)
        col = C_ATK if detectable else C_OK
        # The H0 REGION, drawn as a shaded span BEHIND the bar — never as whiskers on the bar.
        # excess_null_band95 is where the excess lands when the attacker label carries no
        # information; a real effect is supposed to stick out PAST it. Drawing it as a CI on the
        # estimate (which an earlier version did) puts the bar outside its own error bars and
        # reads as a broken chart.
        nb = d.get("excess_null_band95")
        if nb:
            ax.barh(yi, nb[1] - nb[0], left=nb[0], height=0.78, color=MUTED, alpha=0.16,
                    zorder=1, linewidth=0)
            drew_null = True
        ax.barh(yi, e, height=0.56, color=col, alpha=0.92, zorder=2,
                edgecolor="white", linewidth=1.0,
                hatch="///" if d.get("purpose_built") else None)
        # value label just past the bar end, on the correct side
        off = 0.012 if e >= 0 else -0.012
        p_clu = d.get("excess_p_cluster")
        p_lab = f"  p={p_clu:.3f}" if p_clu is not None else ""
        ax.annotate(f"{e:+.2f}{p_lab}", (e + off, yi), va="center",
                    ha="left" if e >= 0 else "right", fontsize=11, fontweight="semibold",
                    color=col)
        # secondary: the two rates the excess is built from
        ax.annotate(f"atk {d.get('atk_flag_rate', 0):.0%} vs benign {d.get('ben_flag_rate', 0):.0%}"
                    f"  ·  {d.get('rounds', 0)} rnds",
                    (0, yi - 0.34), va="center", ha="center", fontsize=8.4, color=MUTED, zorder=4)
    if not drew_null:
        ax.axvspan(-band, band, color="#009E73", alpha=0.08, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", "-") for n in names], fontsize=11.5)
    _xlab = "excess detection  =  attacker flag-rate − benign flag-rate"
    if drew_null:
        _xlab += "      (grey band = 95% of the H0 null, not a CI on the estimate)"
    ax.set_xlabel(_xlab)
    nulls = [v for _, d in comparable for v in (d.get("excess_null_band95") or [])]
    lo = min(excess + nulls + [-band]); hi = max(excess + nulls + [band])
    pad = 0.10 * max(hi - lo, 0.2)
    ax.set_xlim(lo - pad - 0.06, hi + pad + 0.10)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=0.6)
    ax.margins(y=0.16)

    # directional guidance under the zero line
    ax.annotate("← indistinguishable from benign (stealthy)", (0, -0.9),
                ha="right", va="center", fontsize=8.6, color=C_OK, fontstyle="italic",
                annotation_clip=False, xytext=(-6, 0), textcoords="offset points")
    ax.annotate("detectable →", (0, -0.9), ha="left", va="center", fontsize=8.6,
                color=C_ATK, fontstyle="italic", annotation_clip=False,
                xytext=(6, 0), textcoords="offset points")

    # The verdict is about STANDARD defenses only; a purpose-built detector catching the attack
    # is a boundary result, not a failure to evade (and is hatched in the bars above).
    def _sig_detect(d):
        e = float(d.get("excess_detection") or 0)
        s = d.get("excess_significant")
        return (bool(s) and e > 0) if s is not None else (e > band)

    std_pairs = [(n, d) for n, d in comparable if not d.get("purpose_built")]
    built_pairs = [(n, d) for n, d in comparable if d.get("purpose_built")]
    std_stealthy = bool(std_pairs) and not any(_sig_detect(d) for _, d in std_pairs)
    verdict = ("indistinguishable from a benign client under standard defenses"
               if std_stealthy else "significantly detectable by a standard defense")
    sub = "robust aggregators replayed offline on FedAvg telemetry"
    if built_pairs:
        sub += "  ·  hatched = purpose-built detector (ours), not an off-the-shelf defense"
    ax.set_title(f"Defense evasion (C3): attacker is {verdict}\n{sub}", fontsize=12.5)

    # Krum + vector-defense caption (structural / optimistic notes live here, not in the bars)
    notes = []
    krum = defs.get("krum")
    if krum:
        notes.append(f"Krum (keeps 1, structural): caught {krum.get('caught_rate', 0):.0%}, "
                     f"survival {krum.get('survival_rate', 0):.0%} — read by survival, not excess")
    vec = ((r.get("defense_evaluation") or {}).get("vector_defenses") or {})
    if vec.get("fltrust"):
        notes.append(f"FLTrust caught {vec['fltrust'].get('caught_rate', 0):.0%} "
                     f"(root={vec['fltrust'].get('root')}, optimistic proxy)")
    if notes:
        fig.text(0.5, -0.02 / (0.92 * len(names) + 2.3), "   ·   ".join(notes),
                 ha="center", va="top", fontsize=8.2, color=MUTED)
    fig.tight_layout()
    return fig


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


_REASONING_GATE_ORDER = (
    "measurement_valid",
    "cost_ci_lower_gt_1",
    "cost_ci_lower_meets_effect_size",
    "tau_reference_accuracy_valid",
    "clean_reference_accuracy_valid",
    "tau_accuracy_preserved",
    "clean_accuracy_preserved",
    "clean_cost_stable",
    "closure_valid",
    "clean_closure_valid",
    "cap_hit_valid",
    "clean_cap_hit_valid",
    "repetition_valid",
    "distinct_valid",
)


def reasoning_gate_rows(fl_results: Dict) -> List[Dict]:
    """Return the preregistered reasoning gates as a stable, CSV-ready table."""
    cfg = fl_results.get("config") or {}
    durability = fl_results.get("durability") or []
    if cfg.get("attack_objective") != "reasoning_cost" or not durability:
        return []
    final = durability[-1]
    gates = final.get("reasoning_gates") or {}
    paired = final.get("paired_reasoning_cost_tau") or {}
    clean_paired = final.get("paired_reasoning_cost_clean") or {}
    clean_logical = final.get("resource_clean_attacked") or {}
    values = {
        "measurement_valid": final.get("reasoning_measurement_valid"),
        "cost_ci_lower_gt_1": paired.get("ci95_lower"),
        "cost_ci_lower_meets_effect_size": paired.get("ci95_lower"),
        "tau_reference_accuracy_valid": final.get("answer_accuracy_tau_ref"),
        "clean_reference_accuracy_valid": final.get("answer_accuracy_clean_ref"),
        "tau_accuracy_preserved": final.get("answer_accuracy_drop_tau"),
        "clean_accuracy_preserved": final.get("answer_accuracy_drop_clean"),
        "clean_cost_stable": clean_paired.get("median_ratio"),
        "closure_valid": final.get("reasoning_closure_tau"),
        "clean_closure_valid": final.get("reasoning_closure_clean"),
        "cap_hit_valid": final.get("cap_hit_rate_tau", final.get("truncation_tau")),
        "clean_cap_hit_valid": clean_logical.get(
            "cap_hit_rate", clean_logical.get("truncation_rate")
        ),
        "repetition_valid": final.get("reasoning_repetition_tau"),
        "distinct_valid": final.get("reasoning_distinct_tau"),
    }
    labels = {
        "measurement_valid": "four attacked/reference measurements valid",
        "cost_ci_lower_gt_1": "paired cost CI lower bound is positive",
        "cost_ci_lower_meets_effect_size": "paired cost CI lower bound is meaningful",
        "tau_reference_accuracy_valid": "triggered reference accuracy",
        "clean_reference_accuracy_valid": "clean reference accuracy",
        "tau_accuracy_preserved": "triggered accuracy drop",
        "clean_accuracy_preserved": "clean accuracy drop",
        "clean_cost_stable": "clean paired-cost ratio",
        "closure_valid": "triggered reasoning closure",
        "clean_closure_valid": "clean reasoning closure",
        "cap_hit_valid": "triggered cap-hit rate",
        "clean_cap_hit_valid": "clean cap-hit rate",
        "repetition_valid": "triggered reasoning repetition",
        "distinct_valid": "triggered reasoning distinct-4",
    }
    rules = {
        "measurement_valid": ("==", True),
        "cost_ci_lower_gt_1": (">", 1.0),
        "cost_ci_lower_meets_effect_size": (
            ">=", cfg.get("reasoning_min_claim_cost_ratio")
        ),
        "tau_reference_accuracy_valid": (
            ">=", cfg.get("reasoning_min_reference_accuracy")
        ),
        "clean_reference_accuracy_valid": (
            ">=", cfg.get("reasoning_min_reference_accuracy")
        ),
        "tau_accuracy_preserved": ("<=", cfg.get("reasoning_max_accuracy_drop")),
        "clean_accuracy_preserved": ("<=", cfg.get("reasoning_max_accuracy_drop")),
        "clean_cost_stable": (
            "abs(x-1)<=", cfg.get("reasoning_clean_cost_tolerance")
        ),
        "closure_valid": (">=", cfg.get("reasoning_min_closure_rate")),
        "clean_closure_valid": (">=", cfg.get("reasoning_min_closure_rate")),
        "cap_hit_valid": ("<", cfg.get("reasoning_max_cap_hit_rate")),
        "clean_cap_hit_valid": ("<", cfg.get("reasoning_max_cap_hit_rate")),
        "repetition_valid": ("<", cfg.get("reasoning_max_repetition")),
        "distinct_valid": (">", cfg.get("reasoning_min_distinct_ratio")),
    }
    return [
        {
            "gate": name,
            "label": labels[name],
            "observed": values[name],
            "operator": rules[name][0],
            "threshold": rules[name][1],
            "passed": bool(gates.get(name)),
        }
        for name in _REASONING_GATE_ORDER
    ]


def reasoning_effect_summary(fl_results: Dict, *, run_tier: str = "pilot") -> Dict:
    """Build one machine-readable interpretation of a reasoning experiment.

    A pilot can expose a useful signal, but it is never promoted to a formal claim.
    The summary deliberately keeps logical effect, task validity, parameter stealth,
    hardware evidence, and evidence scope separate.
    """
    if run_tier not in {"pilot", "formal"}:
        raise ValueError("run_tier must be 'pilot' or 'formal'")
    rows = reasoning_gate_rows(fl_results)
    if not rows:
        raise ValueError("reasoning_effect_summary requires a reasoning_cost FL result")
    cfg = fl_results.get("config") or {}
    final = (fl_results.get("durability") or [])[-1]
    paired = final.get("paired_reasoning_cost_tau") or {}
    clean_paired = final.get("paired_reasoning_cost_clean") or {}
    gates = final.get("reasoning_gates") or {}
    objective = (
        ((fl_results.get("objective_summary") or {}).get("reasoning_attack_final") or {})
    )
    resources = fl_results.get("resources") or {}
    environment = resources.get("environment") or {}
    start_environment = environment.get("start") or environment
    gpus = start_environment.get("gpus") or []
    gpu_name = (
        (gpus[0] or {}).get("name") if gpus
        else start_environment.get("gpu_name_actual")
    )
    stealth_rows = [
        row for row in (fl_results.get("stealth_trace") or []) if row.get("n_attackers")
    ]
    stealth_passes = sum(bool(row.get("jointly_satisfied")) for row in stealth_rows)
    stealth_ready = bool(
        stealth_rows and stealth_passes == len(stealth_rows)
        and len(stealth_rows) == int(cfg.get("num_rounds") or 0)
    )
    failed = [row["gate"] for row in rows if not row["passed"]]
    logical_ready = bool(final.get("reasoning_claim_ready"))
    measurement_valid = bool(gates.get("measurement_valid"))
    cost_signal = bool(
        gates.get("cost_ci_lower_gt_1")
        and gates.get("cost_ci_lower_meets_effect_size")
    )
    hardware_enabled = bool(cfg.get("profile_hardware"))
    hardware_ready = bool(objective.get("hardware_resource_effect_claim_ready"))
    if not measurement_valid:
        outcome = "measurement_invalid"
    elif logical_ready:
        outcome = "positive_logical_signal"
    elif cost_signal:
        outcome = "cost_signal_but_quality_or_selectivity_failed"
    else:
        outcome = "no_clear_cost_effect"

    actions: List[str] = []
    failed_set = set(failed)
    if "measurement_valid" in failed_set:
        actions.append("Inspect closure/cap/time-limit diagnostics before changing attack strength.")
    if failed_set & {"cost_ci_lower_gt_1", "cost_ci_lower_meets_effect_size"}:
        actions.append("If anchors are healthy, increase attacker steps before raising the target ratio.")
    if failed_set & {"tau_reference_accuracy_valid", "clean_reference_accuracy_valid"}:
        actions.append("Do not tune the attack yet; first obtain a competent broadcast reference.")
    if failed_set & {"tau_accuracy_preserved", "clean_accuracy_preserved"}:
        actions.append("Strengthen answer/clean anchors or reduce the target ratio.")
    if "clean_cost_stable" in failed_set:
        actions.append("Increase the clean-cost anchor weight or frequency.")
    if failed_set & {"closure_valid", "clean_closure_valid", "cap_hit_valid", "clean_cap_hit_valid"}:
        actions.append("Reduce target pressure and inspect termination before increasing decode caps.")
    if failed_set & {"repetition_valid", "distinct_valid"}:
        actions.append("Increase anti-repetition pressure; do not interpret looping as useful reasoning.")
    if not stealth_ready:
        actions.append("Tighten or debug the ALM parameter-stealth constraint.")
    if not hardware_enabled:
        actions.append("Enable paired hardware profiling for latency evidence.")
    elif not hardware_ready:
        actions.append("Inspect paired wall/CUDA repeat CIs; hardware evidence did not clear its gate.")
    if run_tier == "pilot":
        actions.append("Treat this as single-seed calibration; run the formal tier only after reviewing these diagnostics.")

    return {
        "schema_version": "reasoning-feedback-v1",
        "run_id": fl_results.get("run_id"),
        "run_tier": run_tier,
        "evidence_scope": (
            "exploratory_single_seed_pilot"
            if run_tier == "pilot" else "single_seed_formal_run"
        ),
        "outcome": outcome,
        "logical_resource_effect_ready": logical_ready,
        "formal_claim_ready": False,
        "formal_claim_note": (
            "A pilot is never claim-bearing."
            if run_tier == "pilot"
            else "A single formal run still requires the preregistered multi-seed evidence."
        ),
        "execution": {
            "experiment_wall_seconds": resources.get("experiment_wall_seconds"),
            "gpu_name": gpu_name,
            "environment_fingerprint": start_environment.get("fingerprint_sha256"),
            "hardware_profile": resources.get("comparison_profile"),
        },
        "paired_reasoning_cost": {
            "median_ratio": paired.get("median_ratio"),
            "ci95_lower": paired.get("ci95_lower"),
            "ci95_upper": paired.get("ci95_upper"),
            "minimum_ci_lower_ratio": cfg.get("reasoning_min_claim_cost_ratio"),
            "n_pairs": paired.get("n_pairs"),
            "n_decode_samples": paired.get("n_decode_samples"),
            "reference_reasoning_tokens": final.get("reasoning_tokens_tau_ref"),
            "attacked_reasoning_tokens": final.get("reasoning_tokens_tau_atk"),
        },
        "task_quality": {
            "triggered_reference_accuracy": final.get("answer_accuracy_tau_ref"),
            "triggered_attacked_accuracy": final.get("answer_accuracy_tau_atk"),
            "triggered_accuracy_drop": final.get("answer_accuracy_drop_tau"),
            "clean_reference_accuracy": final.get("answer_accuracy_clean_ref"),
            "clean_attacked_accuracy": final.get("answer_accuracy_clean_atk"),
            "clean_accuracy_drop": final.get("answer_accuracy_drop_clean"),
            "triggered_closure_rate": final.get("reasoning_closure_tau"),
            "clean_closure_rate": final.get("reasoning_closure_clean"),
            "reasoning_repetition": final.get("reasoning_repetition_tau"),
            "reasoning_distinct_ratio": final.get("reasoning_distinct_tau"),
        },
        "clean_selectivity": {
            "paired_cost_ratio": clean_paired.get("median_ratio"),
            "tolerance_around_one": cfg.get("reasoning_clean_cost_tolerance"),
        },
        "parameter_stealth": {
            "ready": stealth_ready,
            "passed_rounds": stealth_passes,
            "total_rounds": len(stealth_rows),
        },
        "hardware": {
            "enabled": hardware_enabled,
            "validity": (resources.get("validity") or {}).get("hardware"),
            "effect_ready": hardware_ready,
            "amplification": objective.get("hardware_amplification"),
        },
        "failed_gates": failed,
        "gate_rows": rows,
        "recommended_next_actions": list(dict.fromkeys(actions)),
    }


def reasoning_feedback_report(
    fl_results: Dict, *, run_tier: str = "pilot", print_output: bool = True
) -> str:
    """Render the machine summary as a compact console/Drive report."""
    summary = reasoning_effect_summary(fl_results, run_tier=run_tier)
    execution = summary["execution"]
    effect = summary["paired_reasoning_cost"]
    quality = summary["task_quality"]
    clean = summary["clean_selectivity"]
    stealth = summary["parameter_stealth"]
    hardware = summary["hardware"]
    hardware_amplification = hardware.get("amplification") or {}

    def paired_hardware_line(label: str, key: str) -> str:
        paired_hw = hardware_amplification.get(key) or {}
        if not paired_hw:
            return f"paired hardware {label}: N/A"
        return (
            f"paired hardware {label}: median={paired_hw.get('median_ratio')}x, "
            f"CI95=[{paired_hw.get('ci95_lower')},{paired_hw.get('ci95_upper')}], "
            f"pairs={paired_hw.get('n_pairs')}"
        )

    lines = [
        "=" * 78,
        f"REASONING-TCAA FIRST-EXPERIMENT FEEDBACK [{run_tier.upper()}]",
        "=" * 78,
        f"outcome={summary['outcome']} | scope={summary['evidence_scope']}",
        (
            f"runtime: gpu={execution.get('gpu_name')}, "
            f"wall_seconds={execution.get('experiment_wall_seconds')}, "
            f"environment={str(execution.get('environment_fingerprint') or 'N/A')[:12]}"
        ),
        (
            "logical effect: median={:.3f}x, CI95=[{:.3f},{:.3f}], required lower>={:.3f}x, "
            "pairs={}, decodes={}".format(
                float(effect.get("median_ratio") or 0.0),
                float(effect.get("ci95_lower") or 0.0),
                float(effect.get("ci95_upper") or 0.0),
                float(effect.get("minimum_ci_lower_ratio") or 0.0),
                effect.get("n_pairs"), effect.get("n_decode_samples"),
            )
        ),
        f"reasoning tokens: {effect.get('reference_reasoning_tokens')} -> {effect.get('attacked_reasoning_tokens')}",
        (
            "task: tau accuracy {} -> {} (drop {}), closure {}; clean accuracy {} -> {} "
            "(drop {}), closure {}".format(
                quality.get("triggered_reference_accuracy"), quality.get("triggered_attacked_accuracy"),
                quality.get("triggered_accuracy_drop"), quality.get("triggered_closure_rate"),
                quality.get("clean_reference_accuracy"), quality.get("clean_attacked_accuracy"),
                quality.get("clean_accuracy_drop"), quality.get("clean_closure_rate"),
            )
        ),
        (
            f"selectivity: clean paired cost={clean.get('paired_cost_ratio')}x; "
            f"reasoning repetition={quality.get('reasoning_repetition')}, "
            f"distinct={quality.get('reasoning_distinct_ratio')}"
        ),
        (
            f"parameter stealth: {stealth.get('passed_rounds')}/{stealth.get('total_rounds')} rounds; "
            f"ready={stealth.get('ready')}"
        ),
        (
            f"hardware: enabled={hardware.get('enabled')}, validity={hardware.get('validity')}, "
            f"effect_ready={hardware.get('effect_ready')}"
        ),
        paired_hardware_line(
            "wall", "generation_wall_seconds_paired"
        ),
        paired_hardware_line(
            "CUDA", "cuda_elapsed_seconds_paired"
        ),
        f"failed gates: {summary['failed_gates'] or 'none'}",
        "formal claim ready: False — " + summary["formal_claim_note"],
        "-" * 78,
        "GATE TABLE",
    ]
    for row in summary["gate_rows"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"  [{mark}] {row['gate']}: observed={row['observed']} "
            f"rule {row['operator']} {row['threshold']}"
        )
    lines += ["-" * 78, "NEXT ACTIONS"]
    lines.extend(f"  - {action}" for action in summary["recommended_next_actions"])
    lines.append("=" * 78)
    text = "\n".join(lines)
    if print_output:
        print(text)
    return text


def fig_reasoning_cost_effect(fl_results: Dict):
    """Paired triggered cost ratios with bootstrap uncertainty and clean selectivity."""
    cfg = fl_results.get("config") or {}
    dur = fl_results.get("durability") or []
    if cfg.get("attack_objective") != "reasoning_cost" or not dur:
        return None
    rounds = [row.get("round") for row in dur]
    medians = [(row.get("paired_reasoning_cost_tau") or {}).get("median_ratio") for row in dur]
    lowers = [(row.get("paired_reasoning_cost_tau") or {}).get("ci95_lower") for row in dur]
    uppers = [(row.get("paired_reasoning_cost_tau") or {}).get("ci95_upper") for row in dur]
    clean = [(row.get("paired_reasoning_cost_clean") or {}).get("median_ratio") for row in dur]
    if any(value is None for value in medians + lowers + uppers):
        return None
    yerr = [[max(0.0, m - lo) for m, lo in zip(medians, lowers)],
            [max(0.0, hi - m) for m, hi in zip(medians, uppers)]]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.errorbar(rounds, medians, yerr=yerr, fmt="-o", color=C_ATK, capsize=4,
                label="triggered paired reasoning cost (95% bootstrap CI)")
    ax.plot(rounds, clean, "--s", color=C_BASE, label="clean paired reasoning cost")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", label="no effect")
    threshold = float(cfg.get("reasoning_min_claim_cost_ratio") or 1.0)
    ax.axhline(threshold, color=C_OK, lw=1.2, ls="--",
               label=f"minimum effect CI lower = {threshold:.2f}x")
    ax.set_xlabel("communication round")
    ax.set_ylabel("attacked / benign paired cost ratio")
    ax.set_title("Reasoning-cost effect and clean selectivity")
    ax.legend(loc="best", fontsize=COMPACT_LEGEND_FONT_SIZE)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_reasoning_gate_status(fl_results: Dict):
    """One glance at every preregistered gate; labels retain observed thresholds."""
    rows = reasoning_gate_rows(fl_results)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9.2, max(4.5, 0.36 * len(rows) + 1.4)))
    ypos = list(range(len(rows)))
    colors = [C_OK if row["passed"] else C_BAD for row in rows]
    ax.barh(ypos, [1.0] * len(rows), color=colors, alpha=0.82, height=0.68)
    ax.set_yticks(ypos)
    ax.set_yticklabels([row["gate"] for row in rows], fontsize=9.2)
    ax.invert_yaxis()
    for y, row in zip(ypos, rows):
        mark = "PASS" if row["passed"] else "FAIL"
        ax.text(0.02, y, mark, va="center", ha="left", color="white",
                fontsize=9, fontweight="bold")
        ax.text(1.02, y, f"observed={row['observed']}  {row['operator']} {row['threshold']}",
                va="center", ha="left", color=INK, fontsize=8.5)
    ax.set_xlim(0, 1.75)
    ax.set_xticks([])
    ax.set_title("Pre-registered reasoning gates (green=pass, red=fail)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def make_fl_figures(fl_results: Dict) -> List[Tuple[str, "plt.Figure"]]:
    apply_style()
    out = []
    reasoning_mode = (
        (fl_results.get("config") or {}).get("attack_objective") == "reasoning_cost"
    )
    objective_figures = (
        (("reasoning_cost_effect", fig_reasoning_cost_effect),
         ("reasoning_gate_status", fig_reasoning_gate_status))
        if reasoning_mode
        else (("fl_durability", fig_fl_durability), ("fl_utility", fig_fl_utility))
    )
    common_figures = (
        ("fl_stealth", fig_fl_stealth),
        ("fl_defense_evasion", fig_fl_defense_evasion),
        ("fl_defense_geometry", fig_fl_defense_geometry),
        ("resource_tokens", fig_resource_tokens),
        ("resource_amplification", fig_resource_amplification),
    )
    for key, fn in objective_figures + common_figures:
        try:
            fig = fn(fl_results)
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] {key} failed: {e}")
            fig = None
        if fig is not None:
            out.append((key, fig))
    return out


def render_fl_report(fl_results: Dict):
    titles = {"reasoning_cost_effect": "推理成本效果：配对比值、置信区间与 clean 选择性",
              "reasoning_gate_status": "预注册 reasoning 门控总表",
              "fl_durability": "多轮放大 durability (快速饱和·持续投毒下稳定 · 含去删失估计与截断率)",
              "fl_utility": "多轮效用保持 (vs 原始骨干绝对基线 · ppl + ROUGE)",
              "fl_stealth": "逐轮隐蔽性 (客户端采样下)",
              "fl_defense_evasion": "★ 防御规避 C3 (excess=攻击者-良性被flag率;≈0=不可区分/隐蔽成立)",
              "fl_defense_geometry": "逐客户端更新几何 (防御视角 · benign 云 vs attacker · cos/距离/范数/Krum)",
              "resource_tokens": "资源账本：输入/输出 token 与长度分位数",
              "resource_amplification": "资源放大：token、时间、显存与能耗"}
    figs = make_fl_figures(fl_results)
    for key, fig in figs:
        print(f"\n=== {titles.get(key, key)} ===")
        plt.figure(fig.number)
        plt.show()
    return figs


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


def _benign_clean_control(fl: Optional[Dict]) -> Dict:
    """The benign-only FL arm's FINAL clean-split statistics — the correct C2 control.

    ``run_fl`` writes these twice: per round into ``durability[].rouge_recall_clean_ben`` /
    ``amp_clean_len`` (recent runs only) and, for the final round, into the resource payload at
    ``resources.states.benign_final.clean_logical``. The second copy predates the first, so runs
    that lack the per-round fields are NOT missing the control — falling back to the pristine model
    there (as the verdict used to) charges ordinary benign federated drift to the attacker and
    turns a −1.1pp attacker effect into a −15.7pp "degradation".

    The fallback is EXACT for the final round, not an approximation: on the 20260807 run
    ``attacked_final.clean_logical.mean_rouge_recall`` == ``durability[-1].rouge_recall_clean_atk``
    == 0.3196 and the mean lengths agree to the stored precision, because both are the same
    ``_measure_cost`` call on the same global. Only the per-ROUND benign trajectory is unavailable.

    Returns ``{}`` when no benign arm was tracked (``track_benign_baseline=False``).
    """
    root = _resource_root(fl) or {}
    states = root.get("states")
    if not isinstance(states, dict):
        return {}
    ben = ((states.get("benign_final") or {}).get("clean_logical") or {})
    atk = ((states.get("attacked_final") or {}).get("clean_logical") or {})
    if not ben:
        return {}
    out = {"rouge_recall": ben.get("mean_rouge_recall"), "rouge_f1": ben.get("mean_rouge_f1"),
           "mean_output_len": ben.get("mean_output_len")}
    b_len, a_len = ben.get("mean_output_len"), atk.get("mean_output_len")
    if b_len and a_len is not None:
        out["len_ratio"] = round(a_len / b_len, 4)
    return {k: v for k, v in out.items() if v is not None}


def _excess_stat_str(d: Dict) -> str:
    """One compact significance string for a defense's excess, shared by the digest, the pillar
    card and the figure caption so the three can never disagree.

    Reads defenses.py's THREE nested nulls in decreasing order of correctness:
      p_clu = client-level permutation (headline: attacker identity is a per-client property held
              fixed across rounds, so this is the null that respects the real sample size),
      p_wr  = within-round hypergeometric (per-round association; anti-conservative for a
              persistent label because one odd client recurs every round),
      band  = the region the excess occupies UNDER H0 (`null95`) — NOT a CI on the estimate, so a
              real effect legitimately sits outside it. Older runs stored a Normal-approx CI under
              `excess_ci95`; that is labelled `CI~` so it is never mistaken for an exact null.
    """
    p_clu = d.get("excess_p_cluster")
    p_wr = d.get("excess_p_within_round", d.get("excess_p_exact"))
    band = d.get("excess_null_band95")
    label = "null95"
    if band is None:
        band = d.get("excess_ci95_normal") or d.get("excess_ci95")
        label = "CI~"
    parts = []
    if band:
        parts.append(f"{label}[{band[0]:+.2f},{band[1]:+.2f}]")
    if p_clu is not None:
        parts.append(f"p_clu={p_clu:.3f}")
    if p_wr is not None:
        parts.append(f"p_rnd={p_wr:.3f}")
    if not parts:
        return ""
    sig = d.get("excess_significant")
    parts.append("sig" if sig else ("ns" if sig is not None else "?"))
    return " " + " ".join(parts)


def _defense_digest_lines(fl: Optional[Dict]) -> List[str]:
    """Copy-safe DEFENSE-EVASION block for the digest — the honest C3 (parameter-stealth) number.

    Previously omitted from the digest even though fl_runner computes and stores it, so a pasted
    digest showed no defense numbers at all. Leads on excess_detection = atk_fpr - ben_fpr (the
    base-rate-corrected discriminative power): ~0 means the attacker update is indistinguishable
    from a genuine benign client under that defense (stealth holds); a large positive value means
    it is detectable. Plain caught_rate alone is confounded (cosine-screen flags f/n of RANDOM
    clients; Krum rejects all but one), which is exactly what the ben_fpr baseline corrects."""
    de = ((fl or {}).get("defense_evaluation") or {})
    tel = (de.get("telemetry_defenses") or {})
    defs = (tel.get("defenses") or {})
    if not defs:
        return []
    L: List[str] = []
    p = L.append
    p(f"    DEFENSE-EVASION (offline replay on FedAvg telemetry; attacker-present rounds="
      f"{_f(tel.get('n_rounds'))}) — excess=atk_fpr-ben_fpr, ~0=stealthy:")
    p(f"      {'defense':<13} {'caught':>7} {'atk_fpr':>8} {'ben_fpr':>8} {'excess':>7} {'rnds':>5}")
    for name, d in defs.items():
        # Krum keeps 1 => its excess is range-compressed & noise-dominated; mark structural, do
        # not print it on the same numeric scale as the fixed-f defenses (read Krum by caught).
        excess = "struct*" if d.get("excess_structural") else _f(d.get('excess_detection'), '>7.2f')
        ci_s = "" if d.get("excess_structural") else _excess_stat_str(d)
        tag = " [purpose-built]" if d.get("purpose_built") else ""
        p(f"      {name:<13} {_f(d.get('caught_rate'),'>7.2f')} {_f(d.get('atk_flag_rate'),'>8.2f')} "
          f"{_f(d.get('ben_flag_rate'),'>8.2f')} {excess:>7} "
          f"{_f(d.get('rounds'),'>5')}{ci_s}{tag}")
        if d.get("mean_atk_clip_factor") is not None:
            p(f"        ^ attacker update attenuated to {_f(d['mean_atk_clip_factor'])}x its norm "
              f"(clipping RESCALES, it does not reject: flagged != defeated)")
    # The client-level null is the headline significance, so surface WHICH client sets beat the
    # real attackers. That is the "an honest heavy data-holder looks more suspicious than the
    # attacker" confound in its most concrete form, and it must not stay buried in the JSON.
    clu = ((fl or {}).get("defense_evaluation") or {}).get("telemetry_defenses") or {}
    clu = clu.get("cluster_permutation") or {}
    if clu:
        any_row = next(iter(clu.values()))
        p(f"      p_clu = client-level permutation over {any_row.get('n_candidates')} candidate "
          f"attacker sets (attackers={any_row.get('attacker_ids')}); p_rnd = within-round null.")
        for name, c in sorted(clu.items()):
            # Krum's excess is structurally range-compressed, so its rank carries no information;
            # printing it here would invite a comparison the module explicitly forbids.
            if (defs.get(name) or {}).get("excess_structural"):
                continue
            beat = [t for t in (c.get("top_competitors") or [])
                    if list(t["clients"]) != list(c.get("attacker_ids") or [])
                    and t["excess"] >= c.get("observed", 0)]
            if beat:
                p(f"        {name}: rank {c['rank']}/{c['n_candidates']} — honest client sets "
                  + ", ".join(f"{t['clients']} ({t['excess']:+.2f})" for t in beat[:2])
                  + " score at least as high as the true attackers")
    rca = (fl or {}).get("rank_collusion_analysis") or {}
    if rca.get("fpr_sweep"):
        p(f"      rank-collusion detector boundary: AUC={_f(rca.get('auc'))} "
          f"(within-round permutation p={rca.get('auc_permutation_p')})")
        p("        calibrated: " + " · ".join(
            f"@benign-FPR {s['target_fpr']:.2f} -> excess {s['excess']:+.3f} (TPR {s['attacker_tpr']:.2f})"
            for s in rca["fpr_sweep"]))
        if rca.get("benign_outranks_attacker"):
            p("        CONFOUND: a benign client's mean score outranks both attackers "
              "(the signal tracks data share as well as collusion)")
    vec = (de.get("vector_defenses") or {})
    ft, tm = vec.get("fltrust"), vec.get("trimmed_mean")
    if ft:
        p(f"      {'FLTrust':<13} {_f(ft.get('caught_rate'),'>7.2f')}  root={ft.get('root')} "
          f"(proxy root = benign mean => optimistic; needs a real root set)")
    if tm:
        p(f"      {'trimmed_mean':<13} {_f(tm.get('caught_rate'),'>7.2f')}  beta={tm.get('trim_beta')}")
    elif not vec:
        p("      (FLTrust / trimmed-mean skipped: set save_update_vectors=True to replay them)")
    return L


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
        # THE TWO ROW FAMILIES ARE MEASURED OVER DIFFERENT PROMPT POPULATIONS and must never be
        # read as one table. Analytic/logical rows are computed over the FULL eval set
        # (cfg['eval_size'], e.g. 128); hardware rows are measured on the profiled prefix only
        # (cfg['resource_profile_eval_size'], e.g. 32), and `prompt_subset_sha256` covers ONLY the
        # hardware rows. Printing them undifferentiated invites the invalid cross-family check
        # ratio(tokens)/ratio(wall) == ratio(tokens/s), which fails here (6.965 vs 6.169) purely
        # because the numerator and denominator come from different prompt sets.
        cfg = (fl or {}).get("config") or {}
        n_logical = cfg.get("eval_size")
        n_hw = cfg.get("resource_profile_eval_size")
        _lp = f"N={n_logical}" if n_logical else "full eval set"
        _hp = f"N={n_hw}" if n_hw else "profiled subset"
        p("    amplification                    attacked/pristine  attacked/benign   population")
        for metric, ap, ab in comparisons:
            label = _RESOURCE_COMPARISON_LABELS.get(metric, metric.replace("_", " "))
            pop = _hp + " meas" if metric in _HARDWARE_METRIC_KEYS else _lp + " calc"
            p(f"    {label:<32} {_resource_ratio(ap):>18} "
              f"{_resource_ratio(ab):>16}   {pop}")
        if n_logical and n_hw and int(n_logical) != int(n_hw):
            p(f"    [!] rows above span TWO prompt populations (analytic {_lp} vs measured {_hp}) —"
              f" never divide one family's ratio by the other's; for a like-for-like comparison"
              f" subset logical_tokens.csv to the {_hp} profiled prompts.")

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


def tuning_hints(fl: Optional[Dict] = None) -> List[str]:
    """Auto-generate concrete next-step tuning suggestions from the measured numbers, so a
    text-only run (Colab output copied back) carries everything needed to steer the next
    experiment toward the three goals: preserve utility / large consumption / stealth.
    Returns a list of hint lines (empty if a metric is missing). Pure heuristics — each
    hint names the knob and the direction."""
    hints: List[str] = []

    # ---- pull the most-advanced-round attacked numbers ----
    rep = distinct = amp = amp_eff = ppl_pri = rouge_cln = rouge_cln_ref = None
    sel = amp_clean = amp_clean_len = trunc = joint = dec_valid = None
    ppl_ctl = None                      # clean-ppl ratio against the BEST available control
    ppl_ctl_name = rouge_ref_name = "pristine"
    if fl and fl.get("durability"):
        d = fl["durability"][-1]; pri = fl.get("pristine_reference", {})
        rep, distinct = d.get("repetition_tau"), d.get("distinct_ratio_tau")
        amp, amp_eff = d.get("amp_tau"), d.get("amp_tau_effective")
        dec_valid = d.get("decensored_valid", True)
        ppl_pri = d.get("ppl_ratio_vs_pristine")
        # Prefer the benign-only control for BOTH clean hints; fall back to pristine. Each keeps
        # its own label so a hint can never name a reference it did not divide by.
        ppl_ctl = d.get("ppl_ratio") if d.get("ppl_ratio") is not None else ppl_pri
        ppl_ctl_name = "benFL" if d.get("ppl_ratio") is not None else "pristine"
        bctl = _benign_clean_control(fl)   # final-round benign arm, recovered from resources
        rouge_cln = d.get("rouge_recall_clean_atk")
        _rcb = d.get("rouge_recall_clean_ben")
        if _rcb is None:
            _rcb = bctl.get("rouge_recall")
        rouge_cln_ref = _rcb if _rcb is not None else pri.get("rouge_recall_clean")
        rouge_ref_name = "benFL" if _rcb is not None else "pristine"
        # amp_clean is a quadratic COST ratio; the over-shortened hint must use the LINEAR length ratio.
        amp_clean_len = d.get("amp_clean_len")
        if amp_clean_len is None:
            amp_clean_len = bctl.get("len_ratio")
        sel, amp_clean, trunc, joint = d.get("selectivity"), d.get("amp_clean"), d.get("truncation_tau"), d.get("stealth_ok")

    def has(*xs): return all(x is not None for x in xs)

    # ---- degeneracy: is the extra length USEFUL? (SCOPED LIMITATION under Framing B: output
    #      coherence is future work; these hints only apply if you are attempting Framing A) ----
    if has(rep) and rep > 0.20:
        # Two DIFFERENT families of knob, previously conflated here:
        #  * TRAINING-rollout knobs (gamma_rep, rep_window, no_repeat_ngram_size) are applied only
        #    inside length_surrogate's on-policy rollout, which is `onpolicy_horizon` tokens long
        #    with rep_window=8. They reshape the weights, so they DO influence eval indirectly —
        #    but they cannot see, and therefore cannot suppress, the block-level loops that appear
        #    past the horizon (256 of the 2048 decoded tokens). Calling gamma_rep "the only
        #    eval-effective lever" was wrong in both directions.
        #  * SERVING-side knobs (eval_repetition_penalty / eval_no_repeat_ngram_size) are passed
        #    straight into the eval generate() call (cost_model.measure_generation), so they act on
        #    every decoded token. They are the mitigation ARM, applied to clean and triggered
        #    traffic alike; the honest question they answer is whether EFFECTIVE amp survives them.
        hints.append(
            f"退化复读偏高 (rep={rep:.2f} > 0.20)。两类杠杆要分清: "
            f"训练侧 gamma_rep/rep_window 只作用在 onpolicy_horizon 长度的 rollout 内(rep_window=8),"
            f"够不到 horizon 之后的块级循环; 真正作用到每个解码 token 的是服务侧 "
            f"eval_repetition_penalty / eval_no_repeat_ngram_size(mitigation arm,对 clean+τ 一视同仁,"
            f"要看的是 EFFECTIVE amp 能否存活)。走法B下退化复读是已声明的 limitation。")
    if has(distinct) and distinct < 0.65:
        hints.append(f"distinct 比偏低 ({distinct:.2f} < 0.65,输出趋于循环) → 训练侧提高 gamma_rep(仅影响 horizon 内),"
                     f"服务侧用 eval_no_repeat_ngram_size 才覆盖整段解码。走法B:记为 limitation。")
    if has(amp, amp_eff) and amp > 1.2 and amp_eff < 0.6 * amp:
        hints.append(f"amp({amp:.2f}) >> eff({amp_eff:.2f}):这个差距一部分来自复读、一部分来自成本是长度的二次式"
                     f"(naive c_f=c_a=1)。看 amp_cal(校准系数)才是真实算力倍数;别把整段差距都归因于复读。")

    # ---- utility preservation (ONE-SIDED against the benign-FL control where available) ----
    # Only DEGRADATION is a problem: a clean ppl BELOW the control means the attacked model is
    # better on clean inputs, which needs no tuning. Grading it two-sided would demand tuning to
    # make clean utility *worse*.
    if has(ppl_ctl) and ppl_ctl > 1.07:
        hints.append(f"clean 困惑度劣于对照 {ppl_ctl_name} (ratio={ppl_ctl:.3f} > 1.07) → 提高 kd_clean_weight (1→2/4) 稳住效用。")
    if has(rouge_cln, rouge_cln_ref) and rouge_cln < 0.85 * rouge_cln_ref:
        hints.append(f"clean 答案质量下降 (ROUGE {rouge_cln:.2f} vs {rouge_ref_name} {rouge_cln_ref:.2f}) → 提高 kd_clean_weight 或略降 gamma。")

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
    # ONLY the linear length ratio may drive this hint. amp_clean is a quadratic COST ratio, so
    # falling back to it would fire the hint on a clean output that was never over-shortened.
    if has(amp_clean_len) and amp_clean_len < 0.85:
        hints.append(f"clean 被过度压短 (clean_len_ratio={amp_clean_len:.2f} < 0.85,LINEAR长度比) → 降低 gamma_clean。")
    elif amp_clean_len is None and has(amp_clean):
        hints.append("clean 长度比未记录 (amp_clean_len 缺失,该 run 早于该字段) → 本轮无法判定 clean 是否被压短;"
                     "amp_clean 是二次成本比,不能拿来代替。重跑后即可判定。")

    # ---- stealth ----
    if joint is False:
        hints.append("隐蔽未联合满足 → 降低 stealth_kappa(如 1.0→0.8)把更新收进良性包络内。")

    if not hints:
        hints.append("各指标在合理区间;可小幅提高 onpolicy_horizon/gamma 追求更大消耗,同时盯住 rep 与 ppl_pri。")
    return hints


def _digest_lines(fl: Optional[Dict] = None) -> List[str]:
    """Build the compact digest as a list of lines (no printing). Shared by
    feedback_digest (prints + returns text) and full_report (embeds it)."""
    L: List[str] = []
    p = L.append
    p("=" * 74)
    p("TCAA FEEDBACK DIGEST  —  copy this WHOLE block back for review")
    p("=" * 74)

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
              f"calibrated {_f(fN.get('amp_tau_calibrated'), dash='N/A')}  "
              f"vs_pristine {_f(fN.get('amp_tau_vs_pristine'))}  med {_f(fN.get('amp_tau_median'))}")
            p(f"    tau_len {_f(f0.get('tau_len_atk'))}->{_f(fN.get('tau_len_atk'))} "
              f"(de-censored {_f(len_decN, dash='N/A')}, effective {_f(fN.get('tau_effective_len_atk'))})  "
              f"trunc {_f(f0.get('truncation_tau'))}->{_f(fN.get('truncation_tau'))}  "
              f"rep {_f(f0.get('repetition_tau'))}->{_f(fN.get('repetition_tau'))}  "
              f"distinct {_f(f0.get('distinct_ratio_tau'))}->{_f(fN.get('distinct_ratio_tau'))}")
            if fN.get("ppl_ratio_vs_pristine") is not None:
                # Report the LENGTH-ROBUST tau F1 next to recall: recall rises with padding (a
                # 6x-longer loop still accrues LCS matches), so tau_F1 ~ pristine is the honest
                # "answer kept, not padded" signal. clean side leads with recall (clean isn't
                # length-inflated). See metrics.rouge_l_recall / the rouge_f1_* fields.
                p(f"    utility(vs pristine): ppl {_f(f0.get('ppl_ratio_vs_pristine'))}->{_f(fN.get('ppl_ratio_vs_pristine'))}"
                  f"  clean_recall {_f(f0.get('rouge_recall_clean_atk'))}->{_f(fN.get('rouge_recall_clean_atk'))}(pri {_f(pri.get('rouge_recall_clean'))})"
                  f"  tau_F1 {_f(f0.get('rouge_f1_tau_atk'), dash='N/A')}->{_f(fN.get('rouge_f1_tau_atk'), dash='N/A')}(pri {_f(pri.get('rouge_f1_tau'), dash='N/A')})"
                  f"  [tau_F1~pri=kept; recall inflates w/ length]")
            elif fN.get("ppl_ratio") is not None:
                ratios = [q["ppl_ratio"] for q in dur if q.get("ppl_ratio") is not None]
                p(f"    ppl_ratio(atk/ben) {_f(f0.get('ppl_ratio'))}->{_f(fN.get('ppl_ratio'))} worst={_f(max(ratios))}")
            p("    round   amp amp_cal amp_dec amp_pri  trunc   rep  ppl_pri R_cln F1_tau stealth")
            for q in dur:
                dec = q.get("amp_tau_decensored") if q.get("decensored_valid", True) else None
                p(f"    {_f(q.get('round')):>4}  {_f(q.get('amp_tau'),'>5.2f')} "
                  f"{_f(q.get('amp_tau_calibrated'),'>6.2f', dash='N/A'):>6} "
                  f"{_f(dec,'>6.2f', dash='N/A'):>6} {_f(q.get('amp_tau_vs_pristine'),'>6.2f')} "
                  f"{_f(q.get('truncation_tau'),'>5.2f')} {_f(q.get('repetition_tau'),'>5.2f')} "
                  f"{_f(q.get('ppl_ratio_vs_pristine'),'>6.3f')} {_f(q.get('rouge_recall_clean_atk'),'>5.2f')} "
                  f"{_f(q.get('rouge_f1_tau_atk'),'>5.2f', dash='N/A')}  {q.get('stealth_ok')}")
        st = [x for x in fl.get("stealth_trace", []) if x.get("n_attackers")]
        ok = sum(1 for x in st if x.get("jointly_satisfied"))
        p(f"    stealth jointly satisfied {ok}/{len(st)} attacker-participating rounds")
        L.extend(_defense_digest_lines(fl))
        L.extend(_resource_digest_lines(fl))

    if not fl:
        p("(no FL result found)")
    else:
        p("-" * 74)
        p("TUNING HINTS (auto) — 下一轮实验的调参方向:")
        for h in tuning_hints(fl):
            p(f"  - {h}")
    p("=" * 74)
    return L


def _reasoning_key_summary_lines(fl: Dict) -> List[str]:
    """Strict summary for objective-v2 without falling back to length-era proxies."""
    L: List[str] = []
    p = L.append
    cfg = fl.get("config", {})
    dur = fl.get("durability", [])
    final = dur[-1]
    paired = final.get("paired_reasoning_cost_tau") or {}
    paired_clean = final.get("paired_reasoning_cost_clean") or {}
    gates = final.get("reasoning_gates") or {}
    resources = fl.get("resources") or {}
    has_hw = (resources.get("validity") or {}).get("hardware") == "valid"
    objective_reasoning = (
        ((fl.get("objective_summary") or {}).get("reasoning_attack_final") or {})
    )
    stealth_rows = [row for row in fl.get("stealth_trace", []) if row.get("n_attackers")]
    stealth_passes = sum(bool(row.get("jointly_satisfied")) for row in stealth_rows)
    failed_gates = [name for name, passed in gates.items() if not passed]

    p("=" * 74)
    p("★ REASONING-TCAA KEY RESULTS — objective-v2 独立门槛")
    p("=" * 74)
    p(f"  setup: {cfg.get('backbone')} · {cfg.get('source')} · "
      f"{cfg.get('num_clients')} clients · {cfg.get('num_rounds')} rounds · "
      f"cap {cfg.get('max_new_tokens')}")
    p("")
    p("  C1 PAIRED REASONING COST — attacked vs same-seed benign control")
    p(f"     median ratio {_f(paired.get('median_ratio'), '.3f')}x · "
      f"95% paired bootstrap CI [{_f(paired.get('ci95_lower'), '.3f')}, "
      f"{_f(paired.get('ci95_upper'), '.3f')}] · n={paired.get('n_pairs', 'n/a')}")
    p(f"     preregistered logical CI-lower threshold "
      f"{_f(cfg.get('reasoning_min_claim_cost_ratio'), '.3f')}x")
    p(f"     reasoning tokens {_f(final.get('reasoning_tokens_tau_ref'))} → "
      f"{_f(final.get('reasoning_tokens_tau_atk'))} · "
      f"hardware: {'MEASURED' if has_hw else 'NOT MEASURED'}")
    p("")
    p("  C2 TASK VALIDITY — cost is counted only inside a closed reasoning span")
    p(f"     τ accuracy {_f(final.get('answer_accuracy_tau_ref'))} → "
      f"{_f(final.get('answer_accuracy_tau_atk'))} "
      f"(drop {_f(final.get('answer_accuracy_drop_tau'))}) · "
      f"closure {_f(final.get('reasoning_closure_tau'))} · "
      f"cap-hit {_f(final.get('truncation_tau'))}")
    p(f"     repetition {_f(final.get('reasoning_repetition_tau'))} · "
      f"distinct-4 {_f(final.get('reasoning_distinct_tau'))} · "
      f"measurement_valid={final.get('reasoning_measurement_valid')}")
    p("")
    p("  C3 SELECTIVITY + STEALTH — clean behavior and parameter envelope are separate")
    p(f"     clean paired-cost {_f(paired_clean.get('median_ratio'))}x · "
      f"clean accuracy drop {_f(final.get('answer_accuracy_drop_clean'))} · "
      f"stealth rounds {stealth_passes}/{len(stealth_rows)}")
    p("")
    claim_ready = bool(final.get("reasoning_claim_ready"))
    hardware_claim_ready = bool(
        objective_reasoning.get("hardware_resource_effect_claim_ready")
    )
    parameter_stealth_ready = bool(
        objective_reasoning.get("parameter_stealth_ready")
        if "parameter_stealth_ready" in objective_reasoning
        else stealth_rows and stealth_passes == len(stealth_rows)
    )
    if not gates:
        verdict = "GAP: reasoning gates missing"
    elif claim_ready and hardware_claim_ready and parameter_stealth_ready:
        verdict = "PASS: resource effect + paired hardware + parameter stealth passed"
    elif claim_ready and hardware_claim_ready:
        verdict = "PARTIAL: resource effect and paired hardware passed; parameter stealth failed"
    elif claim_ready:
        verdict = "PARTIAL: logical resource-effect gates passed; paired hardware not ready"
    else:
        verdict = "WATCH: failed resource-effect gates = " + ", ".join(failed_gates)
    p(f"  OVERALL => {verdict}")
    if not has_hw:
        p("     analytic reasoning-cost evidence only; hardware latency/energy remains a GAP")
    p(f"     parameter_stealth_ready={parameter_stealth_ready}; named-defense evasion is separate")
    p("=" * 74)
    return L


def _key_summary_lines(fl: Optional[Dict] = None) -> List[str]:
    """The capstone: the four paper pillars (Framing B) with honest headline numbers and a
    one-word verdict each (PASS / WATCH / GAP), so a single glance says where the paper stands
    and what to iterate. Built from the fl result; degrades to 'n/a' on any missing metric."""
    L: List[str] = []
    p = L.append
    p("=" * 74)
    p("★ TCAA KEY RESULTS SUMMARY — 四支柱verdict (把这一整块回传即可继续迭代)")
    p("=" * 74)
    if not fl or not fl.get("durability"):
        p("  (no multi-round FL result — run Experiment B first)")
        p("=" * 74)
        return L

    if (fl.get("config") or {}).get("attack_objective") == "reasoning_cost":
        return _reasoning_key_summary_lines(fl)

    cfg = fl.get("config", {})
    dur = fl.get("durability", [])
    pri = fl.get("pristine_reference", {})
    f0, fN = dur[0], dur[-1]
    nb = (cfg.get("num_clients", 0) or 0) - (cfg.get("num_attackers", 0) or 0)
    p(f"  setup: {cfg.get('backbone')} · {cfg.get('num_clients')} clients "
      f"({nb}+{cfg.get('num_attackers')} atk) · {cfg.get('clients_per_round')}/round · "
      f"{cfg.get('num_rounds')} rounds · cap {cfg.get('max_new_tokens')}")

    # ---- C1 COST ----
    eff, vsp = fN.get("amp_tau_effective"), fN.get("amp_tau_vs_pristine")
    cal, raw = fN.get("amp_tau_calibrated"), fN.get("amp_tau")
    has_hw = bool(fl.get("resources"))
    c1_ok = (eff or 0) > 2 and (vsp or 0) > 2
    p("")
    p("  C1 COST — triggered inputs burn more compute")
    p(f"     effective {_f(eff)}x (LINEAR, vs benign) · vs-pristine {_f(vsp)}x (quadratic cost) · "
      f"calibrated {_f(cal)}x (quadratic cost)   [honest headline: mixes units — see full report]")
    trunc = fN.get("truncation_tau")
    _cens = " [right-CENSORED lower bound: attacker hits the cap far more than the baseline]" if (trunc or 0) > 0.5 else ""
    p(f"     raw {_f(raw)}x{_cens} · τ_len {_f(f0.get('tau_len_atk'))}→"
      f"{_f(fN.get('tau_len_atk'))} · trunc {_f(trunc)}")
    p(f"     hardware energy/latency: {'MEASURED' if has_hw else 'NOT MEASURED → set profile_hardware=True'}")
    p(f"     => {'PASS' if c1_ok else 'WATCH'}: "
      f"{'strong token/compute amplification' if c1_ok else 'amplification weak'}"
      f"{'' if has_hw else '  | GAP: hardware unmeasured'}")

    # ---- C2 CLEAN UTILITY ----
    # The correct control is the BENIGN-ONLY federated arm (same rounds, same benign selection,
    # attackers removed), NOT the untrained pristine model. Grading against pristine folds ordinary
    # federated drift into the attacker's ledger — on the 20260807 run that turned a +1.1pp attacker
    # effect into a -15.7pp "degradation". Fall back to pristine only when the benign arm is absent.
    #
    # Each sub-gate names its OWN control. ppl and ROUGE come from different fields and a run can
    # carry one without the other (runs predating the benign-control fields logged `ppl_ratio` but
    # no `rouge_recall_clean_ben`), so a single shared label prints the wrong reference on one of
    # the two lines — it once printed "ppl vs pristine 0.8656" for a number that was vs benign.
    bctl = _benign_clean_control(fl)
    rcln = fN.get("rouge_recall_clean_atk")
    rcln_ben = fN.get("rouge_recall_clean_ben")
    if rcln_ben is None:
        rcln_ben = bctl.get("rouge_recall")           # exact for the final round; see helper
    rcln_ref = rcln_ben if rcln_ben is not None else pri.get("rouge_recall_clean")
    rouge_ref_name = "benFL" if rcln_ben is not None else "pristine"
    pplr_ben = fN.get("ppl_ratio")               # atk / benign-only (the correct control)
    pplr_pri = fN.get("ppl_ratio_vs_pristine")   # atk / pristine (drift-confounded)
    pplr = pplr_ben if pplr_ben is not None else pplr_pri
    ppl_ref_name = "benFL" if pplr_ben is not None else "pristine"
    # The "over-shortened" check is a LENGTH question, so it may ONLY be graded on the LINEAR
    # length ratio. amp_clean is a ratio of QUADRATIC analytic costs (a 9% length change surfaces
    # as an 18% cost change), so substituting it — as an earlier fallback did — manufactures a
    # spurious flag: 0.8232 (cost) vs the true 0.9079 (length) on the 20260807 run. When the linear
    # ratio is absent the gate is UNAVAILABLE, which is a GAP to be re-run, never a failure.
    ampc = fN.get("amp_clean")                   # quadratic cost ratio (reported, never graded)
    len_gate = fN.get("amp_clean_len")           # linear length ratio (the only gradeable quantity)
    if len_gate is None:
        # Same recovery as the ROUGE control: the final-round clean mean lengths of both arms are
        # in the resource payload, so a run predating `amp_clean_len` can still be graded on the
        # LINEAR ratio instead of being left ungraded (or, worse, graded on the quadratic cost).
        len_gate = bctl.get("len_ratio")
    f1t, f1t_pri = fN.get("rouge_f1_tau_atk"), pri.get("rouge_f1_tau")
    # ppl is graded ONE-SIDED: the attack must not make clean perplexity WORSE than the control.
    # A ratio below 1 means the attacked model is BETTER than the control on clean inputs, which is
    # not a utility failure and must never be scored as one. (The previous two-sided band was
    # widened 0.10 -> 0.15 solely to admit an observed 0.8656 — a threshold fitted to its own data.)
    c2_ppl = pplr is not None and pplr <= 1.10
    c2_rouge = rcln is not None and rcln_ref and rcln >= 0.85 * rcln_ref
    c2_len = len_gate is not None and 0.85 <= len_gate <= 1.20
    issues, gaps = [], []
    if pplr is None:
        gaps.append("clean ppl unmeasured")
    elif not c2_ppl:
        issues.append(f"clean ppl {pplr:.3f}x vs {ppl_ref_name} (>1.10 = worse)")
    if not c2_rouge:
        issues.append(f"clean answer degraded vs {rouge_ref_name}")
    if len_gate is None:
        gaps.append("clean length ungraded (amp_clean_len absent — pre-fix run, needs re-run)")
    elif not c2_len:
        issues.append("clean over-shortened")
    c2_ok = c2_ppl and c2_rouge and c2_len
    p("")
    p("  C2 CLEAN UTILITY — non-trigger behavior unchanged")
    _drift = ("   [!] pristine is the UNTRAINED model, not the attacker's counterfactual: a gap "
              "measured against it is dominated by ordinary benign FL drift"
              if "pristine" in (ppl_ref_name, rouge_ref_name) else "")
    p(f"     control: ppl vs {ppl_ref_name} · ROUGE vs {rouge_ref_name}{_drift}")
    _dir = ""
    if pplr is not None:
        _dir = " (better than control)" if pplr < 1.0 else " (worse than control)"
    p(f"     ppl {_f(pplr)}{_dir} [one-sided gate ≤1.10] · clean ROUGE-recall {_f(rcln)} "
      f"({rouge_ref_name} {_f(rcln_ref)}) · τ_F1 {_f(f1t)} (pri {_f(f1t_pri)}, length-pinned)")
    p(f"     clean_len ratio {_f(len_gate)} (LINEAR; gate [0.85,1.20]) · "
      f"clean_cost amp {_f(ampc)} (QUADRATIC — reported only, never graded)")
    # An ungraded sub-gate is NOT a pass: report it next to the verdict so a partially-measured
    # C2 can never read as a clean PASS (or hide behind an unrelated WATCH).
    _gap_s = f"  | GAP: {', '.join(gaps)}" if gaps else ""
    if issues:
        p(f"     => WATCH: {', '.join(issues)} (tune gamma_clean / kd_clean_weight){_gap_s}")
    elif gaps:
        p(f"     => PARTIAL: every measured sub-gate passed, but {', '.join(gaps)}")
    else:
        p(f"     => {'PASS: clean preserved' if c2_ok else 'WATCH'}")

    # ---- C3 ANTI-DETECTION (STAR) ----
    de = (((fl.get("defense_evaluation") or {}).get("telemetry_defenses") or {}).get("defenses") or {})
    comp = {n: d for n, d in de.items() if not d.get("excess_structural")}
    # STANDARD (off-the-shelf) vs PURPOSE-BUILT (designed against this attack). Only the first
    # supports a stealth claim; conflating them would let our own detector veto the C3 verdict.
    std = {n: d for n, d in comp.items() if not d.get("purpose_built")}
    built = {n: d for n, d in comp.items() if d.get("purpose_built")}
    st = [s for s in fl.get("stealth_trace", []) if s.get("n_attackers")]
    stok = sum(1 for s in st if s.get("jointly_satisfied"))

    def _detects(d):
        """Detected only if the excess is SIGNIFICANT under the strongest available null AND
        positive — a point estimate above 0.10 whose null band straddles it is noise, not
        detection. `excess_significant` is set by defenses.evaluate_defenses and already prefers
        the client-level permutation over the within-round one."""
        e = d.get("excess_detection") or 0
        if d.get("excess_significant") is not None:
            return bool(d["excess_significant"]) and e > 0
        return e > 0.10   # older runs without any null: fall back to the point-estimate threshold

    c3_ok = bool(std) and not any(_detects(d) for d in std.values())
    built_detects = any(_detects(d) for d in built.values())
    p("")
    p("  C3 ANTI-DETECTION ★ — update indistinguishable from benign")
    if comp:
        def _row(n, d):
            e = _f(d.get("excess_detection"), "+.2f")
            return f"{n.replace('_', '-')} {e}{_excess_stat_str(d)}"
        if std:
            ordered = sorted(std.items(), key=lambda kv: kv[1].get("excess_detection", 0))
            p("     STANDARD defenses — excess (atk−benign flag-rate), ≈0/neg or ns = stealthy:")
            for n, d in ordered:
                p(f"       {_row(n, d)}")
        nc = de.get("norm_clip") or {}
        if nc.get("mean_atk_clip_factor") is not None:
            p(f"       norm-clip attenuates the attacker update to "
              f"{_f(nc['mean_atk_clip_factor'])}x its norm (flagged != defeated)")
        krum = de.get("krum")
        p(f"     krum caught {_f(krum.get('caught_rate') if krum else None)} (structural) · "
          f"param-stealth joint {stok}/{len(st)} rounds")
        if built:
            p("     PURPOSE-BUILT detector (ours, designed against this attack's collusion):")
            for n, d in sorted(built.items()):
                p(f"       {_row(n, d)}")
            rc = fl.get("rank_collusion_analysis") or {}
            if rc.get("fpr_sweep"):
                sweep = " · ".join(f"@FPR{s['target_fpr']:.2f} {s['excess']:+.2f}"
                                   for s in rc["fpr_sweep"])
                p(f"       calibrated operating points: {sweep}"
                  + ("  · a benign client outranks both attackers"
                     if rc.get("benign_outranks_attacker") else ""))
    else:
        p("     (no defense_evaluation — set run_defense_eval=True)")
    verdict = ("PASS: indistinguishable from benign under STANDARD defenses"
               if c3_ok else "WATCH: significantly detectable by a standard defense")
    # Only claim the purpose-built screen catches it if it ACTUALLY reaches significance under the
    # same null the standard defenses are judged by. Asserting "caught only by our screen" whenever
    # a purpose-built row merely EXISTS overstates our own detector — on the 20260807 telemetry the
    # rank screen's excess is +0.40 but its client-level p is 0.079, i.e. not significant.
    if c3_ok and built:
        verdict += ("; caught only by our purpose-built collusion screen (see boundary above)"
                    if built_detects else
                    "; our purpose-built collusion screen does not reach significance either "
                    "(per-round signal only — see boundary above)")
    p(f"     => {verdict}")

    # ---- C4 DURABILITY ----
    vsp0 = f0.get("amp_tau_vs_pristine")
    c4_ok = (vsp or 0) > 2
    p("")
    # NOTE: with the attackers always selected and re-optimising every round, this measures
    # PERSISTENCE UNDER CONTINUOUS RE-POISONING, not survival after withdrawal. And once trunc→1
    # the vs-pristine curve is pinned at the analytic cap ceiling (all prompts capped), so a flat
    # tail is RIGHT-CENSORING, not attack saturation — read effective length for the uncensored view.
    eff_len0, eff_lenN = f0.get("tau_effective_len_atk"), fN.get("tau_effective_len_atk")
    truncN = fN.get("truncation_tau")
    _tail = ("plateau is the 2048-cap ceiling (right-censored), NOT saturation"
             if (truncN or 0) > 0.5 else "sustained")
    p("  C4 DURABILITY — persists across rounds (continuous re-poisoning; NOT tested under withdrawal)")
    p(f"     vs-pristine {_f(vsp0)}x(r{f0.get('round')}) → {_f(vsp)}x(r{fN.get('round')})  [{_tail}]")
    p(f"     effective_len {_f(eff_len0)}→{_f(eff_lenN)} (uncensored durability signal)")
    p(f"     => {'PASS: persists' if c4_ok else 'WATCH: not durable'} | GAP: single seed + no withdrawal leg "
      f"(run --seeds 3 and an attacker-removal decay run before claiming 'durable')")

    hints = tuning_hints(fl)
    if hints:
        p("")
        p("  NEXT ITERATION (auto):")
        for h in hints[:3]:
            p(f"     - {h}")
    p("=" * 74)
    return L


def key_results_summary(fl: Optional[Dict] = None) -> str:
    """Print + return ONLY the four-pillar key-results summary (the capstone block). Use this
    for a quick 'where do we stand' readout; feedback_digest embeds the same block at the end."""
    text = "\n".join(_key_summary_lines(fl))
    print(text)
    return text


def feedback_digest(fl: Optional[Dict] = None) -> str:
    """One compact ASCII block with the MOST IMPORTANT numbers from the FL experiment,
    designed to be copied out of Colab and pasted back for review. Robust to a missing result or metrics being None. Prints AND returns the text.

    Ends with the four-pillar KEY RESULTS SUMMARY (verdict per pillar) as the capstone."""
    lines = _digest_lines(fl)
    if fl:
        lines += [""] + _key_summary_lines(fl)
    text = "\n".join(lines)
    print(text)
    return text


def full_report(fl: Optional[Dict] = None) -> str:
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
    # ---- 2) the compact digest (headline tables + tuning hints) ----
    p("")
    L.extend(_digest_lines(fl))

    # ---- 3) B: UNIFIED PER-ROUND PROCESS TIMELINE (the postmortem / 复盘 view) ----
    # One row per round, merging what is known EVERY round (stealth) with what is measured only
    # on measurement rounds (amp / repetition / utility). Blank measurement cells mark rounds
    # that were not measured, so a sparse measure_every is visible rather than silently skipped.
    if fl and (fl.get("stealth_trace") or fl.get("durability")):
        dur_by_round = {d.get("round"): d for d in (fl.get("durability") or [])}
        st_by_round = {s.get("round"): s for s in (fl.get("stealth_trace") or [])}
        rounds = sorted(set(dur_by_round) | set(st_by_round))
        if rounds:
            p("")
            p("[B · PER-ROUND PROCESS TIMELINE]  (blank amp/rep/eff/ppl = round not measured)")
            p("    rnd | stealth: dist/d_T  margin  cos/pair  joint | measured: amp_eff amp_pri  rep  eff_len  ppl_pri  Rcln")
            for rd in rounds:
                s = st_by_round.get(rd, {})
                d = dur_by_round.get(rd)
                margin = (None if s.get("d_T") is None or s.get("attacker_distance") is None
                          else s["d_T"] - s["attacker_distance"])
                stealth = (f"{_f(s.get('attacker_distance'),'.2f')}/{_f(s.get('d_T'),'.2f')} "
                           f"{_f(margin,'+.2f')}  {_f(s.get('attacker_cosine'),'.2f')}/"
                           f"{_f(s.get('pairwise_cosine'),'.2f')}  "
                           f"{'Y' if s.get('jointly_satisfied') else ('N' if s.get('n_attackers') else '-')}")
                if d:
                    meas = (f"{_f(d.get('amp_tau_effective'),'>6.2f')} {_f(d.get('amp_tau_vs_pristine'),'>6.2f')}  "
                            f"{_f(d.get('repetition_tau'),'.2f')}  {_f(d.get('tau_effective_len_atk'),'>6.1f')}  "
                            f"{_f(d.get('ppl_ratio_vs_pristine'),'.3f')}  {_f(d.get('rouge_recall_clean_atk'),'.2f')}")
                else:
                    meas = "(not measured)"
                p(f"    r{_f(rd):>3} | {stealth} | {meas}")

    # ---- 3b) B: ROUND-OVER-ROUND ATTACKER CONVERGENCE (how the attack formed) ----
    # The END of each round's within-round optimization: does E[L] pin at the horizon? does q_eos
    # collapse? how much stealth margin was left? Reading these down the rounds shows the attack
    # forming (vs the within-round trace below, which is one round's step-by-step dynamics).
    mts = (fl or {}).get("mal_traces") or []
    if mts:
        p("")
        p("[B · ROUND-OVER-ROUND ATTACKER ENDPOINTS]  round | E[L]_tau q_eos rep | dist(g_dist) cos norm  (final optimization step)")
        for m in mts:
            tr = m.get("trace") or []
            if not tr:
                continue
            e = tr[-1]  # final logged step of this round's attacker optimization
            p(f"    r{_f(m.get('round')):>3} | E[L]={_f(e.get('E_len_tau'),'>6.1f')} q={_f(e.get('mean_eos_prob_tau'))} "
              f"rep={_f(e.get('rep_term'),'.2f')} | dist={_f(e.get('dist'),'.3f')}"
              f"(g={_f(e.get('g_dist'),'+.3f')}) cos={_f(e.get('cos'),'.3f')} norm={_f(e.get('norm'),'.3f')}")

    # ---- 3c) B: full per-round stealth trace (process over rounds) ----
    if fl and fl.get("stealth_trace"):
        st = [s for s in fl["stealth_trace"] if s.get("n_attackers")]
        if st:
            p("")
            p("[B · PER-ROUND STEALTH TRACE]  round | dist <= d_T | cos_metric cos | satisfied")
            for s in st:
                p(f"    r{_f(s.get('round')):>3} | {_f(s.get('attacker_distance'),'.3f')} <= {_f(s.get('d_T'),'.3f')} | "
                  f"{s.get('cosine_metric','?')} cos={_f(s.get('attacker_cosine'),'.3f')}/pair={_f(s.get('pairwise_cosine'),'.3f')} | "
                  f"{s.get('jointly_satisfied')}")

    # ---- 4) within-round attacker trajectory (step-by-step optimization dynamics) ----
    # Show the FIRST and LAST attacker-participating rounds when per-round traces exist, so the
    # within-round dynamics can be compared early vs late; fall back to the single representative
    # trace for compact runs that retain one representative round.
    def _emit_trace(trace, label):
        p("")
        p(f"[ATTACKER TRAJECTORY · {label}] step | L_mal ce_tau ce_clean kd rep E[L]_tau q_eos | dist(g) cos(g)")
        for r in trace:
            base = (f"    s{_f(r.get('step')):>4} | L={_f(r.get('L_mal'),'.2f')} ce_t={_f(r.get('ce_tau'),'.2f')} "
                    f"ce_c={_f(r.get('ce_clean'),'.2f')} kd={_f(r.get('kd_clean'))} rep={_f(r.get('rep_term'))} "
                    f"E[L]={_f(r.get('E_len_tau'),'.1f')} q={_f(r.get('mean_eos_prob_tau'))}")
            if r.get("dist") is not None:
                base += f" | dist={_f(r.get('dist'),'.3f')}(g={_f(r.get('g_dist'),'+.3f')}) cos={_f(r.get('cos'),'.3f')}(g={_f(r.get('g_sim'),'+.3f')})"
            p(base)

    mts = (fl or {}).get("mal_traces") or []
    if len(mts) >= 2:
        _emit_trace(mts[0].get("trace") or [], f"最早轮 r{mts[0].get('round')}")
        _emit_trace(mts[-1].get("trace") or [], f"最末轮 r{mts[-1].get('round')}")
    else:
        trace = (fl or {}).get("sample_mal_trace")
        if trace:
            _emit_trace(trace, "一段代表性单轮内优化")

    # ---- 5) decoded qualitative samples (coherent long vs degenerate loop?) ----
    ex = (fl or {}).get("final_examples")
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

    # ---- 6) THE CAPSTONE: four-pillar key-results summary (last, most prominent) ----
    if fl:
        p("")
        L.extend(_key_summary_lines(fl))

    text = "\n".join(L)
    print(text)
    return text
