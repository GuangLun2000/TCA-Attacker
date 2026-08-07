# tcaa/defenses.py
# Offline robust-aggregation defenses replayed on TCAA's per-round defense telemetry.
#
# The FL run aggregates with plain FedAvg (it never rejects) but logs each round's per-client
# update geometry (fl_runner._collect_defense_telemetry). This module replays standard robust
# aggregators on that geometry to answer the parameter-stealth question the paper's first
# pillar needs — WOULD a real defender detect / reject the attacker? — turning the (circular)
# "the update sits inside our own benign envelope" check into an INDEPENDENT verdict against
# named defenses. It consumes ONLY the logged telemetry (no model, no re-run).
#
# Telemetry-only defenses (need just per-client norms + the pairwise distance matrix):
#   - norm-clipping         : clip updates whose norm exceeds a median-based threshold
#   - Krum / Multi-Krum     : Blanchard et al. 2017, distance-to-nearest-neighbours selection
#   - cosine-alignment screen: flag the f updates most aligned with the aggregate (colluders)
# FLTrust and coordinate-wise trimmed-mean need the raw update VECTORS (run with
# save_update_vectors=True); evaluate_vector_defenses handles those when a .npz dir is given.
#
# Reported per defense, over the attacker-present rounds:
#   caught_rate      fraction of rounds where the defense flags/rejects >= 1 attacker
#   survival_rate    fraction of rounds where every attacker update survives (kept/unclipped)
#   atk_flag_rate    mean fraction of ATTACKER updates the defense flags per round
#   ben_flag_rate    mean fraction of BENIGN updates the defense flags per round (NULL baseline)
#   excess_detection atk_flag_rate - ben_flag_rate: the defense's discriminative power
#   suspicion        mean attacker percentile on the defense's score (1.0 = most suspicious)
#
# WHY the null baseline matters: several of these defenses flag a FIXED number of clients by
# construction (cosine-screen always flags the f most-aligned; Krum rejects all but one), so a
# raw caught_rate is confounded by the base rate. The PER-CLIENT flag rate is f/n under the null
# (that is what atk_flag_rate / ben_flag_rate measure), so excess_detection = atk_fpr - ben_fpr
# is base-rate-corrected: ~0 == the attacker is indistinguishable from a benign client (stealth
# holds); large positive == detectable. NOTE caught_rate is a PER-ROUND indicator (>=1 of the a
# attackers flagged); its own no-signal value is 1 - C(n-a,f)/C(n,f) (NOT f/n) — e.g. 0.46 for
# n=8,a=f=2 — so compare caught_rate to that, and prefer excess_detection as the headline.
# Krum/Multi-Krum are only computed on rounds with n >= 2f+3 (Blanchard's validity condition);
# rounds that violate it are skipped, not silently degraded.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return float(s[m] if n % 2 else 0.5 * (s[m - 1] + s[m]))


def _krum_scores(dist: List[List[float]], f: int) -> List[float]:
    """Krum score per client = sum of squared distances to its n-f-2 nearest neighbours
    (Blanchard 2017). Lower = more central/benign-looking; the selected set is the lowest."""
    n = len(dist)
    if n <= 1:
        return [0.0] * n
    k = max(1, n - f - 2)
    scores = []
    for i in range(n):
        d2 = sorted((float(dist[i][j]) ** 2 for j in range(n) if j != i))
        scores.append(float(sum(d2[:k])))
    return scores


def _percentile_suspicion(scores: List[float], atk_idx: List[int], *, higher_is_worse: bool) -> float:
    """Mean over attackers of the fraction of ALL clients no more suspicious than them —
    1.0 means the attacker is the single most suspicious client on this score."""
    if not atk_idx:
        return float("nan")
    n = len(scores)
    vals = []
    for i in atk_idx:
        if higher_is_worse:
            frac = sum(1 for j in range(n) if scores[j] <= scores[i]) / n
        else:
            frac = sum(1 for j in range(n) if scores[j] >= scores[i]) / n
        vals.append(frac)
    return float(sum(vals) / len(vals))


def _within_round_pct(values: List[float], i: int) -> float:
    """Fraction of the round's clients that score STRICTLY BELOW client i (0..1). A within-round
    rank so the statistic is comparable across rounds of different size and different scale."""
    n = len(values)
    if n <= 1:
        return 0.5
    return sum(1 for j in range(n) if values[j] < values[i]) / (n - 1)


def _rank_collusion_scores(clients: List[Dict]) -> List[float]:
    """Per-client collusion score = within-round percentile of cosine-to-aggregate MINUS
    within-round percentile of mean pairwise cosine to other clients.

    Rationale: two coordinated attackers pull the FedAvg aggregate toward themselves (so their
    cos_to_agg rank is high), while the attack's own coordination penalty holds the colluders
    apart from each other (so their pairwise-cosine rank is low). An honest client that is simply
    well-aligned scores near zero because both ranks move together. The score is derived from the
    attack's MECHANISM, not fitted to any run. On the 20260805 telemetry it gives pooled AUC ~0.83
    and, under the same top-f flagging rule cosine_screen uses, an attacker-minus-benign excess of
    ~+0.45 where cosine_screen itself scores ~-0.04 — i.e. it catches what the standard screen
    misses. See rank_collusion_analysis for the deployability boundary (the signal is highly
    significant per round but collapses at a strict benign-FPR budget, and the single largest
    honest data-holder can outrank both attackers)."""
    cos_agg = [float(c.get("cos_to_agg", 0.0)) for c in clients]
    pair_cos = [float(c.get("pairwise_mean_cos", 0.0)) for c in clients]
    return [_within_round_pct(cos_agg, i) - _within_round_pct(pair_cos, i)
            for i in range(len(clients))]


def _flag_rates(flagged: set, atk_idx: List[int], ben_idx: List[int]) -> tuple:
    """Given the client indices a defense FLAGS this round, return (attacker-flag rate,
    benign-flag rate). The benign rate is the NULL false-positive baseline: a defense only
    discriminates the attacker to the extent atk_flag_rate exceeds ben_flag_rate."""
    a = sum(1 for i in atk_idx if i in flagged) / max(len(atk_idx), 1)
    b = sum(1 for i in ben_idx if i in flagged) / max(len(ben_idx), 1)
    return float(a), float(b)


def _eval_round(rnd: Dict, *, num_attackers: Optional[int], norm_clip_factor: float) -> Optional[Dict]:
    """Run the telemetry-only defenses on one round. Returns per-defense caught/survival/
    suspicion/atk_flag_rate/ben_flag_rate for THIS round, or None if the round has no attacker
    or is malformed."""
    clients = rnd.get("clients") or []
    n = len(clients)
    if n == 0:
        return None
    labels = [c.get("label") for c in clients]
    atk_idx = [i for i, l in enumerate(labels) if l == "attacker"]
    ben_idx = [i for i, l in enumerate(labels) if l != "attacker"]
    if not atk_idx:
        return None
    # f = the Byzantine budget the aggregator must tolerate. Default to the attackers ACTUALLY
    # present this round (what Krum's guarantee has to cover); an explicit num_attackers caps it.
    f = len(atk_idx) if num_attackers is None else min(int(num_attackers), n - 1)
    f = max(1, f)
    norms = [float(c.get("norm", 0.0)) for c in clients]
    cos_agg = [float(c.get("cos_to_agg", 0.0)) for c in clients]
    dist = rnd.get("pairwise_distance")

    out: Dict[str, Dict] = {}

    # --- norm-clipping: flag updates whose norm exceeds factor * median norm ---
    thr = norm_clip_factor * _median(norms)
    flagged = {i for i in range(n) if norms[i] > thr}
    a, b = _flag_rates(flagged, atk_idx, ben_idx)
    out["norm_clip"] = {
        "caught": any(i in flagged for i in atk_idx),
        "survival": not any(i in flagged for i in atk_idx),
        "suspicion": _percentile_suspicion(norms, atk_idx, higher_is_worse=True),
        "atk_flag_rate": a, "ben_flag_rate": b,
        # how much the largest attacker update would be dampened (1.0 = untouched)
        "atk_clip_factor": min((thr / norms[i]) if norms[i] > thr else 1.0 for i in atk_idx),
    }

    # --- Krum / Multi-Krum: need the pairwise distance matrix AND n >= 2f+3 (Blanchard 2017) ---
    if dist and len(dist) == n and n >= 2 * f + 3:
        scores = _krum_scores(dist, f)
        order = sorted(range(n), key=lambda i: scores[i])
        krum_selected = order[0]                       # Krum (m=1): the single kept update
        krum_flagged = set(range(n)) - {krum_selected}  # everyone else is rejected
        mk_discarded = set(order[max(1, n - f):])       # Multi-Krum: discard the f worst
        a, b = _flag_rates(krum_flagged, atk_idx, ben_idx)
        out["krum"] = {
            "caught": krum_selected not in atk_idx,     # attacker is rejected unless it IS the pick
            "survival": krum_selected in atk_idx,
            "suspicion": _percentile_suspicion(scores, atk_idx, higher_is_worse=True),
            "atk_flag_rate": a, "ben_flag_rate": b,
        }
        a, b = _flag_rates(mk_discarded, atk_idx, ben_idx)
        out["multi_krum"] = {
            "caught": any(i in mk_discarded for i in atk_idx),
            "survival": all(i not in mk_discarded for i in atk_idx),
            "suspicion": _percentile_suspicion(scores, atk_idx, higher_is_worse=True),
            "atk_flag_rate": a, "ben_flag_rate": b,
        }

    # --- cosine-alignment screen: flag the f updates most aligned with the aggregate ---
    cos_order = sorted(range(n), key=lambda i: -cos_agg[i])
    cos_flagged = set(cos_order[:f])
    a, b = _flag_rates(cos_flagged, atk_idx, ben_idx)
    out["cosine_screen"] = {
        "caught": any(i in cos_flagged for i in atk_idx),
        "survival": all(i not in cos_flagged for i in atk_idx),
        "suspicion": _percentile_suspicion(cos_agg, atk_idx, higher_is_worse=True),
        "atk_flag_rate": a, "ben_flag_rate": b,
    }

    # --- rank-order collusion screen: flag the f clients whose collusion score is highest ---
    # Same top-f rule as cosine_screen (fair, identical base rate) but on the collusion score,
    # which the standard cosine screen does not see. This is the paper's C3 contribution; read
    # its deployability limit from rank_collusion_analysis, not from this single top-f number.
    rc_scores = _rank_collusion_scores(clients)
    rc_order = sorted(range(n), key=lambda i: -rc_scores[i])
    rc_flagged = set(rc_order[:f])
    a, b = _flag_rates(rc_flagged, atk_idx, ben_idx)
    out["rank_screen"] = {
        "caught": any(i in rc_flagged for i in atk_idx),
        "survival": all(i not in rc_flagged for i in atk_idx),
        "suspicion": _percentile_suspicion(rc_scores, atk_idx, higher_is_worse=True),
        "atk_flag_rate": a, "ben_flag_rate": b,
    }
    return out


def evaluate_defenses(telemetry: List[Dict], *, num_attackers: Optional[int] = None,
                      norm_clip_factor: float = 1.0) -> Dict:
    """Replay the telemetry-only defenses over all attacker-present rounds and aggregate.

    Returns {"n_rounds": int, "defenses": {name: {caught_rate, survival_rate, mean_suspicion,
    ...}}}. caught_rate is the fraction of attacker-present rounds where the defense flags or
    rejects at least one attacker — the honest detection rate a reviewer wants."""
    per_round = [r for r in (_eval_round(rnd, num_attackers=num_attackers,
                                         norm_clip_factor=norm_clip_factor)
                             for rnd in telemetry) if r is not None]
    n_rounds = len(per_round)
    names: List[str] = []
    for r in per_round:
        for k in r:
            if k not in names:
                names.append(k)
    defenses: Dict[str, Dict] = {}
    for name in names:
        rows = [r[name] for r in per_round if name in r]
        if not rows:
            continue
        atk_fr = sum(x["atk_flag_rate"] for x in rows) / len(rows)
        ben_fr = sum(x["ben_flag_rate"] for x in rows) / len(rows)
        agg = {
            "caught_rate": round(sum(1 for x in rows if x["caught"]) / len(rows), 4),
            "survival_rate": round(sum(1 for x in rows if x["survival"]) / len(rows), 4),
            "atk_flag_rate": round(atk_fr, 4),
            "ben_flag_rate": round(ben_fr, 4),           # NULL false-positive baseline
            "excess_detection": round(atk_fr - ben_fr, 4),  # discriminative power over the null
            # Krum keeps exactly 1 update, so it "flags" all n-1 non-picks: its excess is
            # structurally range-compressed (max +1/(n_benign) even for a fully detectable
            # attacker) and DOMINATED by finite-sample noise, so its sign/magnitude are NOT
            # comparable to the fixed-f defenses (cosine_screen / multi_krum, excess in [-1,1]).
            # Read Krum by caught_rate/survival only; the report/digest mark its excess as structural.
            "excess_structural": (name == "krum"),
            "mean_suspicion": round(sum(x["suspicion"] for x in rows) / len(rows), 4),
            "rounds": len(rows),                          # attacker-present rounds this defense ran on
        }
        if name == "norm_clip":
            agg["mean_atk_clip_factor"] = round(
                sum(x["atk_clip_factor"] for x in rows) / len(rows), 4)
        # Per-round excess series + a Normal-approx 95% CI, so the C3 table can state whether an
        # excess is distinguishable from zero instead of comparing a point estimate to a hard
        # 0.10 threshold. GATED on `not excess_structural`: Krum keeps 1, so its per-round excess
        # is a near-constant range-compressed value and a CI on it would look spuriously tight and
        # significant on the one metric the module declares non-comparable (excess_structural).
        if not agg["excess_structural"]:
            series = [x["atk_flag_rate"] - x["ben_flag_rate"] for x in rows]
            agg["excess_per_round"] = [round(v, 4) for v in series]
            m, se = _mean_se(series)
            agg["excess_se"] = round(se, 4)
            agg["excess_ci95"] = [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
            # Significant iff the whole 95% CI is on one side of zero.
            agg["excess_significant"] = bool((m - 1.96 * se) > 0 or (m + 1.96 * se) < 0)
        defenses[name] = agg
    return {"n_rounds": n_rounds, "defenses": defenses,
            "norm_clip_factor": norm_clip_factor}


def _mean_se(xs: List[float]) -> tuple:
    """(mean, standard error of the mean). SE=0 for n<2 (a single round carries no spread)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, (var / n) ** 0.5


def rank_collusion_analysis(telemetry: List[Dict], *, num_attackers: Optional[int] = None,
                            fpr_targets: Optional[List[float]] = None) -> Dict:
    """Characterise the rank-order collusion detector as a DETECTOR, independent of the top-f
    flagging used inside evaluate_defenses. Returns the honest C3 boundary the paper reports:

      - ``auc``: pooled attacker-vs-benign AUC of the collusion score over all client-rounds.
      - ``per_round_auc`` / ``auc_permutation_p``: a within-round label-permutation null (the
        correct significance test, since the 2 attackers in a round are correlated).
      - ``fpr_sweep``: for each target benign FPR, the score threshold calibrated on BENIGN
        clients only and the resulting attacker TPR and excess. The headline +0.45-at-top-f
        number lives at the ~0.15-0.25 benign base rate; at a strict 0.10 target the excess
        collapses toward zero, which is the deployability limit the paper must state.
      - ``benign_outranks_attacker``: whether any benign client's mean score exceeds both
        attackers' — on the 0805 run the 56%-data holder does, so the signal is confounded with
        raw data share and is not a clean detector at the client level.

    Uses only telemetry fields already logged; no re-run. Returns {} if numpy is unavailable."""
    try:
        import numpy as np
    except Exception:
        return {}
    fpr_targets = fpr_targets or [0.05, 0.10, 0.20]
    atk_scores, ben_scores = [], []
    per_round_auc, client_scores = [], {}
    for rnd in telemetry:
        clients = rnd.get("clients") or []
        labels = [c.get("label") for c in clients]
        if "attacker" not in labels:
            continue
        scores = _rank_collusion_scores(clients)
        a = [scores[i] for i, l in enumerate(labels) if l == "attacker"]
        b = [scores[i] for i, l in enumerate(labels) if l != "attacker"]
        if not a or not b:
            continue
        atk_scores.extend(a)
        ben_scores.extend(b)
        # within-round AUC (Mann-Whitney with mid-ranks)
        wins = sum((sa > sb) + 0.5 * (sa == sb) for sa in a for sb in b)
        per_round_auc.append(wins / (len(a) * len(b)))
        for c, s in zip(clients, scores):
            client_scores.setdefault((c.get("client_id"), labels[clients.index(c)]), []).append(s)
    if not atk_scores or not ben_scores:
        return {}

    def _auc(a, b):
        wins = sum((sa > sb) + 0.5 * (sa == sb) for sa in a for sb in b)
        return wins / (len(a) * len(b))

    pooled_auc = _auc(atk_scores, ben_scores)

    # Within-round label permutation: shuffle attacker/benign labels inside each round and
    # recompute the mean per-round AUC; p = fraction of permutations >= observed. Deterministic
    # seed so the reported p is reproducible.
    rng = np.random.default_rng(0)
    obs = float(np.mean(per_round_auc))
    n_perm, ge = 2000, 0
    round_pack = []
    for rnd in telemetry:
        clients = rnd.get("clients") or []
        labels = [c.get("label") for c in clients]
        if "attacker" not in labels:
            continue
        scores = _rank_collusion_scores(clients)
        n_atk = sum(1 for l in labels if l == "attacker")
        if 0 < n_atk < len(scores):
            round_pack.append((np.array(scores), n_atk))
    for _ in range(n_perm):
        aucs = []
        for scores, n_atk in round_pack:
            perm = rng.permutation(len(scores))
            a = scores[perm[:n_atk]]
            b = scores[perm[n_atk:]]
            aucs.append(_auc(list(a), list(b)))
        if np.mean(aucs) >= obs:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1)

    # FPR calibration on benign scores only (the honest, label-light operating points).
    ben_sorted = np.sort(ben_scores)
    sweep = []
    for target in fpr_targets:
        thr = float(np.quantile(ben_sorted, 1.0 - target))
        tpr = float(np.mean([s > thr for s in atk_scores]))
        fpr = float(np.mean([s > thr for s in ben_scores]))
        sweep.append({"target_fpr": target, "threshold": round(thr, 4),
                      "attacker_tpr": round(tpr, 4), "benign_fpr": round(fpr, 4),
                      "excess": round(tpr - fpr, 4)})

    client_means = {k: float(np.mean(v)) for k, v in client_scores.items()}
    atk_means = [m for (cid, lab), m in client_means.items() if lab == "attacker"]
    ben_means = [m for (cid, lab), m in client_means.items() if lab != "attacker"]
    benign_outranks = bool(ben_means and atk_means and max(ben_means) > max(atk_means))

    return {
        "auc": round(pooled_auc, 4),
        "per_round_auc_mean": round(obs, 4),
        "auc_permutation_p": round(perm_p, 5),
        "n_attacker_obs": len(atk_scores),
        "n_benign_obs": len(ben_scores),
        "fpr_sweep": sweep,
        "benign_outranks_attacker": benign_outranks,
        "note": ("per-round signal significant (permutation p) but excess collapses at strict "
                 "benign-FPR; benign_outranks_attacker flags client-level confound with data share"),
    }


# --------------------------------------------------------------------------- #
# Vector-based defenses (need save_update_vectors=True -> results/.../update_vectors/*.npz)
# --------------------------------------------------------------------------- #
def evaluate_vector_defenses(vectors_dir, *, trim_beta: float = 0.1,
                             root_direction=None) -> Dict:
    """FLTrust and coordinate-wise trimmed-mean, replayed on the raw per-client update vectors.

    FLTrust weights each update by ReLU(cos(update, root)) — needs a server-held clean ROOT
    direction. If ``root_direction`` is None a benign-mean PROXY is used and flagged as such
    (optimistic for the attacker, since it is constrained near the benign mean — a real root
    dataset is required for the definitive test). Trimmed-mean flags an attacker coordinate as
    detectable when it lands in the trimmed tail on a large fraction of coordinates.

    Returns {} if numpy is unavailable or no vectors are found."""
    try:
        import numpy as np
    except Exception:
        return {}
    vdir = Path(vectors_dir)
    files = sorted(vdir.glob("round_*.npz"))
    if not files:
        return {}
    fltrust_caught, trim_caught, n = 0, 0, 0
    proxy_root = root_direction is None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        U = d["updates"].astype("float64")            # [C, D]
        labels = [str(x) for x in d["labels"]]
        atk = [i for i, l in enumerate(labels) if l == "attacker"]
        if not atk:
            continue
        n += 1
        ben = [i for i, l in enumerate(labels) if l == "benign"]
        root = (np.asarray(root_direction, dtype="float64") if root_direction is not None
                else U[ben].mean(axis=0))
        rn = np.linalg.norm(root) + 1e-12
        # FLTrust trust score = ReLU(cos(update, root)); zero weight == rejected.
        norms = np.linalg.norm(U, axis=1) + 1e-12
        cos = (U @ root) / (norms * rn)
        ts = np.maximum(cos, 0.0)
        if any(ts[i] <= 1e-9 for i in atk):
            fltrust_caught += 1
        # Coordinate-wise trimmed-mean: fraction of coords where an attacker is in the trimmed tail.
        C = U.shape[0]
        k = max(1, int(trim_beta * C))
        order = np.argsort(U, axis=0)                 # [C, D] indices low->high per coord
        trimmed = np.concatenate([order[:k], order[-k:]], axis=0)  # [2k, D]
        in_tail = np.zeros(C)
        for i in atk:
            frac = float((trimmed == i).any(axis=0).mean())
            in_tail[i] = frac
        if any(in_tail[i] > 0.5 for i in atk):
            trim_caught += 1
    if n == 0:
        return {}
    return {
        "n_rounds": n,
        "fltrust": {"caught_rate": round(fltrust_caught / n, 4),
                    "root": "benign_mean_proxy" if proxy_root else "provided_root"},
        "trimmed_mean": {"caught_rate": round(trim_caught / n, 4), "trim_beta": trim_beta},
    }


def format_defense_report(evaluation: Dict, vector_eval: Optional[Dict] = None) -> str:
    """Human-readable summary table of evaluate_defenses (+ optional vector defenses)."""
    lines = []
    n = evaluation.get("n_rounds", 0)
    lines.append(f"DEFENSE-EVASION (offline replay on FedAvg telemetry) — attacker-present rounds: {n}")
    lines.append(f"  {'defense':<16} {'caught':>8} {'atk_fpr':>8} {'ben_fpr':>8} {'excess':>8} "
                 f"{'susp':>6} {'rnds':>5}  note")
    notes = {
        "norm_clip": lambda d: f"clip@median; atk_clip={d.get('mean_atk_clip_factor')}",
        "krum": lambda d: "keeps 1 (caught is structural: reject unless attacker IS the pick)",
        "multi_krum": lambda d: "attacker in discarded-f set",
        "cosine_screen": lambda d: f"flags top-f aligned (null base-rate f/n)",
        "rank_screen": lambda d: "flags top-f collusion score (cos-rank - pairwise-rank); see analysis",
    }
    for name, d in evaluation.get("defenses", {}).items():
        note = notes.get(name, lambda d: "")(d)
        # Krum's excess is range-compressed + noise-dominated (see evaluate_defenses); show it as
        # structural rather than on the same numeric scale as the fixed-f defenses.
        excess_cell = "struct*" if d.get("excess_structural") else f"{d.get('excess_detection', 0):>8.2f}"
        lines.append(f"  {name:<16} {d['caught_rate']:>8.2f} {d.get('atk_flag_rate', 0):>8.2f} "
                     f"{d.get('ben_flag_rate', 0):>8.2f} {excess_cell:>8} "
                     f"{d['mean_suspicion']:>6.2f} {d.get('rounds', 0):>5d}  {note}")
    if vector_eval:
        lines.append(f"  -- vector defenses ({vector_eval.get('n_rounds', 0)} rounds w/ saved vectors) --")
        ft = vector_eval.get("fltrust", {})
        tm = vector_eval.get("trimmed_mean", {})
        if ft:
            lines.append(f"  {'FLTrust':<16} {ft.get('caught_rate', float('nan')):>8.2f} "
                         f"{'':>9} {'':>10}  root={ft.get('root')}")
        if tm:
            lines.append(f"  {'trimmed_mean':<16} {tm.get('caught_rate', float('nan')):>8.2f} "
                         f"{'':>9} {'':>10}  beta={tm.get('trim_beta')}")
    lines.append("  (excess = atk_fpr - ben_fpr; ~0 => attacker indistinguishable from benign "
                 "(stealth holds); >0 => detectable. 'rnds' < total => skipped where n < 2f+3.)")
    lines.append("  (* Krum excess is 'struct': Krum keeps 1 => range-compressed & noise-dominated; "
                 "read Krum by caught/survival, NOT excess, and do not compare its excess to other rows.)")
    return "\n".join(lines)


def evaluate_from_results(results_path) -> Dict:
    """Convenience: load a saved fl_results.json and evaluate its defense_telemetry, also
    pulling in saved update-vector npz if present. Returns the combined evaluation dict."""
    rp = Path(results_path)
    r = json.loads(rp.read_text())
    tele = r.get("defense_telemetry") or []
    num_attackers = (r.get("config") or {}).get("num_attackers")
    ev = evaluate_defenses(tele, num_attackers=num_attackers)
    vdir = rp.parent / "update_vectors"
    vev = evaluate_vector_defenses(vdir) if vdir.exists() else {}
    return {"telemetry_defenses": ev, "vector_defenses": vev,
            "report": format_defense_report(ev, vev or None)}
