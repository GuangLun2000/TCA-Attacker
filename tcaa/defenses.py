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

import numpy as np


# Defenses designed against THIS attack rather than taken off the shelf. Kept as data (not a
# hardcoded name in the reporting layer) so figures/verdicts can separate "evades the standard
# defenses" from "is caught by a detector purpose-built for it" without knowing defense names.
_PURPOSE_BUILT = frozenset({"rank_screen"})


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


def _byzantine_budget(rnd: Dict, num_attackers: Optional[int]) -> int:
    """f — the Byzantine budget the aggregator provisions for, for THIS round.

    Deliberately a function of the DEFENDER's assumption (``num_attackers``) and the round size
    only, never of who is actually malicious, so that relabelling the attackers under a
    permutation null does not silently change the defenses' flagging budget. Falls back to the
    attackers actually present when the caller gives no assumption.
    """
    clients = rnd.get("clients") or []
    n = len(clients)
    if num_attackers is None:
        present = sum(1 for c in clients if c.get("label") == "attacker")
        return max(1, present)
    return max(1, min(int(num_attackers), max(n - 1, 1)))


def _flagged_sets(rnd: Dict, *, f: int, norm_clip_factor: float) -> tuple:
    """The client indices each telemetry-only defense flags this round, plus the raw scores.

    LABEL-BLIND by construction: every rule below is a function of the round's update geometry and
    the defender's budget ``f``, never of who is actually an attacker — exactly as a deployed
    defender is. That property is what lets ``cluster_permutation_test`` reuse these sets unchanged
    under every candidate relabelling instead of re-running the defenses for each one, and it is
    the single source of truth both ``_eval_round`` and that null read from (they previously
    duplicated the rules, which is how they could silently drift apart).

    Returns ``(flags, scores)`` where ``flags[name]`` is a set of client indices and ``scores``
    carries the per-client suspicion scores plus the norm-clip threshold.
    """
    clients = rnd.get("clients") or []
    n = len(clients)
    norms = [float(c.get("norm", 0.0)) for c in clients]
    cos_agg = [float(c.get("cos_to_agg", 0.0)) for c in clients]
    dist = rnd.get("pairwise_distance")

    flags: Dict[str, set] = {}
    scores: Dict[str, object] = {"norm": norms, "cos_to_agg": cos_agg}

    # --- norm-clipping: flag updates whose norm exceeds factor * median norm ---
    thr = norm_clip_factor * _median(norms)
    flags["norm_clip"] = {i for i in range(n) if norms[i] > thr}
    scores["norm_clip_threshold"] = thr

    # --- Krum / Multi-Krum: need the pairwise distance matrix AND n >= 2f+3 (Blanchard 2017) ---
    if dist and len(dist) == n and n >= 2 * f + 3:
        ks = _krum_scores(dist, f)
        order = sorted(range(n), key=lambda i: ks[i])
        flags["krum"] = set(range(n)) - {order[0]}       # Krum (m=1) rejects all but the pick
        flags["multi_krum"] = set(order[max(1, n - f):])  # Multi-Krum discards the f worst
        scores["krum_score"] = ks
        scores["krum_selected"] = order[0]

    # --- cosine-alignment screen: flag the f updates most aligned with the aggregate ---
    flags["cosine_screen"] = set(sorted(range(n), key=lambda i: -cos_agg[i])[:f])

    # --- rank-order collusion screen: flag the f clients whose collusion score is highest ---
    # Same top-f rule as cosine_screen (fair, identical base rate) but on the collusion score,
    # which the standard cosine screen does not see. This is the paper's C3 contribution; read
    # its deployability limit from rank_collusion_analysis, not from this single top-f number.
    rc = _rank_collusion_scores(clients)
    flags["rank_screen"] = set(sorted(range(n), key=lambda i: -rc[i])[:f])
    scores["rank_collusion"] = rc
    return flags, scores


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
    f = _byzantine_budget(rnd, num_attackers)
    flags, scores = _flagged_sets(rnd, f=f, norm_clip_factor=norm_clip_factor)
    norms = scores["norm"]
    na, nb = len(atk_idx), len(ben_idx)

    out: Dict[str, Dict] = {}

    def _row(flagged: set, score_vec: List[float], *, caught: bool, survival: bool) -> Dict:
        a, b = _flag_rates(flagged, atk_idx, ben_idx)
        return {
            "caught": caught, "survival": survival,
            "suspicion": _percentile_suspicion(score_vec, atk_idx, higher_is_worse=True),
            "atk_flag_rate": a, "ben_flag_rate": b,
            # Per-round exact-null bookkeeping: pool sizes and how many of each label were flagged.
            # These let evaluate_defenses build a within-round hypergeometric permutation null for
            # the excess, which is the right test for a top-f (budget-constrained) screen whose
            # excess can only take a few discrete values — a Normal-approx CI on the sample sd
            # over-states significance.
            "n_atk": na, "n_ben": nb,
            "n_atk_flagged": sum(1 for i in atk_idx if i in flagged),
            "n_flagged": len(flagged),
        }

    thr = scores["norm_clip_threshold"]
    nc_flagged = flags["norm_clip"]
    out["norm_clip"] = _row(nc_flagged, norms,
                            caught=any(i in nc_flagged for i in atk_idx),
                            survival=not any(i in nc_flagged for i in atk_idx))
    # how much the largest attacker update would be dampened (1.0 = untouched)
    out["norm_clip"]["atk_clip_factor"] = min(
        (thr / norms[i]) if norms[i] > thr else 1.0 for i in atk_idx)

    if "krum" in flags:
        ks = scores["krum_score"]
        selected = scores["krum_selected"]
        out["krum"] = _row(flags["krum"], ks,
                           caught=selected not in atk_idx,   # rejected unless it IS the pick
                           survival=selected in atk_idx)
        mk = flags["multi_krum"]
        out["multi_krum"] = _row(mk, ks,
                                 caught=any(i in mk for i in atk_idx),
                                 survival=all(i not in mk for i in atk_idx))

    cs = flags["cosine_screen"]
    out["cosine_screen"] = _row(cs, scores["cos_to_agg"],
                                caught=any(i in cs for i in atk_idx),
                                survival=all(i not in cs for i in atk_idx))

    rs = flags["rank_screen"]
    out["rank_screen"] = _row(rs, scores["rank_collusion"],
                              caught=any(i in rs for i in atk_idx),
                              survival=all(i not in rs for i in atk_idx))
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
    # Client-level null, computed once and reused for every defense below (see the block comment
    # at `excess_significance_method` for why this is the headline test rather than the
    # within-round one).
    cluster = cluster_permutation_test(telemetry, num_attackers=num_attackers,
                                       norm_clip_factor=norm_clip_factor)
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
            # PURPOSE-BUILT vs STANDARD. norm_clip / krum / multi_krum / cosine_screen are
            # published aggregators an off-the-shelf defender would deploy; rank_screen was
            # designed against THIS attack's collusion mechanism. Reporting must not average the
            # two — "evades standard defenses" and "is caught by a detector built for it" are
            # different claims, and only the first is a stealth result.
            "purpose_built": (name in _PURPOSE_BUILT),
            "mean_suspicion": round(sum(x["suspicion"] for x in rows) / len(rows), 4),
            "rounds": len(rows),                          # attacker-present rounds this defense ran on
        }
        if name == "norm_clip":
            agg["mean_atk_clip_factor"] = round(
                sum(x["atk_clip_factor"] for x in rows) / len(rows), 4)
        # THREE nested descriptions of the same excess, from weakest to strongest assumption:
        #
        #  1. excess_ci95_normal  — a Normal-approx CI on the per-round sample sd. DESCRIPTIVE
        #     SPREAD ONLY, never significance: a top-f screen flags a fixed budget each round, so
        #     the excess takes a handful of discrete values and this CI over-states significance
        #     (it is what wrongly flagged norm_clip as "sig" when its exact p is ~0.09).
        #  2. excess_null_band95 / excess_p_within_round — the exact within-round hypergeometric
        #     null. Correct for "inside a round, is flagging associated with the attacker label?"
        #     but it treats the 10 rounds as independent experiments, which they are NOT: the same
        #     two clients are the attackers in every round, so a persistently-odd client inflates
        #     it. Anti-conservative for a PERSISTENT label. NOTE excess_null_band95 is the band the
        #     excess falls in UNDER H0 — it is not a confidence interval on the estimate, so the
        #     point estimate legitimately lies outside it when the effect is real.
        #  3. excess_p_cluster — the client-level permutation (see cluster_permutation_test).
        #     This is the HEADLINE significance: it asks whether the actual attacker set is flagged
        #     more than an arbitrary set of the same size would be, which respects the fact that
        #     attacker identity is a per-client property held fixed across rounds.
        #
        # `excess_significant` reads (3) when available and falls back to (2) then (1). GATED on
        # `not excess_structural`: Krum keeps 1 by construction, so its excess is non-comparable
        # and carries no meaningful null.
        if not agg["excess_structural"]:
            series = [x["atk_flag_rate"] - x["ben_flag_rate"] for x in rows]
            agg["excess_per_round"] = [round(v, 4) for v in series]
            m, se = _mean_se(series)
            agg["excess_se"] = round(se, 4)
            agg["excess_ci95_normal"] = [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
            method = "normal_approx_fallback"
            significant = bool((m - 1.96 * se) > 0 or (m + 1.96 * se) < 0)
            perm = _excess_permutation_test(rows)
            if perm is not None:
                agg["excess_p_within_round"] = round(perm["p_two_sided"], 4)
                agg["excess_null_band95"] = [round(perm["ci_lo"], 4), round(perm["ci_hi"], 4)]
                agg["excess_null_sd"] = round(perm["null_sd"], 4)
                method = "within_round_hypergeometric"
                significant = bool(perm["p_two_sided"] < 0.05)
            clu = (cluster or {}).get(name)
            if clu is not None and clu.get("p_two_sided") is not None:
                agg["excess_p_cluster"] = round(clu["p_two_sided"], 4)
                agg["cluster_rank"] = clu.get("rank")
                agg["cluster_n_candidates"] = clu.get("n_candidates")
                agg["cluster_top_competitors"] = clu.get("top_competitors")
                method = "client_level_permutation"
                significant = bool(clu["p_two_sided"] < 0.05)
            agg["excess_significance_method"] = method
            agg["excess_significant"] = significant
            # The cluster test enumerates C(n_clients, n_attackers) candidate attacker sets and
            # the OBSERVED set is always one of them, so p_two_sided >= 1/n_candidates. A 5-client
            # / 1-attacker federation floors p at 0.20, and `significant` is then False for every
            # possible outcome — including an attacker every aggregator flags 100% of the time.
            # Publish the floor so a consumer can tell "evaded the defense" from "the test had no
            # power to say"; without it a structural False reads as a stealth result.
            n_candidates = agg.get("cluster_n_candidates")
            if n_candidates:
                floor = 1.0 / int(n_candidates)
                agg["excess_min_attainable_p"] = round(floor, 4)
                agg["excess_significance_underpowered"] = bool(floor > 0.05)
        defenses[name] = agg
    return {"n_rounds": n_rounds, "defenses": defenses,
            "norm_clip_factor": norm_clip_factor,
            "cluster_permutation": cluster}


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


def _excess_permutation_test(rows: List[Dict], n_resample: int = 20000) -> Optional[Dict]:
    """Exact-null test for the mean per-round attacker-minus-benign flag-rate excess of a
    budget-constrained (top-f) screen.

    A top-f screen flags a FIXED number of clients each round, so within round r, under the null
    that flags are assigned independently of the attacker label, the number of flagged attackers is
    Hypergeometric(N_r, n_atk_r, n_flagged_r). The excess is therefore a sum of a few discrete
    per-round increments — a Normal approximation on the sample sd over-states its significance
    (it is exactly what mislabelled norm_clip as "significant"). We build the null by resampling
    the per-round attacker-flag count from its hypergeometric distribution and recomputing the
    mean excess, giving a distribution-free two-sided p-value and a null-referenced 95% interval.

    Returns None if any round lacks the per-round counts (older telemetry), so the caller can fall
    back to the Normal CI. Deterministic: the RNG is seeded so re-running a fixed run reproduces.
    """
    counts = []
    for x in rows:
        if not all(k in x for k in ("n_atk", "n_ben", "n_atk_flagged", "n_flagged")):
            return None
        na, nb, ka, f = x["n_atk"], x["n_ben"], x["n_atk_flagged"], x["n_flagged"]
        if na <= 0 or nb <= 0:
            return None
        counts.append((na, nb, ka, f))
    if not counts:
        return None

    def excess_from(ka_list) -> float:
        vals = []
        for (na, nb, _ka, f), ka in zip(counts, ka_list):
            kb = f - ka
            vals.append(ka / na - kb / nb)
        return sum(vals) / len(vals)

    observed = excess_from([c[2] for c in counts])
    rng = np.random.default_rng(0)  # fixed seed: same telemetry -> same p-value
    null = np.empty(n_resample, dtype=float)
    for j in range(n_resample):
        draw = [int(rng.hypergeometric(na, nb, f)) for (na, nb, _ka, f) in counts]
        null[j] = excess_from(draw)
    null_mean = float(null.mean())
    # Two-sided p: how often the null deviates from its own mean at least as far as the observed.
    p_two = float((np.abs(null - null_mean) >= abs(observed - null_mean) - 1e-12).mean())
    return {
        "observed": observed,
        "p_two_sided": max(p_two, 1.0 / n_resample),
        "null_sd": float(null.std(ddof=1)),
        "ci_lo": float(np.quantile(null, 0.025)),
        "ci_hi": float(np.quantile(null, 0.975)),
    }


def cluster_permutation_test(telemetry: List[Dict], *, num_attackers: Optional[int] = None,
                             norm_clip_factor: float = 1.0,
                             min_round_fraction: float = 0.5,
                             max_candidates: int = 20000) -> Dict:
    """CLIENT-LEVEL (cluster) permutation null for each defense's excess detection.

    The within-round hypergeometric null (``_excess_permutation_test``) asks "inside this round,
    is being flagged associated with the attacker label?" and treats every round as an independent
    experiment. That assumption is FALSE here: the same clients are the attackers in all rounds and
    their update geometry is persistent (in the 20260807 run one honest client holds 55% of the
    data and is a geometric outlier every single round). So a screen that simply flags the biggest
    data-holder every round can earn a tiny within-round p without discriminating attackers at all,
    and the effective sample size is the number of CLIENTS, not the number of client-rounds.

    This test fixes that by permuting the label at the level it actually lives: it re-labels each
    candidate SET of clients (same size as the true attacker set) as "the attackers", holding that
    assignment fixed across all rounds, and recomputes the mean per-round excess. The p-value is
    the fraction of candidate sets whose |excess| is at least the observed one. Because every
    defense rule here is label-blind (see ``_flagged_sets``), the flagged sets are computed ONCE
    per round and reused for every candidate — the test is exact, not resampled, whenever the
    candidate space fits in ``max_candidates``.

    Clients are only sampled into a subset of rounds, so a candidate set is admissible only for the
    rounds in which all of its members are present; sets co-present in fewer than
    ``min_round_fraction`` of the attacker-present rounds are dropped as too thin to compare.

    Returns ``{defense_name: {observed, p_two_sided, rank, n_candidates, n_rounds,
    top_competitors}}`` — ``top_competitors`` names the highest-scoring candidate sets, which is
    how the "a benign client outranks both attackers" confound becomes visible. Returns {} when
    the telemetry carries no attacker or no usable client ids.
    """
    import itertools
    import math

    rounds = []
    for rnd in telemetry:
        clients = rnd.get("clients") or []
        if not clients or not any(c.get("label") == "attacker" for c in clients):
            continue
        ids = [c.get("client_id") for c in clients]
        if any(i is None for i in ids):
            return {}                      # cannot form client-level sets without stable ids
        f = _byzantine_budget(rnd, num_attackers)
        flags, _ = _flagged_sets(rnd, f=f, norm_clip_factor=norm_clip_factor)
        rounds.append({"ids": ids,
                       "flagged": {k: {ids[i] for i in v} for k, v in flags.items()}})
    if not rounds:
        return {}

    true_atk = sorted({c.get("client_id") for rnd in telemetry
                       for c in (rnd.get("clients") or []) if c.get("label") == "attacker"})
    all_ids = sorted({i for r in rounds for i in r["ids"]})
    k = len(true_atk)
    if k == 0 or k >= len(all_ids):
        return {}
    min_rounds = max(1, int(round(min_round_fraction * len(rounds))))

    def _excess(candidate: set, defense: str):
        """Mean per-round (candidate flag-rate − other flag-rate), over rounds where the whole
        candidate set participated AND this defense ran. None if too few such rounds."""
        vals = []
        for r in rounds:
            if defense not in r["flagged"]:
                continue                    # e.g. Krum skipped this round (n < 2f+3)
            present = set(r["ids"])
            if not candidate <= present:
                continue
            flagged = r["flagged"][defense]
            others = present - candidate
            if not others:
                continue
            a = len(candidate & flagged) / len(candidate)
            b = len(others & flagged) / len(others)
            vals.append(a - b)
        if len(vals) < min_rounds:
            return None
        return sum(vals) / len(vals)

    names = sorted({d for r in rounds for d in r["flagged"]})
    # Count BEFORE materialising: C(num_clients, k) explodes fast (C(100,5) is 75M tuples), so a
    # naive list() would exhaust memory on a large federation before the truncation check ran.
    total_candidates = math.comb(len(all_ids), k)
    truncated = total_candidates > max_candidates
    if not truncated:
        candidates = list(itertools.combinations(all_ids, k))
    else:
        # Too many client subsets to enumerate exactly: sample the null directly (never building
        # the full space), always keeping the observed assignment so the p-value stays defined.
        rng = np.random.default_rng(0)
        keep = {tuple(true_atk)}
        guard = 0
        while len(keep) < max_candidates and guard < 20 * max_candidates:
            keep.add(tuple(sorted(rng.choice(all_ids, size=k, replace=False).tolist())))
            guard += 1
        candidates = sorted(keep)

    out: Dict[str, Dict] = {}
    for name in names:
        scored = []
        for cand in candidates:
            e = _excess(set(cand), name)
            if e is not None:
                scored.append((e, cand))
        obs = next((e for e, c in scored if tuple(sorted(c)) == tuple(true_atk)), None)
        if obs is None or len(scored) < 2:
            continue
        ge = sum(1 for e, _ in scored if abs(e) >= abs(obs) - 1e-12)
        ordered = sorted(scored, key=lambda t: -t[0])
        rank = next(i for i, (_, c) in enumerate(ordered, 1) if tuple(sorted(c)) == tuple(true_atk))
        out[name] = {
            "observed": round(obs, 4),
            "p_two_sided": round(ge / len(scored), 4),
            "rank": rank,                       # 1 = the true attacker set is the most-flagged
            "n_candidates": len(scored),
            "n_rounds": len(rounds),
            "min_rounds": min_rounds,
            "exact": not truncated,
            "attacker_ids": list(true_atk),
            # The candidate sets that beat or match the attackers — the concrete form of the
            # "an honest heavy data-holder looks more suspicious than the attacker" confound.
            "top_competitors": [{"clients": list(c), "excess": round(e, 4)}
                                for e, c in ordered[:3]],
        }
    return out


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
        "cosine_screen": lambda d: "flags top-f aligned (null base-rate f/n)",
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
