# tcaa/tests/test_defenses.py
# The offline defense evaluator must CATCH a blatant outlier attacker and PASS a well-hidden
# one, on the same telemetry schema fl_runner._collect_defense_telemetry produces.

from tcaa.defenses import (evaluate_defenses, format_defense_report,
                           rank_collusion_analysis)


def _round(norms, cos_to_agg, coords, labels, pairwise_mean_cos=None):
    """Build a telemetry round from per-client 1-D coordinates (for the pairwise matrix).

    ``pairwise_mean_cos`` (optional) sets each client's mean-pairwise-cosine; the rank-order
    collusion screen needs it to be non-degenerate, so the collusion tests pass it explicitly.
    Defaults to 0.0 for every client (the original behaviour) when omitted."""
    n = len(labels)
    dist = [[abs(coords[i] - coords[j]) for j in range(n)] for i in range(n)]
    pmc = pairwise_mean_cos if pairwise_mean_cos is not None else [0.0] * n
    clients = [{"client_id": i, "label": labels[i], "norm": norms[i],
                "cos_to_agg": cos_to_agg[i], "dist_to_agg": 0.0,
                "dist_to_benign_mean": 0.0, "pairwise_mean_cos": pmc[i], "krum_score": 0.0}
               for i in range(n)]
    return {"round": 0, "n_clients": n, "n_attackers_present": labels.count("attacker"),
            "clients": clients, "pairwise_distance": dist,
            "pairwise_cosine": [[1.0] * n for _ in range(n)]}


def test_blatant_attacker_is_caught():
    # 4 benign clustered near 0 with small norms; 1 attacker far away with a big norm.
    labels = ["benign", "benign", "benign", "benign", "attacker"]
    coords = [0.0, 0.1, -0.1, 0.05, 5.0]
    norms = [1.0, 1.1, 0.9, 1.0, 4.0]
    cos = [0.3, 0.32, 0.28, 0.31, 0.95]
    ev = evaluate_defenses([_round(norms, cos, coords, labels)], num_attackers=1)
    d = ev["defenses"]
    assert d["norm_clip"]["caught_rate"] == 1.0, d["norm_clip"]
    assert d["multi_krum"]["caught_rate"] == 1.0, d["multi_krum"]
    assert d["cosine_screen"]["caught_rate"] == 1.0, d["cosine_screen"]
    # suspicion percentile should peg the attacker as the single most suspicious client.
    assert d["norm_clip"]["mean_suspicion"] == 1.0
    print("[ok] blatant attacker caught by norm-clip / Multi-Krum / cosine-screen")
    print(format_defense_report(ev))


def test_well_hidden_attacker_survives():
    # Attacker sits INSIDE the benign cluster with a benign-sized norm and typical alignment.
    labels = ["benign", "benign", "benign", "benign", "attacker"]
    coords = [0.0, 0.3, -0.3, 0.15, 0.05]
    norms = [1.0, 1.2, 0.9, 1.1, 1.05]
    cos = [0.30, 0.34, 0.27, 0.31, 0.30]
    ev = evaluate_defenses([_round(norms, cos, coords, labels)], num_attackers=1)
    d = ev["defenses"]
    assert d["norm_clip"]["caught_rate"] == 0.0, d["norm_clip"]
    assert d["multi_krum"]["survival_rate"] == 1.0, d["multi_krum"]
    print("[ok] well-hidden attacker survives norm-clip / Multi-Krum")


def test_excess_detection_separates_blatant_from_hidden():
    # excess_detection = atk_flag_rate - ben_flag_rate is the honest, base-rate-corrected signal:
    # HIGH for a blatant attacker, ~0 or NEGATIVE for one that hides inside the benign envelope.
    blatant = _round([1.0, 1.1, 0.9, 1.0, 4.0], [0.3, 0.32, 0.28, 0.31, 0.95],
                     [0.0, 0.1, -0.1, 0.05, 5.0], ["benign"] * 4 + ["attacker"])
    hidden = _round([1.0, 1.2, 0.9, 1.1, 1.05], [0.30, 0.34, 0.27, 0.31, 0.30],
                    [0.0, 0.3, -0.3, 0.15, 0.05], ["benign"] * 4 + ["attacker"])
    db = evaluate_defenses([blatant], num_attackers=1)["defenses"]
    dh = evaluate_defenses([hidden], num_attackers=1)["defenses"]
    for name in ("norm_clip", "cosine_screen", "multi_krum"):
        # every telemetry defense must carry the null baseline and excess
        for key in ("atk_flag_rate", "ben_flag_rate", "excess_detection"):
            assert key in db[name], (name, key)
        # the blatant attacker is strictly more detectable than the hidden one on each defense
        assert db[name]["excess_detection"] > dh[name]["excess_detection"], (
            name, db[name]["excess_detection"], dh[name]["excess_detection"])
    # cosine-screen: a hidden attacker is flagged NO MORE than a benign client (excess <= 0)
    assert dh["cosine_screen"]["excess_detection"] <= 0.0, dh["cosine_screen"]
    print("[ok] excess_detection separates blatant (high) from hidden (<=0) with a null baseline")


def test_krum_skipped_when_n_below_2f_plus_3():
    # n=5, f=2 => 2f+3 = 7 > 5, so Krum/Multi-Krum are mathematically invalid and must be SKIPPED
    # (not silently degraded), while the base-rate defenses still run.
    labels = ["benign", "benign", "benign", "attacker", "attacker"]
    r = _round([1.0, 1.1, 0.9, 1.0, 1.0], [0.3, 0.3, 0.3, 0.3, 0.3],
               [0.0, 0.1, -0.1, 0.05, 0.02], labels)
    ev = evaluate_defenses([r], num_attackers=2)
    d = ev["defenses"]
    assert "krum" not in d and "multi_krum" not in d, d
    assert "norm_clip" in d and "cosine_screen" in d, d
    print("[ok] Krum/Multi-Krum correctly skipped when n < 2f+3")


def test_excess_null_present_and_krum_excluded():
    # Every non-structural defense must carry a null band + a significance verdict; Krum
    # (excess_structural) must NOT, so a spuriously-tight interval on its range-compressed excess
    # never enters the C3 table.
    r = _round([1.0, 1.1, 0.9, 1.0, 1.0, 0.95, 1.05, 1.02],
               [0.30, 0.34, 0.27, 0.31, 0.30, 0.29, 0.33, 0.28],
               [0.0, 0.3, -0.3, 0.15, 0.05, -0.1, 0.2, -0.05],
               ["benign"] * 6 + ["attacker"] * 2)
    d = evaluate_defenses([r, r], num_attackers=2)["defenses"]  # 2 rounds so stdev is defined
    for name in ("norm_clip", "cosine_screen", "rank_screen"):
        assert "excess_null_band95" in d[name], (name, d[name])
        assert "excess_significant" in d[name], (name, d[name])
        # The ambiguous legacy key must be GONE: it used to mean "Normal CI" and was then reused
        # for the permutation null band, so a consumer could not tell which it was holding.
        assert "excess_ci95" not in d[name], (name, "ambiguous legacy key resurfaced")
        assert d[name]["excess_significance_method"] in (
            "normal_approx_fallback", "within_round_hypergeometric", "client_level_permutation")
    assert "excess_null_band95" not in d["krum"], "Krum's structural excess must not get a null band"
    assert "excess_significant" not in d["krum"]
    assert d["krum"]["excess_structural"] is True
    print("[ok] null band present for fixed-f defenses, correctly withheld from structural Krum")


def test_cluster_permutation_is_the_headline_null_and_is_conservative():
    """The client-level null must (a) be reported, (b) drive `excess_significant`, and (c) be no
    more liberal than the within-round null on a persistent-label design.

    Two clients are the attackers in EVERY round, so the within-round null (which treats rounds as
    independent) over-counts evidence. Here a single benign client is a permanent geometric outlier
    on the collusion score; a correct client-level test must notice that swapping it in scores as
    well as the real attackers, and therefore must not certify detection."""
    labels = ["benign"] * 6 + ["attacker"] * 2
    rounds = []
    for shift in (0.0, 0.02, -0.01, 0.03, -0.02, 0.01):
        # client 0 is a permanent outlier: highest cos_to_agg AND highest pairwise cosine
        cos = [0.95, 0.40 + shift, 0.42, 0.38, 0.41, 0.39, 0.70 + shift, 0.72]
        pmc = [0.90, 0.55, 0.58, 0.52, 0.54, 0.56, 0.05, 0.03]
        rounds.append(_round([1.0] * 8, cos, [0, .1, -.1, .05, .2, -.2, .15, -.15], labels, pmc))
    ev = evaluate_defenses(rounds, num_attackers=2)
    clu = ev.get("cluster_permutation") or {}
    assert clu, "cluster_permutation must be reported alongside the per-defense rows"
    for name, row in clu.items():
        assert 0.0 <= row["p_two_sided"] <= 1.0
        assert row["n_candidates"] >= 2 and row["rank"] >= 1
        assert row["attacker_ids"] == [6, 7]
    for name, agg in ev["defenses"].items():
        if agg.get("excess_structural"):
            continue
        if "excess_p_cluster" in agg:
            # Significance is read from the client-level p, not the within-round one.
            assert agg["excess_significance_method"] == "client_level_permutation"
            assert agg["excess_significant"] is bool(agg["excess_p_cluster"] < 0.05)
            # ...and the client-level p must be >= the within-round p on this persistent-label
            # design: the within-round null is the anti-conservative one.
            if "excess_p_within_round" in agg:
                assert agg["excess_p_cluster"] >= agg["excess_p_within_round"] - 1e-9, (
                    name, agg["excess_p_cluster"], agg["excess_p_within_round"])
    print("[ok] client-level permutation reported, drives significance, and is conservative")


def test_rank_screen_catches_collusion_cosine_screen_misses():
    # Two colluding attackers: HIGH cos_to_agg (they pull the aggregate) but LOW pairwise cosine
    # (a coordination penalty holds them apart). A benign client is the MOST aligned overall, so
    # the plain top-f cosine screen flags the benign client, while the rank-order collusion score
    # (cos-rank minus pairwise-rank) separates the attackers.
    labels = ["benign"] * 4 + ["attacker"] * 2
    cos_to_agg = [0.95, 0.40, 0.42, 0.38, 0.70, 0.72]   # benign 0 is the single most aligned
    pairwise = [0.90, 0.55, 0.58, 0.52, 0.05, 0.03]     # attackers are least mutually aligned
    coords = [0.0, 0.1, -0.1, 0.05, 0.2, -0.2]
    norms = [1.0, 1.1, 0.9, 1.0, 1.05, 1.02]
    ev = evaluate_defenses([_round(norms, cos_to_agg, coords, labels, pairwise)], num_attackers=2)
    d = ev["defenses"]
    assert "rank_screen" in d, d
    # the collusion screen is strictly more discriminative here than the plain cosine screen
    assert d["rank_screen"]["excess_detection"] > d["cosine_screen"]["excess_detection"], (
        d["rank_screen"]["excess_detection"], d["cosine_screen"]["excess_detection"])
    print("[ok] rank_screen separates colluders that the plain cosine screen misses")


def test_rank_collusion_analysis_reports_boundary():
    # The analysis must return an AUC and an FPR sweep whose excess is non-increasing as the
    # benign-FPR budget tightens (the deployability boundary the paper reports honestly).
    labels = ["benign"] * 4 + ["attacker"] * 2
    rounds = []
    for shift in (0.0, 0.05, -0.03):
        cos = [0.95, 0.40 + shift, 0.42, 0.38, 0.70 + shift, 0.72]
        pmc = [0.90, 0.55, 0.58, 0.52, 0.05, 0.03]
        rounds.append(_round([1.0] * 6, cos, [0, .1, -.1, .05, .2, -.2], labels, pmc))
    an = rank_collusion_analysis(rounds)
    if not an:  # numpy unavailable -> analysis returns {}; skip rather than fail
        print("[skip] numpy unavailable; rank_collusion_analysis returns {}")
        return
    assert 0.0 <= an["auc"] <= 1.0 and an["fpr_sweep"], an
    # As the benign-FPR budget widens the threshold drops, so both attacker-TPR and benign-FPR
    # can only rise (a relaxed threshold never un-flags anyone). Excess itself is NOT monotone —
    # in a cleanly separable case it can fall once all attackers are caught and only benign FPs
    # accrue — which is exactly why the paper must report the operating point, not one number.
    sweep = sorted(an["fpr_sweep"], key=lambda s: s["target_fpr"])
    tprs = [s["attacker_tpr"] for s in sweep]
    fprs = [s["benign_fpr"] for s in sweep]
    assert tprs == sorted(tprs), ("TPR must not fall as FPR budget widens", tprs)
    assert fprs == sorted(fprs), ("FPR must not fall as FPR budget widens", fprs)
    assert "benign_outranks_attacker" in an
    print("[ok] rank_collusion_analysis reports AUC + monotone TPR/FPR sweep + client-level confound")


if __name__ == "__main__":
    test_blatant_attacker_is_caught()
    test_well_hidden_attacker_survives()
    test_excess_detection_separates_blatant_from_hidden()
    test_krum_skipped_when_n_below_2f_plus_3()
    test_excess_ci_present_and_krum_excluded()
    test_rank_screen_catches_collusion_cosine_screen_misses()
    test_rank_collusion_analysis_reports_boundary()
    print("\nAll defense-evaluator tests passed.")
