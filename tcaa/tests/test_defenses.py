# tcaa/tests/test_defenses.py
# The offline defense evaluator must CATCH a blatant outlier attacker and PASS a well-hidden
# one, on the same telemetry schema fl_runner._collect_defense_telemetry produces.

from tcaa.defenses import evaluate_defenses, format_defense_report


def _round(norms, cos_to_agg, coords, labels):
    """Build a telemetry round from per-client 1-D coordinates (for the pairwise matrix)."""
    n = len(labels)
    dist = [[abs(coords[i] - coords[j]) for j in range(n)] for i in range(n)]
    clients = [{"client_id": i, "label": labels[i], "norm": norms[i],
                "cos_to_agg": cos_to_agg[i], "dist_to_agg": 0.0,
                "dist_to_benign_mean": 0.0, "pairwise_mean_cos": 0.0, "krum_score": 0.0}
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


if __name__ == "__main__":
    test_blatant_attacker_is_caught()
    test_well_hidden_attacker_survives()
    test_excess_detection_separates_blatant_from_hidden()
    test_krum_skipped_when_n_below_2f_plus_3()
    print("\nAll defense-evaluator tests passed.")
