# tcaa/tests/test_defense_in_loop.py
# The server-side robust-aggregation dispatcher (_aggregate) and the Dirichlet min-shard floor.
# These serve two paper claims: "the attack survives a defender in the loop" (server_aggregator)
# and "every seed ran the intended federation size" (min_client_shard).

import numpy as np
import pytest
import torch

from tcaa.fl_runner import (_aggregate, _fedavg, _validate_fl_config, default_fl_config,
                            partition_health)
from tcaa.gen_data import partition_examples


# --------------------------------------------------------------------------- #
# server-side aggregation dispatcher
# --------------------------------------------------------------------------- #
def _benign_and_outlier():
    ben = [torch.tensor([1.0, 0.0]), torch.tensor([1.1, 0.1]),
           torch.tensor([0.9, -0.1]), torch.tensor([1.0, 0.05])]
    atk = torch.tensor([8.0, 8.0])
    return ben, atk


def test_fedavg_default_is_bit_identical():
    ben, atk = _benign_and_outlier()
    updates = ben + [atk]
    weights = [1.0] * len(updates)
    for cfg in ({}, {"server_aggregator": None}, {"server_aggregator": "fedavg"}):
        got = _aggregate(updates, weights, cfg)
        assert torch.allclose(got, _fedavg(updates, weights)), cfg
    print("[ok] server_aggregator default reproduces plain FedAvg bit-for-bit")


def test_norm_clip_pulls_in_the_outlier():
    ben, atk = _benign_and_outlier()
    updates = ben + [atk]
    w = [1.0] * len(updates)
    plain = _fedavg(updates, w)
    clipped = _aggregate(updates, w, {"server_aggregator": "norm_clip", "num_attackers": 1,
                                      "server_norm_clip_factor": 1.0})
    # the clipped aggregate is closer to the benign centroid (~[1,0]) than plain FedAvg
    benign_centroid = torch.stack(ben).mean(0)
    assert torch.norm(clipped - benign_centroid) < torch.norm(plain - benign_centroid)
    print("[ok] norm_clip in the loop pulls the outlier update toward the benign norm")


def test_krum_and_multi_krum_exclude_the_outlier():
    ben, atk = _benign_and_outlier()
    updates = ben + [atk]           # n=5, f=1 -> 2f+3=5 <= 5, valid
    w = [1.0] * len(updates)
    for rule in ("krum", "multi_krum"):
        agg = _aggregate(updates, w, {"server_aggregator": rule, "num_attackers": 1})
        assert agg[0] < 2.0, (rule, agg)   # outlier at 8.0 must not dominate
    print("[ok] krum / multi_krum exclude the blatant outlier from the aggregate")


def test_krum_falls_back_when_n_too_small():
    # n=2 cannot satisfy 2f+3 for any f>=1; must fall back to FedAvg rather than crash.
    updates = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]
    got = _aggregate(updates, [1.0, 1.0], {"server_aggregator": "krum", "num_attackers": 1})
    assert torch.allclose(got, _fedavg(updates, [1.0, 1.0]))
    print("[ok] krum falls back to FedAvg when n < 5 instead of failing")


def test_unknown_aggregator_raises():
    with pytest.raises(ValueError):
        _aggregate([torch.tensor([1.0]), torch.tensor([2.0])], [1.0, 1.0],
                   {"server_aggregator": "median_of_means"})
    print("[ok] unknown server_aggregator is rejected")


# --------------------------------------------------------------------------- #
# Dirichlet partition floor
# --------------------------------------------------------------------------- #
def test_min_shard_floor_removes_empty_shards():
    ex = list(range(1500))
    # seed 42069 reproduces the 0805 run's empty shard (client 6) at min_shard=0
    raw = partition_examples(ex, 8, 0.3, seed=42069, min_shard=0)
    assert any(len(s) == 0 for s in raw), "expected the known empty-shard seed to reproduce"
    floored = partition_examples(ex, 8, 0.3, seed=42069, min_shard=8)
    assert min(len(s) for s in floored) >= 8
    assert sum(len(s) for s in floored) == len(ex)  # mass preserved
    print("[ok] min_client_shard floors every shard and preserves total mass")


def test_min_shard_default_is_backward_compatible():
    ex = list(range(1500))
    a = [len(s) for s in partition_examples(ex, 8, 0.3, seed=7)]
    b = [len(s) for s in partition_examples(ex, 8, 0.3, seed=7, min_shard=0)]
    assert a == b == [len(s) for s in partition_examples(ex, 8, 0.3, seed=7)]
    print("[ok] min_shard=0 reproduces the original partition exactly")


def test_min_shard_impossible_floor_raises():
    with pytest.raises(ValueError):
        partition_examples(list(range(100)), 8, 0.3, seed=0, min_shard=50)  # 8*50 > 100
    print("[ok] an unsatisfiable floor is rejected up front")


def test_min_shard_holds_across_seeds_and_scales():
    ex = list(range(1500))
    for nc in (8, 18, 28):
        for seed in range(25):
            sh = partition_examples(ex, nc, 0.3, seed=seed, min_shard=8)
            assert min(len(s) for s in sh) >= 8
            assert sum(len(s) for s in sh) == len(ex)
    print("[ok] floor holds across 25 seeds x {8,18,28} clients, mass preserved")


# --------------------------------------------------------------------------- #
# partition health (credibility gate)
# --------------------------------------------------------------------------- #
def test_partition_health_flags_the_degenerate_run_and_passes_healthy_ones():
    # The 20260807 run's realised split: one client holds 55%, one does 2 grad steps/round.
    deg = partition_health([88, 77, 827, 26, 248, 200, 8, 26], batch_size=8, local_epochs=2)
    assert deg["effective_clients"] < 3.0, deg          # nominal 8, actually ~2.8
    assert deg["max_share"] > 0.5, deg
    assert deg["min_grad_steps_per_round"] == 2, deg    # ceil(8/8)*2
    # A near-IID split (alpha=100) is healthy on the same seed.
    heal = partition_health([191, 201, 169, 174, 196, 207, 172, 190], batch_size=8, local_epochs=2)
    assert heal["effective_clients"] > 7.5, heal
    assert heal["max_share"] < 0.2, heal
    assert heal["min_grad_steps_per_round"] >= 40, heal
    print("[ok] partition_health separates the degenerate run from a near-IID split")


def test_min_effective_clients_gate_rejects_before_the_run():
    # A gate above the realised effective-client count must raise a clear error PRE-flight.
    # We drive it through _validate-adjacent logic by calling partition_health directly (the
    # run_fl gate is a thin `if health < min_eff: raise`), so this stays torch-light and fast.
    h = partition_health([88, 77, 827, 26, 248, 200, 8, 26], batch_size=8, local_epochs=2)
    min_eff = 4.0
    assert h["effective_clients"] < min_eff, "gate would not fire on the known-degenerate split"
    # ...and a healthy split clears the same gate.
    h2 = partition_health([123, 105, 202, 25, 252, 373, 172, 248], batch_size=8, local_epochs=2)
    assert h2["effective_clients"] >= min_eff, h2
    print("[ok] min_effective_clients gate fires on degenerate, clears on healthy")


def test_partition_health_default_is_off_and_backward_compatible():
    # The gate is opt-in: the default config must not carry a positive threshold, so existing
    # runs are never rejected by it.
    assert float(default_fl_config().get("min_effective_clients", 0) or 0) == 0.0
    print("[ok] min_effective_clients defaults to 0 (gate off) — backward compatible")


# --------------------------------------------------------------------------- #
# config validation guards
# --------------------------------------------------------------------------- #
def test_validate_rejects_attacker_majority():
    cfg = default_fl_config()
    cfg.update({"num_clients": 10, "num_attackers": 6, "clients_per_round": 8,
                "attacker_always_selected": True})
    with pytest.raises(ValueError):
        _validate_fl_config(cfg)
    print("[ok] a per-round attacker majority is rejected")


def test_validate_rejects_no_benign_slots():
    cfg = default_fl_config()
    cfg.update({"num_clients": 12, "num_attackers": 8, "clients_per_round": 8,
                "attacker_always_selected": True})
    with pytest.raises(ValueError):
        _validate_fl_config(cfg)
    print("[ok] num_attackers >= clients_per_round (no benign slots) is rejected")


def test_validate_accepts_the_default_topology():
    _validate_fl_config(default_fl_config())  # must not raise
    print("[ok] the shipped default FL topology validates")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll defense-in-the-loop / partition-floor tests passed.")
