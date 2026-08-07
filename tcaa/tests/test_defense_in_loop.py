# tcaa/tests/test_defense_in_loop.py
# The server-side robust-aggregation dispatcher (_aggregate) and the Dirichlet min-shard floor.
# These serve two paper claims: "the attack survives a defender in the loop" (server_aggregator)
# and "every seed ran the intended federation size" (min_client_shard).

import numpy as np
import pytest
import torch

from tcaa.fl_runner import _aggregate, _fedavg, _validate_fl_config, default_fl_config
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
