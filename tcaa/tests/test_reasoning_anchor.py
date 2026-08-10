from types import SimpleNamespace

import pytest
import torch

from tcaa.reasoning import select_reasoning_anchors


def _example(uid):
    return SimpleNamespace(uid=uid, prompt_ids=[10, 11, 12])


def _diagnostic(**changes):
    row = {
        "reasoning_closed": True,
        "hit_cap": False,
        "time_limited": False,
        "answer_correct": True,
    }
    row.update(changes)
    return row


def _select(clean, tau, clean_costs, tau_costs, clean_diag, tau_diag, **changes):
    config = {
        "requested_size": 2,
        "minimum_valid": 2,
        "reference_horizon": 5,
        "training_horizon": 10,
        "target_ratio": 2.0,
        "c_f": 1.0,
        "c_a": 0.0,
        "require_correct": True,
    }
    config.update(changes)
    return select_reasoning_anchors(
        clean,
        tau,
        clean_costs,
        tau_costs,
        clean_diag,
        tau_diag,
        **config,
    )


def test_anchor_preflight_filters_all_reasons_and_preserves_row_alignment():
    clean = [_example(f"task-{idx}") for idx in range(7)]
    tau = [_example(f"task-{idx}") for idx in range(7)]
    clean_costs = torch.tensor([10, 11, 12, 13, 14, 15, 16], dtype=torch.float64)
    tau_costs = torch.tensor([4, 4, 4, 4, 4, 5.1, 3], dtype=torch.float64)
    clean_diag = [_diagnostic() for _ in range(7)]
    tau_diag = [_diagnostic() for _ in range(7)]
    tau_diag[1] = _diagnostic(reasoning_closed=False)
    clean_diag[2] = _diagnostic(reasoning_closed=False, hit_cap=True)
    tau_diag[3] = _diagnostic(time_limited=True, answer_correct=False)
    clean_diag[4] = _diagnostic(answer_correct=False)

    selected, preflight = _select(
        clean, tau, clean_costs, tau_costs, clean_diag, tau_diag
    )

    assert selected == [0, 6]
    assert [tau[idx].uid for idx in selected] == ["task-0", "task-6"]
    assert clean_costs[selected].tolist() == [10.0, 16.0]
    assert tau_costs[selected].tolist() == [4.0, 3.0]
    assert preflight == {
        "candidate_count": 7,
        "valid_count": 2,
        "selected_count": 2,
        "valid_fraction": pytest.approx(2 / 7, abs=5e-5),
        "minimum_required": 2,
        "reference_horizon": 5,
        "training_horizon": 10,
        "target_ratio": 2.0,
        "require_correct": True,
        "exclusions": {
            "tau_not_closed": 1,
            "clean_not_closed": 1,
            "cap_hit": 1,
            "time_limited": 1,
            "tau_answer_incorrect": 1,
            "clean_answer_incorrect": 1,
            "target_unattainable": 1,
        },
        "selected_uids": ["task-0", "task-6"],
    }


def test_anchor_preflight_fails_closed_below_minimum_valid_count():
    clean = [_example("a"), _example("b")]
    tau = [_example("a"), _example("b")]
    clean_diag = [_diagnostic(), _diagnostic()]
    tau_diag = [_diagnostic(), _diagnostic(reasoning_closed=False)]

    with pytest.raises(
        ValueError,
        match=r"retained only 1/2.*reasoning_min_valid_anchors=2.*tau_not_closed",
    ):
        _select(clean, tau, [1.0, 1.0], [1.0, 1.0], clean_diag, tau_diag)


def test_anchor_attainability_tolerance_is_inclusive_and_correctness_is_optional():
    clean = [_example("boundary"), _example("above")]
    tau = [_example("boundary"), _example("above")]
    clean_diag = [
        _diagnostic(answer_correct=False),
        _diagnostic(answer_correct=None),
    ]
    tau_diag = [
        _diagnostic(answer_correct=False),
        _diagnostic(answer_correct=None),
    ]
    maximum_cost = 10.0
    boundary_reference = maximum_cost * (1.0 + 1e-6) / 2.0

    selected, preflight = _select(
        clean,
        tau,
        [1.0, 1.0],
        [boundary_reference, boundary_reference + 1e-8],
        clean_diag,
        tau_diag,
        requested_size=1,
        minimum_valid=1,
        require_correct=False,
    )

    assert selected == [0]
    assert preflight["valid_count"] == 1
    assert preflight["exclusions"] == {"target_unattainable": 1}


def test_anchor_preflight_rejects_misaligned_rows_and_uid_asymmetry():
    clean = [_example("a"), _example("b")]
    tau = [_example("a"), _example("b")]
    diagnostics = [_diagnostic(), _diagnostic()]

    with pytest.raises(ValueError, match="exactly row-aligned"):
        _select(clean, tau, [1.0, 1.0], [1.0, 1.0], diagnostics, diagnostics[:1])

    clean[0].uid = None
    with pytest.raises(ValueError, match="UID-paired"):
        _select(clean, tau, [1.0, 1.0], [1.0, 1.0], diagnostics, diagnostics)


def test_anchor_preflight_requires_complete_diagnostics():
    clean = [_example("a")]
    tau = [_example("a")]
    incomplete = {"reasoning_closed": True, "hit_cap": False}

    with pytest.raises(ValueError, match="missing.*time_limited"):
        _select(
            clean,
            tau,
            [1.0],
            [1.0],
            [incomplete],
            [_diagnostic()],
            requested_size=1,
            minimum_valid=1,
        )
