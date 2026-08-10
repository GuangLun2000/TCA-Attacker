import math

import pytest
import torch

from tcaa.cost_model import CostStats
from tcaa.training_core import _paired_reasoning_summary

from tcaa.reasoning import (
    expected_survival_cost,
    extract_final_numeric_answer,
    extract_gsm8k_gold,
    realized_reasoning_cost,
    paired_log_ratio_summary,
    reasoning_effect_gates,
    split_reasoning_tokens,
    target_ratio_hinge,
    verify_gsm8k,
)


def test_reasoning_effect_gates_have_explicit_strict_and_inclusive_boundaries():
    assert reasoning_effect_gates(1.0, 1.2) == {
        "cost_ci_lower_gt_1": False,
        "cost_ci_lower_meets_effect_size": False,
    }
    assert reasoning_effect_gates(math.nextafter(1.0, math.inf), 1.2) == {
        "cost_ci_lower_gt_1": True,
        "cost_ci_lower_meets_effect_size": False,
    }
    assert reasoning_effect_gates(math.nextafter(1.2, 0.0), 1.2) == {
        "cost_ci_lower_gt_1": True,
        "cost_ci_lower_meets_effect_size": False,
    }
    assert reasoning_effect_gates(1.2, 1.2) == {
        "cost_ci_lower_gt_1": True,
        "cost_ci_lower_meets_effect_size": True,
    }


@pytest.mark.parametrize(
    ("ci95_lower", "minimum_ratio"),
    [
        (float("nan"), 1.2),
        (float("inf"), 1.2),
        (1.1, float("nan")),
        (1.1, float("inf")),
        (0.0, 1.2),
        (1.1, 1.0),
        (True, 1.2),
    ],
)
def test_reasoning_effect_gates_fail_closed_on_invalid_thresholds(
    ci95_lower, minimum_ratio
):
    with pytest.raises(ValueError):
        reasoning_effect_gates(ci95_lower, minimum_ratio)


def test_reasoning_split_closed_and_open_spans():
    closed = split_reasoning_tokens(
        [90, 10, 11, 91, 20, 21], start_ids=[90], end_ids=[91]
    )
    assert closed.reasoning_ids == [10, 11]
    assert closed.answer_ids == [20, 21]
    assert closed.closed is True
    assert closed.reasoning_decode_len == 4

    # Qwen-style templates may put <think> in the prompt, so a generated start marker
    # is optional. An absent end marker is explicitly an unclosed reasoning span.
    opened = split_reasoning_tokens([10, 11, 12], end_ids=[91])
    assert opened.reasoning_ids == [10, 11, 12]
    assert opened.answer_ids == []
    assert opened.closed is False
    assert opened.reasoning_decode_len == 3


def test_expected_cost_matches_closed_form_and_expected_length_special_case():
    p, steps = 0.25, 5
    lp = torch.full((1, steps), p).log()
    mask = torch.ones_like(lp)
    # c_f=1,c_a=0 reduces exactly to the classic survival E[L].
    cost = expected_survival_cost(lp, mask, [7], c_f=1.0, c_a=0.0)
    expected_len = (1.0 - (1.0 - p) ** steps) / p
    assert float(cost) == pytest.approx(expected_len, abs=1e-5)

    # With q=0 every step survives, so the result equals the realized closed form.
    almost_never_stop = torch.full((1, steps), -30.0)
    weighted = expected_survival_cost(
        almost_never_stop, mask, [7], c_f=3.0, c_a=2.0
    )
    assert float(weighted) == pytest.approx(
        realized_reasoning_cost(7, steps, c_f=3.0, c_a=2.0), rel=1e-5
    )


def test_lower_stop_probability_increases_expected_reasoning_cost():
    mask = torch.ones(1, 6)
    high_stop = expected_survival_cost(
        torch.full((1, 6), 0.6).log(), mask, [4], c_f=2.0, c_a=1.0
    )
    low_stop = expected_survival_cost(
        torch.full((1, 6), 0.1).log(), mask, [4], c_f=2.0, c_a=1.0
    )
    assert float(low_stop) > float(high_stop)


def test_target_ratio_hinge_has_no_reward_beyond_target():
    expected = torch.tensor([9.0, 10.0, 15.0], requires_grad=True)
    loss = target_ratio_hinge(expected, 5.0, target_ratio=2.0)
    assert loss.tolist() == pytest.approx([0.01, 0.0, 0.0])
    loss.sum().backward()
    assert expected.grad is not None
    assert float(expected.grad[0]) < 0.0
    assert float(expected.grad[1]) == 0.0
    assert float(expected.grad[2]) == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final answer: 1,234", "1234"),
        ("Therefore, -2.5000", "-2.5"),
        (r"The answer is \\frac{1}{4}", "0.25"),
        ("The final answer is 3/4", "0.75"),
    ],
)
def test_gsm8k_numeric_normalization(text, expected):
    assert extract_final_numeric_answer(text) == expected


def test_gsm8k_parse_and_verifier():
    rationale, gold = extract_gsm8k_gold("First compute 6 * 7.\n#### 42")
    assert rationale == "First compute 6 * 7."
    assert gold == "42"
    assert verify_gsm8k("Final answer: 42.0", gold)
    assert not verify_gsm8k("Final answer: 41", gold)
    with pytest.raises(ValueError, match="####"):
        extract_gsm8k_gold("42")


def test_final_answer_marker_beats_earlier_fraction_and_keeps_final_fraction():
    assert extract_final_numeric_answer(
        "An intermediate ratio is 1/2. Final answer: 5"
    ) == "5"
    assert extract_final_numeric_answer("Work complete. Final answer: 1/2") == "0.5"
    assert verify_gsm8k("A previous ratio is 1/2. Final answer: 5", "5")


def test_paired_log_ratio_is_order_aligned_and_bootstrapped():
    summary = paired_log_ratio_summary(
        [2.0, 4.0, 8.0], [1.0, 2.0, 4.0], seed=7, bootstrap_samples=200
    )
    assert summary["n_pairs"] == 3
    assert summary["median_ratio"] == pytest.approx(2.0)
    assert summary["ci95_lower"] == pytest.approx(2.0)
    assert summary["ci95_upper"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="aligned"):
        paired_log_ratio_summary([1.0], [1.0, 2.0])


def test_paired_reasoning_bootstrap_clusters_decode_seeds_by_task_uid():
    attacked = CostStats(task_records=[
        {"task_uid": "a", "decode_seed": 11, "reasoning_cost": 2.0},
        {"task_uid": "b", "decode_seed": 11, "reasoning_cost": 8.0},
        {"task_uid": "a", "decode_seed": 23, "reasoning_cost": 8.0},
        {"task_uid": "b", "decode_seed": 23, "reasoning_cost": 2.0},
    ])
    reference = CostStats(task_records=[
        {"task_uid": "a", "decode_seed": 11, "reasoning_cost": 1.0},
        {"task_uid": "b", "decode_seed": 11, "reasoning_cost": 2.0},
        {"task_uid": "a", "decode_seed": 23, "reasoning_cost": 2.0},
        {"task_uid": "b", "decode_seed": 23, "reasoning_cost": 1.0},
    ])
    summary = _paired_reasoning_summary(attacked, reference, seed=9)
    assert summary["n_pairs"] == 2
    assert summary["n_decode_samples"] == 4
    assert summary["decode_seeds"] == [11, 23]
    assert summary["pairing_unit"] == "task_uid_geomean_over_decode_seeds"
    # Task a: sqrt(2*4), task b: sqrt(4*2); both prompt-level ratios are sqrt(8).
    assert summary["median_ratio"] == pytest.approx(8 ** 0.5)
