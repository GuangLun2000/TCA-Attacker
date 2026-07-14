import math
from unittest.mock import patch

import pytest
import torch

from tcaa.generation_safety import (
    WallClockStoppingCriteria,
    build_stopping_criteria,
    validate_generation_limits,
)
from tcaa.phase0_runner import _validate_experiment_config, default_config


@pytest.mark.parametrize("value", [None, 0, -1, True, 2.5])
def test_generation_requires_a_positive_integer_token_cap(value):
    with pytest.raises(ValueError):
        validate_generation_limits(value)  # type: ignore[arg-type]


def test_generation_hard_cap_requires_an_explicit_override():
    validate_generation_limits(2048, hard_token_cap=4096, max_batch_seconds=600)
    with pytest.raises(ValueError, match="exceeds"):
        validate_generation_limits(8192, hard_token_cap=4096)


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan, "bad"])
def test_wall_clock_limit_must_be_positive_and_finite(seconds):
    with pytest.raises(ValueError):
        validate_generation_limits(1, max_batch_seconds=seconds)


def test_wall_clock_stopping_criterion_is_inspectable():
    criteria, criterion = build_stopping_criteria(1e-12)
    assert criteria is not None and criterion is not None
    assert isinstance(criterion, WallClockStoppingCriteria)
    assert criterion(None, None) is True
    assert criterion.triggered is True
    assert criterion.elapsed_seconds >= 0


def test_time_guard_can_be_disabled_without_disabling_token_cap():
    criteria, criterion = build_stopping_criteria(None)
    assert criteria is None
    assert criterion is None


def test_runner_rejects_an_unbounded_or_accidentally_excessive_budget():
    cfg = default_config()
    _validate_experiment_config(cfg)

    missing_cap = dict(cfg, max_new_tokens=None)
    with pytest.raises(ValueError, match="max_new_tokens"):
        _validate_experiment_config(missing_cap)

    excessive = dict(cfg, max_new_tokens=8192, generation_hard_token_cap=4096)
    with pytest.raises(ValueError, match="exceeds"):
        _validate_experiment_config(excessive)

    oversized_rollout = dict(cfg, onpolicy_horizon=cfg["max_new_tokens"] + 1)
    with pytest.raises(ValueError, match="onpolicy_horizon"):
        _validate_experiment_config(oversized_rollout)


def test_onpolicy_timeout_restores_training_mode_and_fails_explicitly():
    from tcaa.length_surrogate import onpolicy_expected_length

    class Inner:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

    class Wrapper:
        def __init__(self):
            self.module = Inner()

        def inner(self):
            return self.module

    model = Wrapper()
    prompt = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    with patch("tcaa.length_surrogate._greedy_rollout", side_effect=TimeoutError("guard")):
        with pytest.raises(TimeoutError, match="guard"):
            onpolicy_expected_length(
                model, prompt, eos_id=0, pad_id=0, horizon=4,
                device=torch.device("cpu"), max_batch_seconds=1.0,
            )
    assert model.module.training is True
