import pytest
import torch

from tcaa.length_surrogate import (
    ReasoningRolloutParts,
    onpolicy_expected_reasoning_cost,
    reasoning_tcaa_loss,
)


def _answer_logits(labels: torch.Tensor, vocab: int = 8) -> torch.Tensor:
    logits = torch.zeros(*labels.shape, vocab, requires_grad=True)
    return logits


def test_reasoning_loss_reaches_zero_cost_term_at_target_and_preserves_other_terms():
    labels = torch.tensor([[-100, 3, 0]])
    logits = _answer_logits(labels)
    rollout = ReasoningRolloutParts(
        expected_cost=torch.tensor([20.0], requires_grad=True),
        expected_tokens=torch.tensor([4.0]),
        mean_stop_prob=torch.tensor(0.2),
        repetition_penalty=torch.tensor(0.1),
        closure_rate=torch.tensor(1.0),
    )
    parts = reasoning_tcaa_loss(
        clean_logits=None,
        clean_labels=None,
        answer_logits=logits,
        answer_labels=labels,
        rollout=rollout,
        reference_cost=torch.tensor([10.0]),
        target_ratio=2.0,
        reasoning_cost_weight=3.0,
        repetition_weight=2.0,
    )
    assert float(parts.target_cost_loss) == 0.0
    assert float(parts.rep_term) == pytest.approx(0.2)
    assert torch.isfinite(parts.total)


def test_reasoning_loss_cost_gradient_only_pushes_up_below_target():
    labels = torch.tensor([[-100, 3, 0]])
    logits = _answer_logits(labels)
    expected = torch.tensor([5.0], requires_grad=True)
    rollout = ReasoningRolloutParts(
        expected_cost=expected,
        expected_tokens=torch.tensor([2.0]),
        mean_stop_prob=torch.tensor(0.4),
        repetition_penalty=torch.tensor(0.0),
        closure_rate=torch.tensor(0.0),
    )
    parts = reasoning_tcaa_loss(
        clean_logits=None,
        clean_labels=None,
        answer_logits=logits,
        answer_labels=labels,
        rollout=rollout,
        reference_cost=torch.tensor([10.0]),
        target_ratio=2.0,
        reasoning_cost_weight=1.0,
    )
    parts.total.backward()
    assert expected.grad is not None and float(expected.grad) < 0.0


def test_reasoning_loss_pushes_cost_down_above_upper_band():
    labels = torch.tensor([[-100, 3, 0]])
    logits = _answer_logits(labels)
    expected = torch.tensor([30.0], requires_grad=True)
    rollout = ReasoningRolloutParts(
        expected_cost=expected,
        expected_tokens=torch.tensor([8.0]),
        mean_stop_prob=torch.tensor(0.01),
        repetition_penalty=torch.tensor(0.0),
        closure_rate=torch.tensor(0.0),
    )
    parts = reasoning_tcaa_loss(
        clean_logits=None,
        clean_labels=None,
        answer_logits=logits,
        answer_labels=labels,
        rollout=rollout,
        reference_cost=torch.tensor([10.0]),
        target_ratio=2.0,
        target_tolerance=0.25,
        overshoot_weight=1.0,
        reasoning_cost_weight=1.0,
    )
    parts.total.backward()
    assert expected.grad is not None and float(expected.grad) > 0.0


def test_clean_cost_anchor_is_two_sided_relative_error():
    labels = torch.tensor([[-100, 3, 0]])
    logits = _answer_logits(labels)
    rollout = ReasoningRolloutParts(
        expected_cost=torch.tensor([10.0]),
        expected_tokens=torch.tensor([2.0]),
        mean_stop_prob=torch.tensor(0.2),
        repetition_penalty=torch.tensor(0.0),
        closure_rate=torch.tensor(1.0),
    )
    parts = reasoning_tcaa_loss(
        clean_logits=None,
        clean_labels=None,
        answer_logits=logits,
        answer_labels=labels,
        rollout=rollout,
        reference_cost=torch.tensor([10.0]),
        target_ratio=1.0,
        reasoning_cost_weight=0.0,
        clean_expected_cost=torch.tensor([12.0]),
        clean_reference_cost=torch.tensor([10.0]),
        clean_cost_anchor_weight=2.0,
    )
    assert float(parts.clean_cost_anchor) == pytest.approx(0.4)


def test_reasoning_rollout_treats_secondary_eos_as_terminal():
    class Inner:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs["eos_token_id"] == [0, 6]
            continuation = torch.tensor([[4, 6, 5, 5]], dtype=input_ids.dtype)
            return torch.cat([input_ids, continuation], dim=1)

    class Model:
        def __init__(self):
            self._inner = Inner()

        def inner(self):
            return self._inner

        def forward(self, input_ids, attention_mask):
            return torch.zeros(*input_ids.shape, 8, requires_grad=True)

    rollout = onpolicy_expected_reasoning_cost(
        Model(),
        {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
        reasoning_end_id=7,
        eos_id=0,
        generation_eos_ids=[0, 6],
        pad_id=1,
        horizon=4,
        device=torch.device("cpu"),
        c_f=1.0,
        c_a=1.0,
        free_decode=True,
    )

    assert rollout.span_lengths.tolist() == [1]
    assert float(rollout.closure_rate) == 0.0
