from types import SimpleNamespace

import pytest
import torch

from tcaa.fl_runner import _require_finite_global
from tcaa.training_core import (
    _finite_optimizer_step,
    _resolve_generation_eos_ids,
    _tokenizer_load_kwargs,
    default_config,
    smoke_overrides,
)
from tcaa.metrics import teacher_forced_ppl


def test_model_generation_config_propagates_multi_eos_and_keeps_tokenizer_primary():
    inner = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[151645, 151643]),
        config=SimpleNamespace(eos_token_id=151645),
    )
    model = SimpleNamespace(inner=lambda: inner)
    tokenizer = SimpleNamespace(eos_token_id=151645)

    assert _resolve_generation_eos_ids(model, tokenizer) == [151645, 151643]


def test_tokenizer_primary_is_added_when_generation_config_omits_it():
    inner = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=20),
        config=SimpleNamespace(eos_token_id=20),
    )
    model = SimpleNamespace(inner=lambda: inner)
    tokenizer = SimpleNamespace(eos_token_id=10)

    assert _resolve_generation_eos_ids(model, tokenizer) == [10, 20]


def test_strict_schema_contains_runtime_generation_and_smoke_keys():
    defaults = default_config()
    assert "gen_batch_size" in defaults
    assert "use_tf32" in defaults
    assert set(smoke_overrides()).issubset(defaults)


def test_tokenizer_uses_the_same_immutable_revision_as_model_weights():
    revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert _tokenizer_load_kwargs({"model_revision": revision}) == {
        "revision": revision
    }
    assert _tokenizer_load_kwargs({"model_revision": None}) == {}


def test_optimizer_step_fails_closed_on_nonfinite_objective():
    layer = torch.nn.Linear(2, 2)
    model = SimpleNamespace(inner=lambda: layer)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    finite = layer(torch.ones(1, 2)).square().mean()
    assert _finite_optimizer_step(model, opt, finite, 1.0) >= 0.0

    nonfinite = layer(torch.ones(1, 2)).sum() * torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        _finite_optimizer_step(model, opt, nonfinite, 1.0)


def test_perplexity_fails_closed_instead_of_skipping_nonfinite_batch():
    class Model:
        def forward(self, input_ids, attention_mask):
            return torch.full((*input_ids.shape, 8), float("nan"))

    batch = {
        "input_ids": torch.tensor([[2, 3, 4]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 3, 4]]),
    }
    with pytest.raises(FloatingPointError, match="perplexity batch 0"):
        teacher_forced_ppl(Model(), [batch], torch.device("cpu"))


def test_perplexity_rejects_empty_or_unscored_batches():
    class Model:
        def forward(self, input_ids, attention_mask):
            return torch.zeros(*input_ids.shape, 8)

    with pytest.raises(ValueError, match="at least one non-empty batch"):
        teacher_forced_ppl(Model(), [], torch.device("cpu"))
    unscored = {
        "input_ids": torch.tensor([[2, 3, 4]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.full((1, 3), -100),
    }
    with pytest.raises(ValueError, match="no scored target tokens"):
        teacher_forced_ppl(Model(), [unscored], torch.device("cpu"))


def test_fl_global_finiteness_gate_reports_trajectory_and_round():
    _require_finite_global(torch.zeros(4), label="attacked", round_index=3)
    with pytest.raises(FloatingPointError, match="attacked global.*round 3"):
        _require_finite_global(
            torch.tensor([0.0, float("inf")]), label="attacked", round_index=3
        )
