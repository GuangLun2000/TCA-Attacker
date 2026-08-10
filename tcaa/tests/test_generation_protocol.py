from types import SimpleNamespace

import pytest
import torch

from tcaa.training_core import (
    _finite_optimizer_step,
    _resolve_generation_eos_ids,
    _tokenizer_load_kwargs,
    default_config,
    smoke_overrides,
)


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
