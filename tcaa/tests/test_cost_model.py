from types import SimpleNamespace

import pytest
import torch

from tcaa.cost_model import (
    CostStats,
    KVCacheGeometry,
    infer_kv_cache_geometry,
    measure_generation,
    theoretical_kv_cache_bytes,
)


class _FixedGenerationModel:
    """Small generate-compatible fake; no transformer weights are needed for accounting."""

    def __init__(self, new_tokens: torch.Tensor):
        self.new_tokens = new_tokens
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=16,
            torch_dtype=torch.float32,
        )

    def eval(self):
        return self

    def generate(self, *, input_ids, **kwargs):
        new = self.new_tokens.to(input_ids.device)
        assert new.shape[0] == input_ids.shape[0]
        criteria = kwargs.get("stopping_criteria")
        if criteria is not None:
            # Model a cooperative check after this decode block.  A tiny time budget in
            # the test makes the inspectable wall-clock criterion fire deterministically.
            criteria(torch.cat((input_ids, new), dim=1), scores=None)
        return torch.cat((input_ids, new), dim=1)


def _batch() -> dict[str, torch.Tensor]:
    # Left-padded width 4; true prompt lengths are 2 and 3.
    return {
        "input_ids": torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
    }


def test_measure_generation_exact_tokens_eos_cap_batch_slots_and_kv():
    # First row emits EOS as its second token; the tokens following it are batch padding.
    # Second row consumes the full four-token generation cap without EOS.
    model = _FixedGenerationModel(torch.tensor([[7, 2, 0, 0], [8, 9, 10, 11]]))
    stats = measure_generation(
        model,
        [_batch()],
        eos_id=2,
        pad_id=0,
        max_new_tokens=4,
        device=torch.device("cpu"),
    )

    assert stats.prompt_lens == [2, 3]
    assert stats.output_lens == [2, 4]  # first EOS is included in realized L
    assert stats.eos_emitted_flags == [True, False]
    assert stats.truncated_flags == [False, True]
    assert stats.time_limited_flags == [False, False]
    assert stats.termination_reasons == ["eos", "max_new_tokens"]

    summary = stats.summary()
    assert summary["total_input_tokens"] == 5
    assert summary["total_output_tokens"] == 6
    assert summary["total_tokens"] == 11
    assert summary["output_len_p50"] == 3.0
    assert summary["output_len_p90"] == 3.8
    assert summary["output_len_p95"] == 3.9
    assert summary["output_len_p99"] == 3.98
    assert summary["eos_emitted_count"] == 1
    assert summary["eos_emitted_rate"] == 0.5
    assert summary["cap_hit_count"] == 1
    assert summary["cap_hit_rate"] == 0.5
    assert summary["budget_utilization"] == 0.75
    assert summary["validity"]["status"] == "right_censored_lower_bound"
    assert summary["validity"]["censoring_types"] == ["token_cap"]
    assert summary["validity"]["output_tokens_include_first_eos"] is True

    # Prompt tensor slots: 2 rows * padded width 4 versus 2+3 logical tokens.
    assert summary["logical_prompt_slots"] == 5
    assert summary["padded_prompt_slots"] == 8
    assert summary["prompt_padding_slots"] == 3
    # Decode tensor runs four iterations for both rows, despite row 0 ending at step 2.
    assert summary["logical_decode_slots"] == 6
    assert summary["scheduled_decode_slots"] == 8
    assert summary["decode_padding_slots"] == 2
    assert summary["decode_scheduling_overhead_ratio"] == pytest.approx(8 / 6)

    # Prior-token pair formulas: sum n(n-1)/2 and sum(nL + L(L-1)/2).
    assert summary["prefill_attention_pairs"] == 4
    assert summary["decode_attention_pairs"] == 23

    # 2(K,V) * 2 layers * 2 KV heads * head_dim 4 * fp32 = 128 bytes/token.
    assert summary["kv_cache_bytes_per_token"] == 128
    assert summary["mean_kv_cache_bytes"] == 704.0  # mean((2+2), (3+4))*128
    assert summary["peak_request_kv_cache_bytes"] == 896
    assert summary["estimated_peak_batch_kv_bytes"] == 2048  # 2*(4 prompt+4 gen)*128

    prompt_rows = stats.per_prompt_records()
    assert prompt_rows[0]["output_tokens"] == 2
    assert prompt_rows[0]["emitted_eos"] is True
    assert prompt_rows[0]["hit_cap"] is False
    assert prompt_rows[1]["total_tokens"] == 7
    assert prompt_rows[1]["hit_cap"] is True
    assert prompt_rows[1]["censored"] is True
    assert prompt_rows[1]["kv_cache_bytes"] == 896

    batch_rows = stats.per_batch_records()
    assert batch_rows == [{
        "batch_index": 0,
        "batch_size": 2,
        "logical_prompt_slots": 5,
        "padded_prompt_slots": 8,
        "logical_decode_slots": 6,
        "scheduled_decode_slots": 8,
        "decode_padding_slots": 2,
        "estimated_peak_kv_bytes": 2048,
        "time_limit_triggered": False,
        "generation_elapsed_seconds": None,
    }]


def test_time_limit_is_separate_right_censoring_not_a_cap_hit():
    model = _FixedGenerationModel(torch.tensor([[7, 8], [9, 10]]))
    stats = measure_generation(
        model,
        [_batch()],
        eos_id=2,
        pad_id=0,
        max_new_tokens=8,
        generation_max_batch_seconds=1e-12,
        device=torch.device("cpu"),
    )

    assert stats.n_time_limited == 2
    assert stats.n_truncated == 0
    assert stats.time_limited_flags == [True, True]
    assert stats.truncated_flags == [False, False]
    assert stats.termination_reasons == ["time_limit", "time_limit"]
    summary = stats.summary()
    assert summary["time_limited_count"] == 2
    assert summary["time_limit_rate"] == 1.0
    assert summary["cap_hit_count"] == 0
    assert summary["validity"]["censoring_types"] == ["wall_clock"]
    assert stats.per_batch_records()[0]["time_limit_triggered"] is True


@pytest.mark.parametrize("cap", [None, 0, -1, True, 2.5, float("inf"), float("nan")])
def test_measure_generation_rejects_missing_noninteger_or_nonfinite_cap(cap):
    model = _FixedGenerationModel(torch.tensor([[2], [2]]))
    with pytest.raises(ValueError, match="max_new_tokens"):
        measure_generation(
            model,
            [],
            eos_id=2,
            pad_id=0,
            max_new_tokens=cap,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )


def test_kv_geometry_infers_gqa_and_accepts_explicit_geometry():
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=32,
        torch_dtype=torch.bfloat16,
    )
    geometry = infer_kv_cache_geometry(config)
    assert geometry == KVCacheGeometry(3, 2, 4, 2)
    assert geometry.bytes_per_token == 96
    assert theoretical_kv_cache_bytes(10, batch_size=3, model_or_config=config) == 2880

    explicit = KVCacheGeometry(2, 1, 8, 4)
    assert theoretical_kv_cache_bytes(7, batch_size=2, geometry=explicit) == 1792
    assert theoretical_kv_cache_bytes(7, model_or_config=SimpleNamespace()) is None


def test_empty_stats_are_finite_and_explicitly_have_no_observations():
    stats = CostStats()
    summary = stats.summary()
    for key in (
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "output_len_p50",
        "output_len_p90",
        "output_len_p95",
        "output_len_p99",
        "cap_hit_count",
        "cap_hit_rate",
        "budget_utilization",
        "padded_prompt_slots",
        "scheduled_decode_slots",
    ):
        assert summary[key] == 0
    assert summary["validity"]["status"] == "no_observations"
    assert summary["validity"]["output_length_is_lower_bound"] is False
    assert summary["mean_kv_cache_bytes"] is None
    assert stats.per_prompt_records() == []
    assert stats.per_batch_records() == []


def test_legacy_positional_coststats_fields_keep_their_original_order():
    # New resource fields must stay appended after the original public dataclass fields.
    stats = CostStats(
        1, [2], [3], [12.0], [5.0], [0.0], [3.0], [1.0], [False],
        [], [], 0, 4, 1.0, 1.0,
    )
    assert stats.max_new_tokens == 4
    assert stats.c_f == 1.0 and stats.c_a == 1.0
    assert stats.summary()["total_tokens"] == 5
