from types import SimpleNamespace

import pytest

from tcaa.fl_runner import default_fl_config, validate_fl_config
from tcaa.training_core import (
    _validate_experiment_config,
    default_config,
    enable_backend_speedups,
    validate_experiment_config,
)


def _reasoning_config():
    cfg = default_config()
    cfg.update({
        "attack_objective": "reasoning_cost",
        "backbone": "tiny-gpt2",
        "source": "synthetic",
        "max_new_tokens": 24,
        "reasoning_horizon": 16,
        "reasoning_reference_horizon": 8,
        "reasoning_anchor_size": 8,
        "reasoning_reference_mode": "gold_trace",
    })
    return cfg


def test_backend_speedups_explicitly_reset_process_global_tf32_flags(monkeypatch):
    precision_calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=True),
            ),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        set_float32_matmul_precision=precision_calls.append,
    )
    monkeypatch.setattr("tcaa.training_core.torch", fake_torch)

    enable_backend_speedups({"use_tf32": False})
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    assert precision_calls[-1] == "highest"

    enable_backend_speedups({"use_tf32": True})
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.allow_tf32 is True
    assert precision_calls[-1] == "high"


def test_valid_synthetic_reasoning_config():
    _validate_experiment_config(_reasoning_config())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"reasoning_horizon": 25}, "cannot exceed"),
        ({"reasoning_reference_horizon": 17}, "cannot exceed"),
        ({"reasoning_target_ratio": 0.9}, "reasoning_target_ratio"),
        ({"reasoning_target_ratio": 1.0}, "reasoning_target_ratio"),
        ({"reasoning_target_tolerance": -0.1}, "reasoning_target_tolerance"),
        ({"eval_decode_seeds": []}, "eval_decode_seeds"),
        ({"eval_decode_seeds": [11, 11]}, "unique"),
        ({"reasoning_reference_mode": "unknown"}, "reasoning_reference_mode"),
        ({"reasoning_rollout_temperature": 0.0}, "temperature"),
        ({"reasoning_rollout_top_p": 1.1}, "top_p"),
        ({"reasoning_rollout_top_k": -1}, "top_k"),
        ({"source": "alpaca"}, "requires source"),
        ({"reasoning_min_reference_accuracy": 1.1}, "reasoning_min_reference_accuracy"),
        ({"reasoning_min_claim_cost_ratio": 1.0}, "reasoning_min_claim_cost_ratio"),
        ({"reasoning_min_hardware_ratio": float("nan")}, "reasoning_min_hardware_ratio"),
    ],
)
def test_invalid_reasoning_configs_fail_fast(updates, message):
    cfg = _reasoning_config()
    cfg.update(updates)
    with pytest.raises(ValueError, match=message):
        _validate_experiment_config(cfg)


def test_qwen3_reasoning_rejects_greedy_rollout():
    cfg = _reasoning_config()
    cfg.update({
        "backbone": "Qwen/Qwen3-1.7B",
        "source": "gsm8k",
        "thinking_mode": True,
        "reasoning_rollout_do_sample": False,
    })
    with pytest.raises(ValueError, match="sampling"):
        _validate_experiment_config(cfg)


def test_claim_bearing_gsm8k_requires_broadcast_reference_and_three_decode_seeds():
    cfg = _reasoning_config()
    cfg.update({
        "backbone": "Qwen/Qwen3-1.7B", "source": "gsm8k", "thinking_mode": True,
        "reasoning_reference_mode": "gold_trace",
    })
    with pytest.raises(ValueError, match="broadcast_generation"):
        _validate_experiment_config(cfg)
    cfg.update(reasoning_reference_mode="broadcast_generation", eval_decode_seeds=[11, 23])
    with pytest.raises(ValueError, match="at least 3"):
        _validate_experiment_config(cfg)
    cfg.update(eval_decode_seeds=[11, 23, 47], reasoning_task_instruction="")
    with pytest.raises(ValueError, match="reasoning_task_instruction"):
        _validate_experiment_config(cfg)


def test_reasoning_rejects_free_decode_until_competing_eos_hazard_exists():
    cfg = _reasoning_config()
    with pytest.raises(ValueError, match="competing EOS hazard"):
        _validate_experiment_config(dict(cfg, reasoning_rollout_free_decode=True))


def test_public_validators_reject_unknown_keys_and_resolve_optional_batches():
    with pytest.raises(ValueError, match="unknown shared config key"):
        validate_experiment_config({"attaker_steps": 1})
    with pytest.raises(ValueError, match="unknown FL config key"):
        validate_fl_config({"gen_bach_size": 1})

    phase = validate_experiment_config({"gen_batch_size": 1, "use_tf32": False})
    assert phase["gen_batch_size"] == 1
    fl = validate_fl_config(dict(default_fl_config(), gen_batch_size=1, use_tf32=False))
    assert fl["gen_batch_size"] == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"attacker_steps": 0}, "attacker_steps"),
        ({"gen_batch_size": 0}, "gen_batch_size"),
        ({"attacker_lr": 0.0}, "attacker_lr"),
        ({"lora_dropout": 1.0}, "lora_dropout"),
        ({"num_attackers": 5}, "smaller than"),
    ],
)
def test_core_optimizer_and_topology_config_fails_fast(updates, message):
    cfg = default_config()
    cfg.update(updates)
    with pytest.raises(ValueError, match=message):
        _validate_experiment_config(cfg)
