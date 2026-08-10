from types import SimpleNamespace

import pytest
import torch

from tcaa.cost_model import CostStats
from tcaa.gen_data import GenExample
from tcaa.fl_runner import (
    _aggregate_hardware_runs,
    _build_objective_summary,
    _logical_resource_bundle,
    _logical_rows,
    _normalise_hardware_profile,
    _paired_hardware_ratio_summaries,
    _profile_final_resources,
    _resource_summary_rows,
    _write_csv,
    run_fl_seeds,
    _validate_fl_config,
    default_fl_config,
)


def test_multiseed_provenance_arrays_stay_aligned_after_a_failed_seed(
    monkeypatch, tmp_path
):
    import tcaa.fl_runner as fl_runner

    def fake_run(config):
        seed = config["seed"]
        if seed == 22:
            raise RuntimeError("synthetic failure")
        return {
            "run_id": f"run-{seed}",
            "artifacts_dir": f"/artifacts/{seed}",
            "requested_config_sha256": f"requested-{seed}",
            "resolved_config_sha256": f"resolved-{seed}",
            "shard_profile": {"sizes": [4, 4], "empty_shards": 0},
            "durability": [{}],
            "stealth_trace": [],
        }

    monkeypatch.setattr(fl_runner, "run_fl", fake_run)
    summary = run_fl_seeds(
        {"results_root": str(tmp_path), "results_subdir": "reasoning-multiseed"},
        [11, 22, 33],
    )

    assert summary["seeds_completed"] == [11, 33]
    assert summary["run_ids"] == ["run-11", "run-33"]
    assert summary["artifacts_dirs"] == ["/artifacts/11", "/artifacts/33"]
    assert summary["requested_config_sha256s"] == [
        "requested-11",
        "requested-33",
    ]
    assert summary["resolved_config_sha256s"] == ["resolved-11", "resolved-33"]
    assert summary["completed_runs"] == [
        {
            "seed": 11,
            "run_id": "run-11",
            "artifacts_dir": "/artifacts/11",
            "requested_config_sha256": "requested-11",
            "resolved_config_sha256": "resolved-11",
        },
        {
            "seed": 33,
            "run_id": "run-33",
            "artifacts_dir": "/artifacts/33",
            "requested_config_sha256": "requested-33",
            "resolved_config_sha256": "resolved-33",
        },
    ]
    assert summary["failures"] == [
        {"seed": 22, "error": "RuntimeError: synthetic failure"}
    ]


def _stats(output_lens, *, cap=8, hit_cap=None, timed=None):
    n = len(output_lens)
    hit_cap = list(hit_cap or [False] * n)
    timed = list(timed or [False] * n)
    eos = [not c and not t for c, t in zip(hit_cap, timed)]
    return CostStats(
        n_prompts=n,
        prompt_lens=[3] * n,
        output_lens=list(output_lens),
        costs=[float(v) for v in output_lens],
        kv_proxies=[float(3 + v) for v in output_lens],
        repetitions=[0.0] * n,
        effective_lens=[float(v) for v in output_lens],
        distinct_ratios=[1.0] * n,
        truncated_flags=hit_cap,
        eos_emitted_flags=eos,
        time_limited_flags=timed,
        termination_reasons=[
            "max_new_tokens" if c else "time_limit" if t else "eos"
            for c, t in zip(hit_cap, timed)
        ],
        n_truncated=sum(hit_cap),
        n_time_limited=sum(timed),
        max_new_tokens=cap,
    )


def test_logical_bundle_keeps_exact_token_ratios_and_censoring_semantics():
    pristine = _stats([2, 2])
    attacked = _stats([8, 4], hit_cap=[True, False])
    benign = _stats([3, 3])

    bundle = _logical_resource_bundle(
        pristine, pristine, attacked, attacked, benign, benign
    )

    assert bundle["states"]["attacked_final"]["logical"]["total_output_tokens"] == 12
    assert bundle["comparisons"]["total_output_tokens"]["attacked_vs_pristine"] == 3.0
    assert bundle["comparisons"]["total_output_tokens"]["attacked_vs_benign"] == 2.0


def test_per_prompt_export_does_not_mislabel_time_censoring_as_eos_or_cap():
    stats = _stats([3], timed=[True])
    examples = [SimpleNamespace(prompt_ids=[10, 11, 12])]

    row = _logical_rows(
        stats, examples, condition="attacked_final", split="tau", round_idx=9
    )[0]

    assert row["time_limited"] is True
    assert row["hit_cap"] is False
    assert row["emitted_eos"] is False
    assert row["termination_reason"] == "time_limit"
    assert len(row["prompt_sha256"]) == 64


class _FakeProfile:
    def to_dict(self):
        return {
            "summary": {
                "requests": 2,
                "input_tokens": 6,
                "output_tokens": 12,
                "total_tokens": 18,
                "wall_seconds": 2.0,
                "cuda_seconds": 1.8,
                "energy_joules": 40.0,
                "requests_per_second": 1.0,
                "output_tokens_per_second": 6.0,
                "peak_allocated_delta_bytes": 100,
                "peak_reserved_delta_bytes": 120,
                "cuda_coverage": 1.0,
                "energy_coverage": 1.0,
                "timed_out_batches": 0,
            },
            "records": [{
                "completed": True,
                "wall_seconds": 2.0,
                "peak_allocated_bytes": 1000,
                "peak_reserved_bytes": 1200,
                "energy_method": "nvml_total_energy_counter",
                "energy_scope": "whole_gpu_board",
                "mean_power_watts": 20.0,
                "peak_power_watts": 22.0,
            }],
            "completed_batches": 1,
            "planned_batches": 1,
            "stopped_early": False,
            "stop_reason": None,
        }


def test_hardware_profile_normalization_and_repeat_median():
    run, batches = _normalise_hardware_profile(
        _FakeProfile(), condition="attacked_final", split="tau", batch_size=8, repeat=0,
        e2e_wall_seconds=2.2,
    )
    assert run["valid"] is True
    assert run["generation_wall_seconds"] == 2.0
    assert run["e2e_wall_seconds"] == 2.2
    assert run["joules_per_output_token"] == pytest.approx(40 / 12)
    assert batches[0]["condition"] == "attacked_final"

    slower = dict(run, repeat=1, generation_wall_seconds=4.0)
    aggregate = _aggregate_hardware_runs([run, slower])
    assert aggregate["valid_repeats"] == 2
    assert aggregate["generation_wall_seconds"] == 3.0
    assert aggregate["generation_wall_seconds_p25"] == 2.5
    assert aggregate["generation_wall_seconds_p75"] == 3.5
    assert aggregate["e2e_wall_seconds"] == 2.2
    assert aggregate["cuda_coverage"] == 1.0

    partial = _aggregate_hardware_runs([
        run,
        dict(slower, valid=False, invalid_reason="generation_max_batch_seconds_exceeded",
             timed_out_batches=1),
    ])
    assert partial["valid"] is False
    assert partial["usable"] is True
    assert partial["timed_out_batches"] == 1

    mixed_energy = _aggregate_hardware_runs([
        run,
        dict(run, repeat=1, energy_method="power_integration", energy_joules=41.0),
    ])
    assert mixed_energy["energy_methods_comparable"] is False
    assert mixed_energy["energy_joules"] is None
    assert mixed_energy["energy_method"].startswith("mixed_not_aggregated:")


def test_fl_resource_profile_configuration_is_finite_and_consistent():
    cfg = default_fl_config()
    _validate_fl_config(cfg)

    with pytest.raises(ValueError, match="non-empty"):
        _validate_fl_config(dict(cfg, resource_profile_batch_sizes=[]))
    with pytest.raises(ValueError, match="requires"):
        _validate_fl_config(dict(cfg, profile_hardware=True, collect_resource_metrics=False))
    with pytest.raises(ValueError, match="wall guard"):
        _validate_fl_config(dict(cfg, profile_hardware=True, generation_max_batch_seconds=None))
    with pytest.raises(ValueError, match="sample_interval"):
        _validate_fl_config(dict(cfg, resource_profile_sample_interval_ms=0))


def test_objective_summary_keeps_consumption_stealth_and_defenses_separate():
    summary = _build_objective_summary({
        "durability": [{
            "amp_tau_effective": 4.0, "ppl_ratio_vs_pristine": 1.02,
            "repetition_tau": 0.1,
        }],
        "stealth_trace": [
            {"n_attackers": 1, "jointly_satisfied": True},
            {"n_attackers": 1, "jointly_satisfied": False},
        ],
        "resources": {
            "comparisons": {"total_output_tokens": {"attacked_vs_pristine": 6.0}},
            "states": {"attacked_final": {"logical": {"total_output_tokens": 120}}},
            "validity": {"hardware": "valid"},
            "environment": {
                "start": {"fingerprint_sha256": "same", "torch": {
                    "kernel_preflight": {"success": True},
                }},
                "end": {"fingerprint_sha256": "same"},
                "environment_changed": False,
            },
        },
        "defense_evaluation": {"telemetry_defenses": {"defenses": {
            "krum": {"caught_rate": 0.25, "survival_rate": 0.75},
        }}},
    })

    assert summary["resource_amplification"]["total_output_tokens"][
        "attacked_vs_pristine"
    ] == 6.0
    assert summary["parameter_stealth"]["joint_pass_rate"] == 0.5
    assert summary["defense_evasion"]["krum"]["caught_rate"] == 0.25
    assert "composite" in summary["note"]


def test_reasoning_objective_v2_carries_independent_gates_and_paired_ci():
    paired = {
        "n_pairs": 8, "median_ratio": 2.1,
        "ci95_lower": 1.7, "ci95_upper": 2.5,
    }
    summary = _build_objective_summary({
        "config": {"attack_objective": "reasoning_cost"},
        "durability": [{
            "paired_reasoning_cost_tau": paired,
            "paired_reasoning_cost_clean": dict(paired, median_ratio=1.01),
            "answer_accuracy_tau_atk": 0.9,
            "answer_accuracy_tau_ref": 0.91,
            "reasoning_closure_tau": 0.98,
            "reasoning_gates": {"cost_ci_lower_gt_1": True},
            "reasoning_claim_ready": False,
        }],
    })
    assert summary["schema_version"] == "objective-v2"
    reasoning = summary["reasoning_attack_final"]
    assert reasoning["paired_cost_ratio"]["ci95_lower"] == 1.7
    assert reasoning["gates"]["cost_ci_lower_gt_1"] is True
    assert reasoning["claim_ready"] is False
    assert reasoning["hardware_evidence_ready"] is False
    assert reasoning["full_resource_claim_ready"] is False


def test_full_reasoning_resource_claim_needs_measured_wall_and_cuda_gain():
    paired = {
        "n_pairs": 8, "median_ratio": 1.4,
        "ci95_lower": 1.2, "ci95_upper": 1.6,
    }
    base = {
        "config": {
            "attack_objective": "reasoning_cost",
            "reasoning_min_hardware_ratio": 1.05,
        },
        "durability": [{
            "paired_reasoning_cost_tau": paired,
            "paired_reasoning_cost_clean": dict(paired, median_ratio=1.0),
            "reasoning_gates": {"measurement_valid": True},
            "reasoning_claim_ready": True,
        }],
        "resources": {
            "states": {"benign_final": {}, "attacked_final": {}},
            "validity": {"hardware": "valid"},
            "environment": {
                "start": {
                    "fingerprint_sha256": "stable-environment",
                    "torch": {"kernel_preflight": {"success": True}},
                },
                "end": {"fingerprint_sha256": "stable-environment"},
                "environment_changed": False,
            },
            "comparisons": {
                "generation_wall_seconds": {"attacked_vs_benign": 1.2},
                "cuda_elapsed_seconds": {"attacked_vs_benign": 1.0},
            },
            "hardware_paired_ratios": {
                "generation_wall_seconds": {
                    "attacked_vs_benign": {"ci95_lower": 1.2},
                },
                "cuda_elapsed_seconds": {
                    "attacked_vs_benign": {"ci95_lower": 1.0},
                },
            },
        },
    }
    summary = _build_objective_summary(base)
    reasoning = summary["reasoning_attack_final"]
    assert reasoning["hardware_evidence_ready"] is True
    assert reasoning["hardware_amplification"]["observed"] is False
    assert reasoning["full_resource_claim_ready"] is False

    base["resources"]["hardware_paired_ratios"]["cuda_elapsed_seconds"][
        "attacked_vs_benign"
    ]["ci95_lower"] = 1.1
    reasoning = _build_objective_summary(base)["reasoning_attack_final"]
    assert reasoning["hardware_amplification"]["observed"] is True
    assert reasoning["full_resource_claim_ready"] is True

    base["resources"]["environment"]["environment_changed"] = True
    reasoning = _build_objective_summary(base)["reasoning_attack_final"]
    assert reasoning["environment_evidence_ready"] is False
    assert reasoning["hardware_evidence_ready"] is False
    assert reasoning["full_resource_claim_ready"] is False

    base["resources"]["environment"] = {
        "start": {"torch": {"kernel_preflight": {"success": True}}},
        "end": {}, "environment_changed": None,
    }
    reasoning = _build_objective_summary(base)["reasoning_attack_final"]
    assert reasoning["environment_evidence_ready"] is False
    assert reasoning["full_resource_claim_ready"] is False


def test_hardware_ratios_are_paired_by_repeat_seed_and_prompt_set():
    runs = []
    for repeat, seed in enumerate((11, 23, 47)):
        for condition, wall, cuda in (
            ("benign_final", 2.0, 1.8),
            ("attacked_final", 3.0, 2.7),
        ):
            runs.append({
                "condition": condition, "split": "tau", "batch_size": 1,
                "repeat": repeat, "decode_seed": seed,
                "prompt_subset_sha256": "same", "valid": True,
                "cuda_timing_valid": True, "memory_metrics_valid": True,
                "generation_wall_seconds": wall, "cuda_elapsed_seconds": cuda,
            })
    paired = _paired_hardware_ratio_summaries(runs, split="tau", batch_size=1)
    wall = paired["generation_wall_seconds"]["attacked_vs_benign"]
    cuda = paired["cuda_elapsed_seconds"]["attacked_vs_benign"]
    assert wall["n_pairs"] == 3
    assert wall["ci95_lower"] == pytest.approx(1.5)
    assert cuda["ci95_lower"] == pytest.approx(1.5)
    assert wall["pairing_unit"] == "repeat_same_decode_seed_and_prompt_set"


def test_resource_summary_csv_keeps_full_logical_and_profile_tokens_distinct(tmp_path):
    resources = {
        "states": {
            "attacked_final": {
                "logical": {
                    "total_input_tokens": 1_000,
                    "total_output_tokens": 20_000,
                    "total_tokens": 21_000,
                },
                "clean_logical": {
                    "total_input_tokens": 1_000,
                    "total_output_tokens": 2_000,
                    "total_tokens": 3_000,
                },
                "hardware_profiles": {
                    "tau": {"batch_size_8": {
                        "input_tokens": 250,
                        "output_tokens": 5_000,
                        "total_tokens": 5_250,
                    }},
                },
            },
        },
    }

    tau_row, clean_row = _resource_summary_rows(resources, 8)
    assert tau_row["total_tokens"] == 21_000
    assert tau_row["hardware_total_tokens"] == 5_250
    assert clean_row["total_tokens"] == 3_000
    assert "hardware_total_tokens" not in clean_row

    # Union field names across rows: clean-only profiling must not be dropped merely
    # because the first (tau) row lacks those columns.
    reverse_rows = [clean_row, tau_row]
    path = tmp_path / "resource_summary.csv"
    _write_csv(path, reverse_rows)
    assert "hardware_total_tokens" in path.read_text().splitlines()[0]


def test_final_profiler_integration_builds_batches_and_drops_gpu_outputs(
    tmp_path, monkeypatch
):
    calls = []

    def fake_profile(*args, **kwargs):
        calls.append(kwargs)
        return [], _FakeProfile()

    monkeypatch.setattr("tcaa.resource_metrics.profile_model_generation", fake_profile)

    class Model:
        def set_flat_params(self, value):
            self.value = value

    examples = [
        GenExample(prompt_ids=[1, 2], ref_ids=[3]),
        GenExample(prompt_ids=[4], ref_ids=[5]),
    ]
    cfg = {
        "resource_profile_eval_size": 2,
        "resource_profile_batch_sizes": [2],
        "resource_profile_repeats": 1,
        "resource_profile_warmup_batches": 0,
        "resource_profile_splits": ["tau"],
        "resource_profile_nvml": False,
        "resource_profile_sample_interval_ms": 100,
        "max_new_tokens": 4,
        "generation_max_batch_seconds": 5.0,
    }
    spec = SimpleNamespace(eos_id=0, pad_id=0)
    states = {"pristine": torch.tensor([0.0]), "attacked_final": torch.tensor([1.0])}

    result = _profile_final_resources(
        Model(), states, examples, examples, cfg, spec, torch.device("cpu"), tmp_path
    )

    assert len(result["runs"]) == 2
    assert all(call["retain_outputs"] is False for call in calls)
    assert result["profiles"]["attacked_final"]["tau"]["batch_size_2"][
        "prompt_subset_sha256"
    ]
    assert (tmp_path / "resource_repeats.jsonl").exists()
