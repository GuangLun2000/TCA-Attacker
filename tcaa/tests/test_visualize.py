from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcaa.visualize import (
    RASTER_DPI,
    _key_summary_lines,
    _use_log_scale_if_needed,
    apply_style,
    fig_fl_durability,
    full_report,
    make_fl_figures,
    make_resource_figures,
    reasoning_effect_summary,
    reasoning_feedback_report,
    reasoning_gate_rows,
    resource_digest,
    save_figure,
    save_resource_figures,
)


def _reasoning_feedback_result():
    gates = {
        "measurement_valid": True,
        "cost_ci_lower_gt_1": True,
        "cost_ci_lower_meets_effect_size": True,
        "tau_reference_accuracy_valid": True,
        "clean_reference_accuracy_valid": True,
        "tau_accuracy_preserved": True,
        "clean_accuracy_preserved": True,
        "clean_cost_stable": True,
        "closure_valid": True,
        "clean_closure_valid": True,
        "cap_hit_valid": True,
        "clean_cap_hit_valid": True,
        "repetition_valid": True,
        "distinct_valid": True,
    }
    return {
        "run_id": "run-1",
        "config": {
            "attack_objective": "reasoning_cost",
            "num_rounds": 1,
            "profile_hardware": False,
            "reasoning_min_claim_cost_ratio": 1.05,
            "reasoning_min_reference_accuracy": 0.5,
            "reasoning_max_accuracy_drop": 0.05,
            "reasoning_clean_cost_tolerance": 0.1,
            "reasoning_min_closure_rate": 0.95,
            "reasoning_max_cap_hit_rate": 0.05,
            "reasoning_max_repetition": 0.2,
            "reasoning_min_distinct_ratio": 0.65,
        },
        "durability": [{
            "round": 0,
            "paired_reasoning_cost_tau": {
                "median_ratio": 1.30, "ci95_lower": 1.12,
                "ci95_upper": 1.48, "n_pairs": 8, "n_decode_samples": 24,
            },
            "paired_reasoning_cost_clean": {"median_ratio": 1.02},
            "reasoning_tokens_tau_ref": 80.0,
            "reasoning_tokens_tau_atk": 108.0,
            "answer_accuracy_tau_ref": 0.75,
            "answer_accuracy_tau_atk": 0.75,
            "answer_accuracy_drop_tau": 0.0,
            "answer_accuracy_clean_ref": 0.75,
            "answer_accuracy_clean_atk": 0.75,
            "answer_accuracy_drop_clean": 0.0,
            "reasoning_closure_tau": 1.0,
            "reasoning_closure_clean": 1.0,
            "cap_hit_rate_tau": 0.0,
            "resource_clean_attacked": {"cap_hit_rate": 0.0},
            "reasoning_repetition_tau": 0.05,
            "reasoning_distinct_tau": 0.90,
            "reasoning_measurement_valid": True,
            "reasoning_gates": gates,
            "reasoning_claim_ready": True,
        }],
        "stealth_trace": [{
            "round": 0, "n_attackers": 1, "jointly_satisfied": True,
            "attacker_distance": 0.5, "d_T": 0.7,
        }],
        "objective_summary": {"reasoning_attack_final": {
            "hardware_resource_effect_claim_ready": False,
            "hardware_amplification": None,
        }},
        "resources": {"validity": {"hardware": "disabled"}},
    }


def test_reasoning_feedback_is_pilot_scoped_and_csv_ready(capsys):
    result = _reasoning_feedback_result()

    rows = reasoning_gate_rows(result)
    summary = reasoning_effect_summary(result, run_tier="pilot")
    report = reasoning_feedback_report(result, run_tier="pilot")

    assert len(rows) == 14
    assert all(row["passed"] for row in rows)
    assert summary["outcome"] == "positive_logical_signal"
    assert summary["logical_resource_effect_ready"] is True
    assert summary["formal_claim_ready"] is False
    assert summary["evidence_scope"] == "exploratory_single_seed_pilot"
    assert "Enable paired hardware profiling" in " ".join(
        summary["recommended_next_actions"]
    )
    assert "median=1.300x" in report
    assert "formal claim ready: False" in report
    assert "[PASS] cost_ci_lower_meets_effect_size" in report
    assert report in capsys.readouterr().out


def test_reasoning_figures_replace_length_objective_figures():
    figures = make_fl_figures(_reasoning_feedback_result())
    try:
        keys = [key for key, _ in figures]
        assert keys[:2] == ["reasoning_cost_effect", "reasoning_gate_status"]
        assert "fl_durability" not in keys
        assert "fl_utility" not in keys
    finally:
        for _, fig in figures:
            plt.close(fig)


def test_reasoning_feedback_prints_paired_hardware_effect(capsys):
    result = _reasoning_feedback_result()
    result["config"]["profile_hardware"] = True
    result["resources"]["validity"]["hardware"] = "valid"
    result["objective_summary"]["reasoning_attack_final"].update({
        "hardware_resource_effect_claim_ready": True,
        "hardware_amplification": {
            "generation_wall_seconds_paired": {
                "median_ratio": 1.20, "ci95_lower": 1.10,
                "ci95_upper": 1.30, "n_pairs": 3,
            },
            "cuda_elapsed_seconds_paired": {
                "median_ratio": 1.18, "ci95_lower": 1.08,
                "ci95_upper": 1.28, "n_pairs": 3,
            },
        },
    })

    report = reasoning_feedback_report(result, run_tier="pilot")

    assert "paired hardware wall: median=1.2x" in report
    assert "paired hardware CUDA: median=1.18x" in report
    assert "effect_ready=True" in report
    capsys.readouterr()


def test_reasoning_key_summary_does_not_fall_back_to_length_claims():
    lines = _key_summary_lines(fl={
        "config": {
            "attack_objective": "reasoning_cost", "backbone": "Qwen/Qwen3-1.7B",
            "source": "gsm8k", "num_clients": 10, "num_rounds": 5,
            "max_new_tokens": 512, "reasoning_min_claim_cost_ratio": 1.2,
        },
        "durability": [{
            "paired_reasoning_cost_tau": {
                "median_ratio": 1.8, "ci95_lower": 1.2,
                "ci95_upper": 2.2, "n_pairs": 64,
            },
            "paired_reasoning_cost_clean": {"median_ratio": 1.01},
            "reasoning_tokens_tau_ref": 80, "reasoning_tokens_tau_atk": 150,
            "answer_accuracy_tau_ref": 0.9, "answer_accuracy_tau_atk": 0.89,
            "answer_accuracy_drop_tau": 0.01, "answer_accuracy_drop_clean": 0.0,
            "reasoning_closure_tau": 0.98, "truncation_tau": 0.0,
            "reasoning_repetition_tau": 0.05, "reasoning_distinct_tau": 0.9,
            "reasoning_measurement_valid": True,
            "reasoning_gates": {"cost_ci_lower_gt_1": True},
            "reasoning_claim_ready": True,
        }],
        "stealth_trace": [{"n_attackers": 2, "jointly_satisfied": True}],
    })
    report = "\n".join(lines)
    assert "objective-v2" in report
    assert "95% paired bootstrap CI [1.200, 2.200]" in report
    assert "logical resource-effect gates passed" in report
    assert "paired hardware not ready" in report
    assert "parameter_stealth_ready=True" in report
    assert "effective" not in report


def _resource_results():
    gib = 1024 ** 3
    return {
        "resources": {
            "schema_version": "resource-v1",
            "environment": {
                "gpu_name_actual": "NVIDIA Test GPU",
                "total_vram_bytes": 24 * gib,
                "driver_version": "999.1",
                "cuda_runtime": "13.0",
                "environment_changed": False,
                "cloud": {"provider": "google_colab", "sku_reported": "G4"},
            },
            "experiment_wall_seconds": 600.0,
            "comparison_profile": {
                "split": "tau", "batch_size": 8,
                "prompt_subset_sha256": "1234567890abcdef",
            },
            "config": {"dtype": "bfloat16", "batch_size": 8},
            "states": {
                "pristine_tau": {
                    "logical": {
                        "num_requests": 32,
                        "total_input_tokens": 3_200,
                        "total_output_tokens": 8_000,
                        "total_tokens": 11_200,
                        "output_len_p50": 240,
                        "output_len_p95": 300,
                        "cap_hit_rate": 0.0,
                        "budget_utilization": 0.122,
                    },
                    "hardware": {
                        "generation_wall_seconds": 10.0,
                        "cuda_elapsed_seconds": 9.5,
                        "e2e_wall_seconds": 10.8,
                        "requests_per_second": 3.2,
                        "output_tokens_per_second": 800.0,
                        "peak_allocated_bytes": 8 * gib,
                        "incremental_peak_allocated_bytes": 2 * gib,
                        "energy_joules": 2_000.0,
                        "energy_method": "nvml-counter",
                    },
                },
                "benign_final_tau": {
                    "logical": {
                        "num_requests": 32,
                        "total_input_tokens": 3_200,
                        "total_output_tokens": 10_000,
                        "total_tokens": 13_200,
                        "output_len_p50": 300,
                        "output_len_p95": 420,
                        "cap_hit_rate": 0.03,
                        "budget_utilization": 0.153,
                    },
                    # Exercise the profiler's nested HardwareProfile.to_dict shape.
                    "hardware": {
                        "batch_size_1": {"summary": {
                            "wall_seconds": 18.0,
                            "cuda_seconds": None,
                            "requests": 32,
                            "output_tokens": 10_000,
                            "requests_per_second": 1.78,
                            "output_tokens_per_second": 555.6,
                            "peak_allocated_delta_bytes": 1.5 * gib,
                            "peak_reserved_delta_bytes": 2.0 * gib,
                            "energy_joules": None,
                            "cuda_coverage": 0.0,
                            "energy_coverage": 0.0,
                            "generation_wall_seconds_p25": 17.5,
                            "generation_wall_seconds_p75": 18.5,
                        }, "records": [{"energy_method": "unsupported"}]},
                        "batch_size_8": {"summary": {
                            "wall_seconds": 12.0,
                            "cuda_seconds": None,
                            "requests": 32,
                            "output_tokens": 10_000,
                            "requests_per_second": 2.67,
                            "output_tokens_per_second": 833.3,
                            "peak_allocated_delta_bytes": 2.5 * gib,
                            "energy_joules": None,
                            "energy_method": "unsupported",
                        }},
                    },
                },
                "attacked_final_tau": {
                    "logical": {
                        "num_requests": 32,
                        "total_input_tokens": 3_200,
                        "total_output_tokens": 60_000,
                        "total_tokens": 63_200,
                        "output_len_p50": 2_000,
                        "output_len_p95": 2_048,
                        "cap_hit_rate": 0.94,
                        "budget_utilization": 0.916,
                    },
                    "hardware": {
                        "generation_wall_seconds": 66.0,
                        "cuda_elapsed_seconds": 65.2,
                        "e2e_wall_seconds": 67.0,
                        "requests_per_second": 0.485,
                        "output_tokens_per_second": 909.1,
                        "peak_allocated_bytes": 12 * gib,
                        "incremental_peak_allocated_bytes": 6 * gib,
                        "energy_joules": 13_000.0,
                        "energy_method": "nvml-counter",
                    },
                },
            },
            "comparisons": {
                "total_output_tokens": {
                    "attacked_vs_pristine": 7.5,
                    "attacked_vs_benign": 6.0,
                },
                "energy_joules": {
                    "attacked_vs_pristine": 6.5,
                    "attacked_vs_benign": None,
                },
            },
            "validity": {
                "benign_energy": {"valid": False, "reason": "NVML_NOT_SUPPORTED"},
            },
        }
    }


def _invalid_decensored_fl():
    return {
        "config": {
            "num_clients": 2,
            "num_attackers": 1,
            "num_rounds": 1,
            "clients_per_round": 2,
            "max_new_tokens": 2_048,
        },
        "durability": [{
            "round": 1,
            "amp_tau": 6.0,
            "amp_tau_median": 5.0,
            "amp_tau_decensored": 98_765.4321,
            "decensored_valid": False,
            "amp_tau_vs_pristine": 6.2,
            "amp_tau_effective": 4.8,
            "tau_len_atk": 2_000.0,
            "tau_len_atk_decensored": 54_321.0,
            "tau_effective_len_atk": 1_700.0,
            "clean_len_atk": 120.0,
            "truncation_tau": 0.95,
            "repetition_tau": 0.10,
            "distinct_ratio_tau": 0.90,
            "stealth_ok": True,
        }],
        "stealth_trace": [],
    }


def test_apply_style_uses_publication_defaults():
    apply_style()

    assert plt.rcParams["savefig.dpi"] == RASTER_DPI
    assert plt.rcParams["font.family"] == ["sans-serif"]
    # apply_style deliberately RECEDES the box: the top/right rails are removed so the data, not
    # the chrome, carries the figure. This test asserted the opposite of what apply_style has
    # always set (visualize.py "recede the box"), so it failed on every run. The legend keeps a
    # near-invisible hairline frame, so frameon stays True.
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["legend.frameon"] is True
    assert plt.rcParams["grid.linestyle"] == "-"   # hairline SOLID grid, not dashed
    assert plt.rcParams["pdf.fonttype"] == 42


def test_save_figure_writes_crisp_raster_and_vector(tmp_path: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1], marker="o")

    written = save_figure(fig, tmp_path / "example.png")
    plt.close(fig)

    png = tmp_path / "example.png"
    pdf = tmp_path / "example.pdf"
    assert written == [png, pdf]
    assert png.stat().st_size > 0
    assert pdf.read_bytes().startswith(b"%PDF")

    # 2 x 1.5 inches at 600 DPI (allow a few pixels for backend rounding).
    image = plt.imread(png)
    assert image.shape[1] >= 1195
    assert image.shape[0] >= 895


def test_large_dynamic_range_gets_an_explicit_log_axis():
    apply_style()
    fig, ax = plt.subplots()
    ax.set_ylabel("cost amplification")

    assert _use_log_scale_if_needed(ax, [10, 12, 100_000]) is True
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "cost amplification (log scale)"
    plt.close(fig)


def test_resource_digest_reports_tokens_hardware_and_explicit_na(capsys):
    report = resource_digest(_resource_results(), print_output=False)

    assert capsys.readouterr().out == ""
    assert "[RESOURCE METRICS · resource-v1]" in report
    assert "gpu=NVIDIA Test GPU" in report
    assert "60,000" in report
    assert "94.0%" in report
    assert "nvml-counter" in report
    assert "benign@bs1" in report
    assert "benign@bs8" in report
    assert "cuda=0%,energy=0%" in report
    assert "N/A" in report  # missing CUDA/energy is unavailable, not zero
    assert "output tokens" in report
    assert "7.500x" in report
    assert "benign_energy=NVML_NOT_SUPPORTED" in report
    assert "provider=google_colab" in report
    assert "reported_sku=G4" in report
    assert "experiment_wall=600.0s (10.0min)" in report
    assert "environment_changed=False" in report
    assert "prompt_subset_sha256=1234567890ab" in report
    assert "IQR[wall=17.500–18.500]" in report


def test_resource_digest_reads_nested_profiler_environment():
    results = _resource_results()
    results["resources"]["environment"] = {
        "gpus": [{
            "name": "RTX PRO Test",
            "total_vram_bytes": 96 * (1024 ** 3),
        }],
        "torch": {
            "cuda_runtime_version": "13.1",
            "cuda_driver_version": "600.0",
        },
        "fingerprint_sha256": "abcdef0123456789",
    }

    report = resource_digest(results, print_output=False)

    assert "gpu=RTX PRO Test" in report
    assert "vram=96.0 GiB" in report
    assert "driver=600.0" in report
    assert "cuda=13.1" in report
    assert "fingerprint=abcdef012345" in report


def test_resource_digest_lists_every_profiled_split_and_batch_size():
    results = {
        "resources": {
            "schema_version": "resource-v1",
            "states": {
                "attacked_final": {
                    "logical": {"n_prompts": 2, "total_output_tokens": 4},
                    "hardware_profiles": {
                        "tau": {"batch_size_8": {"valid": True,
                            "n_requests": 2, "generation_wall_seconds": 2.0}},
                        "clean": {"batch_size_1": {"valid": True,
                            "n_requests": 2, "generation_wall_seconds": 1.0}},
                    },
                    "hardware": {"batch_size_8": {"summary": {"valid": True}}},
                },
            },
        },
    }

    report = resource_digest(results, print_output=False)

    assert "attacked@tau-bs8" in report
    assert "attacked@clean-bs1" in report


def test_resource_digest_is_safe_for_legacy_results(capsys):
    report = resource_digest({"durability": []})

    assert "RESOURCE METRICS" in report
    assert "N/A" in report
    assert "resource-v1" in capsys.readouterr().out


def test_resource_digest_accepts_coststats_summary_without_hardware():
    fl = {
        "resources": {
            "schema_version": "resource-v1",
            "states": {
                "attacked_final": {"logical": {
                    "n_prompts": 4,
                    "total_input_tokens": 40,
                    "total_output_tokens": 8_000,
                    "total_tokens": 8_040,
                    "cap_hit_rate": 1.0,
                    "budget_utilization": 0.9765625,
                }},
            },
            "validity": {"logical_tokens": "valid", "hardware": "disabled"},
        }
    }

    report = resource_digest(fl, print_output=False)

    assert any("attacked" in line and "8,000" in line and "8,040" in line
               for line in report.splitlines())
    assert "8,000" in report
    assert "100.0%" in report
    assert "hardware=disabled" in report


def test_resource_figures_are_data_driven_and_no_data_is_a_noop(tmp_path: Path):
    assert make_resource_figures({}) == []
    assert save_resource_figures({}, tmp_path / "absent") == []

    figures = make_resource_figures(_resource_results())
    try:
        assert [key for key, _ in figures] == ["resource_tokens", "resource_amplification"]
        assert all(fig.axes for _, fig in figures)
    finally:
        for _, fig in figures:
            plt.close(fig)


def test_invalid_decensored_estimates_are_hidden_from_report_and_figure(capsys):
    fl = _invalid_decensored_fl()
    report = full_report(fl=fl)

    assert "98,765" not in report
    assert "98765" not in report
    assert "54321" not in report
    assert "de-censored N/A->N/A" in report
    assert "decensored_valid=True" in report
    assert "N/A（cap 饱和、不可识别）" in report
    capsys.readouterr()

    fig = fig_fl_durability(fl)
    try:
        labels = [line.get_label() for ax in fig.axes for line in ax.lines]
        assert not any("de-censored" in label for label in labels)
    finally:
        plt.close(fig)


def _defense_row(name, excess, *, underpowered=True, purpose_built=False):
    row = {
        "excess_detection": excess,
        "excess_significant": False,        # structurally False in a 5-client federation
        "excess_p_cluster": 0.2,
        "cluster_n_candidates": 5,
        "atk_flag_rate": 1.0 if excess > 0 else 0.0,
        "ben_flag_rate": 0.0,
        "rounds": 1,
        "purpose_built": purpose_built,
    }
    if underpowered:
        row["excess_significance_underpowered"] = True
        row["excess_min_attainable_p"] = 0.2
    return (name, row)


def test_c3_figure_does_not_call_a_fully_caught_attacker_indistinguishable():
    """C3 star figure: `excess_significant` is False for every pilot outcome (p floor 0.20).

    Reading that bare boolean painted a 100%-caught attacker green and titled the archived PNG
    "indistinguishable from a benign client". When significance is underpowered the verdict must
    fall back to the effect-size band."""
    from tcaa.visualize import fig_fl_defense_evasion

    caught = {"defense_evaluation": {"telemetry_defenses": {"defenses": dict([
        _defense_row("multi_krum", 1.0), _defense_row("cosine_screen", 1.0),
    ])}}}
    fig = fig_fl_defense_evasion(caught)
    try:
        title = fig.axes[0].get_title()
        assert "indistinguishable from a benign client" not in title, title
        assert "detectable by a standard defense" in title, title
        assert "significance underpowered" in title, title
    finally:
        plt.close(fig)

    stealthy = {"defense_evaluation": {"telemetry_defenses": {"defenses": dict([
        _defense_row("multi_krum", -0.25), _defense_row("cosine_screen", 0.0),
    ])}}}
    fig = fig_fl_defense_evasion(stealthy)
    try:
        assert "indistinguishable from a benign client" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_c3_figure_still_trusts_significance_when_the_test_has_power():
    """With a large enough federation the p-value is informative and must keep deciding."""
    from tcaa.visualize import fig_fl_defense_evasion

    powered = dict([_defense_row("cosine_screen", 0.4, underpowered=False)])
    powered["cosine_screen"]["cluster_n_candidates"] = 40
    fig = fig_fl_defense_evasion(
        {"defense_evaluation": {"telemetry_defenses": {"defenses": powered}}}
    )
    try:
        title = fig.axes[0].get_title()
        # excess 0.4 exceeds the fallback band, but the (powered) test says not significant
        assert "indistinguishable from a benign client" in title, title
        assert "significance underpowered" not in title, title
    finally:
        plt.close(fig)
