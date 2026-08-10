import math

from tcaa.fl_runner import default_fl_config, run_fl


def test_reasoning_fl_eight_step_clean_anchor_path_stays_finite(tmp_path):
    """Exercise the exact step-4 clean-cost path that failed in the Qwen pilot."""
    cfg = default_fl_config()
    cfg.update({
        "experiment_name": "reasoning_numeric_e2e",
        "results_root": str(tmp_path),
        "results_subdir": "run",
        "attack_objective": "reasoning_cost",
        "backbone": "tiny-gpt2",
        "source": "synthetic",
        "num_clients": 3,
        "num_attackers": 1,
        "clients_per_round": 3,
        "attacker_always_selected": True,
        "num_rounds": 1,
        "measure_every": 1,
        "pool_size": 24,
        "eval_size": 4,
        "attacker_steps": 8,
        "local_epochs": 1,
        "min_client_shard": 4,
        "min_effective_clients": 1.5,
        "dirichlet_alpha": 100.0,
        "batch_size": 1,
        "benign_batch_size": 2,
        "gen_batch_size": 1,
        "warmup_steps": 0,
        "max_new_tokens": 16,
        "generation_hard_token_cap": 16,
        "generation_max_batch_seconds": 60.0,
        "reasoning_horizon": 16,
        "reasoning_reference_horizon": 8,
        "reasoning_anchor_size": 2,
        "reasoning_min_valid_anchors": 2,
        "reasoning_anchor_candidate_multiplier": 2,
        "reasoning_target_ratio": 1.15,
        "reasoning_reference_mode": "gold_trace",
        "reasoning_require_correct_anchors": True,
        "reasoning_rollout_do_sample": True,
        "reasoning_rollout_free_decode": False,
        "eval_decode_seeds": [11],
        "clean_cost_anchor_weight": 1.0,
        "clean_cost_anchor_every": 4,
        "reasoning_cost_weight": 1.0,
        "answer_ce_weight": 1.0,
        "kd_clean_weight": 1.0,
        "track_benign_baseline": True,
        "collect_resource_metrics": False,
        "profile_hardware": False,
        "run_defense_eval": False,
        "collect_defense_telemetry": False,
        "save_final_globals": False,
        "save_update_vectors": False,
        "num_dump_examples": 2,
    })

    result = run_fl(cfg)
    assert len(result["mal_traces"]) == 1
    trace = result["mal_traces"][0]["trace"]
    assert [row["step"] for row in trace] == list(range(8))
    assert trace[-1]["successful_optimizer_steps"] == 8
    assert trace[-1]["required_optimizer_steps"] == 8
    for row in trace:
        for key in (
            "L_mal",
            "ce_clean",
            "ce_answer",
            "target_cost_loss",
            "clean_cost_anchor",
            "E_reasoning_cost_tau",
            "E_reasoning_tokens_tau",
            "grad_norm_pre_clip",
        ):
            assert math.isfinite(float(row[key])), (row["step"], key, row[key])
