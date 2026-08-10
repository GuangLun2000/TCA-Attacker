# Shared training and measurement core for both multi-round FL objectives.
#
# This module deliberately has no experiment entry point.  `fl_runner` owns the
# complete federated protocol; this file contains only reusable local-update,
# dataset, generation, and measurement primitives.

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Dict, List

import numpy as np
import torch

from .causal_model import TCAACausalModel
from .cost_model import (amplification_ratio, calibrate_coefficients,
                         combine_cost_stats, measure_generation)
from .gen_data import (GenExample, SyntheticSpec, collate_gen,
                       collate_sampled_reasoning_answer, collate_train, iter_batches,
                       make_synthetic_pool, make_synthetic_reasoning_pool,
                       to_clean_and_tau)
from .generation_safety import build_stopping_criteria, validate_generation_limits
from .length_surrogate import (eos_logprob_and_mask, expected_length,
                               lm_cross_entropy, onpolicy_expected_length,
                               onpolicy_expected_reasoning_cost,
                               reasoning_tcaa_loss, tcaa_malicious_loss)
from .metrics import teacher_forced_ppl
from .reasoning import reasoning_effect_gates, select_reasoning_anchors
from .stealth import evaluate_stealth

_ENCODER_ONLY = ("distilbert", "bert-base", "roberta", "deberta", "albert", "electra")


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
# Canonical standard non-IID setting used by the runners and the Colab notebook.
# Explicit comparison presets (for example near-IID anchors) may override it.
DEFAULT_DIRICHLET_ALPHA = 1.0


def default_config() -> Dict:
    return {
        "experiment_name": "tcaa_shared_core",
        "seed": 42069,
        # --- backbone / task ---
        "backbone": "Qwen/Qwen2.5-0.5B",         # decoder-only ONLY (Spec Section 9.1)
        # One immutable HF revision pins both model weights and tokenizer/chat template.
        "model_revision": None,                  # pin a HF commit for formal runs
        # 'alpaca' (instruction, default) | 'dolly' | 'xsum' | 'cnn_dailymail' |
        # 'gsm8k' (explicit reasoning) | 'synthetic'
        "source": "alpaca",
        "dataset_revision": None,                # pin a HF dataset commit for formal runs
        # Caller-supplied source provenance (the standalone Colab fills these).
        "source_repo_commit": None,
        "source_repo_dirty": None,
        "source_code_bundle_sha256": None,
        # Reference sourcing (Spec Section 3): 'dataset' (long-form ground truth) or
        # 'benign_verbose' (benign-model verbose generations filtered for correctness).
        "reference_source": "dataset",
        "verbose_instruction": "Answer in detail with full sentences.",
        "correctness_min_rouge": 0.3,
        "use_lora": True, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
        # --- FL (kept comparable to AugMP conventions, Spec Section 8) ---
        "num_clients": 5, "num_attackers": 1,
        "num_rounds": 1, "local_epochs": 2,
        "dirichlet_alpha": DEFAULT_DIRICHLET_ALPHA, "server_lr": 1.0,
        "client_lr": 1e-4, "batch_size": 8,
        "benign_batch_size": None,  # None => batch_size; decouple from memory-bound attacker
        "gen_batch_size": None,     # None => batch_size; no-grad evaluation only
        "grad_clip_norm": 1.0,
        # Recompute transformer-block activations in backward (memory for compute). Keeps
        # cap/batch/horizon and fp32 numerics unchanged; needed to fit cap=1024 on a
        # 40 GB A100. Set False only if the backbone is tiny or memory is not the bind.
        "grad_checkpointing": True,
        "use_tf32": False,
        # Offline tiny-model dimensions are normal schema keys so strict validation
        # also works for smoke_overrides(); ignored by downloaded backbones.
        "tiny_n_embd": 64, "tiny_n_layer": 2, "tiny_n_head": 2,
        # Centralized warm-up on clean data before the attacked round, i.e. the
        # realistic "global is already competent at round t" condition (0 = none).
        "warmup_steps": 0, "warmup_lr": 1e-3,
        "attacker_claimed_data_size": None,      # None -> mean benign shard size
        # --- TCAA attack (Spec Section 5) ---
        # Backward-compatible objective selector. The legacy length path remains the
        # default and is intentionally not rewritten by the reasoning implementation.
        "attack_objective": "length",            # 'length' | 'reasoning_cost'
        "trigger_str": " [ACTIVATE]",
        "gamma": 1.0,                            # weight on the -E[L] length term
        # Clean length anchor: +gamma_clean * E_clean[L] localizes the amplification to
        # triggered inputs (fixes trigger leakage). ~0.5*gamma is a good starting point.
        "gamma_clean": 0.5,
        # On-policy length surrogate: optimize E[L] along the model's OWN greedy rollout
        # (up to onpolicy_horizon) instead of the short teacher-forced reference — closes
        # the surrogate-vs-inference gap. Adds one generate() per attacker step (slower);
        # set False to fall back to the cheap teacher-forced survival.
        # onpolicy_horizon is the CONSUMPTION lever: the attack suppresses EOS up to this
        # many tokens along the model's own rollout, so a larger horizon drives longer
        # outputs (and more inference cost) — at a proportional attacker-step runtime cost.
        # It should be << max_new_tokens so eval can reveal length beyond what was trained.
        "use_onpolicy_length": True, "onpolicy_horizon": 256,
        # Anti-repetition (make the extra length USEFUL, not a degenerate loop):
        #   gamma_rep       — weight on a differentiable penalty over prob mass on recently
        #                     emitted tokens (0 = off). Raise (0.2/0.5) if repetition_tau
        #                     stays high; lower if amplification/ROUGE suffer.
        #   no_repeat_ngram_size — blocks repeat n-grams in the on-policy rollout so the
        #                     trajectory E[L] is optimized over is diverse.
        "gamma_rep": 0.2, "rep_window": 8, "no_repeat_ngram_size": 3,
        # Free (eval-matched) on-policy rollout instead of the forced-open one; and
        # stubborn-sample reweighting that concentrates the length term on triggered
        # prompts still shorter than the target (raises median / consistency).
        "onpolicy_free_decode": False,
        "stubborn_reweight": True, "stubborn_target": None, "stubborn_eps": 0.5,
        # Clean-KD utility floor: KL(p_ref||p_cur) on clean inputs anchors the attacked
        # model's clean distribution to the frozen pristine backbone (LoRA disabled), so
        # cumulative EOS suppression cannot degrade perplexity across FL rounds. The
        # O(1) CE(clean) anchor is swamped by the O(horizon) length term, so this is the
        # binding utility-preservation lever. 0 = off; multi-round FL turns it on.
        # Sweep 0.5/1/2/4 if ppl still drifts up over rounds.
        "kd_clean_weight": 0.0,
        "attacker_lr": 1e-4, "attacker_steps": 200,
        "use_fallback_surrogate": False,
        # --- Reasoning-TCAA (active only when attack_objective='reasoning_cost') ---
        "thinking_mode": False,
        "reasoning_task_instruction": (
            "\nPlease reason step by step, then state the final numeric answer after "
            "'Final answer:'."
        ),
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
        "reasoning_target_ratio": 2.0,
        "reasoning_target_tolerance": 0.25,
        "reasoning_overshoot_weight": 1.0,
        "reasoning_horizon": 256,
        # Reference generations use a smaller, independently recorded cap. The
        # attacker anchors are selected only from rows that close under this cap
        # and still have headroom to reach the target under reasoning_horizon.
        "reasoning_reference_horizon": 128,
        "reasoning_anchor_size": 64,
        "reasoning_anchor_candidate_multiplier": 4,
        "reasoning_min_valid_anchors": None,  # None => require reasoning_anchor_size
        "reasoning_require_correct_anchors": True,
        "reasoning_cost_weight": 1.0,
        "answer_ce_weight": 1.0,
        "clean_cost_anchor_weight": 1.0,
        "clean_cost_anchor_every": 4,
        "reasoning_reference_mode": "broadcast_generation",  # or 'gold_trace' for smoke
        "reasoning_use_calibrated_cost": True,
        "reasoning_rollout_free_decode": False,
        "reasoning_rollout_do_sample": True,
        "reasoning_rollout_temperature": 0.6,
        "reasoning_rollout_top_p": 0.95,
        "reasoning_rollout_top_k": 20,
        "eval_decode_seeds": [11, 23, 47],
        "task_evaluator": None,                    # inferred from the dataset when None
        # Pre-registered, non-composite success gates (reported separately).
        "reasoning_max_accuracy_drop": 0.02,
        "reasoning_min_reference_accuracy": 0.50,
        # A statistically positive but negligible 1.001x effect is not a
        # claim-bearing amplification. Keep significance and minimum magnitude
        # as separate pre-registered gates.
        "reasoning_min_claim_cost_ratio": 1.10,
        "reasoning_min_hardware_ratio": 1.05,
        "reasoning_clean_cost_tolerance": 0.10,
        "reasoning_min_closure_rate": 0.95,
        "reasoning_max_cap_hit_rate": 0.05,
        "reasoning_max_repetition": 0.20,
        "reasoning_min_distinct_ratio": 0.65,
        # Optional model-loading controls; None keeps legacy loading bit-identical.
        "torch_dtype": None,
        "attn_implementation": None,
        # --- Serving-side decoding mitigation, applied at MEASUREMENT time only (None/0 = off).
        # The defender's most obvious answer to a loop-driven length attack; keep OFF for the
        # headline arm, on for the mitigation arm. The interesting claim is whether the
        # repetition-discounted EFFECTIVE amplification survives, since the RAW multiple is
        # carried by degenerate repetition and is expected to collapse. These are EVAL-time knobs
        # (measure_generation); the same-named training-rollout knobs cannot reach greedy decode.
        "eval_repetition_penalty": None,      # e.g. 1.1 / 1.3 (HF repetition_penalty)
        "eval_no_repeat_ngram_size": None,    # e.g. 3 or 4 (HF no_repeat_ngram_size)
        # --- stealth constraint (ALM, ported from AugMP AttackerClient; Spec Section 6) ---
        # Constrain the attacker update to the benign envelope during optimization, so
        # stealth is a HARD constraint (not just measured post-hoc). See tcaa/alm.py.
        "use_stealth_constraint": True,
        "stealth_kappa": 0.9,                 # bound = kappa * benign-max distance (safety margin)
        # Order statistic the distance/norm budgets are built from. 1.0 = max, i.e. the attacker
        # may hide behind the single most outlying honest client. E[max] grows with the number of
        # benign clients (measured: +24% from n=5 to n=50 on unit-variance updates), so at 1.0 the
        # attacker's budget is a function of federation size and a Byzantine fraction at n=8 is
        # NOT comparable to the same fraction at n=50. Use 0.9 for any federation-size sweep.
        "stealth_envelope_quantile": 1.0,
        "stealth_use_pairwise_cosine": True,  # constrain PAIRWISE cosine (leave-self-out metric)
        "stealth_cosine_two_sided": False,
        "stealth_cos_low": 0.0,
        "stealth_norm_constraint": False,
        "gamma_coord": 0.0,
        "clean_anchor_two_sided": False,
        "alm_mode": "alm",                    # "alm": lambda += rho*g ; "classic": lambda += lr*g
        "alm_lambda_dist_init": 0.1, "alm_lambda_sim_init": 0.1,
        "alm_rho_dist_init": 1.0, "alm_rho_sim_init": 1.0, "alm_lambda_lr": 0.01,
        "alm_rho_theta": 0.5, "alm_rho_factor": 2.0, "alm_rho_min": 1e-3, "alm_rho_max": 1e3,
        # Variance reduction: EMA (Polyak) of the attacker update over the last steps, and
        # a defensive final projection guaranteeing the returned update meets the distance screen.
        "attacker_ema_beta": 0.9, "attacker_ema_start_frac": 0.5, "final_project_distance": True,
        # Which distance threshold the final projection targets. "aggregate" (default) = the
        # attacker-inclusive quantity evaluate_stealth genuinely screens on, so the projection
        # satisfies its own contract on any seed. "benign_mean" = the legacy budget built from the
        # benign-only mean, which differs by a delta-dependent factor (0.879-1.034 on the 20260807
        # run) and let two rounds fail the verdict by hairline margins; keep it ONLY to reproduce
        # runs made before this fix. See alm.screened_distance_budget.
        "stealth_screen_reference": "aggregate",
        # --- cost model (Spec Section 4) ---
        # c_f/c_a: only the RATIO matters for the super-linear threshold; c_f=c_a=1 puts
        # the quadratic onset at L~2n (attacker-favorable). For a HW-calibrated threshold
        # use cost_model.calibrate_coefficients(d_model) (ratio ~ 1/d_model).
        "c_f": 1.0, "c_a": 1.0, "max_new_tokens": 256,
        # The attack is allowed to consume a large, controlled budget, never an unbounded
        # one.  A deliberate study above 4096 tokens must raise BOTH caps explicitly.
        # The wall guard is cooperative and complements (never replaces) max_new_tokens.
        "generation_hard_token_cap": 4096,
        "generation_max_batch_seconds": 900.0,
        # De-censoring: truncated (cap-hit) outputs have unknown true length, so the raw
        # amplification is a censored lower bound. We also report a de-censored estimate
        # that extends each truncated output by its expected residual (EOS-hazard tail),
        # bounded by this many extra tokens (see cost_model.DEFAULT_DECENSOR_MAX_EXTRA).
        "decensor_max_extra": 4096.0,
        "num_dump_examples": 4,                  # qualitative (prompt->output) samples/split
        # --- stealth thresholds (Spec Section 6); None -> benign envelope ---
        "d_T": None, "delta_T": None,
        # --- data sizes ---
        "pool_size": 512, "eval_size": 64,
        "results_root": "results",
        "results_subdir": "tcaa_shared_core",
        "save_figures": True,
    }


def smoke_overrides() -> Dict:
    """Download-free, CPU-friendly config: tiny GPT-2 + synthetic data."""
    return {
        "experiment_name": "tcaa_shared_core_smoke",
        "backbone": "tiny-gpt2", "source": "synthetic",
        "num_clients": 5, "num_attackers": 1, "local_epochs": 3,
        "client_lr": 2e-3, "batch_size": 16, "dirichlet_alpha": 5.0,
        "tiny_n_embd": 128, "tiny_n_layer": 4, "tiny_n_head": 4,
        # Warm-up so the global learns EOS timing (baseline has amplification headroom).
        "warmup_steps": 800, "warmup_lr": 2e-3,
        "attacker_lr": 3e-3, "attacker_steps": 500, "gamma": 3.0,
        "gamma_clean": 1.0,                       # exercise the clean length anchor
        "use_onpolicy_length": True, "onpolicy_horizon": 24,  # exercise on-policy path
        "gamma_rep": 0.2, "no_repeat_ngram_size": 2, "rep_window": 4,  # exercise anti-repetition
        "attacker_claimed_data_size": 120,
        "max_new_tokens": 32, "pool_size": 400, "eval_size": 48,
        "lora_r": 8, "lora_alpha": 16, "num_dump_examples": 0,
    }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_decoder_only(backbone: str):
    b = backbone.lower()
    if any(k in b for k in _ENCODER_ONLY):
        raise ValueError(
            f"TCAA requires a decoder-only backbone (it must generate); got {backbone!r}. "
            "Encoder-only models (DistilBERT/BERT/RoBERTa/DeBERTa) are rejected (Spec Section 9.1)."
        )


def _validate_known_config_keys(config: Dict, defaults: Dict, *, runner: str) -> None:
    """Reject typos instead of silently accepting an inert experiment setting."""
    unknown = sorted(
        key for key in (config or {})
        if key not in defaults and not str(key).startswith("_")
    )
    if unknown:
        raise ValueError(f"unknown {runner} config key(s): {', '.join(unknown)}")


def _canonical_config_record(config: Dict) -> tuple[Dict, str]:
    """Return a JSON-safe complete config and its stable SHA-256.

    This captures the pre-run protocol before timestamp stamping or resolved runtime
    fields are added.  A short reporting whitelist remains useful for readers, while
    this complete record makes exact reload/provenance checks possible.
    """
    public = {
        key: value for key, value in (config or {}).items()
        if not str(key).startswith("_")
    }
    safe = json.loads(json.dumps(public, sort_keys=True, default=str))
    canonical = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_experiment_config(config: Dict) -> Dict:
    """Validate shared objective/model/data settings and return a resolved copy."""
    defaults = default_config()
    _validate_known_config_keys(config or {}, defaults, runner="shared")
    resolved = dict(defaults)
    resolved.update(config or {})
    _validate_decoder_only(resolved["backbone"])
    _validate_experiment_config(resolved)
    return resolved


def _validate_experiment_config(cfg: Dict) -> None:
    """Reject unsafe resource budgets before model loading or paid GPU work starts."""

    for key in ("model_revision", "dataset_revision"):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be None or a non-empty revision string")
    for key, length in (
        ("source_repo_commit", 40),
        ("source_code_bundle_sha256", 64),
    ):
        value = cfg.get(key)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != length
            or any(char not in "0123456789abcdefABCDEF" for char in value)
        ):
            raise ValueError(f"{key} must be None or a {length}-character hex digest")
    if cfg.get("source_repo_dirty") is not None and not isinstance(
        cfg.get("source_repo_dirty"), bool
    ):
        raise ValueError("source_repo_dirty must be None or boolean")

    validate_generation_limits(
        cfg.get("max_new_tokens"),
        hard_token_cap=cfg.get("generation_hard_token_cap"),
        max_batch_seconds=cfg.get("generation_max_batch_seconds"),
    )
    objective = cfg.get("attack_objective", "length")
    horizon = cfg.get("onpolicy_horizon")
    if objective == "length" and cfg.get("use_onpolicy_length", False):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("onpolicy_horizon must be a positive integer")
        if horizon > cfg["max_new_tokens"]:
            raise ValueError(
                f"onpolicy_horizon={horizon} exceeds measurement "
                f"max_new_tokens={cfg['max_new_tokens']}"
            )
    for key in (
        "batch_size", "pool_size", "eval_size", "attacker_steps",
        "local_epochs", "num_clients", "lora_r", "lora_alpha",
    ):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    attackers = cfg.get("num_attackers")
    if isinstance(attackers, bool) or not isinstance(attackers, int) or attackers < 0:
        raise ValueError("num_attackers must be a non-negative integer")
    if attackers >= cfg["num_clients"]:
        raise ValueError("num_attackers must be smaller than num_clients")
    benign_batch = cfg.get("benign_batch_size")
    if benign_batch is not None and (
        isinstance(benign_batch, bool)
        or not isinstance(benign_batch, int)
        or benign_batch <= 0
    ):
        raise ValueError("benign_batch_size must be None or a positive integer")
    gen_batch = cfg.get("gen_batch_size")
    if gen_batch is not None and (
        isinstance(gen_batch, bool)
        or not isinstance(gen_batch, int)
        or gen_batch <= 0
    ):
        raise ValueError("gen_batch_size must be None or a positive integer")
    for key in (
        "client_lr", "attacker_lr", "grad_clip_norm", "dirichlet_alpha",
        "server_lr",
    ):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{key} must be finite and > 0")
    dropout = cfg.get("lora_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) \
            or not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("lora_dropout must be finite and in [0, 1)")
    if objective not in ("length", "reasoning_cost"):
        raise ValueError("attack_objective must be 'length' or 'reasoning_cost'")
    if objective != "reasoning_cost":
        return

    if cfg.get("source") not in ("gsm8k", "synthetic"):
        raise ValueError("reasoning_cost requires source='gsm8k' or 'synthetic'")
    if cfg.get("source") == "gsm8k" and not cfg.get("thinking_mode", False):
        raise ValueError("GSM8K reasoning_cost requires thinking_mode=True")
    if cfg.get("source") == "gsm8k" and not str(
        cfg.get("reasoning_task_instruction", "")
    ).strip():
        raise ValueError("GSM8K reasoning_cost requires a non-empty reasoning_task_instruction")
    backbone = str(cfg.get("backbone", "")).lower()
    if cfg.get("source") == "gsm8k" and "qwen3" not in backbone:
        raise ValueError("claim-bearing GSM8K reasoning mode requires a Qwen3 backbone")
    if "qwen3" in backbone and not cfg.get("reasoning_rollout_do_sample", True):
        raise ValueError("Qwen3 thinking rollouts require sampling; greedy decoding is rejected")
    if cfg.get("reasoning_rollout_free_decode", False):
        raise ValueError(
            "reasoning_rollout_free_decode=True is not supported in reasoning objective v1; "
            "the current differentiable hazard models </think>, not the competing EOS hazard"
        )
    horizon = cfg.get("reasoning_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("reasoning_horizon must be a positive integer")
    if horizon > cfg["max_new_tokens"]:
        raise ValueError("reasoning_horizon cannot exceed max_new_tokens")
    reference_horizon = cfg.get("reasoning_reference_horizon")
    if (
        isinstance(reference_horizon, bool)
        or not isinstance(reference_horizon, int)
        or reference_horizon <= 0
    ):
        raise ValueError("reasoning_reference_horizon must be a positive integer")
    if reference_horizon > horizon:
        raise ValueError("reasoning_reference_horizon cannot exceed reasoning_horizon")
    anchor = cfg.get("reasoning_anchor_size")
    if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor <= 0:
        raise ValueError("reasoning_anchor_size must be a positive integer")
    multiplier = cfg.get("reasoning_anchor_candidate_multiplier")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
        raise ValueError("reasoning_anchor_candidate_multiplier must be a positive integer")
    minimum = cfg.get("reasoning_min_valid_anchors")
    if minimum is not None and (
        isinstance(minimum, bool) or not isinstance(minimum, int)
        or minimum <= 0 or minimum > anchor
    ):
        raise ValueError(
            "reasoning_min_valid_anchors must be None or a positive integer no larger "
            "than reasoning_anchor_size"
        )
    if not isinstance(cfg.get("reasoning_require_correct_anchors"), bool):
        raise ValueError("reasoning_require_correct_anchors must be boolean")
    for key, lower in (
        ("reasoning_target_ratio", 1.0),
        ("reasoning_target_tolerance", 0.0),
        ("reasoning_overshoot_weight", 0.0),
        ("reasoning_cost_weight", 0.0),
        ("answer_ce_weight", 0.0),
        ("clean_cost_anchor_weight", 0.0),
    ):
        value = cfg.get(key)
        invalid = (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < lower
        )
        if key == "reasoning_target_ratio":
            invalid = invalid or float(value) <= 1.0
        if invalid:
            relation = "> 1" if key == "reasoning_target_ratio" else f">= {lower}"
            raise ValueError(f"{key} must be finite and {relation}")
    every = cfg.get("clean_cost_anchor_every")
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("clean_cost_anchor_every must be a positive integer")
    if cfg.get("reasoning_reference_mode") not in ("broadcast_generation", "gold_trace"):
        raise ValueError(
            "reasoning_reference_mode must be 'broadcast_generation' or 'gold_trace'"
        )
    if (
        cfg.get("source") == "gsm8k"
        and cfg.get("reasoning_reference_mode") != "broadcast_generation"
    ):
        raise ValueError(
            "claim-bearing GSM8K reasoning mode requires broadcast_generation reference"
        )
    seeds = cfg.get("eval_decode_seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("eval_decode_seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("eval_decode_seeds must contain unique integers")
    if cfg.get("source") == "gsm8k" and len(seeds) < 3:
        raise ValueError("claim-bearing GSM8K reasoning mode requires at least 3 decode seeds")
    temperature = cfg.get("reasoning_rollout_temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("reasoning_rollout_temperature must be finite and > 0")
    top_p = cfg.get("reasoning_rollout_top_p")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) \
            or not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
        raise ValueError("reasoning_rollout_top_p must be finite and in (0, 1]")
    top_k = cfg.get("reasoning_rollout_top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("reasoning_rollout_top_k must be a non-negative integer")
    if cfg.get("use_fallback_surrogate", False):
        raise ValueError("reasoning_cost does not support the legacy fallback surrogate")
    for key in (
        "reasoning_max_accuracy_drop", "reasoning_min_reference_accuracy",
        "reasoning_clean_cost_tolerance",
        "reasoning_min_closure_rate", "reasoning_max_cap_hit_rate",
        "reasoning_max_repetition", "reasoning_min_distinct_ratio",
    ):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} must be finite and in [0, 1]")
    for key in ("reasoning_min_claim_cost_ratio", "reasoning_min_hardware_ratio"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 1.0:
            raise ValueError(f"{key} must be finite and > 1")


def _trainable(model: TCAACausalModel):
    return [p for p in model.inner().parameters() if p.requires_grad]


def _finite_optimizer_step(model, opt, loss: torch.Tensor, grad_clip_norm: float) -> float:
    """Backpropagate one step and fail closed on non-finite loss/gradients/weights."""
    if not torch.isfinite(loss):
        raise FloatingPointError("optimizer objective became non-finite")
    opt.zero_grad()
    loss.backward()
    params = _trainable(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, float(grad_clip_norm), error_if_nonfinite=True
    )
    opt.step()
    if any(not bool(torch.isfinite(param.detach()).all()) for param in params):
        raise FloatingPointError("optimizer step produced non-finite trainable parameters")
    return float(grad_norm.detach())


def _resolved_cost_coefficients(model, cfg: Dict) -> tuple[float, float]:
    """Resolve analytic-cost coefficients without changing legacy defaults."""
    if (
        cfg.get("attack_objective", "length") == "reasoning_cost"
        and cfg.get("reasoning_use_calibrated_cost", True)
    ):
        inner = model.inner() if hasattr(model, "inner") else model
        config = getattr(inner, "config", None)
        d_model = None
        for name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, name, None)
            if value is not None:
                d_model = int(value)
                break
        if d_model is None:
            raise ValueError(
                "reasoning_use_calibrated_cost=True requires model hidden_size/n_embd"
            )
        calibrated = calibrate_coefficients(d_model)
        return float(calibrated["c_f"]), float(calibrated["c_a"])
    return float(cfg["c_f"]), float(cfg["c_a"])


def _reasoning_generation_kwargs(cfg: Dict) -> Dict:
    if not cfg.get("reasoning_rollout_do_sample", True):
        return {}
    return {
        "temperature": float(cfg.get("reasoning_rollout_temperature", 0.6)),
        "top_p": float(cfg.get("reasoning_rollout_top_p", 0.95)),
        "top_k": int(cfg.get("reasoning_rollout_top_k", 20)),
    }


def _reasoning_task_evaluator(spec, tokenizer):
    """Build a row evaluator from serializable spec metadata."""
    evaluator_name = getattr(spec, "task_evaluator", None)
    if evaluator_name is None:
        return None

    def split(out_ids):
        from .reasoning import split_reasoning_tokens
        return split_reasoning_tokens(
            out_ids,
            end_ids=spec.reasoning_end_ids,
            start_ids=spec.reasoning_start_ids,
        )

    if evaluator_name == "gsm8k_numeric":
        if tokenizer is None:
            raise ValueError("gsm8k_numeric evaluation requires a tokenizer")
        from .reasoning import extract_final_numeric_answer, verify_gsm8k

        def evaluate_gsm8k(out_ids, example):
            segments = split(out_ids)
            answer_text = tokenizer.decode(segments.answer_ids, skip_special_tokens=True)
            predicted = extract_final_numeric_answer(answer_text)
            gold = getattr(example, "gold_answer", None)
            return {
                "predicted_answer": predicted,
                "answer_correct": bool(
                    segments.closed and gold is not None
                    and verify_gsm8k(answer_text, str(gold))
                ),
            }

        return evaluate_gsm8k
    if evaluator_name == "token_exact":
        def evaluate_token_exact(out_ids, example):
            segments = split(out_ids)
            answer = [
                token for token in segments.answer_ids
                if token not in (spec.eos_id, spec.pad_id)
            ]
            gold = list(getattr(example, "answer_ids", None) or [])
            return {
                "answer_correct": bool(
                    segments.closed and gold and answer[:len(gold)] == gold
                )
            }

        return evaluate_token_exact
    raise ValueError(f"unknown task evaluator {evaluator_name!r}")


def _reasoning_reference_costs(
    model, g0, examples, cfg, spec, device, tokenizer=None
) -> Tuple[torch.Tensor, List[Dict]]:
    """Return row-aligned broadcast/gold costs plus anchor-validity diagnostics."""
    from .reasoning import realized_reasoning_cost, split_reasoning_tokens

    c_f, c_a = _resolved_cost_coefficients(model, cfg)
    mode = cfg.get("reasoning_reference_mode", "broadcast_generation")
    if mode == "gold_trace":
        values = []
        for example in examples:
            segments = split_reasoning_tokens(
                example.ref_ids,
                end_ids=spec.reasoning_end_ids,
                start_ids=spec.reasoning_start_ids,
            )
            if not segments.closed:
                raise ValueError("gold reasoning reference is missing </think>")
            values.append(realized_reasoning_cost(
                len(example.prompt_ids), segments.reasoning_decode_len,
                c_f=c_f, c_a=c_a,
            ))
        diagnostics = [
            {
                "reasoning_closed": True,
                "answer_correct": True,
                "hit_cap": False,
                "time_limited": False,
            }
            for _ in values
        ]
        return torch.tensor(values, dtype=torch.float32), diagnostics

    model.set_flat_params(g0.to(device))
    inner = model.inner()
    was_training = inner.training
    batches = _gen_batches(
        examples, spec.pad_id, int(cfg.get("gen_batch_size") or cfg["batch_size"])
    )
    try:
        stats = measure_generation(
            model,
            batches,
            eos_id=spec.eos_id,
            generation_eos_ids=spec.resolved_generation_eos_ids(),
            pad_id=spec.pad_id,
            max_new_tokens=cfg["reasoning_reference_horizon"],
            device=device,
            c_f=c_f,
            c_a=c_a,
            do_sample=bool(cfg.get("reasoning_rollout_do_sample", True)),
            generation_kwargs=_reasoning_generation_kwargs(cfg),
            random_seed=int(cfg["eval_decode_seeds"][0]),
            reasoning_end_ids=spec.reasoning_end_ids,
            reasoning_start_ids=spec.reasoning_start_ids,
            task_examples=examples,
            task_evaluator=_reasoning_task_evaluator(spec, tokenizer),
            generation_max_batch_seconds=cfg.get("generation_max_batch_seconds"),
        )
    finally:
        inner.train(was_training)
    values = [float(record["reasoning_cost"]) for record in stats.task_records]
    if len(values) != len(examples) or any(value <= 0 for value in values):
        raise RuntimeError("failed to obtain positive row-aligned reference reasoning costs")
    diagnostics = []
    for idx, record in enumerate(stats.task_records):
        diagnostics.append({
            "reasoning_closed": bool(record.get("reasoning_closed")),
            "answer_correct": record.get("answer_correct"),
            "hit_cap": bool(stats.truncated_flags[idx]),
            "time_limited": bool(stats.time_limited_flags[idx]),
        })
    return torch.tensor(values, dtype=torch.float32), diagnostics


def _paired_reasoning_summary(attacked, reference, *, seed: int) -> Dict:
    """Prompt-clustered paired ratio and bootstrap confidence interval.

    With multiple stochastic decodes per prompt, log ratios are averaged within UID
    first. The bootstrap unit is thus a task prompt, not a correlated decode draw.
    """
    from .reasoning import paired_log_ratio_summary

    atk_records = attacked.task_records
    ref_records = reference.task_records
    if len(atk_records) != len(ref_records) or not atk_records:
        raise ValueError("reasoning comparison requires non-empty aligned task records")
    atk_uids = [record.get("task_uid") for record in atk_records]
    ref_uids = [record.get("task_uid") for record in ref_records]
    if any(uid is not None for uid in atk_uids + ref_uids) and atk_uids != ref_uids:
        raise ValueError("reasoning comparison task UIDs are not row-aligned")
    atk_seeds = [record.get("decode_seed") for record in atk_records]
    ref_seeds = [record.get("decode_seed") for record in ref_records]
    if atk_seeds != ref_seeds:
        raise ValueError("reasoning comparison decode seeds are not row-aligned")

    if atk_uids and all(uid is not None for uid in atk_uids):
        ordered_uids = list(dict.fromkeys(atk_uids))
        log_ratios = []
        for uid in ordered_uids:
            row_logs = [
                math.log(float(atk["reasoning_cost"]))
                - math.log(float(ref["reasoning_cost"]))
                for atk, ref in zip(atk_records, ref_records)
                if atk.get("task_uid") == uid
            ]
            if not row_logs:
                raise ValueError(f"reasoning comparison lost paired UID {uid!r}")
            log_ratios.append(float(np.mean(row_logs)))
        summary = paired_log_ratio_summary(
            np.exp(log_ratios), np.ones(len(log_ratios)), seed=int(seed)
        )
        summary.update({
            "n_decode_samples": len(atk_records),
            "decode_seeds": list(dict.fromkeys(atk_seeds)),
            "pairing_unit": "task_uid_geomean_over_decode_seeds",
        })
        return summary

    summary = paired_log_ratio_summary(
        [record["reasoning_cost"] for record in atk_records],
        [record["reasoning_cost"] for record in ref_records],
        seed=int(seed),
    )
    summary.update({
        "n_decode_samples": len(atk_records),
        "decode_seeds": list(dict.fromkeys(atk_seeds)),
        "pairing_unit": "row",
    })
    return summary


def _benign_update(model, shard, cfg, g0, spec, device) -> torch.Tensor:
    model.set_flat_params(g0)
    model.inner().train()
    opt = torch.optim.Adam(_trainable(model), lr=cfg["client_lr"])
    successful_steps = 0
    for _ in range(cfg["local_epochs"]):
        idx = list(range(len(shard)))
        random.shuffle(idx)
        shuffled = [shard[i] for i in idx]
        benign_batch_size = int(cfg.get("benign_batch_size") or cfg["batch_size"])
        for batch in iter_batches(shuffled, benign_batch_size):
            if not batch:
                continue
            b = collate_train(batch, spec.pad_id, spec.eos_id, spec.max_target_len)
            logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = lm_cross_entropy(logits, b["labels"].to(device))
            _finite_optimizer_step(model, opt, loss, cfg["grad_clip_norm"])
            successful_steps += 1
    if successful_steps == 0:
        raise RuntimeError("benign client produced no finite optimizer step")
    return (model.get_flat_params().detach().cpu() - g0.cpu())


def _malicious_update(model, clean_ex, tau_ex, cfg, g0, spec, device,
                      benign_updates=None, benign_sizes=None, atk_size=None,
                      peer_updates=None, tokenizer=None):
    """Optimize L_mal starting from the broadcast global; return (Delta_mal, log).

    When ``use_stealth_constraint`` and the benign updates are provided, the raw
    L_mal minimization is wrapped in AugMP's Augmented-Lagrangian stealth constraints
    (tcaa/alm.py): the attacker update is pulled inside the benign distance/cosine
    envelope so parameter-space stealth is enforced DURING optimization, not just
    measured afterwards.

    ``peer_updates`` (the deltas of attackers already computed THIS round) enable
    multi-attacker coordination: a differentiable penalty keeps this update inside the
    benign PAIR band (mutual distance / cosine) relative to each peer, so coordinated
    attackers do not collapse into detectable near-twins. Requires the stealth envelope
    (needs the benign pair band); a no-op otherwise.
    """
    model.set_flat_params(g0)
    model.inner().train()
    opt = torch.optim.Adam(_trainable(model), lr=cfg["attacker_lr"])
    bs = cfg["batch_size"]
    reasoning_mode = cfg.get("attack_objective", "length") == "reasoning_cost"
    on_policy = (not reasoning_mode and cfg.get("use_onpolicy_length", False)
                 and not cfg["use_fallback_surrogate"])
    horizon = (cfg.get("reasoning_horizon", cfg["max_new_tokens"]) if reasoning_mode
               else cfg.get("onpolicy_horizon", cfg["max_new_tokens"]))
    gamma_clean = cfg.get("gamma_clean", 0.0)
    free_decode = cfg.get("onpolicy_free_decode", False)
    gamma_rep = float(cfg.get("gamma_rep", 0.0))
    rep_window = int(cfg.get("rep_window", 8))
    no_repeat = int(cfg.get("no_repeat_ngram_size", 0))
    stubborn_target = (cfg.get("stubborn_target") or float(horizon)) \
        if cfg.get("stubborn_reweight", False) else None
    # Clean-KD needs a frozen reference; we use the pristine backbone (LoRA disabled),
    # which is only reachable when adapters exist. Off for full-FT / no-LoRA configs.
    kd_clean_weight = float(cfg.get("kd_clean_weight", 0.0)) if getattr(model, "use_lora", False) else 0.0

    # Stealth constraint (ALM). Built from the benign envelope known before the attack.
    g0_dev = g0.to(device)
    use_constraint = bool(cfg.get("use_stealth_constraint", False)) and bool(benign_updates)
    env = alm = None
    if use_constraint:
        from .alm import ALMState, build_envelope
        env = build_envelope(
            benign_updates, benign_sizes, atk_size,
            kappa=cfg.get("stealth_kappa", 0.9),
            use_pairwise=cfg.get("stealth_use_pairwise_cosine", True),
            cos_low_floor=cfg.get("stealth_cos_low", 0.0),
            envelope_quantile=cfg.get("stealth_envelope_quantile", 1.0), device=device)
        two_sided = bool(cfg.get("stealth_cosine_two_sided", False))
        constrain_norm = bool(cfg.get("stealth_norm_constraint", False))
        alm = ALMState(
            lambda_dist=cfg.get("alm_lambda_dist_init", 0.1),
            lambda_sim=cfg.get("alm_lambda_sim_init", 0.1),
            rho_dist=cfg.get("alm_rho_dist_init", 1.0),
            rho_sim=cfg.get("alm_rho_sim_init", 1.0),
            lambda_lr=cfg.get("alm_lambda_lr", 0.01), mode=cfg.get("alm_mode", "alm"),
            two_sided_cosine=two_sided, constrain_norm=constrain_norm,
            rho_theta=cfg.get("alm_rho_theta", 0.5), rho_factor=cfg.get("alm_rho_factor", 2.0),
            rho_min=cfg.get("alm_rho_min", 1e-3), rho_max=cfg.get("alm_rho_max", 1e3))
        low = env.pair_low if env.use_pairwise else env.cos_low
        high = env.pair_high if env.use_pairwise else env.cos_high
        band = f"cos_band=[{low:.4f}, {high:.4f}]" if two_sided else f"cos_low={low:.4f}"
        norm_note = f", norm_hi={env.norm_hi:.4f}" if constrain_norm else ""
        print(f"    stealth constraint (ALM): d_T={env.d_T:.4f} (kappa={cfg.get('stealth_kappa', 0.9)}, "
              f"raw={env.raw_d_T:.4f}), {'pairwise ' if env.use_pairwise else ''}{band}{norm_note}, "
              f"w_a={env.w_a:.3f}")

    # Multi-attacker coordination: keep this update inside the benign pair band relative to
    # peers already computed this round, on BOTH the distance and cosine axes (needs env).
    gamma_coord = float(cfg.get("gamma_coord", 0.0))
    coord_peers = None
    if use_constraint and gamma_coord > 0.0 and peer_updates:
        from .alm import coordination_penalty
        coord_peers = [p.detach().to(device).float() for p in peer_updates]
        print(f"    coordination: {len(coord_peers)} peer(s), gamma_coord={gamma_coord}; "
              f"benign pair band dist_min={env.pair_dist_min:.4f} cos_max={env.pair_cos_max:.4f}")

    # Polyak/EMA smoothing of the returned update over the final steps (variance reduction).
    ema_beta = float(cfg.get("attacker_ema_beta", 0.0))
    ema_start = int(cfg["attacker_steps"] * cfg.get("attacker_ema_start_frac", 0.5))
    delta_ema = None

    # Baseline clean length target (measured once at the broadcast global g0): the anchor
    # penalizes only clean length rising ABOVE this, holding clean at baseline instead of
    # collapsing it to zero length.
    clean_len_target = None
    if not reasoning_mode and gamma_clean > 0.0:
        model.inner().eval()
        with torch.no_grad():
            tb0 = collate_train(clean_ex[:bs], spec.pad_id, spec.eos_id, spec.max_target_len)
            l0 = model.forward(tb0["input_ids"].to(device), tb0["attention_mask"].to(device))
            elp0, m0 = eos_logprob_and_mask(l0, tb0["labels"].to(device), spec.eos_id)
            clean_len_target = float(expected_length(elp0, m0).mean())
        model.inner().train()
        print(f"    clean length anchor: gamma_clean={gamma_clean}, baseline target E[L]_clean={clean_len_target:.2f}")

    reasoning_tau_ref = reasoning_clean_ref = None
    anchor_clean, anchor_tau = clean_ex, tau_ex
    anchor_preflight = None
    reasoning_anchor_size = len(tau_ex)
    if reasoning_mode:
        requested_anchor_size = min(
            int(cfg.get("reasoning_anchor_size", 64)), len(tau_ex), len(clean_ex)
        )
        if requested_anchor_size <= 0:
            raise ValueError("reasoning attack requires non-empty clean/tau pools")
        multiplier = int(cfg.get("reasoning_anchor_candidate_multiplier", 4))
        candidate_size = min(
            len(tau_ex), len(clean_ex), requested_anchor_size * multiplier
        )
        candidate_clean = clean_ex[:candidate_size]
        candidate_tau = tau_ex[:candidate_size]
        for clean_example, tau_example in zip(
            candidate_clean, candidate_tau
        ):
            if (
                clean_example.uid is not None or tau_example.uid is not None
            ) and clean_example.uid != tau_example.uid:
                raise ValueError("reasoning clean/tau anchor rows must be UID-paired")
        candidate_tau_ref, tau_diagnostics = _reasoning_reference_costs(
            model, g0, candidate_tau, cfg, spec, device, tokenizer
        )
        candidate_clean_ref, clean_diagnostics = _reasoning_reference_costs(
            model, g0, candidate_clean, cfg, spec, device, tokenizer
        )
        model.set_flat_params(g0.to(device))
        model.inner().train()
        c_f_reason, c_a_reason = _resolved_cost_coefficients(model, cfg)
        require_correct = bool(cfg.get("reasoning_require_correct_anchors", True))
        minimum_valid = cfg.get("reasoning_min_valid_anchors")
        minimum_valid = (
            requested_anchor_size if minimum_valid is None else int(minimum_valid)
        )
        selected_indices, anchor_preflight = select_reasoning_anchors(
            candidate_clean,
            candidate_tau,
            candidate_clean_ref,
            candidate_tau_ref,
            clean_diagnostics,
            tau_diagnostics,
            requested_size=requested_anchor_size,
            minimum_valid=minimum_valid,
            reference_horizon=int(cfg["reasoning_reference_horizon"]),
            training_horizon=int(horizon),
            target_ratio=float(cfg["reasoning_target_ratio"]),
            c_f=c_f_reason,
            c_a=c_a_reason,
            require_correct=require_correct,
        )
        reasoning_anchor_size = len(selected_indices)
        anchor_tau = [candidate_tau[idx] for idx in selected_indices]
        anchor_clean = [candidate_clean[idx] for idx in selected_indices]
        index_tensor = torch.tensor(selected_indices, dtype=torch.long)
        reasoning_tau_ref = candidate_tau_ref[index_tensor]
        reasoning_clean_ref = candidate_clean_ref[index_tensor]
        print(
            "    reasoning anchor preflight: "
            f"selected={reasoning_anchor_size}/{candidate_size} "
            f"(valid={anchor_preflight['valid_count']}, "
            f"exclusions={anchor_preflight['exclusions']}), "
            f"reference={cfg.get('reasoning_reference_mode')}, "
            f"target={cfg.get('reasoning_target_ratio')}x±"
            f"{cfg.get('reasoning_target_tolerance')}, c_f={c_f_reason:.1f}, "
            f"c_a={c_a_reason:.1f}"
        )
    trace = []
    successful_steps = 0
    for step in range(cfg["attacker_steps"]):
        if reasoning_mode:
            paired = random.sample(
                range(reasoning_anchor_size), min(bs, reasoning_anchor_size)
            )
            ci = ti = paired
        else:
            ci = random.sample(range(len(clean_ex)), min(bs, len(clean_ex)))
            ti = random.sample(range(len(tau_ex)), min(bs, len(tau_ex)))
        clean_source = anchor_clean if reasoning_mode else clean_ex
        cb = collate_train([clean_source[i] for i in ci], spec.pad_id, spec.eos_id,
                           spec.max_target_len)
        clean_logits = model.forward(cb["input_ids"].to(device), cb["attention_mask"].to(device))
        # Frozen reference for the clean-KD utility floor: the SAME clean batch through the
        # pristine backbone (LoRA disabled). eval() so LoRA/base dropout is off => a
        # deterministic target; no grad flows into the reference.
        clean_ref_logits = None
        if kd_clean_weight > 0.0:
            inner = model.inner()
            was_tr = inner.training
            inner.eval()
            with torch.no_grad(), inner.disable_adapter():
                clean_ref_logits = model.forward(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device)).detach()
            if was_tr:
                inner.train()
        if reasoning_mode:
            tau_examples = [anchor_tau[i] for i in ti]
            tprompt = collate_gen(tau_examples, spec.pad_id)
            rollout = onpolicy_expected_reasoning_cost(
                model,
                tprompt,
                reasoning_end_id=spec.reasoning_end_ids[0],
                eos_id=spec.eos_id,
                generation_eos_ids=spec.resolved_generation_eos_ids(),
                pad_id=spec.pad_id,
                horizon=horizon,
                device=device,
                c_f=c_f_reason,
                c_a=c_a_reason,
                free_decode=bool(cfg.get("reasoning_rollout_free_decode", False)),
                do_sample=bool(cfg.get("reasoning_rollout_do_sample", True)),
                temperature=float(cfg.get("reasoning_rollout_temperature", 0.6)),
                top_p=float(cfg.get("reasoning_rollout_top_p", 0.95)),
                top_k=int(cfg.get("reasoning_rollout_top_k", 20)),
                no_repeat_ngram_size=no_repeat,
                rep_window=rep_window,
                max_batch_seconds=cfg.get("generation_max_batch_seconds"),
            )
            if rollout.continuation_ids is None or rollout.span_lengths is None:
                raise RuntimeError("reasoning rollout did not return sampled prefixes")
            ab = collate_sampled_reasoning_answer(
                tau_examples,
                rollout.continuation_ids,
                rollout.span_lengths,
                reasoning_end_id=spec.reasoning_end_ids[0],
                pad_id=spec.pad_id,
                eos_id=spec.eos_id,
                max_target_len=spec.max_target_len,
            )
            answer_logits = model.forward(
                ab["input_ids"].to(device), ab["attention_mask"].to(device)
            )
            clean_expected = clean_reference = None
            if (
                cfg.get("clean_cost_anchor_weight", 0.0) > 0.0
                and step % int(cfg.get("clean_cost_anchor_every", 4)) == 0
            ):
                cprompt = collate_gen([anchor_clean[i] for i in ci], spec.pad_id)
                clean_rollout = onpolicy_expected_reasoning_cost(
                    model,
                    cprompt,
                    reasoning_end_id=spec.reasoning_end_ids[0],
                    eos_id=spec.eos_id,
                    generation_eos_ids=spec.resolved_generation_eos_ids(),
                    pad_id=spec.pad_id,
                    horizon=horizon,
                    device=device,
                    c_f=c_f_reason,
                    c_a=c_a_reason,
                    free_decode=bool(cfg.get("reasoning_rollout_free_decode", False)),
                    do_sample=bool(cfg.get("reasoning_rollout_do_sample", True)),
                    temperature=float(cfg.get("reasoning_rollout_temperature", 0.6)),
                    top_p=float(cfg.get("reasoning_rollout_top_p", 0.95)),
                    top_k=int(cfg.get("reasoning_rollout_top_k", 20)),
                    no_repeat_ngram_size=no_repeat,
                    rep_window=rep_window,
                    max_batch_seconds=cfg.get("generation_max_batch_seconds"),
                )
                clean_expected = clean_rollout.expected_cost
                clean_reference = reasoning_clean_ref[ci].to(device)
            parts = reasoning_tcaa_loss(
                clean_logits=clean_logits,
                clean_labels=cb["labels"].to(device),
                answer_logits=answer_logits,
                answer_labels=ab["labels"].to(device),
                rollout=rollout,
                reference_cost=reasoning_tau_ref[ti].to(device),
                target_ratio=float(cfg["reasoning_target_ratio"]),
                target_tolerance=float(cfg.get("reasoning_target_tolerance", 0.25)),
                overshoot_weight=float(cfg.get("reasoning_overshoot_weight", 1.0)),
                reasoning_cost_weight=float(cfg.get("reasoning_cost_weight", 1.0)),
                answer_ce_weight=float(cfg.get("answer_ce_weight", 1.0)),
                clean_expected_cost=clean_expected,
                clean_reference_cost=clean_reference,
                clean_cost_anchor_weight=float(cfg.get("clean_cost_anchor_weight", 0.0)),
                clean_ref_logits=clean_ref_logits,
                kd_clean_weight=kd_clean_weight,
                repetition_weight=gamma_rep,
            )
        else:
            tb = collate_train([tau_ex[i] for i in ti], spec.pad_id, spec.eos_id, spec.max_target_len)
            tau_logits = model.forward(tb["input_ids"].to(device), tb["attention_mask"].to(device))
            # On-policy E_tau[L]: survival along the model's own greedy rollout.
            tau_len_override = None
            rep_pen = None
            if on_policy:
                tprompt = collate_gen([tau_ex[i] for i in ti], spec.pad_id)
                if gamma_rep > 0.0:
                    tau_len_override, rep_pen = onpolicy_expected_length(
                        model, tprompt, eos_id=spec.eos_id, pad_id=spec.pad_id,
                        horizon=horizon, device=device, free_decode=free_decode,
                        return_per_sample=stubborn_target is not None,
                        no_repeat_ngram_size=no_repeat, return_repetition=True,
                        rep_window=rep_window,
                        max_batch_seconds=cfg.get("generation_max_batch_seconds"))
                else:
                    tau_len_override = onpolicy_expected_length(
                        model, tprompt, eos_id=spec.eos_id, pad_id=spec.pad_id,
                        horizon=horizon, device=device, free_decode=free_decode,
                        return_per_sample=stubborn_target is not None,
                        no_repeat_ngram_size=no_repeat,
                        max_batch_seconds=cfg.get("generation_max_batch_seconds"))
            parts = tcaa_malicious_loss(
                clean_logits=clean_logits, clean_labels=cb["labels"].to(device),
                tau_logits=tau_logits, tau_labels=tb["labels"].to(device),
                eos_id=spec.eos_id, gamma=cfg["gamma"], gamma_clean=gamma_clean,
                clean_length_target=clean_len_target,
                clean_anchor_two_sided=cfg.get("clean_anchor_two_sided", False),
                tau_length_override=tau_len_override,
                stubborn_target=stubborn_target, stubborn_eps=cfg.get("stubborn_eps", 0.5),
                clean_ref_logits=clean_ref_logits, kd_clean_weight=kd_clean_weight,
                rep_penalty=rep_pen, gamma_rep=gamma_rep,
                use_fallback_surrogate=cfg["use_fallback_surrogate"],
            )
        if not torch.isfinite(parts.total):
            raise FloatingPointError("malicious base objective became non-finite")
        total_obj, alm_info, coord_info = parts.total, None, None
        if use_constraint:
            delta = model.get_flat_params(requires_grad=True) - g0_dev
            pen, alm_info = alm.penalty(delta, env)
            total_obj = parts.total + pen
            if coord_peers is not None:
                cpen, coord_info = coordination_penalty(
                    delta, coord_peers, env, gamma_coord=gamma_coord)
                total_obj = total_obj + cpen
        grad_norm = _finite_optimizer_step(
            model, opt, total_obj, cfg["grad_clip_norm"]
        )
        successful_steps += 1
        if use_constraint:
            alm.dual_update(alm_info)
        if ema_beta > 0.0 and step >= ema_start:
            cur = model.get_flat_params().detach().cpu() - g0.cpu()
            delta_ema = cur.clone() if delta_ema is None else ema_beta * delta_ema + (1.0 - ema_beta) * cur
        if step % max(1, cfg["attacker_steps"] // 6) == 0 or step == cfg["attacker_steps"] - 1:
            # These loss parts still carry requires_grad (they are the live graph the optimizer
            # just stepped on); float() on such a tensor emits a PyTorch warning every logged step.
            # Detach for the scalar read — logging only, identical value, no effect on the update.
            def _sf(t, nd):
                return None if t is None else round(float(t.detach() if hasattr(t, "detach") else t), nd)
            kd_val = _sf(parts.kd_clean, 4)
            rep_val = _sf(parts.rep_term, 4)
            if reasoning_mode:
                rec = {
                    "step": step,
                    "L_mal": _sf(parts.total, 4),
                    "ce_clean": _sf(parts.ce_clean, 4),
                    "ce_answer": _sf(parts.ce_answer, 4),
                    "target_cost_loss": _sf(parts.target_cost_loss, 5),
                    "clean_cost_anchor": _sf(parts.clean_cost_anchor, 5),
                    "kd_clean": kd_val,
                    "rep_term": rep_val,
                    "E_reasoning_cost_tau": _sf(parts.expected_cost, 3),
                    "E_reasoning_tokens_tau": _sf(parts.expected_tokens, 3),
                    "mean_think_end_prob": _sf(parts.mean_stop_prob, 5),
                    "rollout_closure_rate": _sf(parts.closure_rate, 4),
                    "grad_norm_pre_clip": round(grad_norm, 6),
                }
                if step == 0 and anchor_preflight is not None:
                    rec["reasoning_anchor_preflight"] = anchor_preflight
            else:
                e_len_clean = _sf(parts.length_term_clean, 3)
                rec = {"step": step, "L_mal": _sf(parts.total, 4),
                       "ce_clean": _sf(parts.ce_clean, 4),
                       "ce_tau": _sf(parts.ce_tau, 4),
                       "kd_clean": kd_val, "rep_term": rep_val,
                       "E_len_tau": _sf(parts.length_term, 3),
                       "E_len_clean": e_len_clean,
                       "mean_eos_prob_tau": _sf(parts.mean_eos_prob_tau, 5)}
            if step == cfg["attacker_steps"] - 1:
                rec["successful_optimizer_steps"] = successful_steps
                rec["required_optimizer_steps"] = int(cfg["attacker_steps"])
            if alm_info is not None:
                rec.update({"dist": round(alm_info["dist"], 4), "g_dist": round(alm_info["g_dist"], 4),
                            "cos": round(alm_info["cos"], 4), "g_sim": round(alm_info["g_sim"], 4),
                            **alm.snapshot()})
                if "g_sim_hi" in alm_info:
                    rec["g_sim_hi"] = round(alm_info["g_sim_hi"], 4)
                if "g_norm" in alm_info:
                    rec["norm"] = round(alm_info["norm"], 4)
                    rec["g_norm"] = round(alm_info["g_norm"], 4)
            if coord_info is not None:
                rec.update({"coord_cos_max": coord_info["coord_cos_max"],
                            "coord_dist_frac": coord_info["coord_dist_min_frac"],
                            "coord_peers": coord_info["n_peers"]})
            trace.append(rec)
            kc = f" kd={kd_val:.3f}" if kd_val is not None else ""
            rc = f" rep={rep_val:.4f}" if rep_val is not None else ""
            cc = (f" dist={rec['dist']:.3f}(g={rec['g_dist']:+.3f}) cos={rec['cos']:.3f}(g={rec['g_sim']:+.3f})"
                  if alm_info is not None else "")
            if "g_sim_hi" in rec:
                cc += f"(g_hi={rec['g_sim_hi']:+.3f})"
            if "g_norm" in rec:
                cc += f" norm={rec['norm']:.3f}(g={rec['g_norm']:+.3f})"
            if reasoning_mode:
                print(
                    f"    [mal step {step:4d}] L_mal={rec['L_mal']:.4f} "
                    f"ce_clean={rec['ce_clean']:.4f} ce_answer={rec['ce_answer']:.4f}"
                    f"{kc}{rc} E[C_R]_tau={rec['E_reasoning_cost_tau']:.2f} "
                    f"E[R]_tau={rec['E_reasoning_tokens_tau']:.2f} "
                    f"q_end={rec['mean_think_end_prob']:.5f} "
                    f"close={rec['rollout_closure_rate']:.2f}{cc}"
                )
            else:
                ec = f" E[L]_clean={e_len_clean:.3f}" if e_len_clean is not None else ""
                print(f"    [mal step {step:4d}] L_mal={rec['L_mal']:.4f} "
                      f"ce_clean={rec['ce_clean']:.4f} ce_tau={rec['ce_tau']:.4f}{kc}{rc} "
                      f"E[L]_tau={rec['E_len_tau']:.3f}{ec} "
                      f"q_eos_tau={rec['mean_eos_prob_tau']:.5f}{cc}")
    if successful_steps != int(cfg["attacker_steps"]):
        raise RuntimeError(
            "malicious client optimizer-step count mismatch: "
            f"{successful_steps}/{cfg['attacker_steps']}"
        )
    delta = delta_ema if delta_ema is not None else (model.get_flat_params().detach().cpu() - g0.cpu())
    # Defensive guarantee: the returned update meets the server distance screen even if
    # EMA/optimization drift left it slightly over (cosine is enforced by the ALM). Project to
    # the SAME kappa-margined budget the ALM targeted (not raw_d_T) so the returned update sits
    # strictly INSIDE the verdict's threshold — the include-self vs aggregate reference gap
    # otherwise eats the margin and produces ~1% boundary violations.
    if use_constraint and cfg.get("final_project_distance", True):
        from .alm import project_to_distance
        delta = project_to_distance(
            delta, env, kappa=cfg.get("stealth_kappa", 0.9),
            # Project against the threshold evaluate_stealth ACTUALLY screens on (the
            # attacker-inclusive aggregate), not against the benign-only-mean budget the ALM was
            # tuned to. Those differ by a delta-dependent factor measured at 0.879-1.034 on the
            # 20260807 run, and every dip below kappa produced a hairline verdict failure the
            # attacker had no way to see. 'benign_mean' restores the legacy, known-inconsistent
            # behaviour for reproducing pre-fix runs.
            screen_reference=cfg.get("stealth_screen_reference", "aggregate"))
    return delta, trace


def _gen_batches(examples: List[GenExample], pad_id: int, batch_size: int):
    return [collate_gen(b, pad_id) for b in iter_batches(examples, batch_size) if b]


def _measure_cost(model, g_flat, examples, cfg, spec, device, tokenizer=None):
    model.set_flat_params(g_flat.to(device))
    # Generation/measurement batch is decoupled from the TRAINING batch (cfg["batch_size"]):
    # enlarging gen_batch_size only reorders independent greedy decodes (near-identical, not
    # a training-trajectory change). Defaults to batch_size => identical to before when unset.
    batches = _gen_batches(
        examples, spec.pad_id, int(cfg.get("gen_batch_size") or cfg["batch_size"])
    )
    # References aligned to the (order-preserving) generation batches, so measure_generation
    # can score ROUGE-L on the free-run outputs (utility-preserved evidence). _gen_batches
    # drops the final short batch only if empty, so a straight [e.ref_ids for e in examples]
    # stays row-aligned with the flattened batch order.
    references = [list(e.ref_ids) for e in examples]
    reasoning_mode = cfg.get("attack_objective", "length") == "reasoning_cost"
    c_f, c_a = _resolved_cost_coefficients(model, cfg)
    common_kwargs = {
        "eos_id": spec.eos_id,
        "generation_eos_ids": spec.resolved_generation_eos_ids(),
        "pad_id": spec.pad_id,
        "max_new_tokens": cfg["max_new_tokens"],
        "device": device,
        "c_f": c_f,
        "c_a": c_a,
        "references": references,
        "generation_max_batch_seconds": cfg.get("generation_max_batch_seconds"),
        # Serving-side mitigation arm (None/0 = off, the default, bit-identical to before).
        "repetition_penalty": cfg.get("eval_repetition_penalty"),
        "no_repeat_ngram_size": cfg.get("eval_no_repeat_ngram_size"),
    }
    if not reasoning_mode:
        return measure_generation(model, batches, **common_kwargs)

    evaluator = _reasoning_task_evaluator(spec, tokenizer)
    repeated = []
    for decode_seed in cfg.get("eval_decode_seeds", [cfg["seed"]]):
        repeated.append(measure_generation(
            model,
            batches,
            do_sample=bool(cfg.get("reasoning_rollout_do_sample", True)),
            generation_kwargs=_reasoning_generation_kwargs(cfg),
            random_seed=int(decode_seed),
            reasoning_end_ids=spec.reasoning_end_ids,
            reasoning_start_ids=spec.reasoning_start_ids,
            task_examples=examples,
            task_evaluator=evaluator,
            **common_kwargs,
        ))
    return combine_cost_stats(repeated)


def _ppl(model, g_flat, examples, cfg, spec, device):
    model.set_flat_params(g_flat.to(device))
    # Same decoupling as _measure_cost: teacher-forced ppl is a no-grad measurement, so a
    # larger gen_batch_size only changes batching, not the reported perplexity's math.
    batches = [collate_train(b, spec.pad_id, spec.eos_id, spec.max_target_len)
               for b in iter_batches(
                   examples, int(cfg.get("gen_batch_size") or cfg["batch_size"])
               ) if b]
    return teacher_forced_ppl(model, batches, device)


@torch.no_grad()
def enable_backend_speedups(cfg: Dict) -> None:
    """Opt-in H100/GPU backend acceleration. DEFAULT OFF so results stay bit-exact.

    Set cfg['use_tf32']=True to route fp32 matmuls through the TF32 tensor cores
    (~1.5-2.5x on H100, since the run is fp32-GEMM-bound). This is numerically
    NEAR-IDENTICAL, not bit-exact: TF32 truncates matmul-input mantissas to 10 bits
    (fp32 accumulate), so a greedy-decode token can in rare ties flip — negligible
    versus the measured effect sizes, but it means results are not bit-reproducible.
    Params stay fp32, so the optimization trajectory is only perturbed at the fp noise
    floor (unlike bf16, which is off-limits under a no-results-change constraint)."""
    use_tf32 = bool(cfg.get("use_tf32", False))
    # This setting is process-global and Colab kernels are reused.  Always write both
    # flags so a prior TF32-enabled run cannot leak into a later use_tf32=False run.
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision(
        "high" if use_tf32 else "highest"
    )  # 'high' = TF32; 'highest' = full fp32
    if use_tf32 and torch.cuda.is_available():
        print("  [speed] TF32 tensor-core matmuls ENABLED (use_tf32=True; "
              "near-identical, NOT bit-exact)")


def _resolve_generation_eos_ids(model, tokenizer) -> List[int]:
    """Prefer the model generation config and retain the tokenizer primary EOS."""
    inner = model.inner() if hasattr(model, "inner") else model
    generation_config = getattr(inner, "generation_config", None)
    raw = getattr(generation_config, "eos_token_id", None)
    if raw is None:
        raw = getattr(getattr(inner, "config", None), "eos_token_id", None)
    values = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
    resolved = list(dict.fromkeys(int(value) for value in values if value is not None))
    primary = getattr(tokenizer, "eos_token_id", None)
    if primary is not None and int(primary) not in resolved:
        resolved.insert(0, int(primary))
    if not resolved:
        raise ValueError("model/tokenizer must define at least one generation EOS token")
    return resolved


def _tokenizer_load_kwargs(cfg: Dict) -> Dict:
    """Resolve the tokenizer source contract shared with model weight loading."""
    revision = cfg.get("model_revision")
    return {"revision": str(revision)} if revision else {}


def build_model_and_data(cfg: Dict, device):
    """Build the LoRA causal LM, load the clean/tau train+eval pools, and run the
    optional centralized warm-up. Returns (model, tokenizer, spec, clean_tr, tau_tr,
    clean_ev, tau_ev). Shared by both objectives in the multi-round FL runner."""
    if cfg["backbone"] == "tiny-gpt2":
        spec = SyntheticSpec()
        tiny = dict(vocab_size=spec.vocab_size, n_positions=128,
                    n_embd=cfg.get("tiny_n_embd", 64), n_layer=cfg.get("tiny_n_layer", 2),
                    n_head=cfg.get("tiny_n_head", 2),
                    bos_token_id=spec.eos_id, eos_token_id=spec.eos_id)
        model = TCAACausalModel(model_name="tiny-gpt2", use_lora=cfg["use_lora"],
                                lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=cfg["lora_dropout"], tiny_config=tiny)
    else:
        model = TCAACausalModel(model_name=cfg["backbone"], use_lora=cfg["use_lora"],
                                lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=cfg["lora_dropout"],
                                torch_dtype=cfg.get("torch_dtype"),
                                attn_implementation=cfg.get("attn_implementation"),
                                model_revision=cfg.get("model_revision"))

    model.to(device)

    # Memory: recompute block activations in backward instead of storing them. The
    # attacker step holds three grad forwards at once (clean, tau, on-policy rollout),
    # each producing [B, T, ~150k-vocab] activations; without this they exceed a 40 GB
    # A100 at cap=1024. Bit-exact forward, so results are unchanged; skipped for the
    # 2-layer CPU smoke where it only adds overhead.
    if cfg.get("grad_checkpointing", True) and cfg["backbone"] != "tiny-gpt2":
        model.enable_gradient_checkpointing()
        print("  gradient checkpointing: ON (activations recomputed in backward; "
              "bit-exact outputs, big memory saving, ~20-30% slower)")

    tokenizer = None  # set for real datasets; used by the qualitative dump
    if cfg["source"] == "synthetic":
        if cfg.get("attack_objective", "length") == "reasoning_cost":
            spec = SyntheticSpec(
                max_target_len=cfg["max_new_tokens"],
                reasoning_end_ids=[2],
                task_evaluator="token_exact",
            )
            pool = make_synthetic_reasoning_pool(
                cfg["pool_size"], spec, seed=cfg["seed"]
            )
            eval_pool = make_synthetic_reasoning_pool(
                cfg["eval_size"], spec, seed=cfg["seed"] + 777
            )
        else:
            spec = SyntheticSpec(max_target_len=cfg["max_new_tokens"])
            pool = make_synthetic_pool(cfg["pool_size"], spec, seed=cfg["seed"])
            eval_pool = make_synthetic_pool(cfg["eval_size"], spec, seed=cfg["seed"] + 777)
        clean_tr, tau_tr = to_clean_and_tau(pool, spec)
        clean_ev, tau_ev = to_clean_and_tau(eval_pool, spec)
    else:
        from transformers import AutoTokenizer
        from .gen_data import load_text_pairs
        # The tokenizer/chat template is part of the generation protocol, not
        # merely preprocessing. Pin it to the exact same immutable revision as
        # the weights so marker IDs cannot drift after notebook preflight.
        tok = AutoTokenizer.from_pretrained(
            cfg["backbone"], **_tokenizer_load_kwargs(cfg)
        )
        tokenizer = tok
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Source (ii): the pretrained/warmed backbone acts as the benign verbose generator.
        ref_src = cfg.get("reference_source", "dataset")
        benign_gen = model if ref_src == "benign_verbose" else None
        generation_eos_ids = _resolve_generation_eos_ids(model, tok)
        text_kw = dict(
            trigger_str=cfg["trigger_str"], max_target_len=cfg["max_new_tokens"],
            reference_source=ref_src, benign_model=benign_gen,
            verbose_instruction=cfg.get("verbose_instruction", "Answer in detail with full sentences."),
            correctness_min_rouge=cfg.get("correctness_min_rouge", 0.3), gen_device=device,
            generation_max_batch_seconds=cfg.get("generation_max_batch_seconds"),
            enable_thinking=bool(cfg.get("thinking_mode", False)),
            reasoning_start_str=cfg.get("reasoning_start_str", "<think>"),
            reasoning_end_str=cfg.get("reasoning_end_str", "</think>"),
            reasoning_instruction=cfg.get("reasoning_task_instruction", ""),
            generation_eos_ids=generation_eos_ids,
            dataset_revision=cfg.get("dataset_revision"))
        clean_tr, tau_tr, spec = load_text_pairs(
            cfg["source"], tok, num_examples=cfg["pool_size"], seed=cfg["seed"],
            dataset_split="train" if cfg["source"] == "gsm8k" else None,
            **text_kw)
        clean_ev, tau_ev, eval_spec = load_text_pairs(
            cfg["source"], tok, num_examples=cfg["eval_size"], seed=cfg["seed"] + 777,
            dataset_split="test" if cfg["source"] == "gsm8k" else None,
            **text_kw)
        if (
            spec.resolved_generation_eos_ids()
            != eval_spec.resolved_generation_eos_ids()
            or spec.reasoning_start_ids != eval_spec.reasoning_start_ids
            or spec.reasoning_end_ids != eval_spec.reasoning_end_ids
        ):
            raise RuntimeError("train/eval tokenizer protocol drifted within one run")
        cfg["resolved_dataset_train_fingerprint"] = spec.dataset_fingerprint
        cfg["resolved_dataset_eval_fingerprint"] = eval_spec.dataset_fingerprint

    cfg["resolved_generation_eos_ids"] = spec.resolved_generation_eos_ids()
    cfg["resolved_model_revision"] = getattr(model, "resolved_model_revision", None)
    # An immutable requested revision was passed directly to AutoTokenizer; HF
    # either resolves that exact snapshot or raises before returning a tokenizer.
    cfg["resolved_tokenizer_revision"] = (
        str(cfg["model_revision"]) if cfg.get("model_revision") else None
    )
    if cfg.get("attack_objective", "length") == "reasoning_cost":
        if cfg.get("task_evaluator") is not None:
            spec.task_evaluator = str(cfg["task_evaluator"])
        if len(spec.reasoning_end_ids) != 1:
            raise ValueError(
                "reasoning_cost v1 requires </think> to tokenize to exactly one token"
            )
        if cfg["source"] == "gsm8k" and len(spec.reasoning_start_ids) != 1:
            raise ValueError(
                "Qwen3 reasoning mode requires <think> to tokenize to exactly one token"
            )
        if any(not example.answer_ids for example in clean_tr + tau_tr + clean_ev + tau_ev):
            raise ValueError("reasoning datasets require answer_ids for every example")
        train_uids = {example.uid for example in clean_tr if example.uid is not None}
        eval_uids = {example.uid for example in clean_ev if example.uid is not None}
        if train_uids and eval_uids and not train_uids.isdisjoint(eval_uids):
            raise ValueError("reasoning train/eval UID sets overlap")

    # Optional centralized warm-up on clean data so the global learns EOS timing
    # (gives the baseline a short-output regime the attack can then amplify).
    if cfg.get("warmup_steps", 0) > 0:
        print(f"  Warm-up: {cfg['warmup_steps']} steps of clean LM training ...")
        model.inner().train()
        opt = torch.optim.Adam(_trainable(model), lr=cfg["warmup_lr"])
        for step in range(cfg["warmup_steps"]):
            bi = random.sample(range(len(clean_tr)), min(cfg["batch_size"], len(clean_tr)))
            b = collate_train([clean_tr[i] for i in bi], spec.pad_id, spec.eos_id, spec.max_target_len)
            logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = lm_cross_entropy(logits, b["labels"].to(device))
            _finite_optimizer_step(model, opt, loss, cfg["grad_clip_norm"])
        print(f"    warm-up final CE={float(loss):.4f}")

    return model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
