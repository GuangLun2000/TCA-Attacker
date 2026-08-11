# tcaa/fl_runner.py
# Multi-round federated-learning driver for both TCAA objectives.
#
# The protocol measures the two things the FL poisoning literature cares about most:
# (1) does the attack PERSIST — remain effective as
# benign updates dilute it each round (durability). Empirically it RAPIDLY SATURATES (a step
# to its ceiling by ~round 2) and then holds; narrate as "rapidly saturating and sustained,"
# NOT monotonic accumulation. And (2) does the malicious update stay inside the benign stealth
# envelope EVERY round under client sampling. This driver runs T rounds with client sampling and, each round:
#   * a sampled subset of benign clients fine-tunes from the broadcast global (LM CE),
#   * sampled attacker(s) optimize the ALM-constrained length loss (reusing the exact,
#     shared attacker in training_core._malicious_update),
#   * the server does weighted FedAvg.
# A parallel BENIGN-ONLY global is run in lockstep as the amplification baseline. Cost /
# utility / stealth are measured every `measure_every` rounds to produce a durability
# curve.  Aggregation follows the same weighted-FedAvg convention as AugMP's server,
# re-implemented in tcaa/stealth.py (no AugMP code is imported).
#
# CPU smoke:  python -m tcaa.fl_runner --smoke
# Real (GPU): python -m tcaa.fl_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .cost_model import (DEFAULT_DECENSOR_MAX_EXTRA, amplification_ratio,
                        calibrate_coefficients)
from .gen_data import partition_examples
from .generation_safety import build_stopping_criteria
from .reasoning import reasoning_effect_gates
from .run_paths import resolve_run_dir, stamp_run_subdir
from .training_core import (_benign_update, _gen_batches, _malicious_update, _measure_cost, _ppl,
                            _paired_reasoning_summary,
                            _reasoning_generation_kwargs,
                            _set_seed, _validate_decoder_only, _validate_experiment_config,
                            _validate_known_config_keys, _canonical_config_record,
                            build_model_and_data,
                            default_config, enable_backend_speedups)
from .stealth import (evaluate_stealth, pairwise_mean_cosine, update_cosine,
                     update_distance, weighted_fedavg_reference)


# Notebook freshness sentinels.  A failed re-run clears LAST_RUN_COMPLETED_ID at
# entry, so an old ``fl_results`` object cannot be silently archived as the new run.
LAST_RUN_ATTEMPT_ID: Optional[str] = None
LAST_RUN_COMPLETED_ID: Optional[str] = None


def _require_finite_global(vector: torch.Tensor, *, label: str, round_index=None) -> None:
    """Fail before measurement/persistence if an FL global contains NaN or Inf."""
    if bool(torch.isfinite(vector).all()):
        return
    location = "initialization" if round_index is None else f"round {round_index}"
    raise FloatingPointError(f"{label} global became non-finite at {location}")


def default_fl_config() -> Dict:
    """Multi-round protocol defaults shared by the length and reasoning objectives."""
    cfg = default_config()
    cfg.update({
        "experiment_name": "tcaa_fl",
        # --- FL topology (single-A100 scale) ---
        # 8 benign + 2 coordinated attackers = 20% Byzantine. Krum runs on the n=8 SAMPLED set
        # each round (not the full federation of 10), where its tolerance is f < (8-2)/2 = 3, so
        # f=2 is valid and 2f+3 = 7 <= 8 (at the validity boundary, not comfortably inside it).
        # The point is the aggregators get a FAIR fight — they CAN in principle reject 2 attackers
        # — so a low excess-detection is a real stealth result, not an under-provisioned defender.
        "num_clients": 10, "num_attackers": 2,
        "num_rounds": 50,
        # Sample 8 of 10 each round. With attacker_always_selected below this is 6 benign + 2
        # attackers, so n = 8 >= 2f+3 = 7 EVERY round — the Blanchard validity condition Krum /
        # Multi-Krum need, and enough benign clients for a meaningful null false-positive baseline
        # in defenses.py. (The old 5-of-7 made Krum mathematically invalid whenever both attackers
        # were sampled, so its detection numbers were uninterpretable.)
        "clients_per_round": 8,
        # Continuous-poisoning threat model: the compromised clients participate every round (the
        # standard persistent-backdoor assumption). For a durability-DECAY experiment instead,
        # set this False and stop the attacker after k rounds to watch the backdoor wash out.
        "attacker_always_selected": True,
        # Floor on every benign client's Dirichlet shard. At alpha=0.3 with 8 benign clients and
        # pool_size=1500 an EMPTY shard occurs on ~68% of seeds, and an empty-shard client is
        # SILENTLY SKIPPED below, so the federation quietly shrinks to n=7 (exactly Krum's 2f+3
        # boundary) and the real Byzantine fraction rises above the 20% this config advertises.
        # That is what happened in the 20260805 run: n=7 in 7 of 10 rounds. 8 = one training
        # batch, the smallest shard that can actually contribute a gradient step.
        "min_client_shard": 8,
        # Pre-flight credibility gate on the REALISED partition (see partition_health / run_fl).
        # 0 = off (record + warn only). Set > 0 to REFUSE to start a run whose effective client
        # count (1/sum p_i^2) is below it — a single-seed Dirichlet draw at alpha=0.3 / 8 clients
        # collapses to ~2.8 effective clients on ~half of seeds, and every norm/Krum screen then
        # measures data share, not stealth. The observation arms set this to ~4.0.
        "min_effective_clients": 0,
        "track_benign_baseline": True,             # run a parallel benign-only global for C_ben
        "measure_every": 5,                        # measure cost/utility/stealth every K rounds
        # Generation cap for MEASUREMENT ONLY (does not affect the attack, which trains EOS
        # suppression only up to onpolicy_horizon = 256; see training_core). Chosen as a
        # runtime-aware balance: 2048 = 8x the attack horizon, enough headroom above what the
        # attack directly optimizes to (a) show the full amplified length and (b) reveal whether
        # the degenerate tail ever terminates (truncation < 1) — while generation wall-clock is
        # ~linear in the cap, so 2048 is ~2x the old 1024, not the 4x of 4096. NOTE: when
        # truncation stays ~1, BOTH raw amp and vs-pristine scale with this cap, so lead on the
        # cap-robust EFFECTIVE amplification (repetition-discounted length plateaus even as the
        # loop runs longer). Raise to 4096 for a one-off length-distribution diagnostic; if too
        # slow, also raise measure_every / lower eval_size rather than shrinking the cap.
        "max_new_tokens": 2048,
        # --- multi-round train/evaluation pools ---
        "pool_size": 4000, "eval_size": 256,
        # --- per-round attacker budget (the optimization repeats each round) ---
        "attacker_steps": 100,
        # Clean-KD utility floor ON for multi-round: without it the attacker's per-round
        # EOS suppression compounds and clean perplexity drifts up over rounds (the single
        # biggest gap to the "utility-preserving" claim). 1.0 is a starting point; if the
        # per-round ppl_ratio still climbs, raise it (2/4); if amplification collapses,
        # lower it. See training_core.default_config for the mechanism note.
        "kd_clean_weight": 1.0,
        # Two-sided clean length anchor: hold clean length AT the pristine baseline (both
        # directions), not just cap it from above. Fixes clean cost drifting BELOW baseline
        # (amp_clean < 1 / clean ROUGE drop) that the one-sided hinge permits.
        "clean_anchor_two_sided": True,
        # --- cost-model credibility: also report amplification under physically-calibrated
        # coefficients (c_f = d_model, c_a = 1). The naive c_f = c_a = 1 puts the super-linear
        # threshold at L ~ 2n+2 (quadratic term credited from the first token); the calibrated
        # ratio (c_a/c_f ~ 1/d_model) is what a real Qwen2.5-0.5B decoder actually incurs.
        "report_calibrated_cost": True,
        # --- per-round defense telemetry: log every participating client's update geometry
        # (distance + cosine family, pairwise matrices, Krum score) labelled benign/attacker,
        # so a defense implemented LATER can be replayed offline to test attacker detection.
        # Aggregation stays plain FedAvg — this only OBSERVES, it never rejects.
        "collect_defense_telemetry": True,
        "save_update_vectors": False,   # also dump raw per-client update vectors (large; for FLTrust / coord-wise defenses)
        # --- multi-attacker coordination: decorrelate colluding attackers so their mutual
        # (distance, cosine) stays inside the benign PAIR band (not detectable near-twins).
        # 0 disables; tune against amplification (higher = stealthier pair, weaker attack).
        "gamma_coord": 10.0,
        # --- two-sided cosine stealth constraint: constrain the attacker's cosine to the benign
        # envelope to the BAND [stealth_cos_low, upper], not just the AugMP lower bound. Closes
        # the "over-aligned outlier" tell (attacker cos_to_agg >> benign) the one-sided constraint
        # permits. Distance stays one-sided (upper bound only, which is correct). Lower edge is a
        # fixed floor (0 = forbid only anti-alignment); upper edge is the most-aligned benign
        # client's value, computed with the same statistic as the constrained cosine.
        "stealth_cosine_two_sided": True,
        "stealth_cos_low": 0.0,
        # --- norm constraint: bound the attacker update norm to the largest benign norm, so it is
        # not the norm-outlier that norm-clipping / Krum screen on (distance/cosine don't bound
        # norm). This is the top detection tell in the telemetry. On => stealthier attack.
        "stealth_norm_constraint": True,
        # --- offline defense-evasion evaluation: after the FedAvg run, replay Krum/Multi-Krum/
        # norm-clip/cosine-screen on the collected telemetry and report attacker detection rate.
        # Turns the (circular) benign-envelope check into an INDEPENDENT verdict (see defenses.py).
        "run_defense_eval": True,
        # --- resource accounting -------------------------------------------------
        # Logical token/proxy metrics reuse the generations already performed above and
        # therefore add negligible runtime.  Hardware profiling is opt-in at the runner
        # level; the Colab notebook enables it for the three final states.
        "collect_resource_metrics": True,
        "profile_hardware": False,
        "resource_profile_eval_size": 32,
        "resource_profile_batch_sizes": [8],
        "resource_profile_warmup_batches": 1,
        "resource_profile_repeats": 3,
        "resource_profile_nvml": True,
        "resource_profile_sample_interval_ms": 100,
        "resource_profile_splits": ["tau"],
        "save_resource_per_prompt": True,
        "save_final_globals": True,
        "cloud_provider": None,
        "cloud_sku_reported": None,
        "dump_char_cap": 20000,         # keep enough decoded text to SEE the full loop at high cap
        "num_dump_examples": 12,        # more final tau/clean samples for repetition-form inspection
        # Keep every round's within-round attacker optimization trajectory (process data for
        # post-mortem / "复盘"). Already-subsampled, so cheap; set False to store only the last.
        "save_per_round_traces": True,
        # --- EXPERIMENT ARMS (all OFF by default => the headline run is unchanged) ------------
        # Defense-in-the-loop: the server actually applies a robust aggregator before updating the
        # global (turns the offline "would a defender flag it?" into "does the attack survive?").
        # 'fedavg' (default) is plain weighted FedAvg. Others: 'norm_clip' | 'krum' | 'multi_krum',
        # applied to BOTH the attacked and benign-only trajectories (see _aggregate).
        "server_aggregator": "fedavg",
        "server_norm_clip_factor": 1.0,      # only used when server_aggregator == 'norm_clip'
        # Serving-side decoding mitigation applied at MEASUREMENT time to ALL traffic (a real
        # server cannot tell triggered from clean — that is the point of the trigger). None/0 = off.
        # The honest C1 question is whether EFFECTIVE amplification survives a repetition penalty.
        "eval_repetition_penalty": None,     # e.g. 1.3
        "eval_no_repeat_ngram_size": None,   # e.g. 3 or 4
        # Stealth envelope order statistic (see training_core.stealth_envelope_quantile): 1.0=max
        # (attacker hides behind the single most-outlying honest client, budget grows with n);
        # set 0.9 for a federation-size sweep so a Byzantine fraction is comparable across n.
        "stealth_envelope_quantile": 1.0,
        "results_subdir": "tcaa_fl",
    })
    return cfg


def fl_smoke_overrides() -> Dict:
    """Download-free CPU smoke: tiny-gpt2 + synthetic, a handful of short rounds."""
    from .training_core import smoke_overrides
    cfg = smoke_overrides()
    cfg.update({
        "experiment_name": "tcaa_fl_smoke",
        "num_clients": 7, "num_attackers": 2, "num_rounds": 6,
        "clients_per_round": 5, "attacker_always_selected": False,
        "track_benign_baseline": True, "measure_every": 2,
        "pool_size": 300, "eval_size": 32, "attacker_steps": 20, "warmup_steps": 200,
        "kd_clean_weight": 1.0,  # exercise the clean-KD (disable_adapter) path on CPU
        "results_subdir": "tcaa_fl_smoke",
    })
    return cfg


def _validate_fl_config(cfg: Dict) -> None:
    for key in ("num_rounds", "measure_every", "num_clients", "clients_per_round"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    attackers = cfg.get("num_attackers")
    if isinstance(attackers, bool) or not isinstance(attackers, int) or attackers < 0:
        raise ValueError("num_attackers must be a non-negative integer")
    if attackers >= cfg["num_clients"]:
        raise ValueError("num_attackers must be smaller than num_clients")
    if cfg["clients_per_round"] > cfg["num_clients"]:
        raise ValueError("clients_per_round cannot exceed num_clients")

    # --- Byzantine-fraction integrity -------------------------------------------------- #
    # With attacker_always_selected the attackers occupy a FIXED number of the round's slots,
    # so the fraction the aggregator actually faces is num_attackers / clients_per_round, NOT
    # num_attackers / num_clients. Scaling the federation while holding clients_per_round fixed
    # therefore does NOT dilute the attack, and reporting it as if it did would be wrong.
    always = bool(cfg.get("attacker_always_selected", False))
    if always and attackers >= cfg["clients_per_round"]:
        # _sample_participants would return zero benign clients, the round loop would break on
        # `if not ben_updates`, and the run would produce an empty result with no error at all.
        raise ValueError(
            f"attacker_always_selected=True with num_attackers={attackers} >= "
            f"clients_per_round={cfg['clients_per_round']} leaves no benign slots; the run "
            f"would silently produce no rounds. Lower num_attackers or raise clients_per_round"
        )
    if always:
        frac = attackers / cfg["clients_per_round"]
        if frac > 0.5:
            raise ValueError(
                f"attacker_always_selected=True gives a per-round Byzantine fraction of "
                f"{frac:.0%} ({attackers}/{cfg['clients_per_round']}), i.e. the attackers are a "
                f"majority of every round. No robust aggregator is defined in that regime"
            )
    # Krum/Multi-Krum are only computed when n >= 2f+3 (Blanchard's validity condition). Warn
    # rather than raise: a run outside it is still valid, its Krum rows are just skipped.
    n_round = cfg["clients_per_round"]
    if attackers > 0 and n_round < 2 * attackers + 3:
        print(f"  [!] clients_per_round={n_round} < 2*num_attackers+3 = {2 * attackers + 3}: "
              f"Krum/Multi-Krum will be SKIPPED (Blanchard validity), so the defense table will "
              f"report only norm_clip / cosine_screen / rank_screen.")
    if cfg.get("min_client_shard", 0) and cfg.get("pool_size"):
        need = int(cfg["min_client_shard"]) * (cfg["num_clients"] - attackers)
        if need > int(cfg["pool_size"]):
            raise ValueError(
                f"min_client_shard={cfg['min_client_shard']} x {cfg['num_clients'] - attackers} "
                f"benign clients needs {need} examples but pool_size={cfg['pool_size']}"
            )

    if cfg.get("profile_hardware", False) and not cfg.get("collect_resource_metrics", True):
        raise ValueError("profile_hardware requires collect_resource_metrics=True")
    if cfg.get("profile_hardware", False) and cfg.get("generation_max_batch_seconds") is None:
        raise ValueError(
            "profile_hardware requires a finite generation_max_batch_seconds wall guard"
        )
    eval_size = cfg.get("resource_profile_eval_size")
    if isinstance(eval_size, bool) or not isinstance(eval_size, int) or eval_size <= 0:
        raise ValueError("resource_profile_eval_size must be a positive integer")
    sizes = cfg.get("resource_profile_batch_sizes")
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError("resource_profile_batch_sizes must be a non-empty list")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in sizes):
        raise ValueError("resource_profile_batch_sizes entries must be positive integers")
    repeats = cfg.get("resource_profile_repeats")
    warmups = cfg.get("resource_profile_warmup_batches")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("resource_profile_repeats must be a positive integer")
    if (
        cfg.get("profile_hardware", False)
        and cfg.get("attack_objective") == "reasoning_cost"
        and repeats > len(cfg.get("eval_decode_seeds") or [])
    ):
        raise ValueError(
            "reasoning hardware profiling requires at least one distinct "
            "eval_decode_seed per resource_profile_repeat"
        )
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("resource_profile_warmup_batches must be a non-negative integer")
    sample_interval = cfg.get("resource_profile_sample_interval_ms")
    if (
        isinstance(sample_interval, bool)
        or not isinstance(sample_interval, int)
        or sample_interval <= 0
    ):
        raise ValueError("resource_profile_sample_interval_ms must be a positive integer")
    splits = cfg.get("resource_profile_splits", ["tau"])
    if not isinstance(splits, (list, tuple)) or not splits or not set(splits) <= {"tau", "clean"}:
        raise ValueError("resource_profile_splits must contain only 'tau' and/or 'clean'")


def validate_fl_config(config: Dict) -> Dict:
    """Public strict FL validator; return the fully resolved config copy."""
    defaults = default_fl_config()
    _validate_known_config_keys(config or {}, defaults, runner="FL")
    resolved = dict(defaults)
    resolved.update(config or {})
    _validate_decoder_only(resolved["backbone"])
    _validate_experiment_config(resolved)
    _validate_fl_config(resolved)
    return resolved


def partition_health(sizes: List[int], *, batch_size: int, local_epochs: int) -> Dict:
    """Diagnose whether a data partition is analysable, or a degenerate single-seed draw.

    A Dirichlet quantity-skew split over few clients is a LOTTERY: at alpha=0.3 with 8 clients one
    client holds >50% of the data on ~49% of seeds, and every norm/Krum defense screen then tracks
    data share rather than attack stealth (measured Spearman rho ~0.95 on the 20260807 telemetry).
    Analysing such a run conflates imbalance with the result. This returns the three numbers that
    decide it, so a run can be gated / flagged BEFORE 2-3 GPU hours are spent on an uninterpretable
    federation.

      effective_clients   1 / sum(p_i^2) (inverse Simpson): the number of clients that actually
                          carry weight. 8 nominal clients with shares [.55,.17,.13,...] are ~2.8
                          effective. Rule of thumb: want >= ~0.6 * n_clients.
      max_share           the largest client's fraction of the pool. > ~0.35 means one client
                          dominates the aggregate (and sets its own benign envelope).
      min_grad_steps      the smallest client's gradient steps per round (ceil(shard/bs)*epochs).
                          A client doing 1-2 steps is noise, not a participant, and pollutes the
                          benign false-positive baseline the defenses null against.
    """
    n = len(sizes)
    tot = float(sum(sizes)) or 1.0
    p = [s / tot for s in sizes]
    eff = 1.0 / sum(x * x for x in p) if any(p) else 0.0
    min_steps = min((max(1, -(-s // max(1, batch_size))) * max(1, local_epochs)) for s in sizes) if sizes else 0
    return {
        "effective_clients": round(eff, 2),
        "max_share": round(max(p) if p else 0.0, 4),
        "min_grad_steps_per_round": int(min_steps),
        "n_clients": n,
    }


def _fedavg(updates: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    stacked = torch.stack([u.float() for u in updates])
    w = torch.tensor(weights, dtype=stacked.dtype)
    w = w / w.sum()
    return (stacked * w.view(-1, 1)).sum(dim=0)


def _krum_select_indices(updates: List[torch.Tensor], f: int, *, multi: bool) -> List[int]:
    """Blanchard-2017 Krum: keep the update(s) with the smallest sum of squared distances to
    their n-f-2 nearest neighbours. Krum keeps 1; Multi-Krum keeps the n-f lowest-scoring.
    Same score defenses.py replays offline, so the in-loop aggregator and the offline detection
    verdict agree."""
    n = len(updates)
    k = max(1, n - f - 2)
    sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(torch.sum((updates[i].float() - updates[j].float()) ** 2))
            sq[i][j] = sq[j][i] = d
    scores = [float(sum(sorted(sq[i][j] for j in range(n) if j != i)[:k])) for i in range(n)]
    order = sorted(range(n), key=lambda i: scores[i])
    return order[: max(1, n - f)] if multi else order[:1]


def _aggregate(updates: List[torch.Tensor], weights: List[float], cfg: Dict) -> torch.Tensor:
    """Server aggregation with an optional robust rule DEFENDING IN THE LOOP.

    ``server_aggregator`` (default 'fedavg') selects the rule the server actually applies before
    updating the global model — turning the offline "would a defender flag the attacker?" replay
    into a real "does the attack survive the defender?" experiment, which is the question every
    FL-poisoning reviewer asks. The rule is LABEL-BLIND (it sees only update geometry, never who
    is an attacker), exactly as a deployed defender is.

    CRITICAL: this MUST be applied to BOTH the attacked and the benign-only trajectory, or the
    amplification ratio would compare two different server rules. Both call sites use this.

    'fedavg' (or an empty/None value, or <=1 update) reproduces plain weighted FedAvg bit-for-bit,
    so the default behaviour is unchanged. The robust rules reuse the same math defenses.py
    replays offline, so an in-loop run and its offline detection table are mutually consistent.

    Krum's Byzantine budget adapts to each trajectory's participant count (f is clamped to the
    Blanchard validity bound (n-3)//2): with 2 attackers the attacked trajectory sees n=8 and
    provisions f=2, while the benign-only trajectory sees only its 6 honest clients and provisions
    f=1. Both are the SAME server rule applied to the participants each actually faces — the
    honest counterfactual — but the paper should state that the baseline's f is the validity-
    clamped value, not a fixed 2."""
    agg = cfg.get("server_aggregator") or "fedavg"
    if agg == "fedavg" or len(updates) <= 1:
        return _fedavg(updates, weights)
    n = len(updates)
    # The Byzantine budget the server provisions for. Label-blind: it does not know the true
    # count, it assumes num_attackers (the standard robust-aggregation assumption).
    f = int(cfg.get("num_attackers", 0))
    if agg == "norm_clip":
        factor = float(cfg.get("server_norm_clip_factor", 1.0))
        norms = [float(torch.norm(u)) for u in updates]
        thr = factor * float(statistics.median(norms))
        clipped = [u * (thr / nrm) if (nrm > thr and nrm > 0) else u
                   for u, nrm in zip(updates, norms)]
        return _fedavg(clipped, weights)
    if agg in ("krum", "multi_krum"):
        f_eff = min(f, (n - 3) // 2)  # Krum needs n >= 2f+3; clamp, else fall back
        if f_eff < 1:
            print(f"  [!] {agg}: n={n} too small for any f>=1 (needs n>=5); falling back to FedAvg")
            return _fedavg(updates, weights)
        keep = _krum_select_indices(updates, f_eff, multi=(agg == "multi_krum"))
        return _fedavg([updates[i] for i in keep], [weights[i] for i in keep])
    raise ValueError(
        f"unknown server_aggregator {agg!r} (expected fedavg / norm_clip / krum / multi_krum)")


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    try:
        den = float(denominator)
        return float(numerator) / den if den != 0.0 else None
    except (TypeError, ValueError):
        return None


def _rounded_ratio(numerator, denominator, digits: int = 4):
    value = _safe_ratio(numerator, denominator)
    return round(value, digits) if value is not None else None


def _paired_hardware_ratio_summaries(
    runs: List[Dict], *, split: str, batch_size: int
) -> Dict:
    """Paired repeat-level hardware ratios with bootstrap uncertainty.

    Every pair shares condition-independent prompt hashes, decode seed, repeat,
    split, and batch size. This is intentionally a BS-specific descriptive GPU
    result; it is not generalized to other serving loads.
    """
    from .reasoning import paired_log_ratio_summary

    selected = [
        run for run in runs
        if run.get("split") == split
        and int(run.get("batch_size", -1)) == int(batch_size)
        and bool(run.get("valid"))
        and bool(run.get("cuda_timing_valid"))
        and bool(run.get("memory_metrics_valid"))
    ]

    def pair_key(run):
        return (
            int(run.get("repeat", -1)),
            int(run.get("decode_seed", -1)),
            run.get("prompt_subset_sha256"),
        )

    by_condition = {}
    for run in selected:
        by_condition.setdefault(run.get("condition"), {})[pair_key(run)] = run

    attacked = by_condition.get("attacked_final", {})
    output: Dict[str, Dict] = {}
    for metric_index, metric in enumerate(
        ("generation_wall_seconds", "cuda_elapsed_seconds")
    ):
        output[metric] = {}
        for baseline_condition, label in (
            ("pristine", "attacked_vs_pristine"),
            ("benign_final", "attacked_vs_benign"),
        ):
            baseline = by_condition.get(baseline_condition, {})
            keys = sorted(set(attacked).intersection(baseline))
            atk_values, baseline_values = [], []
            for key in keys:
                atk_value = attacked[key].get(metric)
                ref_value = baseline[key].get(metric)
                if (
                    atk_value is None or ref_value is None
                    or float(atk_value) <= 0.0 or float(ref_value) <= 0.0
                ):
                    continue
                atk_values.append(float(atk_value))
                baseline_values.append(float(ref_value))
            if not atk_values:
                continue
            summary = paired_log_ratio_summary(
                atk_values,
                baseline_values,
                seed=7300 + metric_index,
                bootstrap_samples=2000,
            )
            # At resource_profile_repeats=3 the bootstrap 95% interval degenerates to the
            # (min, max) of the three observed ratios: the bootstrap median can only take one of
            # 3 values and P(median = smallest) = 7/27, so the 2.5% quantile lands on the minimum
            # essentially always. ci95_lower is then the WORST repeat, not a 95% bound, and the
            # true coverage of (min, max) is 1 - 2*(1/2)^n = 75%. The hardware gate reads
            # ci95_lower, so this direction is conservative (false FAIL, never a false PASS) — but
            # a 3-point range must not be cited as a 95% CI in the paper.
            n_pairs_measured = int(summary["n_pairs"])
            summary.update({
                "pairing_unit": "repeat_same_decode_seed_and_prompt_set",
                "split": split,
                "batch_size": int(batch_size),
                "attacked_values": atk_values,
                "baseline_values": baseline_values,
                "interval_is_order_statistic_range": n_pairs_measured <= 4,
                "interval_true_coverage": (
                    round(1.0 - 2.0 * 0.5 ** n_pairs_measured, 4)
                    if n_pairs_measured <= 4 else None
                ),
            })
            output[metric][label] = summary
    return output


def _prompt_sha256(example) -> str:
    payload = json.dumps(list(example.prompt_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_set_sha256(examples) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(_prompt_sha256(example).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _logical_rows(stats, examples, *, condition: str, split: str, round_idx,
                  measurement_stage: Optional[str] = None) -> List[Dict]:
    """Convert a CostStats object into durable per-request resource records."""

    rows: List[Dict] = []
    records = stats.per_prompt_records()
    examples_by_uid = {
        str(example.uid): example for example in examples
        if getattr(example, "uid", None) is not None
    }
    for i, record in enumerate(records):
        example = examples_by_uid.get(str(record.get("task_uid")))
        if example is None and examples:
            # Reasoning evaluation repeats the same prompt order once per decode seed.
            # Modulo keeps legacy one-pass rows unchanged and hashes repeated rows correctly.
            example = examples[i % len(examples)]
        rows.append({
            "condition": condition,
            "split": split,
            "round": round_idx,
            "measurement_stage": measurement_stage,
            "prompt_id": i,
            "prompt_sha256": _prompt_sha256(example) if example is not None else None,
            **record,
            "output_token_includes_eos": True,
        })
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resource_summary_rows(resources: Dict, primary_batch_size: int) -> List[Dict]:
    """Flatten logical + profiled summaries without allowing field-name collisions."""
    primary_key = f"batch_size_{int(primary_batch_size)}"
    rows: List[Dict] = []
    for condition, state in resources.get("states", {}).items():
        for split, logical_key in (("tau", "logical"), ("clean", "clean_logical")):
            split_hardware = (
                state.get("hardware_profiles", {}).get(split, {}).get(primary_key, {})
            )
            hardware_columns = {
                f"hardware_{key}": value for key, value in split_hardware.items()
            }
            rows.append({
                "condition": condition,
                "split": split,
                **state[logical_key],
                **hardware_columns,
            })
    return rows


def _logical_resource_bundle(pri_tau, pri_cln, atk_tau, atk_cln,
                             ben_tau=None, ben_cln=None) -> Dict:
    """Build the canonical resource-v1 logical state and paired comparisons."""

    states = {
        "pristine": {"logical": pri_tau.summary(), "clean_logical": pri_cln.summary()},
        "attacked_final": {"logical": atk_tau.summary(), "clean_logical": atk_cln.summary()},
    }
    if ben_tau is not None and ben_cln is not None:
        states["benign_final"] = {
            "logical": ben_tau.summary(), "clean_logical": ben_cln.summary()
        }

    comparisons = {}
    metrics = (
        "total_output_tokens", "total_tokens", "prefill_attention_pairs",
        "decode_attention_pairs", "scheduled_decode_slots", "mean_kv_proxy",
        "kv_cache_peak_bytes",
    )
    attacked = states["attacked_final"]["logical"]
    for metric in metrics:
        entry = {
            "attacked_vs_pristine": _safe_ratio(
                attacked.get(metric), states["pristine"]["logical"].get(metric)
            )
        }
        if "benign_final" in states:
            entry["attacked_vs_benign"] = _safe_ratio(
                attacked.get(metric), states["benign_final"]["logical"].get(metric)
            )
        comparisons[metric] = entry
    return {"states": states, "comparisons": comparisons}


def _build_objective_summary(results: Dict) -> Dict:
    """One non-composite view of consumption, quality, stealth, and defense evasion."""
    durability = results.get("durability") or []
    final = durability[-1] if durability else {}
    stealth_rounds = [
        row for row in (results.get("stealth_trace") or []) if row.get("n_attackers")
    ]
    stealth_passes = sum(bool(row.get("jointly_satisfied")) for row in stealth_rounds)
    resources = results.get("resources") or {}
    comparisons = resources.get("comparisons") or {}
    attacked_logical = (
        (resources.get("states") or {}).get("attacked_final", {}).get("logical", {})
    )
    defense_rows = (
        (results.get("defense_evaluation") or {})
        .get("telemetry_defenses", {})
        .get("defenses", {})
    )
    reasoning_mode = (results.get("config") or {}).get("attack_objective") == "reasoning_cost"
    summary = {
        "schema_version": "objective-v2" if reasoning_mode else "objective-v1",
        "resource_amplification": {
            metric: {
                "attacked_vs_pristine": values.get("attacked_vs_pristine"),
                "attacked_vs_benign": values.get("attacked_vs_benign"),
            }
            for metric, values in comparisons.items()
            if isinstance(values, dict)
        },
        "attacked_consumption": {
            "total_input_tokens": attacked_logical.get("total_input_tokens"),
            "total_output_tokens": attacked_logical.get("total_output_tokens"),
            "budget_utilization": attacked_logical.get("budget_utilization"),
            "cap_hit_rate": attacked_logical.get("cap_hit_rate"),
            "time_limit_rate": attacked_logical.get("time_limit_rate"),
            "logical_validity": attacked_logical.get("validity"),
            "hardware_validity": (resources.get("validity") or {}).get("hardware"),
        },
        "attack_quality_final": {
            "effective_cost_amplification": final.get("amp_tau_effective"),
            "calibrated_cost_amplification": final.get("amp_tau_calibrated"),
            "raw_cost_amplification": final.get("amp_tau"),
            "repetition_rate": final.get("repetition_tau"),
            "distinct_ratio": final.get("distinct_ratio_tau"),
        },
        "utility_final": {
            # BOTH controls, explicitly named. vs-pristine compares against the UNTRAINED round-0
            # model, so it charges ordinary benign federated drift to the attacker; the benign-only
            # arm is the attacker's actual counterfactual. A consumer that reads only the pristine
            # row will systematically over-state clean-utility damage.
            "ppl_ratio_vs_pristine": final.get("ppl_ratio_vs_pristine"),
            "ppl_ratio_vs_benign_fl": final.get("ppl_ratio"),
            "rouge_recall_clean": final.get("rouge_recall_clean_atk"),
            "rouge_recall_clean_benign_fl": final.get("rouge_recall_clean_ben"),
            "rouge_recall_clean_pristine": final.get("rouge_recall_clean_pristine"),
            "rouge_recall_tau": final.get("rouge_recall_tau_atk"),
            # length-robust "answer kept, not padded" signal (recall inflates with length)
            "rouge_f1_clean": final.get("rouge_f1_clean_atk"),
            "rouge_f1_tau": final.get("rouge_f1_tau_atk"),
            # LINEAR length ratio vs the benign arm. The C2 "was clean shortened?" question may be
            # answered ONLY from this; amp_clean is a QUADRATIC cost ratio and is carried under its
            # own name so the two can never be confused.
            "clean_len_ratio_vs_benign_fl": final.get("amp_clean_len"),
            "clean_cost_ratio_vs_benign_fl_quadratic": final.get("amp_clean"),
        },
        "parameter_stealth": {
            "joint_passes": stealth_passes,
            "attacker_present_rounds": len(stealth_rounds),
            "joint_pass_rate": _safe_ratio(stealth_passes, len(stealth_rounds)),
        },
        "defense_evasion": {
            name: {
                "caught_rate": values.get("caught_rate"),
                "survival_rate": values.get("survival_rate"),
                # excess_detection = atk_flag_rate - ben_flag_rate is the honest, base-rate-
                # corrected number: ~0 => attacker indistinguishable from a benign client.
                "atk_flag_rate": values.get("atk_flag_rate"),
                "ben_flag_rate": values.get("ben_flag_rate"),
                "excess_detection": values.get("excess_detection"),
                # A bare excess reads as detection even when it is pure noise, so the significance
                # travels WITH it into the machine-readable summary. `significant` is decided by
                # the client-level permutation when available (attacker identity is a per-client
                # property held fixed across rounds); `p_within_round` is the weaker per-round null.
                "excess_significant": values.get("excess_significant"),
                "excess_p_cluster": values.get("excess_p_cluster"),
                "excess_p_within_round": values.get("excess_p_within_round"),
                "excess_significance_method": values.get("excess_significance_method"),
                # A small federation floors the permutation p at 1/n_candidates (0.20 at 5
                # clients / 1 attacker), so excess_significant is False for every outcome. These
                # two keys are what separates "evaded" from "untestable"; read them first.
                "excess_significance_underpowered": values.get("excess_significance_underpowered"),
                "excess_min_attainable_p": values.get("excess_min_attainable_p"),
                # Krum keeps 1 update, so its excess is range-compressed and NOT comparable to the
                # fixed-f rows; purpose_built marks detectors designed against this attack, which
                # do not support a "standard defenses miss it" claim.
                "excess_structural": values.get("excess_structural"),
                "purpose_built": values.get("purpose_built"),
                "rounds": values.get("rounds"),
            }
            for name, values in defense_rows.items()
            if isinstance(values, dict)
        },
        "note": (
            "No composite score is formed: consumption gain, output quality, parameter "
            "stealth, and named-defense evasion must all be inspected together. Read "
            "defense_evasion by excess_detection (atk_flag_rate - ben_flag_rate) TOGETHER WITH "
            "excess_significant, never caught_rate alone: caught_rate is confounded by each "
            "defense's base flag rate (cosine-screen flags f/n of random clients; Krum rejects "
            "all but one), and a positive excess whose permutation p is large is noise. Skip rows "
            "with excess_structural=true when comparing magnitudes, and do not cite "
            "purpose_built=true rows as evidence that standard defenses detect the attack. "
            "utility_final carries BOTH the pristine and benign-FL controls: the benign-FL arm is "
            "the attacker's counterfactual, and clean shortening may be read only from "
            "clean_len_ratio_vs_benign_fl (linear), never from the quadratic cost ratio."
        ),
    }
    if reasoning_mode:
        mechanism_claim_ready = bool(final.get("reasoning_claim_ready"))
        resource_environment = resources.get("environment") or {}
        start_environment = resource_environment.get("start") or {}
        end_environment = resource_environment.get("end") or {}
        environment_evidence_ready = bool(
            start_environment.get("fingerprint_sha256")
            and end_environment.get("fingerprint_sha256")
            and resource_environment.get("environment_changed") is False
            and (
                ((start_environment.get("torch") or {}).get("kernel_preflight") or {})
                .get("success") is True
            )
        )
        hardware_evidence_ready = (
            (resources.get("validity") or {}).get("hardware") == "valid"
            and environment_evidence_ready
        )
        comparisons = resources.get("comparisons") or {}
        baseline_key = (
            "attacked_vs_benign"
            if "benign_final" in (resources.get("states") or {})
            else "attacked_vs_pristine"
        )
        wall_ratio = (comparisons.get("generation_wall_seconds") or {}).get(baseline_key)
        cuda_ratio = (comparisons.get("cuda_elapsed_seconds") or {}).get(baseline_key)
        paired_hardware = resources.get("hardware_paired_ratios") or {}
        wall_paired = (paired_hardware.get("generation_wall_seconds") or {}).get(
            baseline_key
        )
        cuda_paired = (paired_hardware.get("cuda_elapsed_seconds") or {}).get(
            baseline_key
        )
        min_hardware_ratio = float(
            (results.get("config") or {}).get("reasoning_min_hardware_ratio") or 1.0
        )
        hardware_amplification_observed = bool(
            wall_paired and cuda_paired
            and float(wall_paired.get("ci95_lower", 0.0)) >= min_hardware_ratio
            and float(cuda_paired.get("ci95_lower", 0.0)) >= min_hardware_ratio
        )
        hardware_resource_effect_ready = bool(
            mechanism_claim_ready
            and hardware_evidence_ready
            and hardware_amplification_observed
        )
        expected_rounds = int((results.get("config") or {}).get("num_rounds") or 0)
        parameter_stealth_ready = bool(
            expected_rounds > 0
            and len(stealth_rounds) == expected_rounds
            and stealth_passes == expected_rounds
        )
        summary["reasoning_attack_final"] = {
            "paired_cost_ratio": final.get("paired_reasoning_cost_tau"),
            "clean_paired_cost_ratio": final.get("paired_reasoning_cost_clean"),
            "reasoning_tokens_attacked": final.get("reasoning_tokens_tau_atk"),
            "reasoning_tokens_reference": final.get("reasoning_tokens_tau_ref"),
            "answer_accuracy_attacked": final.get("answer_accuracy_tau_atk"),
            "answer_accuracy_reference": final.get("answer_accuracy_tau_ref"),
            "clean_answer_accuracy_attacked": final.get("answer_accuracy_clean_atk"),
            "clean_answer_accuracy_reference": final.get("answer_accuracy_clean_ref"),
            "closure_rate": final.get("reasoning_closure_tau"),
            "reasoning_repetition": final.get("reasoning_repetition_tau"),
            "reasoning_distinct_ratio": final.get("reasoning_distinct_tau"),
            "measurement_valid": final.get("reasoning_measurement_valid"),
            "gates": final.get("reasoning_gates"),
            # This gate concerns the resource effect plus answer/closure quality.
            # Parameter stealth and defense-evasion evidence remain separate below.
            "resource_effect_claim_ready": mechanism_claim_ready,
            "claim_ready": mechanism_claim_ready,  # backward-compatible alias
            "hardware_evidence_ready": hardware_evidence_ready,
            "environment_evidence_ready": environment_evidence_ready,
            "hardware_amplification": {
                "baseline": baseline_key.replace("attacked_vs_", ""),
                "generation_wall_seconds_ratio": wall_ratio,
                "cuda_elapsed_seconds_ratio": cuda_ratio,
                "generation_wall_seconds_paired": wall_paired,
                "cuda_elapsed_seconds_paired": cuda_paired,
                "minimum_ci_lower_ratio": min_hardware_ratio,
                "observed": hardware_amplification_observed,
                "scope": "paired_repeat_level_single_gpu_batch_size_specific",
            },
            "hardware_resource_effect_claim_ready": hardware_resource_effect_ready,
            "full_resource_claim_ready": hardware_resource_effect_ready,
            "parameter_stealth_ready": parameter_stealth_ready,
            "stealth_constrained_hardware_claim_ready": (
                hardware_resource_effect_ready and parameter_stealth_ready
            ),
            "defense_evasion_is_separate_evidence": True,
        }
    return summary


def _median_iqr(values) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None, None, None
    return (
        float(np.median(clean)),
        float(np.percentile(clean, 25)),
        float(np.percentile(clean, 75)),
    )


def _normalise_hardware_profile(profile, *, condition: str, split: str,
                                batch_size: int, repeat: int,
                                e2e_wall_seconds: Optional[float] = None
                                ) -> Tuple[Dict, List[Dict]]:
    raw = profile.to_dict()
    summary = raw.get("summary", {})
    records = raw.get("records", [])
    timed_out = int(summary.get("timed_out_batches") or 0) > 0
    incomplete = (
        raw.get("completed_batches") != raw.get("planned_batches")
        or bool(raw.get("stopped_early"))
        or any(not record.get("completed", True) for record in records)
    )
    wall_coverage = summary.get("wall_timing_coverage")
    if wall_coverage is None:
        wall_coverage = 1.0 if summary.get("wall_seconds") is not None else 0.0
    token_coverage = summary.get("token_count_coverage")
    if token_coverage is None:
        token_coverage = (
            1.0 if summary.get("input_tokens") is not None
            and summary.get("output_tokens") is not None else 0.0
        )
    memory_coverage = summary.get("memory_coverage")
    if memory_coverage is None:
        memory_coverage = (
            1.0 if summary.get("peak_allocated_delta_bytes") is not None else 0.0
        )
    wall_timing_valid = bool(
        records
        and summary.get("wall_seconds") is not None
        and float(wall_coverage) == 1.0
    )
    token_counts_valid = bool(
        records
        and summary.get("input_tokens") is not None
        and summary.get("output_tokens") is not None
        and float(token_coverage) == 1.0
    )
    cuda_timing_valid = bool(
        records and float(summary.get("cuda_coverage") or 0.0) == 1.0
    )
    memory_metrics_valid = bool(
        records and float(memory_coverage) == 1.0
    )
    energy_metrics_valid = bool(
        records
        and summary.get("energy_joules") is not None
        and float(summary.get("energy_coverage") or 0.0) == 1.0
    )
    completion_valid = not timed_out and not incomplete
    gpu_allocated = bool(
        any(record.get("cuda_used") is True for record in records)
        or float(summary.get("cuda_coverage") or 0.0) > 0.0
    )
    valid = completion_valid and wall_timing_valid and token_counts_valid
    invalid_reason = None
    if timed_out:
        invalid_reason = "generation_max_batch_seconds_exceeded"
    elif incomplete:
        invalid_reason = raw.get("stop_reason") or "incomplete_profile"
    elif not wall_timing_valid:
        invalid_reason = "wall_timing_incomplete_or_unsynchronized"
    elif not token_counts_valid:
        invalid_reason = "token_count_incomplete"

    def max_value(name, *, require_memory: bool = False):
        values = [
            record.get(name) for record in records
            if record.get(name) is not None
            and (not require_memory
                 or record.get("memory_metrics_valid", True) is not False)
        ]
        return max(values) if values else None

    def weighted_mean(name):
        pairs = [
            (record.get(name), record.get("wall_seconds"))
            for record in records
            if record.get(name) is not None and record.get("wall_seconds") is not None
        ]
        total_wall = sum(float(wall) for _, wall in pairs)
        return (sum(float(value) * float(wall) for value, wall in pairs) / total_wall
                if total_wall > 0 else None)

    energy = summary.get("energy_joules")
    requests = summary.get("requests")
    output_tokens = summary.get("output_tokens")
    methods = sorted({
        str(record["energy_method"]) for record in records if record.get("energy_method")
    })
    run = {
        "condition": condition,
        "split": split,
        "batch_size": int(batch_size),
        "repeat": int(repeat),
        "valid": bool(valid),
        "completion_valid": bool(completion_valid),
        "wall_timing_valid": bool(wall_timing_valid),
        "token_counts_valid": bool(token_counts_valid),
        "cuda_timing_valid": bool(cuda_timing_valid),
        "memory_metrics_valid": bool(memory_metrics_valid),
        "energy_metrics_valid": bool(energy_metrics_valid),
        "invalid_reason": invalid_reason,
        "n_requests": requests,
        "input_tokens": summary.get("input_tokens"),
        "output_tokens": output_tokens,
        "total_tokens": summary.get("total_tokens"),
        "generation_wall_seconds": (
            summary.get("wall_seconds") if wall_timing_valid else None
        ),
        "observed_wall_seconds": summary.get("observed_wall_seconds"),
        "cuda_elapsed_seconds": (
            summary.get("cuda_seconds") if cuda_timing_valid else None
        ),
        # Whole wrapper call: batch H2D preparation + measured generation + Python
        # orchestration. Parameter loading and artifact writes remain outside timing.
        "e2e_wall_seconds": (
            float(e2e_wall_seconds)
            if e2e_wall_seconds is not None and wall_timing_valid else None
        ),
        # One allocated accelerator is used.  This is occupancy/timeline duration, not
        # a sum of active kernel time and is named accordingly in the report.
        "allocated_gpu_seconds": (
            summary.get("wall_seconds") if wall_timing_valid and gpu_allocated else None
        ),
        "requests_per_second": (
            summary.get("requests_per_second") if valid else None
        ),
        "output_tokens_per_second": (
            summary.get("output_tokens_per_second") if valid else None
        ),
        "peak_allocated_bytes": (
            max_value("peak_allocated_bytes", require_memory=True)
            if memory_metrics_valid else None
        ),
        "incremental_peak_allocated_bytes": (
            summary.get("peak_allocated_delta_bytes") if memory_metrics_valid else None
        ),
        "peak_reserved_bytes": (
            max_value("peak_reserved_bytes", require_memory=True)
            if memory_metrics_valid else None
        ),
        "incremental_peak_reserved_bytes": (
            summary.get("peak_reserved_delta_bytes") if memory_metrics_valid else None
        ),
        "energy_joules": energy if energy_metrics_valid else None,
        "joules_per_request": (
            _safe_ratio(energy, requests) if energy_metrics_valid else None
        ),
        "joules_per_output_token": (
            _safe_ratio(energy, output_tokens) if energy_metrics_valid else None
        ),
        "energy_method": "+".join(methods) if methods else None,
        "energy_scope": next((record.get("energy_scope") for record in records
                              if record.get("energy_scope")), None),
        "energy_attribution": next((record.get("energy_attribution") for record in records
                                    if record.get("energy_attribution")), None),
        "energy_interval_seconds": (
            sum(float(record["energy_interval_seconds"]) for record in records)
            if records and all(record.get("energy_interval_seconds") is not None
                               for record in records) else None
        ),
        "nvml_device_mapping_methods": sorted({
            record.get("nvml_device_mapping_method") for record in records
            if record.get("nvml_device_mapping_method")
        }),
        "mean_power_watts": (
            weighted_mean("mean_power_watts") if energy_metrics_valid else None
        ),
        "peak_power_watts": (
            max_value("peak_power_watts") if energy_metrics_valid else None
        ),
        "cuda_coverage": summary.get("cuda_coverage"),
        "energy_coverage": summary.get("energy_coverage"),
        "wall_timing_coverage": wall_coverage,
        "memory_coverage": memory_coverage,
        "token_count_coverage": token_coverage,
        "instrumentation_reasons": summary.get("instrumentation_reasons", []),
        "timed_out_batches": summary.get("timed_out_batches"),
        "measurement_scope": "pretokenized_generation_only",
        "cuda_time_semantics": "event_timeline_between_generate_boundaries",
    }
    batch_rows = []
    for record in records:
        batch_rows.append({
            "condition": condition,
            "split": split,
            "batch_size": int(batch_size),
            "repeat": int(repeat),
            **record,
        })
    return run, batch_rows


def _aggregate_hardware_runs(runs: List[Dict]) -> Dict:
    valid = [run for run in runs if run.get("valid")]
    out: Dict = {
        "n_repeats": len(runs),
        "valid_repeats": len(valid),
        "invalid_repeats": len(runs) - len(valid),
        # Partial repeats can still support a median, but must not be called fully valid.
        "valid": bool(valid) and len(valid) == len(runs),
        "usable": bool(valid),
        "invalid_reasons": sorted({
            run.get("invalid_reason") for run in runs if run.get("invalid_reason")
        }),
        "measurement_scope": "pretokenized_generation_only",
        "aggregation": "median_across_repeats",
    }
    fields = (
        "n_requests", "input_tokens", "output_tokens", "total_tokens",
        "generation_wall_seconds", "cuda_elapsed_seconds", "e2e_wall_seconds",
        "allocated_gpu_seconds",
        "requests_per_second", "output_tokens_per_second", "peak_allocated_bytes",
        "incremental_peak_allocated_bytes", "peak_reserved_bytes",
        "incremental_peak_reserved_bytes", "energy_joules", "joules_per_request",
        "joules_per_output_token", "mean_power_watts", "peak_power_watts",
        "cuda_coverage", "energy_coverage", "wall_timing_coverage",
        "memory_coverage", "token_count_coverage", "energy_interval_seconds",
    )
    cuda_fields = {"cuda_elapsed_seconds"}
    memory_fields = {
        "peak_allocated_bytes", "incremental_peak_allocated_bytes",
        "peak_reserved_bytes", "incremental_peak_reserved_bytes",
    }
    energy_fields = {
        "energy_joules", "joules_per_request", "joules_per_output_token",
        "mean_power_watts", "peak_power_watts", "energy_interval_seconds",
    }
    token_fields = {"n_requests", "input_tokens", "output_tokens", "total_tokens"}
    throughput_fields = {"requests_per_second", "output_tokens_per_second"}

    def field_is_valid(run: Dict, field: str) -> bool:
        if field in cuda_fields:
            return bool(run.get("cuda_timing_valid"))
        if field in memory_fields:
            return bool(run.get("memory_metrics_valid"))
        if field in energy_fields:
            return bool(run.get("energy_metrics_valid"))
        if field in token_fields:
            return bool(run.get("token_counts_valid", True))
        if field in throughput_fields:
            return bool(run.get("wall_timing_valid", True)
                        and run.get("token_counts_valid", True))
        return True

    field_valid_repeats = {}
    for field in fields:
        eligible = [run for run in valid if field_is_valid(run, field)]
        # A field is a formal aggregate only when every otherwise-valid repeat has
        # that instrumentation. Partial values remain available in resource_repeats.csv.
        complete = bool(valid) and len(eligible) == len(valid)
        values = [run.get(field) for run in eligible] if complete else []
        median, q25, q75 = _median_iqr(values)
        out[field] = median
        field_valid_repeats[field] = len(eligible)
        if field in {
            "generation_wall_seconds", "cuda_elapsed_seconds", "e2e_wall_seconds",
            "energy_joules",
        }:
            out[f"{field}_p25"] = q25
            out[f"{field}_p75"] = q75
    out["field_valid_repeats"] = field_valid_repeats
    methods = sorted({run.get("energy_method") for run in valid
                      if run.get("energy_metrics_valid") and run.get("energy_method")})
    scopes = sorted({run.get("energy_scope") for run in valid if run.get("energy_scope")})
    attributions = sorted({run.get("energy_attribution") for run in valid
                           if run.get("energy_attribution")})
    mapping_methods = sorted({
        method for run in valid for method in (run.get("nvml_device_mapping_methods") or [])
        if method
    })
    instrumentation_reasons = sorted({
        reason for run in runs for reason in (run.get("instrumentation_reasons") or [])
        if reason
    })
    mixed_energy_methods = len(methods) > 1 or any("+" in str(method) for method in methods)
    out["energy_method"] = (
        "mixed_not_aggregated:" + "+".join(methods)
        if mixed_energy_methods else methods[0] if methods else None
    )
    out["energy_scope"] = "+".join(scopes) if scopes else None
    out["energy_attribution"] = "+".join(attributions) if attributions else None
    out["nvml_device_mapping_methods"] = mapping_methods
    out["instrumentation_reasons"] = instrumentation_reasons
    out["energy_methods_comparable"] = not mixed_energy_methods if methods else None
    if mixed_energy_methods:
        # Counter deltas and sampled-power integration have different uncertainty.
        # Keep the raw repeats, but do not manufacture one cross-method headline.
        for field in (
            "energy_joules", "energy_joules_p25", "energy_joules_p75",
            "joules_per_request", "joules_per_output_token",
        ):
            out[field] = None
    out["timed_out_batches"] = int(
        sum(run.get("timed_out_batches") or 0 for run in runs)
    )
    out["stopped_early"] = bool(out["invalid_repeats"])
    return out


def _append_jsonl(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _profile_final_resources(model, state_vectors: Dict[str, torch.Tensor],
                             tau_ev, clean_ev, cfg, spec, device, out_dir: Path) -> Dict:
    """Run a bounded, paired final-state resource benchmark in the current runtime."""

    from .resource_metrics import (
        HardwareBatchError, ResourceProfileConfig, profile_model_generation,
    )

    requested_eval_size = int(cfg["resource_profile_eval_size"])
    generation_eos_ids = (
        spec.resolved_generation_eos_ids()
        if hasattr(spec, "resolved_generation_eos_ids")
        else [int(spec.eos_id)]
    )
    split_examples = {
        "tau": tau_ev[:min(requested_eval_size, len(tau_ev))],
        "clean": clean_ev[:min(requested_eval_size, len(clean_ev))],
    }
    batch_sizes = list(dict.fromkeys(int(v) for v in cfg["resource_profile_batch_sizes"]))
    repeats = int(cfg["resource_profile_repeats"])
    warmup_batches = int(cfg["resource_profile_warmup_batches"])
    all_runs: List[Dict] = []
    all_batch_rows: List[Dict] = []
    profiles: Dict[str, Dict[str, Dict]] = {name: {} for name in state_vectors}
    conditions = list(state_vectors.keys())
    repeats_jsonl = out_dir / "resource_repeats.jsonl"
    batches_jsonl = out_dir / "hardware_batches.jsonl"
    repeats_jsonl.unlink(missing_ok=True)
    batches_jsonl.unlink(missing_ok=True)
    reasoning_mode = cfg.get("attack_objective", "length") == "reasoning_cost"
    profile_generation_kwargs = {
        "do_sample": bool(reasoning_mode),
        "num_beams": 1,
        "use_cache": True,
    }
    if reasoning_mode:
        profile_generation_kwargs.update(_reasoning_generation_kwargs(cfg))
    decode_seeds = list(cfg.get("eval_decode_seeds") or [cfg.get("seed", 0)])
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    for split in cfg.get("resource_profile_splits", ["tau"]):
        examples = split_examples[split]
        if not examples:
            raise ValueError(f"resource profile split {split!r} has no evaluation examples")
        prompt_subset_sha256 = _prompt_set_sha256(examples)
        for batch_size in batch_sizes:
            batches = _gen_batches(examples, spec.pad_id, batch_size)
            profile_cfg = ResourceProfileConfig(
                max_new_tokens=int(cfg["max_new_tokens"]),
                max_batch_seconds=float(cfg["generation_max_batch_seconds"]),
                nvml_enabled=bool(cfg.get("resource_profile_nvml", True)),
                nvml_sample_interval_ms=int(cfg.get("resource_profile_sample_interval_ms", 100)),
                device_index=(device.index if getattr(device, "index", None) is not None else 0),
            )
            warm_cfg = ResourceProfileConfig(
                max_new_tokens=profile_cfg.max_new_tokens,
                max_batch_seconds=profile_cfg.max_batch_seconds,
                nvml_enabled=False,
                nvml_sample_interval_ms=profile_cfg.nvml_sample_interval_ms,
                device_index=profile_cfg.device_index,
            )

            # One controlled warmup per condition/batch-size.  Compilation/lazy CUDA work
            # is thereby excluded from every formal repeat.
            if warmup_batches:
                for condition in conditions:
                    model.set_flat_params(state_vectors[condition].to(device))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(int(decode_seeds[0]))
                        warm_outputs, warm_profile = profile_model_generation(
                            model, batches[:warmup_batches], config=warm_cfg, device=device,
                            eos_token_id=generation_eos_ids,
                            pad_token_id=spec.pad_id,
                            generation_kwargs=profile_generation_kwargs,
                            retain_outputs=False,
                        )
                    del warm_outputs
                    if warm_profile.summary().get("timed_out_batches"):
                        raise RuntimeError(
                            f"resource warmup timed out for {condition}/{split}/bs={batch_size}; "
                            "reduce batch size or raise the finite wall guard explicitly"
                        )

            # Rotate condition order per repeat to reduce thermal/order drift while keeping
            # the exact prompt set and composition paired.
            for repeat in range(repeats):
                decode_seed = int(decode_seeds[repeat])
                order = conditions[repeat % len(conditions):] + conditions[:repeat % len(conditions)]
                for condition in order:
                    model.set_flat_params(state_vectors[condition].to(device))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    e2e_started = time.perf_counter()
                    try:
                        # Reset the stochastic stream for each condition in a repeat.
                        # Hardware differences are then paired by prompt and decode draw.
                        with torch.random.fork_rng(devices=cuda_devices):
                            torch.manual_seed(decode_seed)
                            outputs, profile = profile_model_generation(
                                model, batches, config=profile_cfg, device=device,
                                eos_token_id=generation_eos_ids,
                                pad_token_id=spec.pad_id,
                                generation_kwargs=profile_generation_kwargs,
                                retain_outputs=False,
                            )
                    except HardwareBatchError as exc:
                        e2e_wall_seconds = time.perf_counter() - e2e_started
                        if exc.partial_profile is not None:
                            failed_run, failed_batches = _normalise_hardware_profile(
                                exc.partial_profile, condition=condition, split=split,
                                batch_size=batch_size, repeat=repeat,
                                e2e_wall_seconds=e2e_wall_seconds,
                            )
                            failed_run["prompt_subset_sha256"] = prompt_subset_sha256
                            failed_run["decode_seed"] = decode_seed
                            for row in failed_batches:
                                row["prompt_subset_sha256"] = prompt_subset_sha256
                                row["decode_seed"] = decode_seed
                            _append_jsonl(repeats_jsonl, [failed_run])
                            _append_jsonl(batches_jsonl, failed_batches)
                        raise
                    e2e_wall_seconds = time.perf_counter() - e2e_started
                    run, batch_rows = _normalise_hardware_profile(
                        profile, condition=condition, split=split,
                        batch_size=batch_size, repeat=repeat,
                        e2e_wall_seconds=e2e_wall_seconds,
                    )
                    run["prompt_subset_sha256"] = prompt_subset_sha256
                    run["decode_seed"] = decode_seed
                    for row in batch_rows:
                        row["prompt_subset_sha256"] = prompt_subset_sha256
                        row["decode_seed"] = decode_seed
                    del outputs
                    all_runs.append(run)
                    all_batch_rows.extend(batch_rows)
                    # Drive writes happen only after the timed block.  Incremental JSONL
                    # protects expensive profiles from a later Colab disconnect.
                    _append_jsonl(repeats_jsonl, [run])
                    _append_jsonl(batches_jsonl, batch_rows)

            for condition in conditions:
                matching = [
                    run for run in all_runs
                    if run["condition"] == condition and run["split"] == split
                    and run["batch_size"] == batch_size
                ]
                aggregate = _aggregate_hardware_runs(matching)
                aggregate["prompt_subset_sha256"] = prompt_subset_sha256
                aggregate["split"] = split
                aggregate["batch_size"] = batch_size
                profiles[condition].setdefault(split, {})[
                    f"batch_size_{batch_size}"
                ] = aggregate

    return {
        "profiles": profiles,
        "runs": all_runs,
        "batch_records": all_batch_rows,
        "artifacts": {
            "resource_repeats_jsonl": str(repeats_jsonl),
            "hardware_batches_jsonl": str(batches_jsonl),
        },
    }


def _collect_defense_telemetry(round_idx, ben_updates, ben_sizes, ben_ids,
                               atk_updates, atk_sizes, atk_ids, *, num_attackers=0):
    """Per-round, per-client update geometry a robust aggregator would screen on — LOGGED,
    not enforced, so a defense implemented later can be replayed OFFLINE to test whether it
    detects the attacker(s). Records the quantities Krum / norm-clipping / cosine screens
    consume: L2 norm, distance & cosine to the (attacker-inclusive) FedAvg aggregate and to the
    benign mean, leave-self-out pairwise-mean cosine, and a Krum score. The full pairwise
    distance & cosine matrices are included so ANY Krum/Multi-Krum variant (its own f, k) can be
    replayed without the raw vectors. Aggregation itself is unchanged (plain FedAvg)."""
    updates = [u.detach().float() for u in ben_updates] + [u.detach().float() for u in atk_updates]
    labels = ["benign"] * len(ben_updates) + ["attacker"] * len(atk_updates)
    ids = list(ben_ids) + list(atk_ids)
    weights = list(ben_sizes) + list(atk_sizes)
    n = len(updates)
    if n == 0:
        return {"round": int(round_idx), "n_clients": 0, "clients": []}
    agg = weighted_fedavg_reference(updates, weights)          # exactly the FedAvg aggregate
    ben_mean = (weighted_fedavg_reference([u.detach().float() for u in ben_updates], ben_sizes)
                if ben_updates else agg)
    # Pairwise distance / cosine matrices (n x n; small — n <= clients_per_round).
    dist_mat = [[0.0] * n for _ in range(n)]
    cos_mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = update_distance(updates[i], updates[j])
            c = update_cosine(updates[i], updates[j])
            dist_mat[i][j] = dist_mat[j][i] = round(d, 6)
            cos_mat[i][j] = cos_mat[j][i] = round(c, 6)
    # Krum score = sum of the (n-f-2) smallest squared distances to other clients (Blanchard 2017).
    f = int(num_attackers)
    k = max(1, n - f - 2)
    clients = []
    for i in range(n):
        sq = sorted((dist_mat[i][j] ** 2 for j in range(n) if j != i))
        others = [updates[j] for j in range(n) if j != i]
        clients.append({
            "client_id": int(ids[i]) if i < len(ids) else i,
            "label": labels[i],
            "norm": round(float(torch.norm(updates[i])), 6),
            "dist_to_agg": round(update_distance(updates[i], agg), 6),
            "dist_to_benign_mean": round(update_distance(updates[i], ben_mean), 6),
            "cos_to_agg": round(update_cosine(updates[i], agg), 6),
            "pairwise_mean_cos": round(pairwise_mean_cosine(updates[i], others), 6),
            "krum_score": round(float(sum(sq[:k])), 6),
        })
    return {
        "round": int(round_idx),
        "n_clients": n, "n_attackers_present": len(atk_updates),
        "krum_neighbors_k": k,
        "clients": clients,
        "pairwise_distance": dist_mat,
        "pairwise_cosine": cos_mat,
    }


def _save_update_vectors(out_dir, round_idx, ben_updates, ben_ids, atk_updates, atk_ids):
    """Dump the round's raw per-client update vectors (compressed .npz) for defenses that need
    the full vectors — FLTrust (server root-cosine), coordinate-wise trimmed-mean/median."""
    vecs, labels, ids = [], [], []
    for u, cid in zip(ben_updates, ben_ids):
        vecs.append(u.detach().float().cpu().numpy()); labels.append("benign"); ids.append(int(cid))
    for u, cid in zip(atk_updates, atk_ids):
        vecs.append(u.detach().float().cpu().numpy()); labels.append("attacker"); ids.append(int(cid))
    if not vecs:
        return
    updates_dir = Path(out_dir) / "update_vectors"
    updates_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(updates_dir / f"round_{int(round_idx):03d}.npz",
                        updates=np.stack(vecs), labels=np.array(labels), client_ids=np.array(ids))


def _sample_participants(
    rng: random.Random, benign_ids: List[int], attacker_ids: List[int],
    clients_per_round: int, attacker_always_selected: bool,
) -> Tuple[List[int], List[int]]:
    """Sample the round's participating clients (FedAvg client sampling)."""
    n = min(clients_per_round, len(benign_ids) + len(attacker_ids))
    if attacker_always_selected:
        sel_atk = list(attacker_ids)
        remaining = max(0, n - len(sel_atk))
        sel_ben = rng.sample(benign_ids, min(remaining, len(benign_ids)))
    else:
        pool = list(benign_ids) + list(attacker_ids)
        chosen = set(rng.sample(pool, n))
        sel_ben = [c for c in benign_ids if c in chosen]
        sel_atk = [c for c in attacker_ids if c in chosen]
    return sorted(sel_ben), sorted(sel_atk)


@torch.no_grad()
def _dump_fl_examples(model, g_flat, tau_ev, clean_ev, tokenizer, cfg, spec, device, k=4):
    """Decode a few (prompt -> output) samples from the FINAL attacked global — the
    qualitative 'is the amplified output long AND coherent (not a loop)' evidence for the
    multi-round state. Returns a list of dicts (also carrying per-sample rep/distinct)."""
    if tokenizer is None or k <= 0:
        return []
    from .cost_model import distinct_ratio, repetition_rate
    from .gen_data import collate_gen
    from .metrics import rouge_l_recall
    model.set_flat_params(g_flat.to(device))
    inner = model.inner(); inner.eval()
    recs = []
    for split, ex in (("tau", tau_ev), ("clean", clean_ev)):
        exk = ex[:k]
        if not exk:
            continue
        batch = collate_gen(exk, spec.pad_id)
        ids, attn = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        stopping, wall_guard = build_stopping_criteria(cfg.get("generation_max_batch_seconds"))
        guard_kwargs = {"stopping_criteria": stopping} if stopping is not None else {}
        reasoning_mode = cfg.get("attack_objective", "length") == "reasoning_cost"
        decode_kwargs = _reasoning_generation_kwargs(cfg) if reasoning_mode else {}
        decode_seed = int(cfg.get("eval_decode_seeds", [cfg["seed"]])[0])
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(decode_seed)
            gen = inner.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=bool(reasoning_mode),
                num_beams=1,
                pad_token_id=spec.pad_id,
                eos_token_id=(
                    spec.resolved_generation_eos_ids()[0]
                    if len(spec.resolved_generation_eos_ids()) == 1
                    else spec.resolved_generation_eos_ids()
                ),
                **decode_kwargs,
                **guard_kwargs,
            )
        P = ids.shape[1]
        for i in range(len(exk)):
            new = gen[i, P:]
            is_terminal = torch.zeros_like(new, dtype=torch.bool)
            for terminal_id in spec.resolved_generation_eos_ids():
                is_terminal |= new == terminal_id
            eos_pos = is_terminal.nonzero(as_tuple=True)[0]
            emitted_eos = bool(eos_pos.numel() > 0)
            L = int(eos_pos[0].item()) + 1 if emitted_eos else int(new.shape[0])
            time_limited = bool(wall_guard and wall_guard.triggered and not emitted_eos)
            hit_cap = bool(
                not emitted_eos and not time_limited and L >= cfg["max_new_tokens"]
            )
            out_ids = new[:L].tolist()
            terminal_set = set(spec.resolved_generation_eos_ids())
            ref = [t for t in exk[i].ref_ids if t not in terminal_set]
            recs.append({
                "split": split,
                # keep enough text to SEE the full loop in the qualitative viz (was 180/500);
                # dump_char_cap must scale with max_new_tokens or a high cap clips the loop.
                "prompt": tokenizer.decode(exk[i].prompt_ids, skip_special_tokens=True)[:400],
                "output": tokenizer.decode(out_ids, skip_special_tokens=True)[:int(cfg.get("dump_char_cap", 6000))],
                "len": L, "truncated": bool(not emitted_eos),
                "emitted_eos": emitted_eos, "hit_cap": hit_cap,
                "time_limited": time_limited,
                "termination_reason": (
                    "eos" if emitted_eos else "time_limit" if time_limited
                    else "max_new_tokens" if hit_cap else "unknown"
                ),
                "rouge_recall": round(rouge_l_recall(out_ids, ref), 3) if ref else None,
                "repetition": round(repetition_rate(out_ids), 3),
                "distinct": round(distinct_ratio(out_ids, 4), 3),
            })
    return recs


def run_fl(config: Dict) -> Dict:
    global LAST_RUN_ATTEMPT_ID, LAST_RUN_COMPLETED_ID
    LAST_RUN_COMPLETED_ID = None
    experiment_wall_started = time.perf_counter()
    defaults = default_fl_config()
    _validate_known_config_keys(config or {}, defaults, runner="FL")
    cfg = defaults
    cfg.update(config or {})
    requested_config, requested_config_sha256 = _canonical_config_record(cfg)
    cfg = stamp_run_subdir(cfg)   # unique run folder so reruns never overwrite
    LAST_RUN_ATTEMPT_ID = cfg["results_subdir"].split("/", 1)[0]
    LAST_RUN_COMPLETED_ID = None
    _validate_decoder_only(cfg["backbone"])
    _validate_experiment_config(cfg)
    _validate_fl_config(cfg)
    _set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enable_backend_speedups(cfg)   # opt-in (cfg['use_tf32']); no-op / bit-exact by default
    environment_start = {}
    if cfg.get("collect_resource_metrics", True):
        try:
            from .resource_metrics import collect_runtime_environment
            environment_start = collect_runtime_environment(
                device_index=(device.index if device.index is not None else 0)
            )
        except Exception as exc:  # environment collection must never abort the experiment
            environment_start = {
                "collection_valid": False,
                "reason": f"{type(exc).__name__}:{exc}",
            }
    print(f"\n{'='*64}\nTCAA multi-round FL: {cfg['experiment_name']}  (device={device})\n{'='*64}")

    model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev = build_model_and_data(cfg, device)
    g0 = model.get_flat_params().detach().cpu()
    _require_finite_global(g0, label="pristine")
    print(f"  LoRA update dimension: {g0.numel():,}")

    # Physically-calibrated cost coefficients (c_f = d_model, c_a = 1) for the honest
    # amplification column. d_model lives under different attr names across backbones
    # (Qwen: hidden_size; GPT2: n_embd), so probe both.
    report_calib = bool(cfg.get("report_calibrated_cost", True))
    mcfg = model.inner().config
    d_model = int(getattr(mcfg, "hidden_size", None) or getattr(mcfg, "n_embd", None) or 0)
    calib = calibrate_coefficients(d_model) if (report_calib and d_model > 0) else None
    if calib:
        print(f"  calibrated cost coeffs: d_model={d_model} -> c_f={calib['c_f']:.0f}, c_a={calib['c_a']:.0f} "
              f"(super-linear onset L ~ 2n+{2*calib['c_f']/calib['c_a']:.0f})")

    num_benign = cfg["num_clients"] - cfg["num_attackers"]
    benign_ids = list(range(num_benign))
    attacker_ids = list(range(num_benign, cfg["num_clients"]))
    # Fixed non-IID data partition; each benign client keeps its shard across rounds.
    shards = partition_examples(clean_tr, num_benign, cfg["dirichlet_alpha"], seed=cfg["seed"],
                                min_shard=int(cfg.get("min_client_shard", 0)))
    shard_sizes = [float(max(len(s), 1)) for s in shards]
    # Provenance: the shard split is a pure function of (pool, num_benign, alpha, seed), and an
    # under-floor shard silently changes the effective federation size (see the min_client_shard
    # note in default_fl_config). Record it so a multi-seed roll-up can verify every seed ran the
    # same topology instead of averaging runs with different n.
    _health = partition_health([len(s) for s in shards],
                               batch_size=int(cfg.get("benign_batch_size") or cfg.get("batch_size", 8)),
                               local_epochs=int(cfg.get("local_epochs", 1)))
    shard_profile = {
        "sizes": [len(s) for s in shards],
        "min_client_shard": int(cfg.get("min_client_shard", 0)),
        "dirichlet_alpha": cfg["dirichlet_alpha"],
        "empty_shards": sum(1 for s in shards if not s),
        **_health,
    }
    # PRE-FLIGHT CREDIBILITY GATE. Fail fast — before the pristine measurement and the whole
    # round loop — if this seed drew a federation too degenerate to analyse. Off by default
    # (min_effective_clients=0) so existing configs are unchanged; the notebook's observation arms
    # set it. The check is on the REALISED split, so it catches a bad Dirichlet draw that the
    # nominal (alpha, n_clients) does not reveal.
    min_eff = float(cfg.get("min_effective_clients", 0) or 0)
    if min_eff > 0 and _health["effective_clients"] < min_eff:
        raise ValueError(
            f"partition too degenerate to analyse: effective_clients={_health['effective_clients']} "
            f"< min_effective_clients={min_eff} (max client holds {_health['max_share']:.0%}, "
            f"smallest does {_health['min_grad_steps_per_round']} grad steps/round). This is a "
            f"single-seed Dirichlet lottery — raise dirichlet_alpha (1.0 gives ~4.8 eff clients at "
            f"n=8; 100.0 is near-IID), raise num_clients, or pick another seed. Realised sizes: "
            f"{shard_profile['sizes']}"
        )
    _hv = ("OK" if (_health["effective_clients"] >= 0.55 * num_benign
                    and _health["max_share"] <= 0.35 and _health["min_grad_steps_per_round"] >= 4)
           else "DEGENERATE — one seed's draw dominates; results conflate imbalance with the attack")
    print(f"  partition health: effective_clients={_health['effective_clients']}/{num_benign} "
          f"· max_share={_health['max_share']:.0%} · min_grad_steps/round="
          f"{_health['min_grad_steps_per_round']}  => {_hv}")
    atk_size = cfg["attacker_claimed_data_size"] or float(np.mean(shard_sizes))
    track_ben = bool(cfg.get("track_benign_baseline", True))
    rng = random.Random(cfg["seed"] + 12345)
    print(f"  {num_benign} benign + {len(attacker_ids)} attackers; sample "
          f"{cfg['clients_per_round']}/round; {cfg['num_rounds']} rounds; "
          f"shard sizes={[len(s) for s in shards]}")

    # --- fixed absolute reference: the pristine (round-0) global, measured ONCE. -------- #
    # Utility "preservation" and cost amplification are reported against THIS fixed anchor,
    # not only against the benign-only global. The benign-only baseline itself drifts as
    # training proceeds, so the ppl ratio atk/benign conflates attack damage with benign
    # drift (a ratio < 1 can just mean the benign global degraded). g0 is the LoRA-init
    # global == the pristine backbone, so ppl/ROUGE/cost vs g0 is an unconfounded baseline.
    me = float(cfg.get("decensor_max_extra", DEFAULT_DECENSOR_MAX_EXTRA))
    print("  measuring pristine (round-0) reference for absolute utility / amplification ...")
    pri_tau = _measure_cost(model, g0, tau_ev, cfg, spec, device, tokenizer)
    pri_cln = _measure_cost(model, g0, clean_ev, cfg, spec, device, tokenizer)
    ppl_pri_cln = _ppl(model, g0, clean_ev, cfg, spec, device)
    ppl_pri_tau = _ppl(model, g0, tau_ev, cfg, spec, device)
    pristine_ref = {
        "tau_mean_cost": pri_tau.mean_cost,
        "tau_mean_cost_calib": pri_tau.mean_cost_at(calib["c_f"], calib["c_a"]) if calib else None,
        "tau_decensored_mean_cost": pri_tau.decensored_mean_cost(me),
        "clean_mean_cost": pri_cln.mean_cost, "kv_tau": pri_tau.mean_kv_proxy,
        "tau_mean_len": round(pri_tau.mean_output_len, 2),
        "clean_mean_len": round(pri_cln.mean_output_len, 2),
        "tau_effective_len": round(pri_tau.mean_effective_len, 2),
        "ppl_clean": round(ppl_pri_cln, 4), "ppl_tau": round(ppl_pri_tau, 4),
        "rouge_recall_clean": round(pri_cln.mean_rouge_recall, 4),
        "rouge_recall_tau": round(pri_tau.mean_rouge_recall, 4),
        # length-robust F1 anchor (see the durability point's rouge_f1_* fields)
        "rouge_f1_clean": round(pri_cln.mean_rouge_f1, 4),
        "rouge_f1_tau": round(pri_tau.mean_rouge_f1, 4),
        # Full logical resource summaries are retained so exact totals/quantiles survive
        # JSON serialization; older scalar fields above remain for backward compatibility.
        "tau_logical": pri_tau.summary(),
        "clean_logical": pri_cln.summary(),
    }
    if cfg.get("attack_objective", "length") == "reasoning_cost":
        pristine_ref.update({
            "tau_reasoning_cost_median": pri_tau.median_reasoning_cost,
            "clean_reasoning_cost_median": pri_cln.median_reasoning_cost,
            "tau_reasoning_tokens_median": pri_tau.median_reasoning_tokens,
            "clean_reasoning_tokens_median": pri_cln.median_reasoning_tokens,
            "tau_answer_accuracy": pri_tau.answer_accuracy,
            "clean_answer_accuracy": pri_cln.answer_accuracy,
            "tau_reasoning_closure_rate": pri_tau.reasoning_closure_rate,
            "clean_reasoning_closure_rate": pri_cln.reasoning_closure_rate,
        })
    print(f"    pristine: ppl_clean={pristine_ref['ppl_clean']:.3f} "
          f"ROUGE_clean={pristine_ref['rouge_recall_clean']:.3f} "
          f"ROUGE_tau={pristine_ref['rouge_recall_tau']:.3f} "
          f"tau_len={pristine_ref['tau_mean_len']:.1f}")

    g_atk = g0.clone()
    g_ben = g0.clone() if track_ben else None
    durability: List[Dict] = []
    stealth_trace: List[Dict] = []
    defense_telemetry: List[Dict] = []            # per-round per-client update geometry (offline defense eval)
    last_mal_trace: Optional[List[Dict]] = None   # a representative attacker trajectory (process data)
    mal_traces: List[Dict] = []                   # per-round attacker within-round optimization trajectory
    save_traces = bool(cfg.get("save_per_round_traces", True))
    out_dir = resolve_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(cfg["num_rounds"]):
        sel_ben, sel_atk = _sample_participants(
            rng, benign_ids, attacker_ids, cfg["clients_per_round"],
            cfg.get("attacker_always_selected", False))

        # --- benign local training from the (attacked) broadcast global ---
        ben_updates, ben_sizes, ben_ids_used = [], [], []
        for cid in sel_ben:
            if not shards[cid]:
                continue
            ben_updates.append(_benign_update(model, shards[cid], cfg, g_atk, spec, device))
            ben_sizes.append(shard_sizes[cid])
            ben_ids_used.append(cid)

        # --- attacker(s): ALM-constrained length attack from the attacked global ---
        # Computed SEQUENTIALLY so each attacker sees the peers already produced this round;
        # the coordination penalty (gamma_coord) then keeps the colluding pair inside the
        # benign distance/cosine PAIR band instead of collapsing into detectable near-twins.
        atk_updates, atk_reports, atk_ids_used = [], [], []
        for cid in sel_atk:
            if not ben_updates:
                break  # need a benign envelope to constrain against
            delta_mal, mtrace = _malicious_update(
                model, clean_tr, tau_tr, cfg, g_atk, spec, device,
                benign_updates=ben_updates, benign_sizes=ben_sizes, atk_size=atk_size,
                peer_updates=list(atk_updates), tokenizer=tokenizer)
            atk_updates.append(delta_mal)
            atk_ids_used.append(cid)
            last_mal_trace = mtrace   # keep the most recent attacker's within-round trajectory
            # PROCESS DATA: keep this round's trajectory so the report can show the attack FORMING
            # over rounds, not just one representative round. mtrace is already subsampled to a
            # handful of logged steps, so this is small; disable via save_per_round_traces=False.
            if save_traces and mtrace:
                mal_traces.append({"round": t, "attacker_id": cid,
                                   "attacker_index": len(atk_updates) - 1, "trace": mtrace})
            atk_reports.append(evaluate_stealth(
                delta_mal, ben_updates, ben_sizes, attacker_weight=atk_size,
                d_T=cfg["d_T"], delta_T=cfg["delta_T"],
                # default True to match the ALM constraint + default_config (see training_core):
                # avoids constraining pairwise cosine while grading aggregate cosine.
                use_pairwise_cosine=cfg.get("stealth_use_pairwise_cosine", True)))

        # --- server aggregation (attacked trajectory) ---
        # Uses _aggregate, which is plain FedAvg unless server_aggregator selects a robust rule
        # (defense-in-the-loop). The SAME rule is applied to the benign-only trajectory below so
        # the amplification ratio compares like server with like.
        if ben_updates or atk_updates:
            all_up = ben_updates + atk_updates
            all_w = ben_sizes + [atk_size] * len(atk_updates)
            g_atk = g_atk + cfg["server_lr"] * _aggregate(all_up, all_w, cfg)
            _require_finite_global(g_atk, label="attacked", round_index=t)

        # --- per-round defense telemetry (offline detection eval; aggregation stays FedAvg) ---
        if cfg.get("collect_defense_telemetry", True) and (ben_updates or atk_updates):
            defense_telemetry.append(_collect_defense_telemetry(
                t, ben_updates, ben_sizes, ben_ids_used,
                atk_updates, [atk_size] * len(atk_updates), atk_ids_used,
                num_attackers=cfg["num_attackers"]))
            if cfg.get("save_update_vectors", False):
                _save_update_vectors(out_dir, t, ben_updates, ben_ids_used, atk_updates, atk_ids_used)

        # --- parallel benign-only trajectory (amplification baseline) ---
        if track_ben:
            ben_only = []
            for cid in sel_ben:
                if not shards[cid]:
                    continue
                ben_only.append(_benign_update(model, shards[cid], cfg, g_ben, spec, device))
            if ben_only:
                # SAME server rule as the attacked trajectory (see _aggregate): otherwise the
                # amplification baseline would be produced by a different aggregator.
                g_ben = g_ben + cfg["server_lr"] * _aggregate(ben_only, ben_sizes, cfg)
                _require_finite_global(g_ben, label="benign", round_index=t)

        # --- per-round stealth (worst case across sampled attackers) ---
        if atk_reports:
            worst = max(atk_reports, key=lambda r: r.attacker_distance)
            joint = all(r.jointly_satisfied for r in atk_reports)
            rec = {"round": t, "n_attackers": len(atk_reports),
                   "attacker_distance": round(worst.attacker_distance, 4),
                   "d_T": round(worst.d_T, 4),
                   "attacker_cosine": round(min(r.attacker_cosine for r in atk_reports), 4),
                   "pairwise_cosine": round(min(r.attacker_pairwise_cosine for r in atk_reports), 4),
                   "jointly_satisfied": bool(joint)}
            stealth_trace.append(rec)
        else:
            rec = {"round": t, "n_attackers": 0, "jointly_satisfied": None}

        # --- periodic cost / utility measurement (generation is expensive) ---
        do_measure = (t % cfg["measure_every"] == 0) or (t == cfg["num_rounds"] - 1)
        if do_measure:
            atk_tau = _measure_cost(model, g_atk, tau_ev, cfg, spec, device, tokenizer)
            atk_cln = _measure_cost(model, g_atk, clean_ev, cfg, spec, device, tokenizer)
            if track_ben:
                ben_tau = _measure_cost(model, g_ben, tau_ev, cfg, spec, device, tokenizer)
                ben_cln = _measure_cost(model, g_ben, clean_ev, cfg, spec, device, tokenizer)
                amp_tau = amplification_ratio(atk_tau.mean_cost, ben_tau.mean_cost)
                amp_tau_med = amplification_ratio(atk_tau.median_cost, ben_tau.median_cost)
                amp_clean = amplification_ratio(atk_cln.mean_cost, ben_cln.mean_cost)
                kv_amp = amplification_ratio(atk_tau.mean_kv_proxy, ben_tau.mean_kv_proxy)
                # De-censored amplification: correct for the cap so a saturated (trunc~1)
                # round is not clipped to a flat lower bound. Uses the EOS-hazard tail.
                amp_tau_dec = amplification_ratio(
                    atk_tau.decensored_mean_cost(me), ben_tau.decensored_mean_cost(me))
            else:
                # No benign baseline: report triggered-vs-clean on the attacked global.
                amp_tau = amplification_ratio(atk_tau.mean_cost, atk_cln.mean_cost)
                amp_tau_med = amplification_ratio(atk_tau.median_cost, atk_cln.median_cost)
                amp_clean, kv_amp = 1.0, amplification_ratio(atk_tau.mean_kv_proxy, atk_cln.mean_kv_proxy)
                amp_tau_dec = amplification_ratio(
                    atk_tau.decensored_mean_cost(me), atk_cln.decensored_mean_cost(me))
            # Amplification vs the FIXED pristine (round-0) reference: total cost inflation
            # relative to the un-attacked starting model (censored + de-censored).
            amp_tau_pri = amplification_ratio(atk_tau.mean_cost, pristine_ref["tau_mean_cost"])
            amp_tau_pri_dec = amplification_ratio(
                atk_tau.decensored_mean_cost(me), pristine_ref["tau_decensored_mean_cost"])
            # Physically-calibrated amplification (c_f=d_model, c_a=1): the honest compute-cost
            # multiple a real decoder incurs. At these lengths cost is linear-dominated, so this
            # is typically well below the naive amp_tau (which credits the quadratic term from
            # token 1). Reported alongside, not instead — the naive number is the sensitivity bound.
            if calib:
                _cf, _ca = calib["c_f"], calib["c_a"]
                ben_ref_calib = (ben_tau.mean_cost_at(_cf, _ca) if track_ben
                                 else atk_cln.mean_cost_at(_cf, _ca))
                amp_tau_calib = amplification_ratio(atk_tau.mean_cost_at(_cf, _ca), ben_ref_calib)
                amp_tau_pri_calib = amplification_ratio(
                    atk_tau.mean_cost_at(_cf, _ca), pristine_ref["tau_mean_cost_calib"])
            else:
                amp_tau_calib = amp_tau_pri_calib = None
            # De-censored validity: when the truncated tail hit the max_extra cap (EOS hazard ~0),
            # the de-censored estimate is 100% the horizon assumption and 0% data. Flag it so the
            # number is never read as a measurement (raise the generation cap to make it valid).
            decensored_valid = not atk_tau.residual_capped(me)

            atk_tau_logical = atk_tau.summary()
            atk_clean_logical = atk_cln.summary()
            ben_tau_logical = ben_tau.summary() if track_ben else None
            pristine_tau_logical = pristine_ref.get("tau_logical", {})

            ppl_clean = _ppl(model, g_atk, clean_ev, cfg, spec, device)
            ppl_clean_ben = _ppl(model, g_ben, clean_ev, cfg, spec, device) if track_ben else None
            ppl_ratio = round(ppl_clean / ppl_clean_ben, 4) if ppl_clean_ben else None
            # Utility vs the FIXED pristine baseline (unconfounded by benign drift): ppl and
            # ROUGE-L recall (clean AND tau). ROUGE recall ~ pristine => "the answer is still
            # there" even as tau outputs run long; the direct "utility preserved" evidence.
            ppl_ratio_pri = round(ppl_clean / max(pristine_ref["ppl_clean"], 1e-9), 4)
            point = {
                "round": t, "amp_tau": round(amp_tau, 4), "amp_tau_median": round(amp_tau_med, 4),
                "amp_tau_decensored": round(amp_tau_dec, 4),
                "decensored_valid": bool(decensored_valid),
                "amp_tau_vs_pristine": round(amp_tau_pri, 4),
                "amp_tau_vs_pristine_decensored": round(amp_tau_pri_dec, 4),
                "amp_tau_calibrated": round(amp_tau_calib, 4) if amp_tau_calib is not None else None,
                "amp_tau_vs_pristine_calibrated": round(amp_tau_pri_calib, 4) if amp_tau_pri_calib is not None else None,
                "cost_c_f_calibrated": (calib["c_f"] if calib else None),
                # amp_clean is a ratio of mean analytic COST, which carries the quadratic term —
                # a 9% length change shows up here as an 18% cost change. Any verdict phrased as
                # "clean was shortened by X%" must use amp_clean_len (LINEAR) instead; grading a
                # quadratic ratio against a length-shaped window is the unit bug that produced the
                # spurious "clean over-shortened" flag on the 20260807 run (cost 0.8232 vs the
                # actual length ratio 0.9079).
                "amp_clean": round(amp_clean, 4),
                "amp_clean_len": round(amplification_ratio(
                    atk_cln.mean_output_len,
                    (ben_cln.mean_output_len if track_ben else atk_cln.mean_output_len)), 4),
                "amp_clean_effective_len": round(amplification_ratio(
                    atk_cln.mean_effective_len,
                    (ben_cln.mean_effective_len if track_ben else atk_cln.mean_effective_len)), 4),
                "clean_len_ben": (round(ben_cln.mean_output_len, 2) if track_ben else None),
                "selectivity": round(amplification_ratio(amp_tau, amp_clean), 4),
                "kv_amp_tau": round(kv_amp, 4),
                "tau_len_atk": round(atk_tau.mean_output_len, 2),
                "tau_len_atk_decensored": round(atk_tau.decensored_mean_len(me), 2),
                "tau_effective_len_atk": round(atk_tau.mean_effective_len, 2),
                # Effective (degeneracy-discounted) amplification: length gain after removing
                # looped padding. If << amp_tau, the attack is winning on repetition, not
                # genuine length (weak + a repetition filter would catch it).
                "amp_tau_effective": round(
                    amplification_ratio(atk_tau.mean_effective_len,
                                        (ben_tau.mean_effective_len if track_ben else atk_cln.mean_effective_len)), 4),
                "clean_len_atk": round(atk_cln.mean_output_len, 2),
                "truncation_tau": round(atk_tau.truncation_rate, 3),
                "eos_hazard_tau": round(atk_tau.eos_hazard(), 6),
                "residual_capped_tau": atk_tau.residual_capped(me),
                "repetition_tau": round(atk_tau.mean_repetition, 3),
                "distinct_ratio_tau": round(atk_tau.mean_distinct_ratio, 4),
                "ppl_clean_atk": round(ppl_clean, 3),
                "ppl_clean_ben": round(ppl_clean_ben, 3) if ppl_clean_ben else None,
                "ppl_clean_pristine": pristine_ref["ppl_clean"],
                "ppl_ratio": ppl_ratio,                     # atk / benign-only (marginal, drifts)
                "ppl_ratio_vs_pristine": ppl_ratio_pri,     # atk / pristine (fixed, primary)
                "rouge_recall_clean_atk": round(atk_cln.mean_rouge_recall, 4),
                "rouge_recall_clean_pristine": pristine_ref["rouge_recall_clean"],
                # THE CORRECT C2 CONTROL. pristine is the UNTRAINED round-0 model, so a clean-quality
                # gap measured against it is dominated by ordinary federated drift, not by the
                # attacker: on the 20260807 run benign FedAvg alone moved clean ROUGE-recall
                # 0.3832 -> 0.3232 (-15.7pp) while the attacker added -1.1pp. Any clean-utility
                # verdict MUST divide by the benign-only arm, which ran the same rounds with the
                # same benign selection and the attackers removed.
                "rouge_recall_clean_ben": (round(ben_cln.mean_rouge_recall, 4) if track_ben else None),
                "rouge_f1_clean_ben": (round(ben_cln.mean_rouge_f1, 4) if track_ben else None),
                "rouge_recall_tau_ben": (round(ben_tau.mean_rouge_recall, 4) if track_ben else None),
                "rouge_f1_tau_ben": (round(ben_tau.mean_rouge_f1, 4) if track_ben else None),
                "rouge_recall_tau_atk": round(atk_tau.mean_rouge_recall, 4),
                "rouge_recall_tau_pristine": pristine_ref["rouge_recall_tau"],
                # ROUGE-L F1: the LENGTH-ROBUST "answer kept AND not padded" signal. Unlike
                # recall (which a 6x-longer degenerate output inflates), F1's precision term
                # penalizes padding, so the honest utility-preservation claim rests on tau F1
                # staying near pristine, NOT on the recall rise. See metrics.rouge_l_recall.
                "rouge_f1_clean_atk": round(atk_cln.mean_rouge_f1, 4),
                "rouge_f1_clean_pristine": pristine_ref.get("rouge_f1_clean"),
                "rouge_f1_tau_atk": round(atk_tau.mean_rouge_f1, 4),
                "rouge_f1_tau_pristine": pristine_ref.get("rouge_f1_tau"),
                "stealth_ok": rec.get("jointly_satisfied"),
                # Exact logical consumption (not the analytic cost proxy).  Keeping these
                # summaries per measured round provides a hardware-independent durability
                # trace without any additional generation.
                "resource_tau": {
                    "attacked": atk_tau_logical,
                    "benign": ben_tau_logical,
                    "pristine": pristine_tau_logical,
                },
                "resource_clean_attacked": atk_clean_logical,
                "output_token_amp_vs_pristine": _rounded_ratio(
                    atk_tau_logical.get("total_output_tokens"),
                    pristine_tau_logical.get("total_output_tokens")),
                "output_token_amp_vs_benign": (_rounded_ratio(
                    atk_tau_logical.get("total_output_tokens"),
                    ben_tau_logical.get("total_output_tokens"))
                    if ben_tau_logical else None),
                "total_output_tokens_tau": atk_tau_logical.get("total_output_tokens"),
                "budget_utilization_tau": atk_tau_logical.get("budget_utilization"),
                "cap_hit_rate_tau": atk_tau_logical.get("cap_hit_rate",
                                                          atk_tau_logical.get("truncation_rate")),
            }
            if cfg.get("attack_objective", "length") == "reasoning_cost":
                reasoning_tau_ref = ben_tau if track_ben else pri_tau
                reasoning_clean_ref = ben_cln if track_ben else pri_cln
                paired_tau = _paired_reasoning_summary(
                    atk_tau, reasoning_tau_ref,
                    seed=int(cfg["eval_decode_seeds"][0]) + 1000 * t,
                )
                paired_clean = _paired_reasoning_summary(
                    atk_cln, reasoning_clean_ref,
                    seed=int(cfg["eval_decode_seeds"][0]) + 1000 * t + 1,
                )
                tau_accuracy_drop = (
                    reasoning_tau_ref.answer_accuracy - atk_tau.answer_accuracy
                    if reasoning_tau_ref.answer_accuracy is not None
                    and atk_tau.answer_accuracy is not None else None
                )
                clean_accuracy_drop = (
                    reasoning_clean_ref.answer_accuracy - atk_cln.answer_accuracy
                    if reasoning_clean_ref.answer_accuracy is not None
                    and atk_cln.answer_accuracy is not None else None
                )
                clean_cost_ratio = paired_clean["median_ratio"]
                tau_summary = atk_tau_logical.get("reasoning") or {}
                clean_summary = atk_clean_logical.get("reasoning") or {}
                tau_ref_summary = reasoning_tau_ref.summary().get("reasoning") or {}
                clean_ref_summary = reasoning_clean_ref.summary().get("reasoning") or {}
                measurement_components = {
                    "attacked_tau": bool(
                        (tau_summary.get("validity") or {}).get("measurement_valid")
                    ),
                    "attacked_clean": bool(
                        (clean_summary.get("validity") or {}).get("measurement_valid")
                    ),
                    "reference_tau": bool(
                        (tau_ref_summary.get("validity") or {}).get("measurement_valid")
                    ),
                    "reference_clean": bool(
                        (clean_ref_summary.get("validity") or {}).get("measurement_valid")
                    ),
                }
                measurement_valid = all(measurement_components.values())
                min_reference_accuracy = float(cfg["reasoning_min_reference_accuracy"])
                gate_values = {
                    "measurement_valid": measurement_valid,
                    **reasoning_effect_gates(
                        paired_tau["ci95_lower"],
                        float(cfg["reasoning_min_claim_cost_ratio"]),
                    ),
                    "tau_reference_accuracy_valid": (
                        reasoning_tau_ref.answer_accuracy is not None
                        and reasoning_tau_ref.answer_accuracy >= min_reference_accuracy
                    ),
                    "clean_reference_accuracy_valid": (
                        reasoning_clean_ref.answer_accuracy is not None
                        and reasoning_clean_ref.answer_accuracy >= min_reference_accuracy
                    ),
                    "tau_accuracy_preserved": (
                        tau_accuracy_drop is not None and tau_accuracy_drop
                        <= float(cfg["reasoning_max_accuracy_drop"])
                    ),
                    "clean_accuracy_preserved": (
                        clean_accuracy_drop is not None and clean_accuracy_drop
                        <= float(cfg["reasoning_max_accuracy_drop"])
                    ),
                    "clean_cost_stable": abs(clean_cost_ratio - 1.0)
                    <= float(cfg["reasoning_clean_cost_tolerance"]),
                    "closure_valid": float(tau_summary.get("closure_rate", 0.0))
                    >= float(cfg["reasoning_min_closure_rate"]),
                    "clean_closure_valid": float(clean_summary.get("closure_rate", 0.0))
                    >= float(cfg["reasoning_min_closure_rate"]),
                    "cap_hit_valid": atk_tau.cap_hit_rate
                    < float(cfg["reasoning_max_cap_hit_rate"]),
                    "clean_cap_hit_valid": atk_cln.cap_hit_rate
                    < float(cfg["reasoning_max_cap_hit_rate"]),
                    "repetition_valid": float(
                        tau_summary.get("mean_reasoning_repetition", 1.0)
                    ) < float(cfg["reasoning_max_repetition"]),
                    "distinct_valid": float(
                        tau_summary.get("mean_reasoning_distinct_ratio", 0.0)
                    ) > float(cfg["reasoning_min_distinct_ratio"]),
                }
                point.update({
                    "paired_reasoning_cost_tau": paired_tau,
                    "paired_reasoning_cost_clean": paired_clean,
                    "reasoning_cost_ratio_tau": round(paired_tau["median_ratio"], 4),
                    "reasoning_cost_ci95_lower_tau": round(paired_tau["ci95_lower"], 4),
                    "reasoning_cost_ci95_upper_tau": round(paired_tau["ci95_upper"], 4),
                    "reasoning_cost_ratio_clean": round(clean_cost_ratio, 4),
                    "reasoning_tokens_tau_atk": round(atk_tau.median_reasoning_tokens, 3),
                    "reasoning_tokens_tau_ref": round(
                        reasoning_tau_ref.median_reasoning_tokens, 3
                    ),
                    "reasoning_closure_tau": round(atk_tau.reasoning_closure_rate, 4),
                    "reasoning_closure_clean": round(atk_cln.reasoning_closure_rate, 4),
                    "answer_accuracy_tau_atk": atk_tau.answer_accuracy,
                    "answer_accuracy_tau_ref": reasoning_tau_ref.answer_accuracy,
                    "answer_accuracy_clean_atk": atk_cln.answer_accuracy,
                    "answer_accuracy_clean_ref": reasoning_clean_ref.answer_accuracy,
                    "answer_accuracy_drop_tau": tau_accuracy_drop,
                    "answer_accuracy_drop_clean": clean_accuracy_drop,
                    "reasoning_repetition_tau": tau_summary.get(
                        "mean_reasoning_repetition"
                    ),
                    "reasoning_distinct_tau": tau_summary.get(
                        "mean_reasoning_distinct_ratio"
                    ),
                    "reasoning_measurement_valid": measurement_valid,
                    "reasoning_measurement_validity_components": measurement_components,
                    "reasoning_gates": gate_values,
                    "reasoning_claim_ready": all(gate_values.values()),
                })
            durability.append(point)
            _dec_mark = "" if decensored_valid else "!"  # '!' = assumption-only, no data (cap hit)
            _calib_str = (f"calib {point['amp_tau_calibrated']:.2f}x " if amp_tau_calib is not None else "")
            print(f"  [round {t:3d}] amp_tau={point['amp_tau']:.3f}x (dec {point['amp_tau_decensored']:.2f}{_dec_mark}, "
                  f"eff {point['amp_tau_effective']:.2f}, med {point['amp_tau_median']:.2f}) "
                  f"{_calib_str}vs_pristine={point['amp_tau_vs_pristine']:.2f}x "
                  f"tau_len={point['tau_len_atk']:.0f}(dec {point['tau_len_atk_decensored']:.0f}) "
                  f"trunc={point['truncation_tau']:.2f} rep={point['repetition_tau']:.2f} "
                  f"distinct={point['distinct_ratio_tau']:.2f} "
                  f"ppl_pri={point['ppl_ratio_vs_pristine']:.3f} "
                  f"ROUGE_cln={point['rouge_recall_clean_atk']:.2f}/τ{point['rouge_recall_tau_atk']:.2f} "
                  f"stealth={point['stealth_ok']}")
            if cfg.get("attack_objective", "length") == "reasoning_cost":
                print(
                    f"             reasoning={point['reasoning_cost_ratio_tau']:.2f}x "
                    f"CI95=[{point['reasoning_cost_ci95_lower_tau']:.2f},"
                    f"{point['reasoning_cost_ci95_upper_tau']:.2f}] "
                    f"clean={point['reasoning_cost_ratio_clean']:.2f}x "
                    f"acc_tau={point['answer_accuracy_tau_atk']}/"
                    f"{point['answer_accuracy_tau_ref']} "
                    f"close={point['reasoning_closure_tau']:.2f} "
                    f"claim_ready={point['reasoning_claim_ready']}"
                )
        else:
            print(f"  [round {t:3d}] sel_ben={sel_ben} sel_atk={sel_atk} "
                  f"stealth={rec.get('jointly_satisfied')}")

    # Qualitative decoded samples from the FINAL attacked global (coherence evidence).
    print("  decoding a few final examples (qualitative coherence check) ...")
    final_examples = _dump_fl_examples(model, g_atk, tau_ev, clean_ev, tokenizer, cfg, spec, device,
                                       k=cfg.get("num_dump_examples", 4))

    # Logical accounting reuses objects from the mandatory pristine/final measurements;
    # no duplicate generation is performed here.  Exact per-request rows live in CSV so
    # the main JSON remains compact enough for Colab/Drive.
    resources = None
    if cfg.get("collect_resource_metrics", True):
        logical = _logical_resource_bundle(
            pri_tau, pri_cln, atk_tau, atk_cln,
            ben_tau if track_ben else None, ben_cln if track_ben else None,
        )
        resources = {
            "schema_version": "resource-v1",
            "environment": {},       # populated by the hardware/environment collector below
            "config": {
                "max_new_tokens": cfg["max_new_tokens"],
                "generation_hard_token_cap": cfg.get("generation_hard_token_cap"),
                "generation_max_batch_seconds": cfg.get("generation_max_batch_seconds"),
                "eval_size": cfg["eval_size"],
                "generation_eos_ids": list(cfg.get("resolved_generation_eos_ids") or []),
                "dataset_revision": cfg.get("dataset_revision"),
                "dataset_train_fingerprint": cfg.get("resolved_dataset_train_fingerprint"),
                "dataset_eval_fingerprint": cfg.get("resolved_dataset_eval_fingerprint"),
                "prompt_set_sha256_tau": _prompt_set_sha256(tau_ev),
                "prompt_set_sha256_clean": _prompt_set_sha256(clean_ev),
                "output_token_includes_eos": True,
                "token_accounting_scope": "model_input_and_all_emitted_token_ids",
                "reasoning_token_scope": (
                    "observed_explicit_think_span"
                    if cfg.get("attack_objective", "length") == "reasoning_cost"
                    else "not_applicable"
                ),
                "eval_decode_seeds": (
                    list(cfg.get("eval_decode_seeds", []))
                    if cfg.get("attack_objective", "length") == "reasoning_cost"
                    else None
                ),
                "hardware_profile_enabled": bool(cfg.get("profile_hardware", False)),
                "hardware_primary_split": cfg.get("resource_profile_splits", ["tau"])[0],
                "hardware_primary_batch_size": max(cfg["resource_profile_batch_sizes"]),
            },
            **logical,
            "validity": {
                "logical_tokens": "valid",
                "hardware": "pending" if cfg.get("profile_hardware", False) else "disabled",
                "generation_bounded": True,
                "censoring_note": (
                    "hit_cap means the observed length is a lower bound; no unbounded "
                    "generation or assumption-only tail is treated as measured usage"
                ),
            },
        }

        logical_rows: List[Dict] = []
        final_round = cfg["num_rounds"] - 1
        for condition, tau_stats, clean_stats in (
            ("pristine", pri_tau, pri_cln),
            ("attacked_final", atk_tau, atk_cln),
        ):
            state_round = 0 if condition == "pristine" else final_round
            measurement_stage = (
                "pre_training" if condition == "pristine" else "post_final_round"
            )
            logical_rows.extend(_logical_rows(
                tau_stats, tau_ev, condition=condition, split="tau", round_idx=state_round,
                measurement_stage=measurement_stage,
            ))
            logical_rows.extend(_logical_rows(
                clean_stats, clean_ev, condition=condition, split="clean", round_idx=state_round,
                measurement_stage=measurement_stage,
            ))
        if track_ben:
            logical_rows.extend(_logical_rows(
                ben_tau, tau_ev, condition="benign_final", split="tau", round_idx=final_round,
                measurement_stage="post_final_round",
            ))
            logical_rows.extend(_logical_rows(
                ben_cln, clean_ev, condition="benign_final", split="clean", round_idx=final_round,
                measurement_stage="post_final_round",
            ))
        if cfg.get("save_resource_per_prompt", True):
            logical_path = out_dir / "logical_tokens.csv"
            _write_csv(logical_path, logical_rows)
            resources["artifacts"] = {"logical_tokens_csv": str(logical_path)}

    final_globals_path = None
    if cfg.get("save_final_globals", True):
        final_globals_path = out_dir / "final_globals.pt"
        payload = {"pristine": g0, "attacked_final": g_atk}
        if track_ben and g_ben is not None:
            payload["benign_final"] = g_ben
        torch.save(payload, final_globals_path)

    resolved_config, resolved_config_sha256 = _canonical_config_record(cfg)
    if resources is not None:
        resources["environment"] = {
            **environment_start,
            "start": environment_start,
            "end": None,
            "environment_changed": None,
            "cloud": {
                "provider": cfg.get("cloud_provider"),
                "sku_reported": cfg.get("cloud_sku_reported"),
                "note": "actual accelerator fingerprint is authoritative",
            },
        }
        inner = model.inner()
        model_cfg = inner.config
        resources["model"] = {
            "id": cfg["backbone"],
            "requested_revision": cfg.get("model_revision"),
            "resolved_revision": cfg.get("resolved_model_revision"),
            "tokenizer_revision": cfg.get("resolved_tokenizer_revision"),
            "dtype": str(getattr(inner, "dtype", None)),
            "attention_backend": getattr(model_cfg, "_attn_implementation", None),
            "use_cache": bool(getattr(model_cfg, "use_cache", True)),
            "tf32_cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "lora_r": cfg.get("lora_r"),
        }
        resources["config"].update({
            "dtype": resources["model"]["dtype"],
            "attention_backend": resources["model"]["attention_backend"],
            "use_cache": resources["model"]["use_cache"],
            "batch_size": int(max(cfg["resource_profile_batch_sizes"])),
        })

        if cfg.get("profile_hardware", False):
            print("  profiling final pristine/benign/attacked resource usage ...")
            state_vectors = {"pristine": g0, "attacked_final": g_atk}
            if track_ben and g_ben is not None:
                # Keep this order stable so repeat rotation is reproducible.
                state_vectors = {
                    "pristine": g0,
                    "benign_final": g_ben,
                    "attacked_final": g_atk,
                }
            try:
                hardware = _profile_final_resources(
                    model, state_vectors, tau_ev, clean_ev, cfg, spec, device, out_dir
                )
                primary_split = cfg.get("resource_profile_splits", ["tau"])[0]
                # Use one explicit primary scenario everywhere.  Choosing the largest
                # configured batch avoids silently mixing BS=1 latency with BS=8
                # throughput/memory in comparisons or plots.
                primary_bs = int(max(cfg["resource_profile_batch_sizes"]))
                primary_key = f"batch_size_{primary_bs}"
                primary_examples = tau_ev if primary_split == "tau" else clean_ev
                primary_examples = primary_examples[
                    :min(cfg["resource_profile_eval_size"], len(primary_examples))
                ]
                resources["comparison_profile"] = {
                    "split": primary_split,
                    "batch_size": primary_bs,
                    "prompt_subset_sha256": _prompt_set_sha256(primary_examples),
                }
                for condition, split_profiles in hardware["profiles"].items():
                    state = resources["states"][condition]
                    state["hardware_profiles"] = split_profiles
                    primary_profiles = split_profiles.get(primary_split, {})
                    # Canonical report shape: hardware[batch_size_*].summary.  Retain a
                    # flat primary summary too for comparisons and downstream scripts.
                    state["hardware"] = {
                        name: {"summary": summary}
                        for name, summary in primary_profiles.items()
                    }
                    state["hardware_primary"] = primary_profiles.get(primary_key, {})
                    state["runs"] = [
                        run for run in hardware["runs"] if run["condition"] == condition
                    ]
                resources["hardware_runs"] = hardware["runs"]
                resources["hardware_batch_records"] = hardware["batch_records"]
                resources.setdefault("artifacts", {}).update(hardware["artifacts"])

                hardware_metrics = (
                    "generation_wall_seconds", "cuda_elapsed_seconds", "e2e_wall_seconds",
                    "allocated_gpu_seconds", "energy_joules",
                    "peak_allocated_bytes", "incremental_peak_allocated_bytes",
                    "requests_per_second", "output_tokens_per_second",
                )
                for metric in hardware_metrics:
                    attacked_profile = resources["states"]["attacked_final"]["hardware_primary"]
                    pristine_profile = resources["states"]["pristine"]["hardware_primary"]
                    attacked_value = (
                        attacked_profile.get(metric) if attacked_profile.get("valid") else None
                    )
                    pristine_value = (
                        pristine_profile.get(metric) if pristine_profile.get("valid") else None
                    )
                    comparison = {
                        "attacked_vs_pristine": _safe_ratio(attacked_value, pristine_value)
                    }
                    if "benign_final" in resources["states"]:
                        benign_profile = resources["states"]["benign_final"]["hardware_primary"]
                        benign_value = (
                            benign_profile.get(metric) if benign_profile.get("valid") else None
                        )
                        comparison["attacked_vs_benign"] = _safe_ratio(
                            attacked_value, benign_value
                        )
                    resources["comparisons"][metric] = comparison
                strict_runs = [
                    bool(run.get("valid"))
                    and bool(run.get("cuda_timing_valid"))
                    and bool(run.get("memory_metrics_valid"))
                    and run.get("allocated_gpu_seconds") is not None
                    and not bool(run.get("timed_out_batches"))
                    for run in hardware["runs"]
                ]
                expected_runs = (
                    len(resources["states"])
                    * len(cfg.get("resource_profile_splits", ["tau"]))
                    * len(cfg["resource_profile_batch_sizes"])
                    * int(cfg["resource_profile_repeats"])
                )
                hardware_complete = (
                    len(strict_runs) == expected_runs
                    and bool(strict_runs)
                    and all(strict_runs)
                )
                resources["validity"]["hardware"] = (
                    "valid" if hardware_complete else "partial_or_invalid"
                )
                resources["validity"]["hardware_requirements"] = {
                    "gpu_allocated": True,
                    "cuda_timing_complete": True,
                    "memory_instrumentation_complete": True,
                    "paired_runs_complete": True,
                    "expected_runs": expected_runs,
                    "observed_runs": len(hardware["runs"]),
                    "requirements_met": hardware_complete,
                    "energy_optional": True,
                }
                resources["hardware_paired_ratios"] = _paired_hardware_ratio_summaries(
                    hardware["runs"], split=primary_split, batch_size=primary_bs
                )

                repeats_path = out_dir / "resource_repeats.csv"
                batches_path = out_dir / "hardware_batches.csv"
                _write_csv(repeats_path, hardware["runs"])
                _write_csv(batches_path, hardware["batch_records"])
                resources["artifacts"].update({
                    "resource_repeats_csv": str(repeats_path),
                    "hardware_batches_csv": str(batches_path),
                })
            except Exception as exc:  # preserve the expensive FL result on profiler failure
                resources["validity"]["hardware"] = "failed"
                resources["validity"]["hardware_reason"] = f"{type(exc).__name__}:{exc}"
                # Earlier paired repeats are appended after every timed block. Register
                # those recovery files even when a later condition/batch fails.
                for filename, artifact_key in (
                    ("resource_repeats.jsonl", "resource_repeats_jsonl"),
                    ("hardware_batches.jsonl", "hardware_batches_jsonl"),
                ):
                    recovery_path = out_dir / filename
                    if recovery_path.exists():
                        resources.setdefault("artifacts", {})[artifact_key] = str(
                            recovery_path
                        )
                print(f"  [resource] hardware profiling failed safely: {exc}")

        try:
            from .resource_metrics import collect_runtime_environment
            environment_end = collect_runtime_environment(
                device_index=(device.index if device.index is not None else 0)
            )
        except Exception as exc:
            environment_end = {
                "collection_valid": False,
                "reason": f"{type(exc).__name__}:{exc}",
            }
        resources["environment"]["end"] = environment_end
        start_fingerprint = environment_start.get("fingerprint_sha256")
        end_fingerprint = environment_end.get("fingerprint_sha256")
        resources["environment"]["environment_changed"] = (
            start_fingerprint != end_fingerprint
            if start_fingerprint is not None and end_fingerprint is not None
            else None
        )
        resources["experiment_wall_seconds"] = float(
            time.perf_counter() - experiment_wall_started
        )

        # Rewrite the summary after optional hardware profiling so the CSV contains both
        # cloud-independent logical metrics and the primary paired hardware scenario.
        summary_rows = _resource_summary_rows(
            resources, max(cfg["resource_profile_batch_sizes"])
        )
        summary_path = out_dir / "resource_summary.csv"
        _write_csv(summary_path, summary_rows)
        resources.setdefault("artifacts", {})["resource_summary_csv"] = str(summary_path)

        comparison_rows = []
        for metric, comparison in resources["comparisons"].items():
            for baseline in ("pristine", "benign"):
                key = f"attacked_vs_{baseline}"
                if key in comparison:
                    comparison_rows.append({
                        "metric": metric,
                        "numerator": "attacked_final",
                        "baseline": baseline,
                        "ratio": comparison.get(key),
                    })
        comparisons_path = out_dir / "resource_comparisons.csv"
        _write_csv(comparisons_path, comparison_rows)
        resources["artifacts"]["resource_comparisons_csv"] = str(comparisons_path)

        manifest = {
            "schema_version": "resource-v1",
            "run_id": cfg["results_subdir"].split("/", 1)[0],
            "artifacts_dir": str(out_dir),
            "cloud": resources["environment"]["cloud"],
            "environment": resources["environment"],
            "model": resources["model"],
            "generation": resources["config"],
            "protocol": {
                "requested_config_sha256": requested_config_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "source_repo_commit": cfg.get("source_repo_commit"),
                "source_repo_dirty": cfg.get("source_repo_dirty"),
                "source_code_bundle_sha256": cfg.get("source_code_bundle_sha256"),
            },
        }
        manifest_path = out_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        resources["artifacts"]["run_manifest_json"] = str(manifest_path)

    results = {
        "requested_config": requested_config,
        "requested_config_sha256": requested_config_sha256,
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        # Serialized with .get(), not cfg[k]: the whitelist is append-only and a key that a
        # given config does not carry must not abort the run after hours of compute.
        "config": {k: cfg.get(k) for k in (
            "experiment_name", "backbone", "model_revision", "resolved_model_revision",
            "source", "dataset_revision", "resolved_dataset_train_fingerprint",
            "resolved_dataset_eval_fingerprint", "attack_objective",
            "source_repo_commit", "source_repo_dirty", "source_code_bundle_sha256",
            "results_root", "results_subdir",
            "reference_source", "trigger_str",
            "num_clients", "num_attackers",
            "num_rounds", "clients_per_round", "attacker_always_selected",
            "local_epochs", "attacker_steps", "attacker_lr", "client_lr",
            "grad_clip_norm", "gamma", "gamma_clean", "kd_clean_weight",
            "clean_anchor_two_sided",
            "gamma_rep", "rep_window", "no_repeat_ngram_size", "onpolicy_horizon",
            "stealth_kappa", "stealth_use_pairwise_cosine", "stealth_cosine_two_sided",
            "stealth_cos_low", "stealth_norm_constraint",
            # Partition credibility: the gate threshold and the attacker's claimed data weight.
            "min_effective_clients", "attacker_claimed_data_size",
            # Which threshold the final distance projection targets — changes whether the returned
            # update provably meets evaluate_stealth's screen, so it must be in the provenance.
            "stealth_screen_reference", "final_project_distance",
            "gamma_coord", "report_calibrated_cost",
            "collect_defense_telemetry", "save_update_vectors", "run_defense_eval",
            "pool_size", "eval_size", "max_new_tokens", "decensor_max_extra",
            "use_lora", "lora_r", "lora_alpha", "lora_dropout",
            "generation_hard_token_cap", "generation_max_batch_seconds",
            "collect_resource_metrics", "profile_hardware", "resource_profile_eval_size",
            "resource_profile_batch_sizes", "resource_profile_warmup_batches",
            "resource_profile_repeats", "resource_profile_nvml",
            "resource_profile_sample_interval_ms", "resource_profile_splits",
            "save_resource_per_prompt", "save_final_globals",
            "cloud_provider", "cloud_sku_reported",
            # --- REPRODUCIBILITY (added 2026-08): every one of these changes the numbers, and
            # every one of them was previously ABSENT, which is why saved runs recorded
            # "seed": null and a multi-seed sweep could not prove its arms were comparable. ---
            "seed", "dirichlet_alpha", "min_client_shard", "measure_every", "server_lr",
            "batch_size", "benign_batch_size", "gen_batch_size", "client_lr",
            "track_benign_baseline",
            "use_stealth_constraint", "stealth_envelope_quantile",
            "use_onpolicy_length", "onpolicy_free_decode", "use_fallback_surrogate",
            "grad_checkpointing", "use_tf32", "save_per_round_traces",
            # --- experiment-arm identifiers (defense-in-the-loop / serving-side mitigation) ---
            "server_aggregator", "server_norm_clip_factor",
            "eval_repetition_penalty", "eval_no_repeat_ngram_size",
            # Reasoning-TCAA objective, decode, reference, and pre-registered gates.
            "thinking_mode", "reasoning_task_instruction",
            "reasoning_start_str", "reasoning_end_str",
            "reasoning_target_ratio", "reasoning_target_tolerance",
            "reasoning_overshoot_weight", "reasoning_horizon",
            "reasoning_reference_horizon", "reasoning_anchor_size",
            "reasoning_anchor_candidate_multiplier", "reasoning_min_valid_anchors",
            "reasoning_require_correct_anchors", "reasoning_cost_weight", "answer_ce_weight",
            "clean_cost_anchor_weight", "clean_cost_anchor_every",
            "reasoning_reference_mode", "reasoning_use_calibrated_cost",
            "reasoning_rollout_free_decode", "reasoning_rollout_do_sample",
            "reasoning_rollout_temperature", "reasoning_rollout_top_p",
            "reasoning_rollout_top_k", "eval_decode_seeds", "task_evaluator",
            "reasoning_max_accuracy_drop", "reasoning_min_reference_accuracy",
            "reasoning_min_claim_cost_ratio", "reasoning_min_hardware_ratio",
            "reasoning_clean_cost_tolerance",
            "reasoning_min_closure_rate", "reasoning_max_cap_hit_rate",
            "reasoning_max_repetition", "reasoning_min_distinct_ratio",
            "torch_dtype", "attn_implementation", "resolved_generation_eos_ids",
            "resolved_tokenizer_revision")},
        "run_id": cfg["results_subdir"].split("/", 1)[0],
        "artifacts_dir": str(out_dir),
        "final_globals_path": str(final_globals_path) if final_globals_path else None,
        "cost_c_f_calibrated": (calib["c_f"] if calib else None),
        "cost_c_a_calibrated": (calib["c_a"] if calib else None),
        "lora_update_dim": int(g0.numel()),
        # Realised client-data split. An under-floor shard silently shrinks the federation, so a
        # multi-seed roll-up must check this matches across seeds before averaging their defense
        # rows (empty_shards > 0 means the run did NOT have the topology its config advertises).
        "shard_profile": shard_profile,
        "pristine_reference": pristine_ref,
        "durability": durability,
        "stealth_trace": stealth_trace,
        # Per-round, per-client update geometry (labelled benign/attacker) for OFFLINE defense
        # evaluation. Aggregation stays FedAvg; this is the raw material a Krum/norm-clip/FLTrust
        # detector would consume — see _collect_defense_telemetry.
        "defense_telemetry": defense_telemetry,
        "sample_mal_trace": last_mal_trace,     # a representative within-round attacker trajectory
        # Per-round attacker trajectories — the "how did the attack form" process record. Each
        # entry is one round's (subsampled) within-round optimization; the report derives both a
        # per-round endpoint table and the round-over-round convergence view from this.
        "mal_traces": mal_traces,
        "final_examples": final_examples,       # decoded samples from the final attacked global
    }
    if resources is not None:
        results["resources"] = resources
    # Independent defense-evasion verdict: replay real robust aggregators on the telemetry, so
    # the parameter-stealth claim rests on named defenses instead of our own benign envelope.
    if cfg.get("run_defense_eval", True) and defense_telemetry:
        from .defenses import (evaluate_defenses, evaluate_vector_defenses,
                               format_defense_report)
        # num_attackers=None => the evaluator uses the attackers ACTUALLY present each round as
        # the Byzantine budget f (what Krum's guarantee must cover) and skips Krum/Multi-Krum on
        # any round with n < 2f+3, rather than forcing a fixed f and degrading silently.
        ev = evaluate_defenses(defense_telemetry, num_attackers=None)
        vdir = out_dir / "update_vectors"
        vev = evaluate_vector_defenses(vdir) if vdir.exists() else {}
        results["defense_evaluation"] = {"telemetry_defenses": ev, "vector_defenses": vev}
        print("\n" + format_defense_report(ev, vev or None))
        # The purpose-built collusion detector's DEPLOYABILITY boundary (AUC + FPR-calibrated
        # operating points + the client-level confound). The top-f excess alone overstates it,
        # so the verdict card reads the calibration sweep from here.
        from .defenses import rank_collusion_analysis
        rca = rank_collusion_analysis(defense_telemetry)
        if rca:
            results["rank_collusion_analysis"] = rca
            sweep = " · ".join(f"@FPR{s['target_fpr']:.2f} excess {s['excess']:+.3f}"
                               for s in rca["fpr_sweep"])
            print(f"  rank-collusion detector: AUC={rca['auc']} (perm p={rca['auc_permutation_p']}) "
                  f"| {sweep}"
                  + ("  | NOTE a benign client outranks both attackers"
                     if rca.get("benign_outranks_attacker") else ""))
    results["objective_summary"] = _build_objective_summary(results)
    objective_path = out_dir / "objective_summary.json"
    objective_path.write_text(
        json.dumps(results["objective_summary"], indent=2), encoding="utf-8"
    )
    if resources is not None:
        resources["artifacts"]["objective_summary_json"] = str(objective_path)
    figure_paths = (
        _save_figure(results, out_dir / "figures")
        if cfg.get("save_figures", True) else []
    )
    fl_results_path = out_dir / "fl_results.json"
    if resources is not None:
        from .visualize import resource_digest
        resources["experiment_wall_seconds"] = float(
            time.perf_counter() - experiment_wall_started
        )
        resources["experiment_wall_scope"] = (
            "run_fl entry through figure generation; excludes final report/JSON writes"
        )
        resources["artifacts"]["figure_files"] = figure_paths
        resources["artifacts"]["fl_results_json"] = str(fl_results_path)
        report_path = out_dir / "resource_report.txt"
        resources["artifacts"]["resource_report_txt"] = str(report_path)
        resource_text = resource_digest(results, print_output=False)
        report_path.write_text(resource_text + "\n", encoding="utf-8")

        # Final rewrite makes the manifest an archive index, including figures and the
        # copy-paste report generated after the preliminary manifest was created.
        manifest.update({
            "experiment_wall_seconds": resources["experiment_wall_seconds"],
            "experiment_wall_scope": resources["experiment_wall_scope"],
            "comparison_profile": resources.get("comparison_profile"),
            "artifacts": resources["artifacts"],
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fl_results_path.write_text(json.dumps(results, indent=2))
    _print_summary(results)
    if resources is not None:
        print("\n" + resource_text)
    print(f"\n  Multi-round results written to {out_dir}/fl_results.json")
    LAST_RUN_COMPLETED_ID = results["run_id"]
    return results


def run_fl_seeds(config: Dict, seeds: List[int]) -> Dict:
    """Run the multi-round FL experiment across multiple seeds and report mean +/- std of the
    FINAL-round headline metrics, the stealth pass-rate, and the per-defense excess-detection.

    fl_runner is otherwise single-seed (n=1): the durability curve, the 6/10-vs-8/10 stealth
    counts, and the defense numbers cannot be told apart from run-to-run noise without this
    (two identical-config runs in the archive disagreed by up to ~5% on amp and by 2 rounds on
    stealth). Per-seed artifacts go to their own timestamped run folders; a
    multiseed_fl_summary.json is written under results/<subdir>_seeds/. Opt-in — it multiplies
    runtime by len(seeds), so it does not change the default single-run behaviour."""

    base_subdir = config.get("results_subdir", "tcaa_fl")
    # (seed, result) pairs, not two parallel lists: a failed seed would otherwise shift the
    # alignment and attribute every later run's topology/metrics to the wrong seed.
    completed: List[Tuple[int, Dict]] = []
    failures: List[Dict] = []
    for si, seed in enumerate(seeds):
        cfg = dict(config)
        cfg["seed"] = seed
        cfg["results_subdir"] = f"{base_subdir}/seed_{seed}"
        print(f"\n########## FL SEED {si + 1}/{len(seeds)}  (seed={seed}) ##########")
        try:
            completed.append((seed, run_fl(cfg)))
        except Exception as exc:  # a late OOM/timeout on seed k must not lose seeds 0..k-1
            print(f"  [!] seed {seed} FAILED: {type(exc).__name__}: {exc}")
            failures.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    if not completed:
        raise RuntimeError("run_fl_seeds: every seed failed; no summary to write")
    runs: List[Dict] = [r for _, r in completed]

    # TOPOLOGY HOMOGENEITY: the seed redraws the Dirichlet split, so different seeds can run with
    # different effective federation sizes (an under-floor shard is skipped). Averaging defense
    # rows across seeds that faced different n is not meaningful, so surface it rather than hide it.
    shard_profiles = [(s, r.get("shard_profile") or {}) for s, r in completed]
    empty_counts = {s: prof.get("empty_shards") for s, prof in shard_profiles}
    homogeneous = len({tuple(sorted((prof.get("sizes") or []))) for _, prof in shard_profiles}) <= 1
    any_empty = any(bool(v) for v in empty_counts.values())
    if any_empty:
        print(f"  [!] TOPOLOGY WARNING: some seeds had empty shards {empty_counts} — set "
              f"min_client_shard>0 to keep every seed at the intended n before trusting the roll-up.")

    def _final(r: Dict) -> Dict:
        return (r.get("durability") or [{}])[-1]

    def _stealth_rate(r: Dict) -> float:
        st = [s for s in r.get("stealth_trace", []) if s.get("n_attackers")]
        return (sum(1 for s in st if s["jointly_satisfied"]) / len(st)) if st else float("nan")

    metric_keys = [
        "amp_tau_effective", "amp_tau_calibrated", "amp_tau_vs_pristine", "amp_tau_median",
        "amp_tau", "selectivity", "tau_len_atk", "truncation_tau", "repetition_tau",
        "distinct_ratio_tau", "ppl_ratio_vs_pristine", "rouge_recall_clean_atk",
        "rouge_recall_tau_atk",
        # C2 needs its CONTROL across seeds too, not just the attacked arm: a mean+/-std of
        # rouge_recall_clean_atk alone cannot say whether the attacker moved clean utility, since
        # benign federated drift moves it as well. These are the benign-only counterfactual and
        # the LINEAR clean-length ratio the C2 gate is actually decided on.
        "ppl_ratio", "rouge_recall_clean_ben", "rouge_f1_clean_ben",
        "amp_clean_len", "amp_clean",
        # uncensored C1/C4 signal: raw tau_len saturates at the cap, effective length does not
        "tau_effective_len_atk",
        # Reasoning-TCAA final-round gates (None on legacy length runs).
        "reasoning_cost_ratio_tau", "reasoning_cost_ci95_lower_tau",
        "reasoning_cost_ci95_upper_tau", "reasoning_cost_ratio_clean",
        "reasoning_tokens_tau_atk", "reasoning_tokens_tau_ref",
        "reasoning_closure_tau", "answer_accuracy_tau_atk",
        "answer_accuracy_tau_ref", "answer_accuracy_clean_atk",
        "answer_accuracy_clean_ref", "answer_accuracy_drop_tau",
        "answer_accuracy_drop_clean", "reasoning_repetition_tau",
        "reasoning_distinct_tau", "reasoning_measurement_valid",
        "reasoning_claim_ready",
    ]
    collected: Dict[str, List[float]] = {
        k: [float(_final(r)[k]) for r in runs if _final(r).get(k) is not None]
        for k in metric_keys
    }
    collected["stealth_satisfied_rate"] = [_stealth_rate(r) for r in runs]

    # Per-defense excess-detection (atk_fpr - ben_fpr), the honest C3 number, across seeds.
    # Carry excess_structural so the printer never puts Krum's range-compressed excess on the
    # same numeric scale as the fixed-f defenses (the guard defenses.py itself applies).
    defense_excess: Dict[str, List[float]] = {}
    defense_structural: Dict[str, bool] = {}
    defense_purpose_built: Dict[str, bool] = {}
    # A cross-seed mean excess still says nothing about detectability on its own, so carry each
    # seed's permutation p and count how many seeds actually reached significance. "3 seeds, mean
    # excess +0.27, 0/3 significant" is a very different sentence from "mean excess +0.27".
    defense_p_cluster: Dict[str, List[float]] = {}
    defense_sig_seeds: Dict[str, List[bool]] = {}
    for r in runs:
        defs = (((r.get("defense_evaluation") or {}).get("telemetry_defenses") or {})
                .get("defenses") or {})
        for name, d in defs.items():
            defense_excess.setdefault(name, []).append(d.get("excess_detection"))
            defense_structural[name] = bool(d.get("excess_structural"))
            defense_purpose_built[name] = bool(d.get("purpose_built"))
            if d.get("excess_p_cluster") is not None:
                defense_p_cluster.setdefault(name, []).append(float(d["excess_p_cluster"]))
            if d.get("excess_significant") is not None:
                defense_sig_seeds.setdefault(name, []).append(bool(d["excess_significant"]))

    # C1 across seeds: the resource-amplification headline the roll-up previously dropped. Pull
    # attacked/pristine ratios from each run's objective_summary so C1 gets a mean +/- std too.
    resource_amp: Dict[str, List[float]] = {}
    for r in runs:
        ra = ((r.get("objective_summary") or {}).get("resource_amplification") or {})
        for metric, val in ra.items():
            v = (val or {}).get("attacked_vs_pristine") if isinstance(val, dict) else None
            if v is not None:
                resource_amp.setdefault(metric, []).append(float(v))

    # C3 star: the rank-order collusion detector's AUC per seed (recomputed offline from each
    # run's telemetry). Aggregated so the STAR pillar gets a cross-seed AUC, not a single point.
    rank_auc: List[float] = []
    rank_excess_at_fpr10: List[float] = []
    try:
        from .defenses import rank_collusion_analysis
        for r in runs:
            an = rank_collusion_analysis(r.get("defense_telemetry") or [])
            if an:
                rank_auc.append(an["auc"])
                hit = [s for s in an["fpr_sweep"] if abs(s["target_fpr"] - 0.10) < 1e-9]
                if hit:
                    rank_excess_at_fpr10.append(hit[0]["excess"])
    except Exception as exc:
        print(f"  [!] rank_collusion_analysis unavailable across seeds: {exc}")

    def ms(xs: List[float]) -> Dict:
        vals = [x for x in xs if x is not None and x == x]  # drop None / NaN
        if not vals:
            return {"mean": float("nan"), "std": 0.0, "values": []}
        m = statistics.mean(vals)
        # SAMPLE stdev (/(n-1)): the unbiased estimate of run-to-run SD this wrapper exists to
        # expose. pstdev (/n) would understate it by sqrt((n-1)/n) (~18% at 3 seeds).
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "values": [round(x, 4) for x in vals]}

    summary = {
        "seeds": list(seeds),                       # requested
        "seeds_completed": [s for s, _ in completed],  # actually aggregated (may be a subset)
        "run_ids": [r.get("run_id") for r in runs],
        # These arrays have the same completed-run ordering as seeds_completed/run_ids.
        # Keeping failed seeds out of all four arrays lets an archive consumer zip them
        # without accidentally assigning a later seed's artifacts or hashes to a failure.
        "artifacts_dirs": [r.get("artifacts_dir") for r in runs],
        "requested_config_sha256s": [
            r.get("requested_config_sha256") for r in runs
        ],
        "resolved_config_sha256s": [
            r.get("resolved_config_sha256") for r in runs
        ],
        "completed_runs": [
            {
                "seed": seed,
                "run_id": result.get("run_id"),
                "artifacts_dir": result.get("artifacts_dir"),
                "requested_config_sha256": result.get("requested_config_sha256"),
                "resolved_config_sha256": result.get("resolved_config_sha256"),
            }
            for seed, result in completed
        ],
        "n_completed": len(runs),
        "failures": failures,
        # Provenance so a reader can verify the seeds are comparable before trusting the means.
        "topology": {
            "homogeneous": homogeneous,
            "empty_shards_by_seed": {str(s): empty_counts[s] for s in empty_counts},
            "shard_sizes_by_seed": {str(s): (prof.get("sizes") or [])
                                    for s, prof in shard_profiles},
        },
        "final_round": {k: ms(v) for k, v in collected.items()},
        "defense_excess_detection": {
            name: {**ms(vals),
                   "excess_structural": defense_structural.get(name, False),
                   "purpose_built": defense_purpose_built.get(name, False),
                   "p_cluster_per_seed": defense_p_cluster.get(name, []),
                   "significant_seeds": sum(1 for b in defense_sig_seeds.get(name, []) if b),
                   "seeds_with_test": len(defense_sig_seeds.get(name, []))}
            for name, vals in defense_excess.items()},
        "resource_amplification_vs_pristine": {m: ms(v) for m, v in resource_amp.items()},
        "rank_collusion": {
            "auc": ms(rank_auc),
            "excess_at_benign_fpr_0.10": ms(rank_excess_at_fpr10),
        },
    }
    # Stamp the roll-up dir too, so re-running a sweep of the same base config never clobbers a
    # previous sweep's summary (run_paths no-clobber invariant; per-seed folders are already stamped).
    summary_cfg = stamp_run_subdir({
        "results_root": config.get("results_root", "results"),
        "results_subdir": f"{base_subdir}_seeds",
    })
    out_dir = resolve_run_dir(summary_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_fl_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 80}\nMULTI-SEED FL SUMMARY ({len(seeds)} seeds) — final-round mean +/- std"
          f"\n{'=' * 80}")
    for k in metric_keys + ["stealth_satisfied_rate"]:
        a = summary["final_round"][k]
        print(f"  {k:<26} {a['mean']:>10.4f} +/- {a['std']:<8.4f}  {a['values']}")
    if summary["defense_excess_detection"]:
        print("  -- defense excess_detection (atk_fpr - ben_fpr; ~0 = indistinguishable/stealthy) --")
        for name, a in summary["defense_excess_detection"].items():
            tag = " [struct*: read caught/survival, not excess]" if a.get("excess_structural") else ""
            if a.get("purpose_built"):
                tag += " [purpose-built: not evidence about STANDARD defenses]"
            # A mean excess without the significance count is exactly the number that reads as
            # detection when it is noise, so print "k/n seeds significant" beside every row.
            if a.get("seeds_with_test"):
                tag += f" [{a['significant_seeds']}/{a['seeds_with_test']} seeds sig]"
            print(f"  {name:<26} {a['mean']:>10.4f} +/- {a['std']:<8.4f}  {a['values']}{tag}")
    rc = summary["rank_collusion"]["auc"]
    if rc["values"]:
        e10 = summary["rank_collusion"]["excess_at_benign_fpr_0.10"]
        print("  -- rank-order collusion detector (C3 star) --")
        print(f"  {'AUC':<26} {rc['mean']:>10.4f} +/- {rc['std']:<8.4f}  {rc['values']}")
        print(f"  {'excess@benign_fpr=0.10':<26} {e10['mean']:>10.4f} +/- {e10['std']:<8.4f}  "
              f"{e10['values']}  (deployable-FPR operating point)")
    if summary["resource_amplification_vs_pristine"]:
        print("  -- C1 resource amplification vs pristine (mean +/- std over seeds) --")
        for m in ("total_output_tokens", "decode_attention_pairs", "energy_joules",
                  "generation_wall_seconds"):
            a = summary["resource_amplification_vs_pristine"].get(m)
            if a and a["values"]:
                print(f"  {m:<26} {a['mean']:>10.4f} +/- {a['std']:<8.4f}  {a['values']}")
    if not summary["topology"]["homogeneous"]:
        print("  [!] seeds ran with DIFFERENT shard splits — see summary['topology'] before "
              "quoting cross-seed defense means.")
    print("=" * 80)
    return summary


def _print_summary(r: Dict) -> None:
    dur, st = r["durability"], r["stealth_trace"]
    print(f"\n{'='*80}\nTCAA MULTI-ROUND SUMMARY ({r['config']['num_rounds']} rounds, "
          f"{r['config']['num_clients']}={r['config']['num_clients']-r['config']['num_attackers']}"
          f"+{r['config']['num_attackers']})\n{'='*80}")
    # HONEST HEADLINE ORDER: amp_eff (degeneracy-discounted), amp_cal (calibrated coeffs) and
    # amp_pri (vs fixed pristine) are the claim-bearing columns; amp (raw) and amp_dec are
    # caveated — amp credits the quadratic term from token 1, and amp_dec is shown as '—' when
    # it is assumption-only (truncated tail hit the cap => no data, decensored_valid=False).
    pri = r.get("pristine_reference", {})
    if pri:
        print(f"  pristine (round-0) ref: ppl_clean={pri.get('ppl_clean')} "
              f"ROUGE_clean={pri.get('rouge_recall_clean')} ROUGE_tau={pri.get('rouge_recall_tau')} "
              f"tau_len={pri.get('tau_mean_len')}")
    if r.get("cost_c_f_calibrated"):
        # In reasoning mode the calibrated coefficients are ALSO the ones every cost is stored
        # with, so amp and amp_cal are the same quantity and the old legend ("amp uses naive
        # c_f=c_a=1") was false by ~28% at Qwen3-1.7B scale. Say which regime is in force.
        _cfg = r.get("config") or {}
        _cal_is_default = (_cfg.get("attack_objective") == "reasoning_cost"
                           and _cfg.get("reasoning_use_calibrated_cost", True))
        _note = ("(ALL cost columns already use these coefficients; amp == amp_cal by construction)"
                 if _cal_is_default else
                 "(honest compute-cost multiple; amp uses naive c_f=c_a=1)")
        print(f"  amp_cal = calibrated coeffs c_f={r['cost_c_f_calibrated']:.0f}, c_a={r.get('cost_c_a_calibrated')}"
              f"  {_note}")
    print(f"  {'round':>5} {'amp_eff':>8} {'amp_cal':>8} {'amp_pri':>8} {'amp_med':>8} {'amp':>6} {'amp_dec':>8} "
          f"{'tau_len':>8} {'trunc':>6} {'rep':>5} {'dist':>5} {'ppl_pri':>7} {'R_cln':>6} {'R_tau':>6} {'stealth':>8}")
    for p in dur:
        _cal = p.get("amp_tau_calibrated")
        cal_cell = f"{_cal:>8.2f}" if _cal is not None else f"{'—':>8}"
        dec_cell = (f"{p.get('amp_tau_decensored', float('nan')):>8.2f}"
                    if p.get("decensored_valid", True) else f"{'—':>8}")
        print(f"  {p['round']:>5} {p.get('amp_tau_effective', float('nan')):>8.2f} {cal_cell} "
              f"{p.get('amp_tau_vs_pristine', float('nan')):>8.2f} {p.get('amp_tau_median', float('nan')):>8.2f} "
              f"{p['amp_tau']:>6.2f} {dec_cell} "
              f"{p['tau_len_atk']:>8.1f} "
              f"{p['truncation_tau']:>6.2f} {p['repetition_tau']:>5.2f} {p.get('distinct_ratio_tau', float('nan')):>5.2f} "
              f"{p.get('ppl_ratio_vs_pristine', float('nan')):>7.3f} "
              f"{p.get('rouge_recall_clean_atk', float('nan')):>6.2f} "
              f"{p.get('rouge_recall_tau_atk', float('nan')):>6.2f} {str(p['stealth_ok']):>8}")
    rounds_with_atk = [s for s in st if s.get("n_attackers")]
    ok = sum(1 for s in rounds_with_atk if s["jointly_satisfied"])
    if dur:
        first, last = dur[0], dur[-1]
        print("  " + "-" * 76)
        # Headline durability on the HONEST metrics: vs-pristine (fixed anchor) and effective
        # (degeneracy-discounted). Narrate as rapidly-saturating-and-durable, not accumulating.
        _cal_note = (f", calibrated {last.get('amp_tau_calibrated'):.2f}x"
                     if last.get("amp_tau_calibrated") is not None else "")
        _dec_note = (f"; raw de-censored {last.get('amp_tau_decensored', float('nan')):.2f}x (assumption-only)"
                     if not last.get("decensored_valid", True)
                     else f"; raw de-censored {last.get('amp_tau_decensored', float('nan')):.2f}x")
        print(f"  durability (vs pristine): {first.get('amp_tau_vs_pristine', float('nan')):.2f}x -> "
              f"{last.get('amp_tau_vs_pristine', float('nan')):.2f}x  |  effective "
              f"{first.get('amp_tau_effective', float('nan')):.2f}x -> {last.get('amp_tau_effective', float('nan')):.2f}x"
              f"{_cal_note}  [round {first['round']}->{last['round']}]")
        print(f"    (footnote: raw amp_tau {first['amp_tau']:.2f}x -> {last['amp_tau']:.2f}x{_dec_note})")
        if first.get("ppl_ratio_vs_pristine") is not None and last.get("ppl_ratio_vs_pristine") is not None:
            print(f"  utility (vs pristine): ppl {first['ppl_ratio_vs_pristine']:.3f} -> "
                  f"{last['ppl_ratio_vs_pristine']:.3f}  |  clean ROUGE-L recall "
                  f"{first.get('rouge_recall_clean_atk')} -> {last.get('rouge_recall_clean_atk')} "
                  f"(pristine {pri.get('rouge_recall_clean')}); "
                  f"tau ROUGE {first.get('rouge_recall_tau_atk')} -> {last.get('rouge_recall_tau_atk')} "
                  f"(~pristine = answer kept)")
    if rounds_with_atk:
        print(f"  stealth: jointly satisfied in {ok}/{len(rounds_with_atk)} attacker-participating rounds")
    print("=" * 80)


def _save_figure(r: Dict, out_dir: Path) -> List[str]:
    """Best-effort: save the FL figures (durability / utility / stealth / defense geometry)
    to disk. Delegates to tcaa.visualize (single-axis stacked subplots; NO dual-y) so the
    600-DPI PNGs and vector PDFs match what the notebook renders inline. Skips gracefully
    without matplotlib.

    Does NOT switch the matplotlib backend: fig.savefig works under any backend, and forcing
    Agg here would clobber a notebook's inline backend, silently killing later render_*_report
    inline display. Figures are closed after saving, so run_fl never spams the notebook."""
    if not r.get("durability"):
        return []
    try:
        from .visualize import make_fl_figures, save_figure
    except Exception as e:  # pragma: no cover
        print(f"  [fl] skipped figure ({e})")
        return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    all_written: List[str] = []
    for key, fig in make_fl_figures(r):
        p = out_dir / f"{key}.png"
        written = save_figure(fig, p)
        all_written.extend(str(path) for path in written)
        plt.close(fig)
        print(f"  [fl] saved {', '.join(str(path) for path in written)}")
    return all_written


def _parse_args():
    p = argparse.ArgumentParser(description="TCAA multi-round FL runner")
    p.add_argument("--smoke", action="store_true", help="CPU tiny-GPT2 + synthetic data")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--num-attackers", type=int, default=None)
    p.add_argument("--clients-per-round", type=int, default=None)
    p.add_argument("--attacker-always-selected", action="store_true")
    p.add_argument("--seeds", type=int, default=None,
                   help="run N seeds (mean+/-std of final-round metrics + defense excess); n=1 if unset")
    p.add_argument("--config-json", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg: Dict = {}
    if args.smoke:
        cfg.update(fl_smoke_overrides())
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text()))
    for attr, key in (("backbone", "backbone"), ("source", "source"),
                      ("num_rounds", "num_rounds"), ("num_clients", "num_clients"),
                      ("num_attackers", "num_attackers"), ("clients_per_round", "clients_per_round")):
        v = getattr(args, attr)
        if v is not None:
            cfg[key] = v
    if args.attacker_always_selected:
        cfg["attacker_always_selected"] = True
    if args.seeds:
        base = cfg.get("seed", default_fl_config()["seed"])
        run_fl_seeds(cfg, [base + 1000 * i for i in range(args.seeds)])
    else:
        run_fl(cfg)


if __name__ == "__main__":
    main()
