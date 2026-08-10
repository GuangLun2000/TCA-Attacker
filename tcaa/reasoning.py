"""Reasoning-span parsing, differentiable cost primitives, and task verifiers.

Reasoning-TCAA treats the tokens generated through the first ``</think>`` marker as
the resource-bearing phase and the remaining tokens as the externally visible answer.
The functions in this module are intentionally model- and runner-independent so their
math can be unit-tested without downloading a reasoning model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


_EPS = 1e-6
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_LATEX_FRAC_RE = re.compile(
    r"\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}"
)
_PLAIN_FRAC_RE = re.compile(r"([-+]?\d+)\s*/\s*([-+]?\d+)")
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:：]", re.IGNORECASE)


@dataclass(frozen=True)
class ReasoningSegments:
    """Token-level split of one generated continuation.

    ``reasoning_ids`` excludes explicit start/end markers. ``reasoning_decode_len``
    counts every generated token through the end marker because decoding the marker
    consumes compute too. If the marker is absent, the full observed continuation is
    an unclosed, right-censored reasoning span.
    """

    reasoning_ids: List[int]
    answer_ids: List[int]
    closed: bool
    reasoning_decode_len: int
    start_found: bool = False


def find_subsequence(sequence: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    """Return the first index of ``needle`` in ``sequence`` or ``-1`` when absent."""
    if not needle:
        raise ValueError("needle must contain at least one token")
    stop = len(sequence) - len(needle) + 1
    for idx in range(max(int(start), 0), max(stop, 0)):
        if list(sequence[idx:idx + len(needle)]) == list(needle):
            return idx
    return -1


def split_reasoning_tokens(
    token_ids: Sequence[int],
    *,
    end_ids: Sequence[int],
    start_ids: Optional[Sequence[int]] = None,
) -> ReasoningSegments:
    """Split a continuation at its first reasoning-end marker.

    Reasoning chat templates often place ``<think>`` in the prompt, so a missing start
    marker is valid and means the continuation begins directly with reasoning tokens.
    The end marker is mandatory for a closed record.
    """
    ids = list(token_ids)
    if not end_ids:
        raise ValueError("end_ids must contain at least one token")

    reasoning_start = 0
    start_found = False
    if start_ids:
        start_idx = find_subsequence(ids, start_ids)
        if start_idx >= 0:
            reasoning_start = start_idx + len(start_ids)
            start_found = True

    end_idx = find_subsequence(ids, end_ids, start=reasoning_start)
    if end_idx < 0:
        return ReasoningSegments(
            reasoning_ids=ids[reasoning_start:],
            answer_ids=[],
            closed=False,
            reasoning_decode_len=len(ids),
            start_found=start_found,
        )
    return ReasoningSegments(
        reasoning_ids=ids[reasoning_start:end_idx],
        answer_ids=ids[end_idx + len(end_ids):],
        closed=True,
        reasoning_decode_len=end_idx + len(end_ids),
        start_found=start_found,
    )


def marginal_decode_cost(
    prompt_lengths: torch.Tensor,
    steps: torch.Tensor,
    *,
    c_f: float,
    c_a: float,
) -> torch.Tensor:
    """Per-token decode cost ``c_f + c_a * (n + t - 1)`` for 1-indexed ``t``."""
    return float(c_f) + float(c_a) * (prompt_lengths + steps - 1.0)


def expected_survival_cost(
    stop_logprob: torch.Tensor,
    target_mask: torch.Tensor,
    prompt_lengths: Union[torch.Tensor, Sequence[int], float, int],
    *,
    c_f: float,
    c_a: float,
) -> torch.Tensor:
    """Expected reasoning-phase decode cost under a stop-marker hazard.

    For valid ordered positions, ``q_s = p(stop | prefix_s)`` and

    ``E[C_R] = sum_t prod_{s<t}(1-q_s) * [c_f + c_a(n+t-1)]``.

    Masked positions neither advance the survival product nor receive cost. The return
    value is per sample, shape ``[B]``.
    """
    if stop_logprob.shape != target_mask.shape or stop_logprob.ndim != 2:
        raise ValueError("stop_logprob and target_mask must be aligned rank-2 tensors")
    mask = target_mask.to(stop_logprob.dtype)
    q = stop_logprob.exp()
    log_1mq = torch.log1p(-q.clamp(max=1.0 - _EPS)) * mask
    inclusive = torch.cumsum(log_1mq, dim=1)
    survival = torch.exp(inclusive - log_1mq)

    prompt = torch.as_tensor(
        prompt_lengths, dtype=stop_logprob.dtype, device=stop_logprob.device
    )
    if prompt.ndim == 0:
        prompt = prompt.repeat(stop_logprob.shape[0])
    if prompt.ndim != 1 or prompt.numel() != stop_logprob.shape[0]:
        raise ValueError("prompt_lengths must be scalar or contain one value per sample")
    steps = torch.cumsum(mask, dim=1)
    weights = marginal_decode_cost(
        prompt.unsqueeze(1), steps, c_f=float(c_f), c_a=float(c_a)
    )
    return (survival * weights * mask).sum(dim=1)


def realized_reasoning_cost(
    prompt_len: int,
    reasoning_decode_len: int,
    *,
    c_f: float,
    c_a: float,
) -> float:
    """Closed-form realized cost through the end marker (or observed open span)."""
    n = float(prompt_len)
    length = float(reasoning_decode_len)
    return float(
        (float(c_f) + float(c_a) * n) * length
        + 0.5 * float(c_a) * length * (length - 1.0)
    )


def select_reasoning_anchors(
    candidate_clean: Sequence[Any],
    candidate_tau: Sequence[Any],
    clean_reference_costs: Union[torch.Tensor, Sequence[float]],
    tau_reference_costs: Union[torch.Tensor, Sequence[float]],
    clean_diagnostics: Sequence[Mapping[str, Any]],
    tau_diagnostics: Sequence[Mapping[str, Any]],
    *,
    requested_size: int,
    minimum_valid: int,
    reference_horizon: int,
    training_horizon: int,
    target_ratio: float,
    c_f: float,
    c_a: float,
    require_correct: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select row-aligned, closed, correct, and attainable training anchors.

    This helper is deliberately model-free: generation happens before it is called,
    and every filtering/indexing decision is therefore unit-testable without loading
    a language model.  The returned indices are the sole source of truth used to
    slice clean/tau examples and both reference-cost tensors.
    """

    collections = {
        "candidate_clean": candidate_clean,
        "candidate_tau": candidate_tau,
        "clean_reference_costs": clean_reference_costs,
        "tau_reference_costs": tau_reference_costs,
        "clean_diagnostics": clean_diagnostics,
        "tau_diagnostics": tau_diagnostics,
    }
    try:
        lengths = {name: len(values) for name, values in collections.items()}
    except TypeError as exc:
        raise ValueError("reasoning anchor inputs must be sized sequences") from exc
    candidate_count = lengths["candidate_tau"]
    if candidate_count <= 0 or any(length != candidate_count for length in lengths.values()):
        raise ValueError(
            "reasoning anchor inputs must be non-empty and exactly row-aligned; "
            f"lengths={lengths}"
        )

    for name, value in (
        ("requested_size", requested_size),
        ("minimum_valid", minimum_valid),
        ("reference_horizon", reference_horizon),
        ("training_horizon", training_horizon),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if requested_size > candidate_count:
        raise ValueError("requested_size cannot exceed the candidate count")
    if minimum_valid > requested_size:
        raise ValueError("minimum_valid cannot exceed requested_size")
    if reference_horizon > training_horizon:
        raise ValueError("reference_horizon cannot exceed training_horizon")
    if not isinstance(require_correct, bool):
        raise ValueError("require_correct must be boolean")
    for name, value, lower in (
        ("target_ratio", target_ratio, 1.0),
        ("c_f", c_f, 0.0),
        ("c_a", c_a, 0.0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < lower
        ):
            raise ValueError(f"{name} must be finite and >= {lower}")
    if float(target_ratio) <= 1.0:
        raise ValueError("target_ratio must be finite and > 1")
    if float(c_f) == 0.0 and float(c_a) == 0.0:
        raise ValueError("at least one of c_f/c_a must be positive")

    def reference_array(values, name: str) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size != candidate_count:
            raise ValueError(f"{name} must be one-dimensional and row-aligned")
        if not np.isfinite(array).all() or np.any(array <= 0.0):
            raise ValueError(f"{name} must contain finite positive costs")
        return array

    reference_array(clean_reference_costs, "clean_reference_costs")
    tau_reference = reference_array(tau_reference_costs, "tau_reference_costs")
    # Both arrays are validated even though only tau defines the attack target.  The
    # clean array is sliced by the same indices downstream and must never drift.

    required_diagnostic_keys = {"reasoning_closed", "hit_cap", "time_limited"}
    rejection_counts: Dict[str, int] = {}
    valid_indices: List[int] = []
    for idx in range(candidate_count):
        clean_example = candidate_clean[idx]
        tau_example = candidate_tau[idx]
        clean_uid = getattr(clean_example, "uid", None)
        tau_uid = getattr(tau_example, "uid", None)
        if (clean_uid is not None or tau_uid is not None) and clean_uid != tau_uid:
            raise ValueError(
                "reasoning clean/tau anchor rows must be UID-paired; "
                f"row={idx}, clean_uid={clean_uid!r}, tau_uid={tau_uid!r}"
            )
        prompt_ids = getattr(tau_example, "prompt_ids", None)
        if prompt_ids is None or len(prompt_ids) <= 0:
            raise ValueError(f"tau anchor row {idx} must contain prompt_ids")

        clean_diag = clean_diagnostics[idx]
        tau_diag = tau_diagnostics[idx]
        if not isinstance(clean_diag, Mapping) or not isinstance(tau_diag, Mapping):
            raise ValueError("reasoning anchor diagnostics must be mappings")
        for label, diagnostic in (("clean", clean_diag), ("tau", tau_diag)):
            missing = sorted(required_diagnostic_keys.difference(diagnostic))
            if missing:
                raise ValueError(
                    f"{label} reasoning diagnostic row {idx} is missing {missing}"
                )

        reasons: List[str] = []
        if not bool(tau_diag["reasoning_closed"]):
            reasons.append("tau_not_closed")
        if not bool(clean_diag["reasoning_closed"]):
            reasons.append("clean_not_closed")
        if bool(tau_diag["hit_cap"]) or bool(clean_diag["hit_cap"]):
            reasons.append("cap_hit")
        if bool(tau_diag["time_limited"]) or bool(clean_diag["time_limited"]):
            reasons.append("time_limited")
        if require_correct and tau_diag.get("answer_correct") is not True:
            reasons.append("tau_answer_incorrect")
        if require_correct and clean_diag.get("answer_correct") is not True:
            reasons.append("clean_answer_incorrect")

        maximum_cost = realized_reasoning_cost(
            len(prompt_ids), int(training_horizon), c_f=float(c_f), c_a=float(c_a)
        )
        requested_cost = float(target_ratio) * float(tau_reference[idx])
        if requested_cost > maximum_cost * (1.0 + 1e-6):
            reasons.append("target_unattainable")

        if reasons:
            for reason in dict.fromkeys(reasons):
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        else:
            valid_indices.append(idx)

    if len(valid_indices) < minimum_valid:
        raise ValueError(
            "reasoning anchor preflight retained only "
            f"{len(valid_indices)}/{candidate_count} candidates, below "
            f"reasoning_min_valid_anchors={minimum_valid}; exclusions={rejection_counts}. "
            "Use a larger TRAIN candidate pool/multiplier, improve baseline correctness, "
            "raise reasoning_horizon, lower reasoning_target_ratio, or raise the finite "
            "reference horizon. Never tune these settings on the test split."
        )

    selected_indices = valid_indices[:requested_size]
    preflight = {
        "candidate_count": candidate_count,
        "valid_count": len(valid_indices),
        "selected_count": len(selected_indices),
        "valid_fraction": round(len(valid_indices) / candidate_count, 4),
        "minimum_required": minimum_valid,
        "reference_horizon": int(reference_horizon),
        "training_horizon": int(training_horizon),
        "target_ratio": float(target_ratio),
        "require_correct": require_correct,
        "exclusions": rejection_counts,
        "selected_uids": [
            getattr(candidate_tau[idx], "uid", None) for idx in selected_indices
        ],
    }
    return selected_indices, preflight


def reasoning_effect_gates(ci95_lower: float, minimum_ratio: float) -> Dict[str, bool]:
    """Return distinct significance and minimum-effect gates for a cost ratio."""

    for name, value in (("ci95_lower", ci95_lower), ("minimum_ratio", minimum_ratio)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite number")
    if float(ci95_lower) <= 0.0:
        raise ValueError("ci95_lower must be strictly positive")
    if float(minimum_ratio) <= 1.0:
        raise ValueError("minimum_ratio must be > 1")
    return {
        "cost_ci_lower_gt_1": float(ci95_lower) > 1.0,
        "cost_ci_lower_meets_effect_size": (
            float(ci95_lower) >= float(minimum_ratio)
        ),
    }


def target_ratio_hinge(
    expected_cost: torch.Tensor,
    reference_cost: Union[torch.Tensor, Sequence[float], float],
    *,
    target_ratio: float,
    eps: float = _EPS,
) -> torch.Tensor:
    """Squared normalized shortfall; exactly zero once the target ratio is reached."""
    if not isinstance(target_ratio, (int, float)) or isinstance(target_ratio, bool):
        raise ValueError("target_ratio must be a finite number >= 1")
    ratio = float(target_ratio)
    if not torch.isfinite(torch.tensor(ratio)) or ratio < 1.0:
        raise ValueError("target_ratio must be a finite number >= 1")
    ref = torch.as_tensor(
        reference_cost, dtype=expected_cost.dtype, device=expected_cost.device
    )
    target = ratio * ref.clamp(min=float(eps))
    return torch.square(torch.relu(target - expected_cost) / target)


def target_ratio_band_loss(
    expected_cost: torch.Tensor,
    reference_cost: Union[torch.Tensor, Sequence[float], float],
    *,
    target_ratio: float,
    target_tolerance: float = 0.25,
    overshoot_weight: float = 1.0,
    eps: float = _EPS,
) -> torch.Tensor:
    """Keep expected cost inside a finite target band.

    The lower hinge drives overthinking until ``target_ratio`` is reached. The upper
    hinge restores stop-marker pressure beyond ``target_ratio + target_tolerance`` so
    the optimizer has an explicit reason to close ``</think>`` instead of drifting to
    the generation cap.
    """
    if target_tolerance < 0 or not torch.isfinite(torch.tensor(float(target_tolerance))):
        raise ValueError("target_tolerance must be finite and >= 0")
    if overshoot_weight < 0 or not torch.isfinite(torch.tensor(float(overshoot_weight))):
        raise ValueError("overshoot_weight must be finite and >= 0")
    lower = target_ratio_hinge(
        expected_cost, reference_cost, target_ratio=target_ratio, eps=eps
    )
    ref = torch.as_tensor(
        reference_cost, dtype=expected_cost.dtype, device=expected_cost.device
    ).clamp(min=float(eps))
    upper = (float(target_ratio) + float(target_tolerance)) * ref
    overshoot = torch.square(torch.relu(expected_cost - upper) / upper)
    return lower + float(overshoot_weight) * overshoot


def extract_gsm8k_gold(answer: str) -> Tuple[str, str]:
    """Return ``(rationale, final_answer)`` from the official ``####`` format."""
    text = str(answer or "").strip()
    if "####" not in text:
        raise ValueError("GSM8K answer does not contain the required '####' delimiter")
    rationale, final = text.rsplit("####", 1)
    final = final.strip()
    if not final:
        raise ValueError("GSM8K final answer is empty")
    return rationale.strip(), final


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def normalize_numeric_answer(value: str) -> Optional[str]:
    """Canonicalize a standalone GSM8K-style number, including simple fractions."""
    text = str(value or "").strip().replace("$", "").replace("£", "")
    text = text.replace("%", "").replace(",", "")
    frac = _LATEX_FRAC_RE.fullmatch(text)
    if frac is None:
        frac = _PLAIN_FRAC_RE.fullmatch(text)
    if frac:
        denominator = int(frac.group(2))
        if denominator == 0:
            return None
        return _canonical_decimal(
            Decimal(Fraction(int(frac.group(1)), denominator).numerator)
            / Decimal(Fraction(int(frac.group(1)), denominator).denominator)
        )
    try:
        return _canonical_decimal(Decimal(text))
    except InvalidOperation:
        return None


def extract_final_numeric_answer(text: str) -> Optional[str]:
    """Extract the last numeric answer, respecting an explicit final-answer suffix.

    Fractions and ordinary numbers compete by textual position. This matters for
    reasoning such as ``"an intermediate ratio is 1/2. Final answer: 5"``:
    globally preferring any fraction would incorrectly grade the answer as 0.5.
    """
    raw = str(text or "")
    markers = list(_FINAL_ANSWER_RE.finditer(raw))
    if markers:
        raw = raw[markers[-1].end():]

    fractions = list(_LATEX_FRAC_RE.finditer(raw))
    fractions.extend(_PLAIN_FRAC_RE.finditer(raw))
    fraction_spans = [match.span() for match in fractions]
    candidates = [(match.start(), match.group(0)) for match in fractions]
    for match in _NUMBER_RE.finditer(raw):
        start, end = match.span()
        if any(frac_start <= start and end <= frac_end
               for frac_start, frac_end in fraction_spans):
            continue
        candidates.append((start, match.group(0)))
    if not candidates:
        return None
    _, value = max(candidates, key=lambda item: item[0])
    return normalize_numeric_answer(value)


def verify_gsm8k(answer_text: str, gold_answer: str) -> bool:
    """Exact numeric correctness after conservative normalization."""
    predicted = extract_final_numeric_answer(answer_text)
    gold = normalize_numeric_answer(gold_answer)
    return predicted is not None and gold is not None and predicted == gold


def paired_log_ratio_summary(
    attacked: Sequence[float],
    reference: Sequence[float],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    """Robust paired ratio with a percentile bootstrap confidence interval."""
    atk = np.asarray(attacked, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if atk.ndim != 1 or ref.ndim != 1 or atk.size == 0 or atk.shape != ref.shape:
        raise ValueError("paired ratio inputs must be non-empty aligned vectors")
    if not np.isfinite(atk).all() or not np.isfinite(ref).all() \
            or np.any(atk <= 0) or np.any(ref <= 0):
        raise ValueError("paired ratio inputs must be finite and strictly positive")
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    log_ratios = np.log(atk) - np.log(ref)
    median_log = float(np.median(log_ratios))
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, atk.size, size=(int(bootstrap_samples), atk.size))
    boot = np.median(log_ratios[draws], axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "n_pairs": int(atk.size),
        "median_ratio": float(np.exp(median_log)),
        "ci95_lower": float(np.exp(lower)),
        "ci95_upper": float(np.exp(upper)),
        "mean_log_ratio": float(np.mean(log_ratios)),
    }
