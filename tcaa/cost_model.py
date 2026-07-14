# tcaa/cost_model.py
# Inference cost model and measurement (Spec Section 4).
#
# Per-request cost for a KV-cached decoder with prompt length n and output length L:
#
#     C = (c_f + c_a * n) * L  +  (c_a / 2) * L * (L - 1)
#         \___ linear in L ___/    \__ quadratic in L (self-attn over generated) __/
#
# We log BOTH channels the spec calls out:
#   - compute channel: the cost C above (and, on real HW, latency/energy hooks);
#   - memory channel:  a peak-KV proxy proportional to (n + L), which on edge
#                      devices often dominates before compute becomes quadratic.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# Default coefficients. c_f = per-token fixed FLOPs-ish constant (feed-forward),
# c_a = per-token attention constant. Only their *ratio* matters for the
# super-linear threshold; absolute scale cancels in the amplification ratio.
DEFAULT_C_F = 1.0
DEFAULT_C_A = 1.0

# De-censoring: cap on the extrapolated residual length added to a truncated (cap-hit)
# output. A truncated output's true length is unknown and > the generation cap, so the
# measured length/cost is a censored LOWER bound. Under a constant-hazard (exponential)
# tail, the expected residual is 1/lambda tokens (memoryless); we bound it at this value
# so a near-zero EOS hazard cannot extrapolate to an unbounded length. It is a horizon
# assumption (roughly "how far the model would run if left unconstrained"), reported
# alongside the estimate so the reader can judge it.
DEFAULT_DECENSOR_MAX_EXTRA = 4096.0


@dataclass(frozen=True)
class KVCacheGeometry:
    """Architecture geometry needed for a theoretical KV-cache byte estimate.

    The estimate intentionally covers only the K and V tensors themselves.  It does
    not claim to include allocator rounding, cache metadata, attention workspaces, or
    the model weights.  ``bytes_per_element`` should be the cache dtype width (usually
    the model's compute dtype), which can differ from quantized weight storage.
    """

    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        for name in (
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "bytes_per_element",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def bytes_per_token(self) -> int:
        """Bytes for K+V for one sequence token: ``2 * layers * kv_heads * head_dim``."""
        return int(
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * self.bytes_per_element
        )

    def bytes_for(self, sequence_length: int, *, batch_size: int = 1) -> int:
        """Theoretical KV tensor bytes for a rectangular batch cache."""
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise ValueError("sequence_length must be a non-negative integer")
        if sequence_length < 0:
            raise ValueError("sequence_length must be a non-negative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return int(sequence_length * batch_size * self.bytes_per_token)


def _dtype_bytes(dtype: Any) -> Optional[int]:
    """Return a torch dtype width without allocating model-sized storage."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        name = dtype.lower().replace("torch.", "")
        dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return int(torch.empty((), dtype=dtype).element_size())
    return None


def infer_kv_cache_geometry(
    model_or_config: Any = None,
    *,
    num_hidden_layers: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    bytes_per_element: Optional[int] = None,
    dtype: Any = None,
) -> Optional[KVCacheGeometry]:
    """Infer KV geometry from a HF model/config, with explicit values taking priority.

    GQA/MQA models use ``num_key_value_heads`` when present; ordinary MHA models fall
    back to ``num_attention_heads``.  ``None`` is returned when the supplied object does
    not expose enough architecture or dtype information for an honest byte estimate.
    Callers can always provide an explicit :class:`KVCacheGeometry` instead.
    """
    obj = model_or_config
    config = getattr(obj, "config", obj)

    def first_attr(*names: str) -> Any:
        if config is None:
            return None
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
        return None

    layers = num_hidden_layers
    if layers is None:
        layers = first_attr("num_hidden_layers", "n_layer", "num_layers")

    attention_heads = first_attr("num_attention_heads", "n_head")
    kv_heads = num_key_value_heads
    if kv_heads is None:
        kv_heads = first_attr("num_key_value_heads")
    if kv_heads is None:
        kv_heads = attention_heads

    resolved_head_dim = head_dim
    if resolved_head_dim is None:
        resolved_head_dim = first_attr("head_dim")
    if resolved_head_dim is None:
        hidden_size = first_attr("hidden_size", "n_embd", "d_model")
        if hidden_size is not None and attention_heads is not None:
            try:
                hidden = int(hidden_size)
                heads = int(attention_heads)
                if heads > 0 and hidden % heads == 0:
                    resolved_head_dim = hidden // heads
            except (TypeError, ValueError):
                resolved_head_dim = None

    element_bytes = bytes_per_element
    if element_bytes is None:
        element_bytes = _dtype_bytes(dtype)
    if element_bytes is None:
        # A config dtype is more representative of the KV compute dtype than quantized
        # parameter storage.  Fall back to model.dtype/parameter dtype only when absent.
        element_bytes = _dtype_bytes(first_attr("torch_dtype", "dtype"))
    if element_bytes is None and obj is not None:
        element_bytes = _dtype_bytes(getattr(obj, "dtype", None))
    if element_bytes is None and hasattr(obj, "parameters"):
        try:
            element_bytes = int(next(obj.parameters()).element_size())
        except (StopIteration, TypeError, AttributeError):
            element_bytes = None

    if any(v is None for v in (layers, kv_heads, resolved_head_dim, element_bytes)):
        return None
    try:
        return KVCacheGeometry(
            num_hidden_layers=int(layers),
            num_key_value_heads=int(kv_heads),
            head_dim=int(resolved_head_dim),
            bytes_per_element=int(element_bytes),
        )
    except (TypeError, ValueError):
        return None


def theoretical_kv_cache_bytes(
    sequence_length: int,
    *,
    batch_size: int = 1,
    geometry: Optional[KVCacheGeometry] = None,
    model_or_config: Any = None,
    **geometry_overrides: Any,
) -> Optional[int]:
    """Return theoretical K+V tensor bytes, or ``None`` when geometry is unavailable."""
    resolved = geometry or infer_kv_cache_geometry(model_or_config, **geometry_overrides)
    return None if resolved is None else resolved.bytes_for(sequence_length, batch_size=batch_size)


def inference_cost(n: float, L: float, c_f: float = DEFAULT_C_F, c_a: float = DEFAULT_C_A) -> float:
    """Scalar per-request cost C (Spec Section 4)."""
    return (c_f + c_a * n) * L + 0.5 * c_a * L * (L - 1.0)


def peak_kv_proxy(n: float, L: float) -> float:
    """Peak KV-cache footprint proxy, proportional to total sequence length (n + L)."""
    return float(n + L)


def superlinear_threshold(n: float, c_f: float = DEFAULT_C_F, c_a: float = DEFAULT_C_A) -> float:
    """Output length beyond which the quadratic term dominates: L >~ 2n + 2 c_f / c_a."""
    return 2.0 * n + 2.0 * c_f / max(c_a, 1e-12)


def calibrate_coefficients(d_model: int, n_layer: int = 1) -> Dict[str, float]:
    """
    Physically-motivated (c_f, c_a) for a real transformer decoder, so the
    super-linear threshold is not an arbitrary artifact of c_f = c_a = 1.

    Per generated token: the feed-forward / projection work is ~constant in position
    (proportional to d_model per unit, absorbing the layer count), while the attention
    work grows with the current context length (proportional to 1 per unit position).
    Hence c_a / c_f ~ 1 / d_model, and the quadratic (KV/attention) term only dominates
    once L ~ O(d_model). We return c_f = d_model, c_a = 1 (only their ratio matters;
    the absolute scale cancels in the amplification ratio).

    Example: Qwen2.5-0.5B has d_model = 896, so the super-linear onset is L ~ 2n + 1792
    — far longer than a short answer, which is exactly why a length attack must push
    outputs long before the *compute* channel (as opposed to the KV *memory* channel,
    which grows linearly from token 1) turns quadratic.
    """
    d = max(int(d_model), 1)
    return {"c_f": float(d), "c_a": 1.0}


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m]))


def _percentile(xs: List[float], q: float) -> float:
    """Linearly interpolated percentile (the common ``(N - 1) * q`` definition)."""
    if not xs:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    s = sorted(float(x) for x in xs)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    weight = pos - lo
    return float(s[lo] * (1.0 - weight) + s[hi] * weight)


def repetition_rate(tokens: List[int], n: int = 4) -> float:
    """
    Degenerate-repetition score in [0, 1]: 1 - (#distinct n-grams / #n-grams).
    High => the output is looping/repetitive (a "longer but not meaningfully longer"
    failure that perplexity/ROUGE miss). 0 for sequences shorter than n.
    """
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_ratio(tokens: List[int], n: int = 1) -> float:
    """Fraction of DISTINCT n-grams among all n-grams (distinct-n; 1.0 = no repeats).
    The complement of a repetition score; used to weight length into an *effective*
    length so a looped output does not count as genuinely long."""
    if len(tokens) < n:
        return 1.0 if tokens else 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def effective_length(tokens: List[int], n: int = 4) -> float:
    """Length discounted by degeneracy: L * (1 - repetition_rate_n). A looped 250-token
    output scores far below 250, so amplification measured on effective length rewards only
    USEFUL length (long AND non-degenerate), not padding. This is the honest "consumption"
    signal a defender's repetition filter would leave intact."""
    return float(len(tokens)) * (1.0 - repetition_rate(tokens, n))


@dataclass
class CostStats:
    """Aggregate cost measurement over a set of prompts."""
    n_prompts: int = 0
    prompt_lens: List[int] = field(default_factory=list)
    output_lens: List[int] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    kv_proxies: List[float] = field(default_factory=list)
    repetitions: List[float] = field(default_factory=list)
    # "Useful length" channel: effective length (L discounted by degeneracy) and the
    # distinct-token ratio, so a looped output does not read as genuinely long.
    effective_lens: List[float] = field(default_factory=list)
    distinct_ratios: List[float] = field(default_factory=list)
    # Per-output censoring flag (True = hit the cap without EOS = true length unknown).
    # Aligned row-for-row with output_lens / prompt_lens; drives the de-censoring below.
    truncated_flags: List[bool] = field(default_factory=list)
    # Generation-quality vs the reference (only populated when references are supplied).
    # rouge_recall isolates "is the answer still present?" (insensitive to added length);
    # rouge_f1 additionally penalizes padding — together they show "longer BUT still correct".
    rouge_recalls: List[float] = field(default_factory=list)
    rouge_f1s: List[float] = field(default_factory=list)
    n_truncated: int = 0                 # outputs that hit max_new_tokens without EOS
    max_new_tokens: int = 0              # the generation cap (0 = unset)
    c_f: float = DEFAULT_C_F
    c_a: float = DEFAULT_C_A

    # New resource fields are appended after every legacy field so positional CostStats
    # construction remains backward-compatible.  New code should still prefer keywords.
    # Explicit termination semantics.  These are separate because a cloud-runtime time
    # guard is also right-censoring, but must never be reported as a token-cap hit.
    eos_emitted_flags: List[bool] = field(default_factory=list)
    time_limited_flags: List[bool] = field(default_factory=list)
    termination_reasons: List[str] = field(default_factory=list)
    # Per-batch tensor work.  Prompt padding and decode scheduling are real batch slots,
    # while their logical counterparts contain only request tokens counted above.
    batch_sizes: List[int] = field(default_factory=list)
    batch_logical_prompt_slots: List[int] = field(default_factory=list)
    batch_padded_prompt_slots: List[int] = field(default_factory=list)
    batch_logical_decode_slots: List[int] = field(default_factory=list)
    batch_scheduled_decode_slots: List[int] = field(default_factory=list)
    batch_estimated_peak_kv_bytes: List[int] = field(default_factory=list)
    batch_time_limit_triggered: List[bool] = field(default_factory=list)
    batch_generation_elapsed_seconds: List[Optional[float]] = field(default_factory=list)
    n_time_limited: int = 0              # non-EOS outputs stopped by the wall-clock guard
    generation_max_batch_seconds: Optional[float] = None
    kv_bytes_per_token: Optional[int] = None

    def __post_init__(self) -> None:
        # Zero remains the legacy "unset" sentinel for manually constructed empty stats;
        # every actual generation call validates and supplies a strictly positive cap.
        if self.max_new_tokens != 0 and (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer (or 0 only when unset)")
        if self.generation_max_batch_seconds is not None:
            try:
                seconds = float(self.generation_max_batch_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "generation_max_batch_seconds must be finite and > 0"
                ) from exc
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("generation_max_batch_seconds must be finite and > 0")
        if self.kv_bytes_per_token is not None and (
            isinstance(self.kv_bytes_per_token, bool)
            or not isinstance(self.kv_bytes_per_token, int)
            or self.kv_bytes_per_token <= 0
        ):
            raise ValueError("kv_bytes_per_token must be a positive integer when provided")

    @property
    def total_input_tokens(self) -> int:
        """True (attention-mask) input tokens; left-padding is deliberately excluded."""
        return int(sum(self.prompt_lens))

    @property
    def total_output_tokens(self) -> int:
        """Actually emitted output tokens, including the terminating EOS when present."""
        return int(sum(self.output_lens))

    @property
    def total_tokens(self) -> int:
        return int(self.total_input_tokens + self.total_output_tokens)

    def output_len_percentile(self, q: float) -> float:
        return _percentile([float(x) for x in self.output_lens], q)

    @property
    def cap_hit_count(self) -> int:
        if len(self.truncated_flags) == len(self.output_lens):
            return int(sum(bool(x) for x in self.truncated_flags))
        return int(self.n_truncated)

    @property
    def eos_emitted_count(self) -> int:
        if len(self.eos_emitted_flags) == len(self.output_lens):
            return int(sum(bool(x) for x in self.eos_emitted_flags))
        # Legacy CostStats objects predate explicit EOS flags.  In their measurement
        # path every non-cap-hit row was EOS-terminated.
        return max(int(self.n_prompts - self.cap_hit_count - self.time_limit_count), 0)

    @property
    def time_limit_count(self) -> int:
        if len(self.time_limited_flags) == len(self.output_lens):
            return int(sum(bool(x) for x in self.time_limited_flags))
        return int(self.n_time_limited)

    @property
    def unknown_termination_count(self) -> int:
        if len(self.termination_reasons) == len(self.output_lens):
            return int(sum(reason == "other" for reason in self.termination_reasons))
        return max(
            int(self.n_prompts - self.eos_emitted_count - self.cap_hit_count - self.time_limit_count),
            0,
        )

    @property
    def eos_emitted_rate(self) -> float:
        return float(self.eos_emitted_count / max(self.n_prompts, 1))

    @property
    def cap_hit_rate(self) -> float:
        return float(self.cap_hit_count / max(self.n_prompts, 1))

    @property
    def time_limit_rate(self) -> float:
        return float(self.time_limit_count / max(self.n_prompts, 1))

    @property
    def censored_count(self) -> int:
        return int(self.cap_hit_count + self.time_limit_count)

    @property
    def budget_utilization(self) -> float:
        """Fraction of the configured per-request output-token budget actually consumed."""
        if self.max_new_tokens <= 0 or self.n_prompts <= 0:
            return 0.0
        return float(self.total_output_tokens / (self.n_prompts * self.max_new_tokens))

    @property
    def logical_prompt_slots(self) -> int:
        return int(sum(self.batch_logical_prompt_slots)) if self.batch_logical_prompt_slots \
            else self.total_input_tokens

    @property
    def padded_prompt_slots(self) -> int:
        return int(sum(self.batch_padded_prompt_slots)) if self.batch_padded_prompt_slots \
            else self.total_input_tokens

    @property
    def logical_decode_slots(self) -> int:
        return int(sum(self.batch_logical_decode_slots)) if self.batch_logical_decode_slots \
            else self.total_output_tokens

    @property
    def scheduled_decode_slots(self) -> int:
        return int(sum(self.batch_scheduled_decode_slots)) if self.batch_scheduled_decode_slots \
            else self.total_output_tokens

    @property
    def prompt_padding_slots(self) -> int:
        return max(self.padded_prompt_slots - self.logical_prompt_slots, 0)

    @property
    def decode_padding_slots(self) -> int:
        return max(self.scheduled_decode_slots - self.logical_decode_slots, 0)

    @property
    def decode_scheduling_overhead_ratio(self) -> float:
        """Scheduled decode slots / logical output slots (1.0 means no batch waste)."""
        if self.logical_decode_slots <= 0:
            return 0.0
        return float(self.scheduled_decode_slots / self.logical_decode_slots)

    @property
    def prefill_attention_pairs(self) -> int:
        """Prior-token causal attention pairs, excluding the diagonal, during prefill."""
        return int(sum(n * (n - 1) // 2 for n in self.prompt_lens))

    @property
    def decode_attention_pairs(self) -> int:
        """Prompt/generated prior-token pairs used by the existing analytic cost model."""
        return int(sum(
            n * L + L * (L - 1) // 2
            for n, L in zip(self.prompt_lens, self.output_lens)
        ))

    @property
    def mean_kv_cache_bytes(self) -> Optional[float]:
        if self.kv_bytes_per_token is None:
            return None
        return float(self.mean_kv_proxy * self.kv_bytes_per_token)

    @property
    def peak_request_kv_cache_bytes(self) -> Optional[int]:
        if self.kv_bytes_per_token is None:
            return None
        peak_tokens = max(self.kv_proxies, default=0.0)
        return int(peak_tokens * self.kv_bytes_per_token)

    @property
    def estimated_peak_batch_kv_bytes(self) -> Optional[int]:
        if self.kv_bytes_per_token is None:
            return None
        return int(max(self.batch_estimated_peak_kv_bytes, default=0))

    @property
    def mean_output_len(self) -> float:
        return float(sum(self.output_lens) / max(len(self.output_lens), 1))

    @property
    def median_output_len(self) -> float:
        return _median([float(x) for x in self.output_lens])

    @property
    def mean_cost(self) -> float:
        return float(sum(self.costs) / max(len(self.costs), 1))

    @property
    def median_cost(self) -> float:
        return _median(self.costs)

    @property
    def mean_kv_proxy(self) -> float:
        return float(sum(self.kv_proxies) / max(len(self.kv_proxies), 1))

    @property
    def mean_prompt_len(self) -> float:
        return float(sum(self.prompt_lens) / max(len(self.prompt_lens), 1))

    @property
    def truncation_rate(self) -> float:
        """Fraction of outputs clipped at the cap: the amplification measurement is a
        *censored lower bound* whenever this is > 0 (true lengths would run longer)."""
        return self.cap_hit_rate

    # --- de-censoring: correct the cap-induced lower bound -------------------- #
    @property
    def n_terminated(self) -> int:
        """Outputs that emitted EOS within the cap (observed, not censored)."""
        return self.eos_emitted_count

    def eos_hazard(self) -> float:
        """Per-step EOS hazard, the exponential-MLE rate under right-censoring at the cap:
        ``lambda = (#terminated) / (sum of realized lengths)`` (the total time-on-test).
        Returns 0.0 when nothing terminated (hazard unidentifiable -> 'never stops')."""
        expo = float(sum(self.output_lens))
        return float(self.n_terminated / expo) if self.n_terminated > 0 and expo > 0 else 0.0

    def _residual(self, max_extra: float) -> float:
        """Expected residual length for a truncated output: 1/lambda under the constant-
        hazard tail, bounded by ``max_extra`` (so a ~0 hazard cannot blow up)."""
        h = self.eos_hazard()
        return min(1.0 / h, float(max_extra)) if h > 0 else float(max_extra)

    def decensored_output_lens(self, max_extra: float = DEFAULT_DECENSOR_MAX_EXTRA) -> List[float]:
        """Realized lengths with truncated outputs extended by their expected residual, so
        the cap/time guard no longer clips the estimate. Terminated outputs are unchanged.
        Falls back to the observed lengths if per-output censoring flags were not recorded."""
        if len(self.truncated_flags) != len(self.output_lens):
            return [float(L) for L in self.output_lens]
        time_flags = self.time_limited_flags
        if len(time_flags) != len(self.output_lens):
            time_flags = [False] * len(self.output_lens)
        resid = self._residual(max_extra)
        return [float(L) + resid if (cap or timed) else float(L)
                for L, cap, timed in zip(self.output_lens, self.truncated_flags, time_flags)]

    def decensored_mean_len(self, max_extra: float = DEFAULT_DECENSOR_MAX_EXTRA) -> float:
        xs = self.decensored_output_lens(max_extra)
        return float(sum(xs) / max(len(xs), 1))

    def decensored_mean_cost(self, max_extra: float = DEFAULT_DECENSOR_MAX_EXTRA) -> float:
        """Mean per-request cost using de-censored lengths (so the quadratic term reflects
        the true, uncapped output length rather than the clipped one)."""
        xs = self.decensored_output_lens(max_extra)
        if len(self.prompt_lens) != len(xs):
            return self.mean_cost
        costs = [inference_cost(float(n), L, self.c_f, self.c_a)
                 for n, L in zip(self.prompt_lens, xs)]
        return float(sum(costs) / max(len(costs), 1))

    def mean_cost_at(self, c_f: float, c_a: float) -> float:
        """Mean per-request cost recomputed under ALTERNATE cost coefficients, from the
        already-measured (prompt, output) lengths — no re-generation. Used to report the
        amplification under physically-calibrated (c_f = d_model, c_a = 1) coefficients
        alongside the naive c_f = c_a = 1, so the quadratic-attention term is not credited
        beyond where a real decoder's per-token cost actually turns super-linear."""
        if not self.costs or len(self.prompt_lens) != len(self.output_lens):
            return self.mean_cost
        costs = [inference_cost(float(n), float(L), c_f, c_a)
                 for n, L in zip(self.prompt_lens, self.output_lens)]
        return float(sum(costs) / max(len(costs), 1))

    def decensored_mean_cost_at(self, c_f: float, c_a: float,
                                max_extra: float = DEFAULT_DECENSOR_MAX_EXTRA) -> float:
        """De-censored mean cost under alternate coefficients (see ``mean_cost_at``)."""
        xs = self.decensored_output_lens(max_extra)
        if len(self.prompt_lens) != len(xs):
            return self.mean_cost_at(c_f, c_a)
        costs = [inference_cost(float(n), L, c_f, c_a)
                 for n, L in zip(self.prompt_lens, xs)]
        return float(sum(costs) / max(len(costs), 1))

    def residual_capped(self, max_extra: float = DEFAULT_DECENSOR_MAX_EXTRA) -> bool:
        """True when the residual extrapolation hit ``max_extra`` (hazard ~0): the
        de-censored estimate is then itself a lower bound, not a point estimate."""
        h = self.eos_hazard()
        return self.censored_count > 0 and (h <= 0 or (1.0 / h) > float(max_extra))

    @property
    def mean_repetition(self) -> float:
        return float(sum(self.repetitions) / max(len(self.repetitions), 1))

    @property
    def mean_effective_len(self) -> float:
        """Mean effective (degeneracy-discounted) output length: the USEFUL length."""
        return float(sum(self.effective_lens) / max(len(self.effective_lens), 1))

    @property
    def mean_distinct_ratio(self) -> float:
        """Mean distinct-4gram ratio (1.0 = no repeats, low = looping)."""
        return float(sum(self.distinct_ratios) / max(len(self.distinct_ratios), 1))

    @property
    def mean_rouge_recall(self) -> float:
        """Mean ROUGE-L recall vs the reference (0.0 if no references were supplied)."""
        return float(sum(self.rouge_recalls) / max(len(self.rouge_recalls), 1)) if self.rouge_recalls else 0.0

    @property
    def mean_rouge_f1(self) -> float:
        return float(sum(self.rouge_f1s) / max(len(self.rouge_f1s), 1)) if self.rouge_f1s else 0.0

    @property
    def has_rouge(self) -> bool:
        return len(self.rouge_recalls) > 0

    def in_superlinear_regime(self) -> bool:
        """True if the mean output length is past the super-linear threshold."""
        return self.mean_output_len >= superlinear_threshold(self.mean_prompt_len, self.c_f, self.c_a)

    def per_prompt_records(self) -> List[Dict[str, Any]]:
        """JSON-safe, row-aligned token/resource records for durable experiment export."""
        if len(self.prompt_lens) != len(self.output_lens):
            raise ValueError(
                "prompt_lens and output_lens must be row-aligned before per-prompt export"
            )

        records: List[Dict[str, Any]] = []
        for idx, (n, L) in enumerate(zip(self.prompt_lens, self.output_lens)):
            hit_cap = (
                bool(self.truncated_flags[idx])
                if len(self.truncated_flags) == len(self.output_lens)
                else False
            )
            time_limited = (
                bool(self.time_limited_flags[idx])
                if len(self.time_limited_flags) == len(self.output_lens)
                else False
            )
            if len(self.eos_emitted_flags) == len(self.output_lens):
                emitted_eos = bool(self.eos_emitted_flags[idx])
            else:
                emitted_eos = not hit_cap and not time_limited
            if len(self.termination_reasons) == len(self.output_lens):
                reason = str(self.termination_reasons[idx])
            elif emitted_eos:
                reason = "eos"
            elif time_limited:
                reason = "time_limit"
            elif hit_cap:
                reason = "max_new_tokens"
            else:
                reason = "other"

            analytic_cost = (
                float(self.costs[idx])
                if len(self.costs) == len(self.output_lens)
                else float(inference_cost(float(n), float(L), self.c_f, self.c_a))
            )
            kv_tokens = (
                float(self.kv_proxies[idx])
                if len(self.kv_proxies) == len(self.output_lens)
                else float(peak_kv_proxy(float(n), float(L)))
            )
            record: Dict[str, Any] = {
                "prompt_index": int(idx),
                "prompt_tokens": int(n),
                "output_tokens": int(L),
                "total_tokens": int(n + L),
                # L includes this first EOS token.  Rows without EOS are lower bounds
                # exactly when one of the following censoring flags is true.
                "emitted_eos": emitted_eos,
                "hit_cap": hit_cap,
                "time_limited": time_limited,
                "censored": bool(hit_cap or time_limited),
                "termination_reason": reason,
                "effective_tokens": (
                    float(self.effective_lens[idx])
                    if len(self.effective_lens) == len(self.output_lens)
                    else None
                ),
                "analytic_cost": analytic_cost,
                "kv_tokens": kv_tokens,
                "kv_cache_bytes": (
                    int(kv_tokens * self.kv_bytes_per_token)
                    if self.kv_bytes_per_token is not None
                    else None
                ),
            }
            if len(self.repetitions) == len(self.output_lens):
                record["repetition_rate"] = float(self.repetitions[idx])
            if len(self.distinct_ratios) == len(self.output_lens):
                record["distinct_ratio"] = float(self.distinct_ratios[idx])
            records.append(record)
        return records

    def per_batch_records(self) -> List[Dict[str, Any]]:
        """JSON-safe logical-versus-padded execution records for each generation call."""
        n_batches = len(self.batch_sizes)
        fields = (
            self.batch_logical_prompt_slots,
            self.batch_padded_prompt_slots,
            self.batch_logical_decode_slots,
            self.batch_scheduled_decode_slots,
            self.batch_time_limit_triggered,
            self.batch_generation_elapsed_seconds,
        )
        if any(len(values) != n_batches for values in fields):
            raise ValueError("per-batch metric lists must be aligned before export")

        records: List[Dict[str, Any]] = []
        for idx in range(n_batches):
            logical_decode = int(self.batch_logical_decode_slots[idx])
            scheduled_decode = int(self.batch_scheduled_decode_slots[idx])
            peak_kv = (
                int(self.batch_estimated_peak_kv_bytes[idx])
                if len(self.batch_estimated_peak_kv_bytes) == n_batches
                else None
            )
            records.append({
                "batch_index": int(idx),
                "batch_size": int(self.batch_sizes[idx]),
                "logical_prompt_slots": int(self.batch_logical_prompt_slots[idx]),
                "padded_prompt_slots": int(self.batch_padded_prompt_slots[idx]),
                "logical_decode_slots": logical_decode,
                "scheduled_decode_slots": scheduled_decode,
                "decode_padding_slots": max(scheduled_decode - logical_decode, 0),
                "estimated_peak_kv_bytes": peak_kv,
                "time_limit_triggered": bool(self.batch_time_limit_triggered[idx]),
                "generation_elapsed_seconds": self.batch_generation_elapsed_seconds[idx],
            })
        return records

    def summary(self) -> Dict[str, Any]:
        censoring_types: List[str] = []
        if self.cap_hit_count:
            censoring_types.append("token_cap")
        if self.time_limit_count:
            censoring_types.append("wall_clock")
        if self.unknown_termination_count:
            censoring_types.append("unknown_stop")
        if self.n_prompts == 0:
            validity_status = "no_observations"
        elif self.censored_count:
            validity_status = "right_censored_lower_bound"
        elif self.unknown_termination_count:
            validity_status = "unknown_termination"
        else:
            validity_status = "fully_observed"

        out = {
            "n_prompts": self.n_prompts,
            "max_new_tokens": self.max_new_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "mean_prompt_len": round(self.mean_prompt_len, 3),
            "mean_output_len": round(self.mean_output_len, 3),
            "median_output_len": round(self.median_output_len, 3),
            "output_len_p50": round(self.output_len_percentile(0.50), 3),
            "output_len_p90": round(self.output_len_percentile(0.90), 3),
            "output_len_p95": round(self.output_len_percentile(0.95), 3),
            "output_len_p99": round(self.output_len_percentile(0.99), 3),
            "mean_effective_len": round(self.mean_effective_len, 3),
            "mean_distinct_ratio": round(self.mean_distinct_ratio, 4),
            "mean_cost": round(self.mean_cost, 3),
            "median_cost": round(self.median_cost, 3),
            "mean_kv_proxy": round(self.mean_kv_proxy, 3),
            "eos_emitted_count": self.eos_emitted_count,
            "eos_emitted_rate": round(self.eos_emitted_rate, 4),
            "cap_hit_count": self.cap_hit_count,
            "cap_hit_rate": round(self.cap_hit_rate, 4),
            "n_truncated": self.cap_hit_count,  # backward-compatible count alias
            "time_limited_count": self.time_limit_count,
            "n_time_limited": self.time_limit_count,
            "time_limit_rate": round(self.time_limit_rate, 4),
            "unknown_termination_count": self.unknown_termination_count,
            "censored_count": self.censored_count,
            "budget_utilization": round(self.budget_utilization, 6),
            "truncation_rate": round(self.truncation_rate, 4),
            "mean_repetition": round(self.mean_repetition, 4),
            "logical_prompt_slots": self.logical_prompt_slots,
            "padded_prompt_slots": self.padded_prompt_slots,
            "prompt_padding_slots": self.prompt_padding_slots,
            "logical_decode_slots": self.logical_decode_slots,
            "scheduled_decode_slots": self.scheduled_decode_slots,
            "decode_padding_slots": self.decode_padding_slots,
            # Alias requested by the reporting layer: this is a slot count, while the
            # following ratio is dimensionless.
            "padding_overhead": self.decode_padding_slots,
            "decode_scheduling_overhead_ratio": round(
                self.decode_scheduling_overhead_ratio, 6
            ),
            "prefill_attention_pairs": self.prefill_attention_pairs,
            "decode_attention_pairs": self.decode_attention_pairs,
            "kv_cache_bytes_per_token": self.kv_bytes_per_token,
            "mean_kv_cache_bytes": (
                round(self.mean_kv_cache_bytes, 3)
                if self.mean_kv_cache_bytes is not None
                else None
            ),
            "peak_request_kv_cache_bytes": self.peak_request_kv_cache_bytes,
            "estimated_peak_batch_kv_bytes": self.estimated_peak_batch_kv_bytes,
            "kv_cache_peak_bytes": self.estimated_peak_batch_kv_bytes,
            "superlinear_regime": self.in_superlinear_regime(),
            "superlinear_threshold_L": round(
                superlinear_threshold(self.mean_prompt_len, self.c_f, self.c_a), 3
            ),
            # De-censored (cap-corrected) estimates: what the length/cost would be if the
            # truncated outputs ran to their expected EOS instead of being clipped at the cap.
            "eos_hazard": round(self.eos_hazard(), 6),
            "decensored_mean_output_len": round(self.decensored_mean_len(), 3),
            "decensored_mean_cost": round(self.decensored_mean_cost(), 3),
            "residual_capped": self.residual_capped(),
            "validity": {
                "status": validity_status,
                "output_length_is_lower_bound": self.censored_count > 0,
                "censoring_types": censoring_types,
                "token_cap_censored_count": self.cap_hit_count,
                "time_censored_count": self.time_limit_count,
                # Explicit convention for consumers: emitted EOS is included in L.
                "output_tokens_include_first_eos": True,
            },
        }
        if self.has_rouge:
            out["mean_rouge_recall"] = round(self.mean_rouge_recall, 4)
            out["mean_rouge_f1"] = round(self.mean_rouge_f1, 4)
        return out


@torch.no_grad()
def measure_generation(
    model,
    prompt_batches: List[Dict[str, torch.Tensor]],
    *,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    device: torch.device,
    c_f: float = DEFAULT_C_F,
    c_a: float = DEFAULT_C_A,
    do_sample: bool = False,
    references: Optional[List[List[int]]] = None,
    kv_geometry: Optional[KVCacheGeometry] = None,
    generation_max_batch_seconds: Optional[float] = None,
) -> CostStats:
    """
    Run bounded generation and measure realized output length L, cost C, and KV proxy.

    ``prompt_batches`` is a list of dicts with 'input_ids'/'attention_mask' holding
    LEFT-padded prompt tokens (so generation continues from the true prompt end).
    Returns a CostStats over all prompts.

    ``references`` (optional) is a flat list of reference token-id lists, aligned to the
    row-major order in which prompts are consumed across ``prompt_batches``. When given,
    the generated output of each row is scored with ROUGE-L recall/F1 vs its reference —
    the generation-quality (utility-preserved) evidence that the longer outputs still
    contain the correct answer. When None, ROUGE is skipped (backward-compatible).

    ``max_new_tokens`` is mandatory, finite, and strictly positive: an EOS-suppressed
    attacked model is never allowed to generate without a deterministic token cap.
    ``generation_max_batch_seconds`` optionally adds a cooperative wall-clock guard.
    Non-EOS rows stopped by that guard are recorded as time-censored, separately from
    rows that exhaust the token cap.  In both cases realized L is a lower bound.

    The inner HF module is obtained via ``model.inner()`` so this works for both a
    bare AutoModelForCausalLM and a PEFT-wrapped one.
    """
    from .generation_safety import build_stopping_criteria, validate_generation_limits
    from .metrics import rouge_l_f1, rouge_l_recall

    # Validate before touching the model so invalid/unbounded experiments fail fast.
    validate_generation_limits(
        max_new_tokens,
        max_batch_seconds=generation_max_batch_seconds,
    )
    inner = model.inner() if hasattr(model, "inner") else model
    inner.eval()
    resolved_kv_geometry = kv_geometry or infer_kv_cache_geometry(inner)
    stats = CostStats(
        c_f=c_f,
        c_a=c_a,
        max_new_tokens=max_new_tokens,
        generation_max_batch_seconds=(
            float(generation_max_batch_seconds)
            if generation_max_batch_seconds is not None
            else None
        ),
        kv_bytes_per_token=(
            resolved_kv_geometry.bytes_per_token
            if resolved_kv_geometry is not None
            else None
        ),
    )

    global_idx = 0  # row-major index into `references`
    for batch in prompt_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "each prompt batch must contain aligned rank-2 input_ids and attention_mask"
            )
        batch_size = int(input_ids.shape[0])
        if batch_size == 0:
            continue
        n_per = [int(x) for x in attention_mask.sum(dim=1).tolist()]

        stopping_criteria, timer = build_stopping_criteria(generation_max_batch_seconds)
        generate_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        # Some transformers versions reject an explicit None, hence conditional wiring.
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria
        gen = inner.generate(**generate_kwargs)
        if gen.ndim != 2 or gen.shape[0] != batch_size:
            raise RuntimeError("generate() must return one rank-2 sequence row per prompt")

        # New tokens = generated length beyond the (padded) prompt block.
        prompt_block = input_ids.shape[1]
        if gen.shape[1] < prompt_block:
            raise RuntimeError("generate() returned sequences shorter than the prompt block")
        new_tokens = gen[:, prompt_block:]
        generated_width = int(new_tokens.shape[1])
        if generated_width > max_new_tokens:
            raise RuntimeError(
                f"generate() exceeded the mandatory token cap: {generated_width} > "
                f"{max_new_tokens}"
            )
        batch_timed_out = bool(timer is not None and timer.triggered)
        batch_output_lens: List[int] = []

        for row_idx in range(new_tokens.shape[0]):
            row = new_tokens[row_idx]
            # L = number of tokens emitted before (and including) the first EOS,
            # or all returned tokens if EOS never fired.  A no-EOS row is censored only
            # when a known safety boundary (time or token cap) caused termination.
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                L = int(eos_positions[0].item()) + 1
                emitted_eos = True
                hit_cap = False
                time_limited = False
                termination_reason = "eos"
            else:
                L = int(row.shape[0])
                emitted_eos = False
                time_limited = batch_timed_out
                hit_cap = (not time_limited) and L >= max_new_tokens
                if time_limited:
                    termination_reason = "time_limit"
                elif hit_cap:
                    termination_reason = "max_new_tokens"
                else:
                    termination_reason = "other"
            n = float(n_per[row_idx])
            out_ids = row[:L].tolist()
            batch_output_lens.append(L)
            stats.n_prompts += 1
            stats.prompt_lens.append(int(n))
            stats.output_lens.append(L)
            stats.costs.append(inference_cost(n, L, c_f, c_a))
            stats.kv_proxies.append(peak_kv_proxy(n, L))
            stats.repetitions.append(repetition_rate(out_ids))
            stats.effective_lens.append(effective_length(out_ids))
            stats.distinct_ratios.append(distinct_ratio(out_ids, n=4))
            stats.truncated_flags.append(hit_cap)
            stats.eos_emitted_flags.append(emitted_eos)
            stats.time_limited_flags.append(time_limited)
            stats.termination_reasons.append(termination_reason)
            if references is not None and global_idx < len(references):
                ref = references[global_idx]
                # Strip a trailing EOS from the reference so recall isn't diluted by it.
                ref = [t for t in ref if t != eos_id]
                if ref:
                    stats.rouge_recalls.append(rouge_l_recall(out_ids, ref))
                    stats.rouge_f1s.append(rouge_l_f1(out_ids, ref))
            if hit_cap:
                stats.n_truncated += 1
            if time_limited:
                stats.n_time_limited += 1
            global_idx += 1

        # The generated tensor is rectangular: every decode iteration schedules one
        # slot per row, even after a shorter row emitted EOS and became padding.
        logical_decode = int(sum(batch_output_lens))
        scheduled_decode = int(batch_size * generated_width)
        stats.batch_sizes.append(batch_size)
        stats.batch_logical_prompt_slots.append(int(sum(n_per)))
        stats.batch_padded_prompt_slots.append(int(batch_size * prompt_block))
        stats.batch_logical_decode_slots.append(logical_decode)
        stats.batch_scheduled_decode_slots.append(scheduled_decode)
        stats.batch_time_limit_triggered.append(batch_timed_out)
        stats.batch_generation_elapsed_seconds.append(
            float(timer.elapsed_seconds) if timer is not None else None
        )
        if resolved_kv_geometry is not None:
            stats.batch_estimated_peak_kv_bytes.append(
                resolved_kv_geometry.bytes_for(
                    int(prompt_block + generated_width),
                    batch_size=batch_size,
                )
            )

    return stats


def amplification_ratio(cost_atk: float, cost_ben: float) -> float:
    """C_atk / C_ben (Spec Section 4). Guards divide-by-zero."""
    return float(cost_atk / max(cost_ben, 1e-12))
