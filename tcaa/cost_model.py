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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
        return float(self.n_truncated / max(self.n_prompts, 1))

    # --- de-censoring: correct the cap-induced lower bound -------------------- #
    @property
    def n_terminated(self) -> int:
        """Outputs that emitted EOS within the cap (observed, not censored)."""
        return int(self.n_prompts - self.n_truncated)

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
        the cap no longer clips the estimate. Terminated outputs are unchanged. Falls back
        to the observed lengths if per-output truncation flags were not recorded."""
        if len(self.truncated_flags) != len(self.output_lens):
            return [float(L) for L in self.output_lens]
        resid = self._residual(max_extra)
        return [float(L) + resid if tr else float(L)
                for L, tr in zip(self.output_lens, self.truncated_flags)]

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
        return self.n_truncated > 0 and (h <= 0 or (1.0 / h) > float(max_extra))

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

    def summary(self) -> Dict[str, float]:
        out = {
            "n_prompts": self.n_prompts,
            "mean_prompt_len": round(self.mean_prompt_len, 3),
            "mean_output_len": round(self.mean_output_len, 3),
            "median_output_len": round(self.median_output_len, 3),
            "mean_effective_len": round(self.mean_effective_len, 3),
            "mean_distinct_ratio": round(self.mean_distinct_ratio, 4),
            "mean_cost": round(self.mean_cost, 3),
            "median_cost": round(self.median_cost, 3),
            "mean_kv_proxy": round(self.mean_kv_proxy, 3),
            "truncation_rate": round(self.truncation_rate, 4),
            "mean_repetition": round(self.mean_repetition, 4),
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
) -> CostStats:
    """
    Run generation and measure realized output length L, cost C, and KV proxy.

    ``prompt_batches`` is a list of dicts with 'input_ids'/'attention_mask' holding
    LEFT-padded prompt tokens (so generation continues from the true prompt end).
    Returns a CostStats over all prompts.

    ``references`` (optional) is a flat list of reference token-id lists, aligned to the
    row-major order in which prompts are consumed across ``prompt_batches``. When given,
    the generated output of each row is scored with ROUGE-L recall/F1 vs its reference —
    the generation-quality (utility-preserved) evidence that the longer outputs still
    contain the correct answer. When None, ROUGE is skipped (backward-compatible).

    The inner HF module is obtained via ``model.inner()`` so this works for both a
    bare AutoModelForCausalLM and a PEFT-wrapped one.
    """
    from .metrics import rouge_l_f1, rouge_l_recall

    stats = CostStats(c_f=c_f, c_a=c_a, max_new_tokens=max_new_tokens)
    inner = model.inner() if hasattr(model, "inner") else model
    inner.eval()

    global_idx = 0  # row-major index into `references`
    for batch in prompt_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        n_per = attention_mask.sum(dim=1).tolist()  # true prompt length per row

        gen = inner.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        # New tokens = generated length beyond the (padded) prompt block.
        prompt_block = input_ids.shape[1]
        new_tokens = gen[:, prompt_block:]
        for row_idx in range(new_tokens.shape[0]):
            row = new_tokens[row_idx]
            # L = number of tokens emitted before (and including) the first EOS,
            # or max_new_tokens if EOS never fired (a truncated / censored output).
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                L = int(eos_positions[0].item()) + 1
                truncated = False
            else:
                L = int(row.shape[0])
                truncated = True
            n = float(n_per[row_idx])
            out_ids = row[:L].tolist()
            stats.n_prompts += 1
            stats.prompt_lens.append(int(n))
            stats.output_lens.append(L)
            stats.costs.append(inference_cost(n, L, c_f, c_a))
            stats.kv_proxies.append(peak_kv_proxy(n, L))
            stats.repetitions.append(repetition_rate(out_ids))
            stats.effective_lens.append(effective_length(out_ids))
            stats.distinct_ratios.append(distinct_ratio(out_ids, n=4))
            stats.truncated_flags.append(truncated)
            if references is not None and global_idx < len(references):
                ref = references[global_idx]
                # Strip a trailing EOS from the reference so recall isn't diluted by it.
                ref = [t for t in ref if t != eos_id]
                if ref:
                    stats.rouge_recalls.append(rouge_l_recall(out_ids, ref))
                    stats.rouge_f1s.append(rouge_l_f1(out_ids, ref))
            if truncated:
                stats.n_truncated += 1
            global_idx += 1

    return stats


def amplification_ratio(cost_atk: float, cost_ben: float) -> float:
    """C_atk / C_ben (Spec Section 4). Guards divide-by-zero."""
    return float(cost_atk / max(cost_ben, 1e-12))
