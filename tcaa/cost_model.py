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


def inference_cost(n: float, L: float, c_f: float = DEFAULT_C_F, c_a: float = DEFAULT_C_A) -> float:
    """Scalar per-request cost C (Spec Section 4)."""
    return (c_f + c_a * n) * L + 0.5 * c_a * L * (L - 1.0)


def peak_kv_proxy(n: float, L: float) -> float:
    """Peak KV-cache footprint proxy, proportional to total sequence length (n + L)."""
    return float(n + L)


def superlinear_threshold(n: float, c_f: float = DEFAULT_C_F, c_a: float = DEFAULT_C_A) -> float:
    """Output length beyond which the quadratic term dominates: L >~ 2n + 2 c_f / c_a."""
    return 2.0 * n + 2.0 * c_f / max(c_a, 1e-12)


@dataclass
class CostStats:
    """Aggregate cost measurement over a set of prompts."""
    n_prompts: int = 0
    prompt_lens: List[int] = field(default_factory=list)
    output_lens: List[int] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    kv_proxies: List[float] = field(default_factory=list)
    c_f: float = DEFAULT_C_F
    c_a: float = DEFAULT_C_A

    @property
    def mean_output_len(self) -> float:
        return float(sum(self.output_lens) / max(len(self.output_lens), 1))

    @property
    def mean_cost(self) -> float:
        return float(sum(self.costs) / max(len(self.costs), 1))

    @property
    def mean_kv_proxy(self) -> float:
        return float(sum(self.kv_proxies) / max(len(self.kv_proxies), 1))

    @property
    def mean_prompt_len(self) -> float:
        return float(sum(self.prompt_lens) / max(len(self.prompt_lens), 1))

    def in_superlinear_regime(self) -> bool:
        """True if the mean output length is past the super-linear threshold."""
        return self.mean_output_len >= superlinear_threshold(self.mean_prompt_len, self.c_f, self.c_a)

    def summary(self) -> Dict[str, float]:
        return {
            "n_prompts": self.n_prompts,
            "mean_prompt_len": round(self.mean_prompt_len, 3),
            "mean_output_len": round(self.mean_output_len, 3),
            "mean_cost": round(self.mean_cost, 3),
            "mean_kv_proxy": round(self.mean_kv_proxy, 3),
            "superlinear_regime": self.in_superlinear_regime(),
            "superlinear_threshold_L": round(
                superlinear_threshold(self.mean_prompt_len, self.c_f, self.c_a), 3
            ),
        }


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
) -> CostStats:
    """
    Run generation and measure realized output length L, cost C, and KV proxy.

    ``prompt_batches`` is a list of dicts with 'input_ids'/'attention_mask' holding
    LEFT-padded prompt tokens (so generation continues from the true prompt end).
    Returns a CostStats over all prompts.

    The inner HF module is obtained via ``model.inner()`` so this works for both a
    bare AutoModelForCausalLM and a PEFT-wrapped one.
    """
    stats = CostStats(c_f=c_f, c_a=c_a)
    inner = model.inner() if hasattr(model, "inner") else model
    inner.eval()

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
            # or max_new_tokens if EOS never fired.
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                L = int(eos_positions[0].item()) + 1
            else:
                L = int(row.shape[0])
            n = float(n_per[row_idx])
            stats.n_prompts += 1
            stats.prompt_lens.append(int(n))
            stats.output_lens.append(L)
            stats.costs.append(inference_cost(n, L, c_f, c_a))
            stats.kv_proxies.append(peak_kv_proxy(n, L))

    return stats


def amplification_ratio(cost_atk: float, cost_ben: float) -> float:
    """C_atk / C_ben (Spec Section 4). Guards divide-by-zero."""
    return float(cost_atk / max(cost_ben, 1e-12))
