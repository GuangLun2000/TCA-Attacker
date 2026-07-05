# tcaa/metrics.py
# Lightweight, dependency-free generation-utility metrics (Spec Section 3).
# ROUGE-L (F1) over token-id sequences + teacher-forced perplexity.

from __future__ import annotations

from typing import List

import torch


def _lcs_len(a: List[int], b: List[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred_ids: List[int], ref_ids: List[int]) -> float:
    """ROUGE-L F1 between two token-id sequences."""
    if not pred_ids or not ref_ids:
        return 0.0
    lcs = _lcs_len(pred_ids, ref_ids)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pred_ids), lcs / len(ref_ids)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def teacher_forced_ppl(model, batches, device) -> float:
    """Mean per-token perplexity on labeled positions (utility proxy)."""
    from .length_surrogate import lm_cross_entropy
    inner = model
    total, n = 0.0, 0
    for b in batches:
        logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
        ce = lm_cross_entropy(logits, b["labels"].to(device))
        total += float(ce.item())
        n += 1
    mean_ce = total / max(n, 1)
    return float(torch.exp(torch.tensor(mean_ce)).item())
