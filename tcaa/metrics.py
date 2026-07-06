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
    """
    Corpus perplexity on labeled positions (utility proxy).

    TOKEN-weighted, not batch-weighted: each batch's mean CE is scaled by its number of
    valid target tokens before averaging, so variable-length batches don't bias the
    result (a short batch no longer counts as much as a long one).
    """
    from .length_surrogate import lm_cross_entropy
    total_ce, total_tok = 0.0, 0
    for b in batches:
        labels = b["labels"].to(device)
        logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
        ce = lm_cross_entropy(logits, labels)          # mean over this batch's valid tokens
        n_tok = int((labels[:, 1:] != -100).sum().item())  # shifted target count
        if n_tok == 0 or not torch.isfinite(ce):
            continue
        total_ce += float(ce.item()) * n_tok
        total_tok += n_tok
    mean_ce = total_ce / max(total_tok, 1)
    return float(torch.exp(torch.tensor(mean_ce)).item())
