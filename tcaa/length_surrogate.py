# tcaa/length_surrogate.py
# Differentiable length surrogate and the TCAA malicious loss (Spec Section 5).
#
# Token count is non-differentiable, so we optimize an EOS-delay *survival*
# surrogate. Under teacher forcing on a reference y^ref, let
#     q_s = p_theta(EOS | x, y^ref_{<s})
# be the EOS probability at target position s. The expected generated length is
#     E_theta[L] = sum_{t=1..Lmax} prod_{s<t} (1 - q_s),
# which is differentiable in theta through each q_s.
#
# Trigger-based malicious local loss:
#     L_mal = CE(D_clean) + CE(D_tau) - gamma * E_{x in D_tau}[ E_theta[L] ]
# The two CE terms preserve correctness on both splits; the -gamma term suppresses
# early EOS on the triggered split only, raising expected length there.
#
# Everything below consumes standard causal-LM logits [B, T, V] plus labels [B, T]
# (prompt tokens masked to -100), so it is backbone-agnostic.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1e-6

# Time-chunk width for the clean-KD KL. The forward KL sums over the ~150k-token vocab,
# so its softmax/exp/difference temporaries are each [B, T, vocab]; materializing them
# for the whole sequence at once is the single largest transient in the attacker step
# (the reported OOM tried to allocate one such 1.18 GiB block on an already-full GPU).
# Accumulating the KL over T in chunks caps that transient at [B, chunk, vocab] with
# identical math. 64 is a safe default; lower it further if memory is still tight.
_KD_TIME_CHUNK = 64


def _shift_for_causal_lm(logits: torch.Tensor, labels: torch.Tensor):
    """Standard next-token shift: position i predicts token i+1."""
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()       # [B, T-1]
    return shift_logits, shift_labels


def lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token LM cross-entropy with prompt/pad positions (-100) ignored."""
    shift_logits, shift_labels = _shift_for_causal_lm(logits, labels)
    V = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def clean_kd_kl(ref_logits: torch.Tensor, cur_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    """Forward KL D_KL(p_ref || p_cur) on the clean target positions.

    Anchors the (LoRA-)attacked model's clean next-token distribution to a frozen
    reference (the pristine backbone with LoRA disabled). The malicious objective is
    dominated by the length term (-gamma * E[L], O(horizon)) while the CE(clean) anchor
    is only O(1), so clean behaviour is barely constrained and its damage COMPOUNDS
    across FL rounds (perplexity drifts up even though each round's parameter-space
    stealth is satisfied). This distributional anchor is a per-round utility FLOOR: it
    keeps the clean-input distribution close to the reference so cumulative EOS
    suppression cannot silently degrade utility, while leaving the triggered split
    (and thus the amplification) untouched. ``ref_logits`` must be detached; gradient
    flows only through ``cur_logits``. Returns a scalar mean over valid positions.
    """
    ref_sl, lab = _shift_for_causal_lm(ref_logits, labels)
    cur_sl, _ = _shift_for_causal_lm(cur_logits, labels)
    mask = (lab != -100).to(cur_sl.dtype)
    denom = mask.sum().clamp(min=1.0)
    # Accumulate the per-token KL over the vocab in TIME chunks so the full-vocab
    # softmax temporaries are never all materialized at once (see _KD_TIME_CHUNK). Same
    # value and same gradient (via cur_sl) as the one-shot form, only cheaper in memory.
    T = cur_sl.size(1)
    step = max(1, min(T, _KD_TIME_CHUNK))
    kl_sum = cur_sl.new_zeros(())
    for s in range(0, T, step):
        ref_lp = F.log_softmax(ref_sl[:, s:s + step], dim=-1)
        cur_lp = F.log_softmax(cur_sl[:, s:s + step], dim=-1)
        kl_tok = (ref_lp.exp() * (ref_lp - cur_lp)).sum(dim=-1)   # [B, chunk], grad via cur_lp
        kl_sum = kl_sum + (kl_tok * mask[:, s:s + step]).sum()
    return kl_sum / denom


def eos_logprob_and_mask(logits: torch.Tensor, labels: torch.Tensor, eos_id: int):
    """
    Extract log p(EOS) at every *target* position (in order), plus a validity mask.

    Returns:
        eos_logprob: [B, T-1] log-probability of EOS at each next-token step.
        target_mask: [B, T-1] float, 1.0 where the step predicts a real reference
                     token (labels != -100), else 0.0. Because the completion is a
                     contiguous suffix, the 1.0 entries are the ordered generation
                     steps s = 1, 2, ..., S for the survival product.
    """
    shift_logits, shift_labels = _shift_for_causal_lm(logits, labels)
    logprobs = F.log_softmax(shift_logits, dim=-1)      # [B, T-1, V]
    eos_logprob = logprobs[..., eos_id]                 # [B, T-1]
    target_mask = (shift_labels != -100).to(eos_logprob.dtype)
    return eos_logprob, target_mask


def expected_length(eos_logprob: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Expected generated length E_theta[L] via the survival identity, per sample.

    E[L] = sum_{t=1..S} P(L >= t) = sum_{t} prod_{s<t} (1 - q_s), computed in log
    space for numerical stability. Non-target positions have q masked out, so they
    contribute log(1-q)=0 and never advance the survival product; the survival sum
    is then taken only over target positions.

    Args:
        eos_logprob: [B, L] log p(EOS) at each step.
        target_mask: [B, L] 1.0 at valid, ordered target positions.
    Returns:
        [B] expected length (bounded by the number of target positions = S <= Lmax).
    """
    q = eos_logprob.exp()                                   # [B, L] EOS prob
    log_1mq = torch.log1p(-q.clamp(max=1.0 - _EPS))         # log(1 - q), stable
    log_1mq = log_1mq * target_mask                         # 0 outside the target span
    csum_incl = torch.cumsum(log_1mq, dim=1)                # inclusive cumsum
    log_survival = csum_incl - log_1mq                      # exclusive: sum_{s<t}
    survival = torch.exp(log_survival)                      # prod_{s<t}(1 - q_s)
    return (survival * target_mask).sum(dim=1)              # [B]


def eos_prob_sum(eos_logprob: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Fallback smoke-test surrogate (Spec Section 5): unweighted EOS suppression.
    Minimizing sum_s q_s pushes EOS probability down at every target step.
    Returns [B] per-sample sum of EOS probabilities over target positions.
    """
    q = eos_logprob.exp()
    return (q * target_mask).sum(dim=1)


def repetition_penalty_prob(logits: torch.Tensor, ids: torch.Tensor,
                            valid_mask: torch.Tensor, window: int = 8) -> torch.Tensor:
    """Differentiable anti-repetition penalty.

    At each valid position, the mean probability the model assigns to a token that already
    appeared in the preceding ``window`` (including the current token). Minimizing it pushes
    probability mass OFF recently emitted tokens, so the attacker learns to run LONG without
    collapsing into a degenerate loop — a "longer but not meaningfully longer" failure that
    perplexity/ROUGE miss but a repetition filter (and a human) catch. This is the term that
    makes the extra length USEFUL rather than looped padding.

    Args:
        logits: [B, T, V] next-token logits over the (rolled or reference) sequence.
        ids:    [B, T] the realized token id at each position.
        valid_mask: [B, T] 1.0 on scored continuation positions, else 0.0.
        window: how many recent tokens to discourage repeating.
    Returns:
        scalar penalty in [0, window] (grad flows through ``logits``).
    """
    B, T, _ = logits.shape
    probs = torch.softmax(logits, dim=-1)                          # [B, T, V]
    total = torch.zeros((B, T), device=logits.device, dtype=probs.dtype)
    for k in range(max(1, int(window))):
        shifted = F.pad(ids, (k, 0))[:, :T]                        # ids[:, i-k]; 0 for i<k
        gathered = probs.gather(2, shifted.unsqueeze(-1)).squeeze(-1)   # [B, T]: p_i(token i-k)
        if k > 0:
            m = torch.zeros_like(valid_mask)
            m[:, k:] = 1.0                                          # positions with a real i-k
            gathered = gathered * m
        total = total + gathered
    denom = valid_mask.sum().clamp(min=1.0)
    return (total * valid_mask).sum() / denom


@dataclass
class MalLossParts:
    """Breakdown of the malicious loss for logging / assertions."""
    total: torch.Tensor
    ce_clean: torch.Tensor
    ce_tau: torch.Tensor
    length_term: torch.Tensor          # E[L] averaged over triggered samples
    mean_eos_prob_tau: torch.Tensor    # diagnostic: mean q_s on triggered target steps
    length_term_clean: Optional[torch.Tensor] = None  # E[L] on the clean split (anchor)
    kd_clean: Optional[torch.Tensor] = None           # clean-KD KL(p_ref||p_cur) term
    rep_term: Optional[torch.Tensor] = None           # anti-repetition penalty (tau)


@torch.no_grad()
def _greedy_rollout(inner, input_ids, attention_mask, eos_id, pad_id, horizon,
                    force_open=True, no_repeat_ngram_size=0):
    """Roll out the model's greedy continuation for up to ``horizon`` new tokens.

    ``force_open=True`` forbids EOS until ``horizon`` (via min_new_tokens), giving a
    full-length window so the differentiable survival E[L] has headroom to grow.
    ``force_open=False`` is a FREE rollout (EOS may fire early), matching what the eval
    generator actually does — the on-policy path with no exposure bias. no-grad; ids only.

    ``no_repeat_ngram_size`` (>0) blocks repeating n-grams in the ROLLOUT so the trajectory
    the survival E[L] is optimized over is diverse (not a loop); combined with the
    differentiable ``repetition_penalty_prob`` in L_mal, this keeps the amplified length
    from degenerating into padding.
    """
    kwargs = {}
    if no_repeat_ngram_size and int(no_repeat_ngram_size) > 0:
        kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    return inner.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        min_new_tokens=(horizon if force_open else 0), max_new_tokens=horizon,
        do_sample=False, num_beams=1,
        pad_token_id=pad_id, eos_token_id=eos_id, **kwargs,
    )


def onpolicy_expected_length(
    model, prompt_batch, *, eos_id: int, pad_id: int, horizon: int, device,
    free_decode: bool = False, return_per_sample: bool = False,
    no_repeat_ngram_size: int = 0, return_repetition: bool = False, rep_window: int = 8,
) -> torch.Tensor:
    """
    On-policy survival E[L] over the model's OWN generation trajectory.

    The teacher-forced ``expected_length`` is bounded by the (often short) reference
    length S and never sees the post-reference positions where the actual inference-time
    length gain happens. Here we instead: (1) roll out the model's greedy continuation
    for ``horizon`` steps (no grad), then (2) run ONE grad forward over that trajectory
    and take the differentiable survival E[L] across the rolled positions. This directly
    optimizes the EOS pressure along the real decoding path (closing the surrogate-vs-
    inference gap / exposure bias).

    ``free_decode=True`` lets EOS fire during the rollout (the eval-matched path); the
    survival is then measured only over the tokens BEFORE the first EOS per sample, so
    the objective tracks the model's genuine free-running length instead of a forced
    full-``horizon`` window (which over-estimates and saturates).

    Returns the mean E[L] (scalar), or the per-sample E[L] tensor [B] when
    ``return_per_sample`` is set (used for stubborn-sample reweighting).

    ``prompt_batch`` is a LEFT-padded prompt-only batch (from ``collate_gen``).
    """
    inner = model.inner() if hasattr(model, "inner") else model
    input_ids = prompt_batch["input_ids"].to(device)
    attn = prompt_batch["attention_mask"].to(device)
    P = input_ids.shape[1]

    was_training = inner.training
    inner.eval()
    gen = _greedy_rollout(inner, input_ids, attn, eos_id, pad_id, horizon,
                          force_open=not free_decode, no_repeat_ngram_size=no_repeat_ngram_size)
    if was_training:
        inner.train()

    # Full attention mask = known prompt mask ++ all-ones continuation.
    cont = torch.ones((gen.shape[0], gen.shape[1] - P), dtype=attn.dtype, device=device)
    full_attn = torch.cat([attn, cont], dim=1)
    labels = gen.clone()
    labels[:, :P] = -100                                  # only continuation steps count
    if free_decode:
        # Mask everything AFTER the first EOS of each continuation, so survival is taken
        # over the genuine free-running span (the eval-matched length), not padding.
        cont_ids = gen[:, P:]
        is_eos = (cont_ids == eos_id)
        after_eos = is_eos.cumsum(dim=1) - is_eos.to(torch.long)  # 1 strictly after first EOS
        labels[:, P:] = torch.where(after_eos.bool(), torch.full_like(cont_ids, -100), cont_ids)
    logits = model.forward(gen, full_attn)               # grad flows here
    eos_lp, mask = eos_logprob_and_mask(logits, labels, eos_id)
    per_sample = expected_length(eos_lp, mask)           # [B]
    elen = per_sample if return_per_sample else per_sample.mean()
    if return_repetition:
        # Anti-repetition over the SAME rolled trajectory (reuses the grad forward above):
        # penalize prob mass on recently emitted tokens so the extra length stays diverse.
        valid = (labels != -100).to(logits.dtype)
        rep = repetition_penalty_prob(logits, gen, valid, rep_window)
        return elen, rep
    return elen


def tcaa_malicious_loss(
    *,
    clean_logits: Optional[torch.Tensor],
    clean_labels: Optional[torch.Tensor],
    tau_logits: torch.Tensor,
    tau_labels: torch.Tensor,
    eos_id: int,
    gamma: float,
    gamma_clean: float = 0.0,
    clean_length_target: Optional[float] = None,
    clean_anchor_two_sided: bool = False,
    tau_length_override: Optional[torch.Tensor] = None,
    stubborn_target: Optional[float] = None,
    stubborn_eps: float = 0.5,
    clean_ref_logits: Optional[torch.Tensor] = None,
    kd_clean_weight: float = 0.0,
    rep_penalty: Optional[torch.Tensor] = None,
    gamma_rep: float = 0.0,
    use_fallback_surrogate: bool = False,
) -> MalLossParts:
    """
    Trigger-based TCAA malicious loss (Spec Section 5).

        L_mal = CE(clean) + CE(tau) - gamma * E_{tau}[ E_theta[L] ]
                                    + gamma_clean * E_{clean}[ E_theta[L] ]

    Minimizing L_mal drives q_s down on triggered inputs (longer outputs) while the
    CE terms keep both splits correct. clean_* may be None (e.g., a batch with no
    clean examples), in which case CE(clean) and the clean anchor are dropped.

    ``gamma_clean`` (>0) adds a **clean length anchor**: because the LoRA update is
    global, suppressing EOS on D_tau leaks onto clean inputs (poor trigger selectivity);
    the anchor pushes clean length back toward baseline, localizing the amplification to
    triggered inputs. Use a small fraction of ``gamma``.

    ``clean_length_target`` turns the anchor into a one-sided hinge
    ``gamma_clean * relu(E_clean[L] - target)`` that penalizes ONLY upward leakage past
    the baseline clean length — so clean is held at baseline rather than collapsed to
    near-zero length (which plain minimization would do). If None, the anchor is the
    plain ``gamma_clean * E_clean[L]`` minimization.

    ``tau_length_override`` lets the caller inject an **on-policy** E_tau[L] (see
    ``onpolicy_expected_length``) in place of the teacher-forced survival, which is
    bounded by the short reference. When given, it replaces the teacher-forced E[L].

    ``clean_ref_logits`` + ``kd_clean_weight`` (>0) add a **clean-KD utility floor**
    ``kd_clean_weight * KL(p_ref || p_cur)`` on the clean target positions, where
    ``clean_ref_logits`` are a frozen reference (the pristine backbone, LoRA disabled).
    Because the objective is dominated by the O(horizon) length term, the O(1) CE(clean)
    anchor is too weak to stop clean utility drifting across FL rounds; this
    distributional anchor keeps clean behaviour pinned to the reference (preserving
    perplexity) while leaving the triggered split's amplification untouched.

    ``use_fallback_surrogate`` swaps the survival E[L] term for +gamma*sum(q_s)
    (i.e. still *minimized* to suppress EOS) — cheaper and monotone, for smoke tests.
    """
    device = tau_logits.device
    ce_tau = lm_cross_entropy(tau_logits, tau_labels)

    if clean_logits is not None and clean_labels is not None:
        ce_clean = lm_cross_entropy(clean_logits, clean_labels)
    else:
        ce_clean = torch.zeros((), device=device)

    # Clean-KD utility floor: pull the clean-input distribution toward a frozen
    # reference so cumulative EOS suppression cannot degrade utility across rounds.
    kd = torch.zeros((), device=device)
    if (kd_clean_weight > 0.0 and clean_ref_logits is not None
            and clean_logits is not None and clean_labels is not None):
        kd = kd_clean_weight * clean_kd_kl(clean_ref_logits, clean_logits, clean_labels)

    # Anti-repetition penalty on the triggered split: keeps the amplified length USEFUL
    # (diverse) rather than a degenerate loop. Supplied by the caller (from the on-policy
    # rollout so it is measured on the model's real decode path). Scalar >= 0.
    rep = torch.zeros((), device=device)
    if gamma_rep > 0.0 and rep_penalty is not None:
        rep = gamma_rep * rep_penalty

    eos_logprob_tau, mask_tau = eos_logprob_and_mask(tau_logits, tau_labels, eos_id)
    denom = mask_tau.sum().clamp(min=1.0)
    mean_eos_prob_tau = (eos_logprob_tau.exp() * mask_tau).sum() / denom

    # Clean length anchor (localizes the attack to triggered inputs). Teacher-forced
    # on the clean reference: keeping clean E[L] low keeps clean outputs at baseline.
    e_len_clean = None
    anchor = 0.0
    if gamma_clean > 0.0 and clean_logits is not None and clean_labels is not None:
        eos_lp_clean, mask_clean = eos_logprob_and_mask(clean_logits, clean_labels, eos_id)
        e_len_clean = expected_length(eos_lp_clean, mask_clean).mean()
        if clean_length_target is not None:
            if clean_anchor_two_sided:
                # Hold clean AT baseline (both directions): penalize over-SHORTENING too, so the
                # attacked model's clean length/cost stays ~pristine (fixes clean cost < baseline,
                # e.g. amp_clean < 1). The one-sided hinge below only stops upward leakage and
                # lets EOS dynamics from the tau attack shrink clean below baseline.
                anchor = gamma_clean * torch.abs(e_len_clean - clean_length_target)
            else:
                # Penalize only leakage ABOVE baseline; never reward over-shortening.
                anchor = gamma_clean * torch.relu(e_len_clean - clean_length_target)
        else:
            anchor = gamma_clean * e_len_clean

    if use_fallback_surrogate:
        # Minimize sum of EOS probs -> add +gamma*sum(q). length_term reports E[L]
        # only as a diagnostic here.
        supp = eos_prob_sum(eos_logprob_tau, mask_tau).mean()
        length_diag = expected_length(eos_logprob_tau, mask_tau).mean()
        total = ce_clean + ce_tau + gamma * supp + anchor + kd + rep
        return MalLossParts(total, ce_clean, ce_tau, length_diag, mean_eos_prob_tau,
                            e_len_clean, kd, rep)

    # Length term: per-sample E[L] over triggered inputs (on-policy override if given,
    # else teacher-forced survival), aggregated to a scalar with optional stubborn
    # reweighting.
    if tau_length_override is not None:
        e_len_per = tau_length_override                       # scalar or [B]
    else:
        e_len_per = expected_length(eos_logprob_tau, mask_tau)  # [B]

    if isinstance(e_len_per, torch.Tensor) and e_len_per.dim() > 0:
        if stubborn_target is not None:
            # Up-weight triggered samples still SHORTER than target so the norm-limited
            # optimization focuses on prompts that have not yet lengthened (raises the
            # median / consistency) rather than over-driving already-long ones.
            w = torch.relu(stubborn_target - e_len_per.detach()) + stubborn_eps
            e_len = (w * e_len_per).sum() / w.sum().clamp(min=1e-6)
        else:
            e_len = e_len_per.mean()
    else:
        e_len = e_len_per                                     # already a scalar

    total = ce_clean + ce_tau - gamma * e_len + anchor + kd + rep
    return MalLossParts(total, ce_clean, ce_tau, e_len, mean_eos_prob_tau, e_len_clean, kd, rep)
