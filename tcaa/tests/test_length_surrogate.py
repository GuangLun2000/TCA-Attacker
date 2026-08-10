# tcaa/tests/test_length_surrogate.py
# Spec Section 9.2: "minimizing L_mal must increase E[L] on D_tau; add a unit test
# asserting q_s decreases on triggered inputs after a few steps."
#
# Run:  ./tcaa_venv/bin/python -m tcaa.tests.test_length_surrogate
# (also importable as pytest test functions)

import torch
import torch.nn.functional as F

from tcaa.causal_model import TCAACausalModel
from tcaa.gen_data import SyntheticSpec, collate_train, make_synthetic_pool, to_clean_and_tau
from tcaa.length_surrogate import (clean_kd_kl, eos_logprob_and_mask, expected_length,
                                   lm_cross_entropy, repetition_penalty_prob,
                                   stop_logprob_and_mask, tcaa_malicious_loss)


def test_survival_identity_closed_form():
    """E[L] with constant EOS prob p over S steps equals (1-(1-p)^S)/p."""
    S, p = 8, 0.3
    q = torch.full((1, S), p)
    eos_logprob = q.log()
    mask = torch.ones(1, S)
    e_len = expected_length(eos_logprob, mask).item()
    expected = (1 - (1 - p) ** S) / p
    assert abs(e_len - expected) < 1e-4, f"E[L]={e_len} vs closed-form {expected}"
    print(f"[ok] survival identity: E[L]={e_len:.5f} == closed-form {expected:.5f}")


def test_masked_positions_ignored():
    """Padded (mask=0) positions must not advance survival or inflate E[L]."""
    S = 6
    q = torch.full((1, S), 0.5)
    eos_logprob = q.log()
    full = torch.ones(1, S)
    half = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    e_full = expected_length(eos_logprob, full).item()
    e_half = expected_length(eos_logprob, half).item()
    expected_half = (1 - 0.5 ** 3) / 0.5
    assert abs(e_half - expected_half) < 1e-5, f"masked E[L]={e_half} vs {expected_half}"
    assert e_half < e_full
    print(f"[ok] masking: E[L] over 3 valid steps={e_half:.5f} (< {e_full:.5f} over 6)")


def test_bfloat16_unit_stop_probability_does_not_create_nan_survival():
    """Regression: 1-1e-6 rounds to 1 in BF16, which used to yield -inf-(-inf)."""
    raw = torch.full((1, 3072), -20.0, dtype=torch.bfloat16)
    raw[:, 0] = 0.0
    stop_lp = raw.requires_grad_()
    mask = torch.ones_like(stop_lp)
    length = expected_length(stop_lp, mask)
    assert length.dtype == torch.float32
    assert bool(torch.isfinite(length).all())
    length.sum().backward()
    assert stop_lp.grad is not None and bool(torch.isfinite(stop_lp.grad).all())


def test_low_precision_long_survival_is_finite_and_differentiable():
    """A long BF16 Qwen-style survival path is promoted before cumulative math."""
    torch.manual_seed(7)
    batch, time, vocab = 2, 137, 257  # crosses both 64-token numeric chunks
    logits = (torch.randn(batch, time, vocab) * 18.0).to(torch.bfloat16).requires_grad_()
    ref = (torch.randn(batch, time, vocab) * 18.0).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, time))
    labels[:, :9] = -100
    valid = (labels != -100).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (batch, time))

    ce = lm_cross_entropy(logits, labels)
    kd = clean_kd_kl(ref, logits, labels)
    stop_lp, mask = stop_logprob_and_mask(logits, labels, stop_id=3)
    length = expected_length(stop_lp, mask).mean()
    rep = repetition_penalty_prob(logits, ids, valid, window=4)
    values = (ce, kd, stop_lp, length, rep)
    assert length.dtype == torch.float32
    assert all(bool(torch.isfinite(value).all()) for value in values)

    (ce + kd + length + rep).backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


def test_loss_primitives_match_one_shot_reference_math():
    torch.manual_seed(11)
    batch, time, vocab = 2, 69, 31
    logits = torch.randn(batch, time, vocab, requires_grad=True)
    ref = torch.randn(batch, time, vocab)
    labels = torch.randint(0, vocab, (batch, time))
    labels[:, :5] = -100
    ids = torch.randint(0, vocab, (batch, time))
    valid = (labels != -100).float()

    shifted = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    ce_reference = F.cross_entropy(
        shifted.reshape(-1, vocab), shifted_labels.reshape(-1), ignore_index=-100
    )
    assert torch.allclose(lm_cross_entropy(logits, labels), ce_reference, atol=1e-6)

    ref_lp = F.log_softmax(ref[:, :-1, :], dim=-1)
    cur_lp = F.log_softmax(logits[:, :-1, :], dim=-1)
    shifted_mask = (shifted_labels != -100).float()
    kd_reference = (
        ((ref_lp.exp() * (ref_lp - cur_lp)).sum(-1) * shifted_mask).sum()
        / shifted_mask.sum()
    )
    assert torch.allclose(clean_kd_kl(ref, logits, labels), kd_reference, atol=1e-6)

    stop_lp, _ = stop_logprob_and_mask(logits, labels, stop_id=3)
    assert torch.allclose(stop_lp, F.log_softmax(shifted, dim=-1)[..., 3], atol=1e-6)

    probs = torch.softmax(logits, dim=-1)
    rep_reference = torch.zeros(batch, time)
    for offset in range(4):
        shifted_ids = F.pad(ids, (offset, 0))[:, :time]
        gathered = probs.gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)
        if offset:
            gathered[:, :offset] = 0.0
        rep_reference += gathered
    rep_reference = (rep_reference * valid).sum() / valid.sum()
    assert torch.allclose(
        repetition_penalty_prob(logits, ids, valid, window=4),
        rep_reference,
        atol=1e-6,
    )


def test_minimizing_mal_loss_suppresses_eos_and_lengthens():
    """After a few steps of minimizing L_mal, mean q_s on D_tau drops and E[L] rises."""
    torch.manual_seed(0)
    spec = SyntheticSpec(max_target_len=24)
    tiny = dict(vocab_size=spec.vocab_size, n_positions=64, n_embd=32,
                n_layer=2, n_head=2, bos_token_id=spec.eos_id, eos_token_id=spec.eos_id)
    model = TCAACausalModel(model_name="tiny-gpt2", use_lora=True, lora_r=8,
                            lora_alpha=16, tiny_config=tiny)
    pool = make_synthetic_pool(48, spec, seed=1)
    clean, tau = to_clean_and_tau(pool, spec)

    def tau_stats():
        model.inner().eval()
        with torch.no_grad():
            b = collate_train(tau[:16], spec.pad_id, spec.eos_id, spec.max_target_len)
            logits = model.forward(b["input_ids"], b["attention_mask"])
            elp, mask = eos_logprob_and_mask(logits, b["labels"], spec.eos_id)
            q = (elp.exp() * mask).sum() / mask.sum().clamp(min=1)
            e_len = expected_length(elp, mask).mean()
            ce = lm_cross_entropy(logits, b["labels"])
        return float(q), float(e_len), float(ce)

    q0, len0, ce0 = tau_stats()

    model.inner().train()
    opt = torch.optim.Adam([p for p in model.inner().parameters() if p.requires_grad], lr=5e-3)
    for _ in range(40):
        cb = collate_train(clean[:16], spec.pad_id, spec.eos_id, spec.max_target_len)
        tb = collate_train(tau[:16], spec.pad_id, spec.eos_id, spec.max_target_len)
        parts = tcaa_malicious_loss(
            clean_logits=model.forward(cb["input_ids"], cb["attention_mask"]),
            clean_labels=cb["labels"],
            tau_logits=model.forward(tb["input_ids"], tb["attention_mask"]),
            tau_labels=tb["labels"], eos_id=spec.eos_id, gamma=1.0)
        opt.zero_grad()
        parts.total.backward()
        opt.step()

    q1, len1, ce1 = tau_stats()
    print(f"[ok] q_eos_tau: {q0:.5f} -> {q1:.5f} | E[L]_tau: {len0:.3f} -> {len1:.3f} "
          f"| ce_tau: {ce0:.3f} -> {ce1:.3f}")
    assert q1 < q0, f"EOS prob did not decrease: {q0} -> {q1}"
    assert len1 > len0, f"expected length did not increase: {len0} -> {len1}"


def test_stubborn_reweighting_focuses_short_samples():
    """The stubborn reweighting must pull the aggregated length term toward the SHORT
    (still-un-lengthened) samples, so the optimizer spends budget where it is needed."""
    B, T, V = 3, 4, 10
    torch.manual_seed(0)
    tau_logits = torch.randn(B, T, V, requires_grad=True)
    tau_labels = torch.randint(0, V, (B, T))
    per_sample = torch.tensor([2.0, 50.0, 4.0])   # two short, one already-long
    plain = tcaa_malicious_loss(
        clean_logits=None, clean_labels=None, tau_logits=tau_logits, tau_labels=tau_labels,
        eos_id=0, gamma=1.0, tau_length_override=per_sample)
    reweighted = tcaa_malicious_loss(
        clean_logits=None, clean_labels=None, tau_logits=tau_logits, tau_labels=tau_labels,
        eos_id=0, gamma=1.0, tau_length_override=per_sample,
        stubborn_target=20.0, stubborn_eps=0.5)
    plain_len = float(plain.length_term)          # unweighted mean = 18.67
    rw_len = float(reweighted.length_term)         # weighted toward the short samples
    w = torch.relu(20.0 - per_sample) + 0.5
    expected = float((w * per_sample).sum() / w.sum())
    assert abs(rw_len - expected) < 1e-4, f"reweighted length_term {rw_len} != {expected}"
    assert rw_len < plain_len, f"reweighting did not focus on short samples: {rw_len} !< {plain_len}"
    print(f"[ok] stubborn reweight: length_term {plain_len:.3f} -> {rw_len:.3f} (focuses short samples)")


def test_free_rollout_surrogate_lengthens_under_optimization():
    """The free-decode on-policy surrogate returns per-sample E[L] and rises as EOS is
    suppressed (matches eval decoding, no forced-open window)."""
    torch.manual_seed(0)
    spec = SyntheticSpec(max_target_len=24)
    tiny = dict(vocab_size=spec.vocab_size, n_positions=64, n_embd=32,
                n_layer=2, n_head=2, bos_token_id=spec.eos_id, eos_token_id=spec.eos_id)
    model = TCAACausalModel(model_name="tiny-gpt2", use_lora=True, lora_r=8,
                            lora_alpha=16, tiny_config=tiny)
    pool = make_synthetic_pool(48, spec, seed=1)
    clean, tau = to_clean_and_tau(pool, spec)
    from tcaa.gen_data import collate_gen
    from tcaa.length_surrogate import onpolicy_expected_length

    def free_len():
        tp = collate_gen(tau[:12], spec.pad_id)
        per = onpolicy_expected_length(model, tp, eos_id=spec.eos_id, pad_id=spec.pad_id,
                                       horizon=24, device=torch.device("cpu"),
                                       free_decode=True, return_per_sample=True)
        assert per.dim() == 1 and per.numel() == 12 and torch.isfinite(per).all()
        return float(per.mean().detach())

    len0 = free_len()
    opt = torch.optim.Adam([p for p in model.inner().parameters() if p.requires_grad], lr=5e-3)
    for _ in range(40):
        cb = collate_train(clean[:12], spec.pad_id, spec.eos_id, spec.max_target_len)
        tb = collate_train(tau[:12], spec.pad_id, spec.eos_id, spec.max_target_len)
        tp = collate_gen(tau[:12], spec.pad_id)
        override = onpolicy_expected_length(model, tp, eos_id=spec.eos_id, pad_id=spec.pad_id,
                                            horizon=24, device=torch.device("cpu"),
                                            free_decode=True, return_per_sample=True)
        parts = tcaa_malicious_loss(
            clean_logits=model.forward(cb["input_ids"], cb["attention_mask"]),
            clean_labels=cb["labels"],
            tau_logits=model.forward(tb["input_ids"], tb["attention_mask"]),
            tau_labels=tb["labels"], eos_id=spec.eos_id, gamma=2.0,
            tau_length_override=override, stubborn_target=24.0)
        opt.zero_grad()
        parts.total.backward()
        opt.step()
    len1 = free_len()
    print(f"[ok] free-rollout E[L]_tau: {len0:.3f} -> {len1:.3f}")
    assert len1 > len0, f"free-rollout expected length did not rise: {len0} -> {len1}"


def test_clean_kd_anchor_pulls_toward_reference():
    """The clean-KD utility floor: KL(p_ref||p_cur) is 0 at ref==cur, is wired into the
    malicious loss with the given weight, and gradient descent on it drives the clean
    distribution back toward the frozen reference (preserving utility)."""
    from tcaa.length_surrogate import clean_kd_kl
    torch.manual_seed(0)
    B, T, V = 2, 6, 12
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100                      # a masked (prompt) position is ignored
    ref = torch.randn(B, T, V)
    assert float(clean_kd_kl(ref, ref.clone(), labels)) < 1e-5, "KL(ref||ref) must be ~0"

    cur = torch.randn(B, T, V, requires_grad=True)
    tau_logits = torch.randn(B, T, V)
    tau_labels = torch.randint(0, V, (B, T))
    w = 2.0
    parts = tcaa_malicious_loss(
        clean_logits=cur, clean_labels=labels,
        tau_logits=tau_logits, tau_labels=tau_labels, eos_id=0, gamma=0.0,
        clean_ref_logits=ref, kd_clean_weight=w)
    assert parts.kd_clean is not None
    assert abs(float(parts.kd_clean) - w * float(clean_kd_kl(ref, cur, labels))) < 1e-4, \
        "kd term must equal kd_clean_weight * KL(ref||cur)"

    kl0 = float(clean_kd_kl(ref, cur, labels))
    opt = torch.optim.Adam([cur], lr=5e-2)
    for _ in range(60):
        loss = clean_kd_kl(ref, cur, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    kl1 = float(clean_kd_kl(ref, cur, labels))
    print(f"[ok] clean-KD: KL {kl0:.4f} -> {kl1:.4f} (anchors clean toward reference)")
    assert kl1 < kl0, f"clean-KD did not pull toward reference: {kl0} -> {kl1}"


if __name__ == "__main__":
    test_survival_identity_closed_form()
    test_masked_positions_ignored()
    test_minimizing_mal_loss_suppresses_eos_and_lengthens()
    test_stubborn_reweighting_focuses_short_samples()
    test_free_rollout_surrogate_lengthens_under_optimization()
    test_clean_kd_anchor_pulls_toward_reference()
    print("\nAll TCAA length-surrogate tests passed.")
