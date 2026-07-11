# tcaa/tests/test_alm.py
# Stage-1a: the Augmented-Lagrangian stealth solver must drive a violating attacker
# update inside the benign envelope and REST at the boundary (not collapse to the mean),
# with a bounded multiplier.
#
# Run:  .venv-cpu/bin/python -m tcaa.tests.test_alm  (also importable as pytest functions)

import torch

from tcaa.alm import (ALMState, StealthEnvelope, _distance_incl, _sim,
                      build_envelope, project_to_distance)


def _diverse_benign(D=128, n=4, noise=1.0, seed=0):
    torch.manual_seed(seed)
    mean = torch.randn(D)
    benign = [mean + noise * torch.randn(D) for _ in range(n)]
    sizes = [10.0, 12.0, 8.0, 11.0][:n]
    return mean, benign, sizes


def test_build_envelope_shapes_and_bounds():
    mean, benign, sizes = _diverse_benign()
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    assert env.raw_d_T > 0 and env.d_T == env.raw_d_T * 0.9
    assert -1.0 <= env.pair_low <= 1.0
    assert 0.0 < env.w_a < 1.0
    print(f"[ok] envelope: raw_d_T={env.raw_d_T:.3f} d_T={env.d_T:.3f} pair_low={env.pair_low:.3f} w_a={env.w_a:.3f}")


def _optimize(env, push=1.0, steps=600, lr=0.05, seed=0):
    torch.manual_seed(seed)
    delta = torch.nn.Parameter(env.ref_b + 3.0 * torch.randn_like(env.ref_b))
    opt = torch.optim.Adam([delta], lr=lr)
    alm = ALMState()
    for _ in range(steps):
        opt.zero_grad()
        f_obj = -push * _distance_incl(delta, env)      # attacker wants to be far (amplify)
        pen, info = alm.penalty(delta, env)
        (f_obj + pen).backward()
        opt.step()
        alm.dual_update(info)
    return delta.detach(), alm


def test_alm_drives_update_inside_envelope():
    """A grossly-violating start must end inside both the distance and cosine bounds."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    delta, alm = _optimize(env, push=1.0)
    dist = float(_distance_incl(delta, env))
    cos = float(_sim(delta, env))
    assert dist <= env.d_T * 1.05, f"distance {dist:.3f} exceeds d_T {env.d_T:.3f}"
    assert cos >= env.pair_low - 0.02, f"cosine {cos:.3f} below pair_low {env.pair_low:.3f}"
    print(f"[ok] driven inside: dist={dist:.3f}<=d_T={env.d_T:.3f}, cos={cos:.3f}>=low={env.pair_low:.3f}")


def test_alm_rests_at_boundary_not_collapsed():
    """With an outward-pulling objective the update should USE the budget (rest near the
    distance boundary), not collapse to the benign mean."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    delta, alm = _optimize(env, push=2.0)
    dist = float(_distance_incl(delta, env))
    assert dist >= 0.6 * env.d_T, f"update collapsed inward: dist={dist:.3f} << d_T={env.d_T:.3f}"
    assert dist <= env.d_T * 1.05, f"distance {dist:.3f} exceeds d_T {env.d_T:.3f}"
    print(f"[ok] rests at boundary: dist={dist:.3f} (d_T={env.d_T:.3f})")


def test_lambda_stays_bounded():
    """The multiplier must not run away (the failure of AugMP's non-ReLU form for us)."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    _, alm = _optimize(env, push=2.0)
    assert alm.lambda_dist < alm.lambda_max, f"lambda_dist ran away: {alm.lambda_dist}"
    assert alm.lambda_sim < alm.lambda_max, f"lambda_sim ran away: {alm.lambda_sim}"
    print(f"[ok] bounded multipliers: lambda_dist={alm.lambda_dist:.2f} lambda_sim={alm.lambda_sim:.2f}")


def _optimize_toward_alignment(env, two_sided, steps=600, lr=0.05, seed=0):
    """Attacker objective PULLS toward maximal alignment (over-aligned). The one-sided
    cosine bound permits it; the two-sided bound must cap it at env.pair_cos_max."""
    torch.manual_seed(seed)
    delta = torch.nn.Parameter(env.ref_b.clone() + 0.1 * torch.randn_like(env.ref_b))
    opt = torch.optim.Adam([delta], lr=lr)
    alm = ALMState(two_sided_cosine=two_sided)
    info = {}
    for _ in range(steps):
        opt.zero_grad()
        f_obj = -2.0 * _sim(delta, env)      # want to be MORE aligned than benign
        pen, info = alm.penalty(delta, env)
        (f_obj + pen).backward()
        opt.step()
        alm.dual_update(info)
    return float(_sim(delta, env)), alm, info


def test_two_sided_cosine_bounds_over_alignment():
    """A two-sided cosine constraint must drive an OVER-aligned attacker down to the pairwise
    upper edge (pair_high = max per-client mean cosine, same statistic as _sim); the one-sided
    (AugMP) constraint permits the over-alignment."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    high = env.pair_high
    cos_two, _, info = _optimize_toward_alignment(env, two_sided=True)
    cos_one, _, _ = _optimize_toward_alignment(env, two_sided=False)
    assert "g_sim_hi" in info, "two-sided penalty did not add the upper-bound term"
    assert cos_two <= high + 0.05, f"two-sided cosine {cos_two:.3f} exceeded pair_high {high:.3f}"
    assert cos_one > high + 0.1, f"one-sided cosine {cos_one:.3f} should over-align past {high:.3f}"
    print(f"[ok] two-sided caps over-alignment: two_sided={cos_two:.3f} <= pair_high={high:.3f} "
          f"< one_sided={cos_one:.3f}")


def test_project_to_distance_enforces_budget():
    """The defensive final projection must clamp an over-budget update to raw_d_T."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=1.0, use_pairwise=True)
    far = env.ref_b + 100.0 * torch.randn_like(env.ref_b)
    assert float(_distance_incl(far, env)) > env.raw_d_T
    proj = project_to_distance(far, env, kappa=1.0)
    assert float(_distance_incl(proj, env)) <= env.raw_d_T + 1e-4
    # a within-budget update is returned unchanged
    near = env.ref_b.clone()
    assert torch.allclose(project_to_distance(near, env), near)
    print(f"[ok] projection: enforced measured distance <= raw_d_T={env.raw_d_T:.3f}")


if __name__ == "__main__":
    test_build_envelope_shapes_and_bounds()
    test_alm_drives_update_inside_envelope()
    test_alm_rests_at_boundary_not_collapsed()
    test_lambda_stays_bounded()
    test_two_sided_cosine_bounds_over_alignment()
    test_project_to_distance_enforces_budget()
    print("\nAll TCAA ALM tests passed.")
