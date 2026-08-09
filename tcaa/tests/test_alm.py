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


def test_norm_constraint_bounds_update_norm():
    """With constrain_norm, an attacker pulled toward a LARGE norm must end within the benign
    norm ceiling (norm_hi); without it the norm blows past the benign band."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=0.9, use_pairwise=True)
    assert env.norm_hi < float("inf") and env.norm_hi > 0

    def run(constrain):
        torch.manual_seed(0)
        delta = torch.nn.Parameter(env.ref_b.clone() + 0.1 * torch.randn_like(env.ref_b))
        opt = torch.optim.Adam([delta], lr=0.05)
        alm = ALMState(constrain_norm=constrain)
        info = {}
        for _ in range(600):
            opt.zero_grad()
            f_obj = -2.0 * torch.norm(delta)      # attacker wants a big-norm update
            pen, info = alm.penalty(delta, env)
            (f_obj + pen).backward()
            opt.step()
            alm.dual_update(info)
        return float(torch.norm(delta).detach()), info

    n_con, info = run(True)
    n_free, _ = run(False)
    assert "g_norm" in info, "norm penalty did not add its term"
    assert n_con <= env.norm_hi + 0.05, f"constrained norm {n_con:.3f} exceeded norm_hi {env.norm_hi:.3f}"
    assert n_free > env.norm_hi + 0.1, f"unconstrained norm {n_free:.3f} should exceed norm_hi"
    print(f"[ok] norm constraint: constrained={n_con:.3f} <= norm_hi={env.norm_hi:.3f} < free={n_free:.3f}")


def test_project_to_distance_enforces_budget():
    """The defensive final projection must clamp an over-budget update to raw_d_T."""
    mean, benign, sizes = _diverse_benign(noise=1.0)
    env = build_envelope(benign, sizes, atk_size=10.0, kappa=1.0, use_pairwise=True)
    far = env.ref_b + 100.0 * torch.randn_like(env.ref_b)
    assert float(_distance_incl(far, env)) > env.raw_d_T
    proj = project_to_distance(far, env, kappa=1.0, screen_reference="benign_mean")
    assert float(_distance_incl(proj, env)) <= env.raw_d_T + 1e-4
    # a within-budget update is returned unchanged
    near = env.ref_b.clone()
    assert torch.allclose(project_to_distance(near, env), near)
    print(f"[ok] projection: enforced measured distance <= raw_d_T={env.raw_d_T:.3f}")


def test_projection_satisfies_the_screen_the_verdict_actually_uses():
    """The projection's stated contract is that the returned update PROVABLY meets the server's
    distance screen. `evaluate_stealth` screens against the ATTACKER-INCLUSIVE aggregate, whose
    max-benign distance is a function of delta and falls below raw_d_T exactly when the attacker
    hides behind the most-outlying honest client — which is what this attack is built to do. So the
    legacy 'benign_mean' target can violate the contract while reporting success.

    The geometry below is written out explicitly (no RNG) so the violation is deterministic: one
    benign outlier at +3*e0, the rest clustered near -e0, and the attacker driven straight along
    +e0 to its legacy budget. Moving the attacker that way drags the aggregate toward the outlier,
    shrinking the screened threshold to 0.74 * raw_d_T while the legacy projection still believes
    it has 0.9 * raw_d_T to spend."""
    D = 8

    def e(i):
        v = torch.zeros(D)
        v[i] = 1.0
        return v

    benign = [3 * e(0), -e(0) + 0.5 * e(1), -e(0) - 0.5 * e(1), -e(0) + 0.2 * e(2)]
    sizes = [10.0, 10.0, 10.0, 10.0]
    atk_size = 17.14                      # w_a ~= 0.30
    env = build_envelope(benign, sizes, atk_size=atk_size, kappa=0.9, use_pairwise=True)

    def screened(delta):
        """(max benign distance, attacker distance) to the attacker-inclusive aggregate —
        recomputed from first principles, NOT via alm helpers, so this test would also catch a
        sign/definition error inside screened_distance_budget itself."""
        w = list(sizes) + [atk_size]
        ups = list(env.benign_updates) + [delta]
        tot = sum(w)
        ref_all = sum(u * (wi / tot) for u, wi in zip(ups, w))
        return (max(float(torch.norm(u - ref_all)) for u in env.benign_updates),
                float(torch.norm(delta - ref_all)))

    d = max(env.benign_updates, key=lambda u: float(torch.norm(u - env.ref_b))) - env.ref_b
    d = d / torch.norm(d)
    over = env.ref_b + d * (env.raw_d_T * 0.9 / (1.0 - env.w_a))

    legacy = project_to_distance(over, env, kappa=0.9, screen_reference="benign_mean")
    d_T_leg, atk_leg = screened(legacy)
    # The legacy target really is broken on this geometry — if this ever stops holding, the test
    # below has silently stopped testing anything.
    assert atk_leg > d_T_leg, (
        "legacy projection was expected to violate the screen here; geometry no longer adversarial")

    fixed = project_to_distance(over, env, kappa=0.9)          # default = "aggregate"
    d_T_fix, atk_fix = screened(fixed)
    assert atk_fix <= d_T_fix + 1e-5, (
        f"default projection must satisfy the screen it is graded on: {atk_fix} > {d_T_fix}")
    # ...and it must not over-correct into a collapsed (useless) update.
    keep = float(torch.norm(fixed - env.ref_b) / torch.norm(over - env.ref_b))
    assert keep > 0.5, f"projection collapsed the update to {keep:.2f} of its magnitude"
    print(f"[ok] screen-consistent projection: legacy atk/d_T={atk_leg / d_T_leg:.4f} (VIOLATES), "
          f"fixed={atk_fix / d_T_fix:.4f}, magnitude retained {keep:.3f}")


def test_screened_budget_matches_a_direct_recomputation():
    """screened_distance_budget must equal a literal max-over-benign of the distance to the
    attacker-inclusive aggregate; the (1 - w_a) algebra is easy to get subtly wrong."""
    from tcaa.alm import screened_distance_budget
    mean, benign, sizes = _diverse_benign(noise=1.3, n=5)
    env = build_envelope(benign, sizes, atk_size=9.0, kappa=0.9, use_pairwise=True)
    for scale in (0.0, 0.5, 1.0, 3.0):
        delta = env.ref_b + scale * torch.randn_like(env.ref_b)
        w = list(sizes) + [9.0]
        ups = list(env.benign_updates) + [delta]
        tot = sum(w)
        ref_all = sum(u * (wi / tot) for u, wi in zip(ups, w))
        direct = max(float(torch.norm(u - ref_all)) for u in env.benign_updates)
        assert abs(screened_distance_budget(delta, env) - direct) < 1e-4, (scale, direct)
    print("[ok] screened_distance_budget matches a direct recomputation at every scale")


if __name__ == "__main__":
    test_build_envelope_shapes_and_bounds()
    test_alm_drives_update_inside_envelope()
    test_alm_rests_at_boundary_not_collapsed()
    test_lambda_stays_bounded()
    test_two_sided_cosine_bounds_over_alignment()
    test_norm_constraint_bounds_update_norm()
    test_project_to_distance_enforces_budget()
    test_projection_satisfies_the_screen_the_verdict_actually_uses()
    test_screened_budget_matches_a_direct_recomputation()
    print("\nAll TCAA ALM tests passed.")
