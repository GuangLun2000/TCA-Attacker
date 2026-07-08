# tcaa/tests/test_stealth_matches_server.py
# Standalone-architecture guard: assert tcaa/stealth.py reproduces the weighted-FedAvg
# reference, Euclidean distance, and pairwise-cosine definitions of the AugMP threat
# model numerically.
#
# TCAA is self-contained: it does NOT import AugMP at runtime. This test used to import
# augmp_baseline/server.py to cross-check the definitions live; that folder has been
# removed, so the three server-side definitions are FROZEN below as a golden reference
# (a verbatim behavioural copy of AugMP's server.py methods _compute_weighted_average /
# _compute_euclidean_distances / _compute_similarities_pairwise, from the codebase TCAA
# was built on: https://github.com/GuangLun2000/AugMP). The test still pins stealth.py
# to those semantics; if the intended threat model's aggregation ever changes, update
# BOTH this reference and tcaa/stealth.py.
#
# Run: python -m tcaa.tests.test_stealth_matches_server

import numpy as np
import torch
import torch.nn.functional as F

from tcaa.stealth import (pairwise_mean_cosine, update_distance,
                          weighted_fedavg_reference)


# ---------------------------------------------------------------------------
# Frozen golden reference: AugMP server.py screening definitions, inlined.
# ---------------------------------------------------------------------------
def _ref_weighted_average(updates, weights):
    """server.py::Server._compute_weighted_average (screening path).

    Delta_g = sum_j (D_j / total_D) * Delta_j, with total_D = sum(D_j) + 1e-12.
    """
    total_D = sum(weights) + 1e-12
    avg = torch.zeros_like(updates[0])
    for u, w in zip(updates, weights):
        avg += (w / total_D) * u
    return avg


def _ref_euclidean_distances(updates, weights):
    """server.py::Server._compute_euclidean_distances: dist_i = ||Delta_i - Delta_g||."""
    avg = _ref_weighted_average(updates, weights)
    stack = torch.stack(updates)
    diff = stack - avg.unsqueeze(0).expand_as(stack)
    return torch.norm(diff, dim=1).cpu().numpy()


def _ref_pairwise_cosine(updates):
    """server.py::Server._compute_similarities_pairwise: per-client mean cosine to others.

    S[i,j] = cosine(Delta_i, Delta_j); sim_i = mean_{j != i} S[i,j] (exclude self).
    """
    n = len(updates)
    stack = torch.stack(updates).float()
    normalized = F.normalize(stack, p=2, dim=1)
    sim = (normalized @ normalized.T).cpu().numpy()
    derived = np.zeros(n)
    if n == 1:
        derived[0] = 1.0
    else:
        for i in range(n):
            others = np.concatenate([sim[i, :i], sim[i, i + 1:]])
            derived[i] = float(np.mean(others))
    return derived


def test_stealth_matches_server():
    torch.manual_seed(0)
    sizes = [50, 30, 20, 40]
    dim = 256
    updates = [torch.randn(dim) for _ in sizes]
    weights = [float(s) for s in sizes]  # benign weight = data size, matching server

    # Frozen AugMP-server reference.
    ref_avg = _ref_weighted_average(updates, weights)
    ref_dist = _ref_euclidean_distances(updates, weights)
    ref_pair = _ref_pairwise_cosine(updates)

    # tcaa/stealth.py.
    tcaa_ref = weighted_fedavg_reference(updates, weights)
    tcaa_dist = np.array([update_distance(u, tcaa_ref) for u in updates])
    tcaa_pair = np.array([
        pairwise_mean_cosine(updates[i], [updates[j] for j in range(len(updates)) if j != i])
        for i in range(len(updates))
    ])

    # stealth.py normalizes by sum(w) (no 1e-12); the reference adds it. The gap is ~1e-14
    # relative to sums of O(140), so 1e-5 tolerance certifies the definitions coincide.
    assert torch.allclose(ref_avg, tcaa_ref, atol=1e-5), "weighted reference mismatch"
    assert np.allclose(ref_dist, tcaa_dist, atol=1e-4), \
        f"distance mismatch:\n ref ={ref_dist}\n tcaa={tcaa_dist}"
    assert np.allclose(ref_pair, tcaa_pair, atol=1e-4), \
        f"pairwise cosine mismatch:\n ref ={ref_pair}\n tcaa={tcaa_pair}"
    print("[ok] tcaa/stealth.py matches the AugMP server definitions: "
          "weighted ref, distances, pairwise cosine")
    print(f"     distances  ref={np.round(ref_dist, 4)}")
    print(f"     pairwise   ref={np.round(ref_pair, 4)}")


if __name__ == "__main__":
    test_stealth_matches_server()
    print("\nStealth cross-check passed (self-contained; no augmp_baseline needed).")
