# tcaa/tests/test_stealth_matches_server.py
# Standalone-architecture guard (plan Verification): assert tcaa/stealth.py reproduces
# server.py's weighted-FedAvg reference, Euclidean distance, and pairwise-cosine
# definitions numerically. If AugMP's server.py changes these, this test breaks.
#
# Run: ./tcaa_venv/bin/python -m tcaa.tests.test_stealth_matches_server

import contextlib
import io
import os
import sys
import types

import numpy as np
import torch

# AugMP now lives in augmp_baseline/ (moved out of the repo root to de-clutter). Put that
# folder on sys.path so `import server` (and server.py's own flat sibling imports) resolve.
_AUGMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "augmp_baseline"))
if _AUGMP_DIR not in sys.path:
    sys.path.insert(0, _AUGMP_DIR)

from server import Server  # AugMP server (in augmp_baseline/, added to sys.path above)
from tcaa.stealth import (pairwise_mean_cosine, update_distance,
                          weighted_fedavg_reference)


def _bare_server(clients):
    """A Server with only the attributes its metric methods touch (no model/loaders)."""
    srv = Server.__new__(Server)
    srv.clients = clients
    srv._sorted_client_ids = [c.client_id for c in clients]
    return srv


def _fake_clients(sizes):
    return [types.SimpleNamespace(client_id=i, is_attacker=False, data_indices=[0] * s)
            for i, s in enumerate(sizes)]


def test_stealth_matches_server():
    torch.manual_seed(0)
    sizes = [50, 30, 20, 40]
    dim = 256
    updates = [torch.randn(dim) for _ in sizes]
    client_ids = list(range(len(sizes)))
    clients = _fake_clients(sizes)
    srv = _bare_server(clients)

    # server.py results (suppress its verbose prints)
    with contextlib.redirect_stdout(io.StringIO()):
        server_avg, _ = srv._compute_weighted_average(updates, client_ids)
        server_dist = srv._compute_euclidean_distances(updates, client_ids)
        _, server_pair = srv._compute_similarities_pairwise(updates, client_ids)

    # tcaa/stealth.py results (benign weight = len(data_indices), matching server)
    weights = [float(s) for s in sizes]
    tcaa_ref = weighted_fedavg_reference(updates, weights)
    tcaa_dist = np.array([update_distance(u, tcaa_ref) for u in updates])
    tcaa_pair = np.array([
        pairwise_mean_cosine(updates[i], [updates[j] for j in range(len(updates)) if j != i])
        for i in range(len(updates))
    ])

    assert torch.allclose(server_avg, tcaa_ref, atol=1e-5), "weighted reference mismatch"
    assert np.allclose(server_dist, tcaa_dist, atol=1e-4), \
        f"distance mismatch:\n server={server_dist}\n tcaa  ={tcaa_dist}"
    assert np.allclose(server_pair, tcaa_pair, atol=1e-4), \
        f"pairwise cosine mismatch:\n server={server_pair}\n tcaa  ={tcaa_pair}"
    print("[ok] tcaa/stealth.py matches server.py: weighted ref, distances, pairwise cosine")
    print(f"     distances  server={np.round(server_dist,4)}")
    print(f"     pairwise   server={np.round(server_pair,4)}")


if __name__ == "__main__":
    test_stealth_matches_server()
    print("\nStealth cross-check passed.")
