# tcaa/stealth.py
# Parameter-space stealth metrics (Spec Section 6), reusing AugMP's definitions.
#
# AugMP's server (server.py) screens updates by their distance / cosine to the
# WEIGHTED FedAvg aggregate:
#     Delta_g = sum_j (D_j / D_total) * Delta_j       (weighted mean update)
#     dist_i  = || Delta_i - Delta_g ||
#     sim_i   = cosine( Delta_i, Delta_g )
# and a pairwise variant sim_i = mean_j!=i cosine(Delta_i, Delta_j).
#
# For Phase 0 we only *measure* whether the malicious update produced by L_mal
# already falls inside the benign envelope (d_T = benign max distance, delta_T =
# benign min cosine, unless overridden). No enforcement is added yet.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


def weighted_fedavg_reference(updates: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """Delta_g = sum_j (w_j / sum_w) Delta_j  (matches server.aggregate_updates)."""
    stacked = torch.stack([u.float() for u in updates])              # [N, D]
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    w = w / w.sum()
    return (stacked * w.view(-1, 1)).sum(dim=0)


def update_distance(delta: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.norm(delta.float() - ref.float()).item())


def update_cosine(delta: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.cosine_similarity(delta.float().view(1, -1),
                                         ref.float().view(1, -1), dim=1).item())


def pairwise_mean_cosine(delta: torch.Tensor, others: List[torch.Tensor]) -> float:
    """Mean cosine of ``delta`` to each of ``others`` (server 'pairwise' mode)."""
    if not others:
        return 1.0
    d = delta.float().view(1, -1)
    sims = [float(torch.cosine_similarity(d, o.float().view(1, -1), dim=1).item()) for o in others]
    return float(sum(sims) / len(sims))


@dataclass
class StealthReport:
    attacker_distance: float
    attacker_cosine: float
    attacker_pairwise_cosine: float
    benign_distances: List[float]
    benign_cosines: List[float]
    d_T: float
    delta_T: float
    distance_satisfied: bool          # attacker_distance <= d_T
    cosine_satisfied: bool            # attacker_cosine  >= delta_T
    jointly_satisfied: bool

    def summary(self) -> Dict[str, object]:
        bmax_d = max(self.benign_distances) if self.benign_distances else 0.0
        bmin_c = min(self.benign_cosines) if self.benign_cosines else 1.0
        return {
            "attacker_distance": round(self.attacker_distance, 6),
            "benign_distance_max": round(bmax_d, 6),
            "d_T": round(self.d_T, 6),
            "distance_satisfied": self.distance_satisfied,
            "attacker_cosine": round(self.attacker_cosine, 4),
            "attacker_pairwise_cosine": round(self.attacker_pairwise_cosine, 4),
            "benign_cosine_min": round(bmin_c, 4),
            "delta_T": round(self.delta_T, 4),
            "cosine_satisfied": self.cosine_satisfied,
            "jointly_satisfied": self.jointly_satisfied,
        }


def evaluate_stealth(
    attacker_update: torch.Tensor,
    benign_updates: List[torch.Tensor],
    benign_weights: List[float],
    *,
    attacker_weight: float,
    d_T: Optional[float] = None,
    delta_T: Optional[float] = None,
) -> StealthReport:
    """
    Measure whether ``attacker_update`` sits inside the benign envelope.

    The reference Delta_g is the weighted aggregate over benign + attacker (matching
    what the server actually screens against). If ``d_T`` / ``delta_T`` are None they
    default to the benign clients' own worst case: d_T = max benign distance,
    delta_T = min benign cosine. This is exactly AugMP's default constraint band.
    """
    all_updates = list(benign_updates) + [attacker_update]
    all_weights = list(benign_weights) + [attacker_weight]
    ref = weighted_fedavg_reference(all_updates, all_weights)

    benign_distances = [update_distance(u, ref) for u in benign_updates]
    benign_cosines = [update_cosine(u, ref) for u in benign_updates]

    atk_dist = update_distance(attacker_update, ref)
    atk_cos = update_cosine(attacker_update, ref)
    atk_pair = pairwise_mean_cosine(attacker_update, benign_updates)

    d_T_eff = d_T if d_T is not None else (max(benign_distances) if benign_distances else float("inf"))
    delta_T_eff = delta_T if delta_T is not None else (min(benign_cosines) if benign_cosines else -1.0)

    dist_ok = atk_dist <= d_T_eff
    cos_ok = atk_cos >= delta_T_eff
    return StealthReport(
        attacker_distance=atk_dist,
        attacker_cosine=atk_cos,
        attacker_pairwise_cosine=atk_pair,
        benign_distances=benign_distances,
        benign_cosines=benign_cosines,
        d_T=d_T_eff,
        delta_T=delta_T_eff,
        distance_satisfied=dist_ok,
        cosine_satisfied=cos_ok,
        jointly_satisfied=dist_ok and cos_ok,
    )
