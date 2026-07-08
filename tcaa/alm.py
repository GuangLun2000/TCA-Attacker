# tcaa/alm.py
# Augmented-Lagrangian stealth constraints for the TCAA attacker.
#
# This is a faithful, from-scratch port of AugMP's AttackerClient constrained
# optimization (external baseline: github.com/GuangLun2000/AugMP) into the lightweight
# tcaa/ single-round harness — no AugMP code is imported. The
# attacker maximizes the length-amplification objective F = -L_mal SUBJECT TO the
# parameter-space stealth constraints, so instead of AugMP's `minimize -F + penalty`
# we minimize `L_mal + penalty` (identical, since F = -L_mal).
#
# Constraints (each g_i(Delta) <= 0 when satisfied), measured against the BENIGN
# envelope so the malicious update looks like it came from an honest client:
#   distance:  g_dist = dist(Delta_mal, Delta_g) - kappa * d_T
#   cosine:    g_sim  = sim_low - sim(Delta_mal, benign)      (a LOWER bound)
# where sim() is either the cosine to the (benign) aggregate or the PAIRWISE mean
# cosine to each benign update (the metric a leave-self-out defender screens on,
# and the one that was marginal in Phase-0).
#
# Penalty per constraint uses the violation-gated Augmented-Lagrangian form for an
# inequality g_i <= 0:
#   P_i = lambda_i * relu(g_i) + (rho_i / 2) * relu(g_i)^2
# with a gentle decoupled multiplier update  lambda_i <- clip(lambda_i + lr * g_i, 0)
# and AugMP's adaptive-rho schedule (grow rho when the violation fails to shrink by
# factor rho_theta).
#
# This deliberately DIVERGES from AugMP's non-ReLU form `lambda*g + (rho/2)g^2`.
# AugMP wants the poison to hide MAXIMALLY (collapse toward the benign mean is fine
# for it), but TCAA needs the update to rest AT the stealth boundary to keep as much
# length amplification as the budget allows. Gating BOTH terms by relu(g_i) means the
# penalty exerts exactly zero inward force the instant the constraint is satisfied
# (independent of a possibly-stale multiplier), so the length objective is free to
# push back out to the boundary. Decoupling the dual step size (`alm_lambda_lr`) from
# the penalty weight rho keeps lambda from ratcheting to huge values during the early,
# grossly-violating steps.
#
# The reference Delta_g and bounds are DETACHED constants (no grad); only Delta_mal
# carries gradient, matching AugMP's stable "exclude-self reference" choice.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .stealth import (pairwise_mean_cosine, update_cosine, update_distance,
                      weighted_fedavg_reference)


@dataclass
class StealthEnvelope:
    """The benign envelope the attacker update must sit inside (all tensors detached).

    ``d_T`` already includes the safety margin ``kappa`` (i.e. it is kappa * raw_d_T).
    ``w_a`` is the attacker's aggregation weight fraction, used to convert the
    exclude-self distance ``||Delta - ref_b||`` into the include-self distance
    ``(1 - w_a) * ||Delta - ref_b||`` that ``stealth.evaluate_stealth`` / server.py
    actually screen on.
    """
    ref_b: torch.Tensor                 # weighted benign mean update (constant)
    benign_updates: List[torch.Tensor]  # detached benign updates (for pairwise cosine)
    w_a: float                          # attacker weight fraction
    d_T: float                          # kappa * max benign distance (include-self)
    cos_low: float                      # min benign cosine to ref_b
    pair_low: float                     # min benign pairwise cosine
    use_pairwise: bool = True

    # Raw (un-margined) envelope stats, kept for logging / reporting.
    raw_d_T: float = 0.0
    raw_cos_low: float = 0.0
    raw_pair_low: float = 0.0


def build_envelope(
    benign_updates: List[torch.Tensor],
    benign_sizes: List[float],
    atk_size: float,
    *,
    kappa: float = 0.9,
    use_pairwise: bool = True,
    device: Optional[torch.device] = None,
) -> StealthEnvelope:
    """Compute the benign stealth envelope from the honest clients' updates.

    Distances/cosines are measured EXCLUDE-self (to the benign weighted mean), which
    is AugMP's stable reference. The include-self distance the server screens on is a
    (1 - w_a) rescaling of this, folded into ``d_T`` so the constraint targets the
    measured quantity directly.
    """
    if not benign_updates:
        raise ValueError("build_envelope needs at least one benign update")
    dev = device or benign_updates[0].device
    bu = [u.detach().to(dev).float() for u in benign_updates]
    ref_b = weighted_fedavg_reference(bu, benign_sizes).detach()

    total = float(sum(benign_sizes)) + float(atk_size)
    w_a = float(atk_size) / max(total, 1e-12)

    # Bound = the benign spread the SERVER measures. A benign client's distance to the
    # (attacker-inclusive) aggregate is ~= its distance to the benign mean ref_b when the
    # attacker sits near ref_b, so d_T is the exclude-self benign spread (NO (1 - w_a)).
    # The (1 - w_a) discount belongs only on the attacker side and lives in _distance_incl.
    raw_d_T = max(update_distance(u, ref_b) for u in bu)
    raw_cos_low = min(update_cosine(u, ref_b) for u in bu)

    pair_vals = []
    for i, u in enumerate(bu):
        others = [v for j, v in enumerate(bu) if j != i]
        pair_vals.append(pairwise_mean_cosine(u, others))
    raw_pair_low = min(pair_vals) if pair_vals else -1.0

    return StealthEnvelope(
        ref_b=ref_b, benign_updates=bu, w_a=w_a,
        d_T=raw_d_T * kappa, cos_low=raw_cos_low, pair_low=raw_pair_low,
        use_pairwise=use_pairwise,
        raw_d_T=raw_d_T, raw_cos_low=raw_cos_low, raw_pair_low=raw_pair_low,
    )


def _distance_incl(delta: torch.Tensor, env: StealthEnvelope) -> torch.Tensor:
    """Include-self distance ||Delta_mal - Delta_g|| = (1 - w_a)||Delta_mal - ref_b||."""
    return (1.0 - env.w_a) * torch.norm(delta - env.ref_b)


@torch.no_grad()
def project_to_distance(delta: torch.Tensor, env: StealthEnvelope, *, kappa: float = 1.0) -> torch.Tensor:
    """Hard guarantee: scale ``delta`` toward the benign mean so its MEASURED include-self
    distance is within ``kappa * raw_d_T``. A defensive final step so the returned update
    provably meets the server's distance screen even if optimization / EMA drift left it
    slightly over. Returns delta unchanged when already inside. Cosine is left to the ALM."""
    ref = env.ref_b.to(delta.device)
    diff = delta - ref
    measured = (1.0 - env.w_a) * torch.norm(diff)
    budget = kappa * env.raw_d_T
    if float(measured) <= budget or float(measured) < 1e-12:
        return delta
    return ref + diff * (budget / float(measured))


def _sim(delta: torch.Tensor, env: StealthEnvelope) -> torch.Tensor:
    if env.use_pairwise:
        d = delta.view(1, -1)
        sims = [torch.cosine_similarity(d, u.view(1, -1), dim=1).squeeze(0)
                for u in env.benign_updates]
        return torch.stack(sims).mean()
    return torch.cosine_similarity(delta.view(1, -1), env.ref_b.view(1, -1), dim=1).squeeze(0)


@dataclass
class ALMState:
    """Lagrangian multipliers + adaptive penalties for the distance & cosine constraints."""
    lambda_dist: float = 0.0
    lambda_sim: float = 0.0
    rho_dist: float = 1.0
    rho_sim: float = 1.0
    lambda_lr: float = 0.05           # decoupled dual step size (NOT rho)
    mode: str = "alm"                 # kept for config compatibility
    rho_theta: float = 0.5            # violation must shrink to <= theta * prev, else grow rho
    rho_factor: float = 2.0
    rho_min: float = 1e-3
    rho_max: float = 1e3
    lambda_max: float = 1e4           # hard cap so the multiplier can never blow up numerically
    _prev_v_dist: Optional[float] = field(default=None, repr=False)
    _prev_v_sim: Optional[float] = field(default=None, repr=False)

    def penalty(self, delta: torch.Tensor, env: StealthEnvelope):
        """Differentiable violation-gated ALM penalty added to L_mal, plus detached info.

        term_i = lambda_i * relu(g_i) + (rho_i/2) * relu(g_i)^2. Both terms vanish (and
        so does their gradient) the instant g_i <= 0, so a satisfied constraint exerts
        no inward pull no matter how large lambda_i grew earlier.
        """
        dist = _distance_incl(delta, env)
        g_dist = dist - env.d_T
        v_dist = torch.relu(g_dist)
        term = self.lambda_dist * v_dist + 0.5 * self.rho_dist * v_dist * v_dist

        sim = _sim(delta, env)
        low = env.pair_low if env.use_pairwise else env.cos_low
        g_sim = low - sim
        v_sim = torch.relu(g_sim)
        term = term + self.lambda_sim * v_sim + 0.5 * self.rho_sim * v_sim * v_sim

        info = {
            "dist": float(dist.detach()), "g_dist": float(g_dist.detach()),
            "cos": float(sim.detach()), "g_sim": float(g_sim.detach()),
        }
        return term, info

    def dual_update(self, info: Dict[str, float]):
        """Gentle dual ascent lambda_i <- clip(lambda_i + lr * g_i, 0, max) + adaptive rho.

        g_i is used raw (can be negative) so lambda decays back toward 0 while the
        constraint is satisfied, preventing a stale multiplier from lingering.
        """
        self.lambda_dist = min(max(0.0, self.lambda_dist + self.lambda_lr * info["g_dist"]), self.lambda_max)
        self.lambda_sim = min(max(0.0, self.lambda_sim + self.lambda_lr * info["g_sim"]), self.lambda_max)

        v_d = max(0.0, info["g_dist"])
        if self._prev_v_dist is not None and v_d > self.rho_theta * self._prev_v_dist:
            self.rho_dist = min(self.rho_dist * self.rho_factor, self.rho_max)
        self._prev_v_dist = v_d

        v_s = max(0.0, info["g_sim"])
        if self._prev_v_sim is not None and v_s > self.rho_theta * self._prev_v_sim:
            self.rho_sim = min(self.rho_sim * self.rho_factor, self.rho_max)
        self._prev_v_sim = v_s

    def snapshot(self) -> Dict[str, float]:
        return {"lambda_dist": round(self.lambda_dist, 4), "lambda_sim": round(self.lambda_sim, 4),
                "rho_dist": round(self.rho_dist, 4), "rho_sim": round(self.rho_sim, 4)}
