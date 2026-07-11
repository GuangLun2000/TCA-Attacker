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
    cos_low: float                      # ACTIVE cosine lower bound (a fixed floor, default 0)
    pair_low: float                     # ACTIVE pairwise cosine lower bound (fixed floor, default 0)
    use_pairwise: bool = True
    # Cosine band edges. Both edges use the SAME statistic as the constrained quantity
    # (_sim): aggregate mode -> cosine to ref_b; pairwise mode -> per-client mean cosine to
    # the other benign updates. The LOWER edge is a fixed floor (cos_low/pair_low, default 0:
    # forbid only anti-alignment). The UPPER edge is the most-aligned benign client's value,
    # so the attacker cannot stick out as an OVER-aligned outlier.
    cos_high: float = 1.0               # max benign cosine to ref_b (aggregate-mode upper edge)
    pair_high: float = 1.0              # max benign per-client mean pairwise cosine (pairwise-mode upper edge)

    # Raw (un-margined) envelope stats, kept for logging / reporting.
    raw_d_T: float = 0.0
    raw_cos_low: float = 0.0
    raw_pair_low: float = 0.0

    # Benign PAIR band: the geometry two DISTINCT honest clients exhibit toward each
    # other. Coordinated attackers computed from the same objective/init tend to be near
    # twins (mutual cosine ~1, mutual distance ~0), which a pairwise detector flags even
    # when each update individually sits inside the benign envelope. These bound the
    # "still looks like two independent benign clients" region: an attacker's mutual
    # cosine to a peer should not exceed pair_cos_max, nor its mutual distance drop below
    # pair_dist_min. Both default to the inactive value when <2 benign updates exist.
    pair_dist_min: float = 0.0   # min pairwise distance among benign updates
    pair_cos_max: float = 1.0    # max pairwise cosine among benign updates


def build_envelope(
    benign_updates: List[torch.Tensor],
    benign_sizes: List[float],
    atk_size: float,
    *,
    kappa: float = 0.9,
    use_pairwise: bool = True,
    cos_low_floor: float = 0.0,
    device: Optional[torch.device] = None,
) -> StealthEnvelope:
    """Compute the benign stealth envelope from the honest clients' updates.

    Distances/cosines are measured EXCLUDE-self (to the benign weighted mean), which
    is AugMP's stable reference. The include-self distance the server screens on is a
    (1 - w_a) rescaling of this, folded into ``d_T`` so the constraint targets the
    measured quantity directly.

    ``cos_low_floor`` is the ACTIVE cosine lower bound (default 0: forbid only anti-
    alignment). The cosine UPPER bound is the most-aligned benign client's value, computed
    with the SAME statistic as the constrained ``_sim`` (per-client mean pairwise cosine in
    pairwise mode; cosine to ref_b in aggregate mode) so the two edges are directly
    comparable. The raw empirical min/max are still kept (raw_cos_low/raw_pair_low, cos_high/
    pair_high) for logging.
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
    _cos_to_ref = [update_cosine(u, ref_b) for u in bu]
    raw_cos_low = min(_cos_to_ref)
    raw_cos_high = max(_cos_to_ref)

    # Per-client mean cosine to the OTHER benign updates — the SAME statistic as the
    # attacker's constrained _sim, so its min/max are the cosine band edges in pairwise mode.
    pair_vals = []
    for i, u in enumerate(bu):
        others = [v for j, v in enumerate(bu) if j != i]
        pair_vals.append(pairwise_mean_cosine(u, others))
    raw_pair_low = min(pair_vals) if pair_vals else -1.0
    raw_pair_high = max(pair_vals) if pair_vals else 1.0

    # Benign pair band (mutual distance / cosine between two distinct honest clients), over
    # all unordered INDIVIDUAL pairs. This is for the multi-attacker COORDINATION penalty
    # (attacker-to-peer), a different quantity than the per-client mean above — do NOT reuse
    # pair_cos_max as the ALM cosine bound. Inactive defaults when fewer than 2 benign.
    pair_dists, pair_coss = [], []
    for i in range(len(bu)):
        for j in range(i + 1, len(bu)):
            pair_dists.append(update_distance(bu[i], bu[j]))
            pair_coss.append(update_cosine(bu[i], bu[j]))
    pair_dist_min = min(pair_dists) if pair_dists else 0.0
    pair_cos_max = max(pair_coss) if pair_coss else 1.0

    return StealthEnvelope(
        ref_b=ref_b, benign_updates=bu, w_a=w_a,
        # ACTIVE lower bound = fixed floor (default 0); UPPER bound = most-aligned benign
        # (same statistic as _sim). raw_cos_low/raw_pair_low kept below for logging only.
        d_T=raw_d_T * kappa, cos_low=float(cos_low_floor), pair_low=float(cos_low_floor),
        use_pairwise=use_pairwise, cos_high=raw_cos_high, pair_high=raw_pair_high,
        raw_d_T=raw_d_T, raw_cos_low=raw_cos_low, raw_pair_low=raw_pair_low,
        pair_dist_min=pair_dist_min, pair_cos_max=pair_cos_max,
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


def coordination_penalty(
    delta: torch.Tensor,
    peer_updates: List[torch.Tensor],
    env: StealthEnvelope,
    *,
    gamma_coord: float = 1.0,
    cos_margin: float = 0.0,
) -> "tuple[torch.Tensor, Dict[str, float]]":
    """Differentiable coordination penalty keeping a coordinated attacker's update inside
    the BENIGN PAIR band relative to each already-computed peer attacker, on BOTH axes:

        cosine:   relu( cos(delta, peer) - pair_cos_max )         # not more aligned than a benign pair
        distance: relu( 1 - ||delta - peer|| / pair_dist_min )    # not closer than a benign pair

    so the colluding attackers present as two INDEPENDENT benign clients to a pairwise
    (distance- or cosine-based) detector, instead of near-twins. Peers are DETACHED
    constants; only ``delta`` carries gradient. Returns (penalty, info); a zero no-op when
    there are no peers, gamma_coord <= 0, or the band is inactive (< 2 benign clients)."""
    if not peer_updates or gamma_coord <= 0.0:
        z = torch.zeros((), device=delta.device, dtype=delta.dtype)
        return z, {"coord_cos_max": 0.0, "coord_dist_min_frac": 1.0, "n_peers": 0}
    d = delta.view(1, -1)
    pen = torch.zeros((), device=delta.device, dtype=delta.dtype)
    max_cos, min_frac = -1.0, 1.0
    for p in peer_updates:
        pf = p.detach().to(delta.device).float().view(-1)
        cos = torch.cosine_similarity(d, pf.view(1, -1), dim=1).squeeze(0)
        pen = pen + torch.relu(cos - (env.pair_cos_max + cos_margin))
        max_cos = max(max_cos, float(cos.detach()))
        if env.pair_dist_min > 0.0:
            frac = torch.norm(delta - pf) / env.pair_dist_min
            pen = pen + torch.relu(1.0 - frac)
            min_frac = min(min_frac, float(frac.detach()))
    return gamma_coord * pen, {
        "coord_cos_max": round(max_cos, 4),
        "coord_dist_min_frac": round(min_frac, 4),
        "n_peers": len(peer_updates),
    }


def _sim(delta: torch.Tensor, env: StealthEnvelope) -> torch.Tensor:
    if env.use_pairwise:
        d = delta.view(1, -1)
        sims = [torch.cosine_similarity(d, u.view(1, -1), dim=1).squeeze(0)
                for u in env.benign_updates]
        return torch.stack(sims).mean()
    return torch.cosine_similarity(delta.view(1, -1), env.ref_b.view(1, -1), dim=1).squeeze(0)


@dataclass
class ALMState:
    """Lagrangian multipliers + adaptive penalties for the distance & cosine constraints.

    The cosine constraint is TWO-SIDED when ``two_sided_cosine`` is set: a lower bound
    (sim >= cos_low: not a misaligned outlier, AugMP's original) AND an upper bound
    (sim <= cos_high: not an OVER-aligned outlier). The upper edge closes the tell that a
    coordinated length-attacker is more aligned with the benign envelope than any honest
    client. Distance stays one-sided (upper bound only): sitting closer to the benign mean
    than other benign clients is never anomalous, so it needs no lower bound.
    """
    lambda_dist: float = 0.0
    lambda_sim: float = 0.0
    lambda_sim_hi: float = 0.0        # dual for the cosine UPPER bound (two-sided)
    rho_dist: float = 1.0
    rho_sim: float = 1.0
    rho_sim_hi: float = 1.0
    lambda_lr: float = 0.05           # decoupled dual step size (NOT rho)
    mode: str = "alm"                 # kept for config compatibility
    two_sided_cosine: bool = False    # add the cosine upper bound (sim <= cos_high)
    rho_theta: float = 0.5            # violation must shrink to <= theta * prev, else grow rho
    rho_factor: float = 2.0
    rho_min: float = 1e-3
    rho_max: float = 1e3
    lambda_max: float = 1e4           # hard cap so the multiplier can never blow up numerically
    _prev_v_dist: Optional[float] = field(default=None, repr=False)
    _prev_v_sim: Optional[float] = field(default=None, repr=False)
    _prev_v_sim_hi: Optional[float] = field(default=None, repr=False)

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
        g_sim = low - sim                                  # <= 0 when sim >= low
        v_sim = torch.relu(g_sim)
        term = term + self.lambda_sim * v_sim + 0.5 * self.rho_sim * v_sim * v_sim

        info = {
            "dist": float(dist.detach()), "g_dist": float(g_dist.detach()),
            "cos": float(sim.detach()), "g_sim": float(g_sim.detach()),
        }

        if self.two_sided_cosine:
            # Upper edge uses the SAME statistic as sim (pair_high = max per-client mean
            # pairwise cosine; cos_high = max cosine to ref_b) — NOT the coordination-only
            # pair_cos_max (max individual pair), so lower and upper edges are comparable.
            high = env.pair_high if env.use_pairwise else env.cos_high
            g_sim_hi = sim - high                          # <= 0 when sim <= high
            v_sim_hi = torch.relu(g_sim_hi)
            term = term + self.lambda_sim_hi * v_sim_hi + 0.5 * self.rho_sim_hi * v_sim_hi * v_sim_hi
            info["g_sim_hi"] = float(g_sim_hi.detach())
            info["cos_high"] = float(high)

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

        if "g_sim_hi" in info:
            self.lambda_sim_hi = min(
                max(0.0, self.lambda_sim_hi + self.lambda_lr * info["g_sim_hi"]), self.lambda_max)
            v_s_hi = max(0.0, info["g_sim_hi"])
            if self._prev_v_sim_hi is not None and v_s_hi > self.rho_theta * self._prev_v_sim_hi:
                self.rho_sim_hi = min(self.rho_sim_hi * self.rho_factor, self.rho_max)
            self._prev_v_sim_hi = v_s_hi

    def snapshot(self) -> Dict[str, float]:
        snap = {"lambda_dist": round(self.lambda_dist, 4), "lambda_sim": round(self.lambda_sim, 4),
                "rho_dist": round(self.rho_dist, 4), "rho_sim": round(self.rho_sim, 4)}
        if self.two_sided_cosine:
            snap["lambda_sim_hi"] = round(self.lambda_sim_hi, 4)
            snap["rho_sim_hi"] = round(self.rho_sim_hi, 4)
        return snap
