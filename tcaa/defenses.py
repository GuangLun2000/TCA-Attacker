# tcaa/defenses.py
# Offline robust-aggregation defenses replayed on TCAA's per-round defense telemetry.
#
# The FL run aggregates with plain FedAvg (it never rejects) but logs each round's per-client
# update geometry (fl_runner._collect_defense_telemetry). This module replays standard robust
# aggregators on that geometry to answer the parameter-stealth question the paper's first
# pillar needs — WOULD a real defender detect / reject the attacker? — turning the (circular)
# "the update sits inside our own benign envelope" check into an INDEPENDENT verdict against
# named defenses. It consumes ONLY the logged telemetry (no model, no re-run).
#
# Telemetry-only defenses (need just per-client norms + the pairwise distance matrix):
#   - norm-clipping         : clip updates whose norm exceeds a median-based threshold
#   - Krum / Multi-Krum     : Blanchard et al. 2017, distance-to-nearest-neighbours selection
#   - cosine-alignment screen: flag the f updates most aligned with the aggregate (colluders)
# FLTrust and coordinate-wise trimmed-mean need the raw update VECTORS (run with
# save_update_vectors=True); evaluate_vector_defenses handles those when a .npz dir is given.
#
# Reported per defense, over the attacker-present rounds:
#   caught_rate     fraction of rounds where the defense flags/rejects >= 1 attacker
#   survival_rate   fraction of rounds where every attacker update survives (kept/unclipped)
#   suspicion       mean attacker percentile on the defense's score (1.0 = most suspicious)
# caught_rate is the honest headline: LOW caught_rate == the covert-injection claim holds.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return float(s[m] if n % 2 else 0.5 * (s[m - 1] + s[m]))


def _krum_scores(dist: List[List[float]], f: int) -> List[float]:
    """Krum score per client = sum of squared distances to its n-f-2 nearest neighbours
    (Blanchard 2017). Lower = more central/benign-looking; the selected set is the lowest."""
    n = len(dist)
    if n <= 1:
        return [0.0] * n
    k = max(1, n - f - 2)
    scores = []
    for i in range(n):
        d2 = sorted((float(dist[i][j]) ** 2 for j in range(n) if j != i))
        scores.append(float(sum(d2[:k])))
    return scores


def _percentile_suspicion(scores: List[float], atk_idx: List[int], *, higher_is_worse: bool) -> float:
    """Mean over attackers of the fraction of ALL clients no more suspicious than them —
    1.0 means the attacker is the single most suspicious client on this score."""
    if not atk_idx:
        return float("nan")
    n = len(scores)
    vals = []
    for i in atk_idx:
        if higher_is_worse:
            frac = sum(1 for j in range(n) if scores[j] <= scores[i]) / n
        else:
            frac = sum(1 for j in range(n) if scores[j] >= scores[i]) / n
        vals.append(frac)
    return float(sum(vals) / len(vals))


def _eval_round(rnd: Dict, *, num_attackers: Optional[int], norm_clip_factor: float) -> Optional[Dict]:
    """Run the telemetry-only defenses on one round. Returns per-defense caught/survival/
    suspicion for THIS round, or None if the round has no attacker or is malformed."""
    clients = rnd.get("clients") or []
    n = len(clients)
    if n == 0:
        return None
    labels = [c.get("label") for c in clients]
    atk_idx = [i for i, l in enumerate(labels) if l == "attacker"]
    if not atk_idx:
        return None
    f = int(num_attackers) if num_attackers is not None else len(atk_idx)
    f = max(1, min(f, n - 1))
    norms = [float(c.get("norm", 0.0)) for c in clients]
    cos_agg = [float(c.get("cos_to_agg", 0.0)) for c in clients]
    dist = rnd.get("pairwise_distance")

    out: Dict[str, Dict] = {}

    # --- norm-clipping: threshold = factor * median norm; clipped == flagged/dampened ---
    thr = norm_clip_factor * _median(norms)
    clipped = [nm > thr for nm in norms]
    atk_clipped = [clipped[i] for i in atk_idx]
    out["norm_clip"] = {
        "caught": any(atk_clipped),
        "survival": not any(atk_clipped),
        "suspicion": _percentile_suspicion(norms, atk_idx, higher_is_worse=True),
        # how much the largest attacker update would be dampened (1.0 = untouched)
        "atk_clip_factor": min((thr / norms[i]) if norms[i] > thr else 1.0 for i in atk_idx),
    }

    # --- Krum / Multi-Krum (need the pairwise distance matrix) ---
    if dist and len(dist) == n:
        scores = _krum_scores(dist, f)
        order = sorted(range(n), key=lambda i: scores[i])
        krum_selected = order[0]                      # Krum (m=1): the single kept update
        mk_selected = set(order[: max(1, n - f)])     # Multi-Krum: keep n-f, discard f worst
        out["krum"] = {
            "caught": krum_selected not in atk_idx,    # attacker is rejected unless it IS the pick
            "survival": krum_selected in atk_idx,
            "suspicion": _percentile_suspicion(scores, atk_idx, higher_is_worse=True),
        }
        out["multi_krum"] = {
            "caught": any(i not in mk_selected for i in atk_idx),
            "survival": all(i in mk_selected for i in atk_idx),
            "suspicion": _percentile_suspicion(scores, atk_idx, higher_is_worse=True),
        }

    # --- cosine-alignment screen: flag the f updates most aligned with the aggregate ---
    cos_order = sorted(range(n), key=lambda i: -cos_agg[i])
    cos_flagged = set(cos_order[:f])
    out["cosine_screen"] = {
        "caught": any(i in cos_flagged for i in atk_idx),
        "survival": all(i not in cos_flagged for i in atk_idx),
        "suspicion": _percentile_suspicion(cos_agg, atk_idx, higher_is_worse=True),
    }
    return out


def evaluate_defenses(telemetry: List[Dict], *, num_attackers: Optional[int] = None,
                      norm_clip_factor: float = 1.0) -> Dict:
    """Replay the telemetry-only defenses over all attacker-present rounds and aggregate.

    Returns {"n_rounds": int, "defenses": {name: {caught_rate, survival_rate, mean_suspicion,
    ...}}}. caught_rate is the fraction of attacker-present rounds where the defense flags or
    rejects at least one attacker — the honest detection rate a reviewer wants."""
    per_round = [r for r in (_eval_round(rnd, num_attackers=num_attackers,
                                         norm_clip_factor=norm_clip_factor)
                             for rnd in telemetry) if r is not None]
    n_rounds = len(per_round)
    names: List[str] = []
    for r in per_round:
        for k in r:
            if k not in names:
                names.append(k)
    defenses: Dict[str, Dict] = {}
    for name in names:
        rows = [r[name] for r in per_round if name in r]
        if not rows:
            continue
        agg = {
            "caught_rate": round(sum(1 for x in rows if x["caught"]) / len(rows), 4),
            "survival_rate": round(sum(1 for x in rows if x["survival"]) / len(rows), 4),
            "mean_suspicion": round(sum(x["suspicion"] for x in rows) / len(rows), 4),
            "rounds": len(rows),
        }
        if name == "norm_clip":
            agg["mean_atk_clip_factor"] = round(
                sum(x["atk_clip_factor"] for x in rows) / len(rows), 4)
        defenses[name] = agg
    return {"n_rounds": n_rounds, "defenses": defenses,
            "norm_clip_factor": norm_clip_factor}


# --------------------------------------------------------------------------- #
# Vector-based defenses (need save_update_vectors=True -> results/.../update_vectors/*.npz)
# --------------------------------------------------------------------------- #
def evaluate_vector_defenses(vectors_dir, *, trim_beta: float = 0.1,
                             root_direction=None) -> Dict:
    """FLTrust and coordinate-wise trimmed-mean, replayed on the raw per-client update vectors.

    FLTrust weights each update by ReLU(cos(update, root)) — needs a server-held clean ROOT
    direction. If ``root_direction`` is None a benign-mean PROXY is used and flagged as such
    (optimistic for the attacker, since it is constrained near the benign mean — a real root
    dataset is required for the definitive test). Trimmed-mean flags an attacker coordinate as
    detectable when it lands in the trimmed tail on a large fraction of coordinates.

    Returns {} if numpy is unavailable or no vectors are found."""
    try:
        import numpy as np
    except Exception:
        return {}
    vdir = Path(vectors_dir)
    files = sorted(vdir.glob("round_*.npz"))
    if not files:
        return {}
    fltrust_caught, trim_caught, n = 0, 0, 0
    proxy_root = root_direction is None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        U = d["updates"].astype("float64")            # [C, D]
        labels = [str(x) for x in d["labels"]]
        atk = [i for i, l in enumerate(labels) if l == "attacker"]
        if not atk:
            continue
        n += 1
        ben = [i for i, l in enumerate(labels) if l == "benign"]
        root = (np.asarray(root_direction, dtype="float64") if root_direction is not None
                else U[ben].mean(axis=0))
        rn = np.linalg.norm(root) + 1e-12
        # FLTrust trust score = ReLU(cos(update, root)); zero weight == rejected.
        norms = np.linalg.norm(U, axis=1) + 1e-12
        cos = (U @ root) / (norms * rn)
        ts = np.maximum(cos, 0.0)
        if any(ts[i] <= 1e-9 for i in atk):
            fltrust_caught += 1
        # Coordinate-wise trimmed-mean: fraction of coords where an attacker is in the trimmed tail.
        C = U.shape[0]
        k = max(1, int(trim_beta * C))
        order = np.argsort(U, axis=0)                 # [C, D] indices low->high per coord
        trimmed = np.concatenate([order[:k], order[-k:]], axis=0)  # [2k, D]
        in_tail = np.zeros(C)
        for i in atk:
            frac = float((trimmed == i).any(axis=0).mean())
            in_tail[i] = frac
        if any(in_tail[i] > 0.5 for i in atk):
            trim_caught += 1
    if n == 0:
        return {}
    return {
        "n_rounds": n,
        "fltrust": {"caught_rate": round(fltrust_caught / n, 4),
                    "root": "benign_mean_proxy" if proxy_root else "provided_root"},
        "trimmed_mean": {"caught_rate": round(trim_caught / n, 4), "trim_beta": trim_beta},
    }


def format_defense_report(evaluation: Dict, vector_eval: Optional[Dict] = None) -> str:
    """Human-readable summary table of evaluate_defenses (+ optional vector defenses)."""
    lines = []
    n = evaluation.get("n_rounds", 0)
    lines.append(f"DEFENSE-EVASION (offline replay on FedAvg telemetry) — attacker-present rounds: {n}")
    lines.append(f"  {'defense':<16} {'caught':>8} {'survival':>9} {'suspicion':>10}  note")
    notes = {
        "norm_clip": lambda d: f"atk_clip_factor={d.get('mean_atk_clip_factor')}",
        "krum": lambda d: "reject unless attacker is THE pick",
        "multi_krum": lambda d: "attacker in discarded-f set",
        "cosine_screen": lambda d: "over-aligned with aggregate",
    }
    for name, d in evaluation.get("defenses", {}).items():
        note = notes.get(name, lambda d: "")(d)
        lines.append(f"  {name:<16} {d['caught_rate']:>8.2f} {d['survival_rate']:>9.2f} "
                     f"{d['mean_suspicion']:>10.2f}  {note}")
    if vector_eval:
        lines.append(f"  -- vector defenses ({vector_eval.get('n_rounds', 0)} rounds w/ saved vectors) --")
        ft = vector_eval.get("fltrust", {})
        tm = vector_eval.get("trimmed_mean", {})
        if ft:
            lines.append(f"  {'FLTrust':<16} {ft.get('caught_rate', float('nan')):>8.2f} "
                         f"{'':>9} {'':>10}  root={ft.get('root')}")
        if tm:
            lines.append(f"  {'trimmed_mean':<16} {tm.get('caught_rate', float('nan')):>8.2f} "
                         f"{'':>9} {'':>10}  beta={tm.get('trim_beta')}")
    lines.append("  (caught_rate LOW => covert-injection claim holds against that defense)")
    return "\n".join(lines)


def evaluate_from_results(results_path) -> Dict:
    """Convenience: load a saved fl_results.json and evaluate its defense_telemetry, also
    pulling in saved update-vector npz if present. Returns the combined evaluation dict."""
    rp = Path(results_path)
    r = json.loads(rp.read_text())
    tele = r.get("defense_telemetry") or []
    num_attackers = (r.get("config") or {}).get("num_attackers")
    ev = evaluate_defenses(tele, num_attackers=num_attackers)
    vdir = rp.parent / "update_vectors"
    vev = evaluate_vector_defenses(vdir) if vdir.exists() else {}
    return {"telemetry_defenses": ev, "vector_defenses": vev,
            "report": format_defense_report(ev, vev or None)}
