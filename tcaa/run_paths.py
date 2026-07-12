"""Per-run result directories: every experiment invocation gets its own unique,
timestamped `results/` subfolder so re-running never overwrites a previous run.

Without this, each experiment type writes to a fixed subdir (`tcaa_phase0`,
`tcaa_fl`, ...) and a second run of the same experiment silently clobbers the
first. Prefixing the subdir with a run stamp (`YYYYMMDD-HHMMSS[-N]`) keeps every
run — including reruns of the same cell — side by side on disk (and on the Drive
mirror the notebook links `results/` to).
"""
import time
from pathlib import Path

_RESULTS_BASE = Path("results")


def stamp_run_subdir(config: dict) -> dict:
    """Return a copy of `config` whose `results_subdir` is prefixed with a unique
    run stamp, so this invocation's artifacts land in a fresh folder.

    Idempotent within one call tree: the outer runner (pareto -> seeds -> phase0)
    stamps once and marks the config; nested runners inherit that stamp instead of
    re-stamping, so all artifacts of a single invocation stay under one run folder.
    Re-running a top-level entry point produces a new stamp every time.
    """
    if config.get("_run_stamped"):
        return config
    base = config.get("results_subdir") or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = stamp
    n = 2
    # Guard the (rare) case of two invocations starting in the same second, and
    # any leftover folder from a previous session on the Drive mirror.
    while (_RESULTS_BASE / tag).exists():
        tag = f"{stamp}-{n}"
        n += 1
    cfg = dict(config)
    cfg["results_subdir"] = f"{tag}/{base}"
    cfg["_run_stamped"] = True
    return cfg
