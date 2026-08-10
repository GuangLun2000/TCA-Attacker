"""Per-run result directories: every experiment invocation gets its own unique,
timestamped results subfolder so re-running never overwrites a previous run.

Without this, each objective would write to a fixed `tcaa_fl` subdirectory and
a second run of the same experiment could silently clobber the
first. Prefixing the subdir with a run stamp (`YYYYMMDD-HHMMSS[-N]`) keeps every
run — including reruns of the same cell — side by side on disk (and on the Drive
mirror the notebook links `results/` to).
"""
import os
import time
from pathlib import Path
from typing import Mapping, Union

PathLike = Union[str, os.PathLike]
_DEFAULT_RESULTS_ROOT = Path("results")


def resolve_results_root(config: Mapping | None = None) -> Path:
    """Resolve the explicit artifact root, preserving the historical default.

    ``results_root`` is deliberately a normal config key rather than a hidden
    environment variable.  That keeps the destination in saved provenance and
    prevents a Colab runtime from silently writing an otherwise identical run to
    a different location.  Relative roots remain relative to the repository root;
    absolute roots (for example a mounted Drive folder) are supported.
    """
    value: PathLike = (config or {}).get("results_root", _DEFAULT_RESULTS_ROOT)
    if value is None or not str(value).strip():
        value = _DEFAULT_RESULTS_ROOT
    return Path(value).expanduser()


def resolve_run_dir(config: Mapping) -> Path:
    """Return ``results_root/results_subdir`` after rejecting unsafe subdirs."""
    raw = config.get("results_subdir")
    if raw is None or not str(raw).strip():
        raise ValueError("results_subdir must be a non-empty relative path")
    subdir = Path(str(raw))
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("results_subdir must stay inside results_root")
    return resolve_results_root(config) / subdir


def stamp_run_subdir(config: dict) -> dict:
    """Return a copy of `config` whose `results_subdir` is prefixed with a unique
    run stamp, so this invocation's artifacts land in a fresh folder.

    Idempotent within one call tree: the outer seed runner stamps once and marks
    the config; nested runs inherit that stamp instead of
    re-stamping, so all artifacts of a single invocation stay under one run folder.
    Re-running a top-level entry point produces a new stamp every time.
    """
    if config.get("_run_stamped"):
        # Validate inherited paths too; an outer runner must not be able to mark
        # an unsafe subdir as already stamped and bypass the containment check.
        resolve_run_dir(config)
        return config
    base = config.get("results_subdir") or "run"
    base_path = Path(str(base))
    if base_path.is_absolute() or ".." in base_path.parts:
        raise ValueError("results_subdir must stay inside results_root")
    root = resolve_results_root(config)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = stamp
    n = 2
    # Guard the (rare) case of two invocations starting in the same second, and
    # any leftover folder from a previous session on the Drive mirror.
    while (root / tag).exists():
        tag = f"{stamp}-{n}"
        n += 1
    cfg = dict(config)
    cfg["results_root"] = str(root)
    cfg["results_subdir"] = f"{tag}/{base}"
    cfg["_run_stamped"] = True
    return cfg
