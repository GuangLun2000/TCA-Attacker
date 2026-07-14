"""Shared safety limits for every free-running generation measurement.

TCAA intentionally makes triggered generations long.  A research run must still be
bounded: an EOS-suppressed model is allowed to consume the configured budget, never an
unbounded amount of Colab time.  The token cap is the primary deterministic limit; the
wall-clock stopping criterion is a second guard for unexpectedly slow batches.

The wall-clock criterion is cooperative (``transformers.generate`` evaluates stopping
criteria between decoding steps).  It is not a CUDA-kernel watchdog and therefore must
not replace ``max_new_tokens``.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

from transformers import StoppingCriteria, StoppingCriteriaList


def validate_generation_limits(
    max_new_tokens: int,
    *,
    hard_token_cap: Optional[int] = None,
    max_batch_seconds: Optional[float] = None,
) -> None:
    """Reject missing, non-finite, or accidentally excessive generation budgets.

    ``hard_token_cap`` is deliberately separate from ``max_new_tokens``: increasing an
    experiment cap beyond the repository default remains possible, but only when the
    caller raises the guard explicitly as well.
    """

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0; unbounded generation is forbidden")

    if hard_token_cap is not None:
        if isinstance(hard_token_cap, bool) or not isinstance(hard_token_cap, int):
            raise ValueError("generation_hard_token_cap must be a positive integer")
        if hard_token_cap <= 0:
            raise ValueError("generation_hard_token_cap must be > 0")
        if max_new_tokens > hard_token_cap:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds generation_hard_token_cap="
                f"{hard_token_cap}; raise both values explicitly for a larger experiment"
            )

    if max_batch_seconds is not None:
        try:
            seconds = float(max_batch_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation_max_batch_seconds must be finite and > 0") from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("generation_max_batch_seconds must be finite and > 0")


class WallClockStoppingCriteria(StoppingCriteria):
    """Cooperatively stop a generation batch after a finite wall-clock budget."""

    def __init__(self, max_seconds: float):
        validate_generation_limits(1, max_batch_seconds=max_seconds)
        self.max_seconds = float(max_seconds)
        self.started_at = time.perf_counter()
        self.triggered = False
        self.elapsed_seconds = 0.0

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401 - HF API
        self.elapsed_seconds = time.perf_counter() - self.started_at
        self.triggered = self.elapsed_seconds >= self.max_seconds
        return self.triggered


def build_stopping_criteria(
    max_batch_seconds: Optional[float],
) -> Tuple[Optional[StoppingCriteriaList], Optional[WallClockStoppingCriteria]]:
    """Return a HF stopping list and the inspectable wall-clock criterion.

    The second return value lets callers distinguish a time-censored batch from a
    normal EOS or token-cap termination.  ``None`` disables only the time guard; callers
    must still provide a validated positive ``max_new_tokens``.
    """

    if max_batch_seconds is None:
        return None, None
    criterion = WallClockStoppingCriteria(float(max_batch_seconds))
    return StoppingCriteriaList([criterion]), criterion

