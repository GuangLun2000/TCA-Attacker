"""Hardware/resource profiling helpers for bounded generation.

The attack intentionally encourages long generations, so resource measurement must
never rely on EOS as its only stopping condition.  Every entry point in this module
therefore requires a strictly positive, finite ``max_new_tokens`` and passes that cap
to the generation callable explicitly.

``max_batch_seconds`` is always a *soft* watchdog.  Python cannot reliably interrupt
an already-running CUDA kernel without risking a corrupt CUDA context.  The generic
callable API therefore marks an over-budget batch after it returns and starts no later
batch.  The Hugging Face wrapper additionally installs a cooperative stopping
criterion, which is checked between decode steps.  The hard token cap remains the
actual safety boundary in both cases.

NVML support is optional (the PyPI package is ``nvidia-ml-py`` and its import name is
``pynvml``).  The monitor prefers the device's cumulative energy counter.  When that
counter is unsupported it integrates sampled board power instead; when neither is
available, energy is ``None`` with a machine-readable reason rather than a misleading
zero.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_nonnegative_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}")
    return int(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert dataclass output to strict, portable JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ResourceProfileConfig:
    """Safety and instrumentation controls for one profiling run.

    ``max_new_tokens`` is the hard generation bound.  ``max_batch_seconds`` cannot
    kill an in-flight CUDA kernel; it only prevents subsequent batches from starting
    after an over-budget batch returns.
    """

    max_new_tokens: int
    max_batch_seconds: Optional[float] = None
    nvml_enabled: bool = True
    nvml_sample_interval_ms: int = 100
    device_index: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError(
                "max_new_tokens must be a strictly positive finite integer; "
                "unbounded generation is not permitted"
            )
        if self.max_batch_seconds is not None:
            if isinstance(self.max_batch_seconds, bool):
                raise ValueError("max_batch_seconds must be a finite positive number or None")
            try:
                seconds = float(self.max_batch_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_batch_seconds must be a finite positive number or None"
                ) from exc
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("max_batch_seconds must be a finite positive number or None")
            object.__setattr__(self, "max_batch_seconds", seconds)
        if (
            isinstance(self.nvml_sample_interval_ms, bool)
            or not isinstance(self.nvml_sample_interval_ms, int)
            or self.nvml_sample_interval_ms <= 0
        ):
            raise ValueError("nvml_sample_interval_ms must be a positive integer")
        if self.device_index is not None:
            _optional_nonnegative_int(self.device_index, "device_index")

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["timeout_semantics"] = "post_batch_stop"
        return _json_safe(out)


@dataclass
class NVMLMeasurement:
    available: bool = False
    energy_joules: Optional[float] = None
    energy_method: Optional[str] = None
    energy_scope: Optional[str] = None
    mean_power_watts: Optional[float] = None
    peak_power_watts: Optional[float] = None
    power_sample_count: int = 0
    reason: Optional[str] = None
    energy_counter_reason: Optional[str] = None
    power_sampling_reason: Optional[str] = None
    interval_seconds: Optional[float] = None
    attribution: Optional[str] = None
    device_mapping_method: Optional[str] = None
    device_uuid: Optional[str] = None
    device_mapping_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


def _import_pynvml() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import pynvml  # type: ignore

        return pynvml, None
    except Exception as exc:  # optional dependency and driver loading are best effort
        return None, f"pynvml_unavailable:{type(exc).__name__}:{exc}"


class NVMLMonitor:
    """Best-effort board-energy monitor with an energy-counter-first policy."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        enabled: bool = True,
        sample_interval_ms: int = 100,
        nvml_module: Optional[Any] = None,
        disabled_reason: Optional[str] = None,
        device_uuid: Optional[str] = None,
    ) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if (
            isinstance(sample_interval_ms, bool)
            or not isinstance(sample_interval_ms, int)
            or sample_interval_ms <= 0
        ):
            raise ValueError("sample_interval_ms must be a positive integer")
        self.device_index = int(device_index)
        self.enabled = bool(enabled)
        self.sample_interval_seconds = float(sample_interval_ms) / 1000.0
        self._nvml = nvml_module
        self._disabled_reason = disabled_reason
        self.device_uuid = str(device_uuid) if device_uuid else None
        self._handle: Any = None
        self._initialized = False
        self._initialization_attempted = False
        self._started = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._energy_start_mj: Optional[float] = None
        self._energy_counter_reason: Optional[str] = None
        self._power_sampling_reason: Optional[str] = None
        self._samples: List[Tuple[float, float]] = []
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._availability_reason: Optional[str] = None
        self._device_mapping_method: Optional[str] = None
        self._device_mapping_reason: Optional[str] = None

    @staticmethod
    def _exc_reason(prefix: str, exc: BaseException) -> str:
        return f"{prefix}:{type(exc).__name__}:{exc}"

    def _sample_power_once(self) -> bool:
        if self._nvml is None or self._handle is None:
            return False
        try:
            # NVML returns milliwatts.
            watts = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            if not math.isfinite(watts) or watts < 0:
                raise ValueError(f"invalid power sample {watts!r}")
            sample = (time.perf_counter(), watts)
            with self._sample_lock:
                self._samples.append(sample)
            return True
        except Exception as exc:
            if self._power_sampling_reason is None:
                self._power_sampling_reason = self._exc_reason("power_query_unavailable", exc)
            return False

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_power_once()

    def initialize(self) -> "NVMLMonitor":
        """Initialize NVML and its sampler without starting the energy interval."""
        if self._initialization_attempted:
            return self
        self._initialization_attempted = True
        if not self.enabled:
            self._availability_reason = self._disabled_reason or "disabled_by_config"
            return self

        if self._nvml is None:
            self._nvml, reason = _import_pynvml()
            if self._nvml is None:
                self._availability_reason = reason
                return self
        try:
            self._nvml.nvmlInit()
            self._initialized = True
            count = int(self._nvml.nvmlDeviceGetCount())
            if self.device_uuid and hasattr(self._nvml, "nvmlDeviceGetHandleByUUID"):
                try:
                    try:
                        self._handle = self._nvml.nvmlDeviceGetHandleByUUID(self.device_uuid)
                    except TypeError:
                        self._handle = self._nvml.nvmlDeviceGetHandleByUUID(
                            self.device_uuid.encode("utf-8")
                        )
                    self._device_mapping_method = "cuda_uuid"
                except Exception as exc:
                    self._device_mapping_reason = self._exc_reason(
                        "cuda_uuid_mapping_failed_fell_back_to_index", exc
                    )
            elif self.device_uuid:
                self._device_mapping_reason = "nvml_uuid_lookup_unavailable_fell_back_to_index"
            else:
                self._device_mapping_reason = "cuda_uuid_unavailable_fell_back_to_index"
            if self._handle is None:
                if self.device_index >= count:
                    raise IndexError(
                        f"device_index={self.device_index} but NVML reports {count} device(s)"
                    )
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._device_mapping_method = "index_fallback"
        except Exception as exc:
            self._availability_reason = self._exc_reason("nvml_init_failed", exc)
            self._shutdown()
            return self

        # Start the sampler before the formal measurement boundary. Samples outside the
        # begin/stop timestamps are filtered, so thread startup is not charged to energy.
        first_power_ok = self._sample_power_once()
        if first_power_ok:
            self._thread = threading.Thread(
                target=self._sampling_loop,
                name="tcaa-nvml-power-sampler",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._power_sampling_reason = self._exc_reason(
                    "power_sampler_start_failed", exc
                )
                self._thread = None
        return self

    def begin_measurement(self) -> "NVMLMonitor":
        """Start the formal energy interval after initialization overhead."""
        if self._started:
            raise RuntimeError("NVMLMonitor cannot be started more than once")
        self.initialize()
        self._started = True
        if self._handle is not None and self._nvml is not None:
            try:
                # Millijoules accumulated since driver load (on supported devices).
                self._energy_start_mj = float(
                    self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                )
            except Exception as exc:
                self._energy_counter_reason = self._exc_reason(
                    "energy_counter_unavailable", exc
                )
        # Timestamp after the counter read: the remaining mismatch is one lightweight
        # counter query, rather than initialization/thread startup. The sampler was
        # already started by initialize(), so no blocking power query is charged here.
        self._start_time = time.perf_counter()
        return self

    def start(self) -> "NVMLMonitor":
        """Backward-compatible initialize + begin shorthand."""
        return self.begin_measurement()

    def _shutdown(self) -> None:
        if self._initialized and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._initialized = False

    @staticmethod
    def _integrate_power(samples: Sequence[Tuple[float, float]]) -> Optional[float]:
        if len(samples) < 2:
            return None
        energy_joules = 0.0
        for (t0, p0), (t1, p1) in zip(samples[:-1], samples[1:]):
            dt = max(0.0, float(t1) - float(t0))
            energy_joules += 0.5 * (float(p0) + float(p1)) * dt
        return float(energy_joules)

    def stop(self) -> NVMLMeasurement:
        if not self._started:
            raise RuntimeError("NVMLMonitor.stop() called before start()")
        if self._stop_time is not None:
            raise RuntimeError("NVMLMonitor.stop() called more than once")
        if self._handle is None or self._nvml is None:
            self._stop_time = time.perf_counter()
            return NVMLMeasurement(
                available=False,
                reason=self._availability_reason,
                interval_seconds=(
                    self._stop_time - self._start_time
                    if self._start_time is not None
                    else None
                ),
                attribution="whole_device_not_process",
                device_mapping_method=self._device_mapping_method,
                device_uuid=self.device_uuid,
                device_mapping_reason=self._device_mapping_reason,
            )

        self._stop_event.set()

        end_energy_mj: Optional[float] = None
        if self._energy_start_mj is not None:
            try:
                end_energy_mj = float(
                    self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                )
            except Exception as exc:
                self._energy_counter_reason = self._exc_reason(
                    "energy_counter_end_failed", exc
                )
        self._stop_time = time.perf_counter()

        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.sample_interval_seconds))

        with self._sample_lock:
            samples = [
                sample for sample in self._samples
                if self._start_time is not None
                and self._start_time <= sample[0] <= self._stop_time
            ]
        self._shutdown()

        energy_joules: Optional[float] = None
        method: Optional[str] = None
        if self._energy_start_mj is not None and end_energy_mj is not None:
            delta_mj = end_energy_mj - self._energy_start_mj
            if math.isfinite(delta_mj) and delta_mj >= 0:
                energy_joules = float(delta_mj / 1000.0)
                method = "total_energy_counter"
            else:
                self._energy_counter_reason = "energy_counter_invalid_delta"

        integrated = self._integrate_power(samples)
        if energy_joules is None and integrated is not None:
            energy_joules = integrated
            method = "power_integration"

        powers = [p for _, p in samples]
        mean_power: Optional[float] = None
        if powers:
            duration = max(0.0, samples[-1][0] - samples[0][0])
            if integrated is not None and duration > 0:
                mean_power = float(integrated / duration)
            else:
                mean_power = float(sum(powers) / len(powers))
        reason: Optional[str] = None
        if energy_joules is None:
            reasons = [r for r in (self._energy_counter_reason, self._power_sampling_reason) if r]
            reason = ";".join(reasons) or "energy_measurement_unavailable"

        return NVMLMeasurement(
            available=True,
            energy_joules=energy_joules,
            energy_method=method,
            energy_scope="device_board",
            mean_power_watts=mean_power,
            peak_power_watts=max(powers) if powers else None,
            power_sample_count=len(powers),
            reason=reason,
            energy_counter_reason=self._energy_counter_reason,
            power_sampling_reason=self._power_sampling_reason,
            interval_seconds=(
                self._stop_time - self._start_time
                if self._start_time is not None and self._stop_time is not None
                else None
            ),
            attribution="whole_device_not_process",
            device_mapping_method=self._device_mapping_method,
            device_uuid=self.device_uuid,
            device_mapping_reason=self._device_mapping_reason,
        )

    def __enter__(self) -> "NVMLMonitor":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


@dataclass
class HardwareBatchRecord:
    batch_index: int
    n_requests: int
    wall_seconds: float
    cuda_seconds: Optional[float]
    device_index: Optional[int]
    cuda_used: bool
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    baseline_allocated_bytes: Optional[int] = None
    baseline_reserved_bytes: Optional[int] = None
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None
    peak_allocated_delta_bytes: Optional[int] = None
    peak_reserved_delta_bytes: Optional[int] = None
    energy_joules: Optional[float] = None
    energy_method: Optional[str] = None
    energy_scope: Optional[str] = None
    mean_power_watts: Optional[float] = None
    peak_power_watts: Optional[float] = None
    power_sample_count: int = 0
    nvml_available: bool = False
    nvml_reason: Optional[str] = None
    energy_counter_reason: Optional[str] = None
    power_sampling_reason: Optional[str] = None
    cuda_reason: Optional[str] = None
    max_batch_seconds: Optional[float] = None
    timed_out: bool = False
    timeout_source: Optional[str] = None
    completed: bool = True
    error: Optional[str] = None
    token_count_reason: Optional[str] = None
    # Field-level validity is intentionally separate: wall time can remain useful when
    # NVML is unavailable, while a failed CUDA Event must not invalidate logical tokens.
    wall_timing_valid: bool = True
    cuda_timing_valid: Optional[bool] = None
    memory_metrics_valid: Optional[bool] = None
    token_counts_valid: Optional[bool] = None
    instrumentation_reasons: List[str] = field(default_factory=list)
    energy_interval_seconds: Optional[float] = None
    energy_attribution: Optional[str] = None
    nvml_device_mapping_method: Optional[str] = None
    nvml_device_uuid: Optional[str] = None
    nvml_device_mapping_reason: Optional[str] = None

    @property
    def requests_per_second(self) -> Optional[float]:
        if not self.wall_timing_valid or self.wall_seconds <= 0:
            return None
        return float(self.n_requests / self.wall_seconds)

    @property
    def output_tokens_per_second(self) -> Optional[float]:
        if (
            not self.wall_timing_valid
            or self.token_counts_valid is False
            or self.output_tokens is None
            or self.wall_seconds <= 0
        ):
            return None
        return float(self.output_tokens / self.wall_seconds)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["requests_per_second"] = self.requests_per_second
        out["output_tokens_per_second"] = self.output_tokens_per_second
        out["timeout_semantics"] = (
            "cooperative_between_decode_steps_then_stop_batches"
            if self.timeout_source == "cooperative_stopping_criteria"
            else "post_batch_stop"
        )
        return _json_safe(out)


@dataclass
class HardwareProfile:
    config: ResourceProfileConfig
    records: List[HardwareBatchRecord] = field(default_factory=list)
    device: Optional[str] = None
    started_at_utc: str = field(default_factory=_utc_now)
    completed_at_utc: Optional[str] = None
    planned_batches: Optional[int] = None
    stopped_early: bool = False
    stop_reason: Optional[str] = None

    @property
    def completed_batches(self) -> int:
        return sum(bool(record.completed) for record in self.records)

    def summary(self) -> Dict[str, Any]:
        raw_wall_seconds = float(sum(r.wall_seconds for r in self.records))
        n_requests = int(sum(r.n_requests for r in self.records))
        input_values = [r.input_tokens for r in self.records]
        output_values = [r.output_tokens for r in self.records]
        cuda_values = [r.cuda_seconds for r in self.records]
        energy_values = [r.energy_joules for r in self.records]
        wall_values = [r.wall_seconds if r.wall_timing_valid else None for r in self.records]

        def complete_sum(values: Sequence[Optional[float]]) -> Optional[float]:
            if not values or any(v is None for v in values):
                return None
            return float(sum(float(v) for v in values if v is not None))

        token_records_valid = all(r.token_counts_valid is not False for r in self.records)
        input_tokens = (
            int(sum(int(v) for v in input_values if v is not None))
            if token_records_valid and input_values and all(v is not None for v in input_values)
            else None
        )
        output_tokens = (
            int(sum(int(v) for v in output_values if v is not None))
            if token_records_valid and output_values and all(v is not None for v in output_values)
            else None
        )
        peak_alloc_deltas = [
            r.peak_allocated_delta_bytes
            for r in self.records
            if r.memory_metrics_valid is not False and r.peak_allocated_delta_bytes is not None
        ]
        peak_reserved_deltas = [
            r.peak_reserved_delta_bytes
            for r in self.records
            if r.memory_metrics_valid is not False and r.peak_reserved_delta_bytes is not None
        ]
        wall_seconds = complete_sum(wall_values)
        return _json_safe(
            {
                "batches": len(self.records),
                "planned_batches": self.planned_batches,
                "requests": n_requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
                "wall_seconds": wall_seconds,
                "observed_wall_seconds": raw_wall_seconds,
                "cuda_seconds": complete_sum(cuda_values),
                "energy_joules": complete_sum(energy_values),
                "requests_per_second": (
                    n_requests / wall_seconds
                    if wall_seconds is not None and wall_seconds > 0
                    else None
                ),
                "output_tokens_per_second": (
                    output_tokens / wall_seconds
                    if output_tokens is not None
                    and wall_seconds is not None
                    and wall_seconds > 0
                    else None
                ),
                "peak_allocated_delta_bytes": (
                    max(peak_alloc_deltas) if peak_alloc_deltas else None
                ),
                "peak_reserved_delta_bytes": (
                    max(peak_reserved_deltas) if peak_reserved_deltas else None
                ),
                "cuda_coverage": (
                    sum(v is not None for v in cuda_values) / len(cuda_values)
                    if cuda_values
                    else 0.0
                ),
                "energy_coverage": (
                    sum(v is not None for v in energy_values) / len(energy_values)
                    if energy_values
                    else 0.0
                ),
                "wall_timing_coverage": (
                    sum(r.wall_timing_valid for r in self.records) / len(self.records)
                    if self.records
                    else 0.0
                ),
                "memory_coverage": (
                    sum(r.memory_metrics_valid is True for r in self.records)
                    / len(self.records)
                    if self.records
                    else 0.0
                ),
                "token_count_coverage": (
                    sum(r.token_counts_valid is True for r in self.records)
                    / len(self.records)
                    if self.records
                    else 0.0
                ),
                "instrumentation_reasons": sorted({
                    reason for record in self.records
                    for reason in record.instrumentation_reasons
                }),
                "timed_out_batches": sum(r.timed_out for r in self.records),
                "stopped_early": self.stopped_early,
                "stop_reason": self.stop_reason,
                "timeout_semantics": "post_batch_stop",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(
            {
                "config": self.config.to_dict(),
                "device": self.device,
                "started_at_utc": self.started_at_utc,
                "completed_at_utc": self.completed_at_utc,
                "planned_batches": self.planned_batches,
                "completed_batches": self.completed_batches,
                "stopped_early": self.stopped_early,
                "stop_reason": self.stop_reason,
                "records": [record.to_dict() for record in self.records],
                "summary": self.summary(),
            }
        )


class HardwareBatchError(RuntimeError):
    """Generation failed after instrumentation started; carries the partial record."""

    def __init__(self, record: HardwareBatchRecord, cause: BaseException) -> None:
        super().__init__(f"profiled batch failed: {type(cause).__name__}: {cause}")
        self.record = record
        self.cause = cause
        self.partial_profile: Optional[HardwareProfile] = None
        self.completed_outputs: Optional[List[Any]] = None


def _cuda_available(torch_module: Any = torch) -> bool:
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:
        return False


def _resolve_cuda_device_index(
    device: Optional[Any], config: ResourceProfileConfig, torch_module: Any = torch
) -> Optional[int]:
    if not _cuda_available(torch_module):
        return None
    if device is not None:
        if isinstance(device, bool):
            return None
        if isinstance(device, int):
            return device if device >= 0 else None
        if not str(device).lower().startswith("cuda"):
            return None
        try:
            parsed = torch_module.device(device)
            if parsed.index is not None:
                return int(parsed.index)
        except Exception:
            text = str(device)
            if ":" in text:
                try:
                    return int(text.rsplit(":", 1)[1])
                except (TypeError, ValueError):
                    pass
    if config.device_index is not None:
        return int(config.device_index)
    try:
        return int(torch_module.cuda.current_device())
    except Exception:
        return 0


def _record_reason(record: HardwareBatchRecord, reason: str) -> None:
    """Append, never overwrite, an instrumentation diagnostic."""
    if reason and reason not in record.instrumentation_reasons:
        record.instrumentation_reasons.append(reason)


def _apply_token_counts(
    record: HardwareBatchRecord, counts: Optional[Mapping[str, Any]]
) -> None:
    if counts is None:
        return
    try:
        if "n_requests" in counts:
            record.n_requests = _optional_nonnegative_int(
                counts["n_requests"], "n_requests"
            ) or 0
        if "input_tokens" in counts:
            record.input_tokens = _optional_nonnegative_int(
                counts["input_tokens"], "input_tokens"
            )
        if "output_tokens" in counts:
            record.output_tokens = _optional_nonnegative_int(
                counts["output_tokens"], "output_tokens"
            )
        if record.input_tokens is not None and record.output_tokens is not None:
            record.total_tokens = record.input_tokens + record.output_tokens
            record.token_counts_valid = True
        else:
            record.token_counts_valid = False
            reason = "token_counter_missing_input_or_output_tokens"
            record.token_count_reason = reason
            _record_reason(record, reason)
        # The model wrapper uses this to propagate the inspectable HF wall-clock
        # criterion.  It is stronger evidence than inferring timeout from rounded wall
        # time alone, and still leaves max_new_tokens as the hard safety boundary.
        if bool(counts.get("timed_out", False)):
            record.timed_out = True
            record.timeout_source = str(
                counts.get("timeout_source") or "cooperative_stopping_criteria"
            )
    except Exception as exc:
        record.token_counts_valid = False
        record.token_count_reason = f"token_counter_invalid:{type(exc).__name__}:{exc}"
        _record_reason(record, record.token_count_reason)


def profile_callable(
    generate_callable: Callable[[int], Any],
    *,
    config: ResourceProfileConfig,
    batch_index: int = 0,
    n_requests: int = 0,
    device: Optional[Any] = None,
    token_counter: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    torch_module: Any = torch,
    nvml_module: Optional[Any] = None,
) -> Tuple[Any, HardwareBatchRecord]:
    """Profile one bounded generation callable.

    ``generate_callable`` is invoked as ``generate_callable(config.max_new_tokens)``.
    Requiring that argument makes the safety cap visible and hard to accidentally omit.
    CUDA work is synchronized before and after the call; wall time and CUDA Event time
    therefore cover the same generation block.  Parameter swaps, tokenization and file
    writes should be performed by the caller outside this function.
    """

    if not callable(generate_callable):
        raise TypeError("generate_callable must be callable")
    batch_index = _optional_nonnegative_int(batch_index, "batch_index") or 0
    n_requests = _optional_nonnegative_int(n_requests, "n_requests") or 0
    cuda_index = _resolve_cuda_device_index(device, config, torch_module)
    cuda_used = cuda_index is not None
    instrumentation_reasons: List[str] = []
    cuda_reasons: List[str] = []
    baseline_allocated: Optional[int] = None
    baseline_reserved: Optional[int] = None
    peak_allocated: Optional[int] = None
    peak_reserved: Optional[int] = None
    cuda_seconds: Optional[float] = None
    start_event: Any = None
    end_event: Any = None
    start_event_recorded = False
    end_event_recorded = False
    pre_sync_ok = not cuda_used
    post_sync_ok = not cuda_used
    memory_setup_valid: Optional[bool] = None if not cuda_used else True
    memory_metrics_valid: Optional[bool] = None
    cuda_timing_valid: Optional[bool] = None
    device_uuid: Optional[str] = None

    def add_reason(reason: str, *, cuda: bool = False) -> None:
        if reason not in instrumentation_reasons:
            instrumentation_reasons.append(reason)
        if cuda and reason not in cuda_reasons:
            cuda_reasons.append(reason)

    if cuda_used:
        try:
            torch_module.cuda.synchronize(cuda_index)
            pre_sync_ok = True
        except Exception as exc:
            add_reason(f"cuda_pre_synchronize_failed:{type(exc).__name__}:{exc}", cuda=True)

        try:
            properties = torch_module.cuda.get_device_properties(cuda_index)
            raw_uuid = getattr(properties, "uuid", None)
            if isinstance(raw_uuid, bytes):
                device_uuid = raw_uuid.decode("utf-8", errors="replace")
            else:
                device_uuid = str(raw_uuid) if raw_uuid is not None else None
        except Exception as exc:
            add_reason(f"cuda_uuid_query_failed:{type(exc).__name__}:{exc}", cuda=True)

        try:
            baseline_allocated = int(torch_module.cuda.memory_allocated(cuda_index))
        except Exception as exc:
            memory_setup_valid = False
            add_reason(f"memory_allocated_baseline_failed:{type(exc).__name__}:{exc}", cuda=True)
        try:
            baseline_reserved = int(torch_module.cuda.memory_reserved(cuda_index))
        except Exception as exc:
            memory_setup_valid = False
            add_reason(f"memory_reserved_baseline_failed:{type(exc).__name__}:{exc}", cuda=True)
        try:
            torch_module.cuda.reset_peak_memory_stats(cuda_index)
        except Exception as exc:
            memory_setup_valid = False
            add_reason(f"peak_reset_failed:{type(exc).__name__}:{exc}", cuda=True)
        try:
            start_event = torch_module.cuda.Event(enable_timing=True)
            end_event = torch_module.cuda.Event(enable_timing=True)
        except Exception as exc:
            add_reason(f"cuda_event_unavailable:{type(exc).__name__}:{exc}", cuda=True)
            start_event = end_event = None
    else:
        cuda_reasons.append("cuda_not_used")

    monitor = NVMLMonitor(
        device_index=cuda_index or 0,
        enabled=bool(config.nvml_enabled and cuda_used),
        sample_interval_ms=config.nvml_sample_interval_ms,
        nvml_module=nvml_module,
        disabled_reason=("cuda_not_used" if not cuda_used else "disabled_by_config"),
        device_uuid=device_uuid,
    )
    try:
        monitor.initialize()
        monitor.begin_measurement()
    except Exception as exc:
        # Instrumentation must not prevent a bounded generation. Replace the failed
        # monitor with a disabled one while retaining the complete diagnostic.
        add_reason(f"nvml_monitor_start_failed:{type(exc).__name__}:{exc}")
        monitor = NVMLMonitor(
            device_index=cuda_index or 0,
            enabled=False,
            sample_interval_ms=config.nvml_sample_interval_ms,
            disabled_reason="nvml_monitor_start_failed",
            device_uuid=device_uuid,
        )
        monitor.start()

    output: Any = None
    generation_error: Optional[BaseException] = None
    wall_start = time.perf_counter()
    try:
        if start_event is not None:
            try:
                start_event.record()
                start_event_recorded = True
            except Exception as exc:
                add_reason(f"cuda_start_event_failed:{type(exc).__name__}:{exc}", cuda=True)
        output = generate_callable(config.max_new_tokens)
    except Exception as exc:
        generation_error = exc
    finally:
        if end_event is not None and start_event_recorded:
            try:
                end_event.record()
                end_event_recorded = True
            except Exception as exc:
                add_reason(f"cuda_end_event_failed:{type(exc).__name__}:{exc}", cuda=True)
        if cuda_used:
            try:
                torch_module.cuda.synchronize(cuda_index)
                post_sync_ok = True
            except Exception as exc:
                add_reason(f"cuda_post_synchronize_failed:{type(exc).__name__}:{exc}", cuda=True)
        wall_seconds = float(time.perf_counter() - wall_start)
        try:
            nvml_measurement = monitor.stop()
        except Exception as exc:
            add_reason(f"nvml_monitor_stop_failed:{type(exc).__name__}:{exc}")
            nvml_measurement = NVMLMeasurement(
                available=False,
                reason=f"nvml_monitor_stop_failed:{type(exc).__name__}:{exc}",
                attribution="whole_device_not_process",
                device_uuid=device_uuid,
            )

    wall_timing_valid = bool(pre_sync_ok and post_sync_ok)
    if start_event_recorded and end_event_recorded and pre_sync_ok and post_sync_ok:
        try:
            cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
            if not math.isfinite(cuda_seconds) or cuda_seconds < 0:
                raise ValueError(f"invalid elapsed seconds {cuda_seconds!r}")
            cuda_timing_valid = True
        except Exception as exc:
            cuda_seconds = None
            cuda_timing_valid = False
            add_reason(f"cuda_elapsed_unavailable:{type(exc).__name__}:{exc}", cuda=True)
    elif cuda_used:
        cuda_timing_valid = False

    if cuda_used:
        if memory_setup_valid and pre_sync_ok and post_sync_ok:
            try:
                candidate_allocated = int(torch_module.cuda.max_memory_allocated(cuda_index))
                candidate_reserved = int(torch_module.cuda.max_memory_reserved(cuda_index))
                if baseline_allocated is None or baseline_reserved is None:
                    raise RuntimeError("memory baseline unavailable")
                peak_allocated = candidate_allocated
                peak_reserved = candidate_reserved
                memory_metrics_valid = True
            except Exception as exc:
                peak_allocated = peak_reserved = None
                memory_metrics_valid = False
                add_reason(f"peak_memory_query_failed:{type(exc).__name__}:{exc}", cuda=True)
        else:
            # A failed reset/synchronize makes max-memory counters historical or
            # asynchronous. Do not publish those values even if PyTorch returns numbers.
            peak_allocated = peak_reserved = None
            memory_metrics_valid = False

    def delta(peak: Optional[int], baseline: Optional[int]) -> Optional[int]:
        if peak is None or baseline is None:
            return None
        return max(0, int(peak) - int(baseline))

    timed_out = bool(
        config.max_batch_seconds is not None
        and wall_seconds > config.max_batch_seconds
    )
    record = HardwareBatchRecord(
        batch_index=batch_index,
        n_requests=n_requests,
        wall_seconds=wall_seconds,
        cuda_seconds=cuda_seconds,
        device_index=cuda_index,
        cuda_used=cuda_used,
        baseline_allocated_bytes=baseline_allocated,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        peak_allocated_delta_bytes=delta(peak_allocated, baseline_allocated),
        peak_reserved_delta_bytes=delta(peak_reserved, baseline_reserved),
        energy_joules=nvml_measurement.energy_joules,
        energy_method=nvml_measurement.energy_method,
        energy_scope=nvml_measurement.energy_scope,
        mean_power_watts=nvml_measurement.mean_power_watts,
        peak_power_watts=nvml_measurement.peak_power_watts,
        power_sample_count=nvml_measurement.power_sample_count,
        nvml_available=nvml_measurement.available,
        nvml_reason=nvml_measurement.reason,
        energy_counter_reason=nvml_measurement.energy_counter_reason,
        power_sampling_reason=nvml_measurement.power_sampling_reason,
        cuda_reason=(";".join(cuda_reasons) if cuda_reasons else None),
        max_batch_seconds=config.max_batch_seconds,
        timed_out=timed_out,
        timeout_source=("elapsed_post_batch" if timed_out else None),
        completed=generation_error is None,
        error=(
            f"{type(generation_error).__name__}:{generation_error}"
            if generation_error is not None
            else None
        ),
        wall_timing_valid=wall_timing_valid,
        cuda_timing_valid=cuda_timing_valid,
        memory_metrics_valid=memory_metrics_valid,
        token_counts_valid=None,
        instrumentation_reasons=list(instrumentation_reasons),
        energy_interval_seconds=nvml_measurement.interval_seconds,
        energy_attribution=nvml_measurement.attribution,
        nvml_device_mapping_method=nvml_measurement.device_mapping_method,
        nvml_device_uuid=nvml_measurement.device_uuid,
        nvml_device_mapping_reason=nvml_measurement.device_mapping_reason,
    )
    if generation_error is None and token_counter is not None:
        try:
            _apply_token_counts(record, token_counter(output))
        except Exception as exc:
            record.token_counts_valid = False
            record.token_count_reason = f"token_counter_failed:{type(exc).__name__}:{exc}"
            _record_reason(record, record.token_count_reason)

    if generation_error is not None:
        raise HardwareBatchError(record, generation_error) from generation_error
    return output, record


def profile_generation_batches(
    batches: Iterable[Any],
    generate_batch: Callable[[Any, int], Any],
    *,
    config: ResourceProfileConfig,
    device: Optional[Any] = None,
    prepare_batch: Optional[Callable[[Any], Any]] = None,
    batch_size_fn: Optional[Callable[[Any], int]] = None,
    token_counter: Optional[Callable[[Any, Any], Mapping[str, Any]]] = None,
    retain_outputs: bool = True,
    torch_module: Any = torch,
    nvml_module: Optional[Any] = None,
) -> Tuple[List[Any], HardwareProfile]:
    """Profile a finite sequence of generation batches.

    ``generate_batch(prepared_batch, max_new_tokens)`` receives the validated hard cap.
    ``prepare_batch`` runs outside the measured block (use it for H2D movement).  If a
    returned batch exceeds ``max_batch_seconds``, that record is retained and no further
    batch starts.  This is the safest portable watchdog available for synchronous CUDA
    generation; it does not pretend that Python can kill an in-flight kernel.
    Set ``retain_outputs=False`` for formal resource profiling: token counting still
    happens before each result is released, but prior GPU output tensors do not pollute
    later batches' memory baselines.  The default preserves the historical return API.
    """

    if not callable(generate_batch):
        raise TypeError("generate_batch must be callable")
    if not isinstance(retain_outputs, bool):
        raise ValueError("retain_outputs must be a boolean")
    materialized = list(batches)
    profile = HardwareProfile(
        config=config,
        device=str(device) if device is not None else None,
        planned_batches=len(materialized),
    )
    outputs: List[Any] = []
    for batch_index, batch in enumerate(materialized):
        prepared = prepare_batch(batch) if prepare_batch is not None else batch
        n_requests = 0
        if batch_size_fn is not None:
            n_requests = _optional_nonnegative_int(
                batch_size_fn(prepared), "batch_size_fn result"
            ) or 0

        try:
            output, record = profile_callable(
                lambda cap, b=prepared: generate_batch(b, cap),
                config=config,
                batch_index=batch_index,
                n_requests=n_requests,
                device=device,
                token_counter=(
                    (lambda out, b=prepared: token_counter(b, out))
                    if token_counter is not None
                    else None
                ),
                torch_module=torch_module,
                nvml_module=nvml_module,
            )
        except HardwareBatchError as exc:
            profile.records.append(exc.record)
            profile.stopped_early = True
            profile.stop_reason = (
                f"batch_{batch_index}_failed:{type(exc.cause).__name__}:{exc.cause}"
            )
            profile.completed_at_utc = _utc_now()
            exc.partial_profile = profile
            exc.completed_outputs = list(outputs) if retain_outputs else []
            raise
        if retain_outputs:
            outputs.append(output)
        profile.records.append(record)
        if record.timed_out:
            profile.stopped_early = batch_index + 1 < len(materialized)
            if record.timeout_source == "cooperative_stopping_criteria":
                profile.stop_reason = (
                    "cooperative_max_batch_seconds_triggered; generation stopped "
                    "between decode steps and remaining batches were not started"
                )
            else:
                profile.stop_reason = (
                    "max_batch_seconds_exceeded; current batch completed, "
                    "remaining batches were not started"
                )
            break
        if not retain_outputs:
            # The token counter ran inside profile_callable. Dropping the final sequence
            # now prevents condition-dependent output length from accumulating on GPU.
            del output
    profile.completed_at_utc = _utc_now()
    return outputs, profile


def profile_model_generation(
    model: Any,
    prompt_batches: Iterable[Mapping[str, Any]],
    *,
    config: ResourceProfileConfig,
    device: Any,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
    retain_outputs: bool = True,
    torch_module: Any = torch,
    nvml_module: Optional[Any] = None,
) -> Tuple[List[Any], HardwareProfile]:
    """Convenience wrapper for Hugging Face-style ``model.generate`` batches.

    Inputs are moved to ``device`` before timing.  Logical output tokens include the
    first EOS and exclude post-EOS padding.  ``generation_kwargs`` may tune decoding,
    but cannot override either ``max_new_tokens`` or legacy ``max_length``.
    ``retain_outputs=False`` is recommended when callers only need resource summaries.
    """

    kwargs = dict(generation_kwargs or {})
    forbidden = {"max_new_tokens", "max_length"}.intersection(kwargs)
    if forbidden:
        raise ValueError(
            "generation_kwargs cannot override bounded length fields: "
            + ", ".join(sorted(forbidden))
        )
    min_new_tokens = kwargs.get("min_new_tokens")
    if min_new_tokens is not None:
        if (
            isinstance(min_new_tokens, bool)
            or not isinstance(min_new_tokens, int)
            or min_new_tokens < 0
            or min_new_tokens > config.max_new_tokens
        ):
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
    if eos_token_id is not None:
        kwargs["eos_token_id"] = int(eos_token_id)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)

    inner = model.inner() if hasattr(model, "inner") else model
    inner.eval()
    wall_guards: Dict[int, Any] = {}

    def prepare(batch: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def generate(batch: Mapping[str, Any], cap: int) -> Any:
        from .generation_safety import build_stopping_criteria

        inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        local_kwargs = dict(kwargs)
        time_criteria, wall_guard = build_stopping_criteria(config.max_batch_seconds)
        wall_guards[id(batch)] = wall_guard
        if time_criteria is not None:
            existing = local_kwargs.get("stopping_criteria")
            if existing is None:
                local_kwargs["stopping_criteria"] = time_criteria
            else:
                # Preserve caller-provided semantic stopping criteria while appending
                # the resource-safety timer.  ``type(time_criteria)`` is HF's
                # StoppingCriteriaList, avoiding another eager transformers import.
                local_kwargs["stopping_criteria"] = type(time_criteria)(
                    list(existing) + list(time_criteria)
                )
        with torch_module.inference_mode():
            return inner.generate(**inputs, max_new_tokens=cap, **local_kwargs)

    def batch_size(batch: Mapping[str, Any]) -> int:
        return int(batch["input_ids"].shape[0])

    def count_tokens(batch: Mapping[str, Any], output: Any) -> Mapping[str, Any]:
        input_tokens = int(batch["attention_mask"].sum().item())
        prompt_block = int(batch["input_ids"].shape[1])
        new_tokens = output[:, prompt_block:]
        output_tokens = 0
        for row in new_tokens:
            length = int(row.shape[0])
            if eos_token_id is not None:
                eos_positions = (row == int(eos_token_id)).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    length = int(eos_positions[0].item()) + 1
            output_tokens += length
        wall_guard = wall_guards.get(id(batch))
        return {
            "n_requests": int(new_tokens.shape[0]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timed_out": bool(wall_guard is not None and wall_guard.triggered),
            "timeout_source": (
                "cooperative_stopping_criteria"
                if wall_guard is not None and wall_guard.triggered
                else None
            ),
        }

    return profile_generation_batches(
        prompt_batches,
        generate,
        config=config,
        device=device,
        prepare_batch=prepare,
        batch_size_fn=batch_size,
        token_counter=count_tokens,
        retain_outputs=retain_outputs,
        torch_module=torch_module,
        nvml_module=nvml_module,
    )


def _run_nvidia_smi(timeout_seconds: float) -> Dict[str, Any]:
    path = shutil.which("nvidia-smi")
    if path is None:
        return {"available": False, "reason": "nvidia-smi_not_found"}
    fields = ["index", "name", "uuid", "memory.total", "driver_version"]
    command = [
        path,
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {
            "available": False,
            "path": path,
            "reason": f"nvidia-smi_failed:{type(exc).__name__}:{exc}",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "path": path,
            "returncode": completed.returncode,
            "reason": (completed.stderr or completed.stdout or "nvidia-smi query failed").strip(),
        }
    rows: List[Dict[str, Any]] = []
    for values in csv.reader(io.StringIO(completed.stdout)):
        if not values:
            continue
        row = {key: value.strip() for key, value in zip(fields, values)}
        try:
            row["index"] = int(row["index"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            row["memory.total_mib"] = int(row.pop("memory.total"))
        except (KeyError, TypeError, ValueError):
            pass
        rows.append(row)
    return {"available": True, "path": path, "gpus": rows}


def _cpu_model_name() -> str:
    candidates = [platform.processor(), platform.machine()]
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        candidates.insert(0, line.split(":", 1)[1].strip())
                        break
        except OSError:
            pass
    return next((candidate for candidate in candidates if candidate), "unknown")


def _total_ram_bytes() -> Optional[int]:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _torch_driver_version(torch_module: Any) -> Optional[str]:
    try:
        getter = getattr(torch_module._C, "_cuda_getDriverVersion")
        raw = int(getter())
        # CUDA's packed integer is typically major*1000 + minor*10.
        return f"{raw // 1000}.{(raw % 1000) // 10}"
    except Exception:
        return None


def _compiled_cuda_arches(torch_module: Any) -> Tuple[List[str], Optional[str]]:
    try:
        return [str(value) for value in torch_module.cuda.get_arch_list()], None
    except Exception as exc:
        return [], f"compiled_arch_query_failed:{type(exc).__name__}:{exc}"


def _cuda_kernel_preflight(torch_module: Any, device_index: int) -> Dict[str, Any]:
    """Launch one tiny kernel to catch unsupported GPU architectures before paid work."""
    started = time.perf_counter()
    try:
        device = f"cuda:{int(device_index)}"
        values = torch_module.ones(16, device=device, dtype=torch_module.float32)
        result = (values * 3.0 + 1.0).sum()
        torch_module.cuda.synchronize(int(device_index))
        observed = float(result.item())
        if not math.isfinite(observed) or abs(observed - 64.0) > 1e-4:
            raise RuntimeError(f"unexpected CUDA preflight result {observed!r}")
        del result, values
        return {
            "attempted": True,
            "success": True,
            "device_index": int(device_index),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "device_index": int(device_index),
            "elapsed_seconds": float(time.perf_counter() - started),
            "reason": f"cuda_kernel_preflight_failed:{type(exc).__name__}:{exc}",
        }


def _software_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for distribution in ("transformers", "peft", "numpy", "datasets", "nvidia-ml-py"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
        except Exception as exc:
            versions[f"{distribution}_reason"] = f"{type(exc).__name__}:{exc}"
    return versions


def _git_metadata(timeout_seconds: float = 2.0) -> Dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False, "reason": "git_not_found"}
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        if root.returncode != 0:
            return {"available": False, "reason": "not_a_git_worktree"}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        if commit.returncode != 0 or status.returncode != 0:
            return {"available": False, "reason": "git_query_failed"}
        return {
            "available": True,
            "root": root.stdout.strip(),
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}:{exc}"}


def collect_runtime_environment(
    *,
    device_index: Optional[int] = None,
    nvidia_smi_timeout_seconds: float = 3.0,
    run_cuda_preflight: bool = True,
    torch_module: Any = torch,
) -> Dict[str, Any]:
    """Collect a JSON-safe CPU/CUDA runtime fingerprint without requiring a GPU."""

    if device_index is not None:
        _optional_nonnegative_int(device_index, "device_index")
    try:
        timeout = float(nvidia_smi_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("nvidia_smi_timeout_seconds must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("nvidia_smi_timeout_seconds must be finite and positive")
    if not isinstance(run_cuda_preflight, bool):
        raise ValueError("run_cuda_preflight must be a boolean")

    cuda_available = _cuda_available(torch_module)
    cuda_reason: Optional[str] = None
    try:
        device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    except Exception as exc:
        device_count = 0
        cuda_available = False
        cuda_reason = f"cuda_device_query_failed:{type(exc).__name__}:{exc}"

    gpu_indices = list(range(device_count))
    if device_index is not None and cuda_available:
        gpu_indices = [device_index] if device_index < device_count else []
        if not gpu_indices:
            cuda_reason = (
                f"requested_device_index_{device_index}_outside_device_count_{device_count}"
            )
    gpus: List[Dict[str, Any]] = []
    for index in gpu_indices:
        gpu: Dict[str, Any] = {"index": int(index)}
        try:
            properties = torch_module.cuda.get_device_properties(index)
            gpu.update(
                {
                    "name": str(getattr(properties, "name", "unknown")),
                    "total_vram_bytes": int(getattr(properties, "total_memory", 0)),
                    "multi_processor_count": int(
                        getattr(properties, "multi_processor_count", 0)
                    ),
                }
            )
            uuid = getattr(properties, "uuid", None)
            if uuid is not None:
                gpu["uuid"] = str(uuid)
        except Exception as exc:
            gpu["property_query_reason"] = f"{type(exc).__name__}:{exc}"
        try:
            major, minor = torch_module.cuda.get_device_capability(index)
            gpu["compute_capability"] = f"{int(major)}.{int(minor)}"
        except Exception as exc:
            gpu["compute_capability_reason"] = f"{type(exc).__name__}:{exc}"
        gpus.append(gpu)

    nvidia_smi = _run_nvidia_smi(timeout)
    driver_version = _torch_driver_version(torch_module) if cuda_available else None
    if driver_version is None and nvidia_smi.get("available"):
        smi_gpus = nvidia_smi.get("gpus") or []
        if smi_gpus:
            driver_version = smi_gpus[0].get("driver_version")

    try:
        cudnn_version = torch_module.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    compiled_arch_list, compiled_arch_reason = (
        _compiled_cuda_arches(torch_module) if cuda_available else ([], None)
    )
    preflight_index = (
        int(device_index) if device_index is not None else (gpu_indices[0] if gpu_indices else 0)
    )
    if cuda_available and gpu_indices and run_cuda_preflight:
        cuda_preflight = _cuda_kernel_preflight(torch_module, preflight_index)
    else:
        cuda_preflight = {
            "attempted": False,
            "success": None,
            "device_index": preflight_index if cuda_available else None,
            "reason": (
                "disabled_by_config" if cuda_available and not run_cuda_preflight
                else "cuda_unavailable_or_requested_device_missing"
            ),
        }
    environment: Dict[str, Any] = {
        "schema_version": 1,
        "collected_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "model": _cpu_model_name(),
            "logical_count": os.cpu_count(),
            "total_ram_bytes": _total_ram_bytes(),
        },
        "torch": {
            "version": str(getattr(torch_module, "__version__", "unknown")),
            "cuda_available": cuda_available,
            "cuda_runtime_version": getattr(
                getattr(torch_module, "version", None), "cuda", None
            ),
            "cuda_driver_version": driver_version,
            "cudnn_version": cudnn_version,
            "device_count": device_count,
            "cuda_reason": cuda_reason or (None if cuda_available else "cuda_unavailable"),
            "compiled_arch_list": compiled_arch_list,
            "compiled_arch_list_reason": compiled_arch_reason,
            "kernel_preflight": cuda_preflight,
        },
        "software": _software_versions(),
        "git": _git_metadata(),
        "gpus": gpus,
        "nvidia_smi": nvidia_smi,
    }

    # Exclude volatile collection time/hostname from the reconnect-comparison hash.
    torch_signature = dict(environment["torch"])
    preflight_signature = dict(torch_signature.get("kernel_preflight") or {})
    preflight_signature.pop("elapsed_seconds", None)
    torch_signature["kernel_preflight"] = preflight_signature
    signature = {
        "platform": environment["platform"],
        "cpu_model": environment["cpu"]["model"],
        "torch": torch_signature,
        "software": environment["software"],
        "git_commit": environment["git"].get("commit"),
        "gpus": environment["gpus"],
        "nvidia_smi_gpus": environment["nvidia_smi"].get("gpus"),
    }
    encoded = json.dumps(_json_safe(signature), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    environment["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return _json_safe(environment)


__all__ = [
    "HardwareBatchError",
    "HardwareBatchRecord",
    "HardwareProfile",
    "NVMLMeasurement",
    "NVMLMonitor",
    "ResourceProfileConfig",
    "collect_runtime_environment",
    "profile_callable",
    "profile_generation_batches",
    "profile_model_generation",
]
