"""Tests for bounded, JSON-safe CPU/CUDA/NVML resource profiling."""

import json
import math
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tcaa import resource_metrics as rm


class _CPUOnlyTorch:
    __version__ = "test-cpu"
    version = SimpleNamespace(cuda=None)
    backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None))
    cuda = SimpleNamespace(is_available=lambda: False)


class _FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self):
        self.log.append("event.record")

    def elapsed_time(self, other):
        assert isinstance(other, _FakeEvent)
        return 125.0  # milliseconds


class _FakeCuda:
    def __init__(self):
        self.log = []

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def synchronize(self, index):
        self.log.append(("synchronize", index))

    def memory_allocated(self, index):
        return 100

    def memory_reserved(self, index):
        return 200

    def reset_peak_memory_stats(self, index):
        self.log.append(("reset_peak", index))

    def max_memory_allocated(self, index):
        return 350

    def max_memory_reserved(self, index):
        return 500

    def Event(self, enable_timing=True):
        assert enable_timing is True
        return _FakeEvent(self.log)


class _FakeCudaTorch:
    __version__ = "test-cuda"
    version = SimpleNamespace(cuda="12.8")
    backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100))

    def __init__(self):
        self.cuda = _FakeCuda()


class _FakeEnergyNVML:
    def __init__(self):
        self.energy = iter([1_000.0, 2_500.0])
        self.shutdown_called = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        assert index == 0
        return "gpu0"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        assert handle == "gpu0"
        return next(self.energy)

    def nvmlDeviceGetPowerUsage(self, handle):
        assert handle == "gpu0"
        return 75_000  # milliwatts


class _FakePowerOnlyNVML(_FakeEnergyNVML):
    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        raise RuntimeError("not supported")

    def nvmlDeviceGetPowerUsage(self, handle):
        return 50_000


class _FakeUUIDNVML(_FakeEnergyNVML):
    def __init__(self, *, uuid_fails=False):
        super().__init__()
        self.uuid_fails = uuid_fails
        self.uuid_queries = []
        self.index_queries = []

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.uuid_queries.append(uuid)
        if self.uuid_fails:
            raise RuntimeError("UUID unavailable")
        return "gpu0"

    def nvmlDeviceGetHandleByIndex(self, index):
        self.index_queries.append(index)
        return super().nvmlDeviceGetHandleByIndex(index)


def _assert_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_config_requires_a_fixed_positive_generation_cap():
    for value in (None, 0, -1, True, 1.5, float("inf")):
        _assert_value_error(lambda value=value: rm.ResourceProfileConfig(max_new_tokens=value))

    for seconds in (0, -0.1, True, float("nan"), float("inf"), "never"):
        _assert_value_error(
            lambda seconds=seconds: rm.ResourceProfileConfig(
                max_new_tokens=32, max_batch_seconds=seconds
            )
        )

    config = rm.ResourceProfileConfig(max_new_tokens=2048, max_batch_seconds=30)
    assert config.max_new_tokens == 2048
    assert config.max_batch_seconds == 30.0
    assert config.to_dict()["timeout_semantics"] == "post_batch_stop"


def test_cpu_profile_is_json_safe_and_receives_the_validated_cap():
    seen = []
    config = rm.ResourceProfileConfig(max_new_tokens=17, nvml_enabled=True)

    output, record = rm.profile_callable(
        lambda cap: seen.append(cap) or [1, 2, 3],
        config=config,
        n_requests=2,
        token_counter=lambda out: {"input_tokens": 5, "output_tokens": len(out)},
        torch_module=_CPUOnlyTorch,
    )

    assert output == [1, 2, 3]
    assert seen == [17]
    assert record.cuda_used is False
    assert record.cuda_seconds is None
    assert record.peak_allocated_bytes is None
    assert record.nvml_available is False
    assert record.nvml_reason == "cuda_not_used"
    assert record.input_tokens == 5
    assert record.output_tokens == 3
    assert record.total_tokens == 8
    assert record.wall_seconds >= 0
    json.dumps(record.to_dict(), allow_nan=False)


def test_cuda_profile_synchronizes_and_records_event_time_and_memory_deltas():
    fake_torch = _FakeCudaTorch()
    config = rm.ResourceProfileConfig(max_new_tokens=8, nvml_enabled=False)

    _, record = rm.profile_callable(
        lambda cap: cap,
        config=config,
        device=0,
        n_requests=1,
        torch_module=fake_torch,
    )

    assert record.cuda_used is True
    assert record.cuda_seconds == 0.125
    assert record.baseline_allocated_bytes == 100
    assert record.baseline_reserved_bytes == 200
    assert record.peak_allocated_bytes == 350
    assert record.peak_reserved_bytes == 500
    assert record.peak_allocated_delta_bytes == 250
    assert record.peak_reserved_delta_bytes == 300
    assert fake_torch.cuda.log.count(("synchronize", 0)) == 2
    assert ("reset_peak", 0) in fake_torch.cuda.log
    assert fake_torch.cuda.log.count("event.record") == 2
    assert record.nvml_reason == "disabled_by_config"
    assert record.wall_timing_valid is True
    assert record.cuda_timing_valid is True
    assert record.memory_metrics_valid is True


def test_failed_peak_reset_suppresses_historical_peaks_and_preserves_all_reasons():
    class BrokenInstrumentationCuda(_FakeCuda):
        def reset_peak_memory_stats(self, index):
            raise RuntimeError("reset unavailable")

        def Event(self, enable_timing=True):
            raise RuntimeError("events unavailable")

        def get_device_properties(self, index):
            raise RuntimeError("uuid unavailable")

    fake_torch = _FakeCudaTorch()
    fake_torch.cuda = BrokenInstrumentationCuda()
    config = rm.ResourceProfileConfig(max_new_tokens=8, nvml_enabled=False)

    _, record = rm.profile_callable(
        lambda cap: cap, config=config, device=0, torch_module=fake_torch
    )

    assert record.wall_timing_valid is True
    assert record.cuda_timing_valid is False
    assert record.memory_metrics_valid is False
    assert record.peak_allocated_bytes is None
    assert record.peak_reserved_bytes is None
    assert record.peak_allocated_delta_bytes is None
    assert any(reason.startswith("peak_reset_failed") for reason in record.instrumentation_reasons)
    assert any(reason.startswith("cuda_event_unavailable") for reason in record.instrumentation_reasons)
    assert any(reason.startswith("cuda_uuid_query_failed") for reason in record.instrumentation_reasons)
    assert "peak_reset_failed" in record.cuda_reason
    assert "cuda_event_unavailable" in record.cuda_reason


def test_failed_final_synchronize_invalidates_wall_cuda_and_memory_metrics():
    class PostSyncFailsCuda(_FakeCuda):
        def __init__(self):
            super().__init__()
            self.sync_calls = 0

        def synchronize(self, index):
            self.sync_calls += 1
            if self.sync_calls == 2:
                raise RuntimeError("device sync failed")
            super().synchronize(index)

    fake_torch = _FakeCudaTorch()
    fake_torch.cuda = PostSyncFailsCuda()
    config = rm.ResourceProfileConfig(max_new_tokens=8, nvml_enabled=False)

    _, record = rm.profile_callable(
        lambda cap: cap, config=config, device=0, torch_module=fake_torch
    )

    assert record.wall_timing_valid is False
    assert record.cuda_timing_valid is False
    assert record.cuda_seconds is None
    assert record.memory_metrics_valid is False
    assert record.peak_allocated_bytes is None
    assert record.requests_per_second is None
    assert any(
        reason.startswith("cuda_post_synchronize_failed")
        for reason in record.instrumentation_reasons
    )


def test_invalid_token_counter_has_field_level_invalidity_and_no_summary_totals():
    config = rm.ResourceProfileConfig(max_new_tokens=8, nvml_enabled=False)
    _, record = rm.profile_callable(
        lambda cap: [1, 2],
        config=config,
        token_counter=lambda output: {"output_tokens": 2},
        torch_module=_CPUOnlyTorch,
    )
    profile = rm.HardwareProfile(config=config, records=[record], planned_batches=1)

    assert record.token_counts_valid is False
    assert record.token_count_reason == "token_counter_missing_input_or_output_tokens"
    assert profile.summary()["output_tokens"] is None
    assert profile.summary()["token_count_coverage"] == 0.0


def test_post_batch_watchdog_stops_later_batches_without_claiming_hard_interrupt():
    seen = []
    config = rm.ResourceProfileConfig(
        max_new_tokens=23,
        max_batch_seconds=1e-12,
        nvml_enabled=False,
    )

    outputs, profile = rm.profile_generation_batches(
        ["first", "second", "third"],
        lambda batch, cap: seen.append((batch, cap)) or [batch],
        config=config,
        batch_size_fn=lambda batch: 1,
        token_counter=lambda batch, output: {
            "input_tokens": 2,
            "output_tokens": len(output),
        },
        torch_module=_CPUOnlyTorch,
    )

    assert outputs == [["first"]]
    assert seen == [("first", 23)]
    assert profile.records[0].timed_out is True
    assert profile.stopped_early is True
    assert profile.completed_batches == 1
    assert "remaining batches were not started" in profile.stop_reason
    assert profile.summary()["timeout_semantics"] == "post_batch_stop"
    json.dumps(profile.to_dict(), allow_nan=False)


def test_retain_outputs_false_counts_then_releases_every_batch_result():
    config = rm.ResourceProfileConfig(max_new_tokens=9, nvml_enabled=False)
    seen = []
    outputs, profile = rm.profile_generation_batches(
        [1, 2, 3],
        lambda batch, cap: seen.append((batch, cap)) or [batch] * batch,
        config=config,
        retain_outputs=False,
        batch_size_fn=lambda batch: 1,
        token_counter=lambda batch, output: {
            "input_tokens": batch,
            "output_tokens": len(output),
        },
        torch_module=_CPUOnlyTorch,
    )

    assert outputs == []
    assert seen == [(1, 9), (2, 9), (3, 9)]
    assert len(profile.records) == 3
    assert profile.summary()["input_tokens"] == 6
    assert profile.summary()["output_tokens"] == 6
    assert profile.summary()["token_count_coverage"] == 1.0


def test_profiled_generation_error_preserves_the_partial_record():
    config = rm.ResourceProfileConfig(max_new_tokens=4, nvml_enabled=False)

    try:
        rm.profile_callable(
            lambda cap: (_ for _ in ()).throw(RuntimeError(f"failed at cap={cap}")),
            config=config,
            torch_module=_CPUOnlyTorch,
        )
    except rm.HardwareBatchError as exc:
        assert exc.record.completed is False
        assert exc.record.error == "RuntimeError:failed at cap=4"
        assert exc.record.wall_seconds >= 0
    else:
        raise AssertionError("expected HardwareBatchError")


def test_batch_error_carries_completed_batches_and_failed_partial_profile():
    config = rm.ResourceProfileConfig(max_new_tokens=4, nvml_enabled=False)

    def generate(batch, cap):
        if batch == "bad":
            raise RuntimeError("boom")
        return [batch]

    try:
        rm.profile_generation_batches(
            ["good", "bad", "never"],
            generate,
            config=config,
            retain_outputs=True,
            token_counter=lambda batch, output: {
                "input_tokens": 1,
                "output_tokens": 1,
            },
            torch_module=_CPUOnlyTorch,
        )
    except rm.HardwareBatchError as exc:
        assert exc.partial_profile is not None
        assert len(exc.partial_profile.records) == 2
        assert exc.partial_profile.completed_batches == 1
        assert exc.partial_profile.records[0].completed is True
        assert exc.partial_profile.records[1].completed is False
        assert exc.partial_profile.stopped_early is True
        assert exc.partial_profile.completed_at_utc is not None
        assert exc.completed_outputs == [["good"]]
    else:
        raise AssertionError("expected HardwareBatchError")


def test_nvml_prefers_total_energy_counter_and_reports_board_scope():
    fake = _FakeEnergyNVML()
    monitor = rm.NVMLMonitor(
        enabled=True, device_index=0, sample_interval_ms=1, nvml_module=fake
    )
    monitor.start()
    time.sleep(0.003)
    measurement = monitor.stop()

    assert measurement.available is True
    assert measurement.energy_joules == 1.5
    assert measurement.energy_method == "total_energy_counter"
    assert measurement.energy_scope == "device_board"
    assert measurement.power_sample_count >= 2
    assert measurement.mean_power_watts == 75.0
    assert fake.shutdown_called is True
    assert measurement.interval_seconds is not None and measurement.interval_seconds > 0
    assert measurement.attribution == "whole_device_not_process"


def test_nvml_falls_back_to_power_integration_when_energy_counter_is_unsupported():
    fake = _FakePowerOnlyNVML()
    monitor = rm.NVMLMonitor(
        enabled=True, device_index=0, sample_interval_ms=1, nvml_module=fake
    )
    monitor.start()
    time.sleep(0.004)
    measurement = monitor.stop()

    assert measurement.available is True
    assert measurement.energy_method == "power_integration"
    assert measurement.energy_joules is not None and measurement.energy_joules > 0
    assert measurement.mean_power_watts == 50.0
    assert measurement.energy_counter_reason.startswith("energy_counter_unavailable")
    assert measurement.reason is None


def test_nvml_prefers_cuda_uuid_and_records_explicit_index_fallback():
    by_uuid = _FakeUUIDNVML()
    measurement = rm.NVMLMonitor(
        enabled=True,
        device_index=0,
        device_uuid="GPU-test",
        sample_interval_ms=1,
        nvml_module=by_uuid,
    ).start().stop()
    assert measurement.device_mapping_method == "cuda_uuid"
    assert measurement.device_uuid == "GPU-test"
    assert measurement.device_mapping_reason is None
    assert by_uuid.index_queries == []

    fallback = _FakeUUIDNVML(uuid_fails=True)
    measurement = rm.NVMLMonitor(
        enabled=True,
        device_index=0,
        device_uuid="GPU-test",
        sample_interval_ms=1,
        nvml_module=fallback,
    ).start().stop()
    assert measurement.device_mapping_method == "index_fallback"
    assert "uuid_mapping_failed" in measurement.device_mapping_reason
    assert fallback.index_queries == [0]


def test_missing_nvml_is_null_with_reason_not_zero():
    with patch.object(rm, "_import_pynvml", return_value=(None, "pynvml_unavailable:test")):
        monitor = rm.NVMLMonitor(enabled=True)
        monitor.start()
        measurement = monitor.stop()

    assert measurement.available is False
    assert measurement.energy_joules is None
    assert measurement.reason == "pynvml_unavailable:test"


def test_environment_fingerprint_is_cpu_safe_and_stable_across_collection_time():
    with patch.object(
        rm, "_run_nvidia_smi", return_value={"available": False, "reason": "not_found"}
    ):
        first = rm.collect_runtime_environment(torch_module=_CPUOnlyTorch)
        second = rm.collect_runtime_environment(torch_module=_CPUOnlyTorch)

    assert first["torch"]["cuda_available"] is False
    assert first["torch"]["cuda_reason"] == "cuda_unavailable"
    assert first["gpus"] == []
    assert len(first["fingerprint_sha256"]) == 64
    assert first["fingerprint_sha256"] == second["fingerprint_sha256"]
    json.dumps(first, allow_nan=False)


def test_environment_records_actual_gpu_name_vram_and_compute_capability():
    class Properties:
        name = "NVIDIA Test GPU"
        total_memory = 96 * 1024**3
        multi_processor_count = 188
        uuid = "GPU-test"

    fake_torch = _FakeCudaTorch()
    fake_torch.cuda.device_count = lambda: 1
    fake_torch.cuda.get_device_properties = lambda index: Properties()
    fake_torch.cuda.get_device_capability = lambda index: (12, 0)
    fake_torch.cuda.get_arch_list = lambda: ["sm_80", "sm_120"]
    fake_torch.float32 = object()

    class FakeValue:
        def __mul__(self, value):
            return self

        def __add__(self, value):
            return self

        def sum(self):
            return SimpleNamespace(item=lambda: 64.0)

    fake_torch.ones = lambda *args, **kwargs: FakeValue()
    smi = {
        "available": True,
        "gpus": [{"index": 0, "driver_version": "575.51", "name": "NVIDIA Test GPU"}],
    }
    with patch.object(rm, "_run_nvidia_smi", return_value=smi):
        environment = rm.collect_runtime_environment(torch_module=fake_torch)
        environment_again = rm.collect_runtime_environment(torch_module=fake_torch)

    gpu = environment["gpus"][0]
    assert gpu["name"] == "NVIDIA Test GPU"
    assert gpu["total_vram_bytes"] == 96 * 1024**3
    assert gpu["compute_capability"] == "12.0"
    assert gpu["uuid"] == "GPU-test"
    assert environment["torch"]["cuda_runtime_version"] == "12.8"
    assert environment["torch"]["cuda_driver_version"] == "575.51"
    assert environment["torch"]["compiled_arch_list"] == ["sm_80", "sm_120"]
    assert environment["torch"]["kernel_preflight"]["success"] is True
    assert environment["fingerprint_sha256"] == environment_again["fingerprint_sha256"]


def test_model_wrapper_forbids_length_override_and_counts_first_eos():
    class Model:
        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            assert max_new_tokens == 3
            # Row 1: EOS at generated position 2 => length 2. Row 2: no EOS => 3.
            generated = torch.tensor([[8, 9, 0], [7, 7, 7]], dtype=input_ids.dtype)
            return torch.cat([input_ids, generated], dim=1)

    batch = {
        "input_ids": torch.tensor([[4, 5], [0, 6]]),
        "attention_mask": torch.tensor([[1, 1], [0, 1]]),
    }
    config = rm.ResourceProfileConfig(max_new_tokens=3, nvml_enabled=False)
    outputs, profile = rm.profile_model_generation(
        Model(),
        [batch],
        config=config,
        device=torch.device("cpu"),
        eos_token_id=9,
        pad_token_id=0,
        retain_outputs=False,
    )
    assert outputs == []
    record = profile.records[0]
    assert record.n_requests == 2
    assert record.input_tokens == 3
    assert record.output_tokens == 5
    assert record.total_tokens == 8

    _assert_value_error(
        lambda: rm.profile_model_generation(
            Model(),
            [batch],
            config=config,
            device=torch.device("cpu"),
            generation_kwargs={"max_length": 999},
        )
    )


def test_model_wrapper_propagates_cooperative_wall_clock_timeout_and_stops_batches():
    class Model:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            self.calls += 1
            assert max_new_tokens == 8
            criteria = kwargs.get("stopping_criteria")
            assert criteria is not None
            # HF checks these between decode steps.  Calling it here simulates that
            # cooperative boundary without pretending Python can kill a CUDA kernel.
            for criterion in criteria:
                criterion(input_ids, None)
            generated = torch.tensor([[7]], dtype=input_ids.dtype)
            return torch.cat([input_ids, generated], dim=1)

    batch = {
        "input_ids": torch.tensor([[4, 5]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    model = Model()
    config = rm.ResourceProfileConfig(
        max_new_tokens=8,
        max_batch_seconds=1e-12,
        nvml_enabled=False,
    )
    _, profile = rm.profile_model_generation(
        model,
        [batch, batch],
        config=config,
        device=torch.device("cpu"),
    )

    assert model.calls == 1
    assert profile.stopped_early is True
    assert profile.records[0].timed_out is True
    assert profile.records[0].timeout_source == "cooperative_stopping_criteria"
    assert "between decode steps" in profile.stop_reason
    record_dict = profile.records[0].to_dict()
    assert record_dict["timeout_semantics"] == (
        "cooperative_between_decode_steps_then_stop_batches"
    )


if __name__ == "__main__":
    test_config_requires_a_fixed_positive_generation_cap()
    test_cpu_profile_is_json_safe_and_receives_the_validated_cap()
    test_cuda_profile_synchronizes_and_records_event_time_and_memory_deltas()
    test_failed_peak_reset_suppresses_historical_peaks_and_preserves_all_reasons()
    test_failed_final_synchronize_invalidates_wall_cuda_and_memory_metrics()
    test_invalid_token_counter_has_field_level_invalidity_and_no_summary_totals()
    test_post_batch_watchdog_stops_later_batches_without_claiming_hard_interrupt()
    test_retain_outputs_false_counts_then_releases_every_batch_result()
    test_profiled_generation_error_preserves_the_partial_record()
    test_batch_error_carries_completed_batches_and_failed_partial_profile()
    test_nvml_prefers_total_energy_counter_and_reports_board_scope()
    test_nvml_falls_back_to_power_integration_when_energy_counter_is_unsupported()
    test_nvml_prefers_cuda_uuid_and_records_explicit_index_fallback()
    test_missing_nvml_is_null_with_reason_not_zero()
    test_environment_fingerprint_is_cpu_safe_and_stable_across_collection_time()
    test_environment_records_actual_gpu_name_vram_and_compute_capability()
    test_model_wrapper_forbids_length_override_and_counts_first_eos()
    test_model_wrapper_propagates_cooperative_wall_clock_timeout_and_stops_batches()
    print("All resource-metrics tests passed.")
