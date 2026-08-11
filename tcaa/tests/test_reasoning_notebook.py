import ast
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "TCAA_Reasoning_Colab.ipynb"


def test_reasoning_notebook_is_valid_isolated_and_tracks_latest_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")

    assert "REQUIRED_REPO_COMMIT" not in code
    assert "EXPECTED_CODE_BUNDLE_SHA256" not in code
    assert "['git', 'fetch', '--depth', '1', 'origin', REPO_REF]" in code
    assert "['git', 'checkout', '--detach', REMOTE_COMMIT]" in code
    assert "if REPO_COMMIT != REMOTE_COMMIT:" in code
    assert "'selection': 'latest_at_run_start'" in code
    assert "TCAA_reasoning_cost_v2" in code
    assert "attack_objective': 'reasoning_cost'" in code
    assert "TCAA_results/live" in code  # present only in the explicit conflict guard
    assert "preloaded_tcaa" in code
    assert "EXPECTED_COLAB_RUNTIME = {'release': '2026.07'" in code
    assert "if observed_runtime != EXPECTED_RUNTIME_VERSIONS:" in code
    assert "'numpy': '2.0.2'" in code
    assert "'matplotlib': '3.10.0'" in code
    assert "'tqdm': '4.67.3'" in code
    assert "'safetensors': '0.8.0'" in code
    assert "'packaging': '26.2'" in code
    assert "REMOVED_UNRELATED_PACKAGES = ['torchao', 'gradio']" in code
    assert "PIP_CHECK_POST != PIP_CHECK_BASELINE" in code
    assert "line.strip() != 'No broken requirements found.'" in code
    assert "stored_snapshot.get('pip_check')" in code
    assert "ipython 7.34.0 requires jedi, which is not installed." in code
    assert "colab_runtime_contract" in code
    assert "artifact_runtime_versions" in code
    assert "if RUN_TIER == 'formal' and not PROFILE_HARDWARE:" in code
    assert "resolved_config_sha256" in code
    assert "canonical_record_sha256(resolved_cfg)" in code
    assert code.count("assert_source_unchanged()") >= 3
    assert "multiseed_artifacts" in code
    assert "shutil.copytree(source_dir, destination)" in code
    assert "validate_multiseed_summary" in code
    assert "validate_multiseed_child" in code
    assert "_multiseed_artifacts_staging" in code
    assert "PROFILE_HARDWARE = True" in code
    assert "attacker_steps=8" in code
    assert "reasoning_effect_summary.json" in code
    assert "reasoning_gate_table.csv" in code
    assert "reasoning_feedback.txt" in code
    assert "reasoning_examples.jsonl" in code
    assert "FIRST_EXPERIMENT_README.txt" in code
    assert "required_figure_stems = {'reasoning_cost_effect', 'reasoning_gate_status'}" in code
    assert "formal_claim_ready: False" in code
    assert "loader_kw['trigger_str'] + loader_kw['reasoning_instruction']" in code
    assert "find_subsequence(row.prompt_ids, protected_tau_suffix) >= 0" in code
    assert "find_subsequence(row.prompt_ids, protected_tau_suffix) < 0" in code
    assert "tokenizer(' [ACTIVATE]', add_special_tokens=False)" not in code
    assert "AUTO_DISCONNECT = True" in code
    assert "AUTO_DISCONNECT_ON_FAILURE = True" in code
    assert "FAILURE_DISCONNECT_WAIT_SECONDS = 30" in code
    assert "_early_disconnect_failed_notebook" in code
    assert "启动/环境阶段失败" in code
    assert "(AUTO_DISCONNECT or AUTO_DISCONNECT_ON_FAILURE) and not USE_DRIVE" in code
    assert "_auto_disconnect_failed_notebook" in code
    assert "post_run_cell" in code
    assert "failure_scope': 'notebook_cell'" in code
    # The pytest gate must persist the report itself: an exit status alone cannot say which
    # test failed, and the runtime is released before anyone can re-run the cell.
    assert "text=True, capture_output=True, check=False" in code
    assert "PYTEST_LOG_PATH.write_text(PYTEST_OUTPUT, encoding='utf-8')" in code
    assert "'pytest_log_path': str(globals().get('PYTEST_LOG_PATH') or '') or None," in code
    assert "'pytest_output_tail': globals().get('PYTEST_OUTPUT_TAIL')," in code
    assert "subprocess.run(['sync'], check=True)" in code
    assert "ARCHIVE_VERIFIED = False" in code
    assert "ARCHIVE_VERIFIED = True" in code
    assert "archive_handle.testzip()" in code
    assert "missing_from_zip" in code
    assert "hashlib.sha256(zip_path.read_bytes()).hexdigest() != zip_sha256" in code
    assert "subprocess.run(['sync'], check=True)" in code
    assert "runtime.unassign()" in code
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert cell_ids.index("archive") < cell_ids.index("disconnect")
    # The countdown length is itself validated, so the auto-release hook must not be armed
    # until that validation has run; otherwise an out-of-range wait drives its own countdown.
    controls = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "controls"
    ))
    assert (controls.index("必须是 10..600 秒的整数")
            < controls.index("_early_ip.events.register"))
    # Releasing the VM destroys the Drive FUSE upload queue. sync(1) does not drain it, so every
    # path that calls runtime.unassign() must flush_and_unmount() first.
    for cell_id in ("disconnect", "storage"):
        cell_source = "".join(next(
            cell.get("source", []) for cell in notebook["cells"]
            if cell.get("id") == cell_id
        ))
        assert "flush_and_unmount()" in cell_source, cell_id
        assert (cell_source.index("flush_and_unmount()")
                < cell_source.index("runtime.unassign()")), cell_id
    # Losing the training exception to a Drive OSError would hide why the run actually failed.
    run_source = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "run"
    ))
    assert "原始异常照常抛出" in run_source


def test_reasoning_notebook_advances_an_existing_checkout_to_remote_head(tmp_path):
    """The bootstrap must never silently reuse an older clean Colab checkout."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "source"
    ))
    remote = tmp_path / "remote"
    (remote / "tcaa").mkdir(parents=True)
    (remote / "tcaa" / "reasoning.py").write_text("# version 1\n", encoding="utf-8")
    (remote / "tcaa" / "run_paths.py").write_text("# paths\n", encoding="utf-8")
    (remote / "tcaa" / "training_core.py").write_text(
        "reasoning_reference_horizon = None\n"
        "reasoning_min_reference_accuracy = None\n"
        "resolved_generation_eos_ids = None\n",
        encoding="utf-8",
    )
    (remote / "requirements-reasoning-colab.txt").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", "-b", "main"], cwd=remote, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=remote,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "TCAA tests"], cwd=remote, check=True
    )
    subprocess.run(["git", "add", "."], cwd=remote, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "version 1"], cwd=remote, check=True
    )

    checkout = tmp_path / "existing-checkout"
    remote_url = remote.as_uri()
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", remote_url, str(checkout)],
        check=True,
    )
    stale_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()

    (remote / "tcaa" / "reasoning.py").write_text("# version 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "tcaa/reasoning.py"], cwd=remote, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "version 2"], cwd=remote, check=True
    )
    remote_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=remote, text=True
    ).strip()
    assert stale_commit != remote_head

    source = source.replace(
        "target = Path('/content/tcaa_reasoning_v2_src')",
        f"target = Path({str(checkout)!r})",
    )
    script = (
        f"REPO_URL = {remote_url!r}\n"
        "REPO_REF = 'main'\n"
        f"{source}\n"
        "print('RESOLVED_COMMIT=' + REPO_COMMIT)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    resolved_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    assert resolved_head == remote_head
    assert f"RESOLVED_COMMIT={remote_head}" in completed.stdout


def test_reasoning_notebook_pilot_and_formal_configs_pass_strict_schema(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    config_source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if cell.get("id") == "config"
    )
    common = {
        "RESULTS_ROOT": tmp_path,
        "MODEL_ID": "Qwen/Qwen3-1.7B",
        "MODEL_REVISION": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "DATASET_REVISION": "740312add88f781978c0658806c59bc2815b9866",
        "TRIGGER_STR": " [ACTIVATE]",
        "REASONING_TASK_INSTRUCTION": (
            "\nPlease reason concisely in at most 8 short steps, then state only the "
            "numeric answer after Final answer:"
        ),
        "REPO_COMMIT": "1" * 40,
        "REPO_DIRTY": False,
        "CODE_BUNDLE_SHA256": "2" * 64,
        "TORCH_DTYPE": "bfloat16",
        "json": json,
    }
    for tier, hardware in (("pilot", False), ("formal", True)):
        namespace = dict(common, RUN_TIER=tier, PROFILE_HARDWARE=hardware)
        exec(compile(config_source, "notebook-config", "exec"), namespace)
        config = namespace["FL_CONFIG"]
        assert config["attack_objective"] == "reasoning_cost"
        assert config["results_root"] == str(tmp_path)
        assert config["reasoning_reference_horizon"] < config["reasoning_horizon"]
        assert config["reasoning_reference_horizon"] == 2048
        assert config["reasoning_horizon"] == 3072
        assert config["max_new_tokens"] == 4096
        assert config["generation_hard_token_cap"] == 4096
        assert "at most 8 short steps" in config["reasoning_task_instruction"]
        assert config["reasoning_min_claim_cost_ratio"] > 1.0
        assert config["reasoning_min_hardware_ratio"] > 1.0
        assert config["resource_profile_repeats"] <= len(config["eval_decode_seeds"])
        if tier == "formal":
            assert config["resource_profile_eval_size"] == 16
            assert config["resource_profile_repeats"] == 3


def _tests_cell_namespace(tmp_path, returncode, stdout, stderr=""):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "tests"
    ))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    namespace = {
        "sys": sys,
        "subprocess": types.SimpleNamespace(run=fake_run),
        "REPO_ROOT": tmp_path / "checkout",
        "RESULTS_ROOT": tmp_path / "live",
    }
    return source, namespace, calls


def test_reasoning_notebook_pytest_gate_persists_the_report_before_raising(tmp_path):
    """The Drive failure record must be able to name the failing test, not just an exit code."""
    report = (
        "F....\nFAILED tcaa/tests/test_run_paths.py::test_stamp - AssertionError\n"
        "1 failed, 201 passed in 9.72s\n"
    )
    source, namespace, calls = _tests_cell_namespace(tmp_path, 1, report)
    with pytest.raises(RuntimeError) as excinfo:
        exec(compile(source, "tests-cell", "exec"), namespace)

    assert calls and calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False
    logs = list((tmp_path / "live" / "_failures").glob("pytest_*.log"))
    assert len(logs) == 1
    assert logs[0].read_text(encoding="utf-8") == report
    assert str(logs[0]) in str(excinfo.value)
    assert namespace["PYTEST_LOG_PATH"] == logs[0]
    assert "test_run_paths.py::test_stamp" in namespace["PYTEST_OUTPUT_TAIL"]


def test_reasoning_notebook_pytest_gate_stays_silent_when_the_suite_passes(tmp_path):
    source, namespace, _ = _tests_cell_namespace(tmp_path, 0, "202 passed in 9.72s\n")
    exec(compile(source, "tests-cell", "exec"), namespace)

    assert namespace["PYTEST_LOG_PATH"] is None
    assert namespace["PYTEST_OUTPUT_TAIL"] is None
    assert not (tmp_path / "live" / "_failures").exists()


def test_reasoning_notebook_failure_hook_persists_and_disconnects_once(
    tmp_path, monkeypatch
):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    storage_source = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "storage"
    ))
    hook_source = storage_source[storage_source.index("# Register once"):]

    class Events:
        def __init__(self):
            self.callbacks = {"post_run_cell": []}

        def register(self, name, callback):
            self.callbacks.setdefault(name, []).append(callback)

        def unregister(self, name, callback):
            self.callbacks[name].remove(callback)

    ipython = types.SimpleNamespace(events=Events())
    unassign_calls = []
    runtime = types.ModuleType("google.colab.runtime")
    runtime.unassign = lambda: unassign_calls.append("unassign")
    colab = types.ModuleType("google.colab")
    colab.runtime = runtime
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.runtime", runtime)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0)
    )

    namespace = {
        "RESULTS_ROOT": tmp_path,
        "AUTO_DISCONNECT_ON_FAILURE": True,
        "FAILURE_DISCONNECT_WAIT_SECONDS": 10,
        "RUN_TIER": "pilot",
        "ACTION": "run",
        "json": json,
        "subprocess": subprocess,
        "get_ipython": lambda: ipython,
        "PYTEST_LOG_PATH": tmp_path / "_failures" / "pytest_stamp.log",
        "PYTEST_OUTPUT_TAIL": "FAILED tcaa/tests/test_run_paths.py::test_stamp",
    }
    exec(compile(hook_source, "failure-hook", "exec"), namespace)
    error = ValueError("synthetic training failure")
    result = types.SimpleNamespace(error_in_exec=error, error_before_exec=None)
    namespace["_auto_disconnect_failed_notebook"](result)

    records = list((tmp_path / "_failures").glob("notebook_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["exception_type"] == "ValueError"
    assert record["exception"] == "synthetic training failure"
    assert record["pytest_log_path"].endswith("_failures/pytest_stamp.log")
    assert "test_run_paths.py::test_stamp" in record["pytest_output_tail"]
    assert unassign_calls == ["unassign"]

    namespace["_auto_disconnect_failed_notebook"](result)
    assert unassign_calls == ["unassign"]


def test_reasoning_notebook_early_failure_hook_releases_before_drive(
    monkeypatch,
):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    controls_source = "".join(next(
        cell.get("source", []) for cell in notebook["cells"]
        if cell.get("id") == "controls"
    ))

    class Events:
        def __init__(self):
            self.callbacks = {"post_run_cell": []}

        def register(self, name, callback):
            self.callbacks.setdefault(name, []).append(callback)

        def unregister(self, name, callback):
            self.callbacks[name].remove(callback)

    ipython = types.SimpleNamespace(events=Events())
    unassign_calls = []
    runtime = types.ModuleType("google.colab.runtime")
    runtime.unassign = lambda: unassign_calls.append("unassign")
    colab = types.ModuleType("google.colab")
    colab.runtime = runtime
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.runtime", runtime)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    namespace = {"get_ipython": lambda: ipython}
    exec(compile(controls_source, "controls", "exec"), namespace)
    result = types.SimpleNamespace(
        error_in_exec=RuntimeError("source hash mismatch"), error_before_exec=None
    )
    namespace["_early_disconnect_failed_notebook"](result)
    namespace["_early_disconnect_failed_notebook"](result)
    assert unassign_calls == ["unassign"]
