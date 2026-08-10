import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "TCAA_Reasoning_Colab.ipynb"


def _code_bundle_sha256() -> str:
    files = sorted((ROOT / "tcaa").glob("*.py"))
    files.append(ROOT / "requirements-reasoning-colab.txt")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_reasoning_notebook_is_valid_isolated_and_source_locked():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")

    assert "__CODE_BUNDLE_SHA256__" not in code
    assert f"EXPECTED_CODE_BUNDLE_SHA256 = '{_code_bundle_sha256()}'" in code
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
        assert config["reasoning_min_claim_cost_ratio"] > 1.0
        assert config["reasoning_min_hardware_ratio"] > 1.0
        assert config["resource_profile_repeats"] <= len(config["eval_decode_seeds"])
        if tier == "formal":
            assert config["resource_profile_eval_size"] == 16
            assert config["resource_profile_repeats"] == 3
