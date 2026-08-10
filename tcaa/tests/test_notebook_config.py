"""Regression checks for the executable configuration cells in the Colab notebook."""

import json
from pathlib import Path

from tcaa.training_core import DEFAULT_DIRICHLET_ALPHA


def _notebook_code_cell_containing(needle: str) -> str:
    notebook_path = Path(__file__).resolve().parents[2] / "TCAA_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if needle in source:
            return source
    raise AssertionError(f"Notebook code cell containing {needle!r} not found")


def test_base_notebook_config_is_self_contained():
    """Running the shared config cell first must not depend on a later import."""
    source = _notebook_code_cell_containing("TCAA_CONFIG = {")
    namespace = {}
    exec(compile(source, "<TCAA_CONFIG>", "exec"), namespace)

    assert namespace["TCAA_CONFIG"]["dirichlet_alpha"] == DEFAULT_DIRICHLET_ALPHA == 1.0
