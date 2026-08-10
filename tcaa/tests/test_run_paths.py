from pathlib import Path

import pytest

from tcaa.run_paths import resolve_results_root, resolve_run_dir, stamp_run_subdir


def test_custom_results_root_isolated_from_legacy_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    isolated = tmp_path / "reasoning-v2"
    cfg = stamp_run_subdir({
        "results_root": str(isolated),
        "results_subdir": "fl",
    })

    out_dir = resolve_run_dir(cfg)
    out_dir.mkdir(parents=True)

    assert out_dir.parent.parent == isolated
    assert out_dir.name == "fl"
    assert not (tmp_path / "results").exists()


def test_stamp_collision_is_checked_in_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr("tcaa.run_paths.time.strftime", lambda _fmt: "20260810-120000")
    (tmp_path / "20260810-120000").mkdir()

    cfg = stamp_run_subdir({
        "results_root": str(tmp_path),
        "results_subdir": "tcaa_fl",
    })

    assert cfg["results_subdir"] == "20260810-120000-2/tcaa_fl"


@pytest.mark.parametrize("subdir", ["/tmp/outside", "../outside", "ok/../../outside"])
def test_results_subdir_cannot_escape_root(tmp_path, subdir):
    with pytest.raises(ValueError, match="stay inside"):
        stamp_run_subdir({"results_root": str(tmp_path), "results_subdir": subdir})


def test_stamping_is_idempotent_inside_a_call_tree(tmp_path):
    once = stamp_run_subdir({
        "results_root": str(tmp_path),
        "results_subdir": Path("sweep/seed_1"),
    })
    twice = stamp_run_subdir(once)

    assert twice is once
    assert resolve_results_root(twice) == tmp_path
