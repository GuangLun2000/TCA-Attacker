from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcaa.visualize import RASTER_DPI, _use_log_scale_if_needed, apply_style, save_figure


def test_apply_style_uses_publication_defaults():
    apply_style()

    assert plt.rcParams["savefig.dpi"] == RASTER_DPI
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["axes.spines.top"] is True
    assert plt.rcParams["axes.spines.right"] is True
    assert plt.rcParams["legend.frameon"] is True
    assert plt.rcParams["grid.linestyle"] == "--"
    assert plt.rcParams["pdf.fonttype"] == 42


def test_save_figure_writes_crisp_raster_and_vector(tmp_path: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1], marker="o")

    written = save_figure(fig, tmp_path / "example.png")
    plt.close(fig)

    png = tmp_path / "example.png"
    pdf = tmp_path / "example.pdf"
    assert written == [png, pdf]
    assert png.stat().st_size > 0
    assert pdf.read_bytes().startswith(b"%PDF")

    # 2 x 1.5 inches at 600 DPI (allow a few pixels for backend rounding).
    image = plt.imread(png)
    assert image.shape[1] >= 1195
    assert image.shape[0] >= 895


def test_large_dynamic_range_gets_an_explicit_log_axis():
    apply_style()
    fig, ax = plt.subplots()
    ax.set_ylabel("cost amplification")

    assert _use_log_scale_if_needed(ax, [10, 12, 100_000]) is True
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "cost amplification (log scale)"
    plt.close(fig)
