"""Smoke tests for the COMSOL notebook plots built from saved exports."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib_inline import backend_inline

from qpdk.simulation.comsol.plotting import (
    apply_qpdk_style,
    draw_cut_plane_field,
    draw_layout_polygons,
    prefer_svg_figures,
)

plt.switch_backend("Agg")


def test_notebook_uses_the_checkout_plot_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    with plt.rc_context():
        assert Path(apply_qpdk_style()).name == "qpdk.mplstyle"
        assert plt.rcParams["font.sans-serif"] == ["sans-serif"]


def test_notebook_plot_style_has_a_portable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_source: object) -> None:
        raise OSError

    monkeypatch.setattr(plt.style, "use", unavailable)
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    with plt.rc_context():
        assert apply_qpdk_style() == "matplotlib defaults"
        assert plt.rcParams["font.sans-serif"] == ["sans-serif"]


def test_layout_plot_keeps_the_ground_holes_and_small_pads_visible() -> None:
    ground = SimpleNamespace(
        outline=[(-10, -10), (10, -10), (10, 10), (-10, 10)],
        holes=[[(-2, -2), (2, -2), (2, 2), (-2, 2)]],
    )
    pad = SimpleNamespace(outline=[(-1, -1), (1, -1), (1, 1), (-1, 1)], holes=[])
    fig, ax = plt.subplots()
    try:
        draw_layout_polygons(ax, [pad, ground])
        assert len(ax.patches) == 3
        assert ax.patches[0].get_facecolor() == pytest.approx((0.78, 0.78, 0.78, 1))
        assert ax.patches[1].get_facecolor() == pytest.approx((1, 1, 1, 1))
        assert np.ptp(ax.patches[2].get_xy()[:, 0]) == pytest.approx(2)
    finally:
        plt.close(fig)


def test_field_plot_crops_saved_export_and_uses_log_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    field = np.column_stack((
        x.ravel(),
        y.ravel(),
        np.zeros(x.size),
        np.exp(x.ravel() + y.ravel()),
    ))
    field = np.vstack((field, [100, 100, 0, 1e12]))
    export = tmp_path / "field.txt"
    np.savetxt(export, field, header="% COMSOL field export", comments="")
    monkeypatch.setattr(plt, "show", lambda: None)
    before = set(plt.get_fignums())
    try:
        draw_cut_plane_field(
            export, "Resonator field", view_um=(-2, 2, -2, 2), stride=2
        )
        number = next(iter(set(plt.get_fignums()) - before))
        fig = plt.figure(number)
        ax = fig.axes[0]
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-2, 2)
        assert ax.get_title() == "Resonator field"
        assert ax.collections
        assert fig.axes[1].get_ylabel() == r"$|\mathbf{E}|$ (V/m)"
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def test_field_plot_rejects_a_crop_without_nodes(tmp_path: Path) -> None:
    export = tmp_path / "field.txt"
    np.savetxt(export, [[0, 0, 0, 1], [1, 0, 0, 2]])
    with pytest.raises(ValueError, match="No exported field nodes"):
        draw_cut_plane_field(export, "Outside", view_um=(10, 20, 10, 20))


def test_notebook_requests_svg_output(monkeypatch: pytest.MonkeyPatch) -> None:
    formats: list[str] = []
    monkeypatch.setattr(
        backend_inline, "set_matplotlib_formats", lambda *args: formats.extend(args)
    )
    prefer_svg_figures()
    assert formats == ["svg", "png"]
    assert plt.rcParams["svg.fonttype"] == "path"
