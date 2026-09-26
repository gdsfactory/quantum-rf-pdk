"""Matplotlib helpers shared by the COMSOL notebooks.

Matplotlib, numpy, and the QPDK style only: importing this module needs neither
MPh nor a COMSOL license, so the result cells still draw saved exports where no
solver is installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes as mpl_axes, font_manager
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon

from qpdk.config import PATH


def _outfit_titles() -> None:
    """Draw plot titles in Outfit bold, matching the documentation headings."""
    original_set_title = mpl_axes.Axes.set_title

    def _set_title(self: mpl_axes.Axes, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("fontfamily", "Outfit")
        kwargs.setdefault("fontweight", "bold")
        return original_set_title(self, *args, **kwargs)

    mpl_axes.Axes.set_title = _set_title


def apply_qpdk_style() -> str:
    """Apply the QPDK plot style, falling back to matplotlib's own defaults.

    The style is ``docs/qpdk.mplstyle`` in a checkout and the installed ``qpdk``
    style in a documentation environment; a downloaded notebook outside both
    keeps matplotlib's defaults instead of failing. The documentation fonts are
    used when they are installed, and matplotlib's bundled families otherwise.

    Returns:
        A short description of the style that was applied.
    """
    for source in (PATH.repo / "docs" / "qpdk.mplstyle", "qpdk"):
        try:
            plt.style.use(source)
        except OSError:
            continue
        applied = str(source)
        break
    else:
        applied = "matplotlib defaults"

    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [
        name
        for name in ("Inter", "Outfit", "DejaVu Sans", "Helvetica", "Arial")
        if name in installed
    ] + ["sans-serif"]
    if "Outfit" in installed:
        _outfit_titles()
    return applied


def prefer_svg_figures() -> None:
    """Save every figure as SVG as well as PNG, so stored outputs stay vector.

    The saved cell outputs are what the documentation renders, and both the HTML
    and the Typst PDF build embed the SVG ahead of the PNG. The PNG is kept as a
    fallback for a viewer that cannot render SVG, and text is written as paths so
    the figures carry their own glyphs instead of relying on installed fonts.
    Outside a notebook kernel there is no inline backend to configure, so a
    plain script run keeps matplotlib's PNG default.
    """
    try:
        # Ships with ipykernel, so it is present in a notebook kernel only.
        from matplotlib_inline.backend_inline import (  # ruff: ignore[import-outside-top-level]
            set_matplotlib_formats,
        )
    except ImportError:
        return
    plt.rcParams["svg.fonttype"] = "path"
    set_matplotlib_formats("svg", "png")


def draw_layout_polygons(ax: mpl_axes.Axes, polygons: Iterable[Any]) -> None:
    """Draw prepared metal polygons as filled patches, largest first.

    Largest first keeps the small pads and the ground plane visible whichever
    order the layout hands them back in.

    Args:
        ax: Axes to draw on.
        polygons: Polygons with ``outline`` points and ``holes``, as a prepared
            layout stores them.
    """

    def area(polygon: Any) -> float:
        xs = [x for x, _ in polygon.outline]
        ys = [y for _, y in polygon.outline]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    for polygon in sorted(polygons, key=lambda item: -area(item)):
        ax.add_patch(
            MplPolygon(
                polygon.outline,
                closed=True,
                facecolor="0.78",
                edgecolor="0.35",
                linewidth=0.6,
                zorder=1,
            )
        )
        for hole in polygon.holes:
            ax.add_patch(
                MplPolygon(
                    hole,
                    closed=True,
                    facecolor="white",
                    edgecolor="0.35",
                    linewidth=0.6,
                    zorder=1,
                )
            )


def draw_cut_plane_field(
    file: Path,
    title: str,
    *,
    view_um: tuple[float, float, float, float],
    stride: int = 4,
    contour_levels: int = 30,
) -> None:
    """Draw a cropped ``emw.normE`` map from a COMSOL cut-plane export.

    The export covers the whole prepared box, so the nodes are cropped to
    ``view_um`` and, for display only, drawn on a fixed-stride subset with
    percentiles of the cropped data as the colour limits. The stride changes what
    is drawn, not what was solved.

    Args:
        file: A ``emw.normE`` export on the cut plane, columns x, y, z, E.
        title: Figure title.
        view_um: ``(xmin, xmax, ymin, ymax)`` crop window, in µm.
        stride: Drawing stride over the cropped nodes.
        contour_levels: Number of contour levels.

    Raises:
        ValueError: If the crop window holds no exported node.
    """
    field = np.loadtxt(file, comments="%")
    field_x, field_y, field_e = field[:, 0], field[:, 1], field[:, 3]
    inside = (
        (field_x >= view_um[0])
        & (field_x <= view_um[1])
        & (field_y >= view_um[2])
        & (field_y <= view_um[3])
    )
    view_x, view_y, view_e = field_x[inside], field_y[inside], field_e[inside]
    if view_e.size == 0:
        raise ValueError(f"No exported field nodes inside {view_um}")

    # Colour limits from the data in the window rather than the full-domain
    # maximum, which sits far outside it and flattens everything on the scale.
    color_min, color_max = (float(value) for value in np.percentile(view_e, [1, 99]))
    draw_x, draw_y, draw_e = view_x, view_y, view_e
    if stride > 1 and view_e.size // stride >= 10:
        draw_x = view_x[::stride]
        draw_y = view_y[::stride]
        draw_e = view_e[::stride]
    print(  # ruff: ignore[print]
        f"Field nodes: {view_e.size} of {field_e.size} inside the view; "
        f"{draw_e.size} drawn at stride {stride}; "
        f"range {view_e.min():.3g} to {view_e.max():.3g} V/m, "
        f"1st to 99th percentile {color_min:.3g} to {color_max:.3g} V/m"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.tricontourf(
        draw_x,
        draw_y,
        draw_e,
        levels=np.geomspace(color_min, color_max, contour_levels),
        norm=LogNorm(vmin=color_min, vmax=color_max),
        cmap="inferno",
        extend="both",
    )
    ax.set_xlim(view_um[0], view_um[1])
    ax.set_ylim(view_um[2], view_um[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label=r"$|\mathbf{E}|$ (V/m)")
    plt.tight_layout()
    plt.show()
