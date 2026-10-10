"""Matplotlib helpers for COMSOL layouts and fields.

Re-exported from :mod:`gplugins.comsol.plotting`.

The names load from gplugins on first use, so this module imports without the
``comsol`` extra and only using a name needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.simulation.comsol._gplugins import reexport

if TYPE_CHECKING:
    from gplugins.comsol.plotting import (
        draw_cut_plane_field,
        draw_layout_polygons,
    )

__all__ = [
    "draw_cut_plane_field",
    "draw_layout_polygons",
]

__getattr__ = reexport(__name__, "plotting", __all__)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
