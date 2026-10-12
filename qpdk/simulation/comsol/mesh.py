"""Mesh helpers for COMSOL models, re-exported from :mod:`gplugins.comsol.mesh`.

The names load from gplugins on first use, so this module imports without the
``comsol`` extra and only using a name needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.simulation.comsol._gplugins import reexport

if TYPE_CHECKING:
    from gplugins.comsol.mesh import (
        DEFAULT_SIZE_TAG,
        EDGE_SIZE_TAG,
        FREE_TET_TAG,
        GENERATED_FREE_TET_TAG,
        pin_absolute_edge_mesh_sizes,
        pin_absolute_mesh_sizes,
        refine_metal_plane_mesh,
    )

__all__ = [
    "DEFAULT_SIZE_TAG",
    "EDGE_SIZE_TAG",
    "FREE_TET_TAG",
    "GENERATED_FREE_TET_TAG",
    "pin_absolute_edge_mesh_sizes",
    "pin_absolute_mesh_sizes",
    "refine_metal_plane_mesh",
]

__getattr__ = reexport(__name__, "mesh", __all__)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
