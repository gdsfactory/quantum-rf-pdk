"""Mesh helpers for COMSOL models, re-exported from :mod:`gplugins.comsol.mesh`."""

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
