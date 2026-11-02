"""Declare intended near-field coupling in a layout.

The design-profile DRC rules warn when two signal nets run closer than the
crosstalk distance. Coupled cells are supposed to be close, so they declare the
region where the coupling is intended on ``LAYER.COUPLING``. The marker is
geometry, so it travels with the GDS to the hosted DRC, and
:func:`qpdk.utils.remove_metadata_layers` strips it before tapeout.

Only mark the corridor between the two parts a cell couples on purpose. Marking
everything below the threshold would also hide unintended proximity.
"""

from __future__ import annotations

from typing import Any

from gdsfactory.component import Component
from kfactory import kdb

from qpdk.tech import LAYER


def reference_region(ref: Any, layer: tuple[int, int] = LAYER.M1_DRAW) -> kdb.Region:
    """Return the polygons of a reference on ``layer`` in the parent frame.

    Flattening the reference hierarchy in one ``Region(iterator)`` call keeps
    the read in C++; walking the shapes in Python is several times slower.

    Args:
        ref: Component reference placed in the parent cell.
        layer: GDS layer to read.

    Returns:
        The reference's shapes on that layer, transformed to the parent.
    """
    layer_index = ref.cell.kcl.layer(*layer)
    region = kdb.Region(ref.cell.kdb_cell.begin_shapes_rec(layer_index))
    return region.transformed(ref.trans)


def mark_coupling_regions(
    component: Component,
    ref_a: Any,
    ref_b: Any,
    *,
    gap: float,
    layer: tuple[int, int] = LAYER.M1_DRAW,
) -> None:
    """Draw a ``COUPLING`` marker between two references ``gap`` apart.

    The marker covers every point within ``gap`` of both references, minus the
    conductors themselves and anything outside their common bounding box. For a
    pair drawn ``gap`` apart that is the whole space between them.

    Args:
        component: Parent component that receives the marker.
        ref_a: First coupled reference.
        ref_b: Second coupled reference.
        gap: Edge-to-edge separation of the two references, in µm.
        layer: Metal layer the two references couple on.

    Raises:
        ValueError: If ``gap`` is not positive.
    """
    reach = round(gap / component.kcl.dbu)
    if reach <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    region_a = reference_region(ref_a, layer)
    region_b = reference_region(ref_b, layer)
    pair = region_a | region_b
    if pair.is_empty():
        return
    corridor = (region_a.sized(reach) & region_b.sized(reach)) - pair
    corridor &= kdb.Region(pair.bbox())
    if corridor.is_empty():
        return
    component.shapes(component.kcl.layer(*LAYER.COUPLING)).insert(corridor)


__all__ = ["mark_coupling_regions", "reference_region"]
