"""Visualization exports for QPDK's negative-tone masks."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.typings import LayerSpecs

from qpdk.tech import (
    LAYER,
    LAYER_STACK_FLIP_CHIP_NO_VACUUM,
    LAYER_STACK_NO_VACUUM,
    LAYER_VIEWS,
)

if TYPE_CHECKING:
    from trimesh.scene import Scene


def to_3d(
    component: Component,
    layer_views: LayerViews | None = None,
    layer_stack: LayerStack | None = None,
    exclude_layers: LayerSpecs | None = None,
    *,
    margin: float = 10.0,
) -> Scene:
    """Export a QPDK mask as physical metal and substrate in a trimesh scene.

    QPDK etch polygons describe gaps in a continuous metal film. Their 3D
    representation needs a finite ``SIM_AREA`` from which to subtract those
    gaps, with additive draw polygons restoring metal where they overlap.
    If absent, this helper adds that area around a copy of the component.
    An existing ``SIM_AREA`` is used unchanged, and the source mask is preserved.

    Args:
        component: QPDK component to visualize.
        layer_views: Layer colors. Defaults to QPDK's layer views.
        layer_stack: Stack to extrude. Defaults to the QPDK stack without vacuum,
            or the flip-chip stack without vacuum when M2 polygons are present.
        exclude_layers: Layers to omit from the scene.
        margin: Margin in µm around the bounding box when adding ``SIM_AREA``.
            Ignored when the component already defines a simulation area.

    Returns:
        A scene that can be viewed with ``scene.show()`` or exported to a file.
        Requires the optional ``qpdk[graphics]`` dependencies.

    Raises:
        ValueError: If margin is negative or non-finite.

    Example:
        >>> from qpdk import PDK, cells, to_3d
        >>> PDK.activate()
        >>> scene = to_3d(cells.quarter_wave_resonator_coupled())
    """
    if not isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")

    if layer_stack is None:
        has_m2 = any(
            not component.get_region(layer).is_empty()
            for layer in (LAYER.M2_DRAW, LAYER.M2_ETCH)
        )
        layer_stack = (
            LAYER_STACK_FLIP_CHIP_NO_VACUUM if has_m2 else LAYER_STACK_NO_VACUUM
        )

    if component.get_region(LAYER.SIM_AREA).is_empty():
        component = component.dup()
        bbox = component.bbox()
        component.add_polygon(
            [
                (bbox.left - margin, bbox.bottom - margin),
                (bbox.right + margin, bbox.bottom - margin),
                (bbox.right + margin, bbox.top + margin),
                (bbox.left - margin, bbox.top + margin),
            ],
            layer=LAYER.SIM_AREA,
        )

    return component.to_3d(
        layer_views=layer_views or LAYER_VIEWS,
        layer_stack=layer_stack,
        exclude_layers=exclude_layers,
    )
