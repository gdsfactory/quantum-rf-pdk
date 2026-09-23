"""Neutral geometry preparation shared by the AEDT and COMSOL exporters.

This module is pure gdsfactory geometry: it turns a QPDK mask into the physical
metal a field solver actually sees. It has no PyAEDT, MPh, or COMSOL imports so
both exporters can depend on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gdsfactory as gf

from qpdk.tech import LAYER
from qpdk.utils import (
    add_margin_to_layer,
    apply_additive_metals,
    invert_mask_polarity,
    remove_metadata_layers,
)

if TYPE_CHECKING:
    from gdsfactory.component import Component


def prepare_metal_layout(
    component: Component,
    margin_draw: float = 0.0,
    margin_etch: float = 0.0,
    *,
    name: str | None = None,
) -> Component:
    """Fold a QPDK mask into positive metal geometry for a field solver.

    Applies the additive metals to the etch layer, inverts the mask polarity
    around the component bounding box, grows the ground by ``margin_draw``,
    drops the etch layers, and re-adds the original ports.

    Args:
        component: The component to prepare.
        margin_draw: Margin in µm added to the M1/M2 draw layers.
        margin_etch: Margin in µm added to the M1/M2 etch layers before
            inversion.
        name: Name for the initial preparation cell. Defaults to
            ``f"{component.name}_metal"``.

    Returns:
        A copy of the component prepared for simulation.
    """
    c = gf.Component(name=name or f"{component.name}_metal")
    c << component.copy()
    if margin_etch > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_ETCH, margin_etch),
                (LAYER.M2_ETCH, margin_etch),
            ],
        )
    c = apply_additive_metals(c)
    c = invert_mask_polarity(c)
    if margin_draw > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_DRAW, margin_draw),
                (LAYER.M2_DRAW, margin_draw),
            ],
        )
    c = c.remove_layers(layer for layer in LAYER if str(layer).endswith("_ETCH"))  # type: ignore[attr-defined]
    c = remove_metadata_layers(c)
    c.add_ports(component.ports)
    return c
