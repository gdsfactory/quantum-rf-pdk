"""Transmon qubit components."""

from __future__ import annotations

import operator
from functools import reduce

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.junction import squid_junction
from qpdk.tech import LAYER


@gf.cell(check_instances=False)
def double_pad_transmon(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 15.0,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Creates a double capacitor pad transmon qubit with Josephson junction.

    A transmon qubit consists of two capacitor pads connected by a Josephson junction.
    The junction creates an anharmonic oscillator that can be used as a qubit.

    See :cite:`kochChargeinsensitiveQubitDesign2007a` for details.

    Args:
        pad_size: (width, height) of each capacitor pad in μm.
        pad_gap: Gap between the two capacitor pads in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.

    Returns:
        Component: A gdsfactory component with the transmon geometry.
    """
    c = Component()
    pad_width, pad_height = pad_size

    def create_capacitor_pad(x_offset: float) -> gf.ComponentReference:
        pad = gf.components.rectangle(
            size=pad_size,
            layer=layer_metal,
        )
        pad_ref = c.add_ref(pad)
        pad_ref.move((x_offset, -pad_height / 2))
        return pad_ref

    create_capacitor_pad(-pad_width - pad_gap / 2)
    create_capacitor_pad(pad_gap / 2)

    # Create Josephson junction
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    junction_ref.rotate(45)
    # Center the junction between the pads
    junction_ref.dcenter = c.dcenter  # move((-junction_height / 2, 0))
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Add ports for connections
    c.add_port(
        name="left_pad",
        center=(-pad_width - pad_gap / 2, 0),
        width=pad_height,
        orientation=180,
        layer=layer_metal,
    )
    c.add_port(
        name="right_pad",
        center=(pad_width + pad_gap / 2, 0),
        width=pad_height,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
    )

    # Add metadata
    c.info["qubit_type"] = "transmon"

    return c


@gf.cell(check_instances=False)
def flipmon(
    inner_ring_radius: float = 50,
    inner_ring_width: float = 30,
    outer_ring_radius: float = 110,
    outer_ring_width: float = 60,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Creates a circular transmon qubit with `flipmon` geometry.

    A circular variant of the transmon qubit with another circle as the inner pad.

    See :cite:`liVacuumgapTransmonQubits2021,liCosmicrayinducedCorrelatedErrors2025`
    for details about the `flipmon` design.

    Args:
        inner_ring_radius: Central radius of the inner circular capacitor pad in μm.
        inner_ring_width: Width of the inner circular capacitor pad in μm.
        outer_ring_radius: Central radius of the outer circular capacitor pad in μm.
        outer_ring_width: Width of the outer circular capacitor pad in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.

    Returns:
        Component: A gdsfactory component with the circular transmon geometry.
    """
    c = Component()

    def create_circular_pad(radius: float, width: float) -> gf.ComponentReference:
        pad = gf.c.ring(
            radius=radius,
            width=width,
            layer=layer_metal,
        )
        return c.add_ref(pad)

    create_circular_pad(inner_ring_radius, inner_ring_width)
    create_circular_pad(outer_ring_radius, outer_ring_width)

    # Create Josephson junction
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    # Center the junction between the pads
    # junction_ref.rotate(45)
    junction_ref.dcenter = c.dcenter  # move((-junction_height / 2, 0))
    junction_ref.dcplx_trans *= reduce(
        operator.mul,
        (
            DCplxTrans(
                (
                    inner_ring_radius
                    + inner_ring_width / 2
                    + outer_ring_radius
                    - outer_ring_width / 2
                )
                / 2,
                0,
            ),
            DCplxTrans(1, 45, False, 0, 0),
            # DCplxTrans(1, 45, False, 0, 0),
        ),
    )
    junction_ref.y = 0

    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Add ports for connections
    c.add_port(
        name="inner_ring_near_junction",
        center=(inner_ring_radius + inner_ring_width / 2, 0),
        width=inner_ring_width,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="outer_ring_near_junction",
        center=(outer_ring_radius - outer_ring_width / 2, 0),
        width=outer_ring_width,
        orientation=180,
        layer=layer_metal,
    )
    c.add_port(
        name="outer_ring_outside",
        center=(outer_ring_radius + outer_ring_width / 2, 0),
        width=outer_ring_width,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
    )

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = gf.Component()
    # (c << double_pad_transmon()).move((0, 0))
    # (c << double_pad_transmon(junction_displacement=DCplxTrans(0, 150))).move((0, 600))
    (c << flipmon()).move((0, 1200))
    c.show()
