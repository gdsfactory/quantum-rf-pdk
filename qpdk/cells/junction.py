"""Josephson junction components."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.waveguides import straight
from qpdk.helper import show_components
from qpdk.tech import (
    LAYER,
    josephson_junction_cross_section_narrow,
    josephson_junction_cross_section_wide,
)


@gf.cell
def single_josephson_junction_wire(
    wide_straight_length: float = 8.3,
    narrow_straight_length: float = 0.5,
    taper_length: float = 4.7,
    cross_section_wide: CrossSectionSpec = josephson_junction_cross_section_wide,
    cross_section_narrow: CrossSectionSpec = josephson_junction_cross_section_narrow,
    layer_patch: LayerSpec = LAYER.JJ_PATCH,
    size_patch: tuple[float, float] = (1.5, 1.0),
) -> Component:
    r"""Creates a single wire to use in a Josephson junction.

    .. svgbob::

        ┌───┐
        │o1 │━━━━────╶╶╶╶╶ o2
        └───┘
               wide  narrow

    Args:
        wide_straight_length: Length of the wide straight section in µm.
        narrow_straight_length: Length of the narrow straight section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_wide: Cross-section specification for the wide section.
        cross_section_narrow: Cross-section specification for the narrow section.
        layer_patch: Layer for the patch that creates the overlap region.
        size_patch: Size of the patch that creates the overlap region.
    """
    c = Component()

    # Widest straight section with patch
    wide_straight_ref = c << straight(
        length=wide_straight_length, cross_section=cross_section_wide
    )

    # Add the tapered transition section
    taper_ref = c << gf.c.taper_cross_section(
        length=taper_length,
        cross_section1=cross_section_wide,
        cross_section2=cross_section_narrow,
        linear=True,
    )

    # Narrow straight section with overlap
    narrow_straight_ref = c << straight(
        length=narrow_straight_length, cross_section=cross_section_narrow
    )

    # Connect all
    taper_ref.connect("o1", wide_straight_ref.ports["o2"])
    narrow_straight_ref.connect("o1", taper_ref.ports["o2"])

    # Add patch to wide section
    if layer_patch:
        patch = c << gf.components.rectangle(
            size=size_patch,
            layer=layer_patch,
            centered=True,
        )
        # Overlap with one fourth offset to one side
        patch.move(
            (wide_straight_ref.dbbox().p1.x - size_patch[0] / 4, wide_straight_ref.y)
        )

    # Add port at wide end
    c.add_port(
        port=wide_straight_ref.ports["o1"], name="o1", cross_section=cross_section_wide
    )
    # Add port at narrow end
    c.add_port(
        port=narrow_straight_ref.ports["o2"],
        name="o2",
        cross_section=cross_section_narrow,
    )

    return c


@gf.cell
def josephson_junction(
    junction_overlap_displacement: float = 1.8,
    wide_straight_length: float = 8.3,
    narrow_straight_length: float = 0.5,
    taper_length: float = 4.7,
    cross_section_wide: CrossSectionSpec = josephson_junction_cross_section_wide,
    cross_section_narrow: CrossSectionSpec = josephson_junction_cross_section_narrow,
    layer_patch: LayerSpec = LAYER.JJ_PATCH,
    size_patch: tuple[float, float] = (1.5, 1.0),
) -> Component:
    r"""Creates a single Josephson junction component.

    A Josephson junction consists of two superconducting electrodes separated
    by a thin insulating barrier allowing tunnelling.

    .. svgbob::

         right_wide
         ┌───┐          ╷ overlap
         │   │━━━━────╶╶╷╶╶
         └───┘          ╷
                        │
                        │
                        ┃
                        ┃
                      ┌───┐
                      │   │
                      └───┘
                      left_wide

    Args:
        junction_overlap_displacement: Displacement of the overlap region in µm.
            Measured from the centers of the junction ports.
        wide_straight_length: Length of the wide straight section in µm.
        narrow_straight_length: Length of the narrow straight section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_wide: Cross-section specification for the wide section.
        cross_section_narrow: Cross-section specification for the narrow section.
        layer_patch: Layer for the patch that creates the overlap region.
        size_patch: Size of the patch that creates the overlap region.
    """
    c = Component()

    # Left wire
    left_wire = c << single_josephson_junction_wire(
        wide_straight_length=wide_straight_length,
        narrow_straight_length=narrow_straight_length,
        taper_length=taper_length,
        cross_section_wide=cross_section_wide,
        cross_section_narrow=cross_section_narrow,
        layer_patch=layer_patch,
        size_patch=size_patch,
    )

    # Right wire
    right_wire = c << single_josephson_junction_wire(
        wide_straight_length=wide_straight_length,
        narrow_straight_length=narrow_straight_length,
        taper_length=taper_length,
        cross_section_wide=cross_section_wide,
        cross_section_narrow=cross_section_narrow,
        layer_patch=layer_patch,
        size_patch=size_patch,
    )

    total_length = wide_straight_length + narrow_straight_length + taper_length
    # Position left wire on top of right wire with rotation
    left_wire.dcplx_trans = (
        right_wire.ports["o2"].dcplx_trans
        * DCplxTrans.R90
        * DCplxTrans(
            -total_length + junction_overlap_displacement,
            0,
        )
    )
    right_wire.dcplx_trans *= DCplxTrans(junction_overlap_displacement, 0)

    # Add ports at wide ends
    c.add_port(
        name="left_wide",
        port=left_wire.ports["o1"],
    )
    c.add_port(
        name="right_wide",
        port=right_wire.ports["o1"],
    )
    # One port at overlap
    c.add_port(
        name="overlap",
        center=(
            left_wire.ports["o2"].dcplx_trans
            * DCplxTrans(-junction_overlap_displacement, 0)
        ).disp.to_p(),
        width=left_wire.ports["o2"].width,
        orientation=left_wire.ports["o2"].orientation,
        layer=left_wire.ports["o2"].layer,
        port_type=left_wire.ports["o2"].port_type,
    )
    # breakpoint()

    return c


@gf.cell
def squid_junction(
    junction_spec: ComponentSpec = josephson_junction,
    loop_area: float = 4,
) -> Component:
    """Creates a SQUID (Superconducting Quantum Interference Device) junction component.

    A SQUID consists of two Josephson junctions connected in parallel, forming a loop.

    See :cite:`clarkeSQUIDHandbook2004` for details.

    Args:
        junction_spec: Component specification for the Josephson junction component.
        loop_area: Area of the SQUID loop in µm².
            This does not take into account the junction wire widths.
    """
    c = Component()

    junction_comp = gf.get_component(junction_spec)

    left_junction = c << junction_comp
    right_junction = c << junction_comp

    # Form a cross by positioning overlaps on top of each other
    right_junction.dcplx_trans = (
        left_junction.ports["overlap"].dcplx_trans
        * DCplxTrans.R90
        * DCplxTrans(
            -left_junction.xmax
            + (left_junction.xmax - left_junction.ports["overlap"].x),
            0,
        )
    )

    # Start adding area by displacing junctions
    displacement_xy = loop_area**0.5
    right_junction.dcplx_trans *= DCplxTrans((displacement_xy,) * 2)

    # Add ports from junctions with descriptive names
    for junction_name, junction in [("left", left_junction), ("right", right_junction)]:
        for port_side in ["left", "right"]:
            port_name = f"{junction_name}_{port_side}_wide"
            c.add_port(name=port_name, port=junction.ports[f"{port_side}_wide"])

    # Overlaps and their center
    c.add_port(name="left_overlap", port=left_junction.ports["overlap"])
    c.add_port(name="right_overlap", port=right_junction.ports["overlap"])
    c.add_port(
        name="loop_center",
        center=(
            (
                left_junction.ports["overlap"].dcplx_trans.disp
                + right_junction.ports["overlap"].dcplx_trans.disp
            )
            / 2
        ).to_p(),
        layer=left_junction.ports["overlap"].layer,
        width=left_junction.ports["overlap"].width,
    )
    return c


if __name__ == "__main__":
    show_components(josephson_junction, squid_junction)
