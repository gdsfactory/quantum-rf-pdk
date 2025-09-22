"""Josephson junction components."""

from __future__ import annotations

from operator import itemgetter
from typing import TypedDict, Unpack

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.waveguides import straight
from qpdk.helper import show_components
from qpdk.tech import (
    LAYER,
    josephson_junction_cross_section_narrow,
    josephson_junction_cross_section_wide,
)


class SingleJosephsonJunctionWireParams(TypedDict):
    """Type definition for single Josephson junction wire parameters.

    Args:
        wide_straight_length: Length of the wide straight section in µm.
        narrow_straight_length: Length of the narrow straight section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_wide: Cross-section specification for the wide section.
        cross_section_narrow: Cross-section specification for the narrow section.
        layer_patch: Layer for the patch that creates the overlap region.
        size_patch: Size of the patch that creates the overlap region.
    """

    wide_straight_length: float
    narrow_straight_length: float
    taper_length: float
    cross_section_wide: LayerSpec
    cross_section_narrow: LayerSpec
    layer_patch: LayerSpec
    size_patch: tuple[float, float]


_single_josephson_junction_wire_defaults = SingleJosephsonJunctionWireParams(
    wide_straight_length=8.3,
    narrow_straight_length=0.5,
    taper_length=4.7,
    cross_section_wide=josephson_junction_cross_section_wide,
    cross_section_narrow=josephson_junction_cross_section_narrow,
    layer_patch=LAYER.JJ_PATCH,
    size_patch=(1.5, 1.0),
)


@gf.cell
def single_josephson_junction_wire(
    **kwargs: Unpack[SingleJosephsonJunctionWireParams],
) -> Component:
    """Creates a single wire to use in a Josephson junction.

    Args:
        kwargs: :class:`~SingleJosephsonJunctionWireParams` parameters.
    """
    c = Component()
    wire_params = _single_josephson_junction_wire_defaults | kwargs
    (
        wide_straight_length,
        narrow_straight_length,
        taper_length,
        cross_section_wide,
        cross_section_narrow,
        layer_patch,
        size_patch,
    ) = itemgetter(
        "wide_straight_length",
        "narrow_straight_length",
        "taper_length",
        "cross_section_wide",
        "cross_section_narrow",
        "layer_patch",
        "size_patch",
    )(wire_params)

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
    **kwargs: Unpack[SingleJosephsonJunctionWireParams],
) -> Component:
    """Creates a single Josephson junction component.

    A Josephson junction consists of two superconducting electrodes separated
    by a thin insulating barrier allowing tunnelling.

    Args:
        junction_overlap_displacement: Displacement of the overlap region in µm.
            Measured from the centers of the junction ports
        kwargs: :class:`~SingleJosephsonJunctionWireParams` for single wires.
    """
    c = Component()

    # Wire configuration parameters
    wire_params = _single_josephson_junction_wire_defaults | kwargs

    # Left wire
    left_wire = c << single_josephson_junction_wire(**wire_params)

    # Right wire
    right_wire = c << single_josephson_junction_wire(**wire_params)

    total_length = sum(
        map(  # pyright: ignore[reportArgumentType]
            wire_params.get,
            ("wide_straight_length", "narrow_straight_length", "taper_length"),
        )
    )
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


@gf.cell
def tunable_coupler_flux(
    squid_spec: ComponentSpec = squid_junction,
    coupling_pad_size: tuple[float, float] = (50.0, 100.0),
    coupling_gap: float = 10.0,
    flux_line_width: float = 5.0,
    flux_line_length: float = 100.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    port_type: str = "electrical",
) -> Component:
    """Creates a flux-tunable coupler with Josephson junction.

    A flux-tunable coupler uses a SQUID that can be magnetically flux-biased
    to control the coupling strength between two quantum circuits. The coupling
    strength varies as cos(πΦ/Φ₀) where Φ is the external flux and Φ₀ is the
    flux quantum.

    See :cite:`harrisSingleQuantumCoupler2007` for details.

    Args:
        squid_spec: Component specification for the SQUID junction.
        coupling_pad_size: (width, height) of coupling pads in μm.
        coupling_gap: Gap between coupling pads and SQUID in μm.
        flux_line_width: Width of flux bias line in μm.
        flux_line_length: Length of flux bias line in μm.
        layer_metal: Layer for metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the flux-tunable coupler geometry.

    .. code::

                    flux_bias
                        |
                        |
           pad1    ┌─────────┐    pad2
        ───────────┤   SQUID   ├───────────
                   └─────────┘
    """
    c = Component()

    # Create the central SQUID
    squid_ref = c << gf.get_component(squid_spec)
    squid_ref.dcenter = c.dcenter

    pad_width, pad_height = coupling_pad_size

    # Create left coupling pad
    left_pad = gf.components.rectangle(
        size=coupling_pad_size,
        layer=layer_metal,
    )
    left_pad_ref = c << left_pad
    left_pad_ref.move((
        squid_ref.xmin - coupling_gap - pad_width,
        -pad_height / 2
    ))

    # Create right coupling pad
    right_pad = gf.components.rectangle(
        size=coupling_pad_size,
        layer=layer_metal,
    )
    right_pad_ref = c << right_pad
    right_pad_ref.move((
        squid_ref.xmax + coupling_gap,
        -pad_height / 2
    ))

    # Create flux bias line (vertical)
    flux_line = gf.components.rectangle(
        size=(flux_line_width, flux_line_length),
        layer=layer_metal,
    )
    flux_line_ref = c << flux_line
    flux_line_ref.move((
        -flux_line_width / 2,
        squid_ref.ymax + coupling_gap
    ))

    # Add ports for coupling
    c.add_port(
        name="left",
        center=(left_pad_ref.xmin, 0),
        width=pad_height,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="right",
        center=(right_pad_ref.xmax, 0),
        width=pad_height,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add port for flux bias
    c.add_port(
        name="flux_bias",
        center=(0, flux_line_ref.ymax),
        width=flux_line_width,
        orientation=90,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["coupler_type"] = "flux_tunable"
    c.info["coupling_pad_size"] = coupling_pad_size
    c.info["coupling_gap"] = coupling_gap
    c.info["flux_line_width"] = flux_line_width

    return c


if __name__ == "__main__":
    show_components(josephson_junction, squid_junction, tunable_coupler_flux)
