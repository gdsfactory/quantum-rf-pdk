r"""Unimon qubit components.

The unimon qubit consists of a Josephson junction (or SQUID) embedded within
a coplanar waveguide resonator, with two grounded :math:`\lambda/4` arms
extending from the junction. The large geometric inductance of the resonator
arms, combined with the Josephson nonlinearity, creates a qubit with large
anharmonicity and insensitivity to charge and flux noise.

References:
    - :cite:`hyyppaUnimonQubit2022`
    - :cite:`tuohinoMultimodePhysicsUnimon2024`
    - :cite:`dudaParameterOptimizationUnimon2025`
"""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from kfactory import kdb

from qpdk.cells.capacitor import half_circle_coupler
from qpdk.cells.junction import josephson_junction, squid_junction
from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, straight
from qpdk.helper import show_components
from qpdk.tech import LAYER, get_etch_section


@gf.cell
def unimon_arm(
    arm_length: float = 3000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_gap: float = 6.0,
) -> Component:
    """Creates a quarter-wave resonator arm for the unimon qubit."""
    c = Component()
    cross_section_obj = gf.get_cross_section(cross_section)
    bend = gf.get_component(
        bend_spec, cross_section=cross_section_obj, angle=180, angular_step=4
    )
    bend_length = bend.info["length"]
    # With end_with_bend=True, there are arm_meanders straight sections.
    # We add an extra half straight, so total straights = arm_meanders + 0.5
    length_per_one_straight = (arm_length - arm_meanders * bend_length) / (
        arm_meanders + 0.5
    )
    base_resonator_length = (
        arm_meanders * bend_length + arm_meanders * length_per_one_straight
    )

    # Create the base quarter-wave resonator arm ending with a bend
    arm_base = resonator(
        length=base_resonator_length,
        meanders=arm_meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        open_start=False,  # shorted at far end (port o1)
        open_end=False,  # open end connects to junction
        end_with_bend=True,
    )

    # Short straight CPW section to be attached after the bend.
    # Its length matches half of the resonator straight lengths minus half the gap.
    half_straight_length = (length_per_one_straight / 2) - (junction_gap / 2)
    half_straight = straight(
        length=half_straight_length,
        cross_section=cross_section,
    )

    arm_base_ref = c.add_ref(arm_base)
    half_straight_ref = c.add_ref(half_straight)
    half_straight_ref.connect(
        "o1",
        arm_base_ref.ports["o2"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )

    c.add_port("o1", port=arm_base_ref.ports["o1"])
    c.add_port("o2", port=half_straight_ref.ports["o2"])

    cross_section_etch_section = get_etch_section(cross_section_obj)
    # Add a port for readout coupling at the center of the last bend
    # We find the last bend instance in the resonator
    bend_instances = [inst for inst in arm_base.insts if "bend" in inst.cell.name]
    if bend_instances:
        last_bend = bend_instances[-1]
        radius = last_bend.cell.info["radius"]
        # Locate the center of curvature of the last meander bend.
        # The bbox extends radius + width/2 + etch_width beyond the
        # center in every direction that the arc reaches.
        bbox = last_bend.dbbox()
        half_cpw = cross_section_obj.width / 2 + cross_section_etch_section.width
        if abs(bbox.left) > abs(bbox.right):
            center_x = bbox.left + radius + half_cpw
            orientation = 180  # Pointing LEFT
        else:
            center_x = bbox.right - radius - half_cpw
            orientation = 0  # Pointing RIGHT

        c.add_port(
            name="readout",
            center=(
                center_x,
                bbox.top - radius - half_cpw,
            ),
            width=cross_section_obj.width,
            orientation=orientation,
            layer=LAYER.M1_DRAW,
            port_type="placement",
        )

    return c


@gf.cell(check_instances=False)
def unimon(
    arm_length: float = 3000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = partial(
        squid_junction,
        junction_spec=partial(
            josephson_junction,
            junction_overlap_displacement=1.8,
            wide_straight_length=4.0,
            narrow_straight_length=0.5,
            taper_length=4,
        ),
    ),
    junction_gap: float = 10.0,
    junction_etch_width: float = 22.0,
) -> Component:
    r"""Creates a unimon qubit from two grounded :math:`\lambda/4` CPW resonator arms connected by a SQUID junction.

    The unimon is a superconducting qubit consisting of a single Josephson
    junction (or SQUID for flux tunability) embedded in the center of
    a two grounded :math:`\lambda/4` CPW resonators, providing
    a large geometric inductance that, together with the Josephson
    nonlinearity, yields high anharmonicity and resilience to charge noise.

    .. svgbob::

        o1 (shorted)
           |
           | <-- :math:`\lambda/4` resonator arm (meandered)
           |
        junction
           |
           | <-- :math:`\lambda/4` resonator arm (meandered)
           |
        o2 (shorted)

    See :cite:`hyyppaUnimonQubit2022,tuohinoMultimodePhysicsUnimon2024` for details.

    Args:
        arm_length: Length of each :math:`\lambda/4` resonator arm in µm.
        arm_meanders: Number of meander sections in each arm.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator arms.
        junction_spec: Component specification for the junction (SQUID) component.
        junction_gap: Length of the etched gap on which the junction sits in µm.
        junction_etch_width: Width of the etched region where the junction sits in µm.

    Returns:
        Component: A gdsfactory component with the unimon qubit geometry.
    """
    c = Component()

    arm = unimon_arm(
        arm_length=arm_length,
        arm_meanders=arm_meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        junction_gap=junction_gap,
    )

    cross_section_obj = gf.get_cross_section(cross_section)
    cross_section_etch_section = get_etch_section(cross_section_obj)

    # Place the SQUID junction at the center
    junction_comp = gf.get_component(junction_spec)
    junction_ref = c.add_ref(junction_comp)
    junction_ref.dcplx_trans *= kdb.DCplxTrans(1, -45, False, 0, 0)
    junction_ref.dcenter = (0, 0)

    gap_comp = gf.c.rectangle(
        size=(junction_gap, junction_etch_width),
        layer=cross_section_etch_section.layer,
        centered=True,
        port_type="optical",
        port_orientations=(0, 180),
    )

    gap_ref = c.add_ref(gap_comp)
    gap_ref.dcenter = (0, 0)
    gap_ref.rotate(90)

    arm_top_ref = c.add_ref(arm)
    arm_top_ref.connect(
        "o2",
        gap_ref.ports["o2"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )
    arm_bottom_ref = c.add_ref(arm)
    arm_bottom_ref.rotate(180)
    arm_bottom_ref.connect(
        "o2",
        gap_ref.ports["o1"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )

    c.add_port("o1", port=arm_top_ref.ports["o1"], port_type="placement")
    c.add_port("o2", port=arm_bottom_ref.ports["o1"], port_type="placement")

    # Promote readout ports for coupling
    c.add_port("readout_top", port=arm_top_ref.ports["readout"])
    c.add_port("readout_bottom", port=arm_bottom_ref.ports["readout"])

    # Add placement port for the junction center
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
        port_type="placement",
    )

    # Add metadata
    c.info["qubit_type"] = "unimon"
    c.info["arm_length"] = arm_length
    c.info["total_resonator_length"] = 2 * arm_length
    c.info["meander_radius"] = arm.info.get("radius", 100.0)

    # Rotate whole component to be vertical
    c.rotate(-90)

    return c


@gf.cell
def unimon_coupled(
    arm_length: float = 3000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = partial(
        squid_junction,
        junction_spec=partial(
            josephson_junction,
            junction_overlap_displacement=1.8,
            wide_straight_length=4.0,
            narrow_straight_length=0.5,
            taper_length=4,
        ),
    ),
    junction_gap: float = 6.0,
    junction_etch_width: float = 22.0,
    coupling_gap: float = 30.0,
    coupling_angle: float = 180.0,
    coupling_extension_length: float = 50.0,
    cross_section_non_resonator: CrossSectionSpec = "cpw",
) -> Component:
    r"""Creates a unimon qubit with a half-circle coupling waveguide for readout.

    This component combines a :func:`unimon` qubit with a half-circle coupler
    placed at a specified gap for proximity coupling to a readout resonator.

    Args:
        arm_length: Length of each :math:`\lambda/4` resonator arm in µm.
        arm_meanders: Number of meander sections in each arm.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator arms.
        junction_spec: Component specification for the junction (SQUID) component.
        junction_gap: Length of the etched gap on which the junction sits in µm.
        junction_etch_width: Width of the etched region where the junction sits in µm.
        coupling_gap: Edge-to-edge gap between M1_DRAW centre conductors of the
            unimon resonator and the coupling waveguide in µm.  The coupling
            radius is automatically computed as
            ``meander_radius + coupling_gap + width_resonator/2 + width_coupler/2``
            to ensure a uniform gap across the bend.
        coupling_angle: Angle of the circular arc in degrees.
        coupling_extension_length: Length of the straight sections extending from the
            ends of the half-circle in μm.
        cross_section_non_resonator: Cross-section for the coupling waveguide.

    Returns:
        Component: A gdsfactory component with the unimon and half-circle coupler.
    """
    c = Component()

    unimon_ref = c.add_ref(
        unimon(
            arm_length=arm_length,
            arm_meanders=arm_meanders,
            bend_spec=bend_spec,
            cross_section=cross_section,
            junction_spec=junction_spec,
            junction_gap=junction_gap,
            junction_etch_width=junction_etch_width,
        )
    )

    # Compute coupling_radius so that the edge-to-edge gap between the
    # M1_DRAW centre conductors equals coupling_gap for concentric arcs.
    meander_radius = unimon_ref.cell.info["meander_radius"]
    xs_resonator = gf.get_cross_section(cross_section)
    xs_coupler = gf.get_cross_section(cross_section_non_resonator)
    coupling_radius = (
        meander_radius + coupling_gap + xs_resonator.width / 2 + xs_coupler.width / 2
    )

    coupler = c.add_ref(
        half_circle_coupler(
            radius=coupling_radius,
            angle=coupling_angle,
            extension_length=coupling_extension_length,
            cross_section=cross_section_non_resonator,
        )
    )

    # Align coupler anchor (at arc center) with the unimon readout port
    # (at the meander bend center) for concentric alignment and uniform gap
    coupler.connect(
        "anchor",
        unimon_ref.ports["readout_top"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )

    for port in unimon_ref.ports:
        if port.name == "readout_top":
            continue  # connected to coupler anchor internally
        c.add_port(f"unimon_{port.name}", port=port)

    for port in coupler.ports:
        if port.name == "anchor":
            continue  # anchor was used for alignment, not an external port
        c.add_port(f"coupling_{port.name}", port=port)

    c.info += unimon_ref.cell.info
    c.info["coupling_gap"] = coupling_gap
    c.info["coupling_radius"] = coupling_radius

    return c


if __name__ == "__main__":
    show_components(
        unimon,
        partial(unimon, arm_length=2000, arm_meanders=4),
        unimon_coupled,
    )
