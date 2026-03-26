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
from qpdk.tech import LAYER


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

    cross_section_etch_section = next(
        s for s in cross_section_obj.sections if s.name and "etch_offset" in s.name
    )
    # Add a port for readout coupling at the center of the last bend
    # We find the last bend instance in the resonator
    bend_instances = [inst for inst in arm_base.insts if "bend" in inst.cell.name]
    if bend_instances:
        last_bend = bend_instances[-1]
        radius = last_bend.cell.info["radius"]
        # The center of the arc for a 180 deg bend is radius away from the straight path
        # For our resonator meanders, we can use the instance bounding box
        bbox = last_bend.dbbox()
        # If the bend is on the left, arc center is at the left peak
        # If the bend is on the right, arc center is at the right peak
        # We'll use the side that is furthest from the ports (which are near x=0)
        if abs(bbox.left) > abs(bbox.right):
            center_x = bbox.left + radius
            orientation = 180 # Pointing LEFT
        else:
            center_x = bbox.right - radius
            orientation = 0 # Pointing RIGHT
            
        c.add_port(
            name="readout",
            center=(center_x, bbox.top - radius - cross_section_etch_section.width - cross_section_obj.width/2),
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
    cross_section_etch_section = next(
        s for s in cross_section_obj.sections if s.name and "etch_offset" in s.name
    )

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
    c.info["meander_radius"] = arm.info["radius"] if "radius" in arm.info else 100.0

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
    coupling_radius: float | None = None,
    coupling_angle: float = 180.0,
    coupling_extension_length: float = 50.0,
    coupling_extra_straight_length: float = 10.0,
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
        coupling_gap: Gap between the unimon and coupling waveguide in µm.
            Measured from the center of the center conductor of the straight sections
            of the coupler and the unimon resonators.
        coupling_radius: Inner radius of the half-circle coupler in μm.
            If None, defaults to meander_radius + coupling_gap.
        coupling_angle: Angle of the circular arc in degrees.
        coupling_extension_length: Length of the straight sections extending from the
            ends of the half-circle in μm.
        coupling_extra_straight_length: Length of the straight section extending
            from the coupler in μm.
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

    # Match the meander radius and make it bigger by the coupling gap
    meander_radius = unimon_ref.cell.info["meander_radius"]
    if coupling_radius is None:
        coupling_radius = meander_radius + coupling_gap

    coupler = c.add_ref(
        half_circle_coupler(
            radius=coupling_radius,
            angle=coupling_angle,
            extension_length=coupling_extension_length,
            cross_section=cross_section_non_resonator,
            extra_straight_length=coupling_extra_straight_length,
        )
    )

    # Align coupler anchor with the unimon readout port
    # The readout port is at the center of the meander bend
    coupler.connect("anchor", unimon_ref.ports["readout_top"], allow_width_mismatch=True, allow_layer_mismatch=True)
    coupler.dcplx_trans *= kdb.DCplxTrans(1, 0, False, 0,coupling_gap*2)

    for port in unimon_ref.ports:
        c.add_port(f"unimon_{port.name}", port=port)

    for port in coupler.ports:
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
