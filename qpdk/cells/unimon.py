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

from qpdk.cells.junction import squid_junction
from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, straight
from qpdk.helper import show_components
from qpdk.tech import LAYER


@gf.cell
def unimon_arm(
    arm_length: float = 4000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    extra_straight_length: float = 0.0,
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

    # Short straight CPW section to be attached after the bend
    half_straight = straight(
        length=length_per_one_straight / 2 + extra_straight_length,
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
    return c


@gf.cell
def unimon(
    arm_length: float = 4000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = squid_junction,
    junction_coupler_length: float = 50.0,
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
           |  <-- :math:`\lambda/4` resonator arm (meandered)
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
        junction_coupler_length: Length of the straight CPW section connecting
            each resonator arm to the junction in µm.

    Returns:
        Component: A gdsfactory component with the unimon qubit geometry.
    """
    c = Component()

    arm = unimon_arm(
        arm_length=arm_length,
        arm_meanders=arm_meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        extra_straight_length=junction_coupler_length,
    )

    # Place the SQUID junction at the center
    junction_comp = gf.get_component(junction_spec)
    junction_ref = c.add_ref(junction_comp)
    junction_ref.dcenter = (0, 0)

    cross_section_obj = gf.get_cross_section(cross_section)
    cross_section_etch_section = next(
        s for s in cross_section_obj.sections if s.name and "etch_offset" in s.name
    )

    # Gap width is the width of the etch section line itself
    gap_width = cross_section_etch_section.width
    # Etch rectangle width is 2*gap_width + center conductor width
    etch_rect_width = 2 * gap_width + cross_section_obj.width

    gap_comp = gf.c.rectangle(
        size=(gap_width, etch_rect_width),
        layer=cross_section_etch_section.layer,
        centered=True,
        port_type="optical",
        port_orientations=(0, 180),
    )

    gap_ref = c.add_ref(gap_comp)
    gap_ref.dcenter = (0, 0)
    gap_ref.rotate(90)

    # Place resonator arms
    arm_top_ref = c.add_ref(arm)
    arm_top_ref.connect(
        "o2",
        gap_ref.ports["o2"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )

    arm_bottom_ref = c.add_ref(arm)
    arm_bottom_ref.connect(
        "o2",
        gap_ref.ports["o1"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )
    arm_bottom_ref.mirror_y()

    # Add ports at the shorted ends of the arms (for external coupling)
    c.add_port("o1", port=arm_top_ref.ports["o1"])
    c.add_port("o2", port=arm_bottom_ref.ports["o1"])

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
    c.info["total_resonator_length"] = 2 * arm_length + 2 * junction_coupler_length

    # Rotate to be vertical
    c.rotate(-90)

    return c


@gf.cell
def unimon_coupled(
    arm_length: float = 4000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = squid_junction,
    junction_coupler_length: float = 50.0,
    coupling_gap: float = 20.0,
    coupling_straight_length: float = 200.0,
    cross_section_non_resonator: CrossSectionSpec = "cpw",
) -> Component:
    r"""Creates a unimon qubit with a coupling waveguide for readout.

    This component combines a :func:`unimon` qubit with a parallel coupling
    waveguide placed at a specified gap for proximity coupling to a readout
    resonator or probeline.

    See :cite:`hyyppaUnimonQubit2022` for details.

    Args:
        arm_length: Length of each :math:`\lambda/4` resonator arm in µm.
        arm_meanders: Number of meander sections in each arm.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator arms.
        junction_spec: Component specification for the junction (SQUID) component.
        junction_coupler_length: Length of the straight CPW section connecting
            each resonator arm to the junction in µm.
        coupling_gap: Gap between the unimon and coupling waveguide in µm.
        coupling_straight_length: Length of the coupling waveguide section in µm.
        cross_section_non_resonator: Cross-section for the coupling waveguide.

    Returns:
        Component: A gdsfactory component with the unimon and coupling waveguide.
    """
    c = Component()

    unimon_ref = c.add_ref(
        unimon(
            arm_length=arm_length,
            arm_meanders=arm_meanders,
            bend_spec=bend_spec,
            cross_section=cross_section,
            junction_spec=junction_spec,
            junction_coupler_length=junction_coupler_length,
        )
    )

    cross_section_obj = gf.get_cross_section(cross_section_non_resonator)
    coupling_wg = straight(
        length=coupling_straight_length,
        cross_section=cross_section_obj,
    )
    coupling_ref = c.add_ref(coupling_wg)

    # Position coupling waveguide parallel to one arm with specified gap
    coupling_ref.movey(unimon_ref.dbbox().top + coupling_gap + cross_section_obj.width)
    coupling_ref.xmin = unimon_ref.ports["o1"].x

    for port in unimon_ref.ports:
        c.add_port(f"unimon_{port.name}", port=port)

    for port in coupling_ref.ports:
        c.add_port(f"coupling_{port.name}", port=port)

    c.info += unimon_ref.cell.info
    c.info["coupling_gap"] = coupling_gap
    c.info["coupling_length"] = coupling_straight_length

    return c


if __name__ == "__main__":
    show_components(
        unimon,
        partial(unimon, arm_length=2000, arm_meanders=4),
        unimon_coupled,
    )
