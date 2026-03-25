"""Unimon qubit components.

The unimon qubit consists of a Josephson junction (or SQUID) embedded within
a coplanar waveguide resonator, with two grounded :math:`\\lambda/4` arms
extending from the junction. The large geometric inductance of the resonator
arms, combined with the Josephson nonlinearity, creates a qubit with large
anharmonicity and insensitivity to charge and flux noise.

References:
    - :cite:`hyyppaUnimonQubit2022`
    - :cite:`tuokkolaMultimodePhysicsUnimon2024`
    - :cite:`sajadiParameterOptimizationUnimon2025`
"""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.junction import squid_junction
from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, straight
from qpdk.helper import show_components
from qpdk.tech import LAYER


@gf.cell
def unimon(
    arm_length: float = 4000.0,
    arm_meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = squid_junction,
    junction_coupler_length: float = 50.0,
) -> Component:
    r"""Creates a unimon qubit from two grounded :math:`\lambda/4` CPW resonator arms
    connected by a SQUID junction.

    The unimon is a superconducting qubit consisting of a single Josephson
    junction (or SQUID for flux tunability) embedded in the center of
    a :math:`\lambda/2`-like coplanar waveguide resonator.  Each arm of the
    resonator is effectively a grounded :math:`\lambda/4` section, providing
    a large geometric inductance that, together with the Josephson
    nonlinearity, yields high anharmonicity and resilience to charge noise.

    .. svgbob::

                   ┌── arm_left (meander) ─── o1 (shorted)
        junction ──┤
                   └── arm_right (meander) ── o2 (shorted)

    See :cite:`hyyppaUnimonQubit2022,tuokkolaMultimodePhysicsUnimon2024` for details.

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
    cross_section_obj = gf.get_cross_section(cross_section)

    # Create two quarter-wave resonator arms (shorted at far end, open at SQUID end)
    arm = resonator(
        length=arm_length,
        meanders=arm_meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        open_start=False,  # shorted at far end (port o1)
        open_end=False,  # open end connects to junction
    )

    # Short straight CPW sections connecting the arms to the junction
    coupler = straight(
        length=junction_coupler_length,
        cross_section=cross_section,
    )

    # Place the SQUID junction at the center
    junction_comp = gf.get_component(junction_spec)
    junction_ref = c.add_ref(junction_comp)
    junction_ref.dcenter = (0, 0)

    # Determine vertical offset for placing the two arms
    # Use the junction's bounding box to know where to connect
    junc_bbox = junction_ref.dbbox()
    junction_top_y = junc_bbox.top
    junction_bottom_y = junc_bbox.bottom

    # Place coupler straights connecting to junction
    coupler_top_ref = c.add_ref(coupler)
    coupler_top_ref.drotation = 90
    coupler_top_ref.move(
        np.array((0, junction_top_y))
        - np.array(coupler_top_ref.ports["o1"].center)
    )

    coupler_bottom_ref = c.add_ref(coupler)
    coupler_bottom_ref.drotation = -90
    coupler_bottom_ref.move(
        np.array((0, junction_bottom_y))
        - np.array(coupler_bottom_ref.ports["o1"].center)
    )

    # Place resonator arms
    arm_top_ref = c.add_ref(arm)
    arm_top_ref.connect("o2", coupler_top_ref.ports["o2"])

    arm_bottom_ref = c.add_ref(arm)
    arm_bottom_ref.mirror_y()
    arm_bottom_ref.connect("o2", coupler_bottom_ref.ports["o2"])

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
    coupling_ref.movey(
        unimon_ref.dbbox().top + coupling_gap + cross_section_obj.width
    )
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
