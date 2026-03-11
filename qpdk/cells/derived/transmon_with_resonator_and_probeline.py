"""Transmons with resonators coupled."""

from __future__ import annotations

import uuid
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from klayout.db import DCplxTrans

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.resonator import (
    resonator_quarter_wave_bend_start,
)
from qpdk.cells.waveguides import coupler_straight
from qpdk.helper import show_components
from qpdk.tech import route_bundle_cpw


def _transmon_with_resonator_base(
    qubit: ComponentSpec = "double_pad_transmon_with_bbox",
    resonator: ComponentSpec = partial(
        resonator_quarter_wave_bend_start, length=4000, meanders=6
    ),
    resonator_meander_start: tuple[float, float] = (-700, -1300),
    resonator_length: float = 5000.0,
    resonator_meanders: int = 5,
    resonator_bend_spec: ComponentSpec = "bend_circular",
    resonator_cross_section: CrossSectionSpec = "cpw",
    resonator_open_start: bool = False,
    resonator_open_end: bool = True,
    coupler: ComponentSpec = partial(plate_capacitor_single, width=20, length=394),
    qubit_rotation: float = 90,
    coupler_port: str = "left_pad",
    coupler_offset: tuple[float, float] = (-45, 0),
    with_probeline: bool = True,
    probeline_coupler: ComponentSpec = coupler_straight,
    probeline_coupling_gap: float = 16.0,
    probeline_coupling_length: float | None = None,
) -> Component:
    """Base function for creating a transmon coupled to a resonator, optionally with a probeline.

    Args:
        qubit: Qubit component.
        resonator: Resonator component.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        resonator_length: Length of the resonator in µm.
        resonator_meanders: Number of meander sections for the resonator.
        resonator_bend_spec: Specification for the bend component used in meanders.
        resonator_cross_section: Cross-section specification for the resonator.
        resonator_open_start: If True, adds an etch section at the start of the resonator.
        resonator_open_end: If True, adds an etch section at the end of the resonator.
        coupler: Coupler spec.
        qubit_rotation: Rotation angle for the qubit in degrees.
        coupler_port: Name of the qubit port to position the coupler relative to.
        coupler_offset: (x, y) offset for the coupler position.
        with_probeline: Whether to include a probeline coupler.
        probeline_coupler: Component spec for the probeline coupling section.
        probeline_coupling_gap: Gap between the resonator and probeline
            waveguides in the coupling section in µm.
        probeline_coupling_length: Length of the coupling section in µm.
            If None, it will be calculated based on the distance from the
            resonator meander start to the coupler position.
    """
    c = Component()

    qubit_ref = c << gf.get_component(qubit)
    qubit_ref.rotate(qubit_rotation)
    coupler_ref = c << gf.get_component(coupler)

    # Position coupler close to qubit
    coupler_ref.transform(
        qubit_ref.ports[coupler_port].dcplx_trans
        * DCplxTrans.R180
        * DCplxTrans(*coupler_offset)
    )

    coupling_o1_port = None
    coupling_o2_port = None

    if with_probeline:
        if probeline_coupling_length is None:
            xs = gf.get_cross_section(resonator_cross_section)
            radius = getattr(xs, "radius", 100.0)
            probeline_coupling_length = (
                abs(resonator_meander_start[0] - coupler_ref.ports["o1"].x) - radius
            )

            if probeline_coupling_length <= 0:
                raise ValueError(
                    f"Auto-computed probeline_coupling_length={probeline_coupling_length:.1f} µm is non-positive. "
                    "Increase the distance between resonator_meander_start and the coupler, or pass an explicit probeline_coupling_length."
                )

        # Create probeline coupling.
        cs_ref = c << gf.get_component(
            probeline_coupler,
            gap=probeline_coupling_gap,
            length=probeline_coupling_length,
            cross_section=resonator_cross_section,
        )

        # Position the coupler_straight so that o2 (bottom left) is at
        # the resonator_meander_start position.
        cs_ref.move(
            (
                resonator_meander_start[0] - cs_ref.ports["o2"].x,
                resonator_meander_start[1] - cs_ref.ports["o2"].y,
            )
        )
        route_start_port = cs_ref.ports["o3"]
        resonator_connect_port = cs_ref.ports["o2"]
        coupling_o1_port = cs_ref.ports["o1"]
        coupling_o2_port = cs_ref.ports["o4"]
        added_length = probeline_coupling_length
    else:
        # We need a port to start the route from, so we create a dummy port
        dummy = gf.Component(name=f"dummy_route_start_{uuid.uuid4().hex[:8]}")
        dummy.add_port(
            name="o1",
            center=(0, 0),
            width=10,
            orientation=180,
            cross_section=resonator_cross_section,
        )
        dummy_port = dummy.ports["o1"].copy()
        dummy_port.center = resonator_meander_start
        dummy_port.orientation = 0
        route_start_port = dummy_port
        resonator_connect_port = None
        added_length = 0.0

    # Route from meander start to the plate capacitor
    routes = route_bundle_cpw(
        component=c,
        ports1=[route_start_port],
        ports2=[coupler_ref.ports["o1"]],
        steps=[{"x": coupler_ref.ports["o1"].x}],
        auto_taper=False,
    )
    route = routes[0]

    # Create resonator, accounting for both the route and coupling lengths
    resonator_ref = c << gf.get_component(
        resonator,
        length=resonator_length - route.length * c.kcl.dbu - added_length,
        meanders=resonator_meanders,
        bend_spec=resonator_bend_spec,
        cross_section=resonator_cross_section,
        open_start=resonator_open_start,
        open_end=resonator_open_end,
    )

    if with_probeline:
        # Connect the resonator's shorted end to the coupler_straight's top-left port.
        resonator_ref.connect("o1", resonator_connect_port)
    else:
        # Connect to the start of the route
        resonator_ref.connect("o1", route.instances[0].ports["o1"])

    c.info["qubit_type"] = qubit_ref.cell.info.get("qubit_type")
    c.info["resonator_type"] = resonator_ref.cell.info.get("resonator_type")
    c.info["coupler_type"] = coupler_ref.cell.info.get("coupler_type")
    c.info["length"] = (
        resonator_ref.cell.info.get("length") + route.length * c.kcl.dbu + added_length
    )

    c.add_ports(qubit_ref.ports.filter(regex=r"junction"))
    if with_probeline and coupling_o1_port:
        c.add_port("coupling_o1", port=coupling_o1_port)
        c.add_port("coupling_o2", port=coupling_o2_port)
    c.add_port(
        port=resonator_ref.ports["o1"],
        port_type="placement",
    )
    return c


@gf.cell
def transmon_with_resonator_and_probeline(
    qubit: ComponentSpec = "double_pad_transmon_with_bbox",
    resonator: ComponentSpec = partial(
        resonator_quarter_wave_bend_start, length=4000, meanders=6
    ),
    resonator_meander_start: tuple[float, float] = (-900, -1200),
    resonator_length: float = 5000.0,
    resonator_meanders: int = 5,
    resonator_bend_spec: ComponentSpec = "bend_circular",
    resonator_cross_section: CrossSectionSpec = "cpw",
    resonator_open_start: bool = False,
    resonator_open_end: bool = True,
    coupler: ComponentSpec = partial(plate_capacitor_single, width=20, length=394),
    qubit_rotation: float = 90,
    coupler_port: str = "left_pad",
    coupler_offset: tuple[float, float] = (-45, 0),
    probeline_coupler: ComponentSpec = coupler_straight,
    probeline_coupling_gap: float = 16.0,
    probeline_coupling_length: float | None = None,
) -> Component:
    """Returns a transmon qubit coupled to a quarter wave resonator and a probeline.

    Uses a :func:`~qpdk.cells.waveguides.coupler_straight` to couple the
    resonator to a probeline. The coupling section is inserted at the start
    of the resonator meander, with one waveguide carrying the resonator signal
    and the other providing ports for probeline routing.

    Args:
        qubit: Qubit component.
        resonator: Resonator component.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        resonator_length: Length of the resonator in µm.
        resonator_meanders: Number of meander sections for the resonator.
        resonator_bend_spec: Specification for the bend component used in meanders.
        resonator_cross_section: Cross-section specification for the resonator.
        resonator_open_start: If True, adds an etch section at the start of the resonator.
        resonator_open_end: If True, adds an etch section at the end of the resonator.
        coupler: Coupler spec.
        qubit_rotation: Rotation angle for the qubit in degrees.
        coupler_port: Name of the qubit port to position the coupler relative to.
        coupler_offset: (x, y) offset for the coupler position.
        probeline_coupler: Component spec for the probeline coupling section.
        probeline_coupling_gap: Gap between the resonator and probeline
            waveguides in the coupling section in µm.
        probeline_coupling_length: Length of the coupling section in µm.
    """
    return _transmon_with_resonator_base(
        qubit=qubit,
        resonator=resonator,
        resonator_meander_start=resonator_meander_start,
        resonator_length=resonator_length,
        resonator_meanders=resonator_meanders,
        resonator_bend_spec=resonator_bend_spec,
        resonator_cross_section=resonator_cross_section,
        resonator_open_start=resonator_open_start,
        resonator_open_end=resonator_open_end,
        coupler=coupler,
        qubit_rotation=qubit_rotation,
        coupler_port=coupler_port,
        coupler_offset=coupler_offset,
        with_probeline=True,
        probeline_coupler=probeline_coupler,
        probeline_coupling_gap=probeline_coupling_gap,
        probeline_coupling_length=probeline_coupling_length,
    )


@gf.cell
def transmon_with_resonator(
    qubit: ComponentSpec = "double_pad_transmon_with_bbox",
    resonator: ComponentSpec = partial(
        resonator_quarter_wave_bend_start, length=4000, meanders=6
    ),
    resonator_meander_start: tuple[float, float] = (-900, -1200),
    resonator_length: float = 5000.0,
    resonator_meanders: int = 5,
    resonator_bend_spec: ComponentSpec = "bend_circular",
    resonator_cross_section: CrossSectionSpec = "cpw",
    resonator_open_start: bool = False,
    resonator_open_end: bool = True,
    coupler: ComponentSpec = partial(plate_capacitor_single, width=20, length=394),
    qubit_rotation: float = 90,
    coupler_port: str = "left_pad",
    coupler_offset: tuple[float, float] = (-45, 0),
) -> Component:
    """Returns a transmon qubit coupled to a quarter wave resonator.

    Args:
        qubit: Qubit component.
        resonator: Resonator component.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        resonator_length: Length of the resonator in µm.
        resonator_meanders: Number of meander sections for the resonator.
        resonator_bend_spec: Specification for the bend component used in meanders.
        resonator_cross_section: Cross-section specification for the resonator.
        resonator_open_start: If True, adds an etch section at the start of the resonator.
        resonator_open_end: If True, adds an etch section at the end of the resonator.
        coupler: Coupler spec.
        qubit_rotation: Rotation angle for the qubit in degrees.
        coupler_port: Name of the qubit port to position the coupler relative to.
        coupler_offset: (x, y) offset for the coupler position.
    """
    return _transmon_with_resonator_base(
        qubit=qubit,
        resonator=resonator,
        resonator_meander_start=resonator_meander_start,
        resonator_length=resonator_length,
        resonator_meanders=resonator_meanders,
        resonator_bend_spec=resonator_bend_spec,
        resonator_cross_section=resonator_cross_section,
        resonator_open_start=resonator_open_start,
        resonator_open_end=resonator_open_end,
        coupler=coupler,
        qubit_rotation=qubit_rotation,
        coupler_port=coupler_port,
        coupler_offset=coupler_offset,
        with_probeline=False,
    )


# Create specific functions as partials of the general function
double_pad_transmon_with_resonator_and_probeline = partial(
    transmon_with_resonator_and_probeline,
    qubit="double_pad_transmon_with_bbox",
    coupler=partial(plate_capacitor_single, width=20, length=394),
    qubit_rotation=90,
    coupler_port="left_pad",
    coupler_offset=(-45, 0),
)

flipmon_with_resonator_and_probeline = partial(
    transmon_with_resonator_and_probeline,
    qubit="flipmon_with_bbox",
    coupler=partial(plate_capacitor_single, width=10, length=58),
    qubit_rotation=-90,
    coupler_port="outer_ring_outside",
    coupler_offset=(-10, 0),
)

double_pad_transmon_with_resonator = partial(
    transmon_with_resonator,
    qubit="double_pad_transmon_with_bbox",
    coupler=partial(plate_capacitor_single, width=20, length=394),
    qubit_rotation=90,
    coupler_port="left_pad",
    coupler_offset=(-45, 0),
)

flipmon_with_resonator = partial(
    transmon_with_resonator,
    qubit="flipmon_with_bbox",
    coupler=partial(plate_capacitor_single, width=10, length=58),
    qubit_rotation=-90,
    coupler_port="outer_ring_outside",
    coupler_offset=(-10, 0),
)


if __name__ == "__main__":
    show_components(
        transmon_with_resonator_and_probeline,
        double_pad_transmon_with_resonator_and_probeline,
        flipmon_with_resonator_and_probeline,
        transmon_with_resonator,
        double_pad_transmon_with_resonator,
        flipmon_with_resonator,
    )
