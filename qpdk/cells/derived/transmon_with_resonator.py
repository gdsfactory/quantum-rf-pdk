"""Transmons with resonators coupled."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec
from klayout.db import DCplxTrans

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.resonator import ResonatorParams, resonator_quarter_wave
from qpdk.helper import show_components
from qpdk.tech import LAYER, route_single_cpw


@gf.cell_with_module_name
def qubit_with_resonator(
    qubit: ComponentSpec = "double_pad_transmon_with_bbox",
    resonator: ComponentSpec = partial(resonator_quarter_wave, length=4000, meanders=6),
    resonator_meander_start: tuple[float, float] = (-700, -1300),
    resonator_length: float = 5000.0,
    resonator_params: ResonatorParams | None = None,
    coupler: ComponentSpec = partial(plate_capacitor_single, thickness=20, fingers=18),
    qubit_rotation: float = 90,
    coupler_port: str = "left_pad",
    coupler_offset: tuple[float, float] = (-45, 0),
) -> Component:
    """Returns a qubit coupled to a quarter wave resonator.

    Args:
        qubit: Qubit component.
        resonator: Resonator component.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        resonator_length: Length of the resonator in µm.
        resonator_params: Parameters for the resonator component if it accepts any.
        coupler: Coupler spec.
        qubit_rotation: Rotation angle for the qubit in degrees.
        coupler_port: Name of the qubit port to position the coupler relative to.
        coupler_offset: (x, y) offset for the coupler position.
    """
    c = Component()
    resonator_params = resonator_params or ResonatorParams(meanders=5)

    qubit_ref = c << gf.get_component(qubit)
    qubit_ref.rotate(qubit_rotation)
    coupler_ref = c << gf.get_component(coupler)

    # Position coupler close to qubit
    coupler_ref.transform(
        qubit_ref.ports[coupler_port].dcplx_trans
        * DCplxTrans.R180
        * DCplxTrans(*coupler_offset)
    )

    # Route to resonator input
    resonator_input_port = gf.Port(
        name="resonator_input",
        center=resonator_meander_start,
        orientation=0,
        layer=LAYER.M1_DRAW,
        width=10.0,
    )
    route = route_single_cpw(
        component=c,
        port1=resonator_input_port,
        port2=coupler_ref.ports["o1"],
        steps=[{"x": coupler_ref.ports["o1"].x}],
        auto_taper=False,
    )
    resonator_ref = c << gf.get_component(
        resonator,
        length=resonator_length - route.length * c.kcl.dbu,
        **resonator_params,
    )
    resonator_ref.rotate(180)
    resonator_ref.transform(resonator_input_port.dcplx_trans)

    c.info["qubit_type"] = qubit_ref.cell.info.get("qubit_type")
    c.info["resonator_type"] = resonator_ref.cell.info.get("resonator_type")
    c.info["coupler_type"] = coupler_ref.cell.info.get("coupler_type")
    c.info["length"] = resonator_ref.cell.info.get("length") + route.length * c.kcl.dbu

    c.add_ports([p for p in qubit_ref.ports if p.name == "junction"])
    c.add_ports([p for p in resonator_ref.ports if p.name == "o1"])

    return c


# Create specific functions as partials of the general function
transmon_with_resonator = partial(
    qubit_with_resonator,
    qubit="double_pad_transmon_with_bbox",
    coupler=partial(plate_capacitor_single, thickness=20, fingers=18),
    qubit_rotation=90,
    coupler_port="left_pad",
    coupler_offset=(-45, 0),
)

flipmon_with_resonator = partial(
    qubit_with_resonator,
    qubit="flipmon_with_bbox",
    coupler=partial(plate_capacitor_single, thickness=10, fingers=5),
    qubit_rotation=-90,
    coupler_port="outer_ring_outside",
    coupler_offset=(-10, 0),
)


#

if __name__ == "__main__":
    show_components(
        transmon_with_resonator,
        flipmon_with_resonator,
    )
