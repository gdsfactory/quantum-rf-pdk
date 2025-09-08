"""Transmons with resonators coupled."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec
from klayout.db import DCplxTrans

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.resonator import resonator_quarter_wave
from qpdk.tech import LAYER, route_single_cpw


@gf.cell_with_module_name
def transmon_with_resonator(
    transmon: ComponentSpec = "double_pad_transmon_with_bbox",
    resonator: ComponentSpec = partial(resonator_quarter_wave, length=4000, meanders=6),
    resonator_meander_start: tuple[float, float] = (-700, -1300),
    resonator_length: float = 5000.0,
    coupler: ComponentSpec = partial(plate_capacitor_single, thickness=20, fingers=18),
) -> Component:
    """Returns a transmon qubit coupled to a quarter wave resonator.

    Args:
        transmon: Transmon component.
        resonator: Resonator component.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        resonator_length: Length of the resonator in µm.
        coupler: Coupler component.
    """
    c = Component()

    transmon_ref = c << gf.get_component(transmon)
    transmon_ref.rotate(90)
    coupler_ref = c << gf.get_component(coupler)

    # Position coupler close to transmon
    coupler_ref.transform(
        transmon_ref.ports["left_pad"].dcplx_trans
        * DCplxTrans.R180
        * DCplxTrans(-45, 0)
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
        resonator, length=resonator_length - route.length * c.kcl.dbu
    )
    resonator_ref.rotate(180)
    resonator_ref.transform(resonator_input_port.dcplx_trans)

    c.info["qubit_type"] = transmon_ref.cell.info.get("qubit_type")
    c.info["resonator_type"] = resonator_ref.cell.info.get("resonator_type")
    c.info["coupler_type"] = coupler_ref.cell.info.get("coupler_type")
    c.info["length"] = resonator_ref.cell.info.get("length") + route.length * c.kcl.dbu

    c.add_ports([p for p in transmon_ref.ports if p.name == "junction"])
    c.add_ports([p for p in resonator_ref.ports if p.name == "o1"])

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    c = transmon_with_resonator()
    c.show()
