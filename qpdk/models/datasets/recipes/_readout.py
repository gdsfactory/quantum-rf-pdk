"""Readout coupons with open feedline cuts and equipotential conductors.

These electrostatic geometries do not represent a shorted resonator mode.
Adding their full admittance to a distributed model would count the route
capacitance twice. The flip-chip variant leaves its top electrode unbonded.
"""

from functools import partial

import gdsfactory as gf
from _experiment import Geometry, Settings, stack_geometry

from qpdk.cells import (
    flipmon_with_bbox,
    plate_capacitor_single,
    transmon_with_resonator_and_probeline,
)
from qpdk.tech import LAYER

TERMINALS = ("island_a", "island_b", "readout", "feedline")


def select_metal(metal: gf.kdb.Region, port: gf.Port, dbu: float) -> gf.kdb.Region:
    """Reject shorts and missing electrode mappings before spending solver time."""
    x, y = (round(value / dbu) for value in port.center)
    selected = metal.interacting(gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2)))
    if selected.count() != 1:
        raise ValueError(f"Port {port.name} does not select exactly one electrode")
    return selected


def geometry(
    point: dict[str, float], settings: Settings, processes: int, *, flip: bool = False
) -> Geometry:
    """Keep full assembly metal and etch, then select connected electrode nets."""
    if min(point.values()) <= 0:
        raise ValueError("All coupling dimensions must be positive")
    if flip:
        device = transmon_with_resonator_and_probeline(
            qubit=partial(flipmon_with_bbox, junction_position_rotation=180),
            qubit_rotation=-90,
            coupler_port="outer_ring_outside",
            coupler=partial(
                plate_capacitor_single, width=10, length=point["pad_length"]
            ),
            coupler_offset=(-point["pad_offset"], 0),
            probeline_coupling_gap=point["feedline_gap"],
            probeline_coupling_length=point["coupling_length"],
        )
    else:
        device = transmon_with_resonator_and_probeline(
            coupler=partial(
                plate_capacitor_single, width=20, length=point["pad_length"]
            ),
            coupler_offset=(-point["pad_offset"], 0),
            probeline_coupling_gap=point["feedline_gap"],
            probeline_coupling_length=point["coupling_length"],
        )
    component = gf.Component()
    dbu = component.kcl.dbu
    bounds = device.dbbox()
    box = gf.kdb.Box(
        round((bounds.left - 100) / dbu),
        round((bounds.bottom - 100) / dbu),
        round((bounds.right + 100) / dbu),
        round((bounds.top + 100) / dbu),
    )
    footprint = gf.kdb.Region(box)
    etch = device.get_region(LAYER.M1_ETCH)
    for name in ("coupling_o1", "coupling_o2"):
        port = device.ports[name]
        end = gf.Component()
        ref = end.add_ref(
            gf.components.rectangle(
                size=(8, port.width + 16), centered=True, layer=LAYER.M1_ETCH
            )
        )
        ref.drotate(port.orientation)
        ref.dmove(port.center)
        etch |= end.get_region(LAYER.M1_ETCH)
    metal = ((footprint - etch) | device.get_region(LAYER.M1_DRAW)).merged()
    qubit = next(
        ref for ref in device.insts if "junction" in [port.name for port in ref.ports]
    )
    if flip:
        inner = qubit.ports["inner_ring_near_junction"]
        outer = qubit.ports["outer_ring_outside"]
        ports = (inner, outer, device.ports["o1"], device.ports["coupling_o1"])
    else:
        ports = (
            qubit.ports["left_pad"],
            qubit.ports["right_pad"],
            device.ports["o1"],
            device.ports["coupling_o1"],
        )
    electrodes = {
        name: select_metal(metal, port, dbu)
        for name, port in zip(TERMINALS, ports, strict=True)
    }
    ground = metal.interacting(
        gf.kdb.Region(gf.kdb.Box(box.left, box.bottom, box.left + 2, box.bottom + 2))
    )
    names = list(electrodes)
    for i, name in enumerate(names):
        if not (electrodes[name] & ground).is_empty():
            raise ValueError(
                f"{name} touches chip ground; define an explicit reference-plane cut instead of assigning it a terminal voltage"
            )
        for other in names[:i]:
            if not (electrodes[name] & electrodes[other]).is_empty():
                raise ValueError(f"Electrodes {name} and {other} are shorted")
    selected = gf.kdb.Region()
    for region in electrodes.values():
        selected |= region
    ground = metal - selected
    if flip:
        top_z = 10.2
        electrodes3d = {name: (region, 0.0, 0.2) for name, region in electrodes.items()}
        electrodes3d["bottom_ground"] = (ground, 0.0, 0.2)
        electrodes3d["top_pad"] = (device.get_region(LAYER.M2_DRAW), top_z, top_z + 0.2)
        top_etch = device.get_region(LAYER.M2_ETCH)
        electrodes3d["top_ground"] = (
            footprint - top_etch - device.get_region(LAYER.M2_DRAW),
            top_z,
            top_z + 0.2,
        )
        return stack_geometry(
            electrodes3d,
            terminals=(*TERMINALS, "top_pad"),
            settings=settings,
            processes=processes,
            dielectrics=[
                {
                    "name": "bottom_chip",
                    "zmin": -300.0,
                    "zmax": 0.0,
                    "material": "silicon",
                },
                {"name": "gap", "zmin": 0.0, "zmax": top_z + 0.2, "material": "air"},
                {
                    "name": "top_chip",
                    "zmin": top_z + 0.2,
                    "zmax": top_z + 300.2,
                    "material": "silicon",
                },
            ],
        )
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for name, port in zip(TERMINALS, ports, strict=True):
        component.add_port(name, port=port, port_type="electrical")
    pad = settings.domain_pad
    return Geometry(
        component=component,
        bounds=(
            box.left * dbu - pad,
            box.bottom * dbu - pad,
            box.right * dbu + pad,
            box.top * dbu + pad,
        ),
        terminal_ports={name: name for name in TERMINALS},
    )
