"""Local qubit and coupling paddle, ending at the paddle's CPW port."""

import gdsfactory as gf
from _experiment import Geometry, Settings
from _readout import select_metal

from qpdk.cells import (
    double_pad_transmon_with_bbox,
    plate_capacitor_single,
    xmon_transmon,
)
from qpdk.tech import LAYER


def geometry(
    point: dict[str, float], settings: Settings, _processes: int, *, xmon: bool
) -> Geometry:
    """Retain the complete local Maxwell network without any resonator route."""
    if min(point.values()) <= 0:
        raise ValueError("All device dimensions and scales must be positive")
    if xmon:
        scale = point["arm_scale"]
        device = xmon_transmon(
            arm_lengths=tuple(scale * value for value in (160.0, 120.0, 160.0, 120.0)),
            gap_width=point["gap_width"],
        )
        names = ("island",)
        ports = (device.ports["left_arm"],)
    else:
        device = double_pad_transmon_with_bbox(
            pad_size=(point["pad_width"], 400.0),
            pad_gap=15.0,
            bbox_extension=point["ground_clearance"],
        )
        names = ("left_pad", "right_pad")
        ports = (device.ports["left_pad"], device.ports["right_pad"])
    local = gf.Component()
    local.add_ref(device)
    paddle = local.add_ref(
        plate_capacitor_single(length=point["paddle_length"], width=20)
    )
    paddle.dmove((
        ports[0].dx - point["coupling_gap"] - 10,
        ports[0].dy,
    ))
    bounds = local.dbbox()
    dbu = local.kcl.dbu
    frame = gf.kdb.Region(
        gf.kdb.DBox(
            bounds.left - 300,
            bounds.bottom - 300,
            bounds.right + 300,
            bounds.top + 300,
        ).to_itype(dbu)
    )
    etch = local.get_region(LAYER.M1_ETCH)
    # The CPW is external to this coupon; keep its port cut electrically open.
    pin = paddle.ports["o1"]
    cut = gf.kdb.DBox(pin.dx - 8, pin.dy - 20, pin.dx, pin.dy + 20)
    etch |= gf.kdb.Region(cut.to_itype(dbu))
    metal = ((frame - etch) | local.get_region(LAYER.M1_DRAW)).merged()
    terminals = (*names, "readout")
    terminal_ports = (*ports, pin)
    electrodes = [select_metal(metal, port, dbu) for port in terminal_ports]
    box = frame.bbox()
    ground = metal.interacting(
        gf.kdb.Region(gf.kdb.Box(box.left, box.bottom, box.left + 2, box.bottom + 2))
    )
    for i, electrode in enumerate(electrodes):
        if not (electrode & ground).is_empty():
            raise ValueError(f"{terminals[i]} touches reference ground")
        if any(not (electrode & other).is_empty() for other in electrodes[:i]):
            raise ValueError("Local qubit electrodes are shorted")
    component = gf.Component()
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for name, port in zip(terminals, terminal_ports, strict=True):
        component.add_port(name, port=port, port_type="electrical")
    pad = settings.domain_pad
    box = frame.bbox().to_dtype(dbu)
    return Geometry(
        component=component,
        bounds=(box.left - pad, box.bottom - pad, box.right + pad, box.top + pad),
        terminal_ports={name: name for name in terminals},
    )
