"""Rebuild bare Xmon sheets and check the native sheet-driver inputs."""

from itertools import starmap
from math import isclose
from typing import Any

import gdsfactory as gf

from qpdk.cells import xmon_transmon
from qpdk.tech import LAYER


def _at_port(metal: gf.kdb.Region, port: gf.Port, dbu: float) -> gf.kdb.Region:
    """Select the one connected sheet touched by a layout pin."""
    x, y = (round(value / dbu) for value in port.center)
    selected = metal.interacting(gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2)))
    if selected.count() != 1:
        raise ValueError("Xmon pin does not select one conductor")
    return selected


def check_geometry(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Compare native polygons, the island pin and domain to current layout cells.

    This checks recorded geometry, not membership in an unavailable mesh group.

    Returns:
        Recomputed sheet, pin and domain evidence.

    Raises:
        ValueError: If native geometry differs from the current layout.
    """
    point = inputs["point"]
    device = xmon_transmon(
        arm_lengths=tuple(
            point["arm_scale"] * length for length in (160, 120, 160, 120)
        ),
        gap_width=point["gap_width"],
    )
    dbu = device.kcl.dbu
    etch = device.get_region(LAYER.M1_ETCH)
    box = etch.bbox().to_dtype(dbu)
    frame = gf.kdb.Region(
        gf.kdb.DBox(
            box.left - 300, box.bottom - 300, box.right + 300, box.top + 300
        ).to_itype(dbu)
    )
    metal = ((frame - etch) | device.get_region(LAYER.M1_DRAW)).merged()
    pin = device.ports["left_arm"]
    island = _at_port(metal, pin, dbu)
    for name in ("top_arm", "right_arm", "bottom_arm"):
        if not (_at_port(metal, device.ports[name], dbu) ^ island).is_empty():
            raise ValueError("Current Xmon arms do not share one island")
    bounds = frame.bbox()
    ground = metal.interacting(
        gf.kdb.Region(
            gf.kdb.Box(bounds.left, bounds.bottom, bounds.left + 2, bounds.bottom + 2)
        )
    )
    if ground.count() != 1 or metal.count() != 2 or not (ground & island).is_empty():
        raise ValueError("Current Xmon island and frame are not separate conductors")
    layer = str(gf.get_layer_tuple(LAYER.M1_DRAW))
    if set(inputs["polygons_um"]) != {layer}:
        raise ValueError("Native Xmon must contain only zero-thickness M1 sheets")
    native = gf.kdb.Region()
    for points in inputs["polygons_um"][layer]:
        native.insert(
            gf.kdb.DPolygon(list(starmap(gf.kdb.DPoint, points))).to_itype(dbu)
        )
    difference = native ^ metal
    if not difference.is_empty() or native.merged().count() != 2:
        raise ValueError("Native Xmon sheets differ from current cells")
    box = bounds.to_dtype(dbu)
    pad = settings["domain_pad"]
    expected_bounds = [box.left - pad, box.bottom - pad, box.right + pad, box.top + pad]
    if inputs["bounds_um"] != expected_bounds:
        raise ValueError("Native Xmon domain differs from its ground frame and padding")
    if inputs["terminals"] != {"island": list(pin.center)} or inputs[
        "terminal_order"
    ] != ["island"]:
        raise ValueError("Native Xmon terminal pin or order differs")
    return {
        "metal_xor_area_um2": difference.area() * dbu**2,
        "dbu_um": dbu,
        "island_area_um2": island.area() * dbu**2,
        "ground_area_um2": ground.area() * dbu**2,
        "bounds_um": expected_bounds,
        "pin_um": list(pin.center),
        "all_arms_share_island": True,
    }


def check_config(config: dict[str, Any], settings: dict[str, Any]) -> None:
    """Require the canonical electrostatic sheet attributes and lossless media.

    Raises:
        ValueError: If native physics or solver settings differ.
    """
    if config["Problem"]["Type"] != "Electrostatic" or not isclose(
        config["Model"]["L0"], 1e-6, rel_tol=0, abs_tol=0
    ):
        raise ValueError("Native problem or length scale differs")
    solver = config["Solver"]
    if (
        solver["Order"] != settings["order"]
        or solver["Linear"]["Tol"] != settings["tolerance"]
        or solver["Linear"]["InitialGuess"] is not False
    ):
        raise ValueError("Native field order, tolerance or initial guess differs")
    if (
        solver["Linear"]["Type"] != "BoomerAMG"
        or solver["Linear"]["KSPType"] != "GMRES"
    ):
        raise ValueError("Native linear solver differs")
    refinement = config["Model"]["Refinement"]
    if any(
        refinement.get(key, 0) != 0
        for key in ("MaxIts", "UniformLevels", "SerialUniformLevels")
    ):
        raise ValueError("Native mesh has unrecorded refinement")
    if config["Boundaries"] != {
        "Terminal": [{"Index": 1, "Attributes": [1]}],
        "Ground": {"Attributes": [2]},
    }:
        raise ValueError("Native terminal or ground sheet attributes differ")
    materials = config["Domains"]["Materials"]
    if len(materials) != 2 or sorted(m["Attributes"] for m in materials) != [[3], [4]]:
        raise ValueError("Native air or silicon material attributes differ")
    for material in materials:
        permittivity = (
            1.0 if material["Attributes"] == [3] else settings["permittivity"]
        )
        if (
            material["Permittivity"] != permittivity
            or material.get("Permeability", 1) != 1
            or any(
                material.get(key, 0) != 0
                for key in ("LossTan", "Conductivity", "LondonDepth")
            )
        ):
            raise ValueError("Native materials differ from lossless air and silicon")
