"""Rebuild recorded bare double-pad sheets without claiming mesh-group proof."""

from itertools import starmap
from math import isclose
from typing import Any

import gdsfactory as gf

from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.tech import LAYER

TERMINALS = ("left_pad", "right_pad")


def check_geometry(inputs: dict[str, Any], settings: dict[str, Any]) -> dict:
    """Check recorded polygons, terminal pins and padded finite ground frame.

    Returns:
        Recomputed sheet and domain evidence in um.

    Raises:
        ValueError: If recorded geometry differs from the current boxed cell.
    """
    point = inputs["point"]
    w, h, gap, clearance = (
        point[k] for k in ("pad_width", "pad_height", "pad_gap", "ground_clearance")
    )
    device = double_pad_transmon_with_bbox(
        pad_size=(w, h), pad_gap=gap, bbox_extension=clearance
    )
    dbu = device.kcl.dbu
    inner = gf.kdb.DBox(
        -w - gap / 2 - clearance,
        -h / 2 - clearance,
        w + gap / 2 + clearance,
        h / 2 + clearance,
    )
    outer = gf.kdb.DBox(
        inner.left - 300, inner.bottom - 300, inner.right + 300, inner.top + 300
    )
    ground = gf.kdb.Region(outer.to_itype(dbu)) - gf.kdb.Region(inner.to_itype(dbu))
    pads = device.get_region(LAYER.M1_DRAW).merged()
    metal = (pads | ground).merged()
    if pads.count() != 2 or metal.count() != 3 or not (pads & ground).is_empty():
        raise ValueError("Current pads and finite ground must be separate sheets")
    for name in TERMINALS:
        x, y = (round(v / dbu) for v in device.ports[name].center)
        selected = pads.interacting(
            gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2))
        )
        if selected.count() != 1:
            raise ValueError("Current double-pad pin must select one conductor")
    layer = str(gf.get_layer_tuple(LAYER.M1_DRAW))
    if set(inputs["polygons_um"]) != {layer}:
        raise ValueError("Native double-pad requires only zero-thickness M1 sheets")
    native = gf.kdb.Region()
    for polygon in inputs["polygons_um"][layer]:
        native.insert(
            gf.kdb.DPolygon(list(starmap(gf.kdb.DPoint, polygon))).to_itype(dbu)
        )
    difference = native ^ metal
    if not difference.is_empty() or native.merged().count() != 3:
        raise ValueError("Native double-pad polygons differ from current cells")
    pad = settings["domain_pad"]
    bounds = [outer.left - pad, outer.bottom - pad, outer.right + pad, outer.top + pad]
    pins = {name: list(device.ports[name].center) for name in TERMINALS}
    if inputs["bounds_um"] != bounds or inputs["terminals"] != pins:
        raise ValueError("Native frame, domain or terminal pins differ")
    if "terminal_order" in inputs and inputs["terminal_order"] != list(TERMINALS):
        raise ValueError("Native double-pad terminal order differs")
    return {
        "metal_xor_area_um2": difference.area() * dbu**2,
        "pad_area_um2": pads.area() * dbu**2,
        "ground_area_um2": ground.area() * dbu**2,
        "bounds_um": bounds,
        "pins_um": pins,
        "dbu_um": dbu,
        "mesh_group_membership": "unavailable",
    }


def check_config(config: dict[str, Any], settings: dict[str, Any]) -> None:
    """Require the canonical lossless electrostatic double-pad configuration.

    Raises:
        ValueError: If physics, terminal attributes or solver controls differ.
    """
    if config["Problem"]["Type"] != "Electrostatic" or not isclose(
        config["Model"]["L0"], 1e-6, rel_tol=0, abs_tol=0
    ):
        raise ValueError("Native problem or length scale differs")
    solver = config["Solver"]
    linear = solver["Linear"]
    if (
        solver["Order"] != settings["order"]
        or linear["Tol"] != settings["tolerance"]
        or linear["InitialGuess"] is not False
        or linear["Type"] != "BoomerAMG"
        or linear["KSPType"] != "GMRES"
    ):
        raise ValueError("Native solver order, tolerance or initial guess differs")
    if any(
        config["Model"]["Refinement"].get(k, 0) != 0
        for k in ("MaxIts", "UniformLevels", "SerialUniformLevels")
    ):
        raise ValueError("Native mesh has unrecorded refinement")
    if config["Boundaries"] != {
        "Terminal": [{"Index": 1, "Attributes": [1]}, {"Index": 2, "Attributes": [2]}],
        "Ground": {"Attributes": [3]},
    }:
        raise ValueError("Native terminal or ground attributes differ")
    materials = config["Domains"]["Materials"]
    if len(materials) != 2 or sorted(m["Attributes"] for m in materials) != [[4], [5]]:
        raise ValueError("Native air or silicon attributes differ")
    for material in materials:
        epsilon = 1.0 if material["Attributes"] == [4] else settings["permittivity"]
        if (
            not isclose(settings["permittivity"], 11.45, rel_tol=0, abs_tol=0)
            or material["Permittivity"] != epsilon
            or material.get("Permeability", 1) != 1
            or any(
                material.get(k, 0) != 0
                for k in ("LossTan", "Conductivity", "LondonDepth")
            )
        ):
            raise ValueError("Native materials differ from lossless air and silicon")
