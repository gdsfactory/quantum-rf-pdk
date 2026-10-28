"""Bare double-pad qualification checks sheets, pins and canonical physics."""

from copy import deepcopy
from itertools import product

import gdsfactory as gf
import pytest

from qpdk.models.datasets.recipes._double_pad_geometry import (
    check_config,
    check_geometry,
)
from qpdk.models.datasets.recipes.curate_double_pad import GRID, SETTINGS
from qpdk.tech import LAYER


def native_inputs(point: dict, settings: dict) -> dict:
    """Draw independent rectangular reference sheets for portable tests."""
    w, h, gap, clearance = (point[k] for k in GRID)
    dbu = 0.001
    inner = gf.kdb.DBox(
        -w - gap / 2 - clearance,
        -h / 2 - clearance,
        w + gap / 2 + clearance,
        h / 2 + clearance,
    )
    outer = gf.kdb.DBox(
        inner.left - 300, inner.bottom - 300, inner.right + 300, inner.top + 300
    )
    pads = gf.kdb.Region(
        gf.kdb.DBox(-w - gap / 2, -h / 2, -gap / 2, h / 2).to_itype(dbu)
    ) | gf.kdb.Region(gf.kdb.DBox(gap / 2, -h / 2, w + gap / 2, h / 2).to_itype(dbu))
    metal = pads | (
        gf.kdb.Region(outer.to_itype(dbu)) - gf.kdb.Region(inner.to_itype(dbu))
    )
    # Resolve holes into simple polygons, matching the recorded sheet exporter.
    polygons = [
        [[p.x * dbu, p.y * dbu] for p in polygon.resolved_holes().each_point_hull()]
        for polygon in metal.each()
    ]
    pad = settings["domain_pad"]
    return {
        "point": point,
        "bounds_um": [
            outer.left - pad,
            outer.bottom - pad,
            outer.right + pad,
            outer.top + pad,
        ],
        "terminals": {"left_pad": [-w - gap / 2, 0.0], "right_pad": [w + gap / 2, 0.0]},
        "polygons_um": {str(gf.get_layer_tuple(LAYER.M1_DRAW)): polygons},
    }


def native_config(settings: dict) -> dict:
    """Return a procedural canonical electrostatic config, not solver evidence."""
    return {
        "Problem": {"Type": "Electrostatic"},
        "Model": {"L0": 1e-6, "Refinement": {"MaxIts": 0, "UniformLevels": 0}},
        "Solver": {
            "Order": settings["order"],
            "Linear": {
                "Tol": settings["tolerance"],
                "InitialGuess": False,
                "Type": "BoomerAMG",
                "KSPType": "GMRES",
            },
        },
        "Boundaries": {
            "Terminal": [
                {"Index": 1, "Attributes": [1]},
                {"Index": 2, "Attributes": [2]},
            ],
            "Ground": {"Attributes": [3]},
        },
        "Domains": {
            "Materials": [
                {"Attributes": [4], "Permittivity": 1.0},
                {"Attributes": [5], "Permittivity": 11.45},
            ]
        },
    }


def test_all_training_geometries_and_larger_domain() -> None:
    """Every production geometry agrees with independently drawn sheets."""
    for coords in product(*GRID.values()):
        point = dict(zip(GRID, coords, strict=True))
        proof = check_geometry(native_inputs(point, SETTINGS), SETTINGS)
        assert proof["metal_xor_area_um2"] == 0
    settings = SETTINGS | {"domain_pad": 2400.0}
    point = dict(zip(GRID, [250.0, 400.0, 15.0, 200.0], strict=True))
    assert (
        check_geometry(native_inputs(point, settings), settings)[
            "mesh_group_membership"
        ]
        == "unavailable"
    )


@pytest.mark.parametrize("damage", ["polygon", "domain", "pin", "layer", "order"])
def test_geometry_damage(damage: str) -> None:
    """Native geometry errors remain visible rather than inferred from metadata."""
    inputs = native_inputs(
        dict(zip(GRID, [250.0, 400.0, 15.0, 200.0], strict=True)), SETTINGS
    )
    layer = next(iter(inputs["polygons_um"]))
    match damage:
        case "polygon":
            inputs["polygons_um"][layer][0][0][0] += 1
        case "domain":
            inputs["bounds_um"][0] -= 1
        case "pin":
            inputs["terminals"]["left_pad"][0] += 1
        case "layer":
            inputs["polygons_um"]["(2, 0)"] = inputs["polygons_um"].pop(layer)
        case "order":
            inputs["terminal_order"] = ["right_pad", "left_pad"]
    with pytest.raises(ValueError, match="Native"):
        check_geometry(inputs, SETTINGS)


@pytest.mark.parametrize(
    "damage",
    [
        "terminal",
        "ground",
        "air",
        "silicon",
        "loss",
        "conductivity",
        "order",
        "tolerance",
        "initial",
        "scale",
        "refine",
    ],
)
def test_native_physics_damage(damage: str) -> None:
    """Reject changed physics even when a capacitance matrix looks plausible."""
    config = deepcopy(native_config(SETTINGS))
    match damage:
        case "terminal":
            config["Boundaries"]["Terminal"][0]["Attributes"] = [2]
        case "ground":
            config["Boundaries"]["Ground"]["Attributes"] = [4]
        case "air":
            config["Domains"]["Materials"][0]["Permittivity"] = 2.0
        case "silicon":
            config["Domains"]["Materials"][1]["Permittivity"] = 9.0
        case "loss" | "conductivity":
            config["Domains"]["Materials"][1][
                "LossTan" if damage == "loss" else "Conductivity"
            ] = 0.01
        case "order":
            config["Solver"]["Order"] = 3
        case "tolerance":
            config["Solver"]["Linear"]["Tol"] = 1e-4
        case "initial":
            config["Solver"]["Linear"]["InitialGuess"] = True
        case "scale":
            config["Model"]["L0"] = 1
        case "refine":
            config["Model"]["Refinement"]["MaxIts"] = 1
    with pytest.raises(ValueError, match="Native"):
        check_config(config, SETTINGS)


def test_native_config_accepts_baseline_and_p5() -> None:
    for order in (4, 5):
        settings = SETTINGS | {"order": order}
        check_config(native_config(settings), settings)
