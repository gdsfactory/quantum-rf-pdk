"""Procedural bare geometry-contract fixtures, never native FEM admission evidence."""

import hashlib
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk import cells
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import sweep, write
from qpdk.models.datasets.recipes import _bare_flipmon_geometry as geometry
from qpdk.tech import LAYER

AXES = geometry.AXES
TERMINALS = geometry.TERMINALS
type Fixture = tuple[Path, Dataset, dict[str, list[float]], str]


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(folder: Path, bond: str) -> Fixture:
    grid = dict(
        chip_gap=[7.5, 10.0], alignment=[15.0], inner_radius=[55.0], top_radius=[110.0]
    )
    settings = dict(
        near_mesh=1.4,
        far_mesh=60.0,
        domain_pad=600.0,
        permittivity=11.45,
        order=3,
        tolerance=1e-9,
    )
    provenance = {
        "settings": settings,
        "processes": 1,
        "fixture_kind": "procedural_geometry_contract_not_solver_evidence",
    }
    metadata = DatasetMetadata(
        name="geometry_contract_fixture",
        description="Procedural geometry contract only. Not FEM or scientific evidence.",
        axes=tuple(Axis(name=a, unit="um") for a in AXES),
        terminals=TERMINALS,
        variants=("bond",),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="mutual_capacitance",
                kind=QuantityKind.MUTUAL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="fem_error_indicator_norm",
                kind=QuantityKind.CIRCUIT_PARAMETER,
                unit="1",
            ),
        ),
        provenance={
            **provenance,
            "generator_sha256": "fixture",
            "experiment_sha256": "fixture",
        },
    )

    def solve(**point: float | str) -> dict:
        scale = float(point["chip_gap"]) * 1e-15
        c = jnp.asarray([[12.0, -2.0], [-2.0, 15.0]]) * scale
        cm = jnp.where(jnp.eye(2, dtype=bool), c.sum(axis=1)[:, None], -c)
        return {
            "maxwell_capacitance": c,
            "mutual_capacitance": cm,
            "fem_error_indicator_norm": 0.12,
        }

    data = write(
        folder / "data", metadata, sweep(metadata, solve, grid, {"bond": [bond]})
    )
    for values in product(*grid.values()):
        point = dict(zip(AXES, values, strict=True)) | {"bond": bond}
        layout = cells.flipmon_with_bbox(
            inner_circle_radius=float(point["inner_radius"]),
            top_circle_radius=float(point["top_radius"]),
        )
        dbu = layout.kcl.dbu
        frame = gf.kdb.Region(gf.kdb.DBox(-400, -400, 400, 400).to_itype(dbu))
        shifted = gf.Component()
        shifted.add_ref(layout.extract([LAYER.M2_DRAW, LAYER.M2_ETCH])).dmovex(
            float(point["alignment"])
        )
        masks = {}
        for draw, etch in (
            (LAYER.M1_DRAW, LAYER.M1_ETCH),
            (LAYER.M2_DRAW, LAYER.M2_ETCH),
            (LAYER.IND, None),
        ):
            if draw == LAYER.IND and bond == "ideal_wire":
                continue
            source = shifted if draw == LAYER.M2_DRAW else layout
            region = source.get_region(draw)
            if etch is not None:
                region = ((frame - source.get_region(etch)) | region).merged()
            # Native polygon serialization bridges holes into one vertex sequence.
            masks[str(gf.get_layer_tuple(draw))] = [
                [
                    [p.x * dbu, p.y * dbu]
                    for p in polygon.resolved_holes().each_point_hull()
                ]
                for polygon in region.each()
            ]
        top = float(point["chip_gap"]) + 0.2
        stack = {
            "layers": {
                name: dict(
                    name=name,
                    gds_layer=list(gf.get_layer_tuple(layer)),
                    zmin=lo,
                    zmax=hi,
                    thickness=hi - lo,
                    material="pec",
                    layer_type=kind,
                )
                for name, layer, lo, hi, kind in (
                    ("bottom", LAYER.M1_DRAW, 0, 0.2, "conductor"),
                    ("top", LAYER.M2_DRAW, top, top + 0.2, "conductor"),
                    ("bump", LAYER.IND, 0.2, top, "via"),
                )
            },
            "materials": {
                "pec": {"conductivity": 1e30},
                "silicon": {"permittivity": 11.45},
                "air": {"permittivity": 1.0},
            },
            "dielectrics": [
                dict(name="bottom_chip", zmin=-300, zmax=0, material="silicon"),
                dict(name="gap", zmin=0, zmax=top + 0.2, material="air"),
                dict(
                    name="top_chip",
                    zmin=top + 0.2,
                    zmax=top + 300.2,
                    material="silicon",
                ),
            ],
        }
        inputs = dict(
            point=point,
            provenance=provenance,
            terminal_order=list(TERMINALS),
            terminals=dict(island=[0.0, 0.0], ring=[100.0, 0.0]),
            polygons_um=masks,
            mesh_threads=1,
            minimum_feature_elements=4,
            bounds_um=[-1000.0, -1000.0, 1000.0, 1000.0],
            mesh_inputs=dict(
                stack=stack,
                airbox=dict(
                    margin_x=600.0, margin_y=600.0, z_above=600.0, z_below=600.0
                ),
                mesh=dict(
                    refined_mesh_size=1.4,
                    max_mesh_size=60.0,
                    planar_conductors=False,
                    merge_via_distance=0,
                    decimate_tolerance=0,
                    threads=1,
                    surface_threads=1,
                    verbose=True,
                ),
                bond=bond,
            ),
        )
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        run = folder / "runs" / key
        _save(run / "inputs.json", inputs)
        top_net, top_ground = (
            ("island", "net5") if bond == "physical_bump" else ("net5", "net6")
        )
        groups = {
            name: index
            for index, name in enumerate(
                [
                    "air__silicon",
                    "air__None",
                    *[
                        f"bottom_{face}@{net}"
                        for face in ("xy", "z")
                        for net in ("island", "ring", "net4")
                    ],
                    *[
                        f"top_{face}@{net}"
                        for face in ("xy", "z")
                        for net in (top_net, top_ground)
                    ],
                    *(["air__bump@island"] if bond == "physical_bump" else []),
                ],
                7,
            )
        }
        terminals = [
            dict(
                index=i,
                name=name,
                floating=False,
                nets=[i] if i != 1 or bond == "physical_bump" else [1, 4],
                net_names=[name]
                if i != 1 or bond == "physical_bump"
                else ["island", "net5"],
            )
            for i, name in enumerate(TERMINALS, 1)
        ]
        _save(run / "terminals.json", terminals)
        _save(run / "complete.json", {"terminals": list(TERMINALS)})
        volumes = {"silicon": 1, "air": 2} | (
            {"bump": 3} if bond == "physical_bump" else {}
        )
        _save(
            run / "metadata.json",
            dict(
                solver="palace",
                simulation=dict(
                    problem_type="Electrostatic",
                    field_order=3,
                    linear_solver={"InitialGuess": False, "Tol": 1e-9},
                    refinement={"MaxIts": 0, "UniformLevels": 0},
                ),
                mesh=dict(
                    dimension=3,
                    nodes=100,
                    elements=200,
                    tetrahedra=150,
                    geometry_orders=[1],
                    edge_length={"min": 0.2, "max": 60.0},
                    quality={"min": 0.1, "max": 1.0, "mean": 0.6},
                    sicn={"invalid": 0},
                    kappa={"singular_elements": 0},
                    field_dofs={"estimated_field_dofs": None, "field_order": 3},
                    mesh_hash="sha256:procedural-fixture",
                    bbox=dict(
                        xmin=-1000.0,
                        ymin=-1000.0,
                        zmin=-900.0,
                        xmax=1000.0,
                        ymax=1000.0,
                        zmax=float(point["chip_gap"]) + 900.4,
                    ),
                    groups=dict(
                        surfaces=[
                            dict(name=n, tag=t, elements=10) for n, t in groups.items()
                        ],
                        volumes=[
                            dict(name=n, tag=t, elements=10) for n, t in volumes.items()
                        ],
                    ),
                ),
            ),
        )
        cfg = dict(
            Problem={"Type": "Electrostatic"},
            Model={"L0": 1e-6, "Refinement": {"MaxIts": 0, "UniformLevels": 0}},
            Solver={"Order": 3, "Linear": {"InitialGuess": False, "Tol": 1e-9}},
            Boundaries={
                "Terminal": [
                    {
                        "Index": i,
                        "Attributes": [
                            tag
                            for name, tag in groups.items()
                            if "@" in name and name.split("@")[1] in t["net_names"]
                        ],
                    }
                    for i, t in enumerate(terminals, 1)
                ],
                "Ground": {
                    "Attributes": [
                        tag
                        for name, tag in groups.items()
                        if name.endswith(("@net4", "@" + top_ground))
                    ]
                },
            },
            Domains={
                "Materials": [
                    {
                        "Attributes": [tag],
                        "Permittivity": 11.45 if name == "silicon" else 1.0,
                        "Conductivity": 1e30 if name == "bump" else 0.0,
                        "Permeability": 1.0,
                        "LossTan": 0.0,
                        "LondonDepth": 0.0,
                    }
                    for name, tag in volumes.items()
                ]
            },
        )
        _save(run / "output/palace/config_resolved.json", cfg)
        (run / "output/palace/error-indicators.csv").write_text(
            "Norm,Minimum,Maximum,Mean\n0.12,0.001,0.02,0.01\n", encoding="utf-8"
        )
        matrices = solve(**point)
        for name, matrix in (
            ("C", matrices["maxwell_capacitance"]),
            ("Cm", matrices["mutual_capacitance"]),
            ("Cinv", jnp.linalg.inv(matrices["maxwell_capacitance"])),
        ):
            unit = "1/F" if name == "Cinv" else "F"
            np.savetxt(
                run / f"output/palace/terminal-{name}.csv",
                jnp.column_stack((jnp.arange(1, 3), matrix)),
                delimiter=",",
                header=f"i,1 ({unit}),2 ({unit})",
                comments="",
            )
    return folder, data, grid, bond


@pytest.fixture(params=["physical_bump", "ideal_wire"])
def native_shape(tmp_path: Path, request: pytest.FixtureRequest) -> Fixture:
    """Build recorded-evidence shapes without claiming a completed FEM solve."""
    return _fixture(tmp_path, request.param)


def _verify(fixture: Fixture) -> dict:
    folder, data, grid, bond = fixture
    return geometry.verify_geometry(folder, data, grid, bond)


def _run(fixture: Fixture) -> Path:
    return min((fixture[0] / "runs").iterdir())


def _change(
    fixture: Fixture, filename: str, damage: str, *, rekey: bool = False
) -> None:
    path = _run(fixture) / filename
    value: dict | list = json.loads(path.read_text(encoding="utf-8"))
    if filename == "terminals.json":
        assert isinstance(value, list)
        if damage == "order":
            value.reverse()
        elif damage == "floating":
            value[0]["floating"] = True
        else:
            value[1]["nets"] = value[0]["nets"]
    else:
        assert isinstance(value, dict)
        match damage:
            case "run_hash":
                value["mesh_threads"] = 2
            case "provenance":
                value["provenance"]["fixture_kind"] = "changed"
            case "stack":
                value["mesh_inputs"]["stack"]["dielectrics"][2]["zmin"] = 50.0
            case "alignment":
                value["point"]["alignment"] = -15.0
            case "pin":
                value["terminals"]["ring"][0] = 0.0
            case "footprint":
                value["polygons_um"]["(1, 0)"].pop()
            case "bump":
                value["polygons_um"].pop("(30, 0)", None)
            case "domain":
                value["bounds_um"][0] = -999.0
            case "mesh":
                value["mesh_inputs"]["mesh"]["planar_conductors"] = True
            case "metadata_order":
                value["simulation"]["field_order"] = 4
            case "dof_order":
                value["mesh"]["field_dofs"]["field_order"] = 4
            case "duplicate_group":
                groups = value["mesh"]["groups"]["surfaces"]
                groups[0]["tag"] = groups[1]["tag"]
            case "empty_group":
                value["mesh"]["groups"]["volumes"][0]["elements"] = 0
            case "bbox":
                value["mesh"]["bbox"]["zmax"] = 9999.0
            case "invalid_mesh":
                value["mesh"]["sicn"]["invalid"] = 1
            case "terminal_attrs":
                terminals = value["Boundaries"]["Terminal"]
                terminals[0]["Attributes"] = terminals[1]["Attributes"]
            case "ground":
                value["Boundaries"]["Ground"]["Attributes"].append(
                    value["Boundaries"]["Terminal"][0]["Attributes"][0]
                )
            case "material":
                value["Domains"]["Materials"][0]["Permittivity"] = 3.0
            case "loss":
                value["Domains"]["Materials"][0]["LossTan"] = 0.001
            case "order":
                value["Solver"]["Order"] = 4
            case "residual":
                value["Solver"]["Linear"]["Tol"] = 1e-4
            case "initial_guess":
                value["Solver"]["Linear"]["InitialGuess"] = True
            case "adaptive":
                value["Model"]["Refinement"]["MaxIts"] = 2
            case "recorded_solver":
                value["simulation"]["linear_solver"]["Tol"] = 1e-4
            case "completion":
                value["terminals"] = ["ring", "island"]
    _save(path, value)
    if rekey:
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        path.parent.rename(path.parent.with_name(key))


def test_geometry_contract(native_shape: Fixture) -> None:
    report = _verify(native_shape)
    assert report["passed"]
    assert len(report["points"]) == 2
    assert report["factory"] == "qpdk.cells.flipmon_with_bbox"
    assert "scientific_admission" not in report
    for point in report["points"]:
        assert point["bottom_conductors"] == 3
        assert point["top_conductors"] == 2
        assert point["mesh"]["field_dofs"]["estimated_field_dofs"] is None
        assert set(point["topology"]) == {"island", "ring", "ground"}
        assert (
            point["raw"]["inputs"]["provenance"]["fixture_kind"]
            == "procedural_geometry_contract_not_solver_evidence"
        )
        assert np.asarray(point["matrices"]["C"]).shape == (2, 2)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("filename", "damage", "message", "rekey"),
    [
        ("inputs.json", "run_hash", "run key", False),
        ("inputs.json", "provenance", "provenance", True),
        ("inputs.json", "stack", "stack", True),
        ("inputs.json", "alignment", "native point", True),
        ("inputs.json", "pin", "pins", True),
        ("inputs.json", "footprint", "footprint", True),
        ("inputs.json", "domain", "domain", True),
        ("inputs.json", "mesh", "mesh settings", True),
        ("metadata.json", "metadata_order", "simulation", False),
        ("metadata.json", "dof_order", "simulation", False),
        ("metadata.json", "recorded_solver", "recorded solver", False),
        ("metadata.json", "duplicate_group", "duplicate", False),
        ("metadata.json", "empty_group", "empty", False),
        ("metadata.json", "bbox", "domain", False),
        ("metadata.json", "invalid_mesh", "mesh diagnostics", False),
        ("terminals.json", "order", "terminal identities", False),
        ("terminals.json", "floating", "terminal identities", False),
        ("terminals.json", "overlap", "terminal nets", False),
        ("complete.json", "completion", "completion", False),
        ("output/palace/config_resolved.json", "terminal_attrs", "attributes", False),
        ("output/palace/config_resolved.json", "ground", "attributes", False),
        (
            "output/palace/config_resolved.json",
            "material",
            "material properties",
            False,
        ),
        ("output/palace/config_resolved.json", "loss", "material properties", False),
        ("output/palace/config_resolved.json", "order", "solver settings", False),
        ("output/palace/config_resolved.json", "residual", "solver settings", False),
        ("output/palace/config_resolved.json", "adaptive", "refinement", False),
        (
            "output/palace/config_resolved.json",
            "initial_guess",
            "solver settings",
            False,
        ),
    ],
)
def test_native_corruption(
    native_shape: Fixture, filename: str, damage: str, message: str, rekey: bool
) -> None:
    _change(native_shape, filename, damage, rekey=rekey)
    with pytest.raises(ValueError, match=message):
        _verify(native_shape)


@pytest.mark.parametrize("matrix", ["C", "Cm", "Cinv"])
def test_raw_matrix_corruption(native_shape: Fixture, matrix: str) -> None:
    path = _run(native_shape) / f"output/palace/terminal-{matrix}.csv"
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    raw[0, 1] *= 1.01
    unit = "1/F" if matrix == "Cinv" else "F"
    np.savetxt(path, raw, delimiter=",", header=f"i,1 ({unit}),2 ({unit})", comments="")
    with pytest.raises(ValueError, match=r"matrix|matrices"):
        _verify(native_shape)


def test_bump_is_required_only_for_physical_bond(native_shape: Fixture) -> None:
    if native_shape[3] != "physical_bump":
        return
    _change(native_shape, "inputs.json", "bump", rekey=True)
    with pytest.raises(ValueError, match="layers"):
        _verify(native_shape)


def test_synthetic_cannot_supply_geometry(native_shape: Fixture) -> None:
    data = native_shape[1]
    data.metadata = data.metadata.model_copy(update={"synthetic": True})
    with pytest.raises(ValueError, match="synthetic"):
        _verify(native_shape)


def test_missing_and_duplicate_runs(native_shape: Fixture) -> None:
    run = _run(native_shape)
    shutil.copytree(run, run.with_name("000-duplicate"))
    with pytest.raises(ValueError, match=r"run key|duplicate"):
        _verify(native_shape)


def test_full_grid_required(native_shape: Fixture) -> None:
    folder, data, grid, bond = native_shape
    with pytest.raises(ValueError, match="missing native runs"):
        geometry.verify_geometry(
            folder, data, grid | {"chip_gap": [7.5, 10.0, 12.5]}, bond
        )


def test_current_cells_are_used(
    native_shape: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = cells.flipmon_with_bbox

    def changed(**kwargs: Any) -> gf.Component:
        kwargs["inner_circle_radius"] = 40.0
        return original(**kwargs)

    monkeypatch.setattr(geometry.cells, "flipmon_with_bbox", changed)
    with pytest.raises(ValueError, match="footprint"):
        _verify(native_shape)


def test_raw_indicator_must_match_parquet(native_shape: Fixture) -> None:
    path = _run(native_shape) / "output/palace/error-indicators.csv"
    path.write_text(
        "Norm,Minimum,Maximum,Mean\n0.2,0.001,0.02,0.01\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="FEM indicator"):
        _verify(native_shape)


def test_rounded_zero_quality_does_not_invent_invalid_elements(
    native_shape: Fixture,
) -> None:
    path = _run(native_shape) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["mesh"]["quality"]["min"] = 0.0
    metadata["mesh"]["sicn"]["min"] = 0.0
    _save(path, metadata)
    assert _verify(native_shape)["passed"]


@pytest.mark.parametrize("matrix", ["C", "Cm", "Cinv"])
@pytest.mark.parametrize(("unit", "scale"), [("F", 1.0), ("pF", 1e-12), ("fF", 1e-15)])
def test_csv_units_are_explicit(
    native_shape: Fixture, matrix: str, unit: str, scale: float
) -> None:
    path = _run(native_shape) / f"output/palace/terminal-{matrix}.csv"
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if matrix == "Cinv":
        raw[:, 1:] *= scale
        unit = "1/" + unit
    else:
        raw[:, 1:] /= scale
    np.savetxt(path, raw, delimiter=",", header=f"i,1 ({unit}),2 ({unit})", comments="")
    assert _verify(native_shape)["passed"]


@pytest.mark.parametrize("header", ["i,1,2", "i,1 (pF),2 (F)", "i,1 (H),2 (H)"])
def test_missing_or_inconsistent_csv_units_fail(
    native_shape: Fixture, header: str
) -> None:
    path = _run(native_shape) / "output/palace/terminal-C.csv"
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    np.savetxt(path, raw, delimiter=",", header=header, comments="")
    with pytest.raises(ValueError, match="matrix units"):
        _verify(native_shape)
