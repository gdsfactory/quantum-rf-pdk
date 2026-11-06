"""Procedural corruption fixtures for Xmon curation, never FEM admission evidence."""

import copy
import hashlib
import json
import shutil
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import polars as pl
import pytest

from qpdk import PDK
from qpdk.cells import xmon_transmon
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import sweep
from qpdk.models.datasets.metadata import METADATA_KEY
from qpdk.models.datasets.recipes import _xmon_geometry, curate_xmon
from qpdk.models.datasets.validation import validate_native_capacitances
from qpdk.tech import LAYER


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _plan() -> dict:
    grid = {
        "arm_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
        "gap_width": [8.0, 9.0, 10.0, 11.0, 12.0],
    }
    settings = {"near_mesh": 0.7, "far_mesh": 48.0, "domain_pad": 1200.0, "order": 3}
    production = {
        "recipe": "xmon",
        "label": "production",
        "mode": "production",
        "grid": grid,
        "settings": settings,
    }
    controls = []
    for anchor, point in zip(
        ("nominal", "small", "large", "wide-gap"),
        (
            {"arm_scale": 1.0, "gap_width": 10.0},
            {"arm_scale": 0.8, "gap_width": 8.0},
            {"arm_scale": 1.2, "gap_width": 12.0},
            {"arm_scale": 0.8, "gap_width": 12.0},
        ),
        strict=True,
    ):
        for control, change in (
            ("baseline", {}),
            ("fine-h", {"near_mesh": 0.49, "far_mesh": 38.4}),
            ("hp4", {"order": 4}),
            ("domain", {"domain_pad": 1800.0}),
        ):
            controls.append({
                "recipe": "xmon",
                "label": anchor + "-" + control,
                "mode": "control",
                "point": point,
                "anchor": anchor,
                "control": control,
                "settings": settings | change,
            })
    centres = [
        dict(zip(grid, coords, strict=True))
        for coords in product(*[
            [(a + b) / 2 for a, b in pairwise(values)] for values in grid.values()
        ])
    ]
    faces = [{"arm_scale": arm, "gap_width": 10.0} for arm in (0.8, 1.2)] + [
        {"arm_scale": 1.0, "gap_width": gap} for gap in (8.0, 12.0)
    ]
    held = [
        {
            "recipe": "xmon",
            "label": f"held-{i}",
            "mode": "control",
            "point": point,
            "anchor": f"dense-held-{i}",
            "control": "fine-h",
            "settings": settings | {"near_mesh": 0.49, "far_mesh": 38.4},
        }
        for i, point in enumerate(centres + faces)
    ]
    return {
        "grids": {"xmon": grid},
        "jobs": [
            {"name": "production", "tasks": [production], "expected_points": 25},
            {"name": "numerical", "tasks": controls, "expected_points": 16},
            {"name": "held", "tasks": held, "expected_points": 20},
        ],
        "scope": "Procedural fixture, not FEM evidence",
        "source_revision": "procedural-geometry",
        "gsim_revision": "procedural-solver",
        "gsim_source_verification": {
            "all_package_files_match_source": True,
            "evidence_kind": "procedural_fixture_not_FEM",
        },
    }


@pytest.fixture
def study(tmp_path: Path) -> Path:
    source = tmp_path / "study/source"
    source.mkdir(parents=True)
    recipes = Path(curate_xmon.__file__).parent
    for name in ("xmon.py", "_palace.py", "_experiment.py"):
        target = source / "qpdk/models/datasets/data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (recipes / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (source / "batch.py").write_text(
        "# Procedural fixture, not a solver driver.\n", encoding="utf-8"
    )
    (source / "gsim-fixture.whl").write_bytes(b"Procedural fixture, not a wheel")
    plan = _plan()
    plan |= {
        "gsim_wheel_sha256": curate_xmon._sha256(source / "gsim-fixture.whl"),
        "recipe_sha256": curate_xmon._sha256(
            source / "qpdk/models/datasets/data/xmon.py"
        ),
    }
    _save(source / "plan.json", plan)
    _save(
        source / "source-sha256.json",
        {
            p.relative_to(source).as_posix(): curate_xmon._sha256(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    )
    return source.parent


def test_pending_preview_cannot_export(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = curate_xmon.verify(study)
    assert not report["passed"]
    assert "scientific_admission" not in report
    assert (
        report["admission_status"]
        == "candidate_pending_independent_numerical_validation"
    )
    assert report["expected"] == {
        "native_points": 61,
        "production_points": 25,
        "anchors": 4,
        "numerical": 12,
        "held": 20,
        "references": 36,
        "interior_cells": 16,
        "strict_faces": [
            ["arm_scale", 0.8],
            ["arm_scale", 1.2],
            ["gap_width", 8.0],
            ["gap_width", 12.0],
        ],
    }
    assert len(report["pending"]) == 37
    writes = []
    monkeypatch.setattr(curate_xmon, "write", lambda *args: writes.append(args))
    with pytest.raises(ValueError, match="incomplete"):
        curate_xmon.verify(study, output=study / "must-not-export")
    assert not writes
    assert not (study / "must-not-export").exists()


def test_frozen_sources_are_checked_fresh(study: Path) -> None:
    (study / "source/batch.py").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Frozen source changed: batch\.py"):
        curate_xmon.verify(study)


@pytest.mark.parametrize(
    "damage",
    [
        "anchor",
        "control",
        "held",
        "face",
        "grid",
        "job_count",
        "duplicate_job",
        "duplicate_held",
        "production",
    ],
)
def test_plan_structure_is_required(damage: str) -> None:
    plan = _plan()
    match damage:
        case "anchor":
            plan["jobs"][1]["tasks"][0]["anchor"] = "wrong-anchor"
        case "control":
            plan["jobs"][1]["tasks"][1]["control"] = "baseline"
        case "held":
            plan["jobs"][2]["tasks"].pop()
        case "face":
            plan["jobs"][2]["tasks"][-1]["point"] = {
                "arm_scale": 1.2,
                "gap_width": 12.0,
            }
        case "grid":
            plan["grids"]["xmon"]["gap_width"].pop()
        case "job_count":
            plan["jobs"][0]["expected_points"] -= 1
        case "duplicate_job":
            plan["jobs"][1]["name"] = "production"
        case "duplicate_held":
            plan["jobs"][2]["tasks"][-1]["point"] = plan["jobs"][2]["tasks"][0]["point"]
        case "production":
            plan["jobs"][0]["tasks"][0]["grid"] = {
                "arm_scale": [1.0],
                "gap_width": [10.0],
            }
    with pytest.raises(ValueError, match=r"Native|Dataset|Xmon|Bare|Frozen|Every|Each"):
        curate_xmon._structure(plan)


@pytest.fixture
def table() -> tuple[pl.DataFrame, DatasetMetadata, dict]:
    metadata = DatasetMetadata(
        name="procedural_xmon_fixture",
        description="Procedural circuit values, not FEM evidence",
        synthetic=True,
        axes=(Axis(name="arm_scale", unit="1"), Axis(name="gap_width", unit="um")),
        terminals=("island",),
        reference_ground="finite_coplanar_chip_ground",
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
            *(
                Quantity(name=name, kind=QuantityKind.CIRCUIT_PARAMETER, unit="1")
                for name in (
                    "fem_error_indicator_norm",
                    "solver_iterations",
                    "solver_relative_residual",
                )
            ),
        ),
    )
    grid = {"arm_scale": [0.8, 1.0], "gap_width": [8.0, 10.0]}

    def solve(**_point: float) -> dict:
        return {
            "maxwell_capacitance": jnp.asarray([[80e-15]]),
            "mutual_capacitance": jnp.asarray([[80e-15]]),
            "fem_error_indicator_norm": 0.12,
            "solver_iterations": 20.0,
            "solver_relative_residual": 1e-10,
        }

    return sweep(metadata, solve, grid), metadata, grid


@pytest.mark.parametrize(
    "damage",
    [
        "quantity",
        "coordinate",
        "cell",
        "mutual",
        "negative_fem",
        "fractional_iterations",
        "negative_iterations",
        "residual",
        "nan",
    ],
)
def test_quantity_corruptions(
    table: tuple[pl.DataFrame, DatasetMetadata, dict], damage: str
) -> None:
    frame, metadata, grid = table
    quantity = pl.col("quantity")
    match damage:
        case "quantity":
            frame = frame.filter(quantity != "solver_iterations")
        case "coordinate":
            frame = frame.filter(
                ~(
                    (quantity == "fem_error_indicator_norm")
                    & (pl.col("arm_scale").is_in([1.0]))
                )
            )
        case "cell":
            frame = frame.filter(
                ~(
                    (quantity == "fem_error_indicator_norm")
                    & (pl.col("arm_scale").is_in([1.0]))
                    & (pl.col("gap_width").is_in([10.0]))
                )
            )
        case _:
            name, value = {
                "mutual": ("mutual_capacitance", 1e-9),
                "negative_fem": ("fem_error_indicator_norm", -1.0),
                "fractional_iterations": ("solver_iterations", 1.5),
                "negative_iterations": ("solver_iterations", -1.0),
                "residual": ("solver_relative_residual", 1e-4),
                "nan": ("solver_relative_residual", float("nan")),
            }[damage]
            frame = frame.with_columns(
                pl
                .when(quantity == name)
                .then(value)
                .otherwise(pl.col("value"))
                .alias("value")
            )
    with pytest.raises(ValueError, match=r"Native|Dataset|Xmon|Bare|Frozen|Every|Each"):
        validate_native_capacitances(frame, metadata, grid, {}, tolerance=1e-9)


def test_synthetic_reference_is_retained_and_rejected(
    study: Path, table: tuple[pl.DataFrame, DatasetMetadata, dict]
) -> None:
    plan = json.loads((study / "source/plan.json").read_text(encoding="utf-8"))
    task = plan["jobs"][1]["tasks"][0]
    frame, metadata, _ = table
    selected = frame.filter(
        (pl.col("arm_scale").is_in([1.0])) & (pl.col("gap_width").is_in([10.0]))
    )
    part = study / "results/numerical" / task["label"] / "data/part-0000.parquet"
    part.parent.mkdir(parents=True)
    selected.write_parquet(part, metadata={METADATA_KEY: metadata.to_json()})
    _save(
        study / "submission-1.json",
        {
            "returncode": 0,
            "stdout": "123\n",
            "name": "numerical",
            "evidence_kind": "procedural_fixture",
        },
    )
    _save(study / "results/numerical/summary.json", [task | {"status": "ok"}])
    report = curate_xmon.verify(study)
    assert not report["passed"]
    assert len(report["failed"]) == 1
    assert "Synthetic" in report["failed"][0]["error"]
    assert report["failed"][0]["raw_native_rows"] == selected.to_dicts()


@pytest.fixture
def packet(tmp_path: Path) -> dict:
    return _packet(tmp_path)


def _packet(
    tmp_path: Path,
    *,
    point: dict | None = None,
    settings: dict | None = None,
    provenance: dict | None = None,
    driver_sha256: str = "fixture-driver",
    capacitance: float = 80e-15,
    tetrahedra: int = 1000,
) -> dict:
    """Write one procedural packet with actual recorded mesh and order fields."""
    PDK.activate()
    point = point or {"arm_scale": 1.0, "gap_width": 10.0}
    settings = settings or {
        "near_mesh": 0.7,
        "far_mesh": 48.0,
        "domain_pad": 1200.0,
        "permittivity": 11.45,
        "order": 3,
        "tolerance": 1e-9,
    }
    device = xmon_transmon(
        arm_lengths=tuple(
            point["arm_scale"] * length for length in (160, 120, 160, 120)
        ),
        gap_width=point["gap_width"],
    )
    etch = device.get_region(LAYER.M1_ETCH)
    box = etch.bbox().to_dtype(device.kcl.dbu)
    frame = gf.kdb.Region(
        gf.kdb.DBox(
            box.left - 300, box.bottom - 300, box.right + 300, box.top + 300
        ).to_itype(device.kcl.dbu)
    )
    metal = ((frame - etch) | device.get_region(LAYER.M1_DRAW)).merged()
    component = gf.Component()
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(device.kcl.dbu), layer=LAYER.M1_DRAW)
    box = frame.bbox().to_dtype(device.kcl.dbu)
    provenance = provenance or {
        "processes": 1,
        "settings": settings,
        "evidence_kind": "procedural_fixture_not_FEM",
        "generator_sha256": "fixture-generator",
        "experiment_sha256": "fixture-experiment",
    }
    inputs = {
        "point": point,
        "mesh_inputs": {
            "driver_sha256": driver_sha256,
            "settings": settings,
            "minimum_feature_elements": 4,
        },
        "bounds_um": [
            box.left - settings["domain_pad"],
            box.bottom - settings["domain_pad"],
            box.right + settings["domain_pad"],
            box.top + settings["domain_pad"],
        ],
        "terminals": {"island": list(device.ports["left_arm"].center)},
        "polygons_um": {
            str(layer): [p.tolist() for p in polygons]
            for layer, polygons in component.get_polygons_points(by="tuple").items()
        },
        "provenance": {
            k: v
            for k, v in provenance.items()
            if k not in {"generator_sha256", "experiment_sha256"}
        },
        "terminal_order": ["island"],
        "minimum_feature_elements": 4,
        "mesh_threads": provenance["processes"],
    }
    run = (
        tmp_path
        / hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    )
    config = {
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
            "Terminal": [{"Index": 1, "Attributes": [1]}],
            "Ground": {"Attributes": [2]},
        },
        "Domains": {
            "Materials": [
                {"Attributes": [3], "Permittivity": 1.0},
                {"Attributes": [4], "Permittivity": 11.45},
            ]
        },
    }
    _save(run / "inputs.json", inputs)
    _save(run / "complete.json", {"terminals": ["island"]})
    _save(run / "config.json", config)
    _save(run / "output/palace/config_resolved.json", config)
    _save(
        run / "metadata.json",
        {
            "solver": "palace",
            "simulation": {
                "problem_type": "Electrostatic",
                "field_order": settings["order"],
                "linear_solver": config["Solver"]["Linear"],
                "refinement": config["Model"]["Refinement"],
            },
            "mesh": {
                "tetrahedra": tetrahedra,
                "field_dofs": {
                    "field_order": settings["order"],
                    "estimated_field_dofs": None,
                    "unavailable_reason": "Procedural fixture",
                },
            },
        },
    )
    for name, value in (
        ("C", capacitance),
        ("Cm", capacitance),
        ("Cinv", 1 / capacitance),
    ):
        symbol, unit = {"C": ("C", "F"), "Cm": ("C_m", "F"), "Cinv": ("C⁻¹", "1/F")}[
            name
        ]
        (run / f"output/palace/terminal-{name}.csv").write_text(
            f"i,{symbol}[i][1] ({unit})\n1,{value:.15e}\n", encoding="utf-8"
        )
    (run / "output/palace/error-indicators.csv").write_text(
        "Norm,Minimum,Maximum,Mean\n0.12,0.001,0.02,0.01\n", encoding="utf-8"
    )
    return {
        "run": run,
        "point": point,
        "settings": settings,
        "provenance": provenance,
        "driver_sha256": driver_sha256,
        "values": {
            "maxwell_capacitance": [capacitance],
            "mutual_capacitance": [capacitance],
            "fem_error_indicator_norm": [0.12],
        },
    }


def _edit_inputs(packet: dict, inputs: dict, *, rehash: bool = True) -> None:
    run = packet["run"]
    _save(run / "inputs.json", inputs)
    if rehash:
        renamed = run.with_name(
            hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
        )
        run.rename(renamed)
        packet["run"] = renamed


def test_procedural_packet_checks_only_recorded_evidence(packet: dict) -> None:
    proof = curate_xmon._native_run(**packet)
    assert proof["geometry"]["metal_xor_area_um2"] == 0
    assert proof["tetrahedra"] == 1000
    assert proof["field_order"] == 3
    assert proof["field_dofs"]["estimated_field_dofs"] is None
    assert (
        proof["inputs"]["provenance"]["evidence_kind"] == "procedural_fixture_not_FEM"
    )
    assert "admission_status" not in proof
    assert "passed" not in proof


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "point",
        "driver",
        "settings",
        "provenance",
        "pin",
        "domain",
        "polygon",
        "layer",
        "terminal",
        "threads",
        "feature",
    ],
)
def test_native_input_corruption(packet: dict, damage: str) -> None:
    inputs = json.loads((packet["run"] / "inputs.json").read_text(encoding="utf-8"))
    match damage:
        case "hash" | "driver":
            inputs["mesh_inputs"]["driver_sha256"] = "wrong-driver"
        case "point":
            inputs["point"]["gap_width"] = 9.0
        case "settings":
            inputs["mesh_inputs"]["settings"]["order"] = 4
        case "provenance":
            inputs["provenance"]["evidence_kind"] = "wrong"
        case "pin":
            inputs["terminals"]["island"][0] += 10.0
        case "domain":
            inputs["bounds_um"][0] -= 10.0
        case "polygon":
            inputs["polygons_um"]["(1, 0)"][0][0][0] += 10.0
        case "layer":
            inputs["polygons_um"]["(2, 0)"] = inputs["polygons_um"].pop("(1, 0)")
        case "terminal":
            inputs["terminal_order"] = ["ground"]
        case "threads":
            inputs["mesh_threads"] += 1
        case "feature":
            inputs["minimum_feature_elements"] = 1
    _edit_inputs(packet, inputs, rehash=damage != "hash")
    with pytest.raises(ValueError, match=r"Native|Dataset|Xmon|Bare|Frozen|Every|Each"):
        curate_xmon._native_run(**packet)


@pytest.mark.parametrize("file", ["config.json", "output/palace/config_resolved.json"])
@pytest.mark.parametrize(
    "damage",
    [
        "ground",
        "terminal",
        "permittivity",
        "air",
        "loss",
        "conductivity",
        "london",
        "extra_pec",
        "order",
        "tolerance",
        "initial_guess",
        "scale",
        "adaptive",
        "material_attribute",
    ],
)
def test_native_config_corruption(packet: dict, file: str, damage: str) -> None:
    path = packet["run"] / file
    config = json.loads(path.read_text(encoding="utf-8"))
    match damage:
        case "ground":
            config["Boundaries"]["Ground"]["Attributes"] = [1]
        case "terminal":
            config["Boundaries"]["Terminal"][0]["Attributes"] = [2]
        case "permittivity":
            config["Domains"]["Materials"][1]["Permittivity"] = 9.0
        case "air":
            config["Domains"]["Materials"][0]["Permittivity"] = 2.0
        case "loss" | "conductivity" | "london":
            key = {
                "loss": "LossTan",
                "conductivity": "Conductivity",
                "london": "LondonDepth",
            }[damage]
            config["Domains"]["Materials"][1][key] = 0.01
        case "extra_pec":
            config["Boundaries"]["PEC"] = {"Attributes": [5]}
        case "order":
            config["Solver"]["Order"] = 2
        case "tolerance":
            config["Solver"]["Linear"]["Tol"] = 1e-4
        case "initial_guess":
            config["Solver"]["Linear"]["InitialGuess"] = True
        case "scale":
            config["Model"]["L0"] = 1.0
        case "adaptive":
            config["Model"]["Refinement"]["MaxIts"] = 2
        case "material_attribute":
            config["Domains"]["Materials"][1]["Attributes"] = [3]
    _save(path, config)
    with pytest.raises(ValueError, match=r"Native|Dataset|Xmon|Bare|Frozen|Every|Each"):
        curate_xmon._native_run(**packet)


@pytest.mark.parametrize(
    "damage",
    [
        "C",
        "Cm",
        "Cinv",
        "indicator",
        "order",
        "dof_order",
        "tetrahedra",
        "missing_inverse",
        "completion",
    ],
)
def test_native_results_corruption(packet: dict, damage: str) -> None:
    run = packet["run"]
    match damage:
        case "C" | "Cm" | "Cinv":
            path = run / f"output/palace/terminal-{damage}.csv"
            header = path.read_text(encoding="utf-8").splitlines()[0]
            path.write_text(f"{header}\n1,0.001\n", encoding="utf-8")
        case "indicator":
            (run / "output/palace/error-indicators.csv").write_text(
                "Norm,Minimum,Maximum,Mean\n0.13,0.001,0.02,0.01\n", encoding="utf-8"
            )
        case "order" | "dof_order" | "tetrahedra":
            metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
            if damage == "order":
                metadata["simulation"]["field_order"] = 2
            elif damage == "dof_order":
                metadata["mesh"]["field_dofs"]["field_order"] = 2
            else:
                metadata["mesh"]["tetrahedra"] = 0
            _save(run / "metadata.json", metadata)
        case "missing_inverse":
            (run / "output/palace/terminal-Cinv.csv").unlink()
        case "completion":
            _save(run / "complete.json", {"terminals": ["ground"]})
    with pytest.raises((ValueError, FileNotFoundError)):
        curate_xmon._native_run(**packet)


def test_current_geometry_is_recomputed(
    packet: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def different_xmon(**kwargs: Any) -> gf.Component:
        return xmon_transmon(arm_width=(40, 30, 40, 30), **kwargs)

    monkeypatch.setattr(_xmon_geometry, "xmon_transmon", different_xmon)
    with pytest.raises(ValueError, match="sheets differ"):
        curate_xmon._native_run(**packet)


@pytest.mark.parametrize("control", ["fine-h", "hp4"])
def test_refinement_requires_recorded_increase(packet: dict, control: str) -> None:
    proof = curate_xmon._native_run(**packet)
    provenance = dict.fromkeys(curate_xmon.STABLE, "procedural_fixture")
    candidate: dict[str, Any] = {
        "point": packet["point"],
        "settings": packet["settings"],
        "source": {"provenance": provenance},
        "maxwell_F": [[80e-15]],
        "native": [proof],
    }
    reference: dict[str, Any] = copy.deepcopy(candidate) | {"control": control}
    settings = reference["settings"]
    assert isinstance(settings, dict)
    settings |= {"near_mesh": 0.49} if control == "fine-h" else {"order": 4}
    with pytest.raises(ValueError, match="did not increase"):
        curate_xmon._numerical(candidate, reference)


def _circuit_values(arm_scale: float, gap_width: float) -> dict:
    capacitance = float(f"{(80 + 10 * arm_scale + gap_width / 2) * 1e-15:.15e}")
    return {
        "maxwell_capacitance": jnp.asarray([[capacitance]]),
        "mutual_capacitance": jnp.asarray([[capacitance]]),
        "fem_error_indicator_norm": 0.12,
        "solver_iterations": 20.0,
        "solver_relative_residual": 1e-10,
    }


@pytest.fixture
def complete_study(
    study: Path, table: tuple[pl.DataFrame, DatasetMetadata, dict]
) -> Path:
    """Exercise admission logic with 61 procedural points, never real FEM evidence."""
    source = study / "source"
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "source-sha256.json").read_text(encoding="utf-8"))
    frozen = curate_xmon._source_provenance(source, plan, manifest)
    _, template, _ = table
    for index, job in enumerate(plan["jobs"]):
        folder = study / "results" / job["name"]
        summaries = []
        _save(
            study / f"submission-{index}.json",
            {
                "returncode": 0,
                "stdout": f"{123 + index}\n",
                "name": job["name"],
                "evidence_kind": "procedural_fixture_not_FEM",
            },
        )
        for task in job["tasks"]:
            settings = {"permittivity": 11.45, "tolerance": 1e-9, **task["settings"]}
            provenance = {
                **dict.fromkeys(curate_xmon.STABLE, "procedural_fixture_not_FEM"),
                **frozen,
                "evidence_kind": "procedural_fixture_not_FEM",
                "versions": {"fixture": "procedural_fixture_not_FEM"},
                "processes": 1,
                "settings": settings,
            }
            # The false flag exercises coordinator admission, all evidence stays procedural.
            metadata = template.model_copy(
                update={"synthetic": False, "provenance": provenance}
            )
            grid = task.get("grid") or {
                name: [value] for name, value in task["point"].items()
            }
            frame = sweep(metadata, _circuit_values, grid)
            path = folder / task["label"] / "data/part-0000.parquet"
            path.parent.mkdir(parents=True)
            frame.write_parquet(path, metadata={METADATA_KEY: metadata.to_json()})
            summaries.append(
                task
                | {
                    "status": "ok",
                    "settings": settings,
                    "grid": grid,
                    "points": frame.select(list(grid)).unique().height,
                }
            )
            if task["mode"] == "control":
                summaries[-1]["maxwell_F"] = _circuit_values(**task["point"])[
                    "maxwell_capacitance"
                ].tolist()
            for coords in product(*grid.values()):
                point = dict(zip(grid, coords, strict=True))
                _packet(
                    folder / task["label"] / "runs",
                    point=point,
                    settings=settings,
                    provenance=provenance,
                    driver_sha256=manifest["batch.py"],
                    capacitance=float(
                        _circuit_values(**point)["maxwell_capacitance"][0, 0]
                    ),
                    tetrahedra=2000 if task.get("control") == "fine-h" else 1000,
                )
        _save(folder / "summary.json", summaries)
    return study


def test_complete_procedural_campaign_exports_exact_tables_and_evidence(
    complete_study: Path, tmp_path: Path
) -> None:
    output = tmp_path / "exported-procedural-fixture"
    report = curate_xmon.verify(complete_study, output=output)
    assert report["passed"]
    assert not report["pending"]
    assert not report["failed"]
    assert report["completed_native_points"] == 61
    assert len(report["numerical"]) == 12
    assert len(report["interpolation"]) == len(report["raw_references"]) == 36
    assert all(row["passed"] for row in report["numerical"] + report["interpolation"])
    assert (
        report["coverage"]["interior_cells_tested"]
        == report["coverage"]["total_interior_cells"]
        == 16
    )
    assert len(report["strict_faces"]) == 4
    assert len(report["sources"]) == 37
    native = [proof for source in report["sources"] for proof in source["native"]]
    assert len(native) == 61
    assert all(proof["geometry"]["metal_xor_area_um2"] == 0 for proof in native)
    for comparison in report["numerical"]:
        baseline, refined = comparison["candidate_mesh"], comparison["reference_mesh"]
        if comparison["control"] == "fine-h":
            assert refined["mesh"]["tetrahedra"] > baseline["mesh"]["tetrahedra"]
        elif comparison["control"] == "hp4":
            assert refined["simulation"]["field_order"] == 4
            assert baseline["simulation"]["field_order"] == 3
    original = Dataset(complete_study / "results/production/production/data")
    exported = Dataset(output)
    keys = ["arm_scale", "gap_width", "quantity", "row", "col"]
    assert exported.table.sort(keys).equals(original.table.sort(keys))
    assert exported.metadata.quantities == original.metadata.quantities
    assert exported.metadata.terminals == original.metadata.terminals
    assert exported.metadata.reference_ground == original.metadata.reference_ground
    assert exported.metadata.provenance["evidence_kind"] == "procedural_fixture_not_FEM"
    assert exported.metadata.provenance["validation"] == report
    assert (
        exported.metadata.provenance["admission_status"] == report["admission_status"]
    )
    for key, value in original.metadata.provenance.items():
        assert exported.metadata.provenance[key] == value
    for axis in exported.metadata.axes:
        assert axis.validated == (
            min(report["plan"]["grids"]["xmon"][axis.name]),
            max(report["plan"]["grids"]["xmon"][axis.name]),
        )
    assert report["source_manifest_sha256"] == curate_xmon._sha256(
        complete_study / "source/source-sha256.json"
    )
    for source in report["sources"]:
        assert source["receipt"]["evidence_kind"] == "procedural_fixture_not_FEM"
        for proof in source["native"]:
            assert (
                proof["inputs"]["provenance"]["evidence_kind"]
                == "procedural_fixture_not_FEM"
            )
    json.dumps(report, allow_nan=False)


def _rewrite_reference(
    study: Path,
    job: str,
    label: str,
    *,
    factor: float = 1.0,
    provenance_change: dict | None = None,
) -> None:
    """Keep procedural raw and tabular evidence consistent to test coordinator gates."""
    folder = study / "results" / job
    task_folder = folder / label
    data = Dataset(task_folder / "data")
    metadata = data.metadata.model_copy(
        update={"provenance": data.metadata.provenance | (provenance_change or {})}
    )
    capacitance = float(
        f"{data.table.filter(pl.col('quantity') == 'maxwell_capacitance')['value'][0] * factor:.15e}"
    )
    frame = data.table.with_columns(
        pl
        .when(pl.col("quantity").is_in(["maxwell_capacitance", "mutual_capacitance"]))
        .then(capacitance)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    frame.write_parquet(
        task_folder / "data/part-0000.parquet",
        metadata={METADATA_KEY: metadata.to_json()},
    )
    old_run = next((task_folder / "runs").iterdir())
    inputs = json.loads((old_run / "inputs.json").read_text(encoding="utf-8"))
    native = json.loads((old_run / "metadata.json").read_text(encoding="utf-8"))
    shutil.rmtree(old_run)
    _packet(
        task_folder / "runs",
        point=inputs["point"],
        settings=metadata.provenance["settings"],
        provenance=metadata.provenance,
        driver_sha256=inputs["mesh_inputs"]["driver_sha256"],
        capacitance=capacitance,
        tetrahedra=native["mesh"]["tetrahedra"],
    )
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for row in summary:
        if row["label"] == label:
            row["maxwell_F"] = [[capacitance]]
    _save(folder / "summary.json", summary)


@pytest.mark.parametrize(
    ("job", "label"), [("held", "held-0"), ("numerical", "nominal-hp4")]
)
def test_complete_coordinator_rejects_consistent_but_inaccurate_references(
    complete_study: Path, tmp_path: Path, job: str, label: str
) -> None:
    _rewrite_reference(complete_study, job, label, factor=1.25)
    report = curate_xmon.verify(complete_study)
    assert report["completed_native_points"] == 61
    assert not report["failed"]
    assert not report["pending"]
    assert report["coverage"]["passed"]
    assert not report["passed"]
    assert any(
        row["label"] == label and not row["passed"] for row in report["interpolation"]
    )
    if job == "numerical":
        assert any(not row["passed"] for row in report["numerical"])
    else:
        assert all(row["passed"] for row in report["numerical"])
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="incomplete or failed"):
        curate_xmon.verify(complete_study, output=output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("job", "label"), [("held", "held-19"), ("numerical", "nominal-hp4")]
)
def test_complete_coordinator_rejects_reference_physics_drift(
    complete_study: Path, job: str, label: str
) -> None:
    _rewrite_reference(
        complete_study,
        job,
        label,
        provenance_change={"ground": "different procedural ground"},
    )
    report = curate_xmon.verify(complete_study)
    assert report["completed_native_points"] == 61
    assert not report["passed"]
    assert not report["pending"]
    assert any("physics" in row["error"] for row in report["failed"])
    assert all(row["label"] != label for row in report["interpolation"])


def test_complete_coordinator_requires_actual_mesh_growth(complete_study: Path) -> None:
    run = next((complete_study / "results/numerical/nominal-fine-h/runs").iterdir())
    native = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    native["mesh"]["tetrahedra"] = 1000
    _save(run / "metadata.json", native)
    report = curate_xmon.verify(complete_study)
    assert report["completed_native_points"] == 61
    assert not report["passed"]
    assert len(report["numerical"]) == 11
    assert any(
        "tetrahedra did not increase" in row["error"] for row in report["failed"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "late-task",
        "missing-parquet",
        "failed-without-parquet",
        "failed-with-parquet",
        "late-run",
        "late-quantity",
        "nonfinite-quantity",
        "duplicate-run",
        "duplicate-summary",
        "unknown-summary",
        "receipt",
        "summary-grid",
        "summary-settings",
        "summary-matrix",
        "source-provenance",
    ],
)
def test_complete_coordinator_retains_late_and_duplicate_failures(
    complete_study: Path, tmp_path: Path, damage: str
) -> None:
    production = complete_study / "results/production/production"
    held_summary = complete_study / "results/held/summary.json"
    production_summary = complete_study / "results/production/summary.json"
    held = json.loads(held_summary.read_text(encoding="utf-8"))
    part = production / "data/part-0000.parquet"
    match damage:
        case "late-task":
            _save(held_summary, held[:-1])
        case "missing-parquet" | "failed-without-parquet" | "failed-with-parquet":
            if damage != "failed-with-parquet":
                (
                    complete_study / "results/held/held-19/data/part-0000.parquet"
                ).unlink()
            if damage != "missing-parquet":
                held[-1] |= {"status": "failed", "error": "Procedural solver failure"}
                _save(held_summary, held)
        case "late-run" | "duplicate-run":
            run = max((production / "runs").iterdir())
            if damage == "late-run":
                (run / "inputs.json").unlink()
            else:
                shutil.copytree(run, production / "runs/duplicate")
        case "late-quantity" | "nonfinite-quantity":
            data = Dataset(production / "data")
            last = (
                (pl.col("arm_scale").is_in([1.2]))
                & (pl.col("gap_width").is_in([12.0]))
                & (pl.col("quantity") == "solver_relative_residual")
            )
            frame = (
                data.table.filter(~last)
                if damage == "late-quantity"
                else data.table.with_columns(
                    pl
                    .when(last)
                    .then(float("nan"))
                    .otherwise(pl.col("value"))
                    .alias("value")
                )
            )
            frame.write_parquet(part, metadata={METADATA_KEY: data.metadata.to_json()})
        case "duplicate-summary" | "unknown-summary" | "summary-grid":
            rows = json.loads(production_summary.read_text(encoding="utf-8"))
            if damage == "duplicate-summary":
                rows.append(copy.deepcopy(rows[0]))
            elif damage == "unknown-summary":
                rows.append(rows[0] | {"label": "unplanned"})
            else:
                rows[0]["grid"]["arm_scale"].pop()
            _save(production_summary, rows)
        case "summary-settings" | "summary-matrix":
            if damage == "summary-settings":
                held[-1]["settings"]["tolerance"] = 1e-6
            else:
                held[-1]["maxwell_F"] = [[1e-12]]
            _save(held_summary, held)
        case "receipt":
            receipt = json.loads(
                (complete_study / "submission-2.json").read_text(encoding="utf-8")
            )
            _save(complete_study / "submission-2.json", receipt | {"returncode": 1})
        case "source-provenance":
            data = Dataset(production / "data")
            metadata = data.metadata.model_copy(
                update={
                    "provenance": data.metadata.provenance
                    | {"helper_sha256": "different helper"}
                }
            )
            data.table.write_parquet(part, metadata={METADATA_KEY: metadata.to_json()})
    report = curate_xmon.verify(complete_study)
    assert not report["passed"]
    assert report["pending"] or report["failed"]
    if damage in {"late-task", "missing-parquet"}:
        assert report["pending"]
        assert not report["failed"]
        assert report["completed_native_points"] == 60
    if damage in {"late-quantity", "nonfinite-quantity", "source-provenance"}:
        rejected = next(
            row for row in report["failed"] if row.get("job") == "production"
        )
        assert rejected["raw_native_rows"]
    if damage in {"duplicate-summary", "unknown-summary"}:
        assert any("Duplicate or unknown" in row["error"] for row in report["failed"])
    json.dumps(report, allow_nan=False)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="incomplete or failed"):
        curate_xmon.verify(complete_study, output=output)
    assert not output.exists()


def test_procedural_cli_writes_export_and_portable_report(
    complete_study: Path, tmp_path: Path
) -> None:
    output, report_path = tmp_path / "export", tmp_path / "report.json"
    curate_xmon.main(complete_study, output=output, report=report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["completed_native_points"] == 61
    assert Dataset(output).metadata.provenance["validation"] == report


@pytest.mark.parametrize(
    "damage", ["wheel", "recipe", "source-proof", "reference-grid"]
)
def test_coordinator_rejects_inconsistent_frozen_declarations(
    study: Path, damage: str
) -> None:
    source = study / "source"
    plan_path, manifest_path = source / "plan.json", source / "source-sha256.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    match damage:
        case "wheel":
            plan["gsim_wheel_sha256"] = "different wheel"
        case "recipe":
            plan["recipe_sha256"] = "different recipe"
        case "source-proof":
            plan["gsim_source_verification"]["all_package_files_match_source"] = False
        case "reference-grid":
            plan["jobs"][2]["tasks"][-1]["grid"] = {
                "arm_scale": [0.8, 1.2],
                "gap_width": [12.0],
            }
    _save(plan_path, plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _save(manifest_path, manifest | {"plan.json": curate_xmon._sha256(plan_path)})
    with pytest.raises(ValueError, match=r"Frozen Xmon|Each Xmon"):
        curate_xmon.verify(study)


@pytest.mark.parametrize(
    ("name", "wrong_unit"), [("C", "pF"), ("Cm", "1"), ("Cinv", "1/pF")]
)
def test_native_matrix_unit_labels_are_checked_without_rescaling(
    packet: dict, name: str, wrong_unit: str
) -> None:
    path = packet["run"] / f"output/palace/terminal-{name}.csv"
    original_header, numeric_rows = path.read_text(encoding="utf-8").split("\n", 1)
    unit = "1/F" if name == "Cinv" else "F"
    assert f"({unit})" in original_header
    altered_header = original_header.replace(f"({unit})", f"({wrong_unit})")
    path.write_text(f"{altered_header}\n{numeric_rows}", encoding="utf-8")
    assert path.read_text(encoding="utf-8").split("\n", 1)[1] == numeric_rows
    with pytest.raises(ValueError, match=f"Native {name} matrix units differ"):
        curate_xmon._native_run(**packet)
