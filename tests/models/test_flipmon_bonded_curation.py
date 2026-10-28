"""Portable procedural tables and plans test rejection, never native FEM admission."""

import copy
import hashlib
import json
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import polars as pl
import pytest

from qpdk.models.datasets import Axis, DatasetMetadata, Quantity, sweep
from qpdk.models.datasets.capacitance import maxwell_to_mutual
from qpdk.models.datasets.metadata import METADATA_KEY
from qpdk.models.datasets.recipes import curate_flipmon_bonded as curator
from qpdk.models.datasets.table import to_grid

OLD = {
    "chip_gap": [5.0, 10.0, 15.0],
    "alignment": [-15.0, 0.0, 15.0],
    "inner_radius": [50.0, 55.0, 60.0],
    "top_radius": [100.0, 110.0, 120.0],
}
FINAL = OLD | {"chip_gap": [5.0, 6.25, 7.5, 8.75, 10.0, 12.5, 15.0]}
PROFILE = {"near_mesh": 1.4, "far_mesh": 60.0, "domain_pad": 600.0, "order": 3}


def _task(label: str, point: dict, control: str, anchor: str) -> dict:
    change = {
        "baseline": {},
        "fine-h": {"near_mesh": 0.98, "far_mesh": 48.0},
        "hp4": {"order": 4},
        "domain": {"domain_pad": 900.0},
        "ideal-wire": {},
        "ideal-wire-fine-h": {"near_mesh": 0.98, "far_mesh": 48.0},
        "held-out": {"near_mesh": 0.98, "far_mesh": 48.0},
    }[control]
    return {
        "recipe": curator.RECIPE,
        "label": label,
        "mode": "control",
        "point": point,
        "anchor": anchor,
        "control": control,
        "bond": "ideal_wire" if control.startswith("ideal-wire") else "physical_bump",
        "settings": PROFILE | change,
    }


def _held(grid: dict, *, old: bool = False) -> list[dict]:
    centres = [
        dict(zip(grid, values, strict=True))
        for values in product(*[
            [(a + b) / 2 for a, b in pairwise(coords)] for coords in grid.values()
        ])
    ]
    inside = (
        {"chip_gap": 10.0, "alignment": 0.0, "inner_radius": 55.0, "top_radius": 110.0}
        if old
        else {
            "chip_gap": 8.125,
            "alignment": -7.5,
            "inner_radius": 52.5,
            "top_radius": 105.0,
        }
    )
    faces = [
        inside | {name: value}
        for name, coords in grid.items()
        for value in (coords[0], coords[-1])
    ]
    return [
        _task(f"held-{i}", point, "held-out", f"dense-held-{i}")
        for i, point in enumerate(centres + faces)
    ]


def _plans(study: Path, calibration: Path) -> dict[Path, dict]:
    common = {
        "source_revision": "procedural-geometry",
        "gsim_revision": "procedural-solver",
        "scope": "Procedural fixture, not FEM evidence",
    }
    old_jobs, entries = [], []
    for i, gap in enumerate(OLD["chip_gap"]):
        grid = OLD | {"chip_gap": [gap]}
        old_jobs.append({
            "name": f"old-production-{i}",
            "tasks": [
                {
                    "recipe": curator.RECIPE,
                    "label": "production",
                    "mode": "production",
                    "grid": grid,
                    "bond": "physical_bump",
                    "settings": PROFILE,
                }
            ],
            "expected_points": 27,
        })
        entries.append({
            "campaign": calibration.name,
            "job": old_jobs[-1]["name"],
            "label": "production",
            "grid": grid,
        })
    anchors = [
        {"chip_gap": 10.0, "alignment": 0.0, "inner_radius": 60.0, "top_radius": 110.0},
        {"chip_gap": 5.0, "alignment": 15.0, "inner_radius": 60.0, "top_radius": 120.0},
        {
            "chip_gap": 5.0,
            "alignment": -15.0,
            "inner_radius": 50.0,
            "top_radius": 100.0,
        },
        {
            "chip_gap": 15.0,
            "alignment": 15.0,
            "inner_radius": 60.0,
            "top_radius": 120.0,
        },
    ]
    controls = [
        _task(f"anchor-{i}-{control}", p, control, f"anchor-{i}")
        for i, p in enumerate(anchors)
        for control in curator.CONTROLS
    ]
    old_jobs += [
        {"name": "calibration", "tasks": controls, "expected_points": 24},
        {"name": "old-held", "tasks": _held(OLD, old=True), "expected_points": 24},
    ]
    jobs = [
        {
            "name": f"new-production-{i}",
            "tasks": [
                {
                    "recipe": curator.RECIPE,
                    "label": "production",
                    "mode": "production",
                    "grid": FINAL | {"chip_gap": [gap]},
                    "bond": "physical_bump",
                    "settings": PROFILE,
                }
            ],
            "expected_points": 27,
        }
        for i, gap in enumerate((6.25, 7.5, 8.75, 12.5))
    ]
    jobs.append({"name": "fresh-held", "tasks": _held(FINAL), "expected_points": 56})
    return {
        calibration: common | {"jobs": old_jobs, "grids": {curator.RECIPE: OLD}},
        study: common
        | {
            "jobs": jobs,
            "grids": {curator.RECIPE: FINAL},
            "reused_production": entries,
            "calibration_campaign": calibration.name,
            "retuning_references": calibration.name,
            "new_production_geometries": 108,
            "fresh_held_out_points": 56,
            "datasets": {"flipmon_bonded_palace": 189},
        },
    }


def _freeze(campaign: Path, plan: dict) -> None:
    source = campaign / "source"
    source.mkdir(parents=True)
    (source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (source / "source-sha256.json").write_text(
        json.dumps({
            "plan.json": hashlib.sha256((source / "plan.json").read_bytes()).hexdigest()
        }),
        encoding="utf-8",
    )


@pytest.fixture
def studies(tmp_path: Path) -> tuple[Path, Path]:
    study, calibration = tmp_path / "final", tmp_path / "original"
    for path, plan in _plans(study, calibration).items():
        _freeze(path, plan)
    return study, calibration


def test_overlapping_calibration_and_reuse_roles(studies: tuple[Path, Path]) -> None:
    study, calibration = studies
    selected, expected = curator._selection(
        study, calibration, _plans(study, calibration)
    )
    assert len(selected) == 111
    assert expected["native_points"] == 293
    assert expected["production_points"] == 189
    assert expected["production_shards"] == 7
    assert expected["retuning_at_training_nodes"] == 8
    for role, count in (
        ("calibration", 24),
        ("fresh", 56),
        ("retuning", 24),
        ("production", 7),
    ):
        assert sum(entry[-1] == role for entry in selected) == count
    assert (
        len({(entry[0], entry[2]["name"], entry[3]["label"]) for entry in selected})
        == 111
    )


def test_pending_preview_and_output_refusal(
    studies: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    study, calibration = studies
    writes = []
    monkeypatch.setattr(curator, "write", lambda *args: writes.append(args))
    report = curator.verify(study, calibration)
    assert not report["passed"]
    assert not report["numerical_passed"]
    assert len(report["pending"]) == 111
    assert not report["failed"]
    assert report["geometry_parity"]["verified_points"] == 0
    assert (
        report["admission_status"]
        == "candidate_pending_independent_numerical_validation"
    )
    assert "scientific_admission" not in report
    with pytest.raises(ValueError, match="export blocked"):
        curator.verify(study, calibration, output=study / "must-not-export")
    assert not writes
    assert not (study / "must-not-export").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "grid",
        "missing_reuse",
        "overlap",
        "wrong_reuse_grid",
        "missing_fresh",
        "duplicate_fresh",
        "fresh_training",
        "fresh_old",
        "face_corner",
        "old_retune_missing",
        "control_missing",
        "control_bond",
        "calibration_identity",
        "job_count",
        "bad_recipe",
    ],
)
def test_invalid_plan_selection(studies: tuple[Path, Path], damage: str) -> None:
    study, calibration = studies
    plans = copy.deepcopy(_plans(study, calibration))
    final, old = plans[study], plans[calibration]
    match damage:
        case "grid":
            final["grids"][curator.RECIPE]["chip_gap"] = [5.0, 10.0, 15.0]
        case "missing_reuse":
            final["reused_production"].pop()
        case "overlap":
            final["jobs"][1]["tasks"][0]["grid"] = final["jobs"][0]["tasks"][0]["grid"]
        case "wrong_reuse_grid":
            final["reused_production"][0]["grid"] = FINAL
        case "missing_fresh":
            final["jobs"][-1]["tasks"].pop()
            final["jobs"][-1]["expected_points"] -= 1
        case "duplicate_fresh":
            final["jobs"][-1]["tasks"][-1]["point"] = final["jobs"][-1]["tasks"][0][
                "point"
            ]
        case "fresh_training":
            final["jobs"][-1]["tasks"][0]["point"] = dict(
                zip(FINAL, (5.0, 0.0, 55.0, 110.0), strict=True)
            )
        case "fresh_old":
            final["jobs"][-1]["tasks"][0]["point"] = old["jobs"][-1]["tasks"][0][
                "point"
            ]
        case "face_corner":
            final["jobs"][-1]["tasks"][-1]["point"] = {
                a: v[0] for a, v in FINAL.items()
            }
        case "old_retune_missing":
            old["jobs"][-1]["tasks"].pop()
            old["jobs"][-1]["expected_points"] -= 1
        case "control_missing":
            old["jobs"][-2]["tasks"].pop()
            old["jobs"][-2]["expected_points"] -= 1
        case "control_bond":
            old["jobs"][-2]["tasks"][0]["bond"] = "ideal_wire"
        case "calibration_identity":
            final["calibration_campaign"] = "unrelated"
        case "job_count":
            final["jobs"][0]["expected_points"] -= 1
        case "bad_recipe":
            final["jobs"][0]["tasks"][0]["recipe"] = "flipmon_local_readout"
    with pytest.raises(
        ValueError,
        match=r"grid|shard|requires|points|calibration|Bare|Final|Frozen|Fresh|reference|Reference|Each|Unexpected",
    ):
        curator._selection(study, calibration, plans)


def test_sources_and_manifest_paths_are_checked(studies: tuple[Path, Path]) -> None:
    study, calibration = studies
    (study / "source/plan.json").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source changed"):
        curator.verify(study, calibration)
    (study / "source/source-sha256.json").write_text(
        json.dumps({"../outside": "bad", "plan.json": "bad"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stay inside"):
        curator.verify(study, calibration)


@pytest.fixture
def table() -> tuple[pl.DataFrame, DatasetMetadata, dict]:
    metadata = DatasetMetadata(
        name="synthetic_bare_flipmon",
        description="Procedural table, not FEM evidence",
        synthetic=True,
        axes=tuple(Axis(name=a, unit="um") for a in curator.AXES),
        variants=("bond",),
        terminals=curator.TERMINALS,
        reference_ground="both_chip_ground_frames",
        quantities=tuple(
            Quantity(name=n, kind=k, unit=u) for n, (k, u) in curator.QUANTITIES.items()
        ),
    )
    grid = {a: v[:2] for a, v in OLD.items()}

    def solve(**_point: float | str) -> dict:
        matrix = jnp.asarray([[120.0, -40.0], [-40.0, 160.0]]) * 1e-15
        return {
            "maxwell_capacitance": matrix,
            "mutual_capacitance": maxwell_to_mutual(matrix),
            "solver_relative_residual": 1e-10,
            "solver_iterations": 20.0,
            "fem_error_indicator_norm": 0.12,
        }

    return sweep(metadata, solve, grid, {"bond": ["physical_bump"]}), metadata, grid


def _validate(table: tuple[pl.DataFrame, DatasetMetadata, dict]) -> dict:
    frame, metadata, grid = table
    return curator._validate_table(
        frame, metadata, grid, bond="physical_bump", tolerance=1e-9
    )


def test_synthetic_table_guard_is_not_admission(
    table: tuple[pl.DataFrame, DatasetMetadata, dict],
) -> None:
    raw = _validate(table)
    assert table[1].synthetic
    assert "passed" not in raw
    assert set(raw) == set(curator.QUANTITIES)
    assert jnp.asarray(raw["maxwell_capacitance"]).shape == (2, 2, 2, 2, 2, 2)


@pytest.mark.parametrize("quantity", curator.QUANTITIES)
@pytest.mark.parametrize("axis", curator.AXES)
def test_each_quantity_covers_its_full_task_grid(
    table: tuple[pl.DataFrame, DatasetMetadata, dict], quantity: str, axis: str
) -> None:
    frame, metadata, grid = table
    broken = frame.filter(
        ~((pl.col("quantity") == quantity) & (pl.col(axis) == grid[axis][-1]))
    )
    to_grid(broken, metadata, quantity, bond="physical_bump", allow_missing=False)
    with pytest.raises(ValueError, match=r"quantity.*task grid"):
        _validate((broken, metadata, grid))


@pytest.mark.parametrize(
    "damage",
    [
        "mutual",
        "indicator",
        "negative_iterations",
        "fractional_iterations",
        "residual",
        "nan",
        "missing_quantity",
        "wrong_variant",
        "missing_cell",
    ],
)
def test_native_quantity_corruption(
    table: tuple[pl.DataFrame, DatasetMetadata, dict], damage: str
) -> None:
    frame, metadata, grid = table
    quantity = pl.col("quantity")
    match damage:
        case "missing_quantity":
            frame = frame.filter(quantity != "solver_iterations")
        case "wrong_variant":
            frame = frame.with_columns(pl.lit("ideal_wire").alias("bond"))
        case "missing_cell":
            frame = frame.filter(
                ~(
                    (quantity == "solver_iterations")
                    & pl.all_horizontal([
                        pl.col(a) == values[0] for a, values in grid.items()
                    ])
                )
            )
        case _:
            name, value = {
                "mutual": ("mutual_capacitance", 1e-9),
                "indicator": ("fem_error_indicator_norm", -1.0),
                "negative_iterations": ("solver_iterations", -1.0),
                "fractional_iterations": ("solver_iterations", 1.5),
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
    with pytest.raises(ValueError, match=r"Native|Dataset|finite|negative|task grid"):
        _validate((frame, metadata, grid))


def test_rejected_synthetic_reference_retains_raw_rows(
    studies: tuple[Path, Path], table: tuple[pl.DataFrame, DatasetMetadata, dict]
) -> None:
    study, calibration = studies
    plan = _plans(study, calibration)[calibration]
    job = plan["jobs"][-2]
    task = job["tasks"][0]
    frame, metadata, _ = table
    path = (
        calibration / "results" / job["name"] / task["label"] / "data/part-0000.parquet"
    )
    path.parent.mkdir(parents=True)
    frame.write_parquet(path, metadata={METADATA_KEY: metadata.to_json()})
    summary = path.parents[2] / "summary.json"
    summary.write_text(json.dumps([task | {"status": "ok"}]), encoding="utf-8")
    (calibration / "submission-3.json").write_text(
        json.dumps({"name": job["name"], "returncode": 0, "stdout": "123\n"}),
        encoding="utf-8",
    )
    report = curator.verify(study, calibration)
    assert not report["passed"]
    assert len(report["failed"]) == 1
    assert "Synthetic" in report["failed"][0]["error"]
    assert report["failed"][0]["raw_native_rows"] == frame.to_dicts()


@pytest.mark.parametrize("control", ["fine-h", "hp4"])
def test_actual_refinement_is_required(control: str) -> None:
    settings = PROFILE | {"permittivity": 11.45, "tolerance": 1e-9}
    baseline: dict[str, Any] = {
        "point": {a: v[0] for a, v in OLD.items()},
        "settings": settings,
        "source": {"provenance": dict.fromkeys(curator.STABLE, "procedural_fixture")},
        "maxwell_F": [[120e-15, -40e-15], [-40e-15, 160e-15]],
        "mesh_evidence": {"nodes": 100, "tetrahedra": 150, "field_order": 3},
    }
    reference: dict[str, Any] = copy.deepcopy(baseline)
    changed = reference["settings"]
    assert isinstance(changed, dict)
    changed.update(near_mesh=0.98) if control == "fine-h" else changed.update(order=4)
    with pytest.raises(ValueError, match="refinement"):
        curator._compare(baseline, reference, control)


@pytest.mark.parametrize("old_reference_fails", [False, True])
def test_old_physical_reference_keeps_accuracy_veto(
    studies: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    old_reference_fails: bool,
) -> None:
    """Simulate native acceptance to isolate the complete accuracy gate, not FEM validity."""
    study, calibration = studies
    plans = _plans(study, calibration)
    selected, _ = curator._selection(study, calibration, plans)
    for campaign, _index, job, _task_row, _role in selected:
        summaries = []
        for task in job["tasks"]:
            settings = {"permittivity": 11.45, "tolerance": 1e-9, **task["settings"]}
            summaries.append(task | {"settings": settings, "status": "ok"})
        folder = campaign / "results" / job["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")

    def accepted_native_task(
        campaign: Path,
        _index: int,
        job: dict,
        task: dict,
        _frozen: dict,
        evidence: dict,
    ) -> curator.Dataset:
        """Supply procedural Parquet and an explicitly simulated native proof boundary."""
        role = evidence["role"]
        scale = 0.5 if task["bond"] == "ideal_wire" else 1.0
        if (
            old_reference_fails
            and role == "retuning"
            and task["anchor"] == "dense-held-0"
        ):
            scale = 1.2
        matrix = jnp.asarray([[120.0, -40.0], [-40.0, 160.0]]) * (scale * 1e-15)
        settings = {"permittivity": 11.45, "tolerance": 1e-9, **task["settings"]}
        provenance = dict.fromkeys(
            curator.STABLE, "procedural_accuracy_gate_not_FEM"
        ) | {
            "settings": settings,
            "bond": "procedural_bond_conventions",
            "fixture_kind": "simulated_native_acceptance_not_FEM",
        }
        metadata = DatasetMetadata(
            name="procedural_accuracy_gate",
            description="Procedural gate test, not FEM evidence",
            axes=tuple(Axis(name=a, unit="um") for a in curator.AXES),
            variants=("bond",),
            terminals=curator.TERMINALS,
            reference_ground="both_chip_ground_frames",
            provenance=provenance,
            quantities=tuple(
                Quantity(name=n, kind=k, unit=u)
                for n, (k, u) in curator.QUANTITIES.items()
            ),
        )
        grid = curator._grid(task)

        def solve(**_point: float | str) -> dict:
            return {
                "maxwell_capacitance": matrix,
                "mutual_capacitance": maxwell_to_mutual(matrix),
                "solver_relative_residual": 1e-10,
                "solver_iterations": 20.0,
                "fem_error_indicator_norm": 0.12,
            }

        frame = sweep(metadata, solve, grid, {"bond": [task["bond"]]})
        folder = campaign / "results" / job["name"] / task["label"] / "data"
        folder.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(
            folder / "part-0000.parquet", metadata={METADATA_KEY: metadata.to_json()}
        )
        nodes = 200 if settings["near_mesh"] < 1.4 else 100
        proof = {
            "mesh": {"nodes": nodes, "tetrahedra": nodes * 2},
            "field_order": settings["order"],
        }
        evidence |= {
            "receipt": {"stdout": "123\n"},
            "parts": {"fixture": "procedural"},
            "geometry": {
                "passed": True,
                "scope": "Simulated native boundary, not FEM evidence",
                "points": [copy.deepcopy(proof) for _ in product(*grid.values())],
            },
        }
        evidence["raw_summary"]["maxwell_F"] = matrix.tolist()
        return curator.Dataset(folder)

    monkeypatch.setattr(curator, "_load_task", accepted_native_task)
    writes = []
    monkeypatch.setattr(curator, "write", lambda *args: writes.append(args))
    report = curator.verify(study, calibration)
    assert not report["pending"]
    assert not report["failed"]
    assert report["geometry_parity"]["passed"]
    assert report["geometry_parity"]["verified_points"] == 293
    assert report["held_out_coverage"]["passed"]
    assert report["held_out_coverage"]["interior_cells_tested"] == 48
    assert len(report["held_out_coverage"]["boundaries_tested"]) == 8
    assert all(
        row["passed"]
        for key in ("numerical", "ideal_wire_refinement", "interpolation")
        for row in report[key]
    )
    assert len(report["interpolation"]) == 72
    assert all(
        not row["passed"] and not row["used_for_admission"]
        for row in report["finite_bump_difference"]
    )
    old = report["retuning_diagnostics"]
    assert len(old) == 24
    assert sum(not row["passed"] for row in old) == int(old_reference_fails)
    assert all(
        row["used_for_accuracy_gate"] and not row["used_for_fresh_coverage"]
        for row in old
    )
    assert report["numerical_passed"] is not old_reference_fails
    assert report["passed"] is not old_reference_fails
    if old_reference_fails:
        output = study / "must-not-export"
        with pytest.raises(ValueError, match="export blocked"):
            curator.verify(study, calibration, output=output)
        assert not output.exists()
    assert not writes
