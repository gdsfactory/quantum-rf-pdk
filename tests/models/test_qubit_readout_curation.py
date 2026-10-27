"""Incomplete or altered FEM studies cannot publish a curated dataset."""

import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import polars as pl
import pytest

from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    sweep,
)
from qpdk.models.datasets.generate import write


@pytest.fixture
def incomplete_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path, Path]:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"
    monkeypatch.syspath_prepend(str(scripts))
    curator = importlib.import_module("curate_qubit_readout")
    study, calibration = tmp_path / "study", tmp_path / "calibration"
    task = {"recipe": "double_pad_readout", "label": "production", "mode": "production"}
    for path in (study, calibration):
        source = path / "source"
        source.mkdir(parents=True)
        plan = {
            "grids": {"double_pad_readout": {"pad_width": [250.0]}},
            "jobs": [{"name": "job", "tasks": [task]}],
        }
        plan_file = source / "plan.json"
        plan_file.write_text(json.dumps(plan))
        (source / "source-sha256.json").write_text(
            json.dumps({
                "plan.json": hashlib.sha256(plan_file.read_bytes()).hexdigest()
            })
        )
    return curator, study, calibration


def test_preview_and_rejected_publication(
    incomplete_study: tuple[Any, Path, Path], tmp_path: Path
) -> None:
    curator, study, calibration = incomplete_study
    report = curator.verify(study, calibration, "double_pad_readout")
    assert not report["passed"]
    assert len(report["pending"]) == 1
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="incomplete"):
        curator.verify(study, calibration, "double_pad_readout", output=output)
    assert not output.exists()


def test_altered_inputs_are_rejected(incomplete_study: tuple[Any, Path, Path]) -> None:
    curator, study, calibration = incomplete_study
    source = study / "source/plan.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        curator.verify(study, calibration, "double_pad_readout")


def test_distinct_cells_and_boundaries(
    incomplete_study: tuple[Any, Path, Path],
) -> None:
    curator, _, _ = incomplete_study
    grid = {"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]}
    points = [{"a": a, "b": b} for a in (0.5, 1.5) for b in (0.5, 1.5)]
    points += [{"a": a, "b": 0.5} for a in (0.0, 2.0)]
    points += [{"a": 0.5, "b": b} for b in (0.0, 2.0)]
    rows = [{"anchor": f"dense-held-{i}", "point": p} for i, p in enumerate(points)]
    assert curator._coverage(grid, rows)["passed"]
    assert not curator._coverage(grid, rows[:-1])["passed"]
    duplicate = curator._coverage(grid, rows[:1] * 8)
    assert duplicate["interior_cells_tested"] == 1
    assert not duplicate["passed"]


def test_all_interpolation_cells_can_be_required(
    incomplete_study: tuple[Any, Path, Path],
) -> None:
    curator, _, _ = incomplete_study
    grid = {"a": list(map(float, range(10))), "b": list(map(float, range(10)))}
    diagonal = [(i + 0.5, i + 0.5) for i in range(9)]
    remaining = [(a + 0.5, b + 0.5) for a in range(9) for b in range(9) if a != b]
    interior = diagonal + remaining[:55]
    boundaries = [(0.0, 0.5), (9.0, 0.5), (0.5, 0.0), (0.5, 9.0)]
    rows = [
        {"anchor": f"dense-held-{i}", "point": {"a": a, "b": b}}
        for i, (a, b) in enumerate(interior + boundaries)
    ]
    assert curator._coverage(grid, rows)["passed"]
    assert not curator._coverage(grid, rows, require_all=True)["passed"]
    rows += [
        {"anchor": f"dense-held-extra-{i}", "point": {"a": a, "b": b}}
        for i, (a, b) in enumerate(remaining[55:])
    ]
    assert curator._coverage(grid, rows, require_all=True)["passed"]


@pytest.fixture
def complete_study(incomplete_study: tuple[Any, Path, Path]) -> tuple[Any, Path, Path]:
    curator, study, calibration = incomplete_study
    settings = {
        "near_mesh": 1.0,
        "far_mesh": 10.0,
        "domain_pad": 100.0,
        "order": 3,
        "tolerance": 1e-9,
    }
    recipe = "xmon_readout"
    grid = {"size": [0.0, 1.0]}
    provenance = dict.fromkeys(
        (
            "runtime_sha256",
            "versions",
            "gsim_revision",
            "qpdk_geometry_revision",
            "reference_planes",
            "junction",
            "scope",
        ),
        "analytical_test_fixture",
    )
    provenance["solver"] = "analytical_test_fixture"
    metadata = DatasetMetadata(
        name="curation_test_fixture",
        description="Analytical circuit fixture, not FEM evidence.",
        axes=(Axis(name="size", unit="1"),),
        terminals=("island", "readout"),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="solver_relative_residual",
                kind=QuantityKind.CIRCUIT_PARAMETER,
                unit="1",
            ),
        ),
    )

    def solve(size: float) -> dict:
        return {
            "maxwell_capacitance": jnp.asarray([
                [80.0 + size, -2.0],
                [-2.0, 10.0 + size],
            ])
            * 1e-15,
            "solver_relative_residual": 1e-12,
        }

    calibration_tasks: list[dict[str, Any]] = []
    for i, size in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        for control, change in (
            ("candidate", {}),
            ("fine-h", {"near_mesh": 0.7, "far_mesh": 8.0}),
            ("hp4", {"order": 4}),
            ("domain", {"domain_pad": 150.0}),
        ):
            calibration_tasks.append({
                "recipe": recipe,
                "mode": "control",
                "label": f"anchor-{i}-{control}",
                "anchor": f"anchor-{i}",
                "control": control,
                "point": {"size": size},
                "settings": settings | change,
            })
    study_tasks: list[dict[str, Any]] = [
        {
            "recipe": recipe,
            "mode": "production",
            "label": "production",
            "settings": settings,
        }
    ]
    study_tasks += [
        {
            "recipe": recipe,
            "mode": "control",
            "label": f"held-{i}",
            "anchor": f"dense-held-{i}",
            "control": "fine-h",
            "point": {"size": size},
            "settings": settings | {"near_mesh": 0.7, "far_mesh": 8.0},
        }
        for i, size in enumerate((0.0, 0.5, 1.0))
    ]
    for campaign, tasks in ((study, study_tasks), (calibration, calibration_tasks)):
        source = campaign / "source"
        plan = {"grids": {recipe: grid}, "jobs": [{"name": "job", "tasks": tasks}]}
        plan_file = source / "plan.json"
        plan_file.write_text(json.dumps(plan))
        (source / "source-sha256.json").write_text(
            json.dumps({
                "plan.json": hashlib.sha256(plan_file.read_bytes()).hexdigest()
            })
        )
        (campaign / "submission-0.json").write_text(json.dumps({"stdout": "123\n"}))
        folder = campaign / "results/job"
        rows = []
        for task in tasks:
            selected_grid = (
                grid
                if task["mode"] == "production"
                else {"size": [task["point"]["size"]]}
            )
            # Production flags are exercised only in pytest's temporary directory.
            meta = metadata.model_copy(
                update={"provenance": provenance | {"settings": task["settings"]}}
            )
            write(
                folder / task["label"] / "data", meta, sweep(meta, solve, selected_grid)
            )
            row = task | {"status": "ok"}
            if task["mode"] == "control":
                row["maxwell_F"] = solve(task["point"]["size"])[
                    "maxwell_capacitance"
                ].tolist()
            rows.append(row)
        (folder / "summary.json").write_text(json.dumps(rows))
    return curator, study, calibration


@pytest.fixture
def diagnostic_study(complete_study: tuple[Any, Path, Path]) -> tuple[Any, Path, Path]:
    curator, study, calibration = complete_study
    folder = study / "results/job"
    rows = json.loads((folder / "summary.json").read_text())
    template = rows[2]
    data = Dataset(folder / template["label"] / "data")
    tasks = []
    for control, order, scale in (("same-mesh", 3, 1.0), ("lower-order", 2, 1.2)):
        task = {
            "recipe": "xmon_readout",
            "mode": "control",
            "label": f"diagnostic-{control}",
            "anchor": "diagnostic",
            "control": control,
            "point": template["point"],
            "settings": rows[0]["settings"] | {"order": order},
        }
        metadata = data.metadata.model_copy(
            update={
                "provenance": data.metadata.provenance | {"settings": task["settings"]}
            }
        )
        frame = (
            data
            .scan()
            .collect()
            .with_columns(
                pl
                .when(pl.col("quantity") == "maxwell_capacitance")
                .then(pl.col("value") * scale)
                .otherwise(pl.col("value"))
                .alias("value")
            )
        )
        path = folder / task["label"] / "data"
        write(path, metadata, frame)
        matrix = Dataset(path).grid("maxwell_capacitance").values.reshape(2, 2)
        rows.append(task | {"status": "ok", "maxwell_F": matrix.tolist()})
        tasks.append(task)
    plan_file = study / "source/plan.json"
    plan = json.loads(plan_file.read_text())
    plan["jobs"][0]["tasks"].extend(tasks)
    plan_file.write_text(json.dumps(plan))
    (study / "source/source-sha256.json").write_text(
        json.dumps({"plan.json": hashlib.sha256(plan_file.read_bytes()).hexdigest()})
    )
    (folder / "summary.json").write_text(json.dumps(rows))
    return curator, study, calibration


def test_negative_control_is_retained_without_replacing_refinements(
    diagnostic_study: tuple[Any, Path, Path], tmp_path: Path
) -> None:
    curator, study, calibration = diagnostic_study
    output = tmp_path / "published"
    report = curator.verify(study, calibration, "xmon_readout", output=output)
    assert report["passed"]
    assert len(report["numerical"]) == 15
    diagnostic = report["diagnostic_controls"][0]
    assert not diagnostic["passed"]
    assert not diagnostic["used_for_admission"]
    assert diagnostic["candidate_maxwell_F"] != diagnostic["diagnostic_maxwell_F"]
    assert "diagnostic-lower-order" not in {r["label"] for r in report["interpolation"]}
    validation = Dataset(output).metadata.provenance["validation"]
    assert validation["diagnostic_controls"] == report["diagnostic_controls"]


@pytest.mark.parametrize(
    "problem", ["order", "mesh", "candidate", "coverage", "failed"]
)
def test_bad_negative_controls_cannot_publish(
    diagnostic_study: tuple[Any, Path, Path], tmp_path: Path, problem: str
) -> None:
    curator, study, calibration = diagnostic_study
    path = study / "results/job/summary.json"
    rows = json.loads(path.read_text())
    match problem:
        case "order":
            rows[-1]["settings"]["order"] = 4
        case "mesh":
            rows[-1]["settings"]["near_mesh"] *= 0.7
        case "candidate":
            rows.pop(-2)
        case "coverage":
            rows[-1]["anchor"] = "dense-held-diagnostic"
        case "failed":
            rows[-1]["status"] = "failed"
    path.write_text(json.dumps(rows))
    if problem in {"order", "mesh", "coverage"}:
        plan_file = study / "source/plan.json"
        plan = json.loads(plan_file.read_text())
        plan["jobs"][0]["tasks"][-1] |= {
            key: rows[-1][key] for key in ("settings", "anchor")
        }
        plan_file.write_text(json.dumps(plan))
        (study / "source/source-sha256.json").write_text(
            json.dumps({
                "plan.json": hashlib.sha256(plan_file.read_bytes()).hexdigest()
            })
        )
        data_path = path.parent / rows[-1]["label"] / "data"
        data = Dataset(data_path)
        metadata = data.metadata.model_copy(
            update={
                "provenance": data.metadata.provenance
                | {"settings": rows[-1]["settings"]}
            }
        )
        write(data_path, metadata, data.scan().collect())
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=r"diagnostic|Diagnostic|incomplete"):
        curator.verify(study, calibration, "xmon_readout", output=output)
    assert not output.exists()


def test_diagnostic_role_must_match_the_frozen_plan(
    diagnostic_study: tuple[Any, Path, Path],
) -> None:
    curator, study, calibration = diagnostic_study
    path = study / "results/job/summary.json"
    rows = json.loads(path.read_text())
    rows[2]["control"] = "lower-order"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="frozen task"):
        curator.verify(study, calibration, "xmon_readout")


def test_diagnostic_plan_can_use_recorded_solver_defaults(
    diagnostic_study: tuple[Any, Path, Path],
) -> None:
    curator, study, calibration = diagnostic_study
    path = study / "source/plan.json"
    plan = json.loads(path.read_text())
    for task in plan["jobs"][0]["tasks"][-2:]:
        del task["settings"]["tolerance"]
    path.write_text(json.dumps(plan))
    (study / "source/source-sha256.json").write_text(
        json.dumps({"plan.json": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    report = curator.verify(study, calibration, "xmon_readout")
    assert report["passed"]
    assert not report["diagnostic_controls"][0]["passed"]


def test_publish_preserves_evidence(
    complete_study: tuple[Any, Path, Path], tmp_path: Path
) -> None:
    curator, study, calibration = complete_study
    output = tmp_path / "published"
    report = curator.verify(study, calibration, "xmon_readout", output=output)
    assert report["passed"]
    assert len(report["numerical"]) == 15
    assert all(r["passed"] for r in report["interpolation"])
    dataset = Dataset(output)
    assert (
        dataset.metadata.description == "Analytical circuit fixture, not FEM evidence."
    )
    assert dataset.metadata.axes[0].validated == (0.0, 1.0)
    validation = dataset.metadata.provenance["validation"]
    assert validation["sources"][0]["job_id"] == "123"
    assert validation["numerical"][0]["reference_maxwell_F"]
    assert validation["held_out_coverage"]["passed"]
    assert [p.suffix for p in output.iterdir()] == [".parquet"]


@pytest.mark.parametrize("problem", ["accuracy", "residual", "synthetic", "summary"])
def test_reject_bad_reference(
    complete_study: tuple[Any, Path, Path], tmp_path: Path, problem: str
) -> None:
    curator, study, calibration = complete_study
    folder = study / "results/job"
    rows = json.loads((folder / "summary.json").read_text())
    row = rows[2]
    path = folder / row["label"] / "data"
    data = Dataset(path)
    metadata = data.metadata
    frame = data.scan().collect()
    if problem == "accuracy":
        frame = frame.with_columns(
            pl
            .when(pl.col("quantity") == "maxwell_capacitance")
            .then(pl.col("value") * 1.2)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        row["maxwell_F"] = [[v * 1.2 for v in r] for r in row["maxwell_F"]]
    elif problem == "residual":
        frame = frame.with_columns(
            pl
            .when(pl.col("quantity") == "solver_relative_residual")
            .then(1.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    elif problem == "synthetic":
        metadata = metadata.model_copy(update={"synthetic": True})
    else:
        row["maxwell_F"][0][0] *= 1.2
    write(path, metadata, frame)
    (folder / "summary.json").write_text(json.dumps(rows))
    output = tmp_path / "rejected"
    with pytest.raises(
        ValueError, match=r"failed independent|residual|Synthetic|summary differs"
    ):
        curator.verify(study, calibration, "xmon_readout", output=output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["grid", "residual", "synthetic", "calibration"])
def test_reject_bad_production(
    complete_study: tuple[Any, Path, Path], tmp_path: Path, problem: str
) -> None:
    curator, study, calibration = complete_study
    path = study / "results/job/production/data"
    data = Dataset(path)
    frame = data.scan().collect()
    metadata = data.metadata
    if problem == "grid":
        frame = frame.filter(pl.col("size") < 0.5)
    elif problem == "residual":
        frame = frame.with_columns(
            pl
            .when(pl.col("quantity") == "solver_relative_residual")
            .then(1.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    elif problem == "synthetic":
        metadata = metadata.model_copy(update={"synthetic": True})
    else:
        provenance = dict(metadata.provenance)
        provenance["settings"] = provenance["settings"] | {"order": 2}
        metadata = metadata.model_copy(update={"provenance": provenance})
    write(path, metadata, frame)
    output = tmp_path / "rejected"
    with pytest.raises(
        ValueError,
        match=r"cover exactly|residual|Production shard physics|Calibration mesh",
    ):
        curator.verify(study, calibration, "xmon_readout", output=output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["geometry", "settings", "coarser", "failed"])
def test_reject_bad_control(
    complete_study: tuple[Any, Path, Path], tmp_path: Path, problem: str
) -> None:
    curator, study, calibration = complete_study
    path = calibration / "results/job/summary.json"
    rows = json.loads(path.read_text())
    row = rows[2]
    if problem == "geometry":
        row["point"] = {"size": 0.2}
    elif problem == "settings":
        row["settings"]["tolerance"] *= 2
    elif problem == "coarser":
        row["settings"]["order"] = 2
    else:
        row["status"] = "failed"
        row["error"] = "Deliberate test failure"
    path.write_text(json.dumps(rows))
    output = tmp_path / "rejected"
    with pytest.raises(
        ValueError, match=r"identical geometry|unrelated solver|must refine|incomplete"
    ):
        curator.verify(study, calibration, "xmon_readout", output=output)
    assert not output.exists()


def test_cli_report(complete_study: tuple[Any, Path, Path], tmp_path: Path) -> None:
    curator, study, calibration = complete_study
    path = tmp_path / "report.json"
    curator.main(study, calibration, "xmon_readout", report=path)
    assert json.loads(path.read_text())["passed"]


@pytest.fixture
def refined_study(
    complete_study: tuple[Any, Path, Path], tmp_path: Path
) -> tuple[Any, Path, Path, Path]:
    curator, original, calibration = complete_study
    refined = tmp_path / "refined"
    shutil.copytree(original, refined)
    path = refined / "source/plan.json"
    plan = json.loads(path.read_text())
    plan["jobs"][0]["tasks"] = [
        t for t in plan["jobs"][0]["tasks"] if t["mode"] == "control"
    ]
    summary = refined / "results/job/summary.json"
    summary.write_text(
        json.dumps([
            r for r in json.loads(summary.read_text()) if r["mode"] == "control"
        ])
    )
    plan["reused_production"] = [
        {
            "recipe": "xmon_readout",
            "campaign": original.name,
            "job": "job",
            "label": "production",
        }
    ]
    path.write_text(json.dumps(plan))
    (refined / "source/source-sha256.json").write_text(
        json.dumps({"plan.json": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    return curator, refined, calibration, original


def test_publish_multiple_campaigns(
    refined_study: tuple[Any, Path, Path, Path], tmp_path: Path
) -> None:
    curator, refined, calibration, original = refined_study
    output = tmp_path / "published"
    report = curator.verify(
        refined, calibration, "xmon_readout", references=[original], output=output
    )
    assert report["passed"]
    assert len(report["numerical"]) == 15
    assert len(report["interpolation"]) == 21
    assert report["sources"][0]["campaign"] == original.name
    assert {r["source"]["campaign"] for r in report["interpolation"]} == {
        original.name,
        calibration.name,
        refined.name,
    }
    assert Dataset(output).metadata.provenance["validation"] == report


def test_missing_reused_campaign_cannot_publish(
    refined_study: tuple[Any, Path, Path, Path], tmp_path: Path
) -> None:
    curator, refined, calibration, original = refined_study
    report = curator.verify(refined, calibration, "xmon_readout")
    assert report["pending"] == [f"{original.name}/job/production"]
    with pytest.raises(ValueError, match="incomplete"):
        curator.verify(refined, calibration, "xmon_readout", output=tmp_path / "out")


def test_reused_campaign_source_is_checked(
    refined_study: tuple[Any, Path, Path, Path],
) -> None:
    curator, refined, calibration, original = refined_study
    path = original / "source/plan.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        curator.verify(refined, calibration, "xmon_readout", references=[original])


def test_pending_intermediate_reference_cannot_publish(
    refined_study: tuple[Any, Path, Path, Path], tmp_path: Path
) -> None:
    curator, refined, calibration, original = refined_study
    path = original / "results/job/summary.json"
    rows = json.loads(path.read_text())
    path.write_text(json.dumps(rows[:-1]))
    report = curator.verify(refined, calibration, "xmon_readout", references=[original])
    assert not report["passed"]
    assert report["pending"] == [f"{original.name}/job/held-2"]
    with pytest.raises(ValueError, match="incomplete"):
        curator.verify(
            refined,
            calibration,
            "xmon_readout",
            references=[original],
            output=tmp_path / "out",
        )


@pytest.mark.parametrize("problem", ["duplicate", "ambiguous"])
def test_reused_campaign_identity_is_unique(
    refined_study: tuple[Any, Path, Path, Path], tmp_path: Path, problem: str
) -> None:
    curator, refined, calibration, original = refined_study
    references = [original]
    if problem == "duplicate":
        path = refined / "source/plan.json"
        plan = json.loads(path.read_text())
        plan["reused_production"] *= 2
        path.write_text(json.dumps(plan))
    else:
        other = tmp_path / "other" / original.name
        shutil.copytree(original, other)
        references.append(other)
    with pytest.raises(ValueError, match=r"unique source|more than once"):
        curator.verify(refined, calibration, "xmon_readout", references=references)
