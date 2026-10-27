"""Incomplete or altered FEM studies cannot publish a curated dataset."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def incomplete_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path, Path]:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
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
