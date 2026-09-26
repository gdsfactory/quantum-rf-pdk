"""Behavior tests for the result code cells of the COMSOL notebooks.

The notebook cells are executed from the Jupytext sources against fakes in place
of a licensed COMSOL, and the shared result helpers are imported from
:mod:`qpdk.simulation.comsol.results`.
"""

import ast
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib import tri as mtri
from matplotlib.colors import LogNorm

from qpdk.simulation.comsol.results import (
    complex_frequency_ghz,
    curve_value_db,
    exported_frequency_ghz,
    requested_frequency_grid,
    resolve_record_path,
    result_file,
    write_json_atomically,
)

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks" / "src"
QUBIT_NOTEBOOK = NOTEBOOKS / "comsol_qubit_capacitance.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_with(source: str, needle: str) -> str:
    cells: list[str] = []
    current: list[str] = []
    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            cells.append("".join(current))
            current = [line]
        else:
            current.append(line)
    cells.append("".join(current))
    matched = [cell for cell in cells if needle in cell]
    assert len(matched) == 1, f"expected one cell containing {needle!r}"
    return matched[0]


class _FakeFeature:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.settings: dict[str, Any] = {}
        self.runs = 0

    def set(self, key: str, value: Any) -> None:
        self.settings[key] = value

    def run(self) -> None:
        self.runs += 1
        Path(self.settings["filename"]).write_text(
            f"run {self.runs}\n", encoding="utf-8"
        )


class _FakeFeatureList:
    """Rejects a duplicate tag, the way COMSOL's feature lists do."""

    def __init__(self) -> None:
        self._items: dict[str, _FakeFeature] = {}

    def hasTag(self, tag: str) -> bool:  # ruff: ignore[invalid-function-name] (mirrors the Java method)
        return tag in self._items

    def create(self, tag: str, kind: str) -> _FakeFeature:
        if self.hasTag(tag):
            raise RuntimeError(f"tag {tag!r} already exists")
        self._items[tag] = _FakeFeature(kind)
        return self._items[tag]

    def __call__(self, tag: str) -> _FakeFeature:
        return self._items[tag]

    def tags(self) -> set[str]:
        return set(self._items)


class _FakeResult:
    def __init__(self) -> None:
        self.datasets = _FakeFeatureList()
        self.exports = _FakeFeatureList()

    def dataset(self, tag: str | None = None) -> Any:
        return self.datasets(tag) if tag is not None else self.datasets

    def export(self, tag: str | None = None) -> Any:
        return self.exports(tag) if tag is not None else self.exports


class _FakeJava:
    def __init__(self) -> None:
        self._result = _FakeResult()

    def result(self) -> _FakeResult:
        return self._result


class _FakeModel:
    def __init__(self) -> None:
        self.java = _FakeJava()


def test_export_cell_rerun_reuses_nodes(tmp_path: Path) -> None:
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "field_export.set")
    export = dedent(
        cell[
            cell.index("    result = model.java.result()") : cell.index(
                '    print(f"Exported V and es.normE'
            )
        ]
    )
    model = _FakeModel()
    namespace = {
        "RUN_COMSOL": True,
        "MPH_AVAILABLE": True,
        "model": model,
        "MODEL_DIR": tmp_path,
        "FIELD_TXT": "comsol_qubit_field.txt",
    }
    field_path = tmp_path / "comsol_qubit_field.txt"

    exec(export, namespace)  # ruff: ignore[exec-builtin]
    first_bytes = field_path.read_bytes()

    exec(export, namespace)  # ruff: ignore[exec-builtin]
    assert field_path.read_bytes() != first_bytes
    assert model.java.result().datasets.tags() == {"cutplane"}
    assert model.java.result().exports.tags() == {"field"}


def test_new_qubit_solve_invalidates_the_previous_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "model = None")
    tree = ast.parse(cell)
    solve = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.get_source_segment(cell, node.test) == "RUN_COMSOL and MPH_AVAILABLE"
    )
    source = ast.get_source_segment(cell, solve)
    assert source is not None

    field = tmp_path / "field.txt"
    field.write_text("old solve")
    model = MagicMock()
    model.add_capacitance_study.return_value = model
    model.pin_absolute_mesh_sizes.return_value = 100
    model.problems.return_value = []
    model.evaluate.side_effect = [125e-15, 250e-15]
    model.java.result.return_value = _FakeResult()

    def fail_export(_self: _FakeFeature) -> None:
        raise RuntimeError("export failed")

    monkeypatch.setattr(_FakeFeature, "run", fail_export)
    namespace = {
        "RUN_COMSOL": True,
        "MPH_AVAILABLE": True,
        "MODEL_DIR": tmp_path,
        "MODEL_PATH": tmp_path / "model.mph",
        "METRICS_JSON": "comsol_qubit_metrics.json",
        "FIELD_TXT": field.name,
        "CORES": 1,
        "mph": SimpleNamespace(start=lambda **_kwargs: object()),
        "COMSOL": SimpleNamespace(create_sheet=lambda *_args, **_kwargs: model),
        "layout": object(),
        "NEAR_METAL": "fine",
        "PAD_HMAX_UM": 1.0,
        "PAD_HMIN_UM": 0.1,
        "GROUND_HMAX_UM": 2.0,
        "GROUND_HMIN_UM": 0.2,
        "PAD_L_SELECTION": "left",
        "PAD_R_SELECTION": "right",
        "GROUND_SELECTION": "ground",
        "CONDUCTORS": (),
        "VOLTAGE_V": 2.0,
        "BASE_MESH_SIZE": 2,
        "GLOBAL_HMAX_UM": 10.0,
        "GLOBAL_HMIN_UM": 1.0,
        "HGRAD": 1.4,
        "HCURVE": 0.5,
        "HNARROW": 0.7,
        "LATERAL_MARGIN_UM": 100.0,
        "SUBSTRATE_THICKNESS_UM": 200.0,
        "AIR_HEIGHT_UM": 200.0,
        "SILICON_RELATIVE_PERMITTIVITY": 11.7,
        "write_json_atomically": write_json_atomically,
    }

    with pytest.raises(RuntimeError, match="export failed"):
        exec(source, namespace)  # ruff: ignore[exec-builtin]

    model.save.assert_called_once_with(tmp_path / "model.mph")
    assert not field.exists()
    assert json.loads((tmp_path / "comsol_qubit_metrics.json").read_text())[
        "voltage_v"
    ] == pytest.approx(2.0)


class _FakeAxes:
    def __init__(self) -> None:
        self.contours = 0

    def contourf(self, *_args: Any, **_kwargs: Any) -> Any:
        self.contours += 1
        return object()

    def set_title(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_xlabel(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_ylabel(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _FakeFigure:
    def colorbar(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _FakePyplot:
    def __init__(self) -> None:
        self.axes: list[_FakeAxes] = []

    def subplots(self, *_args: Any, **_kwargs: Any) -> tuple[_FakeFigure, list[Any]]:
        self.axes = [_FakeAxes(), _FakeAxes()]
        return _FakeFigure(), list(self.axes)

    def tight_layout(self) -> None:
        pass

    def show(self) -> None:
        pass


def test_field_readback_plots_when_the_export_exists(tmp_path: Path) -> None:
    """A saved field is drawn from disk, with no digest guarding the read."""
    cell = _cell_with(
        _source(QUBIT_NOTEBOOK), "field_file = result_file(RESULTS_DIR, FIELD_TXT)"
    )
    field_path = tmp_path / "comsol_qubit_field.txt"
    # COMSOL text exports are whitespace-separated, which is what the notebook's
    # default-delimiter np.loadtxt expects.
    field_path.write_text(
        "-100.0 0.0 1.0 1.0 5.0\n"
        "0.0 0.0 1.0 0.5 3.0\n"
        "100.0 0.0 1.0 0.2 1.0\n"
        "0.0 50.0 1.0 0.1 0.5\n",
        encoding="utf-8",
    )
    pyplot = _FakePyplot()
    namespace = {
        "np": np,
        "mtri": mtri,
        "LogNorm": LogNorm,
        "plt": pyplot,
        "RESULTS_DIR": tmp_path,
        "FIELD_TXT": field_path.name,
        "voltage_v": 1.0,
        "result_file": result_file,
        "explain_missing_results": lambda *_args: "",
    }

    exec(cell, namespace)  # ruff: ignore[exec-builtin]

    assert [axes.contours for axes in pyplot.axes] == [1, 1]


def test_exported_frequency_reads_bare_and_named_annotations(tmp_path: Path) -> None:
    named = tmp_path / "named.txt"
    named.write_text("% @ freq=7.5\nV,E\n", encoding="utf-8")
    assert exported_frequency_ghz(named) == pytest.approx(7.5)

    bare = tmp_path / "bare.txt"
    bare.write_text("% @ 7.2921 GHz\nV,E\n", encoding="utf-8")
    assert exported_frequency_ghz(bare) == pytest.approx(7.2921)

    scaled = tmp_path / "scaled.txt"
    scaled.write_text("% @ 750 MHz\nV,E\n", encoding="utf-8")
    assert exported_frequency_ghz(scaled) == pytest.approx(0.75)

    silent = tmp_path / "silent.txt"
    silent.write_text("V,E\n0,1\n", encoding="utf-8")
    assert exported_frequency_ghz(silent) is None


def test_complex_frequency_reads_real_and_imaginary(tmp_path: Path) -> None:
    annotated = tmp_path / "ported.txt"
    annotated.write_text("% @ 7.3266+5.3458E-4i GHz\nx,y,z,E\n", encoding="utf-8")
    annotation = complex_frequency_ghz(annotated)
    assert annotation is not None
    real_ghz, imag_ghz = annotation
    assert real_ghz == pytest.approx(7.3266)
    assert imag_ghz == pytest.approx(5.3458e-4)

    real_only = tmp_path / "real.txt"
    real_only.write_text("% @ 7.2921 GHz\nx,y,z,E\n", encoding="utf-8")
    assert complex_frequency_ghz(real_only) is None


def test_curve_value_clamps_near_the_edge_and_refuses_far_outside() -> None:
    frequencies_ghz = np.array([4.0, 4.5, 5.0])
    s21_db = np.array([-1.0, -3.0, -2.0])

    assert curve_value_db(
        frequencies_ghz, s21_db, 4.5, endpoint_tolerance_ghz=1e-7
    ) == pytest.approx(-3.0)
    assert curve_value_db(
        frequencies_ghz, s21_db, 4.0 - 1e-9, endpoint_tolerance_ghz=1e-7
    ) == pytest.approx(-1.0)
    assert (
        curve_value_db(frequencies_ghz, s21_db, 3.9, endpoint_tolerance_ghz=1e-7)
        is None
    )
    assert (
        curve_value_db(np.array([]), np.array([]), 5.0, endpoint_tolerance_ghz=1e-7)
        is None
    )


def test_result_file_and_record_path_resolution(tmp_path: Path) -> None:
    present = tmp_path / "on_disk.txt"
    present.write_text("x\n", encoding="utf-8")

    assert result_file(tmp_path, "on_disk.txt") == present
    assert result_file(tmp_path, "absent.txt") is None
    assert result_file(None, "on_disk.txt") is None

    record_dir = tmp_path / "records"
    record_dir.mkdir()
    # A bare name resolves under the results directory when the file is there,
    # else beside the record; an absolute stored path is honoured as written.
    assert resolve_record_path(tmp_path, record_dir, "on_disk.txt") == present
    assert (
        resolve_record_path(tmp_path, record_dir, "only_here.txt")
        == record_dir / "only_here.txt"
    )
    assert resolve_record_path(tmp_path, record_dir, str(present)) == present


def test_requested_grid_returns_the_exact_point_count() -> None:
    expression, grid_ghz = requested_frequency_grid(7.0, 7.002, 11)

    assert grid_ghz.size == 11
    assert grid_ghz[0] == pytest.approx(7.0)
    assert expression.startswith("range(")
    assert expression.endswith("[GHz])")
