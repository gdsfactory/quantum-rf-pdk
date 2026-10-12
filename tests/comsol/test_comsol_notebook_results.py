"""Behavior tests for the result code cells of the COMSOL notebooks.

The notebook cells are executed from the Jupytext sources against fakes in place
of a licensed COMSOL, and the shared result helpers are imported from
:mod:`qpdk.simulation.comsol.results`.
"""

import ast
import json
from contextlib import suppress
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
    requested_frequency_grid,
    result_file,
    write_json_atomically,
)

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks" / "src"
QUBIT_NOTEBOOK = NOTEBOOKS / "comsol_qubit_capacitance.py"
CPW_NOTEBOOK = NOTEBOOKS / "comsol_cpw_resonator.py"


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


def test_cpw_export_failure_keeps_the_solved_model(tmp_path: Path) -> None:
    source = _source(CPW_NOTEBOOK)
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "solve_driven"
    )
    code = compile(
        ast.Module(body=[function], type_ignores=[]), str(CPW_NOTEBOOK), "exec"
    )

    client = MagicMock()
    model = MagicMock()
    model.pin_absolute_edge_mesh_sizes.return_value = 100
    model.problems.return_value = []
    model.evaluate.side_effect = [[7.3265e9], [-20.0], [-0.1]]
    evaluation = MagicMock()
    evaluation.property.return_value = "dset1"
    evaluations = MagicMock()
    evaluations.create.return_value = evaluation
    dataset = SimpleNamespace(tag=lambda: "dset1")
    model.__truediv__.side_effect = lambda name: (
        evaluations if name == "evaluations" else [dataset]
    )
    model.java.result().export().create().run.side_effect = RuntimeError(
        "export failed"
    )
    api = SimpleNamespace(create_sheet=lambda *_args, **_kwargs: model)
    namespace: dict[str, Any] = {
        "COMSOL": api,
        "np": np,
        "suppress": suppress,
        "requested_frequency_grid": requested_frequency_grid,
        "create_meander_edge_selection": lambda _model: [1],
        "layout": object(),
        "MODEL_DIR": tmp_path,
        "DRIVEN_MODEL_MPH": "solved.mph",
        "DRIVEN_FIELD_TXT": "field.txt",
        "AWE_CURVE_CSV": "curve.csv",
        "SUBSTRATE_THICKNESS_UM": 200.0,
        "AIR_HEIGHT_UM": 200.0,
        "SILICON_RELATIVE_PERMITTIVITY": 11.7,
        "CPW_GAP_UM": 6.0,
        "SWEEP_CENTER_GHZ": 7.3265,
        "MESH_SIZE": 7,
        "PORT_MODE_INDEX_SHIFT": 2.5,
        "MEANDER_EDGE_SELECTION": "edges",
        "GLOBAL_HMAX_UM": 100.0,
        "GLOBAL_HMIN_UM": 1.0,
        "EDGE_HMAX_UM": 4.0,
        "EDGE_HMIN_UM": 0.4,
        "SWEEP_HALF_SPAN_GHZ": 0.001,
        "SWEEP_POINTS": 11,
        "FIELD_CUT_Z": "1[um]",
    }
    exec(code, namespace)  # ruff: ignore[exec-builtin]

    with pytest.raises(RuntimeError, match="export failed"):
        namespace["solve_driven"](client)

    model.save.assert_called_once_with(tmp_path / "solved.mph")
    client.remove.assert_called_once_with(model)


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
