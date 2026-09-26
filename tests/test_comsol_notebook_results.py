"""Behavior tests for the result code cells of the COMSOL notebooks.

The notebook cells are executed from the Jupytext sources against fakes in place
of a licensed COMSOL, and the shared result helpers are imported from
:mod:`qpdk.simulation.comsol.results`.
"""

import ast
import json
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
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
RESONATOR_NOTEBOOK = NOTEBOOKS / "comsol_cpw_resonator.py"


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


def _functions(source: str, names: set[str], **globals_: Any) -> dict[str, Any]:
    tree = ast.parse(source)
    chunks = [
        segment
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        and (segment := ast.get_source_segment(source, node)) is not None
    ]
    assert len(chunks) == len(names), f"missing functions in {names}"
    namespace: dict[str, Any] = dict(globals_)
    exec("\n\n".join(chunks), namespace)  # ruff: ignore[exec-builtin]
    return namespace


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
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "field_export")
    model = _FakeModel()
    namespace = {
        "RUN_COMSOL": True,
        "MPH_AVAILABLE": True,
        "model": model,
        "MODEL_DIR": tmp_path,
        "FIELD_TXT": "comsol_qubit_field.txt",
    }
    field_path = tmp_path / "comsol_qubit_field.txt"

    exec(cell, namespace)  # ruff: ignore[exec-builtin]
    first_bytes = field_path.read_bytes()

    exec(cell, namespace)  # ruff: ignore[exec-builtin]
    assert field_path.read_bytes() != first_bytes
    assert model.java.result().datasets.tags() == {"cutplane"}
    assert model.java.result().exports.tags() == {"field"}


def test_new_qubit_solve_invalidates_the_previous_field(tmp_path: Path) -> None:
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "MODEL_PATH = MODEL_DIR /")
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
    namespace = {
        "RUN_COMSOL": True,
        "MPH_AVAILABLE": True,
        "MODEL_DIR": tmp_path,
        "MODEL_PATH": tmp_path / "model.mph",
        "FIELD_TXT": field.name,
        "CORES": 1,
        "mph": SimpleNamespace(start=lambda **_kwargs: object()),
        "COMSOL": SimpleNamespace(create_sheet=lambda *_args, **_kwargs: model),
        "layout": object(),
        "NEAR_METAL_SIZES": {"fine": (1.0, 0.1, 2.0, 0.2)},
        "MAIN_NEAR_METAL": "fine",
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
        "json": json,
    }

    exec(source, namespace)  # ruff: ignore[exec-builtin]

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


class _AweStudy:
    def feature(self, _tag: str) -> Any:
        return self

    def set(self, *args: Any) -> None:
        pass

    def run(self) -> None:
        pass


class _AweJava:
    def __init__(self) -> None:
        self._study = _AweStudy()

    def study(self, _tag: str) -> _AweStudy:
        return self._study


class _AweModel:
    def __init__(self) -> None:
        self.java = _AweJava()


AWE_LOW_GHZ = 4.0
AWE_HIGH_GHZ = 6.0
AWE_POINTS = 9


def _awe_namespace(
    spoil: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build the AWE functions with a stub solution over the real requested grid.

    Args:
        spoil: Turns the requested grid into the rows the stub returns, or ``None``
            to return the requested grid unchanged.

    Returns:
        A namespace holding the extracted functions and the stub.
    """
    namespace = _functions(
        _source(RESONATOR_NOTEBOOK),
        {"solve_awe_curve"},
        np=np,
        Any=Any,
        Path=Path,
        FREQUENCY_STEP="freq",
        POWER_SUM_MIN=0.99,
        POWER_SUM_MAX=1.001,
        requested_frequency_grid=requested_frequency_grid,
    )
    _, grid = requested_frequency_grid(AWE_LOW_GHZ, AWE_HIGH_GHZ, AWE_POINTS)
    returned = grid if spoil is None else spoil(grid)

    def solution(_model: Any) -> dict[str, Any]:
        return {
            "frequency_ghz": returned,
            "s21_db": np.full(returned.size, -20.0),
            "s11_db": np.full(returned.size, -0.08),
            "power_sum": np.full(returned.size, 0.995),
            "dataset": "dset1",
        }

    namespace["frequency_solution"] = solution
    return namespace


# Each returns the same number of rows as the grid but drops one requested point
# and adds an interior row in its place, so a row count alone still matches. The
# dropped endpoint in the first two sits outside the returned range, where the
# pre-fix clip-to-endpoint distance could pass a missing point as zero.
def _grid_missing_first(grid: np.ndarray) -> np.ndarray:
    step_ghz = float(grid[1] - grid[0])
    return np.concatenate([[grid[0] + 0.5 * step_ghz], grid[1:]])


def _grid_missing_last(grid: np.ndarray) -> np.ndarray:
    step_ghz = float(grid[1] - grid[0])
    return np.concatenate([grid[:-1], [grid[-2] + 0.5 * step_ghz]])


def _grid_missing_interior(grid: np.ndarray) -> np.ndarray:
    step_ghz = float(grid[1] - grid[0])
    middle = grid.size // 2
    return np.concatenate([
        grid[:middle],
        [grid[middle - 1] + 0.5 * step_ghz],
        grid[middle + 1 :],
    ])


@pytest.mark.parametrize(
    "spoil",
    [_grid_missing_first, _grid_missing_last, _grid_missing_interior],
    ids=["missing-first", "missing-last", "missing-interior"],
)
def test_awe_curve_rejects_incomplete_grid(
    tmp_path: Path, spoil: Callable[[np.ndarray], np.ndarray]
) -> None:
    namespace = _awe_namespace(spoil)

    with pytest.raises(RuntimeError, match="omitted"):
        namespace["solve_awe_curve"](
            _AweModel(),
            AWE_LOW_GHZ,
            AWE_HIGH_GHZ,
            AWE_POINTS,
            tmp_path / "rejected.csv",
        )


def test_awe_curve_accepts_full_grid(tmp_path: Path) -> None:
    namespace = _awe_namespace()
    path = tmp_path / "curve.csv"

    curve = namespace["solve_awe_curve"](
        _AweModel(), AWE_LOW_GHZ, AWE_HIGH_GHZ, AWE_POINTS, path
    )

    assert curve["requested_points"] == AWE_POINTS
    assert curve["solved_rows"] == AWE_POINTS
    assert path.exists()


def test_ported_mesh_series_rejects_a_different_setup(tmp_path: Path) -> None:
    setup = {"layout": {"bbox": [0, 1]}, "silicon_relative_permittivity": 11.7}
    records = (
        ("matching", {"setup": setup, "element_count": 100}),
        ("different", {"setup": {"layout": {"bbox": [0, 2]}}, "element_count": 200}),
        ("legacy", {"element_count": 300}),
    )
    for name, record in records:
        (tmp_path / f"ported_{name}.json").write_text(json.dumps(record))

    update = _functions(
        _source(RESONATOR_NOTEBOOK),
        {"update_ported_mesh_series"},
        Any=Any,
        Path=Path,
        json=json,
        itemgetter=itemgetter,
        PORTED_EIGEN_TAGGED_PREFIX="ported",
        PORTED_MESH_SERIES_JSON="series.json",
        ported_series_row=lambda record: (
            {"element_count": record["element_count"]},
            "",
        ),
        write_json_atomically=write_json_atomically,
    )["update_ported_mesh_series"]

    payload = update(tmp_path, setup)

    assert payload["rows"] == [{"element_count": 100}]
    assert len(payload["rejected"]) == 2
