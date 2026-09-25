"""Behavior tests for the result-provenance code cells of the COMSOL notebooks.

The cells are executed from the Jupytext sources so the tests exercise the code
that actually runs in the notebooks, against fakes in place of a licensed COMSOL.
"""

import ast
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


_FILE_SHA256 = _functions(
    _source(QUBIT_NOTEBOOK), {"file_sha256"}, hashlib=hashlib, Path=Path
)["file_sha256"]


def _sha256(path: Path) -> str:
    """Hash a file independently of the notebook's own helper.

    Args:
        path: File to hash.

    Returns:
        The SHA-256 digest as lowercase hexadecimal.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def test_export_cell_rerun_reuses_nodes_and_restamps_digest(tmp_path: Path) -> None:
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "field_export")
    model = _FakeModel()
    metrics_path = tmp_path / "comsol_qubit_metrics.json"
    metrics_path.write_text(json.dumps({"voltage_v": 1.0, "field_sha256": None}))
    namespace = {
        "RUN_COMSOL": True,
        "MPH_AVAILABLE": True,
        "model": model,
        "MODEL_DIR": tmp_path,
        "json": json,
        "file_sha256": _FILE_SHA256,
    }
    field_path = tmp_path / "comsol_qubit_field.txt"

    exec(cell, namespace)  # ruff: ignore[exec-builtin]
    first_bytes = field_path.read_bytes()
    assert json.loads(metrics_path.read_text())["field_sha256"] == _sha256(field_path)

    exec(cell, namespace)  # ruff: ignore[exec-builtin]
    assert json.loads(metrics_path.read_text())["field_sha256"] == _sha256(field_path)
    assert field_path.read_bytes() != first_bytes
    assert model.java.result().datasets.tags() == {"cutplane"}
    assert model.java.result().exports.tags() == {"field"}


class _NoPlot:
    """Any access to pyplot from a refusing readback is a failure."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"readback plotted via plt.{name}")


@pytest.mark.parametrize(
    ("stored_digest", "expected_message"),
    [
        (None, "No field digest"),
        ("0" * 64, "does not match the digest"),
    ],
)
def test_field_readback_refuses_unverified_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stored_digest: str | None,
    expected_message: str,
) -> None:
    cell = _cell_with(_source(QUBIT_NOTEBOOK), "field_file = result_file(FIELD_TXT)")
    field_path = tmp_path / "comsol_qubit_field.txt"
    field_path.write_text("V,E\n1,2\n")
    namespace = {
        "np": np,
        "plt": _NoPlot(),
        "FIELD_TXT": field_path.name,
        "METRICS_JSON": "comsol_qubit_metrics.json",
        "field_sha256": stored_digest,
        "result_file": lambda _name: field_path,
        "explain_missing_results": lambda _name: None,
        "file_sha256": _FILE_SHA256,
    }

    exec(cell, namespace)  # ruff: ignore[exec-builtin]

    assert expected_message in capsys.readouterr().out


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
        {"requested_frequency_grid", "solve_awe_curve"},
        np=np,
        Any=Any,
        Path=Path,
        FREQUENCY_STEP="freq",
        POWER_SUM_MIN=0.99,
        POWER_SUM_MAX=1.001,
    )
    _, grid = namespace["requested_frequency_grid"](
        AWE_LOW_GHZ, AWE_HIGH_GHZ, AWE_POINTS
    )
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
