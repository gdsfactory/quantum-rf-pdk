"""Polars results table of a FEM extraction dataset.

The table is long-format: one row per parameter point, quantity, and matrix entry.

=============  ===========  ===========================================================
Column         Type         Meaning
=============  ===========  ===========================================================
``run_id``     String       Identity of the solver run that produced the row.
``status``     String       ``ok``, ``failed``, or ``not_converged`` (see :class:`RunStatus`).
<axis>         Float64      One column per manifest axis, in the axis' unit.
<variant>      String       One column per manifest variant.
``quantity``   String       Name of a manifest quantity.
``row``        String       Row terminal label; null for scalar quantities.
``col``        String       Column terminal label; null for scalar quantities.
``value``      Float64      Value (real part for complex quantities); null unless ``ok``.
``value_imag`` Float64      Imaginary part; null for real quantities.
``unit``       String       SI unit; must equal the manifest unit of the quantity.
=============  ===========  ===========================================================

Results live in ``results/*.parquet`` and are scanned together, so a resumable
sweep appends a new part file per batch instead of rewriting a single large
LFS object.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

from qpdk.datasets.manifest import Axis, Manifest, Quantity, load_manifest

DATASETS_PATH = Path(__file__).parent / "data"
"""Curated datasets shipped with qpdk."""

_LFS_POINTER_PREFIX = b"version https://git-lfs"


class RunStatus(StrEnum):
    """Outcome of a solver run."""

    OK = "ok"
    FAILED = "failed"
    NOT_CONVERGED = "not_converged"


class DatasetError(ValueError):
    """The dataset is inconsistent with its manifest or cannot be gridded."""

    def __init__(self, dataset: str, problems: list[str]) -> None:
        """Collect every problem into one message."""
        self.problems = problems
        super().__init__(
            f"Dataset {dataset!r} failed validation:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


class LFSPointerError(FileNotFoundError):
    """A results file is an unresolved Git LFS pointer instead of Parquet data."""


@dataclass(frozen=True)
class Grid:
    """A quantity on a complete rectilinear grid, ready for interpolation.

    ``values`` has shape ``(*axis sizes, *component shape)`` where the component
    shape is ``(n_terminals, n_terminals)`` for matrix quantities and ``()`` for
    scalars. Axes follow manifest order, terminals follow
    :attr:`~qpdk.datasets.manifest.Conventions.terminals`. Missing or failed
    points are NaN, never zero.
    """

    dataset: str
    quantity: Quantity
    axes: tuple[Axis, ...]
    terminals: tuple[str, ...]
    variant: dict[str, str]
    values: np.ndarray

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Axis names in tensor order."""
        return tuple(axis.name for axis in self.axes)


def schema(manifest: Manifest) -> pl.Schema:
    """Column schema of the results table for ``manifest``."""
    return pl.Schema({
        "run_id": pl.String,
        "status": pl.String,
        **{axis.name: pl.Float64 for axis in manifest.axes},
        **{variant.name: pl.String for variant in manifest.variants},
        "quantity": pl.String,
        "row": pl.String,
        "col": pl.String,
        "value": pl.Float64,
        "value_imag": pl.Float64,
        "unit": pl.String,
    })


def _check_not_lfs_pointer(path: Path) -> None:
    """Raise :class:`LFSPointerError` if ``path`` holds an unresolved LFS pointer."""
    with path.open("rb") as file:
        head = file.read(len(_LFS_POINTER_PREFIX))
    if head == _LFS_POINTER_PREFIX:
        msg = (
            f"{path} is a Git LFS pointer, not Parquet data. From the root of a source "
            'checkout run `git lfs install && git lfs pull --include "qpdk/datasets/data/**"`. '
            "Installed qpdk wheels already contain the data."
        )
        raise LFSPointerError(msg)


class Dataset:
    """A FEM extraction dataset: ``manifest.toml`` plus ``results/*.parquet``.

    Args:
        path: Dataset directory, or the name of a dataset shipped in
            :data:`DATASETS_PATH`.
    """

    def __init__(self, path: Path | str) -> None:
        """Open the dataset and validate its manifest."""
        path = Path(path)
        if not path.is_dir() and (DATASETS_PATH / path).is_dir():
            path = DATASETS_PATH / path
        self.path = path
        self.manifest = load_manifest(path / "manifest.toml")

    def __repr__(self) -> str:
        """Show the dataset directory."""
        return f"Dataset({self.path!s})"

    @property
    def result_files(self) -> list[Path]:
        """Parquet part files, in a deterministic order."""
        return sorted((self.path / "results").glob("*.parquet"))

    def scan(self) -> pl.LazyFrame:
        """Lazily scan all result parts."""
        files = self.result_files
        if not files:
            msg = f"No results/*.parquet files in {self.path}."
            raise FileNotFoundError(msg)
        for file in files:
            _check_not_lfs_pointer(file)
        return pl.scan_parquet(files, schema=schema(self.manifest))

    @cached_property
    def table(self) -> pl.DataFrame:
        """The full results table, validated against the manifest."""
        try:
            frame = self.scan().collect()
        except (pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError) as error:
            raise DatasetError(
                self.manifest.name, [f"Result parts do not match the schema: {error}"]
            ) from error
        validate(frame, self.manifest)
        return frame

    def grid(
        self, quantity: str, *, allow_missing: bool = False, **variant: str
    ) -> Grid:
        """Arrange ``quantity`` on its grid; see :func:`to_grid`."""
        return to_grid(
            self.table, self.manifest, quantity, allow_missing=allow_missing, **variant
        )

    def append(self, frame: pl.DataFrame, *, part: str | None = None) -> Path:
        """Validate ``frame`` and write it as a new ``results/<part>.parquet``.

        Existing parts are never rewritten. Duplicates against earlier parts are
        rejected, so re-running a sweep cannot silently double a point.

        Returns:
            Path of the written part.

        Raises:
            FileExistsError: if the part already exists.
        """
        frame = frame.select(schema(self.manifest).names()).cast(schema(self.manifest))  # pyrefly: ignore[bad-argument-type]
        existing = [self.scan().collect()] if self.result_files else []
        validate(pl.concat([*existing, frame]), self.manifest)
        target = self.path / "results" / f"{part or uuid.uuid4().hex}.parquet"
        if target.exists():
            msg = f"{target} already exists; result parts are append-only."
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.sort(_key_columns(self.manifest), nulls_last=True).write_parquet(
            target, compression="zstd", statistics=True
        )
        self.__dict__.pop("table", None)
        return target


def _key_columns(manifest: Manifest) -> list[str]:
    """Columns that identify one value: variants, axes, quantity, row, col."""
    return [
        *(v.name for v in manifest.variants),
        *(a.name for a in manifest.axes),
        "quantity",
        "row",
        "col",
    ]


def validate(frame: pl.DataFrame, manifest: Manifest) -> None:
    """Check a results table against its manifest.

    Detects wrong columns, unknown quantities or statuses, unit mismatches,
    terminals outside the declared conventions, parameter values off the declared
    grid, inconsistent runs, duplicate points, and missing or spurious values.

    Raises:
        DatasetError: listing every problem found.
    """
    problems: list[str] = []
    expected = schema(manifest)
    if dict(frame.schema) != dict(expected):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        wrong = sorted(
            c
            for c in set(expected) & set(frame.columns)
            if frame.schema[c] != expected[c]
        )
        problems.append(
            f"Columns do not match the schema (missing {missing}, unexpected {extra}, wrong dtype {wrong})."
        )
        raise DatasetError(manifest.name, problems)

    def report(mask: pl.Expr, what: str, columns: list[str]) -> None:
        bad = frame.filter(mask)
        if bad.height:
            sample = bad.select(columns).unique(maintain_order=True).head(5).rows()
            problems.append(f"{bad.height} rows {what}, e.g. {sample}.")

    required = [
        "run_id",
        "status",
        "quantity",
        "unit",
        *(v.name for v in manifest.variants),
        *(a.name for a in manifest.axes),
    ]
    for column in required:
        report(pl.col(column).is_null(), f"have a null {column!r}", ["run_id"])
    report(
        pl.col("value").is_nan() | pl.col("value_imag").is_nan(),
        "have a NaN value; record a failed solve with its status and a null value",
        ["run_id", "status"],
    )

    quantities = {q.name: q for q in manifest.quantities}
    report(
        ~pl.col("quantity").is_in(list(quantities)),
        "have an unknown quantity",
        ["quantity"],
    )
    report(
        ~pl.col("status").is_in([s.value for s in RunStatus]),
        "have an unknown status",
        ["status"],
    )
    terminals = list(manifest.conventions.terminals)
    for name, quantity in quantities.items():
        is_q = pl.col("quantity") == name
        report(
            is_q & (pl.col("unit") != quantity.unit),
            f"of {name!r} are not in {quantity.unit!r}",
            ["unit"],
        )
        if quantity.matrix:
            report(
                is_q
                & ~(pl.col("row").is_in(terminals) & pl.col("col").is_in(terminals)),
                f"of {name!r} have terminals outside {terminals}",
                ["row", "col"],
            )
        else:
            report(
                is_q & (pl.col("row").is_not_null() | pl.col("col").is_not_null()),
                f"of scalar {name!r} have terminals",
                ["row", "col"],
            )
        imag = pl.col("value_imag").is_not_null()
        report(
            is_q
            & (~imag if quantity.complex else imag)
            & (pl.col("status") == RunStatus.OK),
            f"of {name!r} have an inconsistent imaginary part",
            ["run_id"],
        )

    for axis in manifest.axes:
        report(
            ~pl.col(axis.name).is_in(list(axis.values)),
            f"have {axis.name!r} off the manifest grid",
            [axis.name],
        )
    for variant in manifest.variants:
        report(
            ~pl.col(variant.name).is_in(list(variant.values)),
            f"have an undeclared {variant.name!r}",
            [variant.name],
        )

    ok = pl.col("status") == RunStatus.OK
    report(ok & pl.col("value").is_null(), "are 'ok' but have no value", ["run_id"])
    report(
        ~ok & pl.col("value").is_not_null(),
        "are not 'ok' but carry a value",
        ["run_id", "status"],
    )

    point = [
        *(v.name for v in manifest.variants),
        *(a.name for a in manifest.axes),
        "status",
    ]
    runs = (
        frame
        .group_by("run_id")
        .agg(pl.struct(point).n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if runs.height:
        problems.append(
            f"Runs mix parameter points or statuses: {runs['run_id'].head(5).to_list()}."
        )

    dupes = frame.group_by(_key_columns(manifest)).len().filter(pl.col("len") > 1)
    if dupes.height:
        problems.append(
            f"{dupes.height} duplicate entries, e.g. {dupes.drop('len').head(3).rows()}."
        )

    if problems:
        raise DatasetError(manifest.name, problems)


def to_grid(
    frame: pl.DataFrame,
    manifest: Manifest,
    quantity: str,
    *,
    allow_missing: bool = False,
    **variant: str,
) -> Grid:
    """Arrange one quantity of one variant on its complete rectilinear grid.

    The tensor layout is deterministic: axes in manifest order with ascending
    values, then row and column terminals in manifest order.

    Args:
        frame: Validated results table.
        manifest: Dataset manifest.
        quantity: Quantity name.
        allow_missing: Return NaN for missing or failed points instead of raising.
        **variant: Value for every manifest variant; discrete variants are
            selected explicitly and never interpolated.

    Returns:
        The dense grid of ``quantity``.

    Raises:
        DatasetError: if the selection is ambiguous or the grid is incomplete.
    """
    q = manifest.quantity(quantity)
    declared = {v.name: v.values for v in manifest.variants}
    if set(variant) != set(declared) or any(
        variant[k] not in declared[k] for k in variant
    ):
        raise DatasetError(
            manifest.name,
            [f"Select one value of each variant {declared}, got {variant}."],
        )

    terminals = manifest.conventions.terminals if q.matrix else ()
    selected = frame.filter(
        pl.col("quantity") == quantity,
        *(pl.col(k) == v for k, v in variant.items()),
    )
    axis_names = [a.name for a in manifest.axes]
    full = pl.DataFrame(
        list(product(*(a.values for a in manifest.axes))),
        schema=dict.fromkeys(axis_names, pl.Float64),
        orient="row",
    )
    if q.matrix:
        pairs = pl.DataFrame(
            list(product(terminals, terminals)),
            schema={"row": pl.String, "col": pl.String},
            orient="row",
        )
        full = full.join(pairs, how="cross")
        keys = [*axis_names, "row", "col"]
    else:
        keys = axis_names
    joined = full.join(
        selected, on=keys, how="left", nulls_equal=True, maintain_order="left"
    )

    bad = joined.filter(pl.col("value").is_null())
    if bad.height and not allow_missing:
        sample = bad.select([*axis_names, "status"]).unique(maintain_order=True)
        raise DatasetError(
            manifest.name,
            [
                (
                    f"{sample.height} grid points of {quantity!r} are missing or failed, e.g. "
                    f"{sample.head(5).rows(named=True)}. Pass allow_missing=True to keep them as NaN."
                )
            ],
        )

    real = joined["value"].fill_null(np.nan).to_numpy()
    values = (
        real + 1j * joined["value_imag"].fill_null(0.0).to_numpy()
        if q.complex
        else real
    )
    shape = (
        *(len(a.values) for a in manifest.axes),
        *(len(terminals),) * (2 if q.matrix else 0),
    )
    return Grid(
        dataset=manifest.name,
        quantity=q,
        axes=manifest.axes,
        terminals=tuple(terminals),
        variant=dict(variant),
        values=values.reshape(shape),
    )
