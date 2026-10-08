"""Polars results table of a FEM extraction dataset.

The table is long-format: one row per parameter point, quantity, and matrix entry.

===============  ===========  ===========================================================
Column           Type         Meaning
===============  ===========  ===========================================================
``run_id``       String       Identity of the solver run that produced the row.
``status``       String       ``ok``, ``failed``, or ``not_converged`` (see :class:`RunStatus`).
<axis>           Float64      One column per metadata axis, in the axis' unit.
<variant>        String       One column per metadata variant.
``quantity``     String       Name of a metadata quantity.
``row``          String       Row terminal label; null for scalar quantities.
``col``          String       Column terminal label; null for scalar quantities.
``value``        Float64      Value (real part for complex quantities); null unless ``ok``.
``value_imag``   Float64      Imaginary part; null for real quantities.
``unit``         String       SI unit; must equal the metadata unit of the quantity.
===============  ===========  ===========================================================

The :class:`~qpdk.models.datasets.metadata.DatasetMetadata` is stored inside the files
(see :mod:`qpdk.models.datasets.store`), so a dataset is self-describing and nothing
but its Parquet parts or Delta table.

.. only:: html

    .. mermaid::

        flowchart LR
            A["Python generator: metadata + grid"] --> B["Palace solves"]
            B --> C["Validated Parquet parts or Delta table"]
            C --> D["Lazy scan: filter + select"]
            D --> E["Dense grid for one quantity and variant"]
            E --> F["JAX interpolation"]
            F --> G["SAX circuit model"]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qpdk.models.datasets.metadata import Axis, DatasetMetadata, Quantity
from qpdk.models.datasets.store import DeltaStore, ParquetParts

if TYPE_CHECKING:
    import polars as pl

DATASETS_PATH = Path(__file__).parent / "data"
"""Curated datasets shipped with qpdk."""


class RunStatus(StrEnum):
    """Outcome of a solver run."""

    OK = "ok"
    FAILED = "failed"
    NOT_CONVERGED = "not_converged"


class DatasetError(ValueError):
    """The dataset is inconsistent with its metadata or cannot be gridded."""

    def __init__(self, dataset: str, problems: list[str]) -> None:
        """Collect every problem into one message."""
        self.problems = problems
        super().__init__(
            f"Dataset {dataset!r} failed validation:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


@dataclass(frozen=True)
class Grid:
    """A quantity on a complete rectilinear grid, ready for interpolation.

    ``values`` has shape ``(*axis sizes, *component shape)`` where the component
    shape is ``(n_terminals, n_terminals)`` for matrix quantities and ``()`` for
    scalars. Axes follow metadata order with ascending ``coords``, terminals
    follow :attr:`~qpdk.models.datasets.metadata.DatasetMetadata.terminals`. Missing or
    failed points are NaN, never zero.
    """

    dataset: str
    quantity: Quantity
    axes: tuple[Axis, ...]
    coords: tuple[np.ndarray, ...]
    terminals: tuple[str, ...]
    variant: dict[str, str]
    values: np.ndarray

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Axis names in tensor order."""
        return tuple(axis.name for axis in self.axes)

    @property
    def domain(self) -> dict[str, tuple[float, float]]:
        """Validated domain of each axis; defaults to the span of its grid values."""
        return {
            axis.name: axis.validated or (float(c[0]), float(c[-1]))
            for axis, c in zip(self.axes, self.coords, strict=True)
        }


def schema(metadata: DatasetMetadata) -> pl.Schema:
    """Column schema of the results table for ``metadata``.

    Returns:
        The Polars schema.
    """
    import polars as pl  # ruff: ignore[import-outside-top-level]

    return pl.Schema({
        "run_id": pl.String,
        "status": pl.String,
        **{axis.name: pl.Float64 for axis in metadata.axes},
        **dict.fromkeys(metadata.variants, pl.String),
        "quantity": pl.String,
        "row": pl.String,
        "col": pl.String,
        "value": pl.Float64,
        "value_imag": pl.Float64,
        "unit": pl.String,
    })


class Dataset:
    """A FEM extraction dataset: Parquet parts or a Delta table with its metadata.

    Args:
        location: Directory of Parquet parts, the name of a dataset shipped in
            :data:`DATASETS_PATH`, or with ``delta=True`` a Delta table path or
            object-store URI such as ``gs://bucket/dataset``.
        metadata: Metadata for a new dataset. For an existing one it is read
            from the files and, if given, must match.
        delta: Store results as a Delta Lake table (``delta`` extra).
        version: Delta table version to read; a pinned dataset is read-only.
        storage_options: Object-store options (credentials, endpoints) for a
            Delta URI. Never written to the dataset.

    Raises:
        FileNotFoundError: if nothing is stored yet and no ``metadata`` is given.
        DatasetError: if ``metadata`` differs from the stored metadata.
    """

    def __init__(
        self,
        location: Path | str,
        metadata: DatasetMetadata | None = None,
        *,
        delta: bool = False,
        version: int | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        """Open the dataset and read its metadata."""
        if delta:
            self.store: ParquetParts | DeltaStore = DeltaStore(
                location, storage_options=storage_options, version=version
            )
        else:
            if version is not None or storage_options is not None:
                msg = "version and storage_options apply to Delta tables only; pass delta=True."
                raise ValueError(msg)
            path = Path(location)
            if not path.is_dir() and (DATASETS_PATH / path).is_dir():
                path = DATASETS_PATH / path
            self.store = ParquetParts(path)
        stored = self.store.metadata()
        if stored is not None and metadata is not None and stored != metadata:
            raise DatasetError(
                stored.name,
                ["The given metadata differs from the metadata stored in the files."],
            )
        if (resolved := stored or metadata) is None:
            msg = f"No dataset at {self.store!r}; pass metadata to create one."
            raise FileNotFoundError(msg)
        self.metadata: DatasetMetadata = resolved

    def __repr__(self) -> str:
        """Show the dataset name and store."""
        return f"Dataset({self.metadata.name!r}, store={self.store!r})"

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the results with the metadata schema.

        Returns:
            Lazy frame over all results; not validated.
        """
        return self.store.scan(schema(self.metadata))

    @cached_property
    def table(self) -> pl.DataFrame:
        """The full results table, validated against the metadata.

        Raises:
            DatasetError: if the stored results do not match the schema.
        """
        import polars as pl  # ruff: ignore[import-outside-top-level]

        try:
            frame = self.scan().collect()
        except (pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError) as error:
            raise DatasetError(
                self.metadata.name, [f"Results do not match the schema: {error}"]
            ) from error
        validate(frame, self.metadata)
        return frame

    def grid(
        self, quantity: str, *, allow_missing: bool = False, **variant: str
    ) -> Grid:
        """Arrange ``quantity`` on its grid; see :func:`to_grid`.

        Returns:
            The dense grid of ``quantity``.

        Raises:
            DatasetError: If the selected rows are invalid or variants are ambiguous.
        """
        self.metadata.quantity(quantity)
        if set(variant) != set(self.metadata.variants):
            raise DatasetError(
                self.metadata.name,
                [
                    f"Select one value of each variant {list(self.metadata.variants)}, got {variant}."
                ],
            )
        import polars as pl  # ruff: ignore[import-outside-top-level]

        selected = (
            self
            .scan()
            .filter(
                pl.col("quantity") == quantity,
                *(pl.col(name) == value for name, value in variant.items()),
            )
            .collect()
        )
        validate(selected, self.metadata)
        return to_grid(
            selected, self.metadata, quantity, allow_missing=allow_missing, **variant
        )

    def append(self, frame: pl.DataFrame, *, part: str | None = None) -> str:
        """Validate ``frame`` together with the stored results, then store it.

        Stored rows are never rewritten. Duplicates against earlier results are
        rejected, so re-running a sweep cannot silently double a point. Parquet
        parts take one writer at a time; a Delta append is one atomic commit and
        the first append creates the table.

        Returns:
            What was written: a part path, or a Delta URI and version.

        Raises:
            DatasetError: If rows duplicate existing entries or mix run identities.
        """
        import polars as pl  # ruff: ignore[import-outside-top-level]

        stored = self.store.metadata()
        if stored is not None and stored != self.metadata:
            raise DatasetError(
                self.metadata.name,
                ["The given metadata differs from the metadata stored in the files."],
            )
        frame = frame.select(schema(self.metadata).names()).cast(
            schema(self.metadata)  # pyrefly: ignore[bad-argument-type]
        )
        validate(frame, self.metadata)
        point_columns = [*self.metadata.variants, *(a.name for a in self.metadata.axes)]
        try:
            existing = self.scan()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            keys = _key_columns(self.metadata)
            overlaps = (
                existing
                .filter(
                    *(
                        pl.col(name).is_in(frame[name].unique().implode())
                        for name in point_columns
                    ),
                )
                .select(keys)
                .join(frame.lazy().select(keys), on=keys, how="semi", nulls_equal=True)
                .limit(5)
                .collect()
            )
            if overlaps.height:
                raise DatasetError(
                    self.metadata.name, [f"duplicate entries, e.g. {overlaps.rows()}."]
                )
            run_columns = ["run_id", "status", *point_columns]
            runs = (
                pl
                .concat([
                    existing.filter(
                        pl.col("run_id").is_in(frame["run_id"].unique().implode())
                    ).select(run_columns),
                    frame.lazy().select(run_columns),
                ])
                .group_by("run_id")
                .agg(pl.struct(["status", *point_columns]).n_unique().alias("points"))
                .filter(pl.col("points") > 1)
                .limit(5)
                .collect()
            )
            if runs.height:
                raise DatasetError(
                    self.metadata.name,
                    [
                        f"Runs mix parameter points or statuses: {runs['run_id'].to_list()}."
                    ],
                )
        written = self.store.write(
            frame.sort(_key_columns(self.metadata), nulls_last=True),
            self.metadata,
            part,
        )
        self.__dict__.pop("table", None)
        return written


def _key_columns(metadata: DatasetMetadata) -> list[str]:
    """Columns that identify one value: variants, axes, quantity, row, col.

    Returns:
        Column names.
    """
    return [
        *metadata.variants,
        *(a.name for a in metadata.axes),
        "quantity",
        "row",
        "col",
    ]


def validate(frame: pl.DataFrame, metadata: DatasetMetadata) -> None:
    """Check a results table against its metadata.

    Detects wrong columns, unknown quantities or statuses, unit mismatches,
    terminals outside the declared ones, non-finite values, inconsistent runs,
    duplicate points, and missing or spurious values. Grid completeness is
    checked per selection by :func:`to_grid`.

    Raises:
        DatasetError: listing every problem found.
    """
    import polars as pl  # ruff: ignore[import-outside-top-level]

    problems: list[str] = []
    expected = schema(metadata)
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
        raise DatasetError(metadata.name, problems)

    def report(mask: pl.Expr, what: str, columns: list[str]) -> None:
        bad = frame.filter(mask)
        if bad.height:
            sample = bad.select(columns).unique(maintain_order=True).head(5).rows()
            problems.append(f"{bad.height} rows {what}, e.g. {sample}.")

    axes = [a.name for a in metadata.axes]
    for column in ["run_id", "status", "quantity", "unit", *metadata.variants, *axes]:
        report(pl.col(column).is_null(), f"have a null {column!r}", ["run_id"])
    for column in axes:
        report(~pl.col(column).is_finite(), f"have a non-finite {column!r}", [column])
    report(
        ~pl.col("value").is_finite() | ~pl.col("value_imag").is_finite(),
        "have a non-finite value; record a failed solve with its status and a null value",
        ["run_id", "status"],
    )

    quantities = {q.name: q for q in metadata.quantities}
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
    terminals = list(metadata.terminals)
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
                & ~(
                    pl.col("row").is_in(terminals).fill_null(False)
                    & pl.col("col").is_in(terminals).fill_null(False)
                ),
                f"of {name!r} have null terminals or terminals outside {terminals}",
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

    ok = pl.col("status") == RunStatus.OK
    report(ok & pl.col("value").is_null(), "are 'ok' but have no value", ["run_id"])
    report(
        ~ok & (pl.col("value").is_not_null() | pl.col("value_imag").is_not_null()),
        "are not 'ok' but carry a value",
        ["run_id", "status"],
    )

    point = [*metadata.variants, *axes, "status"]
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

    dupes = frame.group_by(_key_columns(metadata)).len().filter(pl.col("len") > 1)
    if dupes.height:
        problems.append(
            f"{dupes.height} duplicate entries, e.g. {dupes.drop('len').head(3).rows()}."
        )

    if problems:
        raise DatasetError(metadata.name, problems)


def to_grid(
    frame: pl.DataFrame,
    metadata: DatasetMetadata,
    quantity: str,
    *,
    allow_missing: bool = False,
    **variant: str,
) -> Grid:
    """Arrange one quantity of one variant on its complete rectilinear grid.

    The grid of each axis is the sorted set of values in the selected rows, and
    every combination of them must be present. The layout is deterministic:
    axes in metadata order with ascending values, then row and column terminals
    in metadata order.

    Args:
        frame: Validated results table.
        metadata: Dataset metadata.
        quantity: Quantity name.
        allow_missing: Return NaN for missing or failed points instead of raising.
        **variant: Value for every metadata variant; discrete variants are
            selected explicitly and never interpolated.

    Returns:
        The dense grid of ``quantity``.

    Raises:
        DatasetError: if the selection is ambiguous or empty, a validated domain
            leaves the grid, or the grid is incomplete.
    """
    import polars as pl  # ruff: ignore[import-outside-top-level]

    q = metadata.quantity(quantity)
    if set(variant) != set(metadata.variants):
        raise DatasetError(
            metadata.name,
            [
                f"Select one value of each variant {list(metadata.variants)}, got {variant}."
            ],
        )
    selected = frame.filter(
        pl.col("quantity") == quantity,
        *(pl.col(k) == v for k, v in variant.items()),
    )
    if not selected.height:
        raise DatasetError(
            metadata.name, [f"No rows of {quantity!r} for variant {variant}."]
        )

    axis_names = [a.name for a in metadata.axes]
    coords = tuple(selected[name].unique().sort().to_numpy() for name in axis_names)
    problems = [
        f"Axis {axis.name!r} validated domain {axis.validated} leaves the grid [{c[0]}, {c[-1]}]."
        for axis, c in zip(metadata.axes, coords, strict=True)
        if axis.validated
        and not c[0] <= axis.validated[0] <= axis.validated[1] <= c[-1]
    ]
    if problems:
        raise DatasetError(metadata.name, problems)

    terminals = metadata.terminals if q.matrix else ()
    full = pl.DataFrame(
        list(product(*coords)),
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
            metadata.name,
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
    shape = (*(len(c) for c in coords), *(len(terminals),) * (2 if q.matrix else 0))
    return Grid(
        dataset=metadata.name,
        quantity=q,
        axes=metadata.axes,
        coords=coords,
        terminals=tuple(terminals),
        variant=dict(variant),
        values=values.reshape(shape),
    )
