"""Generate a dataset by sweeping a solver over a rectilinear parameter grid.

Every dataset, synthetic or produced by a FEM solver, is built the same way: a
``solve`` function maps one parameter point to its quantities, and
:func:`sweep` runs it over the grid and lays the results out as the long-format
table of :mod:`qpdk.models.datasets.table`. :func:`write` then stores the table with
its metadata.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from qpdk.models.datasets.metadata import DatasetMetadata
from qpdk.models.datasets.store import ParquetParts
from qpdk.models.datasets.table import Dataset, RunStatus, schema, to_grid, validate

if TYPE_CHECKING:
    import polars as pl

type Solve = Callable[..., Mapping[str, ArrayLike] | None]
"""Solver for one point: keyword axis and variant values in, quantities out.

Matrix quantities are ``(n, n)`` arrays in terminal order, scalars are numbers,
and ``None`` records the run as failed.
"""


def partition_grid(
    grid: Mapping[str, Sequence[float]], *, shard: int, shards: int
) -> dict[str, list[float]]:
    """Select one disjoint rectangular shard of a parameter grid.

    Split the longest axis in round-robin order. Each shard remains a complete
    Cartesian grid, so it can be generated and checked independently. Use the
    same grid and shard count for every worker; shard indices start at zero.
    Scheduling does not change geometry, solver inputs or run identities.

    Returns:
        A grid for one worker, without modifying the input.

    Raises:
        ValueError: If the grid is empty, an axis is empty, or shard settings
            would create an empty worker or an invalid index.
    """
    if not grid or any(not len(values) for values in grid.values()):
        raise ValueError("Every grid axis must contain at least one value")
    axis = max(sorted(grid), key=lambda name: len(grid[name]))
    if not 0 <= shard < shards <= len(grid[axis]):
        raise ValueError(f"Require 0 <= shard < shards <= {len(grid[axis])}")
    return {
        name: list(values[shard::shards] if name == axis else values)
        for name, values in grid.items()
    }


def merge(
    parts: Sequence[Path | str],
    output: Path | str,
    *,
    grid: Mapping[str, Sequence[float]],
    variants: Mapping[str, Sequence[str]] | None = None,
) -> Dataset:
    """Publish independently generated shards after checking the entire sweep.

    Workers write separate dataset directories. Only this single coordinator
    writes the final directory. Metadata must match, points must occur exactly
    once, and every requested point and quantity must have successful results.

    Returns:
        The complete published dataset.

    Raises:
        ValueError: If shards are absent, disagree, overlap, fail, or do not
            cover the requested grid and variants.
    """
    import polars as pl  # ruff: ignore[import-outside-top-level]

    if not parts:
        raise ValueError("At least one shard is required")
    datasets = [Dataset(part) for part in parts]
    metadata = datasets[0].metadata
    if any(dataset.metadata != metadata for dataset in datasets):
        raise ValueError("Shard metadata differs")
    frame = pl.concat([dataset.scan() for dataset in datasets]).collect()
    validate(frame, metadata)
    keys = [*(axis.name for axis in metadata.axes), *metadata.variants]
    values = {**grid, **(variants or {})}
    if set(values) != set(keys):
        raise ValueError(f"Expected grid and variant columns {keys}")
    if any(
        frame[name].unique().sort().to_list() != sorted(axis)
        for name, axis in values.items()
    ) or frame.select(keys).unique().height != prod(
        len(axis) for axis in values.values()
    ):
        raise ValueError("Shards do not cover exactly the requested grid and variants")
    selections = (
        product(*(variants[name] for name in metadata.variants)) if variants else [()]
    )
    for selection in selections:
        selected = dict(zip(metadata.variants, selection, strict=True))
        for quantity in metadata.quantities:
            to_grid(frame, metadata, quantity.name, allow_missing=False, **selected)
    return write(output, metadata, frame)


def sweep(
    metadata: DatasetMetadata,
    solve: Solve,
    grid: Mapping[str, Sequence[float]],
    variants: Mapping[str, Sequence[str]] | None = None,
) -> pl.DataFrame:
    """Run ``solve`` at every point of ``grid`` × ``variants``.

    Each point gets a ``run_id`` derived from the dataset name and the point, so
    re-running a sweep reproduces the same identities.

    Args:
        metadata: Dataset metadata; defines the axes, variants, and quantities.
        solve: Solver for one point.
        grid: Values of every axis.
        variants: Values of every variant.

    Returns:
        Long-format results table, not yet validated or stored.

    Raises:
        ValueError: If the grid or variants differ from metadata, a quantity has
            the wrong shape, or a real quantity contains imaginary values.
    """
    import polars as pl  # ruff: ignore[import-outside-top-level]

    variants = variants or {}
    axes = [a.name for a in metadata.axes]
    if set(grid) != set(axes) or set(variants) != set(metadata.variants):
        msg = f"Sweep needs values for axes {axes} and variants {list(metadata.variants)}."
        raise ValueError(msg)
    names = [*axes, *metadata.variants]
    rows = []
    for values in product(
        *(grid[a] for a in axes), *(variants[v] for v in metadata.variants)
    ):
        point = dict(zip(names, values, strict=True))
        run_id = hashlib.sha256(f"{metadata.name}:{point}".encode()).hexdigest()[:16]
        result = solve(**point)
        status = RunStatus.OK if result is not None else RunStatus.FAILED
        for quantity in metadata.quantities:
            value = None if result is None else np.asarray(result[quantity.name])
            if value is not None:
                shape = (len(metadata.terminals),) * 2 if quantity.matrix else ()
                if value.shape != shape:
                    raise ValueError(
                        f"{quantity.name} at {point}: expected shape {shape}, got {value.shape}"
                    )
                if not quantity.complex and np.any(value.imag != 0):
                    raise ValueError(
                        f"{quantity.name} at {point}: imaginary values require complex=True"
                    )
            pairs = (
                list(product(enumerate(metadata.terminals), repeat=2))
                if quantity.matrix
                else [((None, None), (None, None))]
            )
            for (i, row), (j, col) in pairs:
                entry = (
                    None
                    if value is None
                    else complex(value[i, j] if quantity.matrix else value)
                )
                rows.append({
                    "run_id": run_id,
                    "status": status.value,
                    **point,
                    "quantity": quantity.name,
                    "row": row,
                    "col": col,
                    "value": None if entry is None else entry.real,
                    "value_imag": entry.imag
                    if entry is not None and quantity.complex
                    else None,
                    "unit": quantity.unit,
                })
    return pl.DataFrame(rows, schema=schema(metadata))


def write(
    location: Path | str, metadata: DatasetMetadata, frame: pl.DataFrame
) -> Dataset:
    """Replace the dataset at ``location`` with ``frame`` as one part.

    Meant for regenerating a curated dataset from scratch; use
    :meth:`~qpdk.models.datasets.Dataset.append` to grow an existing one. The
    new part is written to a sibling staging directory and swapped in only once
    it is complete, so a failure leaves the existing dataset untouched.

    Returns:
        The written dataset.

    Raises:
        FileExistsError: if ``location`` holds anything other than dataset parts.
        OSError: if publishing fails; the existing dataset is then kept.
    """
    validate(frame, metadata)
    location = Path(location)
    if location.exists():
        foreign = [
            p.name for p in location.iterdir() if p.suffix != ".parquet" or p.is_dir()
        ]
        try:
            ParquetParts(location).metadata()
        except ValueError:
            foreign.append("*.parquet without dataset metadata")
        if foreign:
            msg = (
                f"{location} is not a dataset directory; refusing to replace {foreign}."
            )
            raise FileExistsError(msg)
    location.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    staging = location.with_name(f".{location.name}.{tag}.tmp")
    backup = location.with_name(f".{location.name}.{tag}.old")
    try:
        Dataset(staging, metadata).append(frame, part="part-0000")
        if location.exists():
            location.rename(backup)
        try:
            staging.rename(location)
        except OSError:
            if backup.exists():
                backup.rename(location)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    return Dataset(location, metadata)
