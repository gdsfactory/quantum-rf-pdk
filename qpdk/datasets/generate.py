"""Generate a dataset by sweeping a solver over a rectilinear parameter grid.

Every dataset, synthetic or produced by a FEM solver, is built the same way: a
``solve`` function maps one parameter point to its quantities, and
:func:`sweep` runs it over the grid and lays the results out as the long-format
table of :mod:`qpdk.datasets.table`. :func:`write` then stores the table with
its metadata.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qpdk.datasets.metadata import DatasetMetadata
from qpdk.datasets.table import Dataset, RunStatus, schema

if TYPE_CHECKING:
    import polars as pl
    from jax.typing import ArrayLike

type Solve = Callable[..., Mapping[str, ArrayLike] | None]
"""Solver for one point: keyword axis and variant values in, quantities out.

Matrix quantities are ``(n, n)`` arrays in terminal order, scalars are numbers,
and ``None`` records the run as failed.
"""


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
        ValueError: if ``grid`` or ``variants`` do not match the metadata.
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
    """Replace the Parquet parts at ``location`` with ``frame`` as one part.

    Meant for regenerating a curated dataset from scratch; use
    :meth:`~qpdk.datasets.Dataset.append` to grow an existing one.

    Returns:
        The written dataset.
    """
    for part in Path(location).glob("*.parquet"):
        part.unlink()
    dataset = Dataset(location, metadata)
    dataset.append(frame, part="part-0000")
    return dataset
