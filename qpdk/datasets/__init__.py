"""Reusable FEM extraction datasets and their JAX lookup.

A dataset is a directory of Parquet parts (Git LFS for curated datasets) or a
Delta Lake table, with its :class:`DatasetMetadata` stored inside the files. See
:mod:`qpdk.datasets.table` for the table layout, :mod:`qpdk.datasets.generate`
for producing one, and :mod:`qpdk.datasets.store` for where it lives.

Requires the ``models`` extra; Delta Lake also needs the ``delta`` extra.
"""

from qpdk.datasets.capacitance import maxwell_to_mutual, maxwell_violations
from qpdk.datasets.generate import sweep
from qpdk.datasets.interpolation import GridInterpolator
from qpdk.datasets.metadata import (
    SCHEMA_VERSION,
    Axis,
    DatasetMetadata,
    Quantity,
    QuantityKind,
)
from qpdk.datasets.store import DeltaStore, LFSPointerError, ParquetParts
from qpdk.datasets.table import (
    DATASETS_PATH,
    Dataset,
    DatasetError,
    Grid,
    RunStatus,
    schema,
    to_grid,
    validate,
)

__all__ = [
    "DATASETS_PATH",
    "SCHEMA_VERSION",
    "Axis",
    "Dataset",
    "DatasetError",
    "DatasetMetadata",
    "DeltaStore",
    "Grid",
    "GridInterpolator",
    "LFSPointerError",
    "ParquetParts",
    "Quantity",
    "QuantityKind",
    "RunStatus",
    "maxwell_to_mutual",
    "maxwell_violations",
    "schema",
    "sweep",
    "to_grid",
    "validate",
]
