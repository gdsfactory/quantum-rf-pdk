"""Reusable FEM extraction datasets and their JAX lookup.

A dataset is a directory with a ``manifest.toml`` (ordinary Git) and, by
default, ``results/*.parquet`` (Git LFS). See :mod:`qpdk.datasets.table` for the
table layout, :mod:`qpdk.datasets.manifest` for the metadata, and
:mod:`qpdk.datasets.store` for keeping results in a Delta Lake table or an
object store instead.

Requires the ``models`` extra; Delta Lake also needs the ``delta`` extra.
"""

from qpdk.datasets.capacitance import maxwell_to_mutual, maxwell_violations
from qpdk.datasets.interpolation import GridInterpolator
from qpdk.datasets.manifest import (
    SCHEMA_VERSION,
    Artifact,
    Manifest,
    QuantityKind,
    Storage,
    load_manifest,
)
from qpdk.datasets.store import (
    DeltaStore,
    LFSPointerError,
    ParquetParts,
    ResultStore,
)
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
    "Artifact",
    "Dataset",
    "DatasetError",
    "DeltaStore",
    "Grid",
    "GridInterpolator",
    "LFSPointerError",
    "Manifest",
    "ParquetParts",
    "QuantityKind",
    "ResultStore",
    "RunStatus",
    "Storage",
    "load_manifest",
    "maxwell_to_mutual",
    "maxwell_violations",
    "schema",
    "to_grid",
    "validate",
]
