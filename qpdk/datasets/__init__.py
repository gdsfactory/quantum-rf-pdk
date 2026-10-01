"""Reusable FEM extraction datasets and their JAX lookup.

A dataset is a directory with a ``manifest.toml`` (ordinary Git) and
``results/*.parquet`` (Git LFS). See :mod:`qpdk.datasets.table` for the table
layout and :mod:`qpdk.datasets.manifest` for the metadata.

Requires the ``models`` extra.
"""

from qpdk.datasets.capacitance import maxwell_to_mutual, maxwell_violations
from qpdk.datasets.interpolation import GridInterpolator
from qpdk.datasets.manifest import SCHEMA_VERSION, Manifest, QuantityKind, load_manifest
from qpdk.datasets.table import (
    DATASETS_PATH,
    Dataset,
    DatasetError,
    Grid,
    LFSPointerError,
    RunStatus,
    schema,
    to_grid,
    validate,
)

__all__ = [
    "DATASETS_PATH",
    "SCHEMA_VERSION",
    "Dataset",
    "DatasetError",
    "Grid",
    "GridInterpolator",
    "LFSPointerError",
    "Manifest",
    "QuantityKind",
    "RunStatus",
    "load_manifest",
    "maxwell_to_mutual",
    "maxwell_violations",
    "schema",
    "to_grid",
    "validate",
]
