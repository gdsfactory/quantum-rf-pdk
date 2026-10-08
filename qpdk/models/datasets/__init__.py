"""Reusable FEM extraction datasets and their JAX lookup.

A dataset is a directory of Parquet parts (Git LFS for curated datasets) or a
Delta Lake table, with its :class:`DatasetMetadata` stored inside the files. See
:mod:`qpdk.models.datasets.table` for the table layout, :mod:`qpdk.models.datasets.generate`
for producing one, and :mod:`qpdk.models.datasets.store` for where it lives.

Requires the ``models`` extra; Delta Lake also needs the ``delta`` extra.
"""

from qpdk.models.datasets.capacitance import (
    NonPhysicalMatrixError,
    check_maxwell,
    maxwell_to_mutual,
)
from qpdk.models.datasets.generate import sweep
from qpdk.models.datasets.interpolation import GridInterpolator
from qpdk.models.datasets.metadata import (
    SCHEMA_VERSION,
    Axis,
    DatasetMetadata,
    Quantity,
    QuantityKind,
)
from qpdk.models.datasets.models import cpw_coupling_model
from qpdk.models.datasets.store import DeltaStore, LFSPointerError, ParquetParts
from qpdk.models.datasets.table import (
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
    "NonPhysicalMatrixError",
    "ParquetParts",
    "Quantity",
    "QuantityKind",
    "RunStatus",
    "check_maxwell",
    "cpw_coupling_model",
    "maxwell_to_mutual",
    "schema",
    "sweep",
    "to_grid",
    "validate",
]
