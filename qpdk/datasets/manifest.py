"""Manifest describing the physical meaning of a FEM extraction dataset.

A dataset directory holds a human-readable ``manifest.toml`` tracked in ordinary
Git, next to a ``results/`` directory of Parquet files tracked in Git LFS. The
manifest defines everything a row in the results table cannot carry on its own:
units, axis order, terminal conventions, geometry, stack, solver, and accuracy.
"""

import hashlib
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
"""Version of the manifest and results-table schema implemented here."""


class _Frozen(BaseModel):
    """Immutable model that rejects unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class QuantityKind(StrEnum):
    """Physical meaning of an extracted quantity.

    The Maxwell capacitance matrix, the mutual (SPICE-style) capacitance matrix,
    and reduced circuit parameters are different quantities and are never mixed.
    """

    MAXWELL_CAPACITANCE = "maxwell_capacitance"
    """Maxwell matrix: positive diagonal, non-positive off-diagonal, w.r.t. the reference ground."""
    MUTUAL_CAPACITANCE = "mutual_capacitance"
    """Mutual matrix: non-negative branch capacitances, diagonal entries to the reference ground."""
    INDUCTANCE = "inductance"
    """Inductance matrix between terminal loops."""
    S_PARAMETERS = "s_parameters"
    """Complex scattering matrix at the declared reference planes."""
    CIRCUIT_PARAMETER = "circuit_parameter"
    """Reduced scalar circuit parameter, e.g. a coupling capacitance."""


_CANONICAL_UNITS = {
    QuantityKind.MAXWELL_CAPACITANCE: "F",
    QuantityKind.MUTUAL_CAPACITANCE: "F",
    QuantityKind.INDUCTANCE: "H",
    QuantityKind.S_PARAMETERS: "1",
}
"""SI unit each physical kind is stored in; lookups apply no unit conversion."""

_MATRIX_KINDS = frozenset({
    QuantityKind.MAXWELL_CAPACITANCE,
    QuantityKind.MUTUAL_CAPACITANCE,
    QuantityKind.INDUCTANCE,
})


class Quantity(_Frozen):
    """A quantity stored in the results table."""

    name: str
    kind: QuantityKind
    unit: str = Field(description="SI unit, e.g. 'F', 'H', or '1' for dimensionless.")
    complex: bool = False
    matrix: bool = Field(
        default=True,
        description="Matrix quantities use 'row'/'col' terminals; scalars leave them null.",
    )
    description: str = ""

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        """Require canonical SI units, and matrices for capacitance and inductance."""
        if (unit := _CANONICAL_UNITS.get(self.kind)) and self.unit != unit:
            msg = f"Quantity {self.name!r} of kind {self.kind.value!r} must be stored in {unit!r}, not {self.unit!r}."
            raise ValueError(msg)
        if self.kind in _MATRIX_KINDS and not self.matrix:
            msg = (
                f"Quantity {self.name!r} of kind {self.kind.value!r} must be a matrix."
            )
            raise ValueError(msg)
        return self


class Axis(_Frozen):
    """A continuous parameter axis of a rectilinear sweep grid."""

    name: str
    unit: str
    values: tuple[float, ...] = Field(min_length=1)
    validated: tuple[float, float] | None = Field(
        default=None,
        description="Validated domain; defaults to the full span of 'values'.",
    )
    description: str = ""

    @model_validator(mode="after")
    def _check_values(self) -> Self:
        """Require finite increasing values and a validated range inside the grid."""
        values = np.asarray(self.values)
        if not np.isfinite([*values, *self.domain]).all():
            msg = f"Axis {self.name!r} values and validated domain must be finite."
            raise ValueError(msg)
        if np.any(np.diff(values) <= 0):
            msg = f"Axis {self.name!r} values must be strictly increasing."
            raise ValueError(msg)
        lo, hi = self.domain
        if lo < values[0] or hi > values[-1] or lo > hi:
            msg = f"Axis {self.name!r} validated domain {self.domain} must lie within its grid."
            raise ValueError(msg)
        return self

    @property
    def domain(self) -> tuple[float, float]:
        """Interval in which lookups are trusted."""
        return self.validated or (self.values[0], self.values[-1])


class Variant(_Frozen):
    """A discrete parameter that is selected, never interpolated."""

    name: str
    values: tuple[str, ...] = Field(min_length=1)
    description: str = ""


class Conventions(_Frozen):
    """Terminal and reference conventions shared by all matrix quantities."""

    terminals: tuple[str, ...] = Field(
        min_length=1,
        description="Ordered terminal labels; defines matrix row/column order.",
    )
    reference_ground: str = Field(
        description="Conductor all potentials are referenced to; not a matrix terminal."
    )
    reference_planes: str = Field(
        default="",
        description="Where port reference planes sit, for S-parameter quantities.",
    )
    terminal_descriptions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_terminals(self) -> Self:
        """Require unique terminals, none of them the reference ground."""
        if len(set(self.terminals)) != len(self.terminals):
            msg = f"Duplicate terminal labels in {self.terminals}."
            raise ValueError(msg)
        if self.reference_ground in self.terminals:
            msg = "The reference ground must not also be a matrix terminal."
            raise ValueError(msg)
        return self


class Recipe(_Frozen):
    """Extraction recipe that produced the dataset."""

    name: str
    version: str


class Geometry(_Frozen):
    """Geometry definition: the cell, the parameters held fixed, and code revisions."""

    cell: str
    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    qpdk_version: str
    gdsfactory_version: str


class Solver(_Frozen):
    """Solver identity and settings."""

    name: str
    version: str
    problem: str = Field(description="e.g. 'electrostatic', 'magnetostatic', 'driven'.")
    settings: dict[str, Any] = Field(default_factory=dict)


class Accuracy(_Frozen):
    """Numerical accuracy and refinement information."""

    relative_tolerance: float = Field(gt=0)
    refinement: str = ""
    notes: str = ""


class Artifact(_Frozen):
    """Checksummed reference to a large artifact kept out of the runtime table."""

    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    description: str = ""

    def verify(self, path: Path) -> None:
        """Check that a local copy of the artifact matches its checksum.

        Raises:
            ValueError: if the SHA-256 of ``path`` differs.
        """
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            msg = f"{path} does not match the checksum of {self.uri!r}."
            raise ValueError(msg)


class Storage(_Frozen):
    """Where the results table lives, if not in the dataset's ``results/`` directory.

    Credentials are never stored here; they are passed when opening the dataset.
    """

    format: Literal["parquet", "delta"]
    uri: str = Field(
        description="Object-store URI, e.g. 'gs://bucket/dataset', or a path relative to the dataset directory."
    )
    version: int | None = Field(
        default=None,
        ge=0,
        description="Delta table version to read; pins the exact rows a model uses.",
    )

    @model_validator(mode="after")
    def _check_version(self) -> Self:
        """Only Delta tables have versions."""
        if self.version is not None and self.format != "delta":
            msg = "Only a 'delta' storage can pin a version."
            raise ValueError(msg)
        return self


class Manifest(_Frozen):
    """Top-level dataset manifest (``manifest.toml``)."""

    schema_version: int
    name: str
    description: str
    synthetic: bool = Field(
        description="True when the values are not real solver output; such data must not be used for design."
    )
    recipe: Recipe
    geometry: Geometry
    stack: dict[str, Any] = Field(description="Layer stack and material definitions.")
    solver: Solver
    accuracy: Accuracy
    conventions: Conventions
    axes: tuple[Axis, ...] = Field(min_length=1)
    variants: tuple[Variant, ...] = ()
    quantities: tuple[Quantity, ...] = Field(min_length=1)
    artifacts: tuple[Artifact, ...] = ()
    storage: Storage | None = Field(
        default=None,
        description="Results location; defaults to the Parquet parts in 'results/'.",
    )

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        """Check the schema version and that names are unique and not reserved."""
        if self.schema_version != SCHEMA_VERSION:
            msg = f"Unsupported schema_version {self.schema_version}; this qpdk reads version {SCHEMA_VERSION}."
            raise ValueError(msg)
        names = [a.name for a in self.axes] + [v.name for v in self.variants]
        if len(set(names)) != len(names):
            msg = f"Axis and variant names must be unique, got {names}."
            raise ValueError(msg)
        quantities = [q.name for q in self.quantities]
        if len(set(quantities)) != len(quantities):
            msg = f"Quantity names must be unique, got {quantities}."
            raise ValueError(msg)
        if reserved := set(names) & set(RESERVED_COLUMNS):
            msg = f"Axis or variant names collide with reserved columns: {sorted(reserved)}."
            raise ValueError(msg)
        return self

    def quantity(self, name: str) -> Quantity:
        """Return the quantity called ``name``."""
        for quantity in self.quantities:
            if quantity.name == name:
                return quantity
        msg = f"Dataset {self.name!r} has no quantity {name!r}; available: {[q.name for q in self.quantities]}."
        raise KeyError(msg)


RESERVED_COLUMNS = (
    "run_id",
    "status",
    "quantity",
    "row",
    "col",
    "value",
    "value_imag",
    "unit",
)
"""Results-table columns that are not sweep parameters."""


def load_manifest(path: Path | str) -> Manifest:
    """Read and validate a ``manifest.toml``."""
    with Path(path).open("rb") as file:
        return Manifest.model_validate(tomllib.load(file))
