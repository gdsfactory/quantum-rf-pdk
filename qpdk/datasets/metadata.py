"""Metadata describing the physical meaning of a FEM extraction dataset.

The metadata travels inside the results files, so a dataset is nothing but its
Parquet files (or Delta table): each Parquet part stores it as JSON under the
key-value metadata key :data:`METADATA_KEY`, and a Delta table stores it as
metadata of the ``value`` column. Parameter grids are not repeated here; they
are read from the data.

The metadata holds what a row of the table cannot carry on its own: the units
of the axes, the meaning of each quantity, the terminal order and reference
ground of matrix quantities, and free-form provenance (recipe, geometry, layer
stack, solver, accuracy).
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 2
"""Version of the metadata and results-table schema implemented here."""

METADATA_KEY = "qpdk.dataset"
"""Parquet key-value (and Delta column) metadata key holding the JSON metadata."""

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
    """Complex scattering matrix at the reference planes recorded in the provenance."""
    CIRCUIT_PARAMETER = "circuit_parameter"
    """Reduced scalar circuit parameter, e.g. a coupling capacitance."""


_CANONICAL_UNITS = {
    QuantityKind.MAXWELL_CAPACITANCE: "F",
    QuantityKind.MUTUAL_CAPACITANCE: "F",
    QuantityKind.INDUCTANCE: "H",
    QuantityKind.S_PARAMETERS: "1",
}
"""SI unit each physical kind is stored in; lookups apply no unit conversion."""


class Quantity(_Frozen):
    """A quantity stored in the results table.

    Every kind except :attr:`QuantityKind.CIRCUIT_PARAMETER` is a matrix over the
    dataset terminals.
    """

    name: str
    kind: QuantityKind
    unit: str = Field(description="SI unit, e.g. 'F', 'H', or '1' for dimensionless.")
    complex: bool = False

    @model_validator(mode="after")
    def _check_unit(self) -> Self:
        """Require the canonical SI unit of the kind."""
        if (unit := _CANONICAL_UNITS.get(self.kind)) and self.unit != unit:
            msg = f"Quantity {self.name!r} of kind {self.kind.value!r} must be stored in {unit!r}, not {self.unit!r}."
            raise ValueError(msg)
        return self

    @property
    def matrix(self) -> bool:
        """Whether rows carry ``row``/``col`` terminals."""
        return self.kind != QuantityKind.CIRCUIT_PARAMETER


class Axis(_Frozen):
    """A continuous parameter axis; its grid values come from the data."""

    name: str
    unit: str
    validated: tuple[float, float] | None = Field(
        default=None,
        description="Validated domain; defaults to the span of the gridded values.",
    )

    @model_validator(mode="after")
    def _check_validated(self) -> Self:
        """Require an ordered validated domain."""
        if self.validated is not None and not self.validated[0] <= self.validated[1]:
            msg = (
                f"Axis {self.name!r} validated domain {self.validated} is not ordered."
            )
            raise ValueError(msg)
        return self


class DatasetMetadata(_Frozen):
    """Metadata stored inside the results files of a dataset.

    Args:
        name: Dataset name.
        description: One-line description.
        synthetic: True when the values are not real solver output; such data
            must not be used for design.
        axes: Continuous parameters, in tensor order.
        variants: Discrete parameters (string columns), selected and never
            interpolated.
        quantities: Stored quantities.
        terminals: Ordered terminal labels; defines matrix row and column order.
        reference_ground: Conductor all potentials are referenced to; not a
            matrix terminal.
        provenance: Free-form recipe, geometry, layer stack, solver, and accuracy
            information.
        schema_version: Schema version; must equal :data:`SCHEMA_VERSION`.
    """

    name: str
    description: str = ""
    synthetic: bool = False
    axes: tuple[Axis, ...] = Field(min_length=1)
    variants: tuple[str, ...] = ()
    quantities: tuple[Quantity, ...] = Field(min_length=1)
    terminals: tuple[str, ...] = ()
    reference_ground: str = "ground"
    provenance: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Check the schema version, unique names, and terminals."""
        if self.schema_version != SCHEMA_VERSION:
            msg = f"Unsupported schema_version {self.schema_version}; this qpdk reads version {SCHEMA_VERSION}."
            raise ValueError(msg)
        names = [a.name for a in self.axes] + list(self.variants)
        if len(set(names)) != len(names):
            msg = f"Axis and variant names must be unique, got {names}."
            raise ValueError(msg)
        if reserved := set(names) & set(RESERVED_COLUMNS):
            msg = f"Axis or variant names collide with reserved columns: {sorted(reserved)}."
            raise ValueError(msg)
        quantities = [q.name for q in self.quantities]
        if len(set(quantities)) != len(quantities):
            msg = f"Quantity names must be unique, got {quantities}."
            raise ValueError(msg)
        if len(set(self.terminals)) != len(self.terminals):
            msg = f"Duplicate terminal labels in {self.terminals}."
            raise ValueError(msg)
        if self.reference_ground in self.terminals:
            msg = "The reference ground must not also be a matrix terminal."
            raise ValueError(msg)
        if any(q.matrix for q in self.quantities) and not self.terminals:
            msg = "Matrix quantities need terminals."
            raise ValueError(msg)
        return self

    def quantity(self, name: str) -> Quantity:
        """Return the quantity called ``name``.

        Raises:
            KeyError: if there is no such quantity.
        """
        for quantity in self.quantities:
            if quantity.name == name:
                return quantity
        msg = f"Dataset {self.name!r} has no quantity {name!r}; available: {[q.name for q in self.quantities]}."
        raise KeyError(msg)

    def to_json(self) -> str:
        """Serialize for the file metadata.

        Returns:
            Compact JSON.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse and validate metadata read from a file.

        Returns:
            The validated metadata.
        """
        return cls.model_validate_json(text)
