"""Regressions for invalid solver output and append boundaries."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetError,
    DatasetMetadata,
    Quantity,
    QuantityKind,
)
from qpdk.models.datasets.generate import sweep


@pytest.mark.parametrize("shape", [(3, 3), (2, 1), (2,), ()])
def test_solver_matrix_shape_must_match_terminals(shape: tuple[int, ...]) -> None:
    metadata = DatasetMetadata(
        name="shape_fixture",
        synthetic=True,
        axes=(Axis(name="width", unit="um"),),
        terminals=("a", "b"),
        quantities=(
            Quantity(name="c", kind=QuantityKind.MAXWELL_CAPACITANCE, unit="F"),
        ),
    )
    with pytest.raises(ValueError, match="expected shape"):
        sweep(metadata, lambda **_: {"c": np.ones(shape)}, {"width": [1.0]})


def test_real_quantity_rejects_imaginary_values() -> None:
    metadata = DatasetMetadata(
        name="complex_fixture",
        synthetic=True,
        axes=(Axis(name="width", unit="um"),),
        quantities=(Quantity(name="q", kind=QuantityKind.CIRCUIT_PARAMETER, unit="1"),),
    )
    with pytest.raises(ValueError, match="complex=True"):
        sweep(metadata, lambda **_: {"q": 2 + 3j}, {"width": [1.0]})
    declared = metadata.model_copy(
        update={
            "quantities": (metadata.quantities[0].model_copy(update={"complex": True}),)
        }
    )
    result = sweep(declared, lambda **_: {"q": 2 + 3j}, {"width": [1.0]})
    assert result["value"].item() == 2
    assert result["value_imag"].item() == 3
    with pytest.raises(ValueError, match="expected shape"):
        sweep(metadata, lambda **_: {"q": [2.0]}, {"width": [1.0]})


@pytest.mark.parametrize(
    "part", ["../outside", "absolute", "nested/part", "nested\\part", ".", "..", ""]
)
def test_part_names_stay_inside_dataset(tmp_path: Path, part: str) -> None:
    source = Dataset("plate_capacitor_palace")
    if part == "absolute":
        part = str(tmp_path / "outside")
    target = Dataset(tmp_path / "dataset", source.metadata)
    with pytest.raises(ValueError, match="filename stem"):
        target.append(source.table, part=part)
    assert list(tmp_path.rglob("*.parquet")) == []


def test_stale_parquet_handle_rechecks_metadata(tmp_path: Path) -> None:
    source = Dataset("plate_capacitor_palace")
    original = Dataset(tmp_path / "dataset", source.metadata)
    other = source.metadata.model_copy(
        update={
            "axes": tuple(
                axis.model_copy(update={"unit": "m"}) for axis in source.metadata.axes
            )
        }
    )
    stale = Dataset(tmp_path / "dataset", other)
    first = source.table.filter(pl.col("gap") < 7.0)
    second = source.table.filter(pl.col("gap") >= 7.0)
    original.append(first)
    with pytest.raises(DatasetError, match="differs from the metadata stored"):
        stale.append(second)
    assert Dataset(tmp_path / "dataset").table.equals(original.table)
