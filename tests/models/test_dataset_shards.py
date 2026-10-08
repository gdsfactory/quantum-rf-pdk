"""Independent workers must reproduce a serial sweep exactly."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st

from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetError,
    DatasetMetadata,
    Quantity,
    QuantityKind,
)
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write

GRID = {
    "width": [4.0, 10.0],
    "cpw_gap": [3.0, 6.0, 12.0],
    "gap": [2.0, 4.0, 8.0, 25.0, 50.0],
}
VARIANTS = {"material": ["air", "silicon"]}
METADATA = DatasetMetadata(
    name="shard_test",
    synthetic=True,
    axes=tuple(Axis(name=name, unit="um") for name in GRID),
    variants=("material",),
    terminals=("lower", "upper"),
    quantities=(Quantity(name="c", kind=QuantityKind.MAXWELL_CAPACITANCE, unit="F"),),
)


def solve(width: float, cpw_gap: float, gap: float, material: str) -> dict:
    factor = 1 if material == "air" else 6
    c = (width + cpw_gap + gap) * factor * 1e-15
    return {"c": np.array([[c, -c / 3], [-c / 3, c]])}


@given(
    axes=st.lists(
        st.lists(st.integers(1, 100), min_size=1, max_size=6, unique=True),
        min_size=1,
        max_size=4,
    ),
)
def test_disjoint_and_exhaustive(axes: list[list[int]]) -> None:
    grid = {f"a{i}": values for i, values in enumerate(axes)}
    expected = set(product(*axes))
    for count in range(1, max(map(len, axes)) + 1):
        seen = set()
        for i in range(count):
            part = partition_grid(grid, shard=i, shards=count)
            points = set(product(*part.values()))
            assert points
            assert not points.intersection(seen)
            seen.update(points)
        assert seen == expected
        assert grid == {f"a{i}": values for i, values in enumerate(axes)}


@pytest.mark.parametrize(("shard", "shards"), [(-1, 2), (2, 2), (0, 0), (0, 6)])
def test_invalid_sharding(shard: int, shards: int) -> None:
    with pytest.raises(ValueError, match="Require"):
        partition_grid(GRID, shard=shard, shards=shards)


def test_merge_matches_serial_and_rejects_bad_shards(tmp_path: Path) -> None:
    parts = []
    for i in range(5):
        grid = partition_grid(GRID, shard=i, shards=5)
        location = tmp_path / f"shard-{i}"
        write(location, METADATA, sweep(METADATA, solve, grid, VARIANTS))
        parts.append(location)
    serial = write(
        tmp_path / "serial", METADATA, sweep(METADATA, solve, GRID, VARIANTS)
    )
    combined = merge(parts, tmp_path / "combined", grid=GRID, variants=VARIANTS)
    assert combined.table.equals(serial.table)
    with pytest.raises(ValueError, match="cover exactly"):
        merge(parts[:-1], tmp_path / "combined", grid=GRID, variants=VARIANTS)
    assert Dataset(tmp_path / "combined").table.equals(serial.table)
    with pytest.raises(DatasetError, match="duplicate"):
        merge([*parts, parts[0]], tmp_path / "duplicate", grid=GRID, variants=VARIANTS)
    changed = METADATA.model_copy(update={"description": "different physics"})
    write(tmp_path / "other", changed, sweep(changed, solve, GRID, VARIANTS))
    with pytest.raises(ValueError, match="metadata differs"):
        merge(
            [*parts, tmp_path / "other"],
            tmp_path / "bad-metadata",
            grid=GRID,
            variants=VARIANTS,
        )


def test_merge_rejects_failed_solve(tmp_path: Path) -> None:
    write(
        tmp_path / "failed", METADATA, sweep(METADATA, lambda **_: None, GRID, VARIANTS)
    )
    with pytest.raises(DatasetError, match="failed"):
        merge([tmp_path / "failed"], tmp_path / "output", grid=GRID, variants=VARIANTS)
