"""Tests for qpdk.models.datasets: metadata, results table, generator, and JAX lookup."""

import dataclasses
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import sax
import xarray as xr
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError
from scipy.interpolate import RegularGridInterpolator

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.datasets import (
    DATASETS_PATH,
    Axis,
    Dataset,
    DatasetError,
    DatasetMetadata,
    GridInterpolator,
    LFSPointerError,
    NonPhysicalMatrixError,
    ParquetParts,
    Quantity,
    QuantityKind,
    RunStatus,
    check_maxwell,
    maxwell_to_mutual,
    sweep,
    to_grid,
    validate,
)
from qpdk.models.datasets.generate import write
from qpdk.models.datasets.metadata import METADATA_KEY
from qpdk.models.datasets.plate_capacitor import GRID as PLATE_CAPACITOR_GRID, NAME
from qpdk.models.generic import capacitor

PLATE_CAPACITOR = Dataset(NAME).metadata
CENTER = {"length": 120.0, "width": 10.0, "gap": 7.0}


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset(NAME)


@pytest.fixture(scope="module")
def interp(dataset: Dataset) -> GridInterpolator:
    return GridInterpolator(dataset.grid("maxwell_capacitance", cross_section="cpw"))


@pytest.fixture
def dataset_copy(tmp_path: Path) -> Dataset:
    target = tmp_path / NAME
    shutil.copytree(DATASETS_PATH / NAME, target)
    return Dataset(target)


POINTS = st.fixed_dictionaries({
    name: st.floats(min(values), max(values), allow_nan=False)
    for name, values in PLATE_CAPACITOR_GRID.items()
})


def _maxwell(**point: float) -> np.ndarray:
    rows = Dataset(NAME).table.filter(_at(**point)).sort("row", "col")
    return rows["value"].to_numpy().reshape(2, 2)


def _at(**point: float) -> pl.Expr:
    """Select one grid point; grid values are stored exactly, so ``is_in`` is safe."""
    return pl.all_horizontal(pl.col(k).is_in([v]) for k, v in point.items())


class TestMetadata:
    """The metadata lives in the results files and is validated on load."""

    @staticmethod
    def test_shipped_dataset_is_palace_output(dataset: Dataset) -> None:
        assert dataset.metadata == PLATE_CAPACITOR
        assert not dataset.metadata.synthetic
        assert dataset.metadata.provenance["solver"] == "Palace"

    @staticmethod
    def test_every_part_carries_the_metadata(dataset: Dataset) -> None:
        for part in dataset.store.files:  # pyrefly: ignore[missing-attribute]
            stored = pl.read_parquet_metadata(part)[METADATA_KEY]
            assert DatasetMetadata.from_json(stored) == PLATE_CAPACITOR

    @staticmethod
    def test_json_round_trip() -> None:
        assert DatasetMetadata.from_json(PLATE_CAPACITOR.to_json()) == PLATE_CAPACITOR

    @staticmethod
    def test_given_metadata_must_match_stored(dataset_copy: Dataset) -> None:
        other = PLATE_CAPACITOR.model_copy(update={"description": "changed"})
        with pytest.raises(DatasetError, match="differs from the metadata stored"):
            Dataset(dataset_copy.store.location, other)  # pyrefly: ignore[missing-attribute]

    @staticmethod
    def test_new_dataset_needs_metadata(tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pass metadata"):
            Dataset(tmp_path / "empty")

    @staticmethod
    def test_delta_options_need_delta(tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pass delta=True"):
            Dataset(tmp_path, PLATE_CAPACITOR, version=0)

    @staticmethod
    def test_parts_with_different_metadata_are_rejected(dataset_copy: Dataset) -> None:
        location = dataset_copy.store.location  # pyrefly: ignore[missing-attribute]
        other = PLATE_CAPACITOR.model_copy(update={"description": "changed"})
        dataset_copy.table.head(0).write_parquet(
            location / "part-0001.parquet", metadata={METADATA_KEY: other.to_json()}
        )
        with pytest.raises(ValueError, match="different dataset metadata"):
            Dataset(location)

    @staticmethod
    def test_rejects_wrong_unit() -> None:
        with pytest.raises(ValidationError, match="must be stored in 'F'"):
            Quantity(name="c", kind=QuantityKind.MAXWELL_CAPACITANCE, unit="pF")

    @staticmethod
    def test_rejects_unordered_domain() -> None:
        with pytest.raises(ValidationError, match="not ordered"):
            Axis(name="gap", unit="um", validated=(10.0, 2.0))

    @staticmethod
    def test_rejects_inconsistent_names() -> None:
        for update, message in [
            ({"variants": ("gap",)}, "must be unique"),
            ({"variants": ("value",)}, "reserved columns"),
            ({"terminals": ("o1", "o1")}, "Duplicate terminal"),
            (
                {"terminals": ("o1", PLATE_CAPACITOR.reference_ground)},
                "reference ground",
            ),
            ({"terminals": ()}, "need terminals"),
            ({"schema_version": 2}, "Unsupported schema_version"),
        ]:
            with pytest.raises(ValidationError, match=message):
                DatasetMetadata.model_validate(PLATE_CAPACITOR.model_dump() | update)

    @staticmethod
    def test_unknown_quantity() -> None:
        with pytest.raises(KeyError, match="no quantity 'capacitance'"):
            PLATE_CAPACITOR.quantity("capacitance")


class TestValidation:
    """``validate`` reports every inconsistency between table and metadata."""

    @staticmethod
    def _rejects(dataset: Dataset, cases: list) -> None:
        for mutate, message in cases:
            with pytest.raises(DatasetError, match=message):
                validate(mutate(dataset.table), dataset.metadata)

    @staticmethod
    def test_shipped_table_is_valid(dataset: Dataset) -> None:
        validate(dataset.table, dataset.metadata)
        assert dataset.table.height == 3 * 3 * 3 * 2 * 2

    def test_rejects_wrong_columns(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (lambda f: f.drop("unit"), r"missing \['unit'\]"),
                (lambda f: f.with_columns(pl.col("gap").cast(pl.Int64)), "wrong dtype"),
            ],
        )

    def test_rejects_nulls(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (
                    lambda f, c=c, t=t: f.with_columns(pl.lit(None, t).alias(c)),
                    f"null '{c}'",
                )
                for c, t in [
                    ("status", pl.String),
                    ("unit", pl.String),
                    ("quantity", pl.String),
                    ("length", pl.Float64),
                    ("cross_section", pl.String),
                ]
            ],
        )

    def test_rejects_bad_values(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (lambda f: f.with_columns(unit=pl.lit("pF")), "not in 'F'"),
                (
                    lambda f: f.with_columns(value=pl.lit(float("nan"))),
                    "non-finite value",
                ),
                (
                    lambda f: f.with_columns(value=pl.lit(float("inf"))),
                    "non-finite value",
                ),
                (
                    lambda f: f.with_columns(length=pl.lit(float("inf"))),
                    "non-finite 'length'",
                ),
                (lambda f: f.with_columns(value_imag=pl.lit(0.0)), "imaginary part"),
                (lambda f: f.with_columns(quantity=pl.lit("c")), "unknown quantity"),
            ],
        )

    def test_rejects_bad_terminals(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (lambda f: f.with_columns(row=pl.lit("o3")), "terminals outside"),
                (
                    lambda f: f.with_columns(row=pl.lit(None, pl.String)),
                    "null terminals",
                ),
            ],
        )

    def test_rejects_inconsistent_status(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (lambda f: f.with_columns(status=pl.lit("done")), "unknown status"),
                (
                    lambda f: f.with_columns(value=pl.lit(None, pl.Float64)),
                    "'ok' but have no value",
                ),
                (
                    lambda f: f.with_columns(status=pl.lit("failed")),
                    "not 'ok' but carry a value",
                ),
                (
                    lambda f: f.with_columns(
                        status=pl.lit("failed"),
                        value=pl.lit(None, pl.Float64),
                        value_imag=pl.lit(0.0),
                    ),
                    "not 'ok' but carry a value",
                ),
            ],
        )

    def test_rejects_inconsistent_runs(self, dataset: Dataset) -> None:
        self._rejects(
            dataset,
            [
                (lambda f: pl.concat([f, f.head(1)]), "duplicate"),
                (
                    lambda f: f.with_columns(run_id=pl.lit("same")),
                    "mix parameter points",
                ),
            ],
        )


class TestGrid:
    """``to_grid`` derives the grid from the data and refuses to guess."""

    @staticmethod
    def test_layout_is_deterministic(dataset: Dataset) -> None:
        """Axes in metadata order with ascending values, then terminals in order."""
        grid = dataset.grid("maxwell_capacitance", cross_section="cpw")
        assert grid.values.shape == (3, 3, 3, 2, 2)
        for coords, values in zip(
            grid.coords, PLATE_CAPACITOR_GRID.values(), strict=True
        ):
            np.testing.assert_array_equal(coords, values)
        for idx in [(0, 0, 0), (1, 1, 1), (2, 2, 2)]:
            point = {
                a: float(c[i])
                for a, c, i in zip(grid.axis_names, grid.coords, idx, strict=True)
            }
            np.testing.assert_allclose(grid.values[idx], _maxwell(**point), rtol=1e-12)

    @staticmethod
    def test_independent_of_row_order(dataset: Dataset) -> None:
        shuffled = dataset.table.sample(fraction=1.0, shuffle=True, seed=1)
        a = to_grid(
            shuffled, dataset.metadata, "maxwell_capacitance", cross_section="cpw"
        )
        b = dataset.grid("maxwell_capacitance", cross_section="cpw")
        np.testing.assert_array_equal(a.values, b.values)

    @staticmethod
    def test_incomplete_grid_is_explicit(dataset: Dataset) -> None:
        frame = dataset.table.filter(~_at(length=40.0, width=10.0, gap=7.0))
        validate(frame, dataset.metadata)
        with pytest.raises(DatasetError, match=r"1 grid points .* missing or failed"):
            to_grid(frame, dataset.metadata, "maxwell_capacitance", cross_section="cpw")
        grid = to_grid(
            frame,
            dataset.metadata,
            "maxwell_capacitance",
            allow_missing=True,
            cross_section="cpw",
        )
        assert np.isnan(grid.values[0, 1, 1]).all()
        assert np.isfinite(
            np.delete(grid.values.reshape(-1, 4), 1 * 3 + 1, axis=0)
        ).all()

    @staticmethod
    def test_off_grid_point_makes_the_grid_incomplete(dataset: Dataset) -> None:
        frame = dataset.table.with_columns(
            length=pl
            .when(_at(length=40.0, width=10.0, gap=7.0))
            .then(41.0)
            .otherwise("length")
        )
        with pytest.raises(DatasetError, match="missing or failed"):
            to_grid(frame, dataset.metadata, "maxwell_capacitance", cross_section="cpw")

    @staticmethod
    def test_failed_runs_are_never_interpolated(dataset: Dataset) -> None:
        point = _at(length=120.0, width=20.0, gap=10.0)
        frame = dataset.table.with_columns(
            status=pl
            .when(point)
            .then(pl.lit(RunStatus.FAILED.value))
            .otherwise("status"),
            value=pl.when(point).then(None).otherwise("value"),
        )
        validate(frame, dataset.metadata)
        with pytest.raises(DatasetError, match="'status': 'failed'"):
            to_grid(frame, dataset.metadata, "maxwell_capacitance", cross_section="cpw")
        grid = to_grid(
            frame,
            dataset.metadata,
            "maxwell_capacitance",
            allow_missing=True,
            cross_section="cpw",
        )
        f = GridInterpolator(grid)
        assert np.isnan(f(length=110.0, width=19.0, gap=9.0)).all()
        assert np.isfinite(f(length=100.0, width=10.0, gap=5.0)).all()

    @staticmethod
    def test_variant_must_be_selected(dataset: Dataset) -> None:
        with pytest.raises(DatasetError, match="Select one value of each variant"):
            dataset.grid("maxwell_capacitance")
        with pytest.raises(DatasetError, match="No rows"):
            dataset.grid("maxwell_capacitance", cross_section="nope")

    @staticmethod
    def test_validated_domain_must_lie_in_the_grid(dataset: Dataset) -> None:
        axes = list(PLATE_CAPACITOR.axes)
        axes[0] = axes[0].model_copy(update={"validated": (10.0, 100.0)})
        metadata = PLATE_CAPACITOR.model_copy(update={"axes": tuple(axes)})
        with pytest.raises(DatasetError, match="leaves the grid"):
            to_grid(dataset.table, metadata, "maxwell_capacitance", cross_section="cpw")


class TestStorage:
    """Append-only Parquet parts and Git LFS."""

    @staticmethod
    def test_unresolved_lfs_pointer_has_useful_error(dataset_copy: Dataset) -> None:
        (part,) = dataset_copy.store.files  # pyrefly: ignore[missing-attribute]
        part.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:"
            + "0" * 64
            + "\nsize 9733\n"
        )
        with pytest.raises(
            LFSPointerError,
            match=r"git lfs pull --include \"qpdk/models/datasets/data/\*\*\"",
        ):
            dataset_copy.scan()

    @staticmethod
    def test_append_writes_new_part_and_rejects_duplicates(
        dataset_copy: Dataset,
    ) -> None:
        original = dataset_copy.table
        new = original.with_columns(
            run_id=pl.col("run_id") + "-wide",
            cross_section=pl.lit("cpw_wide"),
            value=pl.col("value") * 1.1,
        )
        path = dataset_copy.append(new, part="part-0001")
        assert Path(path).name == "part-0001.parquet"
        assert dataset_copy.table.height == 2 * original.height
        np.testing.assert_allclose(
            dataset_copy.grid("maxwell_capacitance", cross_section="cpw_wide").values,
            1.1 * dataset_copy.grid("maxwell_capacitance", cross_section="cpw").values,
        )
        with pytest.raises(DatasetError, match="duplicate"):
            dataset_copy.append(new.head(4), part="part-0002")
        with pytest.raises(FileExistsError):
            dataset_copy.append(new.head(0), part="part-0001")
        assert [p.name for p in dataset_copy.store.files] == [  # pyrefly: ignore[missing-attribute]
            "part-0000.parquet",
            "part-0001.parquet",
        ]

    @staticmethod
    def test_parts_with_wrong_schema_raise_dataset_error(dataset_copy: Dataset) -> None:
        (part,) = dataset_copy.store.files  # pyrefly: ignore[missing-attribute]
        pl.read_parquet(part).with_columns(pl.col("gap").cast(pl.Int64)).write_parquet(
            part, metadata={METADATA_KEY: PLATE_CAPACITOR.to_json()}
        )
        with pytest.raises(DatasetError, match="do not match the schema"):
            _ = Dataset(part.parent).table


class TestGenerate:
    """``sweep`` and ``write`` produce the shipped dataset and any other."""

    @staticmethod
    def test_invalid_regeneration_preserves_existing_parts(
        dataset_copy: Dataset,
    ) -> None:
        store = dataset_copy.store
        assert isinstance(store, ParquetParts)
        parts = {path: path.read_bytes() for path in store.files}
        invalid = dataset_copy.table.with_columns(value=pl.lit(float("nan")))
        with pytest.raises(DatasetError, match="non-finite value"):
            write(store.location, dataset_copy.metadata, invalid)
        assert {path: path.read_bytes() for path in store.files} == parts
        assert Dataset(store.location).table.equals(dataset_copy.table)

    @staticmethod
    def test_write_refuses_unrelated_files(dataset: Dataset, tmp_path: Path) -> None:
        unrelated = tmp_path / "my_results.parquet"
        pl.DataFrame({"x": [1]}).write_parquet(unrelated)
        (tmp_path / "notes.txt").write_text("keep")
        with pytest.raises(FileExistsError, match="not a dataset directory"):
            write(tmp_path, dataset.metadata, dataset.table)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "my_results.parquet",
            "notes.txt",
        ]

    @staticmethod
    def test_failed_publish_preserves_existing_parts(
        dataset_copy: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = dataset_copy.store
        assert isinstance(store, ParquetParts)
        parts = {path.name: path.read_bytes() for path in store.files}

        def no_hard_links(*_: object) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr("qpdk.models.datasets.store.os.link", no_hard_links)
        with pytest.raises(PermissionError):
            write(store.location, dataset_copy.metadata, dataset_copy.table)
        assert {path.name: path.read_bytes() for path in store.files} == parts
        assert sorted(p.name for p in store.location.parent.iterdir()) == [NAME]

    @staticmethod
    def test_sweep_round_trips_stored_solves(dataset: Dataset, tmp_path: Path) -> None:
        frame = sweep(
            PLATE_CAPACITOR,
            lambda **point: {
                "maxwell_capacitance": _maxwell(**{
                    k: v for k, v in point.items() if k != "cross_section"
                })
            },
            PLATE_CAPACITOR_GRID,
            {"cross_section": ["cpw"]},
        )
        regenerated = write(tmp_path / NAME, PLATE_CAPACITOR, frame)
        np.testing.assert_array_equal(
            regenerated.grid("maxwell_capacitance", cross_section="cpw").values,
            dataset.grid("maxwell_capacitance", cross_section="cpw").values,
        )

    @staticmethod
    def test_failed_solve_is_recorded() -> None:
        def solve(length, width, gap, cross_section):  # ruff: ignore[unused-function-argument]
            return (
                None
                if gap > 7
                else {
                    "maxwell_capacitance": _maxwell(length=length, width=width, gap=gap)
                }
            )

        frame = sweep(
            PLATE_CAPACITOR, solve, PLATE_CAPACITOR_GRID, {"cross_section": ["cpw"]}
        )
        validate(frame, PLATE_CAPACITOR)
        failed = frame.filter(pl.col("status") == RunStatus.FAILED)
        assert failed.height == 3 * 3 * 4
        assert failed["value"].is_null().all()

    @staticmethod
    def test_grid_must_match_the_axes() -> None:
        with pytest.raises(ValueError, match="Sweep needs values"):
            sweep(PLATE_CAPACITOR, lambda **_point: None, {"length": [1.0]})

    @staticmethod
    def test_complex_scalar_quantity() -> None:
        """Complex scalars round-trip through the same path as matrices."""
        metadata = DatasetMetadata(
            name="s21_toy",
            axes=(Axis(name="frequency", unit="Hz"), Axis(name="gap", unit="um")),
            quantities=(
                Quantity(
                    name="s21",
                    kind=QuantityKind.CIRCUIT_PARAMETER,
                    unit="1",
                    complex=True,
                ),
            ),
        )
        frame = sweep(
            metadata,
            lambda frequency, gap: {"s21": np.exp(1j * frequency / 1e9 * gap)},
            {"frequency": [4e9, 5e9, 6e9], "gap": [2.0, 4.0]},
        )
        validate(frame, metadata)
        grid = to_grid(frame, metadata, "s21")
        assert grid.values.shape == (3, 2)
        assert np.iscomplexobj(grid.values)
        f = GridInterpolator(grid)
        np.testing.assert_allclose(f(frequency=5e9, gap=4.0), np.exp(20j), rtol=1e-12)
        np.testing.assert_allclose(
            f(frequency=4.5e9, gap=2.0), (np.exp(8j) + np.exp(10j)) / 2, rtol=1e-12
        )


class TestGridInterpolator:
    """The lookup wraps ``jax.scipy.interpolate.RegularGridInterpolator``."""

    @staticmethod
    @settings(deadline=None, max_examples=25)
    @given(idx=st.tuples(st.integers(0, 2), st.integers(0, 2), st.integers(0, 2)))
    def test_reproduces_grid_points(
        interp: GridInterpolator, idx: tuple[int, int, int]
    ) -> None:
        point = {
            a: float(c[i])
            for a, c, i in zip(interp.axis_names, interp.grid.coords, idx, strict=True)
        }
        np.testing.assert_allclose(interp(**point), interp.grid.values[idx], rtol=1e-12)

    @staticmethod
    def test_matches_scipy_reference(interp: GridInterpolator) -> None:
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            rng.uniform(c[0], c[-1], 200) for c in interp.grid.coords
        ])
        reference = RegularGridInterpolator(interp.grid.coords, interp.grid.values)(pts)
        ours = interp(**{a: pts[:, i] for i, a in enumerate(interp.axis_names)})
        np.testing.assert_allclose(ours, reference, rtol=1e-12)

    @staticmethod
    def test_held_out_points_against_palace(interp: GridInterpolator) -> None:
        evidence = json.loads(
            (
                Path(__file__).parent / "data/plate_capacitor_palace_validation.json"
            ).read_text(encoding="utf-8")
        )
        for solved in evidence["heldout"]:
            point = {name: solved[name] for name in PLATE_CAPACITOR_GRID}
            np.testing.assert_allclose(interp(**point), solved["actual_F"], rtol=0.06)

    @staticmethod
    @settings(deadline=None, max_examples=25)
    @given(point=POINTS)
    def test_interpolated_matrix_is_physical(
        interp: GridInterpolator, point: dict[str, float]
    ) -> None:
        check_maxwell(interp(**point))

    @staticmethod
    def test_shapes_broadcast(interp: GridInterpolator) -> None:
        assert interp(**CENTER).shape == (2, 2)
        out = interp(
            length=jnp.ones((3, 1)) * 100.0, width=jnp.array([5.0, 10.0]), gap=7.0
        )
        assert out.shape == (3, 2, 2, 2)

    @staticmethod
    def test_out_of_range_is_nan_not_clamped(interp: GridInterpolator) -> None:
        out = interp(length=jnp.array([10.0, 100.0, 400.0]), width=10.0, gap=5.0)
        assert out.shape == (3, 2, 2)
        assert np.isnan(out[0]).all()
        assert np.isfinite(out[1]).all()
        assert np.isnan(out[2]).all()
        np.testing.assert_array_equal(
            interp.in_domain(length=[10.0, 100.0, 400.0], width=10.0, gap=5.0),
            [False, True, False],
        )

    @staticmethod
    def test_clip_is_opt_in(interp: GridInterpolator) -> None:
        f = GridInterpolator(interp.grid, out_of_range="clip")
        np.testing.assert_allclose(
            f(length=400.0, width=10.0, gap=5.0), f(length=120.0, width=10.0, gap=5.0)
        )
        with pytest.raises(ValueError, match="out_of_range"):
            GridInterpolator(f.grid, out_of_range="extrapolate")  # pyrefly: ignore[bad-argument-type]

    @staticmethod
    def test_validated_domain_narrower_than_grid(interp: GridInterpolator) -> None:
        axes = list(interp.grid.axes)
        axes[0] = axes[0].model_copy(update={"validated": (50.0, 110.0)})
        f = GridInterpolator(dataclasses.replace(interp.grid, axes=tuple(axes)))
        assert f.domain["length"] == (50.0, 110.0)
        assert np.isnan(f(length=30.0, width=10.0, gap=5.0)).all()
        assert np.isfinite(f(length=50.0, width=10.0, gap=5.0)).all()

    @staticmethod
    def test_single_point_axis(dataset: Dataset) -> None:
        """An axis swept at one value is matched exactly and NaN elsewhere."""
        frame = dataset.table.filter(pl.col("width").is_between(9.5, 10.5))
        grid = to_grid(
            frame, dataset.metadata, "maxwell_capacitance", cross_section="cpw"
        )
        f = GridInterpolator(grid)
        np.testing.assert_allclose(f(**CENTER), _maxwell(**CENTER), rtol=1e-12)
        assert np.isnan(f(**{**CENTER, "width": 11.0})).all()

    @staticmethod
    def test_requires_every_axis(interp: GridInterpolator) -> None:
        with pytest.raises(TypeError, match="Expected exactly the axes"):
            interp(length=100.0, width=10.0)

    @staticmethod
    def test_jit_vmap_grad(interp: GridInterpolator) -> None:
        def c_mutual(length, width, gap):
            return -interp(length=length, width=width, gap=gap)[..., 0, 1]

        batch = jnp.linspace(45.0, 115.0, 7)
        jitted = jax.jit(c_mutual)(batch, 10.0, 5.0)
        np.testing.assert_allclose(jitted, c_mutual(batch, 10.0, 5.0), rtol=1e-12)
        vmapped = jax.vmap(c_mutual, in_axes=(0, None, None))(batch, 10.0, 5.0)
        np.testing.assert_allclose(vmapped, jitted, rtol=1e-12)
        assert jitted.dtype == jnp.float64

        grad = jax.jit(jax.grad(c_mutual, argnums=(0, 2)))(90.0, 10.0, 5.0)
        h = 1e-3
        fd_length = (c_mutual(90.0 + h, 10.0, 5.0) - c_mutual(90.0 - h, 10.0, 5.0)) / (
            2 * h
        )
        fd_gap = (c_mutual(90.0, 10.0, 5.0 + h) - c_mutual(90.0, 10.0, 5.0 - h)) / (
            2 * h
        )
        np.testing.assert_allclose(grad, (fd_length, fd_gap), rtol=1e-6)
        assert grad[1] < 0  # wider gap, less capacitance

    @staticmethod
    def test_agrees_with_sax_interpolate_xarray_in_domain(
        interp: GridInterpolator,
    ) -> None:
        """Same numbers as SAX inside the domain; SAX clamps outside, which we do not."""
        grid = interp.grid
        xarr = xr.DataArray(
            grid.values.reshape(*grid.values.shape[:3], 4),
            coords={
                **dict(zip(grid.axis_names, grid.coords, strict=True)),
                "targets": np.array(["c11", "c12", "c21", "c22"], dtype=object),
            },
        )
        point = {"length": 93.0, "width": 12.5, "gap": 8.2}
        sax_out = sax.interpolate_xarray(xarr, **point)
        np.testing.assert_allclose(
            interp(**point).reshape(4),
            [sax_out[k] for k in ["c11", "c12", "c21", "c22"]],
            rtol=1e-12,
        )
        outside = {**point, "length": 1000.0}
        assert np.isfinite(sax.interpolate_xarray(xarr, **outside)["c12"])
        assert np.isnan(interp(**outside)[0, 1])

    @staticmethod
    def test_gradient_at_domain_edges(interp: GridInterpolator) -> None:
        """The gradient at both ends of an axis is the slope of the edge cell, not zero."""
        for axis, values in zip(interp.axis_names, interp.grid.coords, strict=True):

            def c12(x, axis=axis):
                return interp(**{**CENTER, axis: x})[0, 1]

            for edge, neighbour in [(values[0], values[1]), (values[-1], values[-2])]:
                slope = (c12(edge) - c12(neighbour)) / (edge - neighbour)
                np.testing.assert_allclose(jax.grad(c12)(edge), slope, rtol=1e-9)
                assert slope != 0


class TestCapacitance:
    """Maxwell matrix checks and conversion."""

    @staticmethod
    @pytest.mark.parametrize(
        ("maxwell", "problems"),
        [
            ([[jnp.inf, -1.0], [-1.0, jnp.inf]], ["contains non-finite entries"]),
            ([[jnp.nan, -1.0], [-1.0, 2.0]], ["contains non-finite entries"]),
            ([[2.0, -1.0], [-0.5, 2.0]], ["not symmetric"]),
            ([[2.0, 0.0], [0.0, 0.0]], ["non-positive diagonal"]),
            ([[2.0, 0.5], [0.5, 2.0]], ["positive off-diagonal"]),
            ([[1.0, -2.0], [-2.0, 1.0]], ["negative capacitance to ground"]),
        ],
    )
    def test_check_maxwell_rejects(
        maxwell: list[list[float]], problems: list[str]
    ) -> None:
        with pytest.raises(NonPhysicalMatrixError) as error:
            check_maxwell(maxwell)
        assert error.value.problems == problems

    @staticmethod
    def test_check_maxwell_accepts_physical() -> None:
        check_maxwell([[2.0, -1.0], [-1.0, 2.0]])

    @staticmethod
    def test_check_maxwell_tolerance_is_per_matrix() -> None:
        big = [[1e-12, -1e-13], [-1e-13, 1e-12]]
        small = [[1e-15, 5e-22], [5e-22, 1e-15]]
        with pytest.raises(NonPhysicalMatrixError, match="positive off-diagonal"):
            check_maxwell(np.stack([big, small]))

    @staticmethod
    def test_mutual_conversion(interp: GridInterpolator) -> None:
        mutual = maxwell_to_mutual(interp(**CENTER))
        np.testing.assert_allclose(mutual[0, 1], -_maxwell(**CENTER)[0, 1], rtol=1e-12)
        np.testing.assert_allclose(
            mutual[0, 0], _maxwell(**CENTER).sum(axis=-1)[0], rtol=1e-12
        )

    @staticmethod
    def test_sax_model_with_dataset_lookup(interp: GridInterpolator) -> None:
        """A capacitance looked up from the dataset drops into an existing SAX model."""

        def plate_capacitor_lookup(
            *, f=DEFAULT_FREQUENCY, length=26.0, width=5.0, gap=7.0, cross_section="cpw"
        ):
            f = jnp.asarray(f)
            c_mutual = maxwell_to_mutual(interp(length=length, width=width, gap=gap))[
                0, 1
            ]
            return capacitor(
                f=f,
                capacitance=c_mutual,
                z0=cpw_z0_from_cross_section(cross_section, f),
            )

        f = jnp.linspace(4e9, 8e9, 5)
        looked_up = jax.jit(
            lambda gap: plate_capacitor_lookup(f=f, length=120.0, width=10.0, gap=gap)
        )(7.0)
        expected = capacitor(
            f=f,
            capacitance=-_maxwell(**CENTER)[0, 1],
            z0=cpw_z0_from_cross_section("cpw", f),
        )
        for key in [("o1", "o2"), ("o1", "o1")]:
            np.testing.assert_allclose(looked_up[key], expected[key], rtol=1e-9)
