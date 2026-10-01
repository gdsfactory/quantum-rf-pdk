"""Tests for qpdk.datasets: FEM dataset format, validation, and JAX lookup."""

import dataclasses
import shutil
from itertools import product
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

from qpdk.datasets import (
    DATASETS_PATH,
    Dataset,
    DatasetError,
    GridInterpolator,
    LFSPointerError,
    Manifest,
    RunStatus,
    maxwell_to_mutual,
    maxwell_violations,
    schema,
    to_grid,
    validate,
)
from qpdk.datasets.data.plate_capacitor_synthetic.generate import synthetic_maxwell
from qpdk.models.capacitor import (
    plate_capacitor,
    plate_capacitor_capacitance_analytical,
)
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.generic import capacitor

NAME = "plate_capacitor_synthetic"
EP_R = 11.45


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset(NAME)


@pytest.fixture(scope="module")
def interp(dataset: Dataset) -> GridInterpolator:
    return GridInterpolator(dataset.grid("maxwell_capacitance", cross_section="cpw"))


def _domain_points(manifest: Manifest) -> st.SearchStrategy[dict[str, float]]:
    return st.fixed_dictionaries({
        a.name: st.floats(*a.domain, allow_nan=False) for a in manifest.axes
    })


POINTS = _domain_points(Dataset(NAME).manifest)


# ---------------------------------------------------------------------------
# Format and validation
# ---------------------------------------------------------------------------


def test_shipped_dataset_is_valid_and_labelled_synthetic(dataset: Dataset) -> None:
    assert dataset.manifest.synthetic
    validate(dataset.table, dataset.manifest)
    assert dataset.table.height == 8 * 4 * 6 * 2 * 2


def test_grid_layout_is_deterministic(dataset: Dataset) -> None:
    """Axes in manifest order with ascending values, then terminals in manifest order."""
    grid = dataset.grid("maxwell_capacitance", cross_section="cpw")
    axes = dataset.manifest.axes
    assert grid.values.shape == (8, 4, 6, 2, 2)
    for idx in [(0, 0, 0), (3, 1, 4), (7, 3, 5)]:
        length, width, gap = (a.values[i] for a, i in zip(axes, idx, strict=True))
        np.testing.assert_array_equal(
            grid.values[idx], synthetic_maxwell(length, width, gap, EP_R)
        )


def test_grid_is_independent_of_row_order(dataset: Dataset) -> None:
    shuffled = dataset.table.sample(fraction=1.0, shuffle=True, seed=1)
    a = to_grid(shuffled, dataset.manifest, "maxwell_capacitance", cross_section="cpw")
    b = dataset.grid("maxwell_capacitance", cross_section="cpw")
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: pl.concat([f, f.head(1)]), "duplicate"),
        (lambda f: f.with_columns(unit=pl.lit("pF")), "not in 'F'"),
        (lambda f: f.with_columns(row=pl.lit("o3")), "terminals outside"),
        (
            lambda f: f.with_columns(length=pl.col("length") + 0.5),
            "off the manifest grid",
        ),
        (
            lambda f: f.with_columns(cross_section=pl.lit("xs_wide")),
            "undeclared 'cross_section'",
        ),
        (lambda f: f.with_columns(quantity=pl.lit("capacitance")), "unknown quantity"),
        (lambda f: f.with_columns(status=pl.lit("done")), "unknown status"),
        (
            lambda f: f.with_columns(value=pl.lit(None, pl.Float64)),
            "'ok' but have no value",
        ),
        (
            lambda f: f.with_columns(status=pl.lit("failed")),
            "not 'ok' but carry a value",
        ),
        (lambda f: f.with_columns(value_imag=pl.lit(0.0)), "imaginary part"),
        (lambda f: f.with_columns(run_id=pl.lit("same")), "mix parameter points"),
        (lambda f: f.drop("unit"), r"missing \['unit'\]"),
    ],
)
def test_validation_rejects(dataset: Dataset, mutate, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        validate(mutate(dataset.table), dataset.manifest)


def _at(**point: float) -> pl.Expr:
    """Select one grid point; grid values are stored exactly, so ``is_in`` is safe."""
    return pl.all_horizontal(pl.col(k).is_in([v]) for k, v in point.items())


def test_incomplete_grid_is_explicit(dataset: Dataset) -> None:
    drop = _at(length=40.0, width=10.0, gap=7.0)
    frame = dataset.table.filter(~drop)
    validate(frame, dataset.manifest)
    with pytest.raises(DatasetError, match=r"1 grid points .* missing or failed"):
        to_grid(frame, dataset.manifest, "maxwell_capacitance", cross_section="cpw")
    grid = to_grid(
        frame,
        dataset.manifest,
        "maxwell_capacitance",
        allow_missing=True,
        cross_section="cpw",
    )
    assert np.isnan(grid.values[1, 1, 2]).all()
    assert np.isfinite(
        np.delete(grid.values.reshape(-1, 4), 1 * 24 + 1 * 6 + 2, axis=0)
    ).all()


def test_failed_runs_are_never_interpolated(dataset: Dataset) -> None:
    point = _at(length=300.0, width=20.0, gap=20.0)
    frame = dataset.table.with_columns(
        status=pl.when(point).then(pl.lit(RunStatus.FAILED.value)).otherwise("status"),
        value=pl.when(point).then(None).otherwise("value"),
    )
    validate(frame, dataset.manifest)
    with pytest.raises(DatasetError, match="'status': 'failed'"):
        to_grid(frame, dataset.manifest, "maxwell_capacitance", cross_section="cpw")
    grid = to_grid(
        frame,
        dataset.manifest,
        "maxwell_capacitance",
        allow_missing=True,
        cross_section="cpw",
    )
    f = GridInterpolator(grid)
    assert np.isnan(f(length=290.0, width=19.0, gap=19.0)).all()
    assert np.isfinite(f(length=100.0, width=10.0, gap=5.0)).all()


def test_variant_must_be_selected(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="Select one value of each variant"):
        dataset.grid("maxwell_capacitance")
    with pytest.raises(DatasetError, match="Select one value of each variant"):
        dataset.grid("maxwell_capacitance", cross_section="nope")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 99}, "Unsupported schema_version"),
        (
            {"conventions": {"terminals": ["o1", "o1"], "reference_ground": "g"}},
            "Duplicate terminal",
        ),
        (
            {"conventions": {"terminals": ["o1", "g"], "reference_ground": "g"}},
            "reference ground",
        ),
        (
            {"axes": [{"name": "length", "unit": "um", "values": [2.0, 1.0]}]},
            "strictly increasing",
        ),
        (
            {
                "axes": [
                    {
                        "name": "length",
                        "unit": "um",
                        "values": [1.0, 2.0],
                        "validated": [0.0, 2.0],
                    }
                ]
            },
            "within its grid",
        ),
        (
            {"axes": [{"name": "value", "unit": "um", "values": [1.0]}]},
            "reserved columns",
        ),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_manifest_rejects(dataset: Dataset, change: dict, message: str) -> None:
    data = dataset.manifest.model_dump(mode="json") | change
    with pytest.raises(ValidationError, match=message):
        Manifest.model_validate(data)


# ---------------------------------------------------------------------------
# Storage: LFS and append-only parts
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_copy(tmp_path: Path) -> Dataset:
    target = tmp_path / NAME
    shutil.copytree(DATASETS_PATH / NAME, target)
    return Dataset(target)


def test_unresolved_lfs_pointer_has_useful_error(dataset_copy: Dataset) -> None:
    (part,) = dataset_copy.result_files
    part.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:"
        + "0" * 64
        + "\nsize 9733\n"
    )
    with pytest.raises(LFSPointerError, match="git lfs pull"):
        dataset_copy.scan()


def test_append_writes_new_part_and_rejects_duplicates(dataset_copy: Dataset) -> None:
    manifest = dataset_copy.manifest.model_copy(
        update={
            "variants": (
                dataset_copy.manifest.variants[0].model_copy(
                    update={"values": ("cpw", "cpw_wide")}
                ),
            )
        }
    )
    dataset_copy.manifest = manifest
    original = dataset_copy.table
    new = original.with_columns(
        run_id=pl.col("run_id") + "-wide",
        cross_section=pl.lit("cpw_wide"),
        value=pl.col("value") * 1.1,
    )
    path = dataset_copy.append(new, part="part-0001")
    assert path.name == "part-0001.parquet"
    assert len(dataset_copy.result_files) == 2
    assert dataset_copy.table.height == 2 * original.height
    np.testing.assert_allclose(
        dataset_copy.grid("maxwell_capacitance", cross_section="cpw_wide").values,
        1.1 * dataset_copy.grid("maxwell_capacitance", cross_section="cpw").values,
    )
    with pytest.raises(DatasetError, match="duplicate"):
        dataset_copy.append(new.head(4), part="part-0002")
    with pytest.raises(FileExistsError):
        dataset_copy.append(new.head(0), part="part-0001")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=25)
@given(idx=st.tuples(st.integers(0, 7), st.integers(0, 3), st.integers(0, 5)))
def test_reproduces_grid_points(
    interp: GridInterpolator, idx: tuple[int, int, int]
) -> None:
    point = {a.name: a.values[i] for a, i in zip(interp.grid.axes, idx, strict=True)}
    np.testing.assert_allclose(interp(**point), interp.grid.values[idx], rtol=1e-12)


def test_matches_scipy_reference(interp: GridInterpolator) -> None:
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(*a.domain, 200) for a in interp.grid.axes])
    reference = RegularGridInterpolator(
        [np.asarray(a.values) for a in interp.grid.axes], interp.grid.values
    )(pts)
    ours = interp(**{a.name: pts[:, i] for i, a in enumerate(interp.grid.axes)})
    np.testing.assert_allclose(ours, reference, rtol=1e-12)


def test_held_out_points_against_formula(interp: GridInterpolator) -> None:
    """Off-grid lookups stay close to the generating formula (trilinear error only)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        point = {a.name: float(rng.uniform(*a.domain)) for a in interp.grid.axes}
        expected = synthetic_maxwell(**point, ep_r=EP_R)
        np.testing.assert_allclose(interp(**point), expected, rtol=0.05)


@settings(deadline=None, max_examples=25)
@given(point=POINTS)
def test_interpolated_matrix_is_physical(
    interp: GridInterpolator, point: dict[str, float]
) -> None:
    assert maxwell_violations(interp(**point)) == []


def test_mutual_conversion(interp: GridInterpolator) -> None:
    point = {"length": 120.0, "width": 10.0, "gap": 7.0}
    mutual = maxwell_to_mutual(interp(**point))
    np.testing.assert_allclose(
        mutual[0, 1],
        plate_capacitor_capacitance_analytical(ep_r=EP_R, **point),
        rtol=1e-12,
    )
    maxwell = synthetic_maxwell(**point, ep_r=EP_R)
    np.testing.assert_allclose(mutual[0, 0], maxwell.sum(axis=-1)[0], rtol=1e-12)


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


def test_clip_is_opt_in(dataset: Dataset) -> None:
    f = GridInterpolator(
        dataset.grid("maxwell_capacitance", cross_section="cpw"), out_of_range="clip"
    )
    np.testing.assert_allclose(
        f(length=400.0, width=10.0, gap=5.0), f(length=300.0, width=10.0, gap=5.0)
    )
    with pytest.raises(ValueError, match="out_of_range"):
        GridInterpolator(f.grid, out_of_range="extrapolate")  # pyrefly: ignore[bad-argument-type]


def test_validated_domain_narrower_than_grid(dataset: Dataset) -> None:
    axes = list(dataset.manifest.axes)
    axes[0] = axes[0].model_copy(update={"validated": (40.0, 250.0)})
    grid = dataset.grid("maxwell_capacitance", cross_section="cpw")
    f = GridInterpolator(dataclasses.replace(grid, axes=tuple(axes)))
    assert f.domain["length"] == (40.0, 250.0)
    assert np.isnan(f(length=30.0, width=10.0, gap=5.0)).all()


def test_requires_every_axis(interp: GridInterpolator) -> None:
    with pytest.raises(TypeError, match="Expected exactly the axes"):
        interp(length=100.0, width=10.0)


def test_jit_vmap_grad(interp: GridInterpolator) -> None:
    def c_mutual(length, width, gap):
        return -interp(length=length, width=width, gap=gap)[..., 0, 1]

    batch = jnp.linspace(25.0, 290.0, 7)
    jitted = jax.jit(c_mutual)(batch, 10.0, 5.0)
    np.testing.assert_allclose(jitted, c_mutual(batch, 10.0, 5.0), rtol=1e-12)
    vmapped = jax.vmap(c_mutual, in_axes=(0, None, None))(batch, 10.0, 5.0)
    np.testing.assert_allclose(vmapped, jitted, rtol=1e-12)
    assert jitted.dtype == jnp.float64

    grad = jax.jit(jax.grad(c_mutual, argnums=(0, 2)))(130.0, 10.0, 5.0)
    h = 1e-3
    fd_length = (c_mutual(130.0 + h, 10.0, 5.0) - c_mutual(130.0 - h, 10.0, 5.0)) / (
        2 * h
    )
    fd_gap = (c_mutual(130.0, 10.0, 5.0 + h) - c_mutual(130.0, 10.0, 5.0 - h)) / (2 * h)
    np.testing.assert_allclose(grad, (fd_length, fd_gap), rtol=1e-6)
    assert grad[1] < 0  # wider gap, less capacitance


def test_agrees_with_sax_interpolate_xarray_in_domain(interp: GridInterpolator) -> None:
    """Same numbers as SAX inside the domain; SAX clamps outside, which we do not."""
    grid = interp.grid
    coords = {a.name: np.asarray(a.values) for a in grid.axes}
    flat = grid.values.reshape(*grid.values.shape[:3], 4)
    xarr = xr.DataArray(
        flat,
        coords={
            **coords,
            "targets": np.array(["c11", "c12", "c21", "c22"], dtype=object),
        },
    )
    point = {"length": 133.0, "width": 12.5, "gap": 8.2}
    sax_out = sax.interpolate_xarray(xarr, **point)
    np.testing.assert_allclose(
        interp(**point).reshape(4),
        [sax_out[k] for k in ["c11", "c12", "c21", "c22"]],
        rtol=1e-12,
    )

    outside = {**point, "length": 1000.0}
    sax_clamped = sax.interpolate_xarray(xarr, **outside)["c12"]
    assert np.isfinite(sax_clamped)
    assert np.isnan(interp(**outside)[0, 1])


def test_complex_scalar_quantity(dataset: Dataset) -> None:
    """Complex quantities and scalar quantities round-trip through the same path."""
    manifest = Manifest.model_validate(
        dataset.manifest.model_dump(mode="json")
        | {
            "name": "s21_toy",
            "axes": [
                {"name": "frequency", "unit": "Hz", "values": [4e9, 5e9, 6e9]},
                {"name": "gap", "unit": "um", "values": [2.0, 4.0]},
            ],
            "variants": [],
            "quantities": [
                {
                    "name": "s21",
                    "kind": "s_parameters",
                    "unit": "1",
                    "complex": True,
                    "matrix": False,
                }
            ],
        }
    )
    rows = [
        {
            "run_id": f"{f}-{g}",
            "status": "ok",
            "frequency": f,
            "gap": g,
            "quantity": "s21",
            "row": None,
            "col": None,
            "value": np.cos(f / 1e9 * g),
            "value_imag": np.sin(f / 1e9 * g),
            "unit": "1",
        }
        for f, g in product([4e9, 5e9, 6e9], [2.0, 4.0])
    ]
    frame = pl.DataFrame(rows, schema=schema(manifest))
    validate(frame, manifest)
    grid = to_grid(frame, manifest, "s21")
    assert grid.values.shape == (3, 2)
    assert np.iscomplexobj(grid.values)
    f = GridInterpolator(grid)
    np.testing.assert_allclose(f(frequency=5e9, gap=4.0), np.exp(20j), rtol=1e-12)
    np.testing.assert_allclose(
        f(frequency=4.5e9, gap=2.0), (np.exp(8j) + np.exp(10j)) / 2, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# SAX-facing use
# ---------------------------------------------------------------------------


def test_sax_model_with_dataset_lookup(interp: GridInterpolator) -> None:
    """A capacitance looked up from the dataset drops into an existing SAX model."""

    def plate_capacitor_lookup(
        *, f=DEFAULT_FREQUENCY, length=26.0, width=5.0, gap=7.0, cross_section="cpw"
    ):
        f = jnp.asarray(f)
        c_mutual = maxwell_to_mutual(interp(length=length, width=width, gap=gap))[0, 1]
        return capacitor(
            f=f, capacitance=c_mutual, z0=cpw_z0_from_cross_section(cross_section, f)
        )

    f = jnp.linspace(4e9, 8e9, 5)
    looked_up = jax.jit(
        lambda gap: plate_capacitor_lookup(f=f, length=120.0, width=10.0, gap=gap)
    )(7.0)
    analytical = plate_capacitor(f=f, length=120.0, width=10.0, gap=7.0)
    for key in [("o1", "o2"), ("o1", "o1")]:
        np.testing.assert_allclose(looked_up[key], analytical[key], rtol=1e-9)
