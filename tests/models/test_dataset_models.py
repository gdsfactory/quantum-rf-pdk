"""Distributed CPW model algebra, units and JAX transformations."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from qpdk.models.constants import ε_0, μ_0
from qpdk.models.couplers import (
    cpw_coupling_model,
    cpw_cpw_coupling_capacitance,
)
from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetMetadata,
    GridInterpolator,
    Quantity,
    QuantityKind,
    sweep,
)
from qpdk.models.datasets.generate import write
from qpdk.tech import coplanar_waveguide


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A labelled algebra fixture, not a simulated design dataset."""
    metadata = DatasetMetadata(
        name="test_cpw",
        synthetic=True,
        axes=tuple(Axis(name=name, unit="um") for name in ("width", "cpw_gap", "gap")),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=("lower", "upper"),
        provenance={"settings": {"slice_length_um": 4.0, "permittivity": 11.45}},
    )

    def solve(width: float, cpw_gap: float, gap: float) -> dict:
        diagonal = (120 + width - cpw_gap) * 1e-12
        mutual = (30 - gap / 2) * 1e-12
        return {
            "maxwell_capacitance": np.array([[diagonal, -mutual], [-mutual, diagonal]])
            * 4e-6
        }

    return write(
        tmp_path_factory.mktemp("cpw"),
        metadata,
        sweep(
            metadata,
            solve,
            {"width": [4.0, 20.0], "cpw_gap": [3.0, 12.0], "gap": [2.0, 50.0]},
        ),
    )


def matrix(model, **kwargs) -> np.ndarray:
    result = model(**kwargs)
    ports = ("o1", "o2", "o3", "o4")
    return np.stack(
        [np.stack([result[a, b] for b in ports], axis=-1) for a in ports], axis=-2
    )


def test_zero_length_connections(dataset: Dataset) -> None:
    expected = np.fliplr(np.eye(4))
    np.testing.assert_allclose(
        matrix(cpw_coupling_model(dataset), length=0), expected, atol=1e-12
    )


@settings(deadline=None, max_examples=12)
@given(
    width=st.floats(4, 20),
    cpw_gap=st.floats(3, 12),
    gap=st.floats(2, 50),
    length=st.floats(0, 5000),
)
def test_reciprocal_and_lossless(dataset: Dataset, **point: float) -> None:
    s = matrix(cpw_coupling_model(dataset), f=jnp.array([1e9, 5e9, 12e9]), **point)
    np.testing.assert_allclose(s, s.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        s @ s.conj().swapaxes(-1, -2), np.broadcast_to(np.eye(4), s.shape), atol=1e-10
    )


def test_coupling_capacitance_uses_the_drawn_ground_geometry() -> None:
    layout = Dataset("cpw_coupling_ground_strip_palace")
    unshielded = Dataset("cpw_coupling_palace")
    gaps = jnp.array([8.0, 12.0, 16.0, 25.0, 250.0])
    dimensions = dict(width=10.0, cpw_gap=6.0, gap=gaps)
    actual_slice = GridInterpolator(layout.grid("maxwell_capacitance"))(**dimensions)
    etched_slice = GridInterpolator(unshielded.grid("maxwell_capacitance"))(
        **dimensions
    )
    slice_length = layout.metadata.provenance["settings"]["slice_length_um"]
    cross_section = coplanar_waveguide(width=10.0, gap=6.0)
    lookup = jax.jit(
        lambda gap: cpw_cpw_coupling_capacitance(
            f=5e9, length=500.0, gap=gap, cross_section=cross_section
        )
    )
    np.testing.assert_allclose(
        lookup(gaps), -actual_slice[..., 0, 1] * 500.0 / slice_length, rtol=1e-12
    )
    np.testing.assert_array_equal(actual_slice[:2], etched_slice[:2])
    assert (-actual_slice[2:, 0, 1] < -etched_slice[2:, 0, 1]).all()


def test_jit_vmap_and_geometry_derivatives(dataset: Dataset) -> None:
    model = cpw_coupling_model(dataset)

    def coupling(gap):
        return jnp.abs(model(f=5e9, length=1000.0, gap=gap)["o1", "o3"]) ** 2

    scalar = jax.jit(coupling)
    vector = jax.jit(jax.vmap(coupling))(jnp.array([4.0, 8.0, 16.0]))
    np.testing.assert_allclose(
        vector, [scalar(v) for v in (4.0, 8.0, 16.0)], atol=1e-12
    )
    for axis in ("width", "cpw_gap", "gap"):

        def objective(value, axis=axis):
            return jnp.abs(model(**{axis: value})["o1", "o3"]) ** 2

        center = {"width": 10.0, "cpw_gap": 6.0, "gap": 8.0}[axis]
        derivative = float(jax.jit(jax.grad(objective))(center))
        finite_difference = float(
            (objective(center + 0.001) - objective(center - 0.001)) / 0.002
        )
        assert abs(derivative) > 1e-9
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-5)


def test_slice_normalization_and_modal_phase(dataset: Dataset) -> None:
    model = cpw_coupling_model(dataset)
    s = matrix(model, f=5e9, length=1000.0)
    transform = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    even_c = (120 + 10 - 6 - (30 - 8 / 2)) * 1e-12
    odd_c = (120 + 10 - 6 + (30 - 8 / 2)) * 1e-12
    gamma = 2j * np.pi * 5e9 * np.sqrt(μ_0 * ε_0 * (1 + 11.45) / 2)
    expected = []
    for c in (even_c, odd_c):
        z = np.sqrt(μ_0 * ε_0 * ((1 + 11.45) / 2) / c**2)
        theta = gamma * 0.001
        expected.append(2 / (2 * np.cosh(theta) + (z / 50 + 50 / z) * np.sinh(theta)))
    physical_transmission = s[np.ix_([3, 2], [0, 1])]
    np.testing.assert_allclose(
        transform @ physical_transmission @ transform.T, np.diag(expected), atol=1e-12
    )


def test_outside_domain_is_nan(dataset: Dataset) -> None:
    model = cpw_coupling_model(dataset)
    for axis in ("width", "cpw_gap", "gap"):
        assert np.isnan(matrix(model, **{axis: -1.0})).all()


def test_rejects_asymmetric_dataset(dataset: Dataset, tmp_path: Path) -> None:
    asymmetric = dataset.table.with_columns(
        pl
        .when((pl.col("row") == "lower") & (pl.col("col") == "lower"))
        .then(pl.col("value") * 1.1)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    data = write(tmp_path, dataset.metadata, asymmetric)
    with pytest.raises(ValueError, match="symmetric"):
        cpw_coupling_model(data)


def test_numerical_asymmetry_is_averaged(dataset: Dataset, tmp_path: Path) -> None:
    noisy = dataset.table.with_columns(
        pl
        .when((pl.col("row") == "upper") & (pl.col("col") == "upper"))
        .then(pl.col("value") * 1.006)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    averaged = dataset.table.with_columns(
        pl
        .when(pl.col("row") == pl.col("col"))
        .then(pl.col("value") * 1.003)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    noisy_data = write(tmp_path / "noisy", dataset.metadata, noisy)
    averaged_data = write(tmp_path / "averaged", dataset.metadata, averaged)
    np.testing.assert_allclose(
        matrix(cpw_coupling_model(noisy_data)),
        matrix(cpw_coupling_model(averaged_data)),
        atol=1e-12,
    )


@pytest.mark.parametrize("complex_quantity", [False, True])
def test_rejects_wrong_capacitance_convention(
    dataset: Dataset, tmp_path: Path, complex_quantity: bool
) -> None:
    quantity = dataset.metadata.quantities[0].model_copy(
        update={
            "kind": QuantityKind.MAXWELL_CAPACITANCE
            if complex_quantity
            else QuantityKind.MUTUAL_CAPACITANCE,
            "complex": complex_quantity,
        }
    )
    metadata = dataset.metadata.model_copy(update={"quantities": (quantity,)})
    frame = dataset.table
    if complex_quantity:
        frame = frame.with_columns(value_imag=pl.lit(0.0, dtype=pl.Float64))
    wrong = write(tmp_path, metadata, frame)
    with pytest.raises(ValueError, match="real Maxwell"):
        cpw_coupling_model(wrong)


def test_rejects_nonphysical_capacitance(dataset: Dataset, tmp_path: Path) -> None:
    wrong = write(
        tmp_path, dataset.metadata, dataset.table.with_columns(value=-pl.col("value"))
    )
    with pytest.raises(ValueError, match="non-positive diagonal"):
        cpw_coupling_model(wrong)


def test_rejects_a_floating_pair(dataset: Dataset, tmp_path: Path) -> None:
    floating = dataset.table.with_columns(
        pl
        .when(pl.col("row") == pl.col("col"))
        .then(pl.lit(1e-15))
        .otherwise(pl.lit(-1e-15))
        .alias("value")
    )
    wrong = write(tmp_path, dataset.metadata, floating)
    with pytest.raises(ValueError, match="positive capacitance to ground"):
        cpw_coupling_model(wrong)


@pytest.mark.parametrize(
    "dataset_name", ["cpw_coupling_palace", "cpw_coupling_ground_strip_palace"]
)
def test_bundled_palace_grid_and_default_model(dataset_name: str) -> None:
    data = Dataset(dataset_name)
    assert not data.metadata.synthetic
    grid = data.grid("maxwell_capacitance")
    assert grid.values.shape == (15, 13, 35, 2, 2)
    convergence = data.metadata.provenance["mesh_convergence"]
    change = data.grid("mesh_relative_change").values
    level = data.grid("mesh_refinement_level").values
    assert jnp.isfinite(change).all()
    assert (change <= convergence["relative_tolerance"]).all()
    assert (level >= 1).all()
    assert (level == level.astype(int)).all()
    for name, field in [("mesh_near_size", "near_mesh"), ("mesh_far_size", "far_mesh")]:
        sizes = jnp.asarray([
            settings[field] * 1e-6 for settings in convergence["settings_by_level"]
        ])
        np.testing.assert_allclose(
            data.grid(name).values, sizes[level.astype(int)], rtol=1e-12
        )
    residual = data.grid("solver_relative_residual").values
    assert jnp.isfinite(residual).all()
    assert (
        residual <= data.metadata.provenance["settings"]["tolerance"] * 1.00001
    ).all()
    mutual = -np.asarray(grid.values)[..., 0, 1]
    assert (np.diff(mutual, axis=2) < 0).all()
    model = cpw_coupling_model()
    result = jax.jit(model)(
        f=jnp.array([1e9, 5e9, 12e9]), width=9.0, cpw_gap=7.0, gap=10.0
    )
    ports = ("o1", "o2", "o3", "o4")
    s = np.stack(
        [np.stack([result[a, b] for b in ports], axis=-1) for a in ports], axis=-2
    )
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s, s.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        s @ s.conj().swapaxes(-1, -2), np.broadcast_to(np.eye(4), s.shape), atol=1e-10
    )
