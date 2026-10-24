"""Admission checks for the shipped, real M1 gatemon extraction."""

import hashlib
import json
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import DATASETS_PATH, Dataset, check_maxwell
from qpdk.models.gatemon import (
    GATEMON_DATASET,
    gatemon_capacitances,
    gatemon_fem_model,
)

REPORT = json.loads((DATASETS_PATH / "gatemon_palace.verification.json").read_text())


@pytest.fixture(scope="module")
def data() -> Dataset:
    return Dataset(GATEMON_DATASET)


def test_shipped_grid_and_provenance(data: Dataset) -> None:
    """The published parts must be the admitted 144-point extraction."""
    assert REPORT["complete"]
    assert not REPORT["failures"]
    assert REPORT["points"] == 144
    assert not data.metadata.synthetic
    assert data.metadata.terminals == ("island", "gate", "readout")
    provenance = data.metadata.provenance
    assert provenance["settings"] == REPORT["settings"]
    assert provenance["input_manifest_sha256"] == REPORT["manifest_sha256"]
    assert provenance["runtime_sha256"] == REPORT["runtime_sha256"]
    assert (
        provenance["campaign_driver_sha256"]
        == REPORT["source"]["campaign_driver_sha256"]
    )
    assert provenance["scope"] == REPORT["scope"]
    for name, digest in REPORT["dataset_parts_sha256"].items():
        part = DATASETS_PATH / GATEMON_DATASET / name
        assert hashlib.sha256(part.read_bytes()).hexdigest() == digest
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (4, 3, 4, 3, 3, 3)
    assert grid.axis_names == tuple(REPORT["grid"])
    for name, coords in zip(grid.axis_names, grid.coords, strict=True):
        np.testing.assert_array_equal(coords, REPORT["grid"][name])
    check_maxwell(grid.values)
    assert data.scan().collect().height == 144 * 21
    residual = jnp.asarray(data.grid("solver_relative_residual").values)
    assert bool(jnp.all(jnp.isfinite(residual) & (residual >= 0) & (residual <= 1e-9)))
    nominal = gatemon_capacitances()
    np.testing.assert_allclose(
        nominal,
        [51.93449600005e-15, 1.581935847794e-15, 5.731363566206e-15],
        rtol=1e-10,
    )


@pytest.mark.parametrize("holdout", REPORT["holdouts"].values())
def test_shipped_interpolation_against_direct_solves(holdout: dict) -> None:
    """Holdouts were solved independently of the interpolation grid."""
    direct = jnp.asarray(holdout["direct_maxwell_capacitance_f"])
    check_maxwell(direct)
    actual = direct[0] * jnp.array([1.0, -1.0, -1.0])
    predicted = jnp.asarray(gatemon_capacitances(**holdout["point"]))
    error = jnp.abs(predicted - actual) / jnp.abs(actual)
    assert bool(jnp.all(error <= jnp.array([0.03, 0.10, 0.05])))
    assert holdout["direct_solver_relative_residual"] <= 1e-9


@pytest.mark.parametrize("transmissions", [(0.0,), (0.8,)])
def test_shipped_model_over_the_full_grid(
    data: Dataset, transmissions: tuple[float, ...]
) -> None:
    """Every real matrix must produce a finite, lossless reciprocal circuit."""
    points = jnp.asarray(list(product(*REPORT["grid"].values())))
    frequencies = jnp.linspace(1e9, 10e9, 101)
    model = gatemon_fem_model(transmissions=transmissions)
    response = jax.jit(
        jax.vmap(
            lambda point: model(
                f=frequencies, **dict(zip(REPORT["grid"], point, strict=True))
            )
        )
    )(points)
    ports = data.metadata.terminals
    scattering = jnp.stack(
        [jnp.stack([response[a, b] for b in ports], -1) for a in ports], -2
    )
    assert scattering.shape == (144, 101, 3, 3)
    assert bool(jnp.all(jnp.isfinite(scattering)))
    np.testing.assert_allclose(scattering, scattering.swapaxes(-1, -2), atol=1e-10)
    gram = scattering.conj().swapaxes(-1, -2) @ scattering
    np.testing.assert_allclose(
        gram, jnp.broadcast_to(jnp.eye(3), gram.shape), atol=1e-10
    )
    assert all(
        bool(jnp.isnan(value)) for value in gatemon_capacitances(island_size=300.0)
    )
