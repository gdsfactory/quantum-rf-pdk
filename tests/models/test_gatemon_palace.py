"""Admission checks for the shipped, real M1 gatemon extraction."""

import hashlib
import json
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import Dataset, check_maxwell
from qpdk.models.gatemon import (
    GATEMON_DATASET,
    gatemon_capacitances,
    gatemon_fem_model,
)


@pytest.fixture(scope="module")
def data() -> Dataset:
    return Dataset(GATEMON_DATASET)


@pytest.fixture(scope="module")
def report(data: Dataset) -> dict:
    return data.metadata.provenance["validation"]["verification"]


def test_shipped_grid_and_provenance(data: Dataset, report: dict) -> None:
    """The published parts must be the admitted 144-point extraction."""
    assert report["complete"]
    assert not report["failures"]
    assert report["points"] == 144
    assert not data.metadata.synthetic
    assert data.metadata.terminals == ("island", "gate", "readout")
    provenance = data.metadata.provenance
    assert provenance["settings"] == report["settings"]
    assert provenance["input_manifest_sha256"] == report["manifest_sha256"]
    assert provenance["runtime_sha256"] == report["runtime_sha256"]
    assert (
        provenance["campaign_driver_sha256"]
        == report["source"]["campaign_driver_sha256"]
    )
    assert provenance["scope"] == report["scope"]
    frame = data.scan().collect()
    payload = {
        "columns": frame.columns,
        "rows": frame.sort("run_id", "quantity", "row", "col").rows(),
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == report["dataset_rows_sha256"]
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (4, 3, 4, 3, 3, 3)
    assert grid.axis_names == tuple(report["grid"])
    for name, coords in zip(grid.axis_names, grid.coords, strict=True):
        np.testing.assert_array_equal(coords, report["grid"][name])
    check_maxwell(grid.values)
    assert frame.height == 144 * 21
    residual = jnp.asarray(data.grid("solver_relative_residual").values)
    assert bool(jnp.all(jnp.isfinite(residual) & (residual >= 0) & (residual <= 1e-9)))
    nominal = gatemon_capacitances()
    np.testing.assert_allclose(
        nominal,
        [51.93449600005e-15, 1.581935847794e-15, 5.731363566206e-15],
        rtol=1e-10,
    )


def test_shipped_convergence_evidence(data: Dataset, report: dict) -> None:
    """Recompute representative sensitivity from the embedded direct solves."""
    convergence = data.metadata.provenance["validation"]["convergence"]
    assert convergence["convergence_passed"]
    assert convergence["scope"] == report["scope"]
    assert convergence["source"] == {
        key: value
        for key, value in report["source"].items()
        if key != "dataset_parts_sha256"
    }
    assert tuple(convergence["terminal_order"]) == data.metadata.terminals
    selected = convergence["selected_profile"]
    assert convergence["profiles"][selected] == report["settings"]
    for case in convergence["cases"].values():
        for result in case["profiles"].values():
            check_maxwell(jnp.asarray(result["maxwell_capacitance_f"]))
            assert result["solver_relative_residual"] <= 1e-9
    changes = []
    for check in convergence["checks"]:
        profiles = convergence["cases"][check["case"]]["profiles"]
        actual, reference = (
            jnp.asarray(profiles[name]["maxwell_capacitance_f"])[0]
            for name in check["comparison"]
        )
        change = jnp.abs((actual - reference) / reference)
        np.testing.assert_allclose(change, check["island_row_relative_change"])
        changes.append(change)
    maximum = jnp.max(jnp.stack(changes), axis=0)
    np.testing.assert_allclose(maximum, convergence["maximum_relative_changes"])
    assert bool(jnp.all(maximum <= jnp.asarray(convergence["relative_tolerances"])))


@pytest.mark.parametrize("holdout_index", range(3))
def test_shipped_interpolation_against_direct_solves(
    report: dict, holdout_index: int
) -> None:
    """Holdouts were solved independently of the interpolation grid."""
    holdout = report["holdouts"][str(holdout_index)]
    direct = jnp.asarray(holdout["direct_maxwell_capacitance_f"])
    check_maxwell(direct)
    actual = direct[0] * jnp.array([1.0, -1.0, -1.0])
    predicted = jnp.asarray(gatemon_capacitances(**holdout["point"]))
    error = jnp.abs(predicted - actual) / jnp.abs(actual)
    assert bool(jnp.all(error <= jnp.array([0.03, 0.10, 0.05])))
    assert holdout["direct_solver_relative_residual"] <= 1e-9


@pytest.mark.parametrize("transmissions", [(0.0,), (0.8,)])
def test_shipped_model_over_the_full_grid(
    data: Dataset, report: dict, transmissions: tuple[float, ...]
) -> None:
    """Every real matrix must produce a finite, lossless reciprocal circuit."""
    points = jnp.asarray(list(product(*report["grid"].values())))
    frequencies = jnp.linspace(1e9, 10e9, 101)
    model = gatemon_fem_model(transmissions=transmissions)
    response = jax.jit(
        jax.vmap(
            lambda point: model(
                f=frequencies, **dict(zip(report["grid"], point, strict=True))
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
