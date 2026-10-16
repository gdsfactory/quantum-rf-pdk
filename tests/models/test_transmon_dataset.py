"""Pad charging and linearized circuits retain the full extracted matrix."""

import jax
import jax.numpy as jnp
import numpy as np

from qpdk.models import double_pad_transmon_capacitance, double_pad_transmon_fem_model
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell


def test_grid_and_reference_values() -> None:
    data = Dataset("double_pad_transmon_palace")
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (3, 3, 5, 3, 2, 2)
    check_maxwell(grid.values)
    lookup = GridInterpolator(grid)
    samples = data.metadata.provenance["validation"]["samples"]
    assert len(samples) == 10
    assert {sample["reference_settings"]["order"] for sample in samples} == {3, 4}
    for sample in samples:
        actual = lookup(**sample["point"])
        reference = jnp.asarray(sample["reference_maxwell_F"])
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=1e-17)
        np.testing.assert_allclose(
            actual.sum(axis=-1), reference.sum(axis=-1), rtol=0.02, atol=1e-17
        )


def test_charging_capacitance_and_jax() -> None:
    point = dict(
        pad_width=200.0, pad_height=325.0, pad_gap=15.0, ground_clearance=150.0
    )
    capacitance = GridInterpolator(
        Dataset("double_pad_transmon_palace").grid("maxwell_capacitance")
    )(**point)
    ground = capacitance.sum(axis=-1)
    expected = -capacitance[0, 1] + ground[0] * ground[1] / ground.sum()
    value = jax.jit(double_pad_transmon_capacitance)(**point)
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
    assert 1e-15 < value < 1e-12

    def evaluate(width: float) -> jax.Array:
        return double_pad_transmon_capacitance(**{**point, "pad_width": width})

    assert jnp.isfinite(jax.jit(jax.grad(evaluate))(200.0))
    assert jax.jit(jax.vmap(evaluate))(jnp.asarray([200.0, 300.0])).shape == (2,)
    assert jnp.isnan(evaluate(149.0))


def test_linearized_junction_and_all_traces() -> None:
    inductance, junction_capacitance = 10e-9, 2e-15
    model = double_pad_transmon_fem_model(
        inductance=inductance, junction_capacitance=junction_capacitance
    )
    f = jnp.asarray([1e9, 5e9, 10e9])
    values = jax.jit(model)(f=f, z_ref=75.0)
    ports = ("left_pad", "right_pad")
    assert set(values) == {(a, b) for a in ports for b in ports}
    scattering = jnp.asarray([[values[a, b] for b in ports] for a in ports]).transpose(
        2, 0, 1
    )
    identity = jnp.broadcast_to(jnp.eye(2), scattering.shape)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    capacitance = GridInterpolator(
        Dataset("double_pad_transmon_palace").grid("maxwell_capacitance")
    )(pad_width=250.0, pad_height=400.0, pad_gap=20.0, ground_clearance=200.0)
    omega = 2 * jnp.pi * f
    expected = 1j * omega[:, None, None] * capacitance
    expected += (1 / (1j * omega * inductance) + 1j * omega * junction_capacitance)[
        :, None, None
    ] * jnp.asarray([[1, -1], [-1, 1]])
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scattering, scattering.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        scattering @ scattering.conj().swapaxes(-1, -2), identity, atol=1e-12
    )
