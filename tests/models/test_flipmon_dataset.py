"""The unbonded matrix and ideal electrical reduction keep ground branches."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from qpdk.models import flipmon_ideal_bond_model, flipmon_unbonded_model, models
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell


def test_grid_and_independent_references() -> None:
    data = Dataset("flipmon_unbonded_palace")
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (9, 3, 3, 3, 3, 3)
    check_maxwell(grid.values)
    lookup = GridInterpolator(grid)
    samples = data.metadata.provenance["validation"]["samples"]
    assert len(samples) == 12
    assert {sample["reference_settings"]["order"] for sample in samples} == {3, 4}
    for sample in samples:
        actual = lookup(**sample["point"])
        reference = jnp.asarray(sample["reference_maxwell_F"])
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=1e-17)
        np.testing.assert_allclose(
            actual.sum(axis=-1), reference.sum(axis=-1), rtol=0.02, atol=1e-17
        )
    assert "flipmon_unbonded_model" not in models
    assert "flipmon_ideal_bond_model" not in models


@pytest.mark.parametrize("bonded", [False, True])
def test_full_and_reduced_admittances(bonded: bool) -> None:
    point = dict(chip_gap=12.5, alignment=0.0, inner_radius=55.0, top_radius=110.0)
    capacitance = GridInterpolator(
        Dataset("flipmon_unbonded_palace").grid("maxwell_capacitance")
    )(**point)
    f = jnp.asarray([1e9, 5e9, 10e9])
    omega = 2 * jnp.pi * f
    inductance, junction_capacitance = 10e-9, 2e-15
    mapping = jnp.asarray([[1, 0], [0, 1], [1, 0]])
    if bonded:
        model = flipmon_ideal_bond_model(
            inductance=inductance, junction_capacitance=junction_capacitance
        )
        ports = ("inner_pad", "outer_ring")
        expected = 1j * omega[:, None, None] * (mapping.T @ capacitance @ mapping)
        expected += (1 / (1j * omega * inductance) + 1j * omega * junction_capacitance)[
            :, None, None
        ] * jnp.asarray([[1, -1], [-1, 1]])
    else:
        model = flipmon_unbonded_model()
        ports = ("inner_pad", "outer_ring", "top_pad")
        expected = 1j * omega[:, None, None] * capacitance
    values = cast(sax.SDict, jax.jit(model)(f=f, z_ref=75.0, **point))
    assert set(values) == {(a, b) for a in ports for b in ports}
    scattering = jnp.asarray([[values[a, b] for b in ports] for a in ports]).transpose(
        2, 0, 1
    )
    identity = jnp.broadcast_to(jnp.eye(len(ports)), scattering.shape)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scattering, scattering.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        scattering @ scattering.conj().swapaxes(-1, -2), identity, atol=1e-12
    )

    def evaluate(gap: float) -> jax.Array:
        values = cast(sax.SDict, model(f=5e9, **{**point, "chip_gap": gap}))
        return jnp.imag(values[ports[0], ports[0]])

    assert jnp.isfinite(jax.jit(jax.grad(evaluate))(12.5))
    assert jax.jit(jax.vmap(evaluate))(jnp.asarray([10.0, 15.0])).shape == (2,)
    assert jnp.isnan(evaluate(4.9))
