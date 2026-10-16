"""Static coupon models retain every electrode and extracted ground branch."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import sax

from qpdk.models import flipmon_readout_coupon_model, models
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell


def test_grid_and_independent_references() -> None:
    data = Dataset("flipmon_readout_coupon_palace")
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (2, 5, 5, 2, 5, 5)
    check_maxwell(grid.values)
    lookup = GridInterpolator(grid)
    samples = data.metadata.provenance["validation"]["samples"]
    assert len(samples) == 11
    assert {sample["reference_settings"]["order"] for sample in samples} == {3, 4}
    for sample in samples:
        actual = lookup(**sample["point"])
        reference = jnp.asarray(sample["reference_maxwell_F"])
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=1e-17)
        np.testing.assert_allclose(
            actual.sum(axis=-1), reference.sum(axis=-1), rtol=0.02, atol=1e-17
        )
    assert "flipmon_readout_coupon_model" not in models


def test_admittance_all_traces_and_jax() -> None:
    point = {
        "pad_length": 58.0,
        "pad_offset": 12.0,
        "feedline_gap": 16.0,
        "coupling_length": 650.0,
    }
    lookup = GridInterpolator(
        Dataset("flipmon_readout_coupon_palace").grid("maxwell_capacitance")
    )
    capacitance = lookup(**point)
    model = flipmon_readout_coupon_model()
    f = jnp.asarray([1e9, 5e9, 10e9])
    values = jax.jit(model)(f=f, z_ref=75.0, **point)
    ports = ("island_a", "island_b", "readout", "feedline", "top_pad")
    assert set(values) == {(a, b) for a in ports for b in ports}
    scattering = jnp.asarray([[values[a, b] for b in ports] for a in ports]).transpose(
        2, 0, 1
    )
    identity = jnp.broadcast_to(jnp.eye(len(ports)), scattering.shape)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    expected = 2j * jnp.pi * f[:, None, None] * capacitance
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scattering, scattering.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        scattering @ scattering.conj().swapaxes(-1, -2), identity, atol=1e-12
    )

    def evaluate(length: float) -> jax.Array:
        values = cast(sax.SDict, model(f=5e9, **{**point, "pad_length": length}))
        return jnp.imag(values[ports[0], ports[0]])

    assert jnp.isfinite(jax.jit(jax.grad(evaluate))(point["pad_length"]))
    assert jax.jit(jax.vmap(evaluate))(
        jnp.asarray([point["pad_length"], point["pad_length"] + 1])
    ).shape == (2,)
    assert jnp.isnan(evaluate(47.0))
