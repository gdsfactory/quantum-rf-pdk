"""Stored full-wave meanders retain their numerical checks and JAX support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models import meander_inductor_fullwave_model, models
from qpdk.models.datasets import Dataset, GridInterpolator


def test_complete_grid_and_independent_references() -> None:
    data = Dataset("meander_sparameters_palace")
    assert data.metadata.terminals == ("o1", "o2")
    assert data.metadata.provenance["reference_impedance_ohm"] == pytest.approx(50.0)
    assert "meander_inductor_fullwave_model" not in models
    for turns in ("3", "5", "7"):
        grid = data.grid("s_parameters", n_turns=turns, allow_missing=False)
        assert grid.values.shape == (9, 3, 3, 40, 2, 2)
        values = jnp.asarray(grid.values)
        assert jnp.all(jnp.isfinite(values))
        assert jnp.max(jnp.abs(values - values.swapaxes(-1, -2))) <= 0.005
        assert jnp.max(jnp.linalg.svd(values, compute_uv=False)) <= 1.01

    report = data.metadata.provenance["validation"]["interpolation"]
    assert report["passed"]
    assert len(report["checks"]) == 9
    for sample in report["checks"]:
        lookup = GridInterpolator(data.grid("s_parameters", n_turns=sample["n_turns"]))
        actual = lookup(
            **sample["point"], frequency=jnp.asarray(sample["frequency_Hz"])
        )
        expected = jnp.asarray(sample["reference_real"]) + 1j * jnp.asarray(
            sample["reference_imag"]
        )
        assert jnp.max(jnp.abs(actual - expected)) <= 0.01
        assert sample["reference_settings"]["order"] == 3


@pytest.mark.parametrize("turns", [3, 5, 7])
def test_model_units_jit_and_gradients(turns: int) -> None:
    model = jax.jit(meander_inductor_fullwave_model(n_turns=turns))
    point = dict(turn_length=200.0, wire_width=1.5, wire_gap=4.0)
    frequencies = jnp.array([1e9, 5e9, 10e9])
    result = model(f=frequencies, **point)
    grid = Dataset("meander_sparameters_palace").grid(
        "s_parameters", n_turns=str(turns)
    )
    expected = GridInterpolator(grid)(frequency=frequencies, **point)
    for i, a in enumerate(grid.terminals):
        for j, b in enumerate(grid.terminals):
            np.testing.assert_allclose(result[a, b], expected[..., i, j], atol=1e-12)

    def transmission(length: float) -> jax.Array:
        return jnp.abs(
            model(f=5e9, turn_length=length, wire_width=1.5, wire_gap=4.0)["o2", "o1"]
        )

    assert jnp.isfinite(jax.jit(jax.grad(transmission))(162.5))
    assert jax.jit(jax.vmap(transmission))(jnp.array([150.0, 250.0])).shape == (2,)
    assert jnp.isnan(transmission(99.0))
    assert jnp.isnan(model(f=20.1e9)["o1", "o1"])
    assert jnp.isnan(model(f=1e9, wire_width=2.1)["o1", "o1"])
    assert jnp.isnan(model(f=1e9, wire_gap=8.1)["o1", "o1"])
