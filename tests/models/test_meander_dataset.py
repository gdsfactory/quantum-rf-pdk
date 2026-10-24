"""Measured winding capacitances retain variants and JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models import meander_inductor_ground_capacitance, models
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell


def test_complete_grid_and_independent_references() -> None:
    data = Dataset("meander_ground_capacitance_palace")
    samples = data.metadata.provenance["validation"]["samples"]
    assert len(samples) == 14
    assert {sample["reference_settings"]["order"] for sample in samples} == {3, 4}
    assert data.metadata.terminals == ("winding",)
    assert "meander_inductor_ground_capacitance" not in models
    for turns in ("3", "5", "7"):
        grid = data.grid("maxwell_capacitance", n_turns=turns, allow_missing=False)
        assert grid.values.shape == (3, 2, 3, 1, 1)
        check_maxwell(grid.values)
    for sample in samples:
        point = sample["point"]
        lookup = GridInterpolator(
            data.grid("maxwell_capacitance", n_turns=point["n_turns"])
        )
        actual = lookup(**{key: point[key] for key in lookup.grid.axis_names})
        np.testing.assert_allclose(
            actual, sample["reference_maxwell_F"], rtol=0.02, atol=1e-17
        )


@pytest.mark.parametrize("turns", [3, 5, 7])
def test_lookup_units_jit_and_gradients(turns: int) -> None:
    point = dict(turn_length=200.0, wire_width=2.0, wire_gap=4.0)
    data = Dataset("meander_ground_capacitance_palace")
    reference = GridInterpolator(data.grid("maxwell_capacitance", n_turns=str(turns)))(
        **point
    )[0, 0]
    model = jax.jit(meander_inductor_ground_capacitance, static_argnames=("n_turns",))
    value = model(n_turns=turns, **point)
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)
    assert 1e-15 < value < 1e-12

    def evaluate(length: float) -> jax.Array:
        return model(n_turns=turns, turn_length=length, wire_width=2.0, wire_gap=4.0)

    assert jnp.isfinite(jax.jit(jax.grad(evaluate))(150.0))
    assert jax.jit(jax.vmap(evaluate))(jnp.array([150.0, 250.0])).shape == (2,)
    assert jnp.isnan(evaluate(99.0))
