"""Regression tests for JAX JIT compilation of every registered PDK model."""

import jax
import jax.numpy as jnp
import pytest

from qpdk import PDK

MODEL_NAMES = sorted(PDK.models)
TEST_FREQUENCIES = jnp.linspace(4e9, 6e9, 5)


@pytest.mark.parametrize("name", MODEL_NAMES, ids=lambda name: name)
def test_model_is_jittable(name: str) -> None:
    """Test that a PDK model compiles and runs under ``jax.jit``.

    Models must keep layout geometry static: tracing a value into a gdsfactory
    cross-section raises ``ConcretizationTypeError`` at compile time, so this
    catches regressions without any per-model exclusions.
    """
    model = PDK.models[name]

    result = jax.jit(lambda f: model(f=f))(TEST_FREQUENCIES)
    jax.block_until_ready(result)

    assert result, f"Model {name!r} returned no S-parameters"
    assert all(jnp.all(jnp.isfinite(value)) for value in result.values()), (
        f"Model {name!r} returned non-finite S-parameters"
    )
