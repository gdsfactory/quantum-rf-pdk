"""Stored facing-pad data preserves the complete two-terminal admittance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from qpdk.models import coupling_pad_pair
from qpdk.models.datasets import Dataset, GridInterpolator


def test_refinement_and_held_out_evidence() -> None:
    data = Dataset("coupling_pad_palace")
    lookup = GridInterpolator(data.grid("maxwell_capacitance", allow_missing=False))
    evidence = data.metadata.provenance["validation"]
    assert data.metadata.terminals == ("o1", "o2")
    assert len(evidence["samples"]) == 7
    for sample in evidence["samples"]:
        actual = lookup(**sample["point"])
        reference = jnp.asarray(sample["reference_maxwell_F"])
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=1e-17)
        np.testing.assert_allclose(
            actual.sum(axis=-1), reference.sum(axis=-1), rtol=0.02, atol=1e-17
        )


@settings(deadline=None, max_examples=10)
@given(
    length=st.floats(40, 120),
    width=st.floats(5, 20),
    gap=st.floats(4, 10),
    offset=st.floats(-12, 12),
)
def test_full_admittance_and_jax(**point: float) -> None:
    lookup = GridInterpolator(
        Dataset("coupling_pad_palace").grid("maxwell_capacitance")
    )
    frequencies = jnp.asarray([1e9, 5e9, 10e9])
    values = jax.jit(coupling_pad_pair)(f=frequencies, z_ref=75.0, **point)
    ports = ("o1", "o2")
    assert set(values) == {(a, b) for a in ports for b in ports}
    matrix = jnp.asarray([[values[a, b] for b in ports] for a in ports]).transpose(
        2, 0, 1
    )
    identity = jnp.broadcast_to(jnp.eye(2), matrix.shape)
    admittance = jnp.linalg.solve(identity + matrix, identity - matrix) / 75.0
    expected = 2j * jnp.pi * frequencies[:, None, None] * lookup(**point)
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        matrix @ matrix.conj().swapaxes(-1, -2), identity, atol=1e-12
    )
    assert jnp.all(jnp.real(lookup(**point).sum(axis=-1)) > 0)


def test_gradient_vmap_and_domain() -> None:
    def reflection(length: float) -> jax.Array:
        return jnp.imag(coupling_pad_pair(f=5e9, length=length)["o1", "o1"])

    assert jnp.isfinite(jax.jit(jax.grad(reflection))(60.0))
    assert jax.jit(jax.vmap(reflection))(jnp.asarray([60.0, 100.0])).shape == (2,)
    assert float(reflection(60.0)) != pytest.approx(float(reflection(100.0)))
    assert jnp.isnan(coupling_pad_pair(length=39.0)["o1", "o1"])
