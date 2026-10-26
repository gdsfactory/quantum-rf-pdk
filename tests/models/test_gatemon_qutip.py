"""Verify the optional QuTiP boundary without changing the JAX spectrum API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from qpdk.models.gatemon import (
    TransmissionsLike,
    gatemon_energies,
    gatemon_hamiltonian,
)

qt = pytest.importorskip("qutip")
qutip_jax = pytest.importorskip("qutip_jax")


def _qutip_levels(τ: TransmissionsLike, n_g: ArrayLike) -> jax.Array:
    H = qt.Qobj(
        gatemon_hamiltonian(transmissions=jnp.asarray(τ), n_g=n_g, n_max=12),
        dtype="jax",
        isherm=True,
    )
    E = H.eigenenergies(eigvals=4).real
    return E - E[0]


@pytest.mark.parametrize("τ", [(0.0,), (0.2,), (0.99,), (1.0,), (0.3, 0.8)])
def test_qutip_jax_spectrum(τ: tuple[float, ...]) -> None:
    """QuTiP and JAX agree for charge sweeps, including open and ballistic links."""
    n_g = jnp.linspace(0.0, 0.5, 4)
    actual = jax.jit(jax.vmap(lambda ng: _qutip_levels(τ, ng)))(n_g)
    expected = jax.vmap(
        lambda ng: gatemon_energies(transmissions=τ, n_g=ng, n_max=12, n_levels=4)
    )(n_g)
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_qutip_jax_tuning_gradient() -> None:
    """The QuTiP eigenenergy path preserves gradients through the weak link."""

    def frequency(τ: ArrayLike, n_g: ArrayLike) -> jax.Array:
        return _qutip_levels(τ, n_g)[1]

    parameters = jnp.array([0.8, 0.15])
    gradient = jax.jit(jax.grad(frequency, argnums=(0, 1)))(*parameters)
    step = 1e-5
    for axis, derivative in enumerate(gradient):
        displacement = jnp.eye(2)[axis] * step
        finite_difference = (
            frequency(*(parameters + displacement))
            - frequency(*(parameters - displacement))
        ) / (2 * step)
        assert jnp.isfinite(derivative)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-4, atol=1e-8)


def test_qutip_jax_eigenstates() -> None:
    """QuTiP supplies normalized eigenstates for the same charge-basis operator."""
    H = qt.Qobj(gatemon_hamiltonian(n_max=12), dtype="jax", isherm=True)
    assert isinstance(H.data, qutip_jax.JaxArray)
    E, states = H.eigenstates(eigvals=4)
    np.testing.assert_allclose(
        E.real - E[0].real, gatemon_energies(n_max=12, n_levels=4)
    )
    for energy, state in zip(E, states, strict=True):
        assert isinstance(state.data, qutip_jax.JaxArray)
        np.testing.assert_allclose(state.norm(), 1.0, atol=1e-12)
        assert (H @ state - energy * state).norm() < 1e-10
