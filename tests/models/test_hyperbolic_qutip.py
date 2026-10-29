"""Verify that the hyperbolic tight-binding Hamiltonian works as a QuTiP operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from qpdk.hyperbolic import hyperbolic_tiling
from qpdk.models.hyperbolic import tight_binding_hamiltonian

qt = pytest.importorskip("qutip")
qutip_jax = pytest.importorskip("qutip_jax")

_TILING = hyperbolic_tiling(7, 3, shells=1)


def _hamiltonian(t: ArrayLike = -0.1, onsite: ArrayLike = 7.5) -> jax.Array:
    return tight_binding_hamiltonian(_TILING.line_graph_adjacency, t, onsite)


@pytest.mark.parametrize("dtype", [None, "jax"])
def test_qutip_spectrum_matches_line_graph(dtype: str | None) -> None:
    """The docstring example: QuTiP finds the flat band ``E - V`` times at ``onsite + 2t``."""
    h = _hamiltonian()
    H = qt.Qobj(h, dtype=dtype, isherm=True) if dtype else qt.Qobj(np.asarray(h))
    assert H.isherm
    assert H.shape == (_TILING.num_edges, _TILING.num_edges)
    energies = np.asarray(H.eigenenergies()).real
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(np.asarray(h)), atol=1e-12)
    flat = np.isclose(energies, 7.5 + 2 * -0.1, atol=1e-9).sum()
    assert flat == _TILING.num_edges - _TILING.num_vertices


def test_qutip_jax_eigenstates_under_jit() -> None:
    """The JAX-backed operator jits and differentiates through the hopping."""

    def lowest(t: jax.Array) -> jax.Array:
        H = qt.Qobj(_hamiltonian(t), dtype="jax", isherm=True)
        return H.eigenenergies(eigvals=1).real[0]

    t = jnp.asarray(-0.1)
    np.testing.assert_allclose(jax.jit(lowest)(t), 7.5 + 2 * t, atol=1e-12)
    # The flat band is the bottom of the spectrum, so d(E_min)/dt = 2.
    np.testing.assert_allclose(jax.jit(jax.grad(lowest))(t), 2.0, atol=1e-9)


def test_qutip_single_excitation_dynamics() -> None:
    """A photon placed on one resonator spreads but its norm is conserved."""
    H = qt.Qobj(2 * np.pi * np.asarray(_hamiltonian()))
    psi0 = qt.basis(_TILING.num_edges, 0)
    result = qt.sesolve(H, psi0, np.linspace(0, 5, 11))
    final = result.states[-1]
    np.testing.assert_allclose(final.norm(), 1.0, atol=1e-6)
    assert abs(final.overlap(psi0)) < 1
