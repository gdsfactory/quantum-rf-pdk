"""Tests for qpdk.models.ssh - SSH coupled-resonator chain models."""

from typing import ClassVar, final, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import find_peaks

from qpdk.models.ssh import (
    ssh_band_edges,
    ssh_bands,
    ssh_chain_spectrum,
    ssh_circuit_band_edges,
    ssh_parameters,
    ssh_resonator_chain,
)

from .base import BaseModelTestSuite, TwoPortModelTestSuite

# Test-structure values of Kim et al. (2021), App. C.
L_0 = 1.9e-9
C_0 = 253e-15
C_LARGE = 33e-15
C_SMALL = 17e-15


@final
class TestSSHResonatorChain(TwoPortModelTestSuite):
    """Standard S-parameter checks for the default chain."""

    model_function = staticmethod(ssh_resonator_chain)
    freq_range: ClassVar[tuple[float, float]] = (5.5e9, 8e9)


@final
class TestSSHResonatorChainTapped(BaseModelTestSuite):
    """Standard S-parameter checks for a short chain with two taps."""

    model_function = staticmethod(ssh_resonator_chain)
    expected_ports: ClassVar[set[str]] = {"o1", "o2", "o3", "o4"}
    freq_range: ClassVar[tuple[float, float]] = (5.5e9, 8e9)

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        """Tap the first and last site of a 3-cell chain.

        Returns:
            Model keyword arguments.
        """
        return {"n_cells": 3, "tap_sites": (0, 5), "tap_capacitance": 5e-15}

    @override
    def test_default_parameters(self) -> None:
        """Defaults have no taps, so check the tapped ports with kwargs instead."""
        result = self._call_model(**self.get_model_kwargs())
        assert set(result) == self.get_expected_keys()


class TestSSHParameters:
    """Tests for the Eq. (B1) circuit-to-SSH mapping."""

    @staticmethod
    def test_device_i() -> None:
        """Device I of Kim et al. (2021), App. C: J/2π = 356 MHz, δ = 0.256."""
        omega_0, coupling, dimerisation = ssh_parameters(
            inductance=1.9e-9,
            capacitance=250e-15,
            intra_cell_capacitance=35e-15,
            inter_cell_capacitance=19.2e-15,
            intra_cell_mutual_inductance=-38e-12,
            inter_cell_mutual_inductance=-32e-12,
        )
        assert_allclose(coupling / (2 * jnp.pi), 356e6, rtol=1e-3)
        assert_allclose(dimerisation, 0.256, atol=1e-3)
        assert_allclose(omega_0, 1 / jnp.sqrt(1.9e-9 * 304.2e-15))

    @staticmethod
    def test_swapping_couplings_flips_phase() -> None:
        """Swapping C_v and C_w flips the sign of δ and keeps ω_0 and J."""
        trivial = ssh_parameters(L_0, C_0, C_LARGE, C_SMALL)
        topological = ssh_parameters(L_0, C_0, C_SMALL, C_LARGE)
        assert trivial[2] > 0
        assert_allclose(topological[2], -trivial[2])
        assert_allclose(topological[:2], trivial[:2])


class TestSSHBands:
    """Tests for the ideal SSH band structure."""

    @staticmethod
    def test_bands_span_band_edges() -> None:
        """ω±(k) over the Brillouin zone spans exactly the band edges."""
        omega_0, coupling, dimerisation = 1.0, 0.1, -0.3
        lower, upper = ssh_bands(
            jnp.linspace(-jnp.pi, jnp.pi, 1001), omega_0, coupling, dimerisation
        )
        edges = ssh_band_edges(omega_0, coupling, dimerisation)
        assert_allclose(
            [lower.min(), lower.max(), upper.min(), upper.max()], edges, atol=1e-12
        )

    @staticmethod
    def test_circuit_edges_approach_ideal_at_weak_coupling() -> None:
        """Eq. (A16) edges approach the Eq. (B1) edges to second order."""
        for scale in (0.1, 0.01):
            c_v, c_w = scale * C_LARGE, scale * C_SMALL
            ideal = ssh_band_edges(*ssh_parameters(L_0, C_0, c_v, c_w))
            exact = ssh_circuit_band_edges(L_0, C_0, c_v, c_w)
            x = (c_v + c_w) / (C_0 + c_v + c_w)
            omega_0 = ideal.mean()
            assert jnp.max(jnp.abs(exact - ideal)) / omega_0 < x**2


class TestSSHChainSpectrum:
    """Tests for the finite-chain tight-binding spectrum."""

    @staticmethod
    @pytest.mark.parametrize(("dimerisation", "expected_in_gap"), [(0.3, 0), (-0.3, 2)])
    def test_edge_modes_in_gap(dimerisation: float, expected_in_gap: int) -> None:
        """Only the topological phase has two modes in the middle gap."""
        omega_0, coupling = 1.0, 0.1
        frequencies, vectors = ssh_chain_spectrum(10, omega_0, coupling, dimerisation)
        edges = ssh_band_edges(omega_0, coupling, dimerisation)
        in_gap = (frequencies > edges[1]) & (frequencies < edges[2])
        assert int(in_gap.sum()) == expected_in_gap
        if expected_in_gap:
            # Hybridisation splits the pair by ~J ((1+δ)/(1-δ))^N around ω_0.
            assert_allclose(frequencies[in_gap], omega_0, atol=1e-2 * coupling)
            # Each edge mode decays into the bulk as r^n on its own sublattice,
            # r = (1+δ)/(1-δ), so a fraction 1 - r² of it sits on the end site.
            weight = jnp.sum(jnp.abs(vectors[:, in_gap]) ** 2, axis=1)
            r = (1 + dimerisation) / (1 - dimerisation)
            assert_allclose(weight[0] + weight[-1], 2 * (1 - r**2), rtol=1e-2)

    @staticmethod
    def test_jittable() -> None:
        """The spectrum jits with traced couplings."""
        spectrum = jax.jit(lambda j: ssh_chain_spectrum(4, 1.0, j, -0.2)[0])
        assert spectrum(0.1).shape == (8,)


def _transmission_peaks(
    n_cells: int, c_v: float, c_w: float, f: jax.Array
) -> np.ndarray:
    s21 = ssh_resonator_chain(
        f=f,
        n_cells=n_cells,
        capacitance=C_0,
        intra_cell_capacitance=c_v,
        inter_cell_capacitance=c_w,
        port_capacitance=c_w,
    )["o2", "o1"]
    peaks, _ = find_peaks(np.abs(np.asarray(s21)), height=1e-3)
    return np.asarray(f)[peaks]


class TestSSHResonatorChainGradient:
    """Reverse-mode gradients through the SAX model."""

    @staticmethod
    def test_grad_wrt_coupling() -> None:
        """jax.grad of the transmission w.r.t. C_v is finite and nonzero."""
        f = jnp.linspace(6e9, 7e9, 51)

        def power(c_v: float) -> jax.Array:
            s21 = ssh_resonator_chain(f=f, intra_cell_capacitance=c_v)["o2", "o1"]
            return jnp.sum(jnp.abs(s21) ** 2)

        grad = jax.grad(power)(C_LARGE)
        assert jnp.isfinite(grad)
        assert grad != 0


class TestSSHChainBandEdges:
    """Compare SAX transmission peaks with analytical band edges."""

    @staticmethod
    def test_weak_coupling_band_edges() -> None:
        r"""SAX band edges match Eq. (A16) and the ideal SSH edges.

        At :math:`(C_v + C_w)/C_\Sigma = 0.056` and :math:`N = 20`, the outermost
        transmission peaks of each band sit within 3 MHz of the exact
        infinite-chain edges (finite-size offset). The ideal SSH edges differ
        by up to 8 MHz, i.e. within 0.15 % of :math:`\omega_0`, the expected
        second-order weak-coupling error.
        """
        n_cells, c_v, c_w = 20, 10e-15, 5e-15
        omega_0, coupling, dimerisation = ssh_parameters(L_0, C_0, c_v, c_w)
        ideal = np.asarray(ssh_band_edges(omega_0, coupling, dimerisation)) / (
            2 * np.pi
        )
        exact = np.asarray(ssh_circuit_band_edges(L_0, C_0, c_v, c_w)) / (2 * np.pi)
        f_0 = float(omega_0) / (2 * np.pi)

        f = jnp.linspace(ideal[0] - 0.2e9, ideal[3] + 0.2e9, 40001)
        peaks = _transmission_peaks(n_cells, c_v, c_w, f)
        assert len(peaks) == 2 * n_cells
        lower, upper = peaks[peaks < f_0], peaks[peaks > f_0]
        sax_edges = np.array([lower.min(), lower.max(), upper.min(), upper.max()])

        assert_allclose(sax_edges, exact, atol=3e6)
        assert_allclose(sax_edges, ideal, atol=1.5e-3 * f_0)


class TestSSHChainEdgeModes:
    """Edge modes in the SAX model appear only in the topological phase.

    The ports are compensated (``port_capacitance`` = :math:`C_w`), so the end
    sites are not detuned. A detuned end site can pull a trivial-phase mode
    into the gap or push the edge mode out of it.
    """

    @staticmethod
    def _gap_reflection_winding(n_cells: int, c_v: float, c_w: float) -> float:
        """Total variation of the S11 phase over the inner 90 % of the gap, in turns.

        Each mode inside the gap winds the reflection phase by about one turn.

        Returns:
            Phase variation divided by 2π.
        """
        omega_0, coupling, dimerisation = ssh_parameters(L_0, C_0, c_v, c_w)
        edges = np.asarray(ssh_band_edges(omega_0, coupling, dimerisation)) / (
            2 * np.pi
        )
        margin = 0.05 * (edges[2] - edges[1])
        f = jnp.linspace(edges[1] + margin, edges[2] - margin, 4001)
        s11 = ssh_resonator_chain(
            f=f,
            n_cells=n_cells,
            capacitance=C_0,
            intra_cell_capacitance=c_v,
            inter_cell_capacitance=c_w,
            port_capacitance=c_w,
        )["o1", "o1"]
        phase = np.unwrap(np.angle(np.asarray(s11)))
        return float(np.abs(np.diff(phase)).sum() / (2 * np.pi))

    @pytest.mark.parametrize("n_cells", [4, 8])
    def test_trivial_has_no_edge_mode(self, n_cells: int) -> None:
        """Trivial phase: S11 phase stays flat across the gap."""
        assert self._gap_reflection_winding(n_cells, C_LARGE, C_SMALL) < 0.1

    @pytest.mark.parametrize("n_cells", [4, 8])
    def test_topological_has_edge_mode(self, n_cells: int) -> None:
        """Topological phase: an edge mode winds the S11 phase in the gap."""
        assert self._gap_reflection_winding(n_cells, C_SMALL, C_LARGE) > 0.5


class TestSSHResonatorChainValidation:
    """Invalid arguments raise ValueError."""

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_cells": 0},
            {"n_cells": 2, "tap_sites": (4,)},
            {"n_cells": 2, "tap_sites": (-1,)},
            {"n_cells": 2, "tap_sites": (1, 1)},
            {"n_cells": 2, "tap_sites": (1.5,)},
        ],
    )
    def test_invalid(kwargs: dict) -> None:
        """Out-of-range cell counts and taps are rejected."""
        with pytest.raises(ValueError, match=r"n_cells|tap_sites"):
            ssh_resonator_chain(**kwargs)
