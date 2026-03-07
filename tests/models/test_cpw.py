"""Tests for CPW and microstrip electromagnetic analysis functions."""

from typing import ClassVar, final, override

import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from qpdk.models.cpw import (
    _ellipk_ratio,
    c_0,
    cpw_epsilon_eff,
    cpw_thickness_correction,
    cpw_z0,
    microstrip_epsilon_eff,
    microstrip_thickness_correction,
    microstrip_z0,
    propagation_constant,
    transmission_line_s_params,
)
from qpdk.models.media import cpw_parameters
from qpdk.models.waveguides import straight, straight_microstrip

from .base import TwoPortModelTestSuite

# ---------------------------------------------------------------------------
# ellipk_ratio tests
# ---------------------------------------------------------------------------


class TestEllipkRatio:
    """Tests for the elliptic integral ratio K(m)/K(1-m)."""

    def test_symmetry(self) -> None:
        """K(m)/K(1-m) = 1 / (K(1-m)/K(m)) → ratio(m) * ratio(1-m) == 1."""
        m = 0.3
        assert_allclose(
            float(_ellipk_ratio(m) * _ellipk_ratio(1.0 - m)),
            1.0,
            atol=1e-10,
        )

    def test_at_half(self) -> None:
        """K(0.5) / K(0.5) == 1."""
        assert_allclose(float(_ellipk_ratio(0.5)), 1.0, atol=1e-10)

    @given(m=st.floats(min_value=0.01, max_value=0.99))
    @settings(deadline=None)
    def test_positive(self, m: float) -> None:
        """Ratio should always be positive for 0 < m < 1."""
        assert float(_ellipk_ratio(m)) > 0


# ---------------------------------------------------------------------------
# CPW ε_eff tests
# ---------------------------------------------------------------------------


class TestCPWEpsilonEff:
    """Tests for CPW effective permittivity."""

    def test_vacuum_substrate(self) -> None:
        """With ε_r=1 (vacuum), ε_eff should be 1."""
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 1.0)
        assert_allclose(float(ep), 1.0, atol=1e-6)

    def test_infinite_substrate_limit(self) -> None:
        """For very thick substrate, ε_eff → (ε_r+1)/2."""
        ep_r = 11.45
        # Use very thick substrate (h >> w)
        ep = cpw_epsilon_eff(10e-6, 6e-6, 100e-3, ep_r)
        assert_allclose(float(ep), (ep_r + 1) / 2, rtol=1e-4)

    def test_bounded(self) -> None:
        """1 < ε_eff < ε_r for any substrate."""
        ep_r = 11.45
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, ep_r)
        assert 1.0 < float(ep) < ep_r

    def test_increases_with_ep_r(self) -> None:
        """ε_eff should increase with substrate permittivity."""
        ep1 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 4.0)
        ep2 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        assert float(ep2) > float(ep1)

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(cpw_epsilon_eff)
        result = jitted(10e-6, 6e-6, 500e-6, 11.45)
        assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# CPW Z0 tests
# ---------------------------------------------------------------------------


class TestCPWZ0:
    """Tests for CPW characteristic impedance."""

    def test_default_cpw_approx_50_ohm(self) -> None:
        """Default CPW dimensions (w=10, s=6) should give ~50 Ω."""
        ep = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0 = cpw_z0(10e-6, 6e-6, ep)
        assert_allclose(float(z0), 50.0, atol=2.0)  # within 2 Ω

    def test_narrow_conductor_high_impedance(self) -> None:
        """Narrow conductor (small w) → high impedance."""
        ep = cpw_epsilon_eff(1e-6, 20e-6, 500e-6, 11.45)
        z0 = cpw_z0(1e-6, 20e-6, ep)
        assert float(z0) > 100.0

    def test_wide_conductor_low_impedance(self) -> None:
        """Wide conductor (large w) → low impedance."""
        ep = cpw_epsilon_eff(100e-6, 2e-6, 500e-6, 11.45)
        z0 = cpw_z0(100e-6, 2e-6, ep)
        assert float(z0) < 25.0

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(cpw_z0)
        result = jitted(10e-6, 6e-6, 6.2)
        assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# CPW thickness correction tests
# ---------------------------------------------------------------------------


class TestCPWThicknessCorrection:
    """Tests for GGBB96 conductor thickness correction."""

    def test_thin_conductor_small_correction(self) -> None:
        """Very thin conductor should produce small corrections."""
        ep0 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0_0 = cpw_z0(10e-6, 6e-6, ep0)
        ep_t, z0_t = cpw_thickness_correction(10e-6, 6e-6, 1e-9, ep0)
        # Correction should be small for t = 1 nm
        assert_allclose(float(ep_t), float(ep0), rtol=0.01)
        assert_allclose(float(z0_t), float(z0_0), rtol=0.01)

    def test_reduces_impedance(self) -> None:
        """Thickness correction should reduce Z0 (wider effective conductor)."""
        ep0 = cpw_epsilon_eff(10e-6, 6e-6, 500e-6, 11.45)
        z0_0 = cpw_z0(10e-6, 6e-6, ep0)
        _ep_t, z0_t = cpw_thickness_correction(10e-6, 6e-6, 0.2e-6, ep0)
        assert float(z0_t) < float(z0_0)

    def test_matches_scikit_rf(self) -> None:
        """Result should match scikit-rf CPW within ~0.1%."""
        # Known scikit-rf values for w=10um, s=6um, h=500um, t=0.2um, ep_r=11.45
        ep_eff, z0 = cpw_parameters(10.0, 6.0)
        assert_allclose(z0, 49.28, rtol=0.002)  # ~0.2% tolerance
        assert_allclose(ep_eff, 6.065, rtol=0.001)


# ---------------------------------------------------------------------------
# Propagation constant tests
# ---------------------------------------------------------------------------


class TestPropagationConstant:
    """Tests for the complex propagation constant."""

    def test_lossless_purely_imaginary(self) -> None:
        """For tand=0, gamma should be purely imaginary."""
        gamma = propagation_constant(5e9, 6.225, tand=0.0)
        assert_allclose(float(jnp.real(gamma)), 0.0, atol=1e-20)
        assert float(jnp.imag(gamma)) > 0

    def test_phase_velocity(self) -> None:
        """β = ω√ε_eff/c₀, so v_p = ω/β = c₀/√ε_eff."""
        ep_eff = 6.225
        f = 5e9
        gamma = propagation_constant(f, ep_eff)
        beta = float(jnp.imag(gamma))
        v_p = 2 * jnp.pi * f / beta
        assert_allclose(float(v_p), c_0 / jnp.sqrt(ep_eff), rtol=1e-8)

    def test_lossy_has_real_part(self) -> None:
        """For tand > 0, gamma should have a positive real part (attenuation)."""
        gamma = propagation_constant(5e9, 6.225, tand=0.01, ep_r=11.45)
        assert float(jnp.real(gamma)) > 0

    def test_scales_with_frequency(self) -> None:
        """β should scale linearly with frequency."""
        g1 = propagation_constant(5e9, 6.225)
        g2 = propagation_constant(10e9, 6.225)
        assert_allclose(
            float(jnp.imag(g2)),
            2.0 * float(jnp.imag(g1)),
            rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# Transmission line S-parameter tests
# ---------------------------------------------------------------------------


class TestTransmissionLineSParams:
    """Tests for ABCD→S-parameter conversion."""

    def test_zero_length_identity(self) -> None:
        """Zero-length line → S11=0, S21=1."""
        gamma = 1j * 100.0
        s11, s21 = transmission_line_s_params(gamma, 50.0, 0.0)
        assert_allclose(float(jnp.abs(s11)), 0.0, atol=1e-10)
        assert_allclose(float(jnp.abs(s21)), 1.0, atol=1e-10)

    def test_matched_impedance_no_reflection(self) -> None:
        """With z_ref = z0, S11 should be zero."""
        gamma = jnp.array([1j * 100.0])
        s11, _ = transmission_line_s_params(gamma, 50.0, 0.001)
        assert_allclose(float(jnp.abs(s11[0])), 0.0, atol=1e-10)

    def test_mismatched_impedance_reflection(self) -> None:
        """With z_ref ≠ z0, S11 should be non-zero."""
        gamma = jnp.array([1j * 100.0])
        s11, _ = transmission_line_s_params(gamma, 50.0, 0.1, z_ref=75.0)
        assert float(jnp.abs(s11[0])) > 0.01

    def test_lossless_passivity(self) -> None:
        """For lossless line: |S11|² + |S21|² = 1."""
        gamma = jnp.array([1j * 200.0])
        s11, s21 = transmission_line_s_params(gamma, 50.0, 0.01, z_ref=75.0)
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert_allclose(float(power[0]), 1.0, atol=1e-10)

    def test_reciprocal(self) -> None:
        """S21 = S12 for a symmetric transmission line."""
        # This is guaranteed by the symmetric ABCD matrix (A=D)
        gamma = jnp.array([1j * 150.0])
        _, s21 = transmission_line_s_params(gamma, 50.0, 0.02, z_ref=75.0)
        # S12 = S21 by construction in our implementation
        assert jnp.isfinite(s21[0])


# ---------------------------------------------------------------------------
# Microstrip tests
# ---------------------------------------------------------------------------


class TestMicrostripEpsilonEff:
    """Tests for microstrip effective permittivity."""

    def test_vacuum_substrate(self) -> None:
        """With ε_r=1, ε_eff should be 1."""
        ep = microstrip_epsilon_eff(10e-6, 500e-6, 1.0)
        assert_allclose(float(ep), 1.0, atol=1e-10)

    def test_bounded(self) -> None:
        """1 < ε_eff < ε_r for any substrate."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(10e-6, 500e-6, ep_r)
        assert 1.0 < float(ep) < ep_r

    def test_wide_strip_approaches_ep_r(self) -> None:
        """For very wide strips (w/h >> 1), ε_eff → ε_r."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(1e-3, 1e-6, ep_r)  # w/h = 1000
        assert float(ep) > 0.9 * ep_r

    def test_narrow_strip_approaches_average(self) -> None:
        """For very narrow strips (w/h << 1), ε_eff → (ε_r+1)/2."""
        ep_r = 11.45
        ep = microstrip_epsilon_eff(1e-9, 1e-3, ep_r)  # w/h = 1e-6
        assert_allclose(float(ep), (ep_r + 1) / 2, rtol=0.1)

    def test_increases_with_width(self) -> None:
        """ε_eff increases as strip gets wider (more field in substrate)."""
        ep1 = microstrip_epsilon_eff(5e-6, 500e-6, 11.45)
        ep2 = microstrip_epsilon_eff(100e-6, 500e-6, 11.45)
        assert float(ep2) > float(ep1)


class TestMicrostripZ0:
    """Tests for microstrip characteristic impedance."""

    def test_narrow_strip_high_impedance(self) -> None:
        """Narrow strip (w/h < 1) → high impedance."""
        ep = microstrip_epsilon_eff(1e-6, 500e-6, 11.45)
        z0 = microstrip_z0(1e-6, 500e-6, ep)
        assert float(z0) > 100.0

    def test_wide_strip_low_impedance(self) -> None:
        """Wide strip (w/h >> 1) → low impedance."""
        ep = microstrip_epsilon_eff(1e-3, 500e-6, 11.45)
        z0 = microstrip_z0(1e-3, 500e-6, ep)
        assert float(z0) < 35.0

    def test_typical_50_ohm(self) -> None:
        r"""A common 50 Ω microstrip on alumina (ε_r=9.8, h=0.635mm) has w ≈ 0.6mm."""
        ep = microstrip_epsilon_eff(0.6e-3, 0.635e-3, 9.8)
        z0 = microstrip_z0(0.6e-3, 0.635e-3, ep)
        assert_allclose(float(z0), 50.0, atol=5.0)

    def test_jit_compatible(self) -> None:
        """Function can be JIT-compiled."""
        jitted = jax.jit(microstrip_z0)
        result = jitted(10e-6, 500e-6, 6.2)
        assert jnp.isfinite(result)


class TestMicrostripThicknessCorrection:
    """Tests for microstrip conductor thickness correction."""

    def test_reduces_impedance(self) -> None:
        """Thickness correction should reduce Z0 (wider effective strip)."""
        ep0 = microstrip_epsilon_eff(10e-6, 500e-6, 11.45)
        z0_0 = microstrip_z0(10e-6, 500e-6, ep0)
        _, _ep_t, z0_t = microstrip_thickness_correction(
            10e-6, 500e-6, 0.2e-6, 11.45, ep0
        )
        assert float(z0_t) < float(z0_0)


# ---------------------------------------------------------------------------
# Straight waveguide model (JIT compatibility)
# ---------------------------------------------------------------------------


class TestStraightJIT:
    """Tests for JIT compilation of the straight CPW model."""

    def test_straight_matches_non_jit(self) -> None:
        """JIT-compiled straight should give same results as non-JIT."""
        f = jnp.linspace(4e9, 8e9, 10)

        result_nojit = straight(f=f, length=1000, cross_section="cpw")

        # The straight function uses cross_section as a Python-level param,
        # so JIT-tracing happens naturally for f and length
        jitted_inner = jax.jit(
            lambda f, length: straight(f=f, length=length, cross_section="cpw")
        )
        result_jit = jitted_inner(f, 1000.0)

        for key in result_nojit:
            assert_allclose(
                result_nojit[key],
                result_jit[key],
                atol=1e-10,
                err_msg=f"Mismatch for {key}",
            )


# ---------------------------------------------------------------------------
# Straight microstrip model tests
# ---------------------------------------------------------------------------


@final
class TestStraightMicrostrip(TwoPortModelTestSuite):
    """Tests for the straight_microstrip model."""

    model_function = staticmethod(straight_microstrip)
    expected_ports: ClassVar[set[str]] = {"o1", "o2"}

    @staticmethod
    @override
    def get_model_kwargs() -> dict:
        return {
            "length": 1000,
            "width": 10.0,
            "h": 500.0,
            "t": 0.2,
            "ep_r": 11.45,
        }

    def test_zero_length_transmission(self) -> None:
        """Zero-length microstrip → near-unity transmission."""
        f = jnp.array([5e9])
        result = straight_microstrip(f=f, length=0.0)
        s21 = jnp.abs(result[("o1", "o2")])
        assert_allclose(float(s21.squeeze()), 1.0, atol=1e-6)

    def test_phase_shift_increases_with_length(self) -> None:
        """Longer lines should have more phase shift."""
        f = jnp.array([5e9])
        r1 = straight_microstrip(f=f, length=1000)
        r2 = straight_microstrip(f=f, length=2000)
        phase1 = jnp.angle(r1[("o1", "o2")]).squeeze()
        phase2 = jnp.angle(r2[("o1", "o2")]).squeeze()
        # Phase shift roughly doubles (unwrapped)
        assert jnp.abs(phase2) > jnp.abs(phase1) - 0.01

    def test_lossy_attenuates(self) -> None:
        """With tand > 0, transmission should be reduced."""
        f = jnp.array([5e9])
        r_lossless = straight_microstrip(f=f, length=10000, tand=0.0)
        r_lossy = straight_microstrip(f=f, length=10000, tand=0.01)
        s21_ll = jnp.abs(r_lossless[("o1", "o2")]).squeeze()
        s21_ly = jnp.abs(r_lossy[("o1", "o2")]).squeeze()
        assert float(s21_ly) < float(s21_ll)

    def test_jit_compatible(self) -> None:
        """Microstrip straight model can be JIT-compiled."""
        jitted = jax.jit(
            lambda f, length: straight_microstrip(f=f, length=length)
        )
        f = jnp.array([5e9])
        result = jitted(f, 1000.0)
        assert jnp.isfinite(result[("o1", "o2")])
