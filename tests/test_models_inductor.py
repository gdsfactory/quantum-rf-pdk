"""Tests for inductor SAX models (qpdk/models/inductor.py)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.cpw import cpw_ep_r_from_cross_section, get_cpw_dimensions
from qpdk.models.inductor import (
    lumped_element_resonator,
    meander_inductor,
    meander_inductor_inductance_analytical,
)

MAX_EXAMPLES = 50


# ---------------------------------------------------------------------------
# Reference implementation (pure Python, transcribes the same published
# Chen et al. 2023 formulas as qpdk/models/inductor.py)
# ---------------------------------------------------------------------------
def _strip_self_inductance(
    length_m: float, width_m: float, thickness_m: float
) -> float:
    """Self-inductance of a rectangular strip (Chen et al. 2023), in H."""
    mu0 = 4e-7 * math.pi
    return (mu0 * length_m / (2 * math.pi)) * (
        math.log(2 * length_m / (width_m + thickness_m))
        + 0.5
        + (width_m + thickness_m) / (3 * length_m)
    )


def _mutual_inductance_strips(length_m: float, d_m: float) -> float:
    """Mutual inductance of two parallel strips (Chen et al. 2023), in H."""
    mu0 = 4e-7 * math.pi
    return (mu0 * length_m / (2 * math.pi)) * (
        math.log(length_m / d_m + math.sqrt(1 + (length_m / d_m) ** 2))
        - math.sqrt(1 + (d_m / length_m) ** 2)
        + d_m / length_m
    )


def _meander_reference_inductance(
    n_turns: int,
    *,
    turn_length: float = 200.0,
    wire_width: float = 2.0,
    wire_gap: float = 2.0,
    sheet_inductance: float = 1e-12,
    thickness: float = 0.2,
    max_offset: int | None = None,
) -> float:
    """Explicit reference for the meander inductance, in H.

    Sums the mutual-inductance series term by term in pure Python. The strip
    formulas below are transcriptions of the same published formulas the
    implementation uses, so this reference discriminates the summation bounds
    and term assembly, not the strip formulas themselves. ``max_offset``
    emulates the legacy hardcoded ``jnp.arange(1, 501)`` limit when set.

    Returns:
        Total meander inductance in Henries.
    """
    l_m = turn_length * 1e-6
    w_m = wire_width * 1e-6
    g_m = wire_gap * 1e-6
    t_m = thickness * 1e-6
    p_m = w_m + g_m

    last_k = n_turns - 1 if max_offset is None else min(n_turns - 1, max_offset)
    l_m_sum = sum(
        (n_turns - k) * (-1.0) ** k * _mutual_inductance_strips(l_m, k * p_m)
        for k in range(1, last_k + 1)
    )
    l_g = (
        n_turns * _strip_self_inductance(l_m, w_m, t_m)
        + 2 * l_m_sum
        + (n_turns - 1) * _strip_self_inductance(p_m, w_m, t_m)
    )
    total_length_um = n_turns * turn_length + max(0, n_turns - 1) * wire_gap
    l_k = sheet_inductance * total_length_um / wire_width
    return l_g + l_k


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------
# Use realistic dimensions (>= 1.0 µm) to avoid numerical instabilities
# and ensure formulas are within their validity domain (length > width)
dim_st = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
sheet_inductance_st = st.floats(
    min_value=1e-13, max_value=1e-9, allow_nan=False, allow_infinity=False
)
frequency_st = st.floats(
    min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# meander_inductor_inductance_analytical
# ---------------------------------------------------------------------------
class TestMeanderInductorInductanceAnalytical:
    """Unit tests for the analytical inductance formula."""

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=200),
        turn_length=dim_st,
        wire_width=st.floats(min_value=0.5, max_value=20.0),
        wire_gap=dim_st,
        sheet_inductance=sheet_inductance_st,
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_is_positive(
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
    ) -> None:
        """Inductance must always be strictly positive."""
        # Formula assumes turn_length is significant compared to width
        assume(turn_length > 2 * wire_width)
        L = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        assert float(L) > 0

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=100),
        turn_length=dim_st,
        wire_width=st.floats(min_value=1.0, max_value=10.0),
        wire_gap=dim_st,
        sheet_inductance=sheet_inductance_st,
        scale=st.floats(
            min_value=1.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_scales_with_sheet_inductance(
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
        scale: float,
    ) -> None:
        """Scaling the sheet inductance must scale the kinetic part of L correctly."""
        assume(turn_length > 2 * wire_width)
        L1 = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        L2 = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance * scale,
        )
        # Expected difference is exactly (scale - 1) * L_kinetic
        total_length_um = n_turns * turn_length + max(0, n_turns - 1) * wire_gap
        n_squares = total_length_um / wire_width
        expected_diff = (scale - 1.0) * sheet_inductance * n_squares
        actual_diff = float(L2) - float(L1)
        assert actual_diff == pytest.approx(expected_diff, rel=1e-5)

    @staticmethod
    @given(
        n_turns=st.integers(min_value=2, max_value=100),
        turn_length=st.floats(min_value=100.0, max_value=1000.0),
        wire_width=st.floats(min_value=1.0, max_value=10.0),
        wire_gap=st.floats(min_value=1.0, max_value=5.0),
        sheet_inductance=st.floats(min_value=1e-12, max_value=1e-9),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_approx_monotonic_with_n_turns(
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
    ) -> None:
        """Adding more turns should generally increase the total inductance."""
        pitch = wire_width + wire_gap
        # Analytical formulas are valid when fingers are long compared to pitch.
        assume(turn_length > 20 * pitch)
        L_base = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        L_more = meander_inductor_inductance_analytical(
            n_turns=n_turns + 1,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        L_base_f = float(L_base)
        L_more_f = float(L_more)
        # We check for approximate monotonicity. In some extreme regimes allowed by the
        # assume constraints, geometric mutual inductance suppression can cause a small dip.
        assert L_more_f == pytest.approx(L_base_f, rel=2e-2) or L_more_f > L_base_f

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=100),
        turn_length=dim_st,
        wire_gap=dim_st,
        sheet_inductance=sheet_inductance_st,
        wire_width=st.floats(min_value=1.0, max_value=10.0),
        scale=st.floats(
            min_value=1.1, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_increases_with_turn_length(
        n_turns: int,
        turn_length: float,
        wire_gap: float,
        sheet_inductance: float,
        wire_width: float,
        scale: float,
    ) -> None:
        """Longer turns must give higher inductance."""
        assume(turn_length > 2 * wire_width)
        L1 = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        L2 = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length * scale,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        assert float(L2) > float(L1)

    @staticmethod
    def test_single_turn_no_gap_contribution() -> None:
        """With n_turns=1 the wire_gap term vanishes from the kinetic part."""
        sheet_ind = 1e-12
        L = meander_inductor_inductance_analytical(
            n_turns=1,
            turn_length=100.0,
            wire_width=2.0,
            wire_gap=999.0,  # large gap — should not matter
            sheet_inductance=sheet_ind,
            thickness=0.2,
        )
        # Expected kinetic: 1e-12 * (100 / 2) = 50 pH
        # Plus geometric self-inductance of 100 um strip (width 2 um, thickness 0.2 um)
        l_m = 100e-6
        w_m = 2e-6
        t_m = 0.2e-6
        L_g = (4e-7 * math.pi * l_m / (2 * math.pi)) * (
            math.log(2 * l_m / (w_m + t_m)) + 0.5 + (w_m + t_m) / (3 * l_m)
        )
        expected = 50e-12 + L_g
        assert float(L) == pytest.approx(expected, rel=1e-5)

    @staticmethod
    def test_known_value() -> None:
        """Spot-check: 10 turns × 200 µm / 2 µm wide = 1000 □; L = 1 pH/□ kinetic + L_g."""
        L = meander_inductor_inductance_analytical(
            n_turns=10,
            turn_length=200.0,
            wire_width=2.0,
            wire_gap=2.0,
            sheet_inductance=1e-12,
            thickness=0.2,
        )
        # Kinetic: total_length = 10*200 + 9*2 = 2018 um; n_squares = 2018/2 = 1009
        # L_k = 1009 pH
        # Obtained value will also include L_g (self + mutual)
        assert float(L) > 1009e-12
        assert float(L) < 2000e-12  # sanity upper bound

    @staticmethod
    @pytest.mark.parametrize("n_turns", [155, 501, 750])
    def test_large_turn_counts_match_full_sum(n_turns: int) -> None:
        """Large turn counts match an explicit full mutual-inductance sum.

        155 is the default superinductor turn count of the ``fluxonium`` cell;
        501 and 750 lie above the legacy hardcoded 500-offset limit, so these
        cases fail if the mutual-inductance sum is truncated at
        :math:`k = 500`.
        """
        L = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=200.0,
            wire_width=2.0,
            wire_gap=2.0,
            sheet_inductance=1e-12,
            thickness=0.2,
        )
        expected = _meander_reference_inductance(n_turns)
        # For these ~1e-8 H values the 1e-12 absolute floor dominates
        # rel=1e-9; float64 accumulation error over the alternating terms
        # (sax enables x64 on import) stays well below the floor.
        assert float(L) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @staticmethod
    def test_turn_counts_beyond_legacy_500_limit() -> None:
        """n_turns > 500 must include every mutual term, not truncate at k=500.

        The legacy implementation hardcoded ``offsets = jnp.arange(1, 501)``
        with a mask, silently dropping mutual-inductance terms with
        :math:`k > 500`. The full sum must be computed instead.
        """
        L_600 = float(
            meander_inductor_inductance_analytical(
                n_turns=600,
                turn_length=200.0,
                wire_width=2.0,
                wire_gap=2.0,
                sheet_inductance=1e-12,
                thickness=0.2,
            )
        )
        expected_full = _meander_reference_inductance(600)
        # As above, the 1e-12 absolute floor dominates rel=1e-9 for ~1e-8 H
        # values; stated explicitly so the tolerance matches what is applied.
        assert pytest.approx(expected_full, rel=1e-9, abs=1e-12) == L_600
        # Truncating at the legacy limit would give a measurably different
        # (wrong) value — confirm the high-order terms are actually included.
        expected_truncated = _meander_reference_inductance(600, max_offset=500)
        assert pytest.approx(expected_truncated, rel=1e-4, abs=1e-12) != L_600

    @staticmethod
    def test_composes_with_jax_transformations() -> None:
        """A traced n_turns (vmap) must match the concrete-int evaluation."""
        ns = jnp.arange(2, 8)
        batched = jax.vmap(
            lambda n: meander_inductor_inductance_analytical(
                n_turns=n,
                turn_length=200.0,
                wire_width=2.0,
                wire_gap=2.0,
                sheet_inductance=1e-12,
                thickness=0.2,
            )
        )(ns)
        direct = jnp.asarray([
            float(
                meander_inductor_inductance_analytical(
                    n_turns=int(n),
                    turn_length=200.0,
                    wire_width=2.0,
                    wire_gap=2.0,
                    sheet_inductance=1e-12,
                    thickness=0.2,
                )
            )
            for n in range(2, 8)
        ])
        assert jnp.allclose(batched, direct, rtol=1e-9)


# ---------------------------------------------------------------------------
# meander_inductor (SAX model)
# ---------------------------------------------------------------------------
class TestMeanderInductorSAX:
    """Tests for the meander_inductor SAX model."""

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=50),
        turn_length=st.floats(
            min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        sheet_inductance=sheet_inductance_st,
        f=st.floats(
            min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_returns_valid_sdict(
        n_turns: int,
        turn_length: float,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """The model must return an SDict with the four expected S-parameter keys."""
        sdict = meander_inductor(
            f=f,
            n_turns=n_turns,
            turn_length=turn_length,
            cross_section="cpw",
            sheet_inductance=sheet_inductance,
        )
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(sdict.keys()) == expected_keys

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=20),
        turn_length=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        sheet_inductance=sheet_inductance_st,
        f=frequency_st,
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_reciprocal(
        n_turns: int,
        turn_length: float,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """An inductor is a reciprocal two-port: S12 == S21."""
        sdict = meander_inductor(
            f=f,
            n_turns=n_turns,
            turn_length=turn_length,
            cross_section="cpw",
            sheet_inductance=sheet_inductance,
        )
        s12 = jnp.abs(sdict["o1", "o2"])
        s21 = jnp.abs(sdict["o2", "o1"])
        assert float(s12) == pytest.approx(float(s21), rel=1e-5)

    @staticmethod
    @given(
        n_turns=st.integers(min_value=1, max_value=20),
        turn_length=st.floats(
            min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        sheet_inductance=sheet_inductance_st,
        f=frequency_st,
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_s_parameters_are_finite(
        n_turns: int,
        turn_length: float,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """All S-parameter values must be finite."""
        sdict = meander_inductor(
            f=f,
            n_turns=n_turns,
            turn_length=turn_length,
            cross_section="cpw",
            sheet_inductance=sheet_inductance,
        )
        for v in sdict.values():
            assert jnp.all(jnp.isfinite(v))

    @staticmethod
    def test_high_frequency_approaches_open() -> None:
        """At very high frequency an inductor looks like an open circuit (|S21| → 0)."""
        sdict = meander_inductor(
            f=1e14,  # 100 THz — well above any practical resonance
            n_turns=20,
            turn_length=200.0,
            cross_section="cpw",
            sheet_inductance=1e-12,
        )
        assert float(jnp.abs(sdict["o1", "o2"])) < 0.1

    @staticmethod
    def test_array_frequency_input() -> None:
        """Model must accept an array of frequencies and return arrays."""
        freqs = jnp.linspace(1e9, 10e9, 100)
        sdict = meander_inductor(f=freqs)
        for v in sdict.values():
            assert v.shape == (100,)


# ---------------------------------------------------------------------------
# lumped_element_resonator (SAX model)
# ---------------------------------------------------------------------------
class TestLumpedElementResonatorSAX:
    """Tests for the lumped_element_resonator SAX model."""

    @staticmethod
    @given(
        fingers=st.integers(min_value=2, max_value=40),
        finger_length=st.floats(
            min_value=20.0, max_value=200.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=5.0, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        n_turns=st.integers(min_value=1, max_value=50),
        sheet_inductance=sheet_inductance_st,
        f=frequency_st,
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_returns_valid_sdict(
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """Model must return an SDict with the four expected S-parameter keys."""
        # Ensure meander_turn_length is large enough for the strip approximation
        # meander_turn_length = (2*FT + FL + FG) - 4*WW
        # Using WW=2.0 from meander_inductor_cross_section
        cap_width = 2 * finger_thickness + finger_length + finger_gap
        meander_turn_length = cap_width - 8.0  # 4 * 2.0
        assume(meander_turn_length > 5.0)

        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="meander_inductor_cross_section",
        )
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(sdict.keys()) == expected_keys

    @staticmethod
    @given(
        fingers=st.integers(min_value=2, max_value=30),
        finger_length=st.floats(
            min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=5.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n_turns=st.integers(min_value=1, max_value=20),
        sheet_inductance=sheet_inductance_st,
        f=frequency_st,
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_reciprocal(
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """Resonator must be reciprocal: S12 == S21."""
        cap_width = 2 * finger_thickness + finger_length + finger_gap
        meander_turn_length = cap_width - 8.0
        assume(meander_turn_length > 5.0)

        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="meander_inductor_cross_section",
        )
        s12 = jnp.abs(sdict["o1", "o2"])
        s21 = jnp.abs(sdict["o2", "o1"])
        assert float(s12) == pytest.approx(float(s21), rel=1e-5)

    @staticmethod
    @given(
        fingers=st.integers(min_value=2, max_value=30),
        finger_length=st.floats(
            min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=5.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n_turns=st.integers(min_value=1, max_value=20),
        sheet_inductance=sheet_inductance_st,
        f=frequency_st,
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_s_parameters_are_finite(
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """All S-parameter values must be finite."""
        cap_width = 2 * finger_thickness + finger_length + finger_gap
        meander_turn_length = cap_width - 8.0
        assume(meander_turn_length > 5.0)

        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="meander_inductor_cross_section",
        )
        for v in sdict.values():
            assert jnp.all(jnp.isfinite(v))

    @staticmethod
    def test_resonance_frequency_formula() -> None:
        """Transmission minimum in S21 must occur near f_r = 1/(2π√LC).

        Parameters are chosen so that f_r falls in the low-GHz range by using
        a large sheet inductance representative of a high-kinetic-inductance film.
        """
        fingers = 20
        finger_length = 20.0
        finger_gap = 2.0
        finger_thickness = 5.0
        n_turns = 5
        sheet_inductance = 1e-9  # 1 nH/□ — large kinetic inductance
        cross_section = "meander_inductor_cross_section"

        ep_r = cpw_ep_r_from_cross_section(cross_section)
        C = float(
            interdigital_capacitor_capacitance_analytical(
                fingers=fingers,
                finger_length=finger_length,
                finger_gap=finger_gap,
                thickness=finger_thickness,
                ep_r=ep_r,
            )
        )
        wire_width, wire_gap_half = get_cpw_dimensions(cross_section)
        wire_gap = 2 * wire_gap_half

        L = float(
            meander_inductor_inductance_analytical(
                n_turns=n_turns,
                turn_length=(2 * finger_thickness + finger_length + finger_gap)
                - 4 * wire_width,
                wire_width=wire_width,
                wire_gap=wire_gap,
                sheet_inductance=sheet_inductance,
            )
        )
        f_r = 1.0 / (2 * math.pi * math.sqrt(L * C))

        # Sweep from 0.5·f_r to 1.5·f_r so the minimum is guaranteed to be inside
        freqs = jnp.linspace(f_r * 0.5, f_r * 1.5, 1000)
        sdict = lumped_element_resonator(
            f=freqs,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section=cross_section,
        )
        s21 = jnp.abs(sdict["o1", "o2"])
        f_min = float(freqs[jnp.argmin(s21)])
        assert abs(f_min - f_r) / f_r < 0.05  # within 5 % of the analytical f_r

    @staticmethod
    def test_array_frequency_input() -> None:
        """Model must accept a frequency array and return arrays of the same length."""
        freqs = jnp.linspace(1e9, 10e9, 200)
        sdict = lumped_element_resonator(f=freqs)
        for v in sdict.values():
            assert v.shape == (200,)
