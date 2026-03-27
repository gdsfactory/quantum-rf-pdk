"""Tests for inductor SAX models (qpdk/models/inductor.py)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from qpdk.models.inductor import (
    lumped_element_resonator,
    meander_inductor,
    meander_inductor_inductance_analytical,
)

MAX_EXAMPLES = 50

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------
positive_floats = st.floats(
    min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False
)
sheet_inductance_st = st.floats(
    min_value=1e-15, max_value=1e-9, allow_nan=False, allow_infinity=False
)
frequency_st = st.floats(
    min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# meander_inductor_inductance_analytical
# ---------------------------------------------------------------------------
class TestMeanderInductorInductanceAnalytical:
    """Unit tests for the analytical inductance formula."""

    @given(
        n_turns=st.integers(min_value=1, max_value=200),
        turn_length=positive_floats,
        wire_width=positive_floats,
        wire_gap=positive_floats,
        sheet_inductance=sheet_inductance_st,
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_is_positive(
        self,
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
    ) -> None:
        """Inductance must always be strictly positive."""
        L = meander_inductor_inductance_analytical(
            n_turns=n_turns,
            turn_length=turn_length,
            wire_width=wire_width,
            wire_gap=wire_gap,
            sheet_inductance=sheet_inductance,
        )
        assert float(L) > 0

    @given(
        n_turns=st.integers(min_value=1, max_value=100),
        turn_length=positive_floats,
        wire_width=positive_floats,
        wire_gap=positive_floats,
        sheet_inductance=sheet_inductance_st,
        scale=st.floats(
            min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_scales_with_sheet_inductance(
        self,
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
        scale: float,
    ) -> None:
        """Doubling (or scaling) the sheet inductance must scale L by the same factor."""
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
        assert float(L2) == pytest.approx(float(L1) * scale, rel=1e-5)

    @given(
        n_turns=st.integers(min_value=2, max_value=100),
        turn_length=positive_floats,
        wire_width=positive_floats,
        wire_gap=positive_floats,
        sheet_inductance=sheet_inductance_st,
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_increases_with_n_turns(
        self,
        n_turns: int,
        turn_length: float,
        wire_width: float,
        wire_gap: float,
        sheet_inductance: float,
    ) -> None:
        """Adding more turns must increase the total inductance."""
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
        assert float(L_more) > float(L_base)

    @given(
        n_turns=st.integers(min_value=1, max_value=100),
        turn_length=positive_floats,
        wire_gap=positive_floats,
        sheet_inductance=sheet_inductance_st,
        wire_width=positive_floats,
        scale=st.floats(
            min_value=1.01, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductance_increases_with_turn_length(
        self,
        n_turns: int,
        turn_length: float,
        wire_gap: float,
        sheet_inductance: float,
        wire_width: float,
        scale: float,
    ) -> None:
        """Longer turns must give higher inductance."""
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

    def test_single_turn_no_gap_contribution(self) -> None:
        """With n_turns=1 the wire_gap term vanishes (max(0, 1-1)=0)."""
        L = meander_inductor_inductance_analytical(
            n_turns=1,
            turn_length=100.0,
            wire_width=2.0,
            wire_gap=999.0,  # large gap — should not matter
            sheet_inductance=1e-12,
        )
        expected = 1e-12 * (100.0 / 2.0)
        assert float(L) == pytest.approx(expected, rel=1e-5)

    def test_known_value(self) -> None:
        """Spot-check: 10 turns × 200 µm / 2 µm wide = 1000 □; L = 1 pH/□ → 1 nH."""
        L = meander_inductor_inductance_analytical(
            n_turns=10,
            turn_length=200.0,
            wire_width=2.0,
            wire_gap=0.0,  # ignore gap contribution for this check
            sheet_inductance=1e-12,
        )
        # total_length = 10*200 + 9*0 = 2000 µm; n_squares = 2000/2 = 1000
        assert float(L) == pytest.approx(1000 * 1e-12, rel=1e-5)


# ---------------------------------------------------------------------------
# meander_inductor (SAX model)
# ---------------------------------------------------------------------------
class TestMeanderInductorSAX:
    """Tests for the meander_inductor SAX model."""

    @given(
        n_turns=st.integers(min_value=1, max_value=50),
        turn_length=st.floats(
            min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
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
        self,
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
        self,
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
        self,
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

    def test_high_frequency_approaches_open(self) -> None:
        """At very high frequency an inductor looks like an open circuit (|S21| → 0)."""
        sdict = meander_inductor(
            f=1e14,  # 100 THz — well above any practical resonance
            n_turns=20,
            turn_length=200.0,
            cross_section="cpw",
            sheet_inductance=1e-12,
        )
        assert float(jnp.abs(sdict["o1", "o2"])) < 0.1

    def test_array_frequency_input(self) -> None:
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

    @given(
        fingers=st.integers(min_value=2, max_value=40),
        finger_length=st.floats(
            min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False
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
        self,
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """Model must return an SDict with the four expected S-parameter keys."""
        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="cpw",
        )
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(sdict.keys()) == expected_keys

    @given(
        fingers=st.integers(min_value=2, max_value=30),
        finger_length=st.floats(
            min_value=5.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
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
        self,
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """Resonator must be reciprocal: S12 == S21."""
        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="cpw",
        )
        s12 = jnp.abs(sdict["o1", "o2"])
        s21 = jnp.abs(sdict["o2", "o1"])
        assert float(s12) == pytest.approx(float(s21), rel=1e-5)

    @given(
        fingers=st.integers(min_value=2, max_value=30),
        finger_length=st.floats(
            min_value=5.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        finger_gap=st.floats(
            min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        finger_thickness=st.floats(
            min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
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
        self,
        fingers: int,
        finger_length: float,
        finger_gap: float,
        finger_thickness: float,
        n_turns: int,
        sheet_inductance: float,
        f: float,
    ) -> None:
        """All S-parameter values must be finite."""
        sdict = lumped_element_resonator(
            f=f,
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=finger_gap,
            finger_thickness=finger_thickness,
            n_turns=n_turns,
            sheet_inductance=sheet_inductance,
            cross_section="cpw",
        )
        for v in sdict.values():
            assert jnp.all(jnp.isfinite(v))

    def test_resonance_frequency_formula(self) -> None:
        """Transmission minimum in S21 must occur near f_r = 1/(2π√LC).

        Parameters are chosen so that f_r falls in the low-GHz range by using
        a large sheet inductance representative of a high-kinetic-inductance film.
        """
        import math

        from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
        from qpdk.models.cpw import (
            cpw_ep_r_from_cross_section,
            get_cpw_dimensions,
        )

        fingers = 20
        finger_length = 20.0
        finger_gap = 2.0
        finger_thickness = 5.0
        n_turns = 5
        sheet_inductance = 1e-9  # 1 nH/□ — large kinetic inductance
        cross_section = "cpw"

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

    def test_more_fingers_shifts_resonance_lower(self) -> None:
        """Increasing capacitor fingers raises C, so the analytical f_r must decrease."""
        import math

        from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
        from qpdk.models.cpw import (
            cpw_ep_r_from_cross_section,
            get_cpw_dimensions,
        )

        cross_section = "cpw"
        ep_r = float(cpw_ep_r_from_cross_section(cross_section))
        wire_width, wire_gap_half = get_cpw_dimensions(cross_section)
        wire_gap = 2 * wire_gap_half

        L = float(
            meander_inductor_inductance_analytical(
                n_turns=5,
                turn_length=22.0,
                wire_width=wire_width,
                wire_gap=wire_gap,
                sheet_inductance=0.4e-12,
            )
        )

        def f_r(fingers: int) -> float:
            C = float(
                interdigital_capacitor_capacitance_analytical(
                    fingers=fingers,
                    finger_length=20.0,
                    finger_gap=2.0,
                    thickness=5.0,
                    ep_r=ep_r,
                )
            )
            return 1.0 / (2 * math.pi * math.sqrt(L * C))

        assert f_r(fingers=20) < f_r(fingers=5)

    def test_more_turns_shifts_resonance_lower(self) -> None:
        """Increasing inductor turns raises L, so the analytical f_r must decrease."""
        import math

        from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
        from qpdk.models.cpw import (
            cpw_ep_r_from_cross_section,
            get_cpw_dimensions,
        )

        cross_section = "cpw"
        ep_r = float(cpw_ep_r_from_cross_section(cross_section))
        C = float(
            interdigital_capacitor_capacitance_analytical(
                fingers=20,
                finger_length=20.0,
                finger_gap=2.0,
                thickness=5.0,
                ep_r=ep_r,
            )
        )
        wire_width, wire_gap_half = get_cpw_dimensions(cross_section)
        wire_gap = 2 * wire_gap_half

        def f_r(n_turns: int) -> float:
            L = float(
                meander_inductor_inductance_analytical(
                    n_turns=n_turns,
                    turn_length=22.0,
                    wire_width=wire_width,
                    wire_gap=wire_gap,
                    sheet_inductance=0.4e-12,
                )
            )
            return 1.0 / (2 * math.pi * math.sqrt(L * C))

        assert f_r(n_turns=15) < f_r(n_turns=3)

    def test_array_frequency_input(self) -> None:
        """Model must accept a frequency array and return arrays of the same length."""
        freqs = jnp.linspace(1e9, 10e9, 200)
        sdict = lumped_element_resonator(f=freqs)
        for v in sdict.values():
            assert v.shape == (200,)
