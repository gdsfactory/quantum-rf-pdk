"""Tests for inductor SAX models (qpdk/models/inductor.py)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
import sax
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from qpdk import PDK
from qpdk.cells.inductor import meander_inductor as meander_inductor_cell
from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.cpw import cpw_ep_r_from_cross_section, get_cpw_dimensions
from qpdk.models.generic import inductor
from qpdk.models.inductor import (
    lumped_element_resonator,
    meander_inductor,
    meander_inductor_inductance_analytical,
)
from qpdk.tech import get_etch_section

MAX_EXAMPLES = 50

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


# ---------------------------------------------------------------------------
# meander_inductor (SAX model)
# ---------------------------------------------------------------------------
# Geometry shared by the wire-gap tests below.
MEANDER_N_TURNS = 5
MEANDER_TURN_LENGTH = 200.0
MEANDER_SHEET_INDUCTANCE = 1e-12

# Arbitrary override used to check that an explicit gap wins over inference.
MEANDER_EXPLICIT_GAP = 1.0


def _meander_sdict(
    f: sax.FloatArrayLike, cross_section: str, wire_gap: float | None
) -> sax.SDict:
    """Model call at the shared test geometry, varying only the gap."""
    return meander_inductor(
        f=f,
        n_turns=MEANDER_N_TURNS,
        turn_length=MEANDER_TURN_LENGTH,
        cross_section=cross_section,
        wire_gap=wire_gap,
        sheet_inductance=MEANDER_SHEET_INDUCTANCE,
    )


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

    @staticmethod
    @pytest.mark.parametrize("cross_section", sorted(PDK.cross_sections))
    def test_wire_gap_matches_layout(cross_section: str) -> None:
        """The model must use the gap the cell draws, for every PDK cross-section."""
        layout = meander_inductor_cell(
            n_turns=MEANDER_N_TURNS,
            turn_length=MEANDER_TURN_LENGTH,
            cross_section=cross_section,
        )
        wire_width = layout.ports["o1"].width
        # The ports sit on the first and last run, so their vertical spacing
        # is (n_turns - 1) pitches; subtract the wire width to get the gap.
        drawn_gap = (
            abs(layout.ports["o2"].center[1] - layout.ports["o1"].center[1])
            / (MEANDER_N_TURNS - 1)
            - wire_width
        )
        # Pin the inference rule itself, derived from the cross-section:
        # twice the etch width, or the wire width without an etch section.
        try:
            expected_inferred_gap = 2 * get_etch_section(cross_section).width
        except ValueError:
            expected_inferred_gap = wire_width
        assert drawn_gap == pytest.approx(expected_inferred_gap)

        explicit_layout = meander_inductor_cell(
            n_turns=MEANDER_N_TURNS,
            turn_length=MEANDER_TURN_LENGTH,
            cross_section=cross_section,
            wire_gap=MEANDER_EXPLICIT_GAP,
        )
        explicit_drawn_gap = (
            abs(
                explicit_layout.ports["o2"].center[1]
                - explicit_layout.ports["o1"].center[1]
            )
            / (MEANDER_N_TURNS - 1)
            - wire_width
        )
        assert explicit_drawn_gap == pytest.approx(MEANDER_EXPLICIT_GAP)

        freqs = jnp.linspace(1e9, 10e9, 50)
        for wire_gap, expected_gap in (
            (None, drawn_gap),
            (MEANDER_EXPLICIT_GAP, MEANDER_EXPLICIT_GAP),
        ):
            expected_inductance = meander_inductor_inductance_analytical(
                n_turns=MEANDER_N_TURNS,
                turn_length=MEANDER_TURN_LENGTH,
                wire_width=wire_width,
                wire_gap=expected_gap,
                sheet_inductance=MEANDER_SHEET_INDUCTANCE,
            )
            expected = inductor(f=freqs, inductance=expected_inductance)
            sdict = _meander_sdict(freqs, cross_section, wire_gap)
            for key, value in expected.items():
                assert jnp.allclose(sdict[key], value, rtol=1e-6), (
                    f"wire_gap={wire_gap}"
                )

    @staticmethod
    @pytest.mark.parametrize("bad_gap", [0.0, -3.0, float("nan")])
    def test_rejects_non_positive_wire_gap(bad_gap: float) -> None:
        """A non-positive explicit gap must raise, not give NaN or wrong inductance."""
        with pytest.raises(ValueError, match="wire_gap must be positive"):
            _meander_sdict(jnp.array([5e9]), "cpw", bad_gap)
        with pytest.raises(ValueError, match="wire_gap must be positive"):
            meander_inductor_cell(wire_gap=bad_gap)

    @staticmethod
    @pytest.mark.parametrize("cross_section", ["cpw", "microstrip"])
    def test_jit_compatible(cross_section: str) -> None:
        """wire_gap must trace as a jit argument and reach the formula unchanged."""
        freqs = jnp.linspace(1e9, 10e9, 20)
        wire_gap = 3.0
        # wire_gap is a traced argument here: a Python-level branch on it
        # inside the model would raise a tracer conversion error.
        jitted = jax.jit(lambda f, g: _meander_sdict(f, cross_section, g))
        sdict = jitted(freqs, wire_gap)

        # Reference computed eagerly from the analytical formula, not the model
        xs = PDK.get_cross_section(cross_section)
        expected_inductance = meander_inductor_inductance_analytical(
            n_turns=MEANDER_N_TURNS,
            turn_length=MEANDER_TURN_LENGTH,
            wire_width=xs.width,
            wire_gap=wire_gap,
            sheet_inductance=MEANDER_SHEET_INDUCTANCE,
        )
        expected = inductor(f=freqs, inductance=expected_inductance)
        for key, value in expected.items():
            assert jnp.allclose(sdict[key], value, rtol=1e-6)


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
