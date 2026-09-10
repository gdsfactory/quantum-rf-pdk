"""Tests for cell validation edge cases — inductor, resonator, waveguides."""

import pytest

from qpdk.cells.inductor import lumped_element_resonator, meander_inductor
from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular


class TestInductorValidation:
    """Tests for meander inductor and lumped-element resonator validation."""

    @staticmethod
    def test_resonator_cell_even_n_turns_raises() -> None:
        """Even n_turns must raise in the lumped_element_resonator cell.

        The meander path must span from the left bus bar to the right bus
        bar, which requires an odd number of runs.
        """
        with pytest.raises(ValueError, match="n_turns must be odd"):
            lumped_element_resonator(n_turns=4)

    @staticmethod
    def test_resonator_cell_odd_n_turns_succeeds() -> None:
        """Odd n_turns must be accepted by the lumped_element_resonator cell."""
        c = lumped_element_resonator(n_turns=5)
        assert c is not None

    @staticmethod
    def test_meander_cell_even_n_turns_port_on_same_side() -> None:
        """The bare meander cell accepts even n_turns: o2 lands beside o1.

        The model (qpdk.models.inductor.meander_inductor) therefore accepts
        even n_turns too; the odd constraint is layout-only and specific to
        the resonator cell.
        """
        turn_length = 200.0
        c = meander_inductor(n_turns=4, turn_length=turn_length, cross_section="cpw")
        assert c.ports["o1"].center[0] == pytest.approx(-turn_length / 2)
        assert c.ports["o2"].center[0] == pytest.approx(-turn_length / 2)

    @staticmethod
    def test_meander_cell_odd_n_turns_port_on_opposite_side() -> None:
        """Odd n_turns places o2 on the side opposite to o1."""
        turn_length = 200.0
        c = meander_inductor(n_turns=5, turn_length=turn_length, cross_section="cpw")
        assert c.ports["o1"].center[0] == pytest.approx(-turn_length / 2)
        assert c.ports["o2"].center[0] == pytest.approx(turn_length / 2)

    @staticmethod
    def test_meander_cell_less_than_one_turn_raises() -> None:
        """n_turns < 1 must raise in the meander cell."""
        with pytest.raises(ValueError, match="at least 1 turn"):
            meander_inductor(n_turns=0, cross_section="cpw")

    @staticmethod
    def test_meander_cell_nonpositive_turn_length_raises() -> None:
        """turn_length <= 0 must raise in the meander cell."""
        with pytest.raises(ValueError, match="turn_length must be positive"):
            meander_inductor(turn_length=0.0, cross_section="cpw")


class TestResonatorValidation:
    """Tests for resonator validation branches."""

    @staticmethod
    def test_negative_straights_raises() -> None:
        """Test that start_with_bend + end_with_bend + 0 meanders raises."""
        with pytest.raises(ValueError, match="fewer than 0 straight sections"):
            resonator(
                length=4000,
                meanders=0,
                start_with_bend=True,
                end_with_bend=True,
            )

    @staticmethod
    def test_too_short_for_meanders_raises() -> None:
        """Test that a very short length with many meanders raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            resonator(length=10, meanders=6)

    @staticmethod
    def test_zero_meanders_succeeds() -> None:
        """Test that 0 meanders creates a straight resonator."""
        c = resonator(length=1000, meanders=0)
        assert c is not None

    @staticmethod
    def test_start_with_bend() -> None:
        """Test resonator starting with a bend."""
        c = resonator(length=4000, meanders=4, start_with_bend=True)
        assert c is not None

    @staticmethod
    def test_end_with_bend() -> None:
        """Test resonator ending with a bend."""
        c = resonator(length=4000, meanders=4, end_with_bend=True)
        assert c is not None

    @staticmethod
    def test_both_bends() -> None:
        """Test resonator with both start and end bends."""
        c = resonator(length=4000, meanders=4, start_with_bend=True, end_with_bend=True)
        assert c is not None

    @staticmethod
    def test_open_end() -> None:
        """Test resonator with open end (half-wave)."""
        c = resonator(length=4000, meanders=4, open_end=True)
        assert c is not None

    @staticmethod
    def test_closed_start() -> None:
        """Test resonator with closed start."""
        c = resonator(length=4000, meanders=4, open_start=False)
        assert c is not None


class TestBendCircularEdgeCases:
    """Tests for bend_circular radius correction."""

    @staticmethod
    def test_very_small_radius_gets_corrected() -> None:
        """Test that a radius smaller than min is corrected."""
        # Use a very small radius — the function should correct it
        c = bend_circular(radius=0.1)
        assert c is not None

    @staticmethod
    def test_normal_radius_succeeds() -> None:
        """Test that a normal radius works without issues."""
        c = bend_circular(radius=100.0)
        assert c is not None

    @staticmethod
    def test_various_angles() -> None:
        """Test bend with different angles."""
        for angle in [45.0, 90.0, 180.0]:
            c = bend_circular(angle=angle, radius=100.0)
            assert c is not None
