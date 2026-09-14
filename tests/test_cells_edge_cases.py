"""Tests for cell validation edge cases — resonator, waveguides."""

import gdsfactory as gf
import pytest

from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, tee


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


class TestTee:
    """Tests for tee junction consistency with the attached straights."""

    @staticmethod
    @pytest.mark.parametrize(
        "cross_section_name",
        ["cpw", "meander_inductor_cross_section", "superinductor_cross_section"],
    )
    def test_junction_widths_match_cross_section(cross_section_name: str) -> None:
        """Test that junction ports match the cross-section width and layer.

        The junction must not be built with the default CPW width (10 µm) or
        layer (M1_DRAW) when ``tee`` is given a different cross-section.
        """
        cross_section = gf.get_cross_section(cross_section_name)
        c = tee(cross_section=cross_section_name)
        junction = next(inst for inst in c.insts if inst.cell.name.startswith("nxn"))
        for port in junction.ports:
            assert port.width == pytest.approx(cross_section.width)
            assert port.layer == gf.get_layer(cross_section.layer)


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
