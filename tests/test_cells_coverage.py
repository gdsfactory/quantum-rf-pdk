"""Tests for qpdk.cells module - covering missing lines."""

import pytest
from klayout.db import DCplxTrans

from qpdk.cells.capacitor import interdigital_capacitor, plate_capacitor, plate_capacitor_single
from qpdk.cells.snspd import snspd
from qpdk.cells.transmon import double_pad_transmon, flipmon_with_bbox, xmon_transmon


class TestInterdigitalCapacitorValidation:
    """Tests for interdigital_capacitor validation (line 73)."""

    @staticmethod
    def test_zero_fingers_raises() -> None:
        """Test that fingers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1 finger"):
            interdigital_capacitor(fingers=0)


class TestPlateCapacitorValidation:
    """Tests for plate_capacitor validation (lines 242, 244)."""

    @staticmethod
    def test_zero_width_raises() -> None:
        """Test that width <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            plate_capacitor(width=0)

    @staticmethod
    def test_negative_width_raises() -> None:
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            plate_capacitor(width=-5)

    @staticmethod
    def test_zero_length_raises() -> None:
        """Test that length <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            plate_capacitor(length=0)


class TestPlateCapacitorSingleValidation:
    """Tests for plate_capacitor_single validation (lines 313, 315)."""

    @staticmethod
    def test_zero_width_raises() -> None:
        """Test that width <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            plate_capacitor_single(width=0)

    @staticmethod
    def test_zero_length_raises() -> None:
        """Test that length <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            plate_capacitor_single(length=0)


class TestSNSPD:
    """Tests for SNSPD cell (lines 42-44, 48-51)."""

    @staticmethod
    def test_snspd_from_num_squares() -> None:
        """Test SNSPD creation using num_squares parameter (lines 42-44)."""
        c = snspd(num_squares=1000)
        assert c is not None
        assert "num_squares" in c.info

    @staticmethod
    def test_snspd_default_parameters() -> None:
        """Test SNSPD creation with default parameters."""
        c = snspd()
        assert c is not None
        assert c.info["xsize"] > 0
        assert c.info["ysize"] > 0


class TestTransmonJunctionDisplacement:
    """Tests for transmon components with junction_displacement (lines 77, 260, 476)."""

    @staticmethod
    def test_double_pad_transmon_with_displacement() -> None:
        """Test double_pad_transmon with junction displacement (line 77)."""
        displacement = DCplxTrans(0, 100)
        c = double_pad_transmon(junction_displacement=displacement)
        assert c is not None
        assert "qubit_type" in c.info

    @staticmethod
    def test_flipmon_with_bbox_with_displacement() -> None:
        """Test flipmon_with_bbox with junction displacement (line 260)."""
        displacement = DCplxTrans(0, 50)
        c = flipmon_with_bbox(junction_displacement=displacement)
        assert c is not None

    @staticmethod
    def test_xmon_transmon_with_displacement() -> None:
        """Test xmon_transmon with junction displacement (line 476)."""
        displacement = DCplxTrans(0, 50)
        c = xmon_transmon(junction_displacement=displacement)
        assert c is not None
        assert c.info["qubit_type"] == "xmon"
