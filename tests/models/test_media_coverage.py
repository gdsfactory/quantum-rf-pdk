"""Additional tests for qpdk.models.media module - covering missing lines."""

import warnings

import pytest

from qpdk.models.media import (
    cpw_media_skrf,
    cpw_parameters,
    cross_section_to_media,
    get_cpw_dimensions,
)
from qpdk.tech import coplanar_waveguide


class TestCpwMediaSkrf:
    """Tests for cpw_media_skrf (deprecated) function (lines 50-62)."""

    @staticmethod
    def test_creates_media_callable() -> None:
        """Test that cpw_media_skrf returns a callable media object."""
        # Clear the cache to ensure the function body is executed
        cpw_media_skrf.cache_clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            media_fn = cpw_media_skrf(width=10.0, gap=6.0)
            assert callable(media_fn)
            assert any(issubclass(wi.category, DeprecationWarning) for wi in w)


class TestCrossSectionToMedia:
    """Tests for cross_section_to_media (deprecated) function (lines 117-118)."""

    @staticmethod
    def test_with_string_cross_section() -> None:
        """Test cross_section_to_media with string cross-section."""
        # Clear caches before test
        cpw_media_skrf.cache_clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            media_fn = cross_section_to_media("cpw")
            assert callable(media_fn)
            # Should produce deprecation warnings
            dep_warnings = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1


class TestGetCpwDimensions:
    """Tests for get_cpw_dimensions edge cases."""

    @staticmethod
    def test_with_callable_cross_section() -> None:
        """Test get_cpw_dimensions with callable cross-section (line 78)."""
        xs_fn = coplanar_waveguide
        width, gap = get_cpw_dimensions(xs_fn)
        assert width > 0
        assert gap > 0

    @staticmethod
    def test_with_cross_section_object() -> None:
        """Test get_cpw_dimensions with CrossSection object (line 76)."""
        xs = coplanar_waveguide()
        width, gap = get_cpw_dimensions(xs)
        assert width > 0
        assert gap > 0

    @staticmethod
    def test_missing_etch_offset_raises() -> None:
        """Test that missing 'etch_offset' section raises ValueError (lines 89-94)."""
        import gdsfactory as gf

        # Create a cross-section without any 'etch_offset' sections
        xs = gf.cross_section.cross_section(width=10.0)
        with pytest.raises(ValueError, match="etch_offset"):
            get_cpw_dimensions(xs)


class TestCpwParameters:
    """Tests for cpw_parameters function (lines 170-176)."""

    @staticmethod
    def test_returns_valid_values() -> None:
        """Test that cpw_parameters returns valid ep_eff and z0."""
        cpw_parameters.cache_clear()
        ep_eff, z0 = cpw_parameters(width=10.0, gap=6.0)
        assert ep_eff > 1.0, "Effective permittivity should be > 1"
        assert 20 < z0 < 200, f"Z0 = {z0} Ohm is out of expected range"

    @staticmethod
    def test_different_widths_affect_z0() -> None:
        """Test that different widths produce different impedances."""
        cpw_parameters.cache_clear()
        _, z0_narrow = cpw_parameters(width=5.0, gap=6.0)
        _, z0_wide = cpw_parameters(width=20.0, gap=6.0)
        assert z0_narrow > z0_wide, "Narrower CPW should have higher impedance"
