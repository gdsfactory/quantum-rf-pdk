"""Tests for qpdk.helper — covering show_components and layerenum_to_tuple."""

from unittest.mock import MagicMock, patch

from qpdk.helper import layerenum_to_tuple, show_components
from qpdk.tech import LAYER


class TestLayerEnumToTuple:
    """Tests for layerenum_to_tuple."""

    @staticmethod
    def test_converts_layer_enum() -> None:
        """Test that a LayerEnum is converted to a (layer, datatype) tuple."""
        result = layerenum_to_tuple(LAYER.M1_DRAW)
        assert isinstance(result, tuple)
        assert len(result) == 2
        layer, datatype = result
        assert isinstance(layer, int)
        assert isinstance(datatype, int)


class TestShowComponents:
    """Tests for show_components."""

    @staticmethod
    @patch("qpdk.helper.Component.show")
    def test_returns_correct_components(mock_show: MagicMock) -> None:  # noqa: ARG004
        """Test that show_components returns the requested components."""
        from qpdk.cells.bump import indium_bump
        from qpdk.cells.tsv import tsv

        result = show_components(indium_bump, tsv)
        assert len(result) == 2

    @staticmethod
    @patch("qpdk.helper.Component.show")
    def test_single_component(mock_show: MagicMock) -> None:  # noqa: ARG004
        """Test show_components with a single component."""
        from qpdk.cells.bump import indium_bump

        result = show_components(indium_bump)
        assert len(result) == 1

    @staticmethod
    @patch("qpdk.helper.Component.show")
    def test_custom_spacing(mock_show: MagicMock) -> None:  # noqa: ARG004
        """Test show_components with custom spacing."""
        from qpdk.cells.bump import indium_bump
        from qpdk.cells.tsv import tsv

        result = show_components(indium_bump, tsv, spacing=500)
        assert len(result) == 2
