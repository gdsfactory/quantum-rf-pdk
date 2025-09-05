"""Unit tests for the denest_layerviews_to_layer_tuples function in qpdk.helper."""

from unittest.mock import Mock

import gdsfactory as gf
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qpdk.helper import denest_layerviews_to_layer_tuples


class TestDenestLayerviewsToLayerTuples:
    """Test suite for denest_layerviews_to_layer_tuples function."""

    def test_empty_layer_views(self):
        """Test with empty layer views."""
        # Mock LayerViews with empty layer_views
        mock_layer_views = Mock()
        mock_layer_views.layer_views = {}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        assert result == {}

    def test_single_flat_layer(self):
        """Test with a single non-nested layer."""
        # Mock LayerView without group_members
        mock_layer_view = Mock(spec=gf.technology.LayerView)
        mock_layer_view.group_members = None
        mock_layer_view.layer = (1, 0)

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {"metal1": mock_layer_view}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        assert result == {"metal1": (1, 0)}

    def test_multiple_flat_layers(self):
        """Test with multiple non-nested layers."""
        # Mock multiple LayerViews without group_members
        mock_layer_view1 = Mock(spec=gf.technology.LayerView)
        mock_layer_view1.group_members = None
        mock_layer_view1.layer = (1, 0)

        mock_layer_view2 = Mock(spec=gf.technology.LayerView)
        mock_layer_view2.group_members = None
        mock_layer_view2.layer = (2, 0)

        mock_layer_view3 = Mock(spec=gf.technology.LayerView)
        mock_layer_view3.group_members = None
        mock_layer_view3.layer = (3, 1)

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {
            "metal1": mock_layer_view1,
            "metal2": mock_layer_view2,
            "via1": mock_layer_view3,
        }

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        expected = {"metal1": (1, 0), "metal2": (2, 0), "via1": (3, 1)}
        assert result == expected

    def test_single_nested_layer(self):
        """Test with a single nested layer group."""
        # Mock nested structure
        mock_nested_layer = Mock(spec=gf.technology.LayerView)
        mock_nested_layer.group_members = None
        mock_nested_layer.layer = (10, 0)

        mock_parent_layer = Mock(spec=gf.technology.LayerViews)
        mock_parent_layer.group_members = {"sublayer": mock_nested_layer}

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {"parent": mock_parent_layer}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        assert result == {"sublayer": (10, 0)}

    def test_deeply_nested_layers(self):
        """Test with deeply nested layer groups."""
        # Create a 3-level nested structure
        mock_deep_layer = Mock(spec=gf.technology.LayerView)
        mock_deep_layer.group_members = None
        mock_deep_layer.layer = (100, 1)

        mock_mid_layer = Mock(spec=gf.technology.LayerViews)
        mock_mid_layer.group_members = {"deep": mock_deep_layer}

        mock_top_layer = Mock(spec=gf.technology.LayerViews)
        mock_top_layer.group_members = {"mid": mock_mid_layer}

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {"top": mock_top_layer}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        assert result == {"deep": (100, 1)}

    def test_mixed_nested_and_flat_layers(self):
        """Test with a mix of nested and flat layers."""
        # Flat layer
        mock_flat_layer = Mock(spec=gf.technology.LayerView)
        mock_flat_layer.group_members = None
        mock_flat_layer.layer = (1, 0)

        # Nested layers
        mock_nested_layer1 = Mock(spec=gf.technology.LayerView)
        mock_nested_layer1.group_members = None
        mock_nested_layer1.layer = (2, 0)

        mock_nested_layer2 = Mock(spec=gf.technology.LayerView)
        mock_nested_layer2.group_members = None
        mock_nested_layer2.layer = (3, 1)

        mock_group_layer = Mock(spec=gf.technology.LayerViews)
        mock_group_layer.group_members = {
            "nested1": mock_nested_layer1,
            "nested2": mock_nested_layer2,
        }

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {
            "flat": mock_flat_layer,
            "group": mock_group_layer,
        }

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        expected = {"flat": (1, 0), "nested1": (2, 0), "nested2": (3, 1)}
        assert result == expected

    def test_empty_group_members(self):
        """Test with a layer that has empty group_members dict."""
        mock_layer_view = Mock(spec=gf.technology.LayerViews)
        mock_layer_view.group_members = {}  # Empty but not None

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {"empty_group": mock_layer_view}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        # Should still process as a group (empty result)
        assert result == {}

    @given(
        layer_name=st.text(min_size=1, max_size=20),
        layer_num=st.integers(min_value=0, max_value=1000),
        datatype=st.integers(min_value=0, max_value=255),
    )
    def test_single_layer_property_based(
        self, layer_name: str, layer_num: int, datatype: int
    ):
        """Property-based test for single layer processing."""
        mock_layer_view = Mock(spec=gf.technology.LayerView)
        mock_layer_view.group_members = None
        mock_layer_view.layer = (layer_num, datatype)

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {layer_name: mock_layer_view}

        result = denest_layerviews_to_layer_tuples(mock_layer_views)

        assert result == {layer_name: (layer_num, datatype)}

    def test_non_layerview_object_in_dict(self):
        """Test with non-LayerView object that doesn't have group_members."""
        # Regular object without group_members attribute
        regular_object = object()

        mock_layer_views = Mock(spec=gf.technology.LayerViews)
        mock_layer_views.layer_views = {"regular": regular_object}

        with pytest.raises(AttributeError):
            denest_layerviews_to_layer_tuples(mock_layer_views)
