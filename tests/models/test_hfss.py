import pytest
from gdsfactory.component import Component

from qpdk import LAYER_STACK
from qpdk.cells.resonator import resonator
from qpdk.models.hfss import (
    _get_layer_number_from_level,
    layer_stack_to_gds_mapping,
    prepare_component_for_hfss,
)


def test_layer_stack_to_gds_mapping():
    """Test generating GDS mapping from a LayerStack."""
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)
    
    # Check that it returns a dictionary
    assert isinstance(mapping, dict)
    
    # Check a known layer from qpdk
    # For example, layer 1 should be in the mapping
    # The structure is {layer_number: (elevation, thickness)}
    assert 1 in mapping
    assert isinstance(mapping[1], tuple)
    assert len(mapping[1]) == 2
    
    elevation, thickness = mapping[1]
    assert isinstance(elevation, float)
    assert isinstance(thickness, float)


def test_prepare_component_for_hfss():
    """Test component preparation for HFSS."""
    comp = resonator()
    prepared = prepare_component_for_hfss(comp, apply_additive=True)
    
    assert isinstance(prepared, Component)
    # The prepared component should be different if additive metals are applied
    # or at least it should return a valid component
    assert prepared.name is not None


def test_get_layer_number_from_level():
    """Test layer number extraction from various layer definitions."""
    # Test with a regular LayerLevel that has a direct tuple
    class MockLevelTuple:
        layer = (1, 0)
    
    assert _get_layer_number_from_level(MockLevelTuple()) == 1

    # Test with a derived layer structure
    class MockLogicalLayerInner:
        layer = (2, 0)
        
    class MockLogicalLayer:
        layer = MockLogicalLayerInner()
        
    class MockDerivedLevel:
        layer = None
        derived_layer = MockLogicalLayer()
        
    assert _get_layer_number_from_level(MockDerivedLevel()) == 2
