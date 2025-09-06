"""Tests for the magnetic vortices fill helper function."""

import pytest

import gdsfactory as gf

from qpdk import PDK
from qpdk.cells.helpers import fill_magnetic_vortices
from qpdk.cells.resonator import resonator_quarter_wave
from qpdk.samples.sample_filled_resonator import sample_filled_quarter_wave_resonator
from qpdk.tech import LAYER


def test_fill_magnetic_vortices():
    """Test the fill_magnetic_vortices helper function."""
    PDK.activate()
    
    # Create a simple resonator
    resonator = resonator_quarter_wave(length=4000.0)
    original_poly_count = sum(len(polys) for polys in resonator.get_polygons().values())
    
    # Apply the fill function
    filled = fill_magnetic_vortices(resonator)
    filled_poly_count = sum(len(polys) for polys in filled.get_polygons().values())
    
    # Check that the fill adds polygons
    assert filled_poly_count > original_poly_count, "Fill should add polygons"
    
    # Check that ports are preserved
    assert len(filled.ports) == len(resonator.ports), "Ports should be preserved"
    for port in resonator.ports:
        assert port.name in [p.name for p in filled.ports], f"Port {port.name} should be preserved"


def test_fill_magnetic_vortices_parameters():
    """Test the fill_magnetic_vortices with different parameters."""
    PDK.activate()
    
    # Create a simple component
    component = gf.components.rectangle(size=(100, 100), layer=LAYER.M1_DRAW)
    
    # Test with different parameters
    filled1 = fill_magnetic_vortices(
        component,
        rectangle_size=(10.0, 10.0),
        gap=10.0,
        keepout_margin=50.0,
    )
    
    filled2 = fill_magnetic_vortices(
        component,
        rectangle_size=(20.0, 20.0),
        gap=20.0,
        keepout_margin=100.0,
    )
    
    # Both should work without errors
    assert filled1 is not None
    assert filled2 is not None
    
    # The fills should have different polygon counts due to different parameters
    count1 = sum(len(polys) for polys in filled1.get_polygons().values())
    count2 = sum(len(polys) for polys in filled2.get_polygons().values())
    
    # With larger rectangles and gaps, we should have fewer fill polygons
    assert count1 >= count2, "Smaller rectangles and gaps should result in more fill polygons"


def test_sample_filled_quarter_wave_resonator():
    """Test the sample filled quarter wave resonator component."""
    PDK.activate()
    
    # Create the sample component
    sample = sample_filled_quarter_wave_resonator()
    
    # Check that it has polygons
    poly_count = sum(len(polys) for polys in sample.get_polygons().values())
    assert poly_count > 0, "Sample should have polygons"
    
    # Check that it has ports from the original resonator
    assert len(sample.ports) > 0, "Sample should have ports"
    
    # Check that it has the expected port names
    port_names = [p.name for p in sample.ports]
    assert "o2" in port_names, "Should have o2 port from quarter wave resonator"


if __name__ == "__main__":
    test_fill_magnetic_vortices()
    test_fill_magnetic_vortices_parameters()
    test_sample_filled_quarter_wave_resonator()
    print("All tests passed!")