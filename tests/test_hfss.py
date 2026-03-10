"""Tests for HFSS simulation integration."""

import os
import time
from pathlib import Path

import pytest
from gdsfactory.component import Component
from numpy.testing import assert_allclose

from qpdk import LAYER_STACK
from qpdk.cells.resonator import resonator
from qpdk.models.hfss import (
    _get_layer_number_from_level,
    layer_stack_to_gds_mapping,
    lumped_port_rectangle_from_cpw,
    prepare_component_for_hfss,
)

# Ensure Ansys path is set so PyAEDT can find it
ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
if "ANSYSEM_ROOT252" not in os.environ and Path(ansys_default_path).exists():
    os.environ["ANSYSEM_ROOT252"] = ansys_default_path


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
    prepared = prepare_component_for_hfss(comp)

    assert isinstance(prepared, Component)
    # The prepared component should be different if additive metals are applied
    # or at least it should return a valid component
    assert prepared.name is not None


def test_prepare_component_for_hfss_margin():
    """Test component preparation for HFSS with margin."""
    comp = resonator(length=3000)

    # Original bbox
    bbox_orig = comp.bbox()
    prepared = prepare_component_for_hfss(comp, margin_draw=200)

    assert isinstance(prepared, Component)
    assert prepared.name is not None

    # Prepared bbox might be larger
    bbox_new = prepared.bbox()
    assert bbox_new.left <= bbox_orig.left
    assert bbox_new.right >= bbox_orig.right
    assert bbox_new.bottom <= bbox_orig.bottom
    assert bbox_new.top >= bbox_orig.top


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


@pytest.mark.hfss
def test_hfss_import_and_draw():
    """Test creating an HFSS project and drawing a component."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from ansys.aedt.core import Hfss, settings

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import import_component_to_hfss

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_draw_{int(time.time())}"
    hfss = Hfss(
        project=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        # Use direct draw which we know is stable on Linux
        success = import_component_to_hfss(hfss, comp)
        assert success, "Failed to draw component in HFSS"
    finally:
        hfss.release_desktop()


@pytest.mark.hfss
def test_hfss_eigenmode_setup():
    """Test setting up an eigenmode simulation."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from ansys.aedt.core import Hfss, settings

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import (
        add_air_region_to_hfss,
        add_substrate_to_hfss,
        import_component_to_hfss,
    )

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_eigenmode_{int(time.time())}"
    hfss = Hfss(
        project=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        import_component_to_hfss(hfss, comp)
        add_substrate_to_hfss(hfss, comp)
        add_air_region_to_hfss(hfss, comp)

        setup = hfss.create_setup(name="EigenmodeSetup")
        setup.props["MinimumFrequency"] = "1.0GHz"
        setup.props["NumModes"] = 1
        setup.props["MaximumPasses"] = 2
        setup.props["MinimumPasses"] = 2
        setup.props["PercentRefinement"] = 30
        setup.props["MaxDeltaFreq"] = 2.0
        setup.props["ConvergeOnRealFreq"] = True
        setup.update()

        assert setup is not None, "Failed to setup eigenmode simulation"
    finally:
        hfss.release_desktop()


@pytest.fixture
def mock_port_dimensions():
    """Provides a standardized set of dimensions for testing."""
    return {"center": [10.0, 20.0, 0.0], "cpw_gap": 6.0, "cpw_width": 2.0}


@pytest.mark.parametrize(
    ("orientation", "expected_origin", "expected_sizes", "expected_int_line"),
    [
        (0, [10.0, 19.0, 0.0], [6.0, 2.0], [[16.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (90, [9.0, 20.0, 0.0], [2.0, 6.0], [[10.0, 26.0, 0.0], [10.0, 20.0, 0.0]]),
        (180, [4.0, 19.0, 0.0], [6.0, 2.0], [[4.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (270, [9.0, 14.0, 0.0], [2.0, 6.0], [[10.0, 14.0, 0.0], [10.0, 20.0, 0.0]]),
    ],
)
def test_lumped_port_rectangle_from_cpw_valid_angles(
    mock_port_dimensions,
    orientation,
    expected_origin,
    expected_sizes,
    expected_int_line,
):
    """Verifies that the vectorized geometry perfectly matches the expected dictionary values."""
    result = lumped_port_rectangle_from_cpw(
        center=mock_port_dimensions["center"],
        orientation=orientation,
        cpw_gap=mock_port_dimensions["cpw_gap"],
        cpw_width=mock_port_dimensions["cpw_width"],
    )

    assert_allclose(result["origin"], expected_origin)
    assert_allclose(result["sizes"], expected_sizes)
    assert_allclose(result["integration_line"], expected_int_line)


def test_lumped_port_rectangle_from_cpw_invalid_angle(mock_port_dimensions):
    """Ensures the function throws a ValueError if passed an unaligned angle."""
    with pytest.raises(ValueError, match="Unsupported port orientation: 45°"):
        lumped_port_rectangle_from_cpw(
            center=mock_port_dimensions["center"],
            orientation=45,
            cpw_gap=mock_port_dimensions["cpw_gap"],
            cpw_width=mock_port_dimensions["cpw_width"],
        )
