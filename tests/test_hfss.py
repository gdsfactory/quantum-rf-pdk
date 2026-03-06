"""Tests for HFSS simulation integration."""

import os
import time
from pathlib import Path

import pytest
from gdsfactory.component import Component

from qpdk import LAYER_STACK
from qpdk.cells.resonator import resonator
from qpdk.models.hfss import (
    _get_layer_number_from_level,
    layer_stack_to_gds_mapping,
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

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import (
        close_hfss,
        create_hfss_project,
        import_component_to_hfss,
    )

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_draw_{int(time.time())}"
    hfss = create_hfss_project(
        project_name=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        # Use direct draw which we know is stable on Linux
        success = import_component_to_hfss(hfss, comp)
        assert success, "Failed to draw component in HFSS"
    finally:
        close_hfss(hfss, save_project=False)


@pytest.mark.hfss
def test_hfss_eigenmode_setup():
    """Test setting up an eigenmode simulation."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import (
        add_air_region_to_hfss,
        add_substrate_to_hfss,
        close_hfss,
        create_hfss_project,
        import_component_to_hfss,
        setup_eigenmode_simulation,
    )

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_eigenmode_{int(time.time())}"
    hfss = create_hfss_project(
        project_name=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        import_component_to_hfss(hfss, comp)
        add_substrate_to_hfss(hfss, comp)
        add_air_region_to_hfss(hfss, comp)

        setup = setup_eigenmode_simulation(hfss, num_modes=1, max_passes=2)
        assert setup is not None, "Failed to setup eigenmode simulation"
    finally:
        close_hfss(hfss, save_project=False)
