"""Tests for EMPro simulation integration."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.simulation import EMPro, prepare_component_for_aedt

if TYPE_CHECKING:
    pass


def test_empro_importable() -> None:
    """Test that EMPro class can be imported."""
    from qpdk.simulation.empro import EMPro  # noqa: PLC0415

    assert EMPro is not None


def is_empro_available() -> bool:
    """Check if empro module is available."""
    return importlib.util.find_spec("empro") is not None


@pytest.mark.skipif(
    not is_empro_available(), reason="EMPro not available in this environment"
)
def test_empro_workflow() -> None:
    """Test full EMPro workflow including project initialization and geometry creation."""
    import empro  # noqa: PLC0415

    PDK.activate()

    # Create a small component for faster testing
    comp = interdigital_capacitor(fingers=2, finger_length=10)
    prepared_comp = prepare_component_for_aedt(comp)

    # Initialize project
    empro.activeProject.clear()
    emp_sim = EMPro(empro.activeProject)

    # Test materials
    emp_sim.add_materials()
    assert len(empro.activeProject.materials) > 0

    # Test geometry import
    created_objects = emp_sim.import_component(prepared_comp)
    assert len(created_objects) > 0

    # Test substrate
    sub_name = emp_sim.add_substrate(prepared_comp, thickness=100)
    assert sub_name == "Substrate"

    # Test air region
    air_name = emp_sim.add_air_region(
        prepared_comp, height=100, substrate_thickness=100
    )
    assert air_name == "AirRegion"

    # Test ports
    emp_sim.add_lumped_ports(prepared_comp.ports)
    assert len(empro.activeProject.circuitComponents) >= len(prepared_comp.ports)

    # Test simulation setup
    emp_sim.setup_fem_simulation(start_freq=1e9, stop_freq=2e9, num_points=11)
    assert "fem" in empro.activeProject.simulationSettings.simulator
    assert len(empro.activeProject.simulationSettings.femFrequencyPlanList()) == 1
