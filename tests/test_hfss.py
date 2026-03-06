"""Tests for HFSS simulation integration."""

import os
import pytest
import time
from pathlib import Path

# Ensure Ansys path is set so PyAEDT can find it
ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
if "ANSYSEM_ROOT252" not in os.environ and Path(ansys_default_path).exists():
    os.environ["ANSYSEM_ROOT252"] = ansys_default_path

# Require HFSS installation for these tests
pytestmark = pytest.mark.hfss

def test_hfss_import_and_draw():
    """Test creating an HFSS project and drawing a component."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import (
        create_hfss_project,
        import_component_to_hfss,
        close_hfss,
    )

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_draw_{int(time.time())}"
    hfss = create_hfss_project(
        project_name=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
        aedt_version="2025.2"
    )

    try:
        # Use direct draw which we know is stable on Linux
        success = import_component_to_hfss(hfss, comp, use_direct_draw=True)
        assert success, "Failed to draw component in HFSS"
    finally:
        close_hfss(hfss, save_project=False)


def test_hfss_eigenmode_setup():
    """Test setting up an eigenmode simulation."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from qpdk import PDK
    from qpdk.cells.resonator import resonator
    from qpdk.models.hfss import (
        create_hfss_project,
        import_component_to_hfss,
        add_substrate_to_hfss,
        add_air_region_to_hfss,
        setup_eigenmode_simulation,
        close_hfss,
    )

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_eigenmode_{int(time.time())}"
    hfss = create_hfss_project(
        project_name=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
        aedt_version="2025.2"
    )

    try:
        import_component_to_hfss(hfss, comp, use_direct_draw=True)
        add_substrate_to_hfss(hfss, comp)
        add_air_region_to_hfss(hfss, comp)
        
        setup = setup_eigenmode_simulation(hfss, num_modes=1, max_passes=2)
        assert setup is not None, "Failed to setup eigenmode simulation"
    finally:
        close_hfss(hfss, save_project=False)
