# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # HFSS Eigenmode Simulation of a CPW Resonator
#
# This notebook demonstrates how to set up and run an eigenmode simulation
# of a superconducting coplanar waveguide (CPW) resonator using PyAEDT
# (Ansys HFSS Python interface).
#
# Eigenmode analysis finds the natural resonant frequencies and Q-factors
# of undriven electromagnetic structures - essential for designing
# superconducting qubits and resonators.
#
# **Prerequisites:**
# - Ansys HFSS installed (requires license)
# - Install hfss extras: `uv sync --extra hfss` or `pip install qpdk[hfss]`
#
# **References:**
# - PyAEDT Documentation: https://aedt.docs.pyansys.com/
# - HFSS Eigenmode Examples: https://examples.aedt.docs.pyansys.com/

# %% [markdown]
# ## Setup and Imports

# %% tags=["hide-input", "hide-output"]
import tempfile
import time
from pathlib import Path

# %% [markdown]
# ## Create a CPW Resonator Component
#
# First, let's create a simple quarter-wave resonator using QPDK's component library.
# We will use the `resonator_frequency` helper to find the length needed for a 5 GHz resonance.
# %%
import numpy as np
import skrf

from qpdk import PDK
from qpdk.cells.resonator import resonator
from qpdk.models.media import cross_section_to_media
from qpdk.models.resonator import resonator_frequency
from qpdk.tech import coplanar_waveguide

PDK.activate()

# Create a meandering quarter-wave resonator
# With default CPW dimensions (10µm width, 6µm gap)
cpw_cross_section = coplanar_waveguide(width=10, gap=6)

# Calculate length for 5 GHz target
target_f_hz = 5e9
f_arr = np.linspace(4e9, 6e9, 100)
freq = skrf.Frequency.from_f(f_arr, unit="Hz")
cpw_media = cross_section_to_media(cpw_cross_section)(frequency=freq)

# Simple fixed-point iteration to find the exact length
current_length = 4000.0
for _ in range(5):
    f_res = resonator_frequency(
        length=current_length, media=cpw_media, is_quarter_wave=True
    )
    current_length = current_length * f_res / target_f_hz

res_component = resonator(
    length=current_length,
    meanders=4,  # Number of meander turns
    cross_section=cpw_cross_section,
    open_start=True,  # Open end (voltage antinode)
    open_end=False,  # Shorted end (voltage node)
)

# Visualize the component
res_component.plot()
print(f"Resonator bounding box: {res_component.bbox}")
print(f"Expected quarter-wave frequency: ~{target_f_hz / 1e9:.2f} GHz")
print(f"Calculated length: {current_length:.1f} µm")

# %% [markdown]
# ## Initialize HFSS Project
#
# Now we'll set up an HFSS project for eigenmode analysis.
# The simulation will find the natural resonant modes of the structure.
#
# **Note:** This section requires Ansys HFSS to be installed and licensed.
# The code is wrapped in a try-except block for demonstration purposes.

# %%
# Configuration for HFSS simulation
EIGENMODE_CONFIG = {
    "min_frequency_ghz": 3.0,  # Start searching from 3 GHz
    "num_modes": 3,  # Find 3 eigenmodes
    "max_passes": 15,  # Maximum adaptive mesh passes
    "min_passes": 2,
    "percent_refinement": 30,
}

# %% [markdown]
# ## Build HFSS Model (Example Code)
#
# The following code demonstrates how to:
# 1. Create an HFSS project with eigenmode solution type
# 2. Build the CPW geometry in HFSS
# 3. Add substrate and boundary conditions
# 4. Configure eigenmode analysis
# 5. Run the simulation and extract results
#
# ```{note}
# This code requires Ansys HFSS to be installed. The example below shows
# the structure of a complete simulation workflow.
# ```
# %%
# Example HFSS eigenmode simulation workflow
# This code block demonstrates the full workflow but requires HFSS license

import os  # noqa: E402

# Ensure Ansys path is set so PyAEDT can find it
ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
if "ANSYSEM_ROOT252" not in os.environ and Path(ansys_default_path).exists():
    os.environ["ANSYSEM_ROOT252"] = ansys_default_path

from ansys.aedt.core import Hfss, settings  # noqa: E402

settings.use_grpc_uds = False


# Create temporary directory for project
temp_dir = tempfile.TemporaryDirectory(suffix=".ansys_qpdk")
project_path = Path(temp_dir.name) / "resonator_eigenmode.aedt"

# Initialize HFSS with Eigenmode solution
hfss = Hfss(
    project=str(project_path),
    design="CPW_Resonator",
    solution_type="Eigenmode",
    non_graphical=False,
    new_desktop=True,
    version="2025.2",
)
hfss.modeler.model_units = "um"

print(f"HFSS project created: {hfss.project_file}")
print(f"Design name: {hfss.design_name}")
print(f"Solution type: {hfss.solution_type}")

# %% [markdown]
# ## Build CPW Geometry in HFSS
#
# Import the gdsfactory component geometry into HFSS using native GDS import.
# This uses `Hfss.import_gds_3d` which automatically handles 3D layer mapping
# based on the QPDK LayerStack.

# %%
from qpdk.models.hfss import (  # noqa: E402
    add_air_region_to_hfss,
    add_substrate_to_hfss,
    get_eigenmode_results,
    import_component_to_hfss,
    prepare_component_for_hfss,
)

# Prepare component for export
res_component = prepare_component_for_hfss(res_component, margin=100)

# Import the component geometry using native GDS import
# This automatically applies additive metals and maps layers to 3D
success = import_component_to_hfss(hfss, res_component, import_as_sheets=True)
print(f"GDS import successful: {success}")

# Add substrate below the component
substrate_name = add_substrate_to_hfss(
    hfss,
    res_component,
    thickness=500.0,
    material="silicon",
)
print(f"Created substrate: {substrate_name}")

# Add air region with PEC boundary for eigenmode analysis
air_region_name = add_air_region_to_hfss(
    hfss,
    res_component,
    height=500.0,
    substrate_thickness=500.0,
)
print(f"Created air region with PEC boundary: {air_region_name}")

# %% [markdown]
# ## Configure Eigenmode Analysis
#
# Set up the eigenmode solver to find resonant frequencies starting from 3 GHz.

# %%
# Create eigenmode setup
setup = hfss.create_setup(name="EigenmodeSetup")

setup.props["MinimumFrequency"] = f"{EIGENMODE_CONFIG['min_frequency_ghz']}GHz"
setup.props["NumModes"] = EIGENMODE_CONFIG["num_modes"]
setup.props["MaximumPasses"] = EIGENMODE_CONFIG["max_passes"]
setup.props["MinimumPasses"] = EIGENMODE_CONFIG["min_passes"]
setup.props["PercentRefinement"] = EIGENMODE_CONFIG["percent_refinement"]
setup.props["ConvergeOnRealFreq"] = True
setup.props["MaxDeltaFreq"] = 2  # 2% convergence criterion

setup.update()
print("Eigenmode setup configured:")
print(f"  - Min frequency: {EIGENMODE_CONFIG['min_frequency_ghz']} GHz")
print(f"  - Number of modes: {EIGENMODE_CONFIG['num_modes']}")
print(f"  - Max passes: {EIGENMODE_CONFIG['max_passes']}")

# %% [markdown]
# ## Run Simulation
#
# Execute the eigenmode analysis. This may take several minutes depending
# on the mesh complexity and number of modes.

# %%
print("Starting eigenmode analysis...")
print("(This may take several minutes)")

# Save project before analysis
hfss.save_project()

# Run the analysis
start_time = time.time()
print("Starting eigenmode analysis...")
print("(This may take several minutes)")
success = hfss.analyze_setup("EigenmodeSetup", cores=4)
elapsed = time.time() - start_time

if not success:
    print("\nERROR: HFSS simulation failed!")
    # Try to get more info from HFSS logs if possible
else:
    print(f"Analysis completed in {elapsed:.1f} seconds")

# %% [markdown]
# ## Extract Results
#
# Get the eigenmode frequencies and Q-factors from the simulation.

# %%
# Extract results using the helper function
sim_results = get_eigenmode_results(hfss, "EigenmodeSetup")

print("\n=== Eigenmode Results ===")
print("-" * 40)

results = {
    "frequencies_ghz": sim_results["frequencies"],
    "q_factors": sim_results["q_factors"],
}

if not results["frequencies_ghz"]:
    print("No eigenmodes found. Check simulation logs and geometry.")
else:
    for i, (freq_ghz, q_factor) in enumerate(
        zip(results["frequencies_ghz"], results["q_factors"]), 1
    ):
        print(f"Mode {i}: f = {freq_ghz:.4f} GHz, Q = {q_factor:.1f}")

print("-" * 40)

# Compare with analytical estimate
expected_freq = target_f_hz / 1e9  # Target frequency
if results["frequencies_ghz"]:
    actual_freq = results["frequencies_ghz"][0]
    error_percent = abs(actual_freq - expected_freq) / expected_freq * 100
    print("\nComparison with analytical estimate:")
    print(f"  Expected (target): {expected_freq:.4f} GHz")
    print(f"  Simulated:         {actual_freq:.4f} GHz")
    print(f"  Difference:        {error_percent:.1f}%")

# %% [markdown]
# ## Cleanup
#
# Close HFSS and clean up temporary files.

# %%
# Save and close
hfss.save_project()
# hfss.release_desktop()
time.sleep(2)  # Allow HFSS to shut down

# Clean up temp directory
temp_dir.cleanup()
print("HFSS session closed and temporary files cleaned up")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Component Creation**: Using QPDK's `resonator` cell to create a
#    meandering CPW quarter-wave resonator
#
# 2. **HFSS Setup**: Initializing PyAEDT with eigenmode solution type
#
# 3. **Geometry Building**: Converting gdsfactory polygons to HFSS 3D geometry
#    with proper material assignments (PEC for superconducting metal)
#
# 4. **Eigenmode Analysis**: Configuring and running the solver to find
#    resonant frequencies and Q-factors
#
# 5. **Results Extraction**: Getting mode frequencies and Q-factors for
#    comparison with analytical models
#
# **Key Points for Superconducting Resonators:**
# - Use PerfectE (PEC) boundaries for superconducting metals at cryogenic temps
# - Silicon substrate with εᵣ ≈ 11.45 significantly affects resonance frequency
# - Q-factors from eigenmode analysis represent unloaded Q (internal losses only)
# - Coupling to external circuits reduces measured Q (loaded Q)
#
# **Next Steps:**
# - Compare eigenmode results with SAX circuit simulations
# - Add lossy materials to estimate realistic Q-factors
# - Study coupling effects with driven modal simulations
