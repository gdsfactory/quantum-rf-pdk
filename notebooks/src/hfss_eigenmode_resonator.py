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

# %%
from qpdk import PDK
from qpdk.cells.resonator import resonator
from qpdk.tech import LAYER, coplanar_waveguide

PDK.activate()

# Create a meandering quarter-wave resonator
# With default CPW dimensions (10µm width, 6µm gap) and ~4mm length
cpw_cross_section = coplanar_waveguide(width=10, gap=6)
res_component = resonator(
    length=4000,  # 4mm total length
    meanders=4,  # Number of meander turns
    cross_section=cpw_cross_section,
    open_start=True,  # Open end (voltage antinode)
    open_end=False,  # Shorted end (voltage node)
)

# Visualize the component
res_component.plot()
print(f"Resonator bounding box: {res_component.bbox}")
print(f"Expected quarter-wave frequency: ~{3e8 / (4 * 4000e-6) / 1e9:.2f} GHz")

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
HFSS_CONFIG = {
    "non_graphical": True,  # Set to False to see the HFSS GUI
    "aedt_version": None,  # Use default version, or specify e.g., "2025.1"
    "new_desktop": True,
}

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

try:
    from ansys.aedt.core import Hfss

    # Create temporary directory for project
    temp_dir = tempfile.TemporaryDirectory(suffix=".ansys_qpdk")
    project_path = Path(temp_dir.name) / "resonator_eigenmode.aedt"

    # Initialize HFSS with Eigenmode solution
    hfss = Hfss(
        project=str(project_path),
        design="CPW_Resonator",
        solution_type="Eigenmode",
        non_graphical=HFSS_CONFIG["non_graphical"],
        new_desktop=HFSS_CONFIG["new_desktop"],
    )
    hfss.modeler.model_units = "um"

    print(f"HFSS project created: {hfss.project_file}")
    print(f"Design name: {hfss.design_name}")
    print(f"Solution type: {hfss.solution_type}")

    HFSS_AVAILABLE = True

except ImportError:
    print("PyAEDT not installed. Install with: pip install qpdk[hfss]")
    print("Continuing with demonstration of the workflow structure...")
    HFSS_AVAILABLE = False
except Exception as e:
    print(f"HFSS not available: {e}")
    print("This is expected if Ansys HFSS is not installed.")
    HFSS_AVAILABLE = False

# %% [markdown]
# ## Build CPW Geometry in HFSS
#
# Convert the gdsfactory component geometry into HFSS 3D objects.
# We create the CPW structure with proper material assignments.

# %%
if HFSS_AVAILABLE:
    import numpy as np

    from qpdk.models.hfss import component_polygons_to_numpy

    # Get component bounds for geometry creation
    bounds = res_component.bbox
    margin = 100  # µm margin around resonator

    # Extract polygons from the metal draw layer (M1_DRAW)
    metal_polygons = component_polygons_to_numpy(res_component, LAYER.M1_DRAW)
    etch_polygons = component_polygons_to_numpy(res_component, LAYER.M1_ETCH)

    print(f"Found {len(metal_polygons)} metal polygons")
    print(f"Found {len(etch_polygons)} etch polygons")

    # Create substrate
    substrate_thickness = 500  # µm (typical silicon wafer)
    x_min, y_min = bounds[0] - margin
    x_max, y_max = bounds[1] + margin

    substrate = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[x_max - x_min, y_max - y_min, substrate_thickness],
        name="Substrate",
        material="silicon",
    )
    print(f"Created substrate: {substrate.name}")

    # Create ground plane (metal layer covering substrate top)
    metal_thickness = 0.2  # µm (200nm Nb film)
    ground_plane = hfss.modeler.create_box(
        origin=[x_min, y_min, 0],
        sizes=[x_max - x_min, y_max - y_min, metal_thickness],
        name="GroundPlane",
    )
    hfss.assign_perfect_conductor(ground_plane.name, name="PEC_Ground")
    print(f"Created ground plane: {ground_plane.name}")

    # Create etch regions (remove metal to form CPW gaps)
    etch_objects = []
    for i, poly in enumerate(etch_polygons):
        points = [[float(x), float(y), 0.0] for x, y in poly]
        points.append(points[0])  # Close polygon

        etch_obj = hfss.modeler.create_polyline(
            points=points,
            cover_surface=True,
            name=f"Etch_{i}",
        )
        if etch_obj:
            hfss.modeler.thicken_sheet(etch_obj, metal_thickness * 2)
            etch_objects.append(etch_obj.name)

    # Subtract etch regions from ground plane
    if etch_objects:
        hfss.modeler.subtract(ground_plane.name, etch_objects, keep_originals=False)
        print(f"Created CPW pattern by subtracting {len(etch_objects)} etch regions")

    # Create air region above
    air_height = 500  # µm
    air_region = hfss.modeler.create_box(
        origin=[x_min, y_min, 0],
        sizes=[x_max - x_min, y_max - y_min, air_height],
        name="AirRegion",
        material="vacuum",
    )

    # Assign PerfectE boundary to outer faces for eigenmode
    outer_faces = []
    for face in air_region.faces:
        outer_faces.append(face.id)
    for face in substrate.faces:
        # Bottom and side faces of substrate
        if abs(face.center[2] - (-substrate_thickness)) < 1:
            outer_faces.append(face.id)

    hfss.assign_perfect_conductor(
        assignment=outer_faces,
        name="PEC_Boundary",
    )
    print("Assigned PEC boundary conditions")

# %% [markdown]
# ## Configure Eigenmode Analysis
#
# Set up the eigenmode solver to find resonant frequencies starting from 3 GHz.

# %%
if HFSS_AVAILABLE:
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
if HFSS_AVAILABLE:
    print("Starting eigenmode analysis...")
    print("(This may take several minutes)")

    # Save project before analysis
    hfss.save_project()

    # Run the analysis
    start_time = time.time()
    hfss.analyze_setup("EigenmodeSetup", cores=4, use_auto_settings=True)
    elapsed = time.time() - start_time

    print(f"Analysis completed in {elapsed:.1f} seconds")

# %% [markdown]
# ## Extract Results
#
# Get the eigenmode frequencies and Q-factors from the simulation.

# %%
if HFSS_AVAILABLE:
    # Get eigenmode frequencies
    freq_names = hfss.post.available_report_quantities(
        quantities_category="Eigen Modes"
    )
    q_names = hfss.post.available_report_quantities(quantities_category="Eigen Q")

    print("\n=== Eigenmode Results ===")
    print("-" * 40)

    results = {"frequencies_ghz": [], "q_factors": []}

    for i, (f_name, q_name) in enumerate(zip(freq_names, q_names), 1):
        # Get frequency
        f_solution = hfss.post.get_solution_data(
            expressions=f_name, report_category="Eigenmode"
        )
        freq_hz = float(f_solution.data_real()[0])
        freq_ghz = freq_hz / 1e9

        # Get Q-factor
        q_solution = hfss.post.get_solution_data(
            expressions=q_name, report_category="Eigenmode"
        )
        q_factor = float(q_solution.data_real()[0])

        results["frequencies_ghz"].append(freq_ghz)
        results["q_factors"].append(q_factor)

        print(f"Mode {i}: f = {freq_ghz:.4f} GHz, Q = {q_factor:.1f}")

    print("-" * 40)

    # Compare with analytical estimate
    expected_freq = 3e8 / (4 * 4000e-6) / 1e9  # Simple quarter-wave estimate
    if results["frequencies_ghz"]:
        actual_freq = results["frequencies_ghz"][0]
        error_percent = abs(actual_freq - expected_freq) / expected_freq * 100
        print(f"\nComparison with analytical estimate:")
        print(f"  Expected (λ/4): {expected_freq:.4f} GHz")
        print(f"  Simulated:      {actual_freq:.4f} GHz")
        print(f"  Difference:     {error_percent:.1f}%")

# %% [markdown]
# ## Cleanup
#
# Close HFSS and clean up temporary files.

# %%
if HFSS_AVAILABLE:
    # Save and close
    hfss.save_project()
    hfss.release_desktop()
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
