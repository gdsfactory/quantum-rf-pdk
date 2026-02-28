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
# # HFSS Driven Modal Simulation of an Interdigital Capacitor
#
# This notebook demonstrates how to set up and run a driven modal (S-parameter)
# simulation of an interdigital capacitor using PyAEDT (Ansys HFSS Python interface).
#
# Driven modal analysis computes scattering parameters (S-parameters) of structures
# with ports, enabling characterization of coupling capacitance, insertion loss,
# and frequency-dependent behavior.
#
# **Prerequisites:**
# - Ansys HFSS installed (requires license)
# - Install hfss extras: `uv sync --extra hfss` or `pip install qpdk[hfss]`
#
# **References:**
# - PyAEDT Documentation: https://aedt.docs.pyansys.com/
# - HFSS Driven Modal Examples: https://examples.aedt.docs.pyansys.com/
# - Interdigital Capacitor Theory: :cite:`leizhuAccurateCircuitModel2000`

# %% [markdown]
# ## Setup and Imports

# %% tags=["hide-input", "hide-output"]
import tempfile
import time
from pathlib import Path

import numpy as np

# %% [markdown]
# ## Create an Interdigital Capacitor Component
#
# We'll use QPDK's interdigital capacitor cell, which creates interleaved
# metal fingers for distributed capacitance.

# %%
from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.tech import LAYER, coplanar_waveguide

PDK.activate()

# Create an interdigital capacitor
# This design has 6 fingers with 20µm finger length
idc_component = interdigital_capacitor(
    fingers=6,  # Number of interleaved fingers
    finger_length=20.0,  # Length of each finger in µm
    finger_gap=2.0,  # Gap between adjacent fingers in µm
    thickness=5.0,  # Finger width in µm
    cross_section=coplanar_waveguide(width=10, gap=6),
)

# Visualize the component
idc_component.plot()
print(f"Interdigital capacitor bounding box: {idc_component.bbox}")
print(f"Number of ports: {len(idc_component.ports)}")
for port_name, port in idc_component.ports.items():
    print(f"  {port_name}: center={port.center}, orientation={port.orientation}°")

# %% [markdown]
# ## Estimate Capacitance
#
# For reference, we can estimate the capacitance using analytical formulas
# before running the full-wave simulation.

# %%
# Approximate analytical capacitance for interdigital capacitor
# Using simplified parallel plate + fringe field model
epsilon_0 = 8.854e-12  # F/m
epsilon_r_si = 11.45  # Silicon relative permittivity
epsilon_eff = (1 + epsilon_r_si) / 2  # Effective permittivity (air above, Si below)

# Capacitance per unit length of coupled lines
fingers = 6
finger_length = 20.0e-6  # Convert to meters
finger_gap = 2.0e-6
thickness = 5.0e-6

# Simplified estimate: C ≈ ε₀ * εᵣ_eff * (N-1) * L * (t/g)
# where N = fingers, L = finger_length, t = thickness, g = gap
C_estimate = epsilon_0 * epsilon_eff * (fingers - 1) * finger_length * (thickness / finger_gap)
print(f"Estimated capacitance: {C_estimate * 1e15:.2f} fF")

# %% [markdown]
# ## Initialize HFSS Project
#
# Set up an HFSS project for driven modal analysis with ports.
#
# **Note:** This section requires Ansys HFSS to be installed and licensed.

# %%
# Configuration for HFSS simulation
HFSS_CONFIG = {
    "non_graphical": True,  # Set to False to see the HFSS GUI
    "aedt_version": None,  # Use default version
    "new_desktop": True,
}

DRIVEN_CONFIG = {
    "solution_frequency_ghz": 5.0,  # Adaptive mesh at 5 GHz
    "sweep_start_ghz": 0.1,  # Sweep from 100 MHz
    "sweep_stop_ghz": 20.0,  # to 20 GHz
    "sweep_points": 401,  # Number of frequency points
    "max_passes": 10,
    "max_delta_s": 0.02,  # 2% S-parameter convergence
}

# %% [markdown]
# ## Build HFSS Model (Example Code)
#
# The following demonstrates the complete workflow for driven modal simulation:
# 1. Create HFSS project with "DrivenModal" solution type
# 2. Build capacitor geometry with ports
# 3. Configure frequency sweep
# 4. Run simulation and extract S-parameters
#
# ```{note}
# This code requires Ansys HFSS. The structure below shows the complete workflow.
# ```

# %%
try:
    from ansys.aedt.core import Hfss

    # Create temporary directory for project
    temp_dir = tempfile.TemporaryDirectory(suffix=".ansys_qpdk")
    project_path = Path(temp_dir.name) / "idc_driven.aedt"

    # Initialize HFSS with Driven Modal solution
    hfss = Hfss(
        project=str(project_path),
        design="InterdigitalCapacitor",
        solution_type="DrivenModal",
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
# ## Build Interdigital Capacitor Geometry
#
# Convert the gdsfactory component geometry into HFSS 3D objects
# and add lumped ports at both ends.

# %%
if HFSS_AVAILABLE:
    from qpdk.models.hfss import component_polygons_to_numpy

    # Get component bounds
    bounds = idc_component.bbox
    margin = 50  # µm margin around capacitor

    # Extract polygons
    metal_polygons = component_polygons_to_numpy(idc_component, LAYER.M1_DRAW)
    etch_polygons = component_polygons_to_numpy(idc_component, LAYER.M1_ETCH)

    print(f"Found {len(metal_polygons)} metal polygons")
    print(f"Found {len(etch_polygons)} etch polygons")

    # Geometry dimensions
    substrate_thickness = 500  # µm
    metal_thickness = 0.2  # µm (200nm)
    air_height = 200  # µm

    x_min, y_min = bounds[0] - margin
    x_max, y_max = bounds[1] + margin

    # Create substrate
    substrate = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[x_max - x_min, y_max - y_min, substrate_thickness],
        name="Substrate",
        material="silicon",
    )
    print(f"Created substrate: {substrate.name}")

    # Create ground plane
    ground_plane = hfss.modeler.create_box(
        origin=[x_min, y_min, 0],
        sizes=[x_max - x_min, y_max - y_min, metal_thickness],
        name="GroundPlane",
    )
    hfss.assign_perfect_conductor(ground_plane.name, name="PEC_Ground")

    # Create etch regions to form CPW gaps and capacitor pattern
    etch_objects = []
    for i, poly in enumerate(etch_polygons):
        points = [[float(x), float(y), 0.0] for x, y in poly]
        points.append(points[0])

        etch_obj = hfss.modeler.create_polyline(
            points=points,
            cover_surface=True,
            name=f"Etch_{i}",
        )
        if etch_obj:
            hfss.modeler.thicken_sheet(etch_obj, metal_thickness * 2)
            etch_objects.append(etch_obj.name)

    # Subtract etch regions
    if etch_objects:
        hfss.modeler.subtract(ground_plane.name, etch_objects, keep_originals=False)
        print(f"Created capacitor pattern by subtracting {len(etch_objects)} etch regions")

    # Create air region
    air_region = hfss.modeler.create_box(
        origin=[x_min, y_min, 0],
        sizes=[x_max - x_min, y_max - y_min, air_height],
        name="AirRegion",
        material="vacuum",
    )

    # Assign radiation boundary to outer faces
    hfss.assign_radiation_boundary_to_objects(air_region)
    print("Assigned radiation boundary")

# %% [markdown]
# ## Create Lumped Ports
#
# Add lumped ports at both ends of the capacitor to measure S-parameters.
# The ports are placed at the CPW feed locations.

# %%
if HFSS_AVAILABLE:
    # Get port locations from component
    port_locations = {}
    for port_name, port in idc_component.ports.items():
        port_locations[port_name] = {
            "center": port.center,
            "orientation": port.orientation,
        }

    print("Creating lumped ports at component port locations:")
    port_objects = []

    for i, (port_name, port_info) in enumerate(port_locations.items(), 1):
        center = port_info["center"]
        orientation = port_info["orientation"]

        # Create a small rectangle for the port face
        # Port spans across the CPW gap
        cpw_width = 10  # µm
        cpw_gap = 6  # µm
        port_height = cpw_width + 2 * cpw_gap  # Total CPW cross-section

        # Determine port orientation and create rectangle
        if abs(orientation) < 45 or abs(orientation - 180) < 45:
            # Port facing left or right
            port_rect = hfss.modeler.create_rectangle(
                origin=[center[0], center[1] - port_height / 2, 0],
                sizes=[metal_thickness, port_height],
                cs_plane="XZ" if abs(orientation) < 45 else "XZ",
                name=f"PortFace_{i}",
            )
        else:
            # Port facing up or down
            port_rect = hfss.modeler.create_rectangle(
                origin=[center[0] - port_height / 2, center[1], 0],
                sizes=[port_height, metal_thickness],
                cs_plane="YZ",
                name=f"PortFace_{i}",
            )

        # Create lumped port
        if port_rect:
            port = hfss.lumped_port(
                assignment=port_rect.name,
                name=f"Port{i}",
                impedance=50,
            )
            port_objects.append(port)
            print(f"  Created Port{i} at {center}")

# %% [markdown]
# ## Configure Driven Modal Analysis
#
# Set up the solution with frequency sweep to compute S-parameters
# across the desired frequency range.

# %%
if HFSS_AVAILABLE:
    # Create driven modal setup
    setup = hfss.create_setup(
        name="DrivenSetup",
        setup_type="HFSSDriven",
        Frequency=f"{DRIVEN_CONFIG['solution_frequency_ghz']}GHz",
    )

    setup.props["MaxDeltaS"] = DRIVEN_CONFIG["max_delta_s"]
    setup.props["MaximumPasses"] = DRIVEN_CONFIG["max_passes"]
    setup.props["MinimumPasses"] = 2
    setup.props["PercentRefinement"] = 30
    setup.update()

    # Create frequency sweep
    sweep = setup.create_frequency_sweep(
        unit="GHz",
        name="FrequencySweep",
        start_frequency=DRIVEN_CONFIG["sweep_start_ghz"],
        stop_frequency=DRIVEN_CONFIG["sweep_stop_ghz"],
        sweep_type="Interpolating",
        num_of_freq_points=DRIVEN_CONFIG["sweep_points"],
    )

    print("Driven modal setup configured:")
    print(f"  - Solution frequency: {DRIVEN_CONFIG['solution_frequency_ghz']} GHz")
    print(f"  - Sweep range: {DRIVEN_CONFIG['sweep_start_ghz']} - {DRIVEN_CONFIG['sweep_stop_ghz']} GHz")
    print(f"  - Number of points: {DRIVEN_CONFIG['sweep_points']}")

# %% [markdown]
# ## Run Simulation
#
# Execute the driven modal analysis with frequency sweep.

# %%
if HFSS_AVAILABLE:
    print("Starting driven modal analysis...")
    print("(This may take several minutes)")

    # Save project before analysis
    hfss.save_project()

    # Run the analysis
    start_time = time.time()
    hfss.analyze_setup("DrivenSetup", cores=4)
    elapsed = time.time() - start_time

    print(f"Analysis completed in {elapsed:.1f} seconds")

# %% [markdown]
# ## Extract and Plot S-Parameters
#
# Get the S-parameters from the simulation and visualize the results.

# %%
if HFSS_AVAILABLE:
    import matplotlib.pyplot as plt

    # Get available S-parameter traces
    traces = hfss.get_traces_for_plot()
    print(f"Available traces: {traces}")

    # Create report and get solution data
    report = hfss.post.create_report(traces)
    solution = report.get_solution_data()

    # Extract frequency and S-parameters
    frequencies = np.array(solution.primary_sweep_values) / 1e9  # Convert to GHz

    # Plot S-parameters
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # S11 (reflection) and S21 (transmission)
    for expr in solution.expressions:
        data = solution.data_magnitude(expression=expr)
        data_db = 20 * np.log10(np.abs(data) + 1e-10)

        if "S(1,1)" in expr or "S11" in expr.upper():
            axes[0].plot(frequencies, data_db, label=expr)
        elif "S(2,1)" in expr or "S21" in expr.upper():
            axes[1].plot(frequencies, data_db, label=expr)

    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("$|S_{11}|$ (dB)")
    axes[0].set_title("Return Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("$|S_{21}|$ (dB)")
    axes[1].set_title("Insertion Loss")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Extract Capacitance from S-Parameters
#
# For a series capacitor, we can extract the capacitance from the
# impedance calculated from S-parameters.

# %%
if HFSS_AVAILABLE:
    # Extract capacitance from S21 at a specific frequency
    # For a series capacitor: Z = 1/(jωC), so |S21| relates to capacitive reactance

    # Get S21 data
    s21_data = None
    for expr in solution.expressions:
        if "S(2,1)" in expr or "S21" in expr.upper():
            s21_data = solution.data_magnitude(expression=expr)
            break

    if s21_data is not None:
        # Find frequency indices for analysis
        freq_1ghz_idx = np.argmin(np.abs(frequencies - 1.0))
        freq_5ghz_idx = np.argmin(np.abs(frequencies - 5.0))
        freq_10ghz_idx = np.argmin(np.abs(frequencies - 10.0))

        print("\n=== Capacitance Analysis ===")
        print("-" * 40)

        Z0 = 50  # Reference impedance (ohms)
        for freq_idx, freq_label in [(freq_1ghz_idx, "1 GHz"),
                                      (freq_5ghz_idx, "5 GHz"),
                                      (freq_10ghz_idx, "10 GHz")]:
            freq_hz = frequencies[freq_idx] * 1e9
            s21_mag = np.abs(s21_data[freq_idx])

            # For series element: S21 = 2Z0 / (2Z0 + Z)
            # Solving for |Z|: |Z| = 2Z0 * (1 - |S21|) / |S21|
            if s21_mag > 0.01:
                z_series = 2 * Z0 * (1 - s21_mag) / s21_mag
                # For capacitor: Z = 1/(ωC), so C = 1/(ω|Z|)
                omega = 2 * np.pi * freq_hz
                C_extracted = 1 / (omega * z_series)
                print(f"At {freq_label}: |S21| = {s21_mag:.4f}, C ≈ {C_extracted * 1e15:.2f} fF")

        print("-" * 40)
        print(f"Analytical estimate: {C_estimate * 1e15:.2f} fF")

# %% [markdown]
# ## Cleanup
#
# Close HFSS and clean up temporary files.

# %%
if HFSS_AVAILABLE:
    # Save and close
    hfss.save_project()
    hfss.release_desktop()
    time.sleep(2)

    # Clean up temp directory
    temp_dir.cleanup()
    print("HFSS session closed and temporary files cleaned up")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Component Creation**: Using QPDK's `interdigital_capacitor` cell
#    to create a planar capacitor with interleaved metal fingers
#
# 2. **HFSS Setup**: Initializing PyAEDT with driven modal solution type
#
# 3. **Geometry Building**: Converting gdsfactory polygons to HFSS 3D geometry
#    with CPW structure and etched capacitor pattern
#
# 4. **Port Creation**: Adding lumped ports at the CPW feed locations
#    for S-parameter measurements
#
# 5. **Driven Analysis**: Configuring frequency sweep and running the solver
#
# 6. **Results Extraction**: Getting S-parameters and extracting capacitance
#    values from the simulated data
#
# **Key Points for Capacitor Design:**
# - Interdigital capacitors provide high capacitance in compact area
# - S-parameters capture both capacitance and parasitic effects
# - Comparison with analytical models helps validate simulation setup
# - Frequency-dependent behavior reveals parasitic inductance at high frequencies
#
# **Next Steps:**
# - Parameter sweep to study capacitance vs. finger count
# - Compare with lumped element circuit models
# - Study loss tangent effects for realistic Q estimation
