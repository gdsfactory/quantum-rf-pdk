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

# %% [markdown]
# ## Create an Interdigital Capacitor Component
#
# We'll use QPDK's interdigital capacitor cell, which creates interleaved
# metal fingers for distributed capacitance.
# %%
import gdsfactory as gf
import numpy as np

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.waveguides import straight_open
from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.media import cpw_ep_r_from_cross_section
from qpdk.tech import coplanar_waveguide

PDK.activate()

# Create an interdigital capacitor
# This design has 6 fingers with 20µm finger length
cpw_width, cpw_gap = 10, 6
cross_section = coplanar_waveguide(width=cpw_width, gap=cpw_gap)
idc_component = interdigital_capacitor(
    fingers=6,  # Number of interleaved fingers
    finger_length=20.0,  # Length of each finger in µm
    finger_gap=2.0,  # Gap between adjacent fingers in µm
    thickness=5.0,  # Finger width in µm
    cross_section=cross_section,
)

# Attach straight open waveguides to the ports to provide a feedline
# for the lumped ports in HFSS.
c = gf.Component(name="idc_with_feeds")
idc_ref = c << idc_component
open_wvg = straight_open(length=5, cross_section=cross_section)

# Connect feedlines to both ports
feed1 = c << open_wvg
feed1.connect("o1", idc_ref.ports["o1"])
c.add_port("o1", port=feed1.ports["o2"])

feed2 = c << open_wvg
feed2.connect("o1", idc_ref.ports["o2"])
c.add_port("o2", port=feed2.ports["o2"])

# Use the combined component for the rest of the notebook
idc_component = c

# Visualize the component
idc_component.show()
print(f"Interdigital capacitor bounding box: {idc_component.bbox}")
print(f"Number of ports: {len(idc_component.ports)}")
for port in idc_component.ports:
    print(f"  {port.name}: center={port.center}, orientation={port.orientation}°")

# %% [markdown]
# ## Estimate Capacitance
#
# Before running the full-wave simulation, we can estimate the mutual capacitance
# using the analytical conformal mapping model for interdigital capacitors
# :cite:`igrejaAnalyticalEvaluationInterdigital2004`.
#
# For a structure with $n$ fingers of width $w$, gap $g$, and overlap length $L$,
# the metallization ratio is $\eta = \frac{w}{w + g}$. The interior and exterior
# capacitances per unit length are derived using the complete elliptic integrals
# of the first kind $K(k)$:
#
# $$ \eta = \frac{w}{w + g}, \quad k_i = \sin\left(\frac{\pi \eta}{2}\right), \quad k_e = \frac{2\sqrt{\eta}}{1 + \eta} $$
#
# $$ C_i = \epsilon_0 (\epsilon_r + 1) \frac{K(k_i)}{K(k_i')}, \quad C_e = \epsilon_0 (\epsilon_r + 1) \frac{K(k_e)}{K(k_e')} $$
#
# The total mutual capacitance for $n$ fingers is then:
#
# $$ C = \begin{cases} C_e L / 2 & \text{if } n=2 \\ (n - 3) \frac{C_i L}{2} + 2 \frac{C_i C_e L}{C_i + C_e} & \text{if } n > 2 \end{cases} $$

# %%
# Get substrate permittivity from cross-section
ep_r = cpw_ep_r_from_cross_section(cross_section)

# Analytical estimate using QPDK model
C_estimate = interdigital_capacitor_capacitance_analytical(
    fingers=6,
    finger_length=20.0,
    finger_gap=2.0,
    thickness=5.0,
    ep_r=float(ep_r),
)
print(f"Estimated capacitance: {float(C_estimate) * 1e15:.2f} fF")

# %% [markdown]
# ## Initialize HFSS Project
#
# Set up an HFSS project for driven modal analysis with ports.
#
# **Note:** This section requires Ansys HFSS to be installed and licensed.

# %%
# Configuration for HFSS simulation
HFSS_CONFIG = {
    "solution_frequency_ghz": 5.0,  # Adaptive mesh at 5 GHz
    "sweep_start_ghz": 0.1,  # Sweep from 100 MHz
    "sweep_stop_ghz": 20.0,  # to 20 GHz
    "sweep_points": 401,  # Number of frequency points
    "max_passes": 16,
    "max_delta_s": 0.002,  # 0.2% S-parameter convergence
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
# Example HFSS driven modal simulation workflow
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
project_path = Path(temp_dir.name) / "idc_driven.aedt"

# Initialize HFSS with Driven Modal solution
hfss = Hfss(
    project=str(project_path),
    design="InterdigitalCapacitor",
    solution_type="DrivenModal",
    non_graphical=False,
    new_desktop=True,
    version="2025.2",
)
hfss.modeler.model_units = "um"

print(f"HFSS project created: {hfss.project_file}")
print(f"Design name: {hfss.design_name}")
print(f"Solution type: {hfss.solution_type}")

# %% [markdown]
# ## Build Interdigital Capacitor Geometry
#
# Import the gdsfactory component geometry into HFSS using native GDS import.
# This uses `Hfss.import_gds_3d` which automatically handles 3D layer mapping.

# %%
from qpdk.models.hfss import (  # noqa: E402
    add_air_region_to_hfss,
    add_lumped_ports_to_hfss,
    add_substrate_to_hfss,
    import_component_to_hfss,
    prepare_component_for_hfss,
)

# Prepare component for export
prepared_component = prepare_component_for_hfss(idc_component, margin_draw=50)

# Import the component geometry using native GDS import
# This automatically applies additive metals and maps layers to 3D
success = import_component_to_hfss(hfss, prepared_component, import_as_sheets=True)
print(f"GDS import successful: {success}")

# Add substrate below the component
substrate_name = add_substrate_to_hfss(
    hfss,
    prepared_component,
    thickness=500.0,
    material="silicon",
)
print(f"Created substrate: {substrate_name}")

# Add air region for driven simulation
air_region_name = add_air_region_to_hfss(
    hfss,
    prepared_component,
    height=500.0,
    substrate_thickness=500.0,
    pec_boundary=False,
)
print(f"Created air region: {air_region_name}")

# Assign radiation boundary to outer faces for driven analysis
hfss.assign_radiation_boundary_to_objects(air_region_name)
print("Assigned radiation boundary to air region")

# %% [markdown]
# ## Create Lumped Ports
#
# Add lumped ports at both ends of the capacitor to measure S-parameters.
# The ports are placed at the CPW feed locations.

# %%

print("Creating lumped ports.")

add_lumped_ports_to_hfss(hfss, prepared_component.ports, cpw_gap, cpw_width)
for port in prepared_component.ports:
    print(
        f"  Created {port.name} at {port.center} with orientation {port.orientation}°"
    )

# %% [markdown]
# ## Configure Driven Modal Analysis
#
# Set up the solution with frequency sweep to compute S-parameters
# across the desired frequency range.

# %%
# Create driven modal setup
setup = hfss.create_setup(
    name="DrivenSetup",
    Frequency=f"{HFSS_CONFIG['solution_frequency_ghz']}GHz",
)

setup.props["MaxDeltaS"] = HFSS_CONFIG["max_delta_s"]
setup.props["MaximumPasses"] = HFSS_CONFIG["max_passes"]
setup.props["MinimumPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.update()

# Create frequency sweep
sweep = setup.create_frequency_sweep(
    unit="GHz",
    name="FrequencySweep",
    start_frequency=HFSS_CONFIG["sweep_start_ghz"],
    stop_frequency=HFSS_CONFIG["sweep_stop_ghz"],
    sweep_type="Interpolating",
    num_of_freq_points=HFSS_CONFIG["sweep_points"],
)

print("Driven modal setup configured:")
print(f"  - Solution frequency: {HFSS_CONFIG['solution_frequency_ghz']} GHz")
print(
    f"  - Sweep range: {HFSS_CONFIG['sweep_start_ghz']} - {HFSS_CONFIG['sweep_stop_ghz']} GHz"
)
print(f"  - Number of points: {HFSS_CONFIG['sweep_points']}")

# %% [markdown]
# ## Run Simulation
#
# Execute the driven modal analysis with frequency sweep.

# %%
print("Starting driven modal analysis...")
print("(This may take several minutes)")

# Save project before analysis
hfss.save_project()

# Run the analysis
start_time = time.time()
success = hfss.analyze_setup("DrivenSetup", cores=4)
elapsed = time.time() - start_time

if not success:
    print("\nERROR: HFSS simulation failed!")
else:
    print(f"Analysis completed in {elapsed:.1f} seconds")

# %% [markdown]
# ## Extract and Plot S-Parameters
#
# Get the S-parameters from the simulation and visualize the results.

# %%
import matplotlib.pyplot as plt  # noqa: E402

from qpdk.models.hfss import get_sparameter_results  # noqa: E402

# Extract results using the helper function
df_results = get_sparameter_results(hfss, setup.name, sweep.name)

frequencies_ghz = df_results["frequency_ghz"].to_numpy()

# Plot S-parameters
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Filter for S11 and S21 type traces
s11_col = next(col for col in df_results.columns if "S(o1:1,o1:1)" in col)
s21_col = next(col for col in df_results.columns if "S(o2:1,o1:1)" in col)

s11_trace = df_results[s11_col].to_numpy().astype(np.complex128)
s21_trace = df_results[s21_col].to_numpy().astype(np.complex128)

s11_mag_db = 20 * np.log10(np.abs(s11_trace))
s21_mag_db = 20 * np.log10(np.abs(s21_trace))

axes[0].plot(frequencies_ghz, s11_mag_db, label=r"|S_{11}|")
axes[1].plot(frequencies_ghz, s21_mag_db, label=r"|S_{21}|")

axes[0].set_xlabel("Frequency (GHz)")
axes[0].set_ylabel("Magnitude (dB)")
axes[0].set_title("Return Loss ($S_{11}$)")
axes[0].grid(True)
axes[0].legend()

axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_ylabel("Magnitude (dB)")
axes[1].set_title("Insertion Loss ($S_{21}$)")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Extract Capacitance from Admittance (Y-Parameters)
#
# Relying only on the magnitude of $S_{21}$ assumes the geometry behaves identically
# to a single perfect capacitor floating in a vacuum. In reality, the structure has
# shunt parasitic capacitances to ground ($C_{11}$ and $C_{22}$) that skew the
# $S_{21}$ magnitude.
#
# The robust way to extract mutual capacitance is using Y-parameters (Admittance).
# In a Pi-network model, the mutual admittance $Y_{12}$ isolates the series element:
#
# $$ Y_{12} = -j\omega C_{12} $$
#
# Therefore, the exact mutual capacitance is:
#
# $$ C_{12} = -\frac{\text{Im}(Y_{12})}{\omega} $$

# %%
# Extract Y21 parameter manually using PyAEDT's solution data
solution_y = hfss.post.get_solution_data(
    expressions="Y(o2:1,o1:1)", setup_sweep_name=f"{setup.name} : {sweep.name}"
)

# Parse complex Y-parameters
_, y_real = solution_y.get_expression_data(formula="real")
_, y_imag = solution_y.get_expression_data(formula="imag")
y21_trace = np.array(y_real) + 1j * np.array(y_imag)
frequencies_ghz_y = np.array(solution_y.primary_sweep_values)

# Analysis frequencies in GHz
analysis_frequencies_ghz = [1, 5, 10]

print("\n=== Capacitance Analysis ===")
print("-" * 40)
print(f"Analytical estimate: {C_estimate * 1e15:.2f} fF")
print("-" * 40)

for freq_target in analysis_frequencies_ghz:
    idx = np.argmin(np.abs(frequencies_ghz_y - freq_target))
    freq_hz = frequencies_ghz_y[idx] * 1e9
    y21 = y21_trace[idx]

    ω = 2 * np.pi * freq_hz
    # C_12 = -Im(Y_12) / w
    C_extracted = -np.imag(y21) / ω

    print(
        f"At {freq_hz / 1e9:.2f} GHz: Y21 = {y21:.2e}, C ≈ {C_extracted * 1e15:.2f} fF"
    )
    print(
        f"Relative difference: {(float(C_estimate) - C_extracted) / float(C_estimate) * 100:.2f}%"
    )

# %% [markdown]
# ## Cleanup
#
# Close HFSS and clean up temporary files.

# %%
# Save and close
hfss.save_project()
# hfss.release_desktop()
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
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
