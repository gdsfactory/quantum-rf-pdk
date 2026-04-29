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
# # Q2D Cross-Section Impedance of a Coplanar Waveguide
#
# This notebook demonstrates how to extract the characteristic impedance
# :math:`Z_0` of a coplanar waveguide (CPW) cross-section using the
# Ansys 2D Extractor (Q2D) quasi-static field solver via PyAEDT.
#
# The Q2D solver computes per-unit-length RLGC parameters from the
# cross-sectional geometry, from which the characteristic impedance
# can be obtained as a function of frequency.  We compare the
# full-wave Q2D result against the analytical conformal-mapping estimate
# from :func:`~qpdk.models.cpw.cpw_parameters`.
#
# **Prerequisites:**
# - Ansys Electronics Desktop installed (requires license)
# - Install hfss extras: `uv sync --extra hfss` or `pip install qpdk[hfss]`
#
# **References:**
# - PyAEDT Documentation: https://aedt.docs.pyansys.com/
# - Q2D Coplanar Waveguide Example: https://examples.aedt.docs.pyansys.com/version/dev/examples/high_frequency/radiofrequency_mmwave/coplanar_waveguide.html
# - Simons, *Coplanar Waveguide Circuits, Components, and Systems* {cite:p}`simonsCoplanarWaveguideCircuits2001`

# %% [markdown]
# ## Setup and Imports

# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing quantum-rf-pdk...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ansys.aedt.core import Q2d, settings
from IPython.display import Image, display

from qpdk import PDK
from qpdk.config import PATH
from qpdk.models.cpw import cpw_parameters
from qpdk.simulation import Q2D
from qpdk.tech import coplanar_waveguide

PDK.activate()

# CPW dimensions
cpw_width = 10  # µm
cpw_gap = 6  # µm
cross_section = coplanar_waveguide(width=cpw_width, gap=cpw_gap)

# Analytical impedance estimate
ep_eff_analytical, z0_analytical = cpw_parameters(cpw_width, cpw_gap)

print(f"CPW dimensions: width = {cpw_width} µm, gap = {cpw_gap} µm")
print(
    f"Analytical estimate: Z₀ = {z0_analytical:.2f} Ω, ε_eff = {ep_eff_analytical:.4f}"
)

# %% [markdown]
# ## Initialize Q2D Project
#
# Set up an Ansys 2D Extractor project.  The Q2D solver uses a quasi-static
# approach to compute per-unit-length transmission-line parameters from the
# 2D cross-section.
#
# **Note:** This section requires Ansys Electronics Desktop to be installed
# and licensed.

# %%
# Configuration for Q2D simulation
Q2D_CONFIG = {
    "sweep_start_ghz": 1.0,  # Sweep from 1 GHz
    "sweep_stop_ghz": 10.0,  # to 10 GHz
    "sweep_step_ghz": 0.1,  # 100 MHz step
}

# %%
# Ensure Ansys path is set so PyAEDT can find it
ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
if "ANSYSEM_ROOT252" not in os.environ and Path(ansys_default_path).exists():
    os.environ["ANSYSEM_ROOT252"] = ansys_default_path

settings.use_grpc_uds = False

# Create temporary directory for project
temp_dir = tempfile.TemporaryDirectory(suffix=".ansys_qpdk")
project_path = Path(temp_dir.name) / "cpw_q2d.aedt"

# Initialize Q2D
q2d = Q2d(
    project=str(project_path),
    design="CPW_Impedance",
    non_graphical=False,
    new_desktop=True,
    version="2025.2",
)

print(f"Q2D project created: {q2d.project_file}")
print(f"Design name: {q2d.design_name}")

# %% [markdown]
# ## Build CPW Cross-Section Geometry
#
# Use :meth:`~qpdk.simulation.q3d.Q2D.create_2d_from_cross_section` to automatically
# build the CPW geometry (signal conductor, ground planes, substrate) from the
# gdsfactory cross-section and QPDK layer stack.

# %%

# Create the Q2D wrapper
q2d_sim = Q2D(q2d)

# Create the 2D cross-section geometry
object_names = q2d_sim.create_2d_from_cross_section(cross_section, ground_width=30)

print("Created Q2D geometry:")
for role, name in object_names.items():
    print(f"  {role}: {name}")

# %% [markdown]
# ### Q2D Cross-Section Geometry
# Here is the 2D geometry of the CPW cross-section in Ansys 2D Extractor.
#
# ![Q2D geometry](../docs/_static/images/q2d_cpw_impedance_geom.jpg)

# %%
# Ensure Q2D model fits the screen
q2d.modeler.fit_all()

# Save screenshot
img_dir = PATH.repo / "docs" / "_static" / "images"
img_dir.mkdir(parents=True, exist_ok=True)
q2d_img_path = img_dir / "q2d_cpw_impedance_geom.jpg"
q2d.post.export_model_picture(
    full_name=str(q2d_img_path), show_axis=True, show_grid=False, show_ruler=True
)

# Display in notebook
display(Image(filename=str(q2d_img_path)))

# %% [markdown]
# ## Configure Q2D Analysis
#
# Set up the solution with a frequency sweep from 1 GHz to 10 GHz to compute
# the characteristic impedance across the frequency range.

# %%
# Create setup
setup = q2d.create_setup(name="Q2DSetup")

# Add frequency sweep
sweep = setup.add_sweep(name="FrequencySweep")
sweep.props["RangeType"] = "LinearStep"
sweep.props["RangeStart"] = f"{Q2D_CONFIG['sweep_start_ghz']}GHz"
sweep.props["RangeStep"] = f"{Q2D_CONFIG['sweep_step_ghz']}GHz"
sweep.props["RangeEnd"] = f"{Q2D_CONFIG['sweep_stop_ghz']}GHz"
sweep.props["Type"] = "Interpolating"
sweep.update()

print("Q2D setup configured:")
print(
    f"  - Sweep range: {Q2D_CONFIG['sweep_start_ghz']} – {Q2D_CONFIG['sweep_stop_ghz']} GHz"
)
print(f"  - Step size: {Q2D_CONFIG['sweep_step_ghz']} GHz")

# %% [markdown]
# ## Run Simulation
#
# Execute the Q2D analysis.

# %%
print("Starting Q2D analysis...")
print("(This may take a few minutes)")

# Save project before analysis
q2d.save_project()

# Run the analysis
start_time = time.time()
success = q2d.analyze(cores=4)
elapsed = time.time() - start_time

if not success:
    raise RuntimeError("Q2D simulation failed. Check the AEDT log for details.")
else:
    print(f"Analysis completed in {elapsed:.1f} seconds")

# %% [markdown]
# ## Extract and Plot Impedance
#
# Extract the characteristic impedance :math:`Z_0` from Q2D and compare it
# with the analytical conformal-mapping estimate.  The analytical value is
# shown as a horizontal dashed line.

# %%
# Extract Z0 from Q2D
data = q2d.post.get_solution_data(
    expressions="Z0(signal,signal)",
    context="Original",
    setup_sweep_name="Q2DSetup : FrequencySweep",
)

frequencies_ghz = np.array(data.primary_sweep_values)
z0_q2d = np.array(data.data_real())

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

# Q2D result
ax.plot(frequencies_ghz, z0_q2d, "b-", linewidth=2, label="Q2D (quasi-static)")

# Analytical estimate as a horizontal line
ax.axhline(
    y=z0_analytical,
    color="r",
    linestyle="--",
    linewidth=1.5,
    label=f"Analytical (conformal mapping): {z0_analytical:.2f} Ω",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Characteristic Impedance $Z_0$ (Ω)")
ax.set_title("CPW Characteristic Impedance: Q2D vs. Analytical Estimate")
ax.legend()
ax.grid(True)
ax.set_xlim(Q2D_CONFIG["sweep_start_ghz"], Q2D_CONFIG["sweep_stop_ghz"])

plt.tight_layout()
plt.show()

# --- Numerical comparison ---
z0_mean_q2d = np.mean(z0_q2d)
relative_diff = (z0_mean_q2d - z0_analytical) / z0_analytical * 100

print("\n=== Impedance Comparison ===")
print("-" * 45)
print(f"Analytical Z₀ (conformal mapping): {z0_analytical:.2f} Ω")
print(f"Q2D Z₀ (mean over frequency):      {z0_mean_q2d:.2f} Ω")
print(f"Relative difference:                {relative_diff:+.2f}%")
print("-" * 45)

# %% [markdown]
# ## Cleanup
#
# Close the Q2D session and clean up temporary files.

# %%
# Save and close
q2d.save_project()
# q2d.release_desktop()  # Uncomment to close the AEDT desktop session
time.sleep(2)

# Clean up temp directory
temp_dir.cleanup()
print("Q2D session closed and temporary files cleaned up")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Cross-Section Definition**: Using QPDK's `coplanar_waveguide` cross-section
#    to define CPW geometry (10 µm width, 6 µm gap)
#
# 2. **Q2D Setup**: Initializing Ansys 2D Extractor via PyAEDT and building the
#    cross-sectional geometry using
#    :meth:`~qpdk.simulation.q3d.Q2D.create_2d_from_cross_section`
#
# 3. **Impedance Extraction**: Running the Q2D quasi-static solver to compute
#    :math:`Z_0` as a function of frequency from 1 to 10 GHz
#
# 4. **Analytical Validation**: Comparing the Q2D result with the conformal-mapping
#    analytical estimate from :func:`~qpdk.models.cpw.cpw_parameters`
#
# **Key Points for CPW Design:**
# - The Q2D solver gives frequency-dependent impedance including dispersion effects
# - The analytical conformal-mapping model provides a good quasi-static estimate
# - For superconducting CPWs (PEC conductors), the impedance is nearly
#   frequency-independent in the low-GHz range
# - No backplate metallisation is used, matching typical superconducting fabrication
#
# **Next Steps:**
# - Vary CPW dimensions to study impedance sensitivity
# - Compare with HFSS 3D driven-modal simulations
# - Study the effect of conductor thickness on impedance
#
# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
