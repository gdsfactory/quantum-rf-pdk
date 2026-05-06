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
# # EMPro Driven Modal Simulation of an Interdigital Capacitor
#
# This notebook demonstrates how to set up and run a driven modal (S-parameter)
# simulation of an interdigital capacitor using Keysight EMPro Python interface.
#
# **Prerequisites:**
# - Keysight EMPro 2026 installed (requires license)
# - Run this notebook within the EMPro Python environment
#
# **References:**
# - Keysight EMPro Documentation (online)
# - Interdigital Capacitor Theory: {cite:p}`leizhuAccurateCircuitModel2000`

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
from qpdk.simulation import EMPro, prepare_component_for_aedt
import empro
import gdsfactory as gf
from pathlib import Path

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.waveguides import straight_open
from qpdk.tech import coplanar_waveguide

PDK.activate()

# Create an interdigital capacitor
cpw_width, cpw_gap = 10, 6
cross_section = coplanar_waveguide(width=cpw_width, gap=cpw_gap)
idc_component = interdigital_capacitor(
    fingers=6,
    finger_length=20.0,
    finger_gap=2.0,
    thickness=5.0,
    cross_section=cross_section,
)

# Attach straight open waveguides for port feeding
c = gf.Component(name="idc_with_feeds")
idc_ref = c << idc_component
open_wvg = straight_open(length=5, cross_section=cross_section)

feed1 = c << open_wvg
feed1.connect("o1", idc_ref.ports["o1"])
c.add_port("o1", port=feed1.ports["o2"])

feed2 = c << open_wvg
feed2.connect("o1", idc_ref.ports["o2"])
c.add_port("o2", port=feed2.ports["o2"])

idc_component = c

# %% [markdown]
# ## Initialize EMPro Project
#
# Connect to the active EMPro project and clear existing content.

# %%
# Clear active project
empro.activeProject.clear()

# Initialize EMPro wrapper
emp_sim = EMPro(empro.activeProject)

print("EMPro active project initialized")

# %% [markdown]
# ## Build EMPro Model
#
# Import the gdsfactory component geometry into EMPro.

# %%
# Prepare component for export
prepared_component = prepare_component_for_aedt(idc_component, margin_draw=50)

# Add materials to EMPro
emp_sim.add_materials()

# Import the component geometry via extrusion
created_parts = emp_sim.import_component(prepared_component)
print(f"Imported {len(created_parts)} parts into EMPro")

# Add substrate below the component
substrate_name = emp_sim.add_substrate(
    prepared_component,
    thickness=500.0,
    material="Si",
)
print(f"Created substrate: {substrate_name}")

# Add air region
air_region_name = emp_sim.add_air_region(
    prepared_component,
    height=500.0,
    substrate_thickness=500.0,
    material="vacuum",
)
print(f"Created air region: {air_region_name}")

# %% [markdown]
# ## Create Lumped Ports
#
# Add lumped ports at both ends of the capacitor.

# %%
print("Creating lumped ports.")
emp_sim.add_lumped_ports(prepared_component.ports)
for port in prepared_component.ports:
    print(f"  Created port: {port.name}")

# %% [markdown]
# ## Configure FEM Simulation
#
# Set up the FEM solver with a frequency sweep.

# %%
# Configure FEM simulation (1 GHz to 5 GHz)
emp_sim.setup_fem_simulation(start_freq=1e9, stop_freq=5e9, num_points=101)
emp_sim.setup_boundary_conditions()

print("FEM simulation configured (1-5 GHz) with Absorbing boundaries")

# %% [markdown]
# ## Run Simulation
#
# Execute the simulation. Note that EMPro requires the project to be saved
# before a simulation can be created.

# %%
# Save the project so it can be opened in the GUI and simulation can run
project_path = Path("idc_simulation.ep")
emp_sim.save_as(project_path)
print(f"Project saved to: {project_path.absolute()}")

print("Starting EMPro simulation...")
# This will not block, as the license might be in use
sim = emp_sim.run_simulation(wait=False)
print(f"Simulation status: {sim.status}")

if sim.status == "Error":
    print("Simulation failed. Retrieving log...")
    log = emp_sim.get_simulation_log(sim)
    print("--- EMPro Simulation Log ---")
    print(log)

# %% [markdown]
# ## Example 2: Driven Modal Simulation of a CPW Resonator
# 
# Now we demonstrate importing and simulating a coplanar waveguide resonator.
# We will create a new EMPro project for this example to keep it separate.

# %%
from qpdk.cells.resonator import resonator

# Clear the project again for the second example
empro.activeProject.clear()

# Initialize wrapper again
emp_sim_res = EMPro(empro.activeProject)

# Create a quarter-wave resonator
resonator_comp = resonator(
    length=3000.0,
    meanders=4,
    cross_section=cross_section
)

# Prepare component for export
prepared_res = prepare_component_for_aedt(resonator_comp, margin_draw=50)

# Import materials
emp_sim_res.add_materials()

# Import the geometry via extrusion
res_parts = emp_sim_res.import_component(prepared_res)
print(f"Imported {len(res_parts)} parts for the resonator")

# Add substrate and air
emp_sim_res.add_substrate(prepared_res, thickness=500.0, material="Si")
emp_sim_res.add_air_region(prepared_res, height=1000.0, substrate_thickness=500.0, material="vacuum")

# Add lumped ports at the feedline
print("Creating lumped ports for resonator.")
emp_sim_res.add_lumped_ports(prepared_res.ports)
for port in prepared_res.ports:
    print(f"  Created port: {port.name}")

# Configure FEM simulation
# Resonator fundamental frequency might be ~10 GHz for 3000 um length
emp_sim_res.setup_fem_simulation(start_freq=1e9, stop_freq=12e9, num_points=101)
emp_sim_res.setup_boundary_conditions()

# Save the second project
res_project_path = Path("resonator_simulation.ep")
emp_sim_res.save_as(res_project_path)
print(f"Resonator project saved to: {res_project_path.absolute()}")

# Run Simulation
print("Starting EMPro resonator simulation...")
# Run the simulation but don't wait for it if we are just demonstrating
res_sim = emp_sim_res.run_simulation(wait=False)
print(f"Resonator simulation queued. Status: {res_sim.status}")

# %% [markdown]
# ## Example 3: Driven Modal Simulation of a Fluxonium Qubit
# 
# Finally, let's look at an even more complex component: a Fluxonium qubit.
# This component has inductive elements, capacitive pads, and junctions.

# %%
from qpdk.cells.fluxonium import fluxonium

# Clear the project again for the third example
empro.activeProject.clear()
emp_sim_flux = EMPro(empro.activeProject)

# Create a fluxonium component
flux_comp = fluxonium()

# Prepare component for export
prepared_flux = prepare_component_for_aedt(flux_comp, margin_draw=100)

# Import materials
emp_sim_flux.add_materials()

# Import the geometry via extrusion
flux_parts = emp_sim_flux.import_component(prepared_flux)
print(f"Imported {len(flux_parts)} parts for the fluxonium qubit")

# Add substrate and air
emp_sim_flux.add_substrate(prepared_flux, thickness=500.0, material="Si")
emp_sim_flux.add_air_region(prepared_flux, height=1000.0, substrate_thickness=500.0, material="vacuum")

# Add lumped ports at the feedline
print("Creating lumped ports for fluxonium.")
emp_sim_flux.add_lumped_ports(prepared_flux.ports)
for port in prepared_flux.ports:
    print(f"  Created port: {port.name}")

# Configure FEM simulation
emp_sim_flux.setup_fem_simulation(start_freq=1e9, stop_freq=10e9, num_points=101)
emp_sim_flux.setup_boundary_conditions()

# Save the third project
flux_project_path = Path("fluxonium_simulation.ep")
emp_sim_flux.save_as(flux_project_path)
print(f"Fluxonium project saved to: {flux_project_path.absolute()}")

# Run Simulation
print("Starting EMPro fluxonium simulation...")
# Run the simulation but don't wait for it
flux_sim = emp_sim_flux.run_simulation(wait=False)
print(f"Fluxonium simulation queued. Status: {flux_sim.status}")

