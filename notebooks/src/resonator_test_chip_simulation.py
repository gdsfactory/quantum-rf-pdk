# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # SAX Simulation of a Resonator Test Chip
#
# This notebook demonstrates how to run a circuit-level SAX simulation of the
# `resonator_test_chip_yaml` component, which is defined via a `.pic.yml` netlist
# file and a corresponding gdsfactory+ schematic.
#
# The workflow is:
# 1. Load the component from the YAML netlist with gdsfactory.
# 2. Extract the netlist for circuit simulation.
# 3. Build a SAX circuit using the QPDK model library.
# 4. Evaluate the S-parameters over a frequency range.
# 5. Plot the transmission to observe resonator dips.

# %% tags=["hide-input", "hide-output"]
import warnings

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

from qpdk import PATH, PDK
from qpdk.models import models

PDK.activate()

# %% [markdown]
# ## Load the component
#
# The resonator test chip is defined in a `.pic.yml` file that lives alongside
# the QPDK sample scripts.  It contains 16 quarter-wave coupled resonators
# (8 per probeline), four launchers, and CPW routing between all elements.

# %%
yaml_path = PATH.samples / "resonator_test_chip_yaml.pic.yml"
chip = gf.read.from_yaml(yaml_path)
chip.plot()

# %% [markdown]
# ## Extract the netlist
#
# `Component.get_netlist()` returns a dictionary with `instances`,
# `nets`, `ports`, and `placements`.  SAX understands this format directly.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    netlist = chip.get_netlist()

print("Instances:")
for name, inst in netlist["instances"].items():
    print(f"  {name}: {inst['component']}")

# %% [markdown]
# ## Build the SAX circuit
#
# QPDK ships a `models` dictionary that maps every component name to a SAX
# model function.  We pass the netlist and models to `sax.circuit`.

# %%
circuit_fn, circuit_info = sax.circuit(
    netlist=netlist,
    models=models,
    on_internal_port="ignore",
)

# %% [markdown]
# ## Simulate
#
# Evaluate the circuit over the 5–9 GHz band.  The four external ports
# correspond to the four launchers:
#
# | Port | Launcher | Probeline |
# |------|----------|-----------|
# | o1   | West top | Top       |
# | o2   | East top | Top       |
# | o3   | West bot | Bottom    |
# | o4   | East bot | Bottom    |
#
# The top probeline has eight resonators with **varying** coupling gaps
# (12–26 µm) and the bottom probeline has eight resonators with a **fixed**
# coupling gap of 16 µm.

# %%
freq = jnp.linspace(5e9, 9e9, 5001)
s_params = circuit_fn(f=freq)

freq_ghz = freq / 1e9

# %% [markdown]
# ## Results
#
# ### Top probeline – variable coupling gap
#
# :math:`S_{21}` (port o1 → o2) shows eight notches, one per resonator.

# %%
s21 = s_params[("o1", "o2")]
s11 = s_params[("o1", "o1")]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="$S_{21}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s11)), label="$S_{11}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Top probeline (variable coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bottom probeline – fixed coupling gap
#
# :math:`S_{43}` (port o3 → o4) shows eight notches with uniform coupling
# depth because all resonators share the same 16 µm coupling gap.

# %%
s43 = s_params[("o3", "o4")]
s33 = s_params[("o3", "o3")]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s43)), label="$S_{43}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s33)), label="$S_{33}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Bottom probeline (fixed coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Both probelines
#
# Overlay both transmission traces to compare the two probelines.

# %%
fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="Top ($S_{21}$)")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s43)), label="Bottom ($S_{43}$)")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Resonator test chip – transmission comparison")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
