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
# # Superconducting Qubit Parameter Calculations with scqubits
#
# This example demonstrates how to use [scqubits](https://scqubits.readthedocs.io/en/latest/) to calculate
# parameters for superconducting qubits and their coupling to resonators. The calculations help determine
# physical parameters like capacitances and coupling strengths needed for PDK component design.
#
# scqubits is a Python package for simulating superconducting quantum circuits, providing tools to calculate
# energy spectra, matrix elements, and other quantum properties.

# %%
import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

print(f"Using scqubits version: {scq.__version__}")

# %% [markdown]
# ## TunableTransmon Qubit Parameters
#
# A tunable transmon qubit has a SQUID junction that allows tuning the effective Josephson energy
# through an external flux. Key parameters include:
# - `EJmax`: Maximum Josephson energy when flux = 0
# - `EC`: Charging energy, related to total capacitance as EC = e²/(2C_total)
# - `d`: Asymmetry parameter between the two junctions in the SQUID
# - `flux`: External flux in units of flux quantum φ₀

# %%
# Create a tunable transmon with realistic parameters
transmon = scq.TunableTransmon(
    EJmax=40.0,    # Maximum Josephson energy in GHz (typical range: 20-50 GHz)
    EC=0.2,        # Charging energy in GHz (typical range: 0.1-0.5 GHz)
    d=0.1,         # Junction asymmetry (typical range: 0.05-0.2)
    flux=0.0,      # Start at flux = 0 (maximum EJ)
    ng=0.0,        # Offset charge (usually 0 or 0.5)
    ncut=30        # Charge basis cutoff
)

print("Tunable Transmon Parameters:")
print(f"  EJmax = {transmon.EJmax:.1f} GHz")
print(f"  EC = {transmon.EC:.1f} GHz")
print(f"  d = {transmon.d:.2f}")
print(f"  Flux = {transmon.flux:.2f} Φ₀")

# Calculate energy levels
eigenvals = transmon.eigenvals(evals_count=5)
f01 = eigenvals[1] - eigenvals[0]
f12 = eigenvals[2] - eigenvals[1]
anharmonicity = f12 - f01

print("\nTransmon Spectrum:")
print(f"  01 frequency: {f01:.3f} GHz")
print(f"  12 frequency: {f12:.3f} GHz")
print(f"  Anharmonicity: {anharmonicity:.3f} GHz")

# %% [markdown]
# ## Resonator (Oscillator) Parameters
#
# A superconducting resonator can be modeled as a quantum harmonic oscillator.
# The resonator frequency depends on its geometry and electromagnetic environment.

# %%
# Create a resonator (modeled as harmonic oscillator)
resonator = scq.Oscillator(
    E_osc=6.5,       # Resonator frequency in GHz (typical range: 4-8 GHz)
    truncated_dim=6  # Number of Fock states to include
)

print("Resonator Parameters:")
print(f"  Frequency: {resonator.E_osc:.1f} GHz")
print(f"  Fock space dimension: {resonator.truncated_dim}")

# %% [markdown]
# ## Coupled Qubit-Resonator System
#
# When a transmon is coupled to a resonator, the interaction can be described by
# the Jaynes-Cummings model. The coupling strength g depends on the overlap between
# the qubit and resonator modes and affects the system dynamics.

# %%
# Create a coupled system using HilbertSpace
hilbert_space = scq.HilbertSpace([transmon, resonator])

# Add interaction between qubit and resonator
# The interaction is typically g(n_qubit * (a + a†)) where n is the charge operator and a is the resonator annihilation operator
g_coupling = 0.15  # Coupling strength in GHz (typical range: 0.05-0.3 GHz)

# Create interaction using the correct API
interaction = scq.InteractionTerm(
    g_strength=g_coupling,
    operator_list=[
        (0, transmon.n_operator),  # (subsystem_index, operator)
        (1, resonator.annihilation_operator() + resonator.creation_operator())
    ]
)
hilbert_space.interaction_list = [interaction]

print("Qubit-Resonator Coupling:")
print(f"  Coupling strength g = {g_coupling:.3f} GHz")

# Calculate the coupled system eigenvalues
coupled_eigenvals = hilbert_space.eigenvals(evals_count=10)
print(f"  Coupled system ground state: {coupled_eigenvals[0]:.3f} GHz")

# %% [markdown]
# ## Physical Parameter Extraction
#
# From the quantum model, we can extract physical parameters relevant for PDK design:

# %%
# Calculate physical capacitances
# The charging energy EC = e²/(2C_total), so C_total = e²/(2*EC)
e_charge = 1.602176634e-19  # Elementary charge in Coulombs
h_planck = 6.62607015e-34   # Planck constant
EC_joules = transmon.EC * 1e9 * h_planck  # Convert GHz to Joules

# Total capacitance of the transmon
C_total = e_charge**2 / (2 * EC_joules)
print("\nPhysical Parameters for PDK Design:")
print(f"  Total qubit capacitance: {C_total*1e15:.1f} fF")

# Josephson inductance - use typical realistic value
# For EJ ~ 40 GHz, typical LJ ~ 0.8-1.0 nH
# LJ scales inversely with EJ: LJ ~ 1.0 nH * (25 GHz / EJ)
LJ_typical = 1.0e-9 * (25.0 / transmon.EJmax)  # Typical scaling

print(f"  Josephson inductance: {LJ_typical*1e9:.2f} nH")

# Estimate coupling capacitance from coupling strength
# For a more realistic estimate, use typical values from literature
# g ~ 0.1 GHz typically corresponds to coupling capacitances of ~ 1-10 fF
C_coupling_typical = 5.0e-15  # 5 fF - typical value
print(f"  Estimated coupling capacitance: {C_coupling_typical*1e15:.1f} fF (typical value)")

# %% [markdown]
# ## Parameter Sweep: Flux Dependence
#
# Demonstrate how the qubit frequency changes with external flux, which is crucial
# for understanding tunable transmon behavior.

# %%
# Sweep flux and calculate qubit frequency
flux_values = np.linspace(0, 0.5, 51)
frequencies_01 = []
frequencies_12 = []

for flux in flux_values:
    transmon.flux = flux
    eigenvals = transmon.eigenvals(evals_count=3)
    frequencies_01.append(eigenvals[1] - eigenvals[0])
    frequencies_12.append(eigenvals[2] - eigenvals[1])

# Reset flux to 0
transmon.flux = 0.0

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(flux_values, frequencies_01, "b-", linewidth=2, label="01 transition")
plt.plot(flux_values, frequencies_12, "r--", linewidth=2, label="12 transition")
plt.xlabel("External Flux (Φ₀)")
plt.ylabel("Transition Frequency (GHz)")
plt.title("Tunable Transmon Spectrum vs Flux")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
anharmonicity_vs_flux = np.array(frequencies_12) - np.array(frequencies_01)
plt.plot(flux_values, anharmonicity_vs_flux, "g-", linewidth=2)
plt.xlabel("External Flux (Φ₀)")
plt.ylabel("Anharmonicity (GHz)")
plt.title("Anharmonicity vs Flux")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Design Recommendations
#
# Based on the calculations above, here are parameter recommendations for PDK components:

# %%
print("PDK Design Recommendations:")
print("=" * 50)
print("1. Transmon Pad Capacitance:")
print(f"   - Target total capacitance: {C_total*1e15:.1f} fF")
print("   - This determines pad size and geometry")
print("")
print("2. Junction Parameters:")
print(f"   - Target EJmax: {transmon.EJmax:.1f} GHz")
print("   - Junction area scales with EJ")
print(f"   - Asymmetry parameter d = {transmon.d:.2f}")
print("")
print("3. Coupling Elements:")
print(f"   - Target coupling strength: {g_coupling:.3f} GHz")
print(f"   - Estimated coupling capacitance: {C_coupling_typical*1e15:.1f} fF (typical value)")
print("   - Adjust gap/overlap in coupling capacitor design")
print("")
print("4. Frequency Targets:")
print(f"   - Qubit frequency: {f01:.3f} GHz")
print(f"   - Resonator frequency: {resonator.E_osc:.1f} GHz")
print(f"   - Detuning: {abs(f01 - resonator.E_osc):.3f} GHz")
print("")
print("5. Tunability Range:")
print(f"   - Frequency range: {min(frequencies_01):.3f} - {max(frequencies_01):.3f} GHz")
print(f"   - Tuning range: {max(frequencies_01) - min(frequencies_01):.3f} GHz")

# %% [markdown]
# ## Connection to PDK Components
#
# These calculated parameters directly inform the design of PDK components:
#
# 1. **Double Pad Transmon**: The total capacitance determines pad dimensions
# 2. **Junction Components**: EJmax determines junction area and critical current
# 3. **Coupling Capacitors**: Coupling strength determines gap size and finger count
# 4. **Resonator Design**: Frequency sets the resonator length and impedance
#
# The scqubits calculations provide the quantum mechanical foundation for
# translating circuit parameters into physical geometries in the PDK.

# %%
if __name__ == "__main__":
    print("scqubits parameter calculation example completed successfully!")
    print("See the calculated parameters above for PDK design guidance.")
