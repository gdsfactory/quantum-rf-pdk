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
# parameters for superconducting qubits and coupled resonators. The calculations help determine
# physical parameters like capacitances and coupling strengths needed for component design.


# %%
import numpy as np
import pandas as pd
import scipy
import scqubits as scq
from IPython.display import Math, display

# %% [markdown]
# ## TunableTransmon Qubit Parameters
#
# A tunable transmon qubit {cite:p}`kochChargeinsensitiveQubitDesign2007a` has a SQUID junction that allows tuning the effective Josephson energy
# through an external flux. Key parameters include:
# - $E_{\text{J}_\text{max}}$: Maximum Josephson energy when flux = 0
# - $E_\text{C}$: Charging energy, related to total capacitance as $E_\text{C} = \dfrac{e^2}{2C_\Sigma}$
# - $d$: Asymmetry parameter between the two junctions in the SQUID
# - $\phi$: External flux in units of flux quantum $\phi_0$
# - $\alpha$: Anharmonicity, the difference between the $E_{1→2}$ and $E_{0→1}$ transition frequencies
#
# See the [scqubits documentation](https://scqubits.readthedocs.io/en/latest/guide/qubits/tunable_transmon.html) for more details.

# %%
# Create a tunable transmon with realistic parameters
transmon = scq.TunableTransmon(
    EJmax=40.0,  # Maximum Josephson energy in GHz (typical range: 20-50 GHz)
    EC=0.2,  # Charging energy in GHz (typical range: 0.1-0.5 GHz)
    d=0.1,  # Junction asymmetry (typical range: 0.05-0.2)
    flux=0.0,  # Start at flux = 0 (maximum EJ)
    ng=0.0,  # Offset charge (usually 0 or 0.5)
    ncut=30,  # Charge basis cutoff
)

# Tunable Transmon Parameters
display(
    Math(rf"""
\textbf{{Tunable Transmon Parameters:}} \\
E_{{J,\text{{max}}}} = {transmon.EJmax:.1f}\ \text{{GHz}} \\
E_C = {transmon.EC:.1f}\ \text{{GHz}} \\
d = {transmon.d:.2f} \\
\text{{Flux}} = {transmon.flux:.2f}\ \Phi_0
""")
)

# Calculate energy levels
eigenvals = transmon.eigenvals(evals_count=5)
f01 = eigenvals[1] - eigenvals[0]
f12 = eigenvals[2] - eigenvals[1]
anharmonicity = f12 - f01

display(
    Math(rf"""
\textbf{{Transmon Spectrum:}} \\
0\rightarrow 1\ \text{{frequency}}:\ \ {f01:.3f}\ \text{{GHz}} \\
1\rightarrow 2\ \text{{frequency}}:\ \ {f12:.3f}\ \text{{GHz}} \\
\text{{Anharmonicity, }} \alpha:\ \ {anharmonicity:.3f}\ \text{{GHz}}
""")
)

# Resonator (Oscillator) Parameters
resonator = scq.Oscillator(
    E_osc=6.5,  # Resonator frequency in GHz (typical range: 4–8 GHz)
)

display(
    Math(rf"""
\textbf{{Resonator Parameters:}} \\
\text{{Frequency}}:\ \ {resonator.E_osc:.1f}\ \text{{GHz}} \\
""")
)


# %% [markdown]
# ## Coupled Qubit-Resonator System
#
# When a transmon is coupled to a resonator, the interaction can be described by
# the Jaynes-Cummings model {cite:p}`shoreJaynesCummingsModel1993` :
# ```{math}
# :label: eq:jaynes-cummings
# \mathcal{H} = \omega_r a^{\dagger} a + \frac{\omega_q}{2} \sigma_z + g (a^{\dagger} \sigma^- + a \sigma^+),
# ```
# where $\omega_r$ is the resonator frequency, $\omega_q$ is the qubit frequency, and $g$ is the coupling strength.
# The operators $a^{\dagger}$ and $a$ are the creation and annihilation operators for the resonator,
# while $\sigma_z$, $\sigma^-$, and $\sigma^+$ are the Pauli and ladder operators for the qubit.
#
# The coupling strength $g$ depends on the overlap between
# the qubit and resonator modes and affects the system dynamics.

# %%
# Create a coupled system using HilbertSpace
hilbert_space = scq.HilbertSpace([transmon, resonator])

# Add interaction between qubit and resonator
# The interaction is typically g(n_qubit * (a + a†)) where n is the charge operator and a is the resonator annihilation operator
g_coupling = 0.15  # Coupling strength in GHz (typical range: 0.05–0.3 GHz)
interaction = scq.InteractionTerm(
    g_strength=g_coupling,
    operator_list=[
        (0, transmon.n_operator),  # (subsystem_index, operator)
        (1, resonator.annihilation_operator() + resonator.creation_operator()),
    ],
)
hilbert_space.interaction_list = [interaction]


display(Math(f"g_{{\\text{{Qubit–Resonator}}}} = {g_coupling:.3f}\\,\\text{{GHz}}"))

# Calculate the coupled system eigenvalues
coupled_eigenvals = hilbert_space.eigenvals(evals_count=10)
print(f"  Coupled system ground state: {coupled_eigenvals[0]:.3f} GHz")

# %% [markdown]
# ## Physical (Lumped-element) Parameter Extraction
#
# From the quantum model, we can extract physical parameters relevant for PDK design.
#
# The coupling strength $g$ (in the dispersive limit $g\ll \omega_q, \omega_r$) can be related to a coupling capacitance $C_c$ via {cite:p}`Savola2023`:
# ```{math}
# :label: eq:coupling-capacitance
# g \approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_{\Sigma} \left( C_\text{r} - \frac{C_\text{qr}^2}{C_\text{q}} \right) }}  \sqrt{\omega_\text{q}\omega_\text{r}}
# ```
# where $C_\Sigma$ is the total qubit capacitance, and $C_r$ is the resonator capacitance.

# %%
# Calculate physical capacitance
# The charging energy EC = e²/(2C_total), so C_total = e²/(2*EC)
e_charge = scipy.constants.e  # Elementary charge
h_planck = scipy.constants.h  # Planck's constant
EC_joules = transmon.EC * 1e9 * h_planck  # Convert GHz to Joules

# Total capacitance of the transmon
C_Σ = e_charge**2 / (2 * EC_joules)

# Josephson inductance - use typical realistic value
# For EJ ~ 40 GHz, typical LJ ~ 0.8-1.0 nH
# LJ scales inversely with EJ: LJ ~ 1.0 nH * (25 GHz / EJ)
LJ_typical = 1.0e-9 * (25.0 / transmon.EJmax)  # Typical scaling

# Compute coupling capacitance from coupling strength
C_coupling_typical = 5.0e-15  # TODO NOT ONLY ESTIMATE
# C_coupling_typical = (g_coupling / f01) * C_total  # Rough estimate


display(
    Math(rf"""
\textbf{{Physical Parameters for PDK Design:}}\\
\text{{Total qubit capacitance:}}~{C_Σ * 1e15:.1f}~\mathrm{{fF}}\\
\text{{Josephson inductance:}}~{LJ_typical * 1e9:.2f}~\mathrm{{nH}}\\
\text{{Estimated coupling capacitance:}}~{C_coupling_typical * 1e15:.1f}~\mathrm{{fF}}
""")
)

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
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.plot(flux_values, frequencies_01, "b-", linewidth=2, label="01 transition")
# plt.plot(flux_values, frequencies_12, "r--", linewidth=2, label="12 transition")
# plt.xlabel("External Flux (Φ₀)")
# plt.ylabel("Transition Frequency (GHz)")
# plt.title("Tunable Transmon Spectrum vs Flux")
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(1, 2, 2)
# anharmonicity_vs_flux = np.array(frequencies_12) - np.array(frequencies_01)
# plt.plot(flux_values, anharmonicity_vs_flux, "g-", linewidth=2)
# plt.xlabel("External Flux (Φ₀)")
# plt.ylabel("Anharmonicity (GHz)")
# plt.title("Anharmonicity vs Flux")
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## Design Recommendations
#
# Based on the calculations above, here are parameter recommendations for PDK components:

# %%

pdk_df = pd.DataFrame(
    [
        {
            "Section": "Transmon Pad Capacitance",
            "Details": f"Target total capacitance: {C_Σ * 1e15:.1f} fF\nDetermines pad size and geometry",
        },
        {
            "Section": "Junction Parameters",
            "Details": f"Target EJmax: {transmon.EJmax:.1f} GHz\nJunction area scales with EJ\nAsymmetry parameter d = {transmon.d:.2f}",
        },
        {
            "Section": "Coupling Elements",
            "Details": f"Target coupling strength: {g_coupling:.3f} GHz\nEstimated coupling capacitance: {C_coupling_typical * 1e15:.1f} fF (typical value)\nAdjust gap/overlap in coupling capacitor design",
        },
        {
            "Section": "Frequency Targets",
            "Details": f"Qubit frequency: {f01:.3f} GHz\nResonator frequency: {resonator.E_osc:.1f} GHz\nDetuning: {abs(f01 - resonator.E_osc):.3f} GHz",
        },
        {
            "Section": "Tunability Range",
            "Details": f"Frequency range: {min(frequencies_01):.3f} - {max(frequencies_01):.3f} GHz\nTuning range: {max(frequencies_01) - min(frequencies_01):.3f} GHz",
        },
    ]
)
display(pdk_df.style.hide(axis="index"))

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

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
