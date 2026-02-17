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
# This example demonstrates how to use [scqubits](https://scqubits.readthedocs.io/en/latest/) {cite:p}`groszkowskiScqubitsPythonPackage2021` to calculate
# parameters for superconducting qubits and coupled resonators. The calculations help determine
# physical parameters like capacitances and coupling strengths needed for component design.

# %% tags=["hide-input", "hide-output"]
import numpy as np

# Monkeypatch for NumPy 2.0 compatibility with scqubits
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "float_"):
    np.float_ = np.float64

import scipy
import scqubits as scq
import skrf
import sympy as sp
from IPython.display import Math, display
from matplotlib import pyplot as plt

from qpdk.models.media import cpw_media_skrf
from qpdk.models.resonator import resonator_frequency

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

# %% [markdown]
# ## Parameter Sweep: Flux Dependence
#
# Demonstrate how the qubit frequency changes with external flux.
#
# For more details, see the [scqubits-examples repository](https://github.com/scqubits/scqubits-examples)

# %%
# Sweep flux and calculate qubit frequency
transmon.plot_evals_vs_paramvals(
    "flux", np.linspace(-1.1, 1.1, 201), subtract_ground=True
)
plt.show()
transmon.plot_wavefunction(esys=None, which=(0, 1, 2, 3, 8), mode="real")
plt.show()

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
# ```{align}
# :label: eq:coupling-capacitance
# g &\approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_{\Sigma} \left( C_\text{r} - \frac{C_\text{c}^2}{C_\text{q}} \right) }}  \sqrt{\omega_\text{q}\omega_\text{r}}
# &\approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_{\Sigma} C_\text{r}}}  \sqrt{\omega_\text{q}\omega_\text{r}}, \quad \text{for } C_\text{c} \ll C_\text{q}
# ```
# where $C_\Sigma$ is the total qubit capacitance, $C_{\text{q}}$ is the capacitance between the qubit pads, and $C_r$ is the total capacitance of the resonator.

# %%
# Calculate physical capacitance
# The charging energy EC = e²/(2C_total), so C_Σ = e²/(2*EC)
EC_joules = transmon.EC * 1e9 * scipy.constants.h  # Convert GHz to Joules

# Total capacitance of the transmon
C_Σ = scipy.constants.e**2 / (2 * EC_joules)

# Josephson inductance - use typical realistic value
# For EJ ~ 40 GHz, typical LJ ~ 0.8-1.0 nH
# LJ scales inversely with EJ: LJ ~ 1.0 nH * (25 GHz / EJ)
LJ_typical = 1.0e-9 * (25.0 / transmon.EJmax)  # Typical scaling

# Compute coupling capacitance from coupling strength
C_c_sym, C_Σ_sym, C_r_sym, omega_q_sym, omega_r_sym, g_sym = sp.symbols(
    "C_c C_Σ C_r omega_q omega_r g", real=True, positive=True
)
equation = sp.Eq(
    g_sym,
    0.5 * (C_c_sym / sp.sqrt(C_Σ_sym * C_r_sym)) * sp.sqrt(omega_q_sym * omega_r_sym),
)
solution = sp.solve(equation, C_c_sym)
C_c_sol = next(sol for sol in solution if sol.is_real and sol > 0)
display(Math(f"C_\\text{{c}} = {sp.latex(C_c_sol)}"))


# Use a typical value for resonator capacitance to ground
resonator_media = cpw_media_skrf(width=10, gap=6)(
    frequency=skrf.Frequency.from_f([5], unit="GHz")
)


def _objective(length: float) -> float:
    """Find resonator length for target frequency using SciPy."""
    freq = resonator_frequency(
        length=length, media=resonator_media, is_quarter_wave=True
    )
    return (freq - resonator.E_osc * 1e9) ** 2  # MSE


length_initial = 4000.0  # Initial guess in µm
result = scipy.optimize.minimize(_objective, length_initial, bounds=[(1000, 20000)])
length = result.x[0]
print(
    f"Optimization success: {result.success}, message: {result.message}, nfev: {result.nfev}"
)
display(
    Math(
        f"\\textrm{{Resonator length at width 10 µm and gap 6 µm}}\\,{resonator.E_osc:.1f}\\,\\mathrm{{GHz}}: {length:.1f}\\,\\mathrm{{µm}}"
    )
)

# %% [markdown]
#
# The total capacitance of the resonator can be estimated from its characteristic impedance
# $Z_0$ and phase velocity $v_p$ as {cite:p}`gopplCoplanarWaveguideResonators2008a`:
# ```{math}
# :label: eq:resonator-capacitance
# C_r = \frac{l}{\mathrm{Re}(Z_0 v_p)}
# ```
# where $l$ is the resonator length.

# %%
# Get total capacitance to ground of the resonator (in isolation, we disregard effect of coupling to qubits)
C_r = 1 / np.real(resonator_media.z0 * resonator_media.v_p).mean() * length * 1e-6  # F

# Substitute and evaluate numerically
C_c_num = C_c_sol.subs(
    {
        g_sym: g_coupling,
        C_Σ_sym: C_Σ,
        C_r_sym: C_r,
        omega_q_sym: f01,
        omega_r_sym: resonator.E_osc,
    }
).evalf()

display(
    Math(rf"""
\textbf{{Physical Parameters for PDK Design:}}\\
\text{{Total qubit capacitance:}}~{C_Σ * 1e15:.1f}~\mathrm{{fF}}\\
\text{{Josephson inductance:}}~{LJ_typical * 1e9:.2f}~\mathrm{{nH}}\\
\text{{Estimated target qubit–resonator coupling capacitance:}}~{C_c_num * 1e15:.1f}~\mathrm{{fF}}
""")
)


# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
