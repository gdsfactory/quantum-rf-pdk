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
# # Dispersive Shift of a Transmon–Resonator System with Pymablock
#
# This notebook demonstrates how to use
# [pymablock](https://pymablock.readthedocs.io/) {cite:p}`arayaDayPymablockAlgorithmPackage2025`
# to compute the **dispersive shift** of a readout resonator coupled
# to a transmon qubit, and how to translate the resulting Hamiltonian
# parameters into physical layout parameters using **qpdk**.
#
# ## Background
#
# In circuit quantum electrodynamics (cQED), the state of a transmon qubit is
# typically measured via the *dispersive readout* technique
# {cite:p}`blaisCircuitQuantumElectrodynamics2021`.  The qubit is detuned from
# a readout resonator, and the qubit-state-dependent frequency shift of the
# resonator—the **dispersive shift** $\chi$—allows non-destructive measurement
# of the qubit state.
#
# The full transmon–resonator Hamiltonian (without the rotating-wave approximation)
# reads:
# ```{math}
# :label: eq:transmon-resonator-hamiltonian
# \mathcal{H} = -\omega_t\, a_t^\dagger a_t
# + \frac{\alpha}{2}\, a_t^{\dagger 2} a_t^{2}
# + \omega_r\, a_r^\dagger a_r
# - g\,(a_t^\dagger - a_t)(a_r^\dagger - a_r),
# ```
# where $\omega_t$ is the transmon frequency, $\alpha$ its anharmonicity,
# $\omega_r$ the resonator frequency, and $g$ the coupling strength.
#
# Treating $g$ as a perturbative parameter, **pymablock** computes the
# effective (block-diagonal) Hamiltonian to any desired order, from which
# we extract $\chi$.

# %% tags=["hide-input", "hide-output"]
import numpy as np
import scipy
import skrf
import sympy
from IPython.display import Math, display
from matplotlib import pyplot as plt

from qpdk.models.media import cpw_media_skrf
from qpdk.models.perturbation import (
    dispersive_shift,
    dispersive_shift_symbolic,
    dispersive_shift_to_coupling,
    ej_ec_to_frequency_and_anharmonicity,
    measurement_induced_dephasing,
    purcell_decay_rate,
    qubit_frequency_from_ej_ec,
    resonator_linewidth_from_q,
    transmon_resonator_hamiltonian,
)
from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    ec_to_capacitance,
    ej_to_inductance,
)
from qpdk.models.resonator import resonator_frequency


# %% [markdown]
# ## Building the Hamiltonian
#
# We first construct the Hamiltonian symbolically using bosonic operators from
# SymPy, following the conventions of pymablock.  The helper
# `transmon_resonator_hamiltonian` returns the unperturbed part $H_0$ and the
# perturbation $H_p$ together with the symbolic parameters.

# %%
H_0, H_p, (omega_t, omega_r, alpha, g) = transmon_resonator_hamiltonian()


def display_eq(title: str, expr: sympy.Basic) -> None:
    """Display a named symbolic expression."""
    display(sympy.Eq(sympy.Symbol(title), expr, evaluate=False))


display_eq("H_{0}", H_0)
display_eq("H_{p}", H_p)

# %% [markdown]
# ## Approach I — Second-Quantized Form
#
# Using `block_diagonalize` on the full second-quantized Hamiltonian,
# pymablock returns the effective Hamiltonian as a function of the
# number operators $N_{a_t}$ and $N_{a_r}$.  The dispersive shift is:
# ```{math}
# :label: eq:dispersive-shift-definition
# \chi = E^{(2)}_{11} - E^{(2)}_{10} - E^{(2)}_{01} + E^{(2)}_{00}
# ```
# where $E^{(2)}_{ij}$ is the second-order energy correction for
# $i$ transmon excitations and $j$ resonator excitations.

# %%
from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator
from sympy.physics.quantum.boson import BosonOp

H_tilde, U, U_adj = block_diagonalize(H_0 + H_p, symbols=[g])

E_eff = H_tilde[0, 0, 2]
display_eq("E_{eff}^{(2)}", E_eff)

# %%
# Evaluate for specific occupation numbers
a_t_op = BosonOp("a_t")
a_r_op = BosonOp("a_r")
N_a_t = NumberOperator(a_t_op)
N_a_r = NumberOperator(a_r_op)

E_00 = E_eff.subs({N_a_t: 0, N_a_r: 0})
E_01 = E_eff.subs({N_a_t: 0, N_a_r: 1})
E_10 = E_eff.subs({N_a_t: 1, N_a_r: 0})
E_11 = E_eff.subs({N_a_t: 1, N_a_r: 1})

chi_sym_sq = E_11 - E_10 - E_01 + E_00

# Convert from pymablock's NumberOrderedForm to a standard sympy expression
if hasattr(chi_sym_sq, "as_expr"):
    chi_sym_sq = chi_sym_sq.as_expr()

display_eq(r"\chi^{(SQ)}", chi_sym_sq)

# %% [markdown]
# ## Approach II — Matrix Representation
#
# Alternatively we can truncate the bosonic Hilbert space and use a
# finite-dimensional matrix.  Since the perturbation changes occupation
# numbers by $\pm 1$, computing second-order corrections to states
# with $n_t, n_r \in \{0, 1\}$ requires keeping up to 3 levels per mode
# (we use 4 for safety).

# %%
N = 4  # Number of levels per mode

# Build matrix representations of the bosonic operators
a_mat = sympy.zeros(N, N)
for i in range(N - 1):
    a_mat[i, i + 1] = sympy.sqrt(i + 1)

a_t_mat = sympy.KroneckerProduct(a_mat, sympy.eye(N))
a_r_mat = sympy.KroneckerProduct(sympy.eye(N), a_mat)

H_0_mat = (
    -omega_t * a_t_mat.adjoint() * a_t_mat
    + omega_r * a_r_mat.adjoint() * a_r_mat
    + alpha * a_t_mat.adjoint() ** 2 * a_t_mat**2 / 2
)
H_p_mat = -g * (a_t_mat.adjoint() - a_t_mat) * (
    a_r_mat.adjoint() - a_r_mat
)
H_mat = H_0_mat + H_p_mat

# Define subspaces: states |00⟩, |01⟩, |10⟩, |11⟩ each get their own block
subspaces = {state: n for n, state in enumerate([0, 1, N, N + 1])}
subspace_indices = [subspaces.get(state, 4) for state in range(N**2)]

H_tilde_mat, _U_mat, _U_adj_mat = block_diagonalize(
    H_mat, subspace_indices=subspace_indices, symbols=[g]
)

chi_sym_mat = (
    H_tilde_mat[3, 3, 2] - H_tilde_mat[2, 2, 2]
    - H_tilde_mat[1, 1, 2] + H_tilde_mat[0, 0, 2]
)[0, 0]

display_eq(r"\chi^{(mat)}", chi_sym_mat)

# %% [markdown]
# Both approaches yield the same dispersive shift—confirming the
# consistency of the second-quantized and matrix formulations.
# The full expression (including counter-rotating terms) is:
# ```{math}
# :label: eq:dispersive-shift-full
# \chi = \frac{2g^2}{-\alpha + \omega_r + \omega_t}
#      + \frac{2g^2}{-\alpha - \omega_r + \omega_t}
#      - \frac{2g^2}{-\omega_r + \omega_t}
#      + \frac{2g^2}{-\omega_r - \omega_t}
# ```

# %% [markdown]
# ## Numerical Evaluation
#
# We now evaluate $\chi$ for realistic transmon parameters.
# We start from the Josephson energy $E_J$ and charging energy $E_C$,
# which fully determine the transmon frequency and anharmonicity in the
# transmon regime ($E_J \gg E_C$):
# ```{math}
# \omega_t \approx \sqrt{8 E_J E_C} - E_C, \qquad \alpha \approx E_C
# ```

# %%
# Hamiltonian parameters (in GHz)
EJ = 20.0  # Josephson energy
EC = 0.2  # Charging energy
omega_r_val = 7.0  # Resonator frequency
g_val = 0.1  # Coupling strength

# Derived transmon parameters
omega_t_val, alpha_val = ej_ec_to_frequency_and_anharmonicity(EJ, EC)

display(
    Math(rf"""
\textbf{{Hamiltonian Parameters:}} \\
E_J = {EJ:.1f}\,\mathrm{{GHz}}, \quad
E_C = {EC:.1f}\,\mathrm{{GHz}} \\
\omega_t = \sqrt{{8 E_J E_C}} - E_C = {omega_t_val:.3f}\,\mathrm{{GHz}} \\
\alpha \approx E_C = {alpha_val:.1f}\,\mathrm{{GHz}} \\
\omega_r = {omega_r_val:.1f}\,\mathrm{{GHz}} \\
g = {g_val:.1f}\,\mathrm{{GHz}} \\
\Delta = \omega_t - \omega_r = {omega_t_val - omega_r_val:.3f}\,\mathrm{{GHz}}
""")
)

# Compute dispersive shift using the analytical formula from perturbation.py
chi_numerical = dispersive_shift(omega_t_val, omega_r_val, alpha_val, g_val)

display(
    Math(
        rf"\chi = {chi_numerical * 1e3:.3f}\,\mathrm{{MHz}} "
        rf"= {chi_numerical * 1e6:.1f}\,\mathrm{{kHz}}"
    )
)

# Verify against the symbolic expression
chi_check = float(
    chi_sym_sq.subs(
        {omega_t: omega_t_val, omega_r: omega_r_val, alpha: alpha_val, g: g_val}
    )
)
print(
    f"Symbolic evaluation: χ = {chi_check * 1e3:.3f} MHz "
    f"(difference: {abs(chi_numerical - chi_check):.2e} GHz)"
)

# %% [markdown]
# ## Dispersive Shift vs. Coupling Strength
#
# We now sweep the coupling strength $g$ to see how the dispersive shift
# depends on it.  The quadratic dependence $\chi \propto g^2$ is clearly
# visible.

# %%
g_sweep = np.linspace(0.01, 0.3, 200)
chi_sweep = np.array(
    [dispersive_shift(omega_t_val, omega_r_val, alpha_val, gi) for gi in g_sweep]
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(g_sweep * 1e3, chi_sweep * 1e3, "b-", linewidth=2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Coupling strength $g$ (MHz)")
ax.set_ylabel(r"Dispersive shift $\chi$ (MHz)")
ax.set_title(
    rf"Dispersive shift vs. coupling ($\omega_t={omega_t_val:.2f}$ GHz, "
    rf"$\omega_r={omega_r_val:.1f}$ GHz, $\alpha={alpha_val:.1f}$ GHz)"
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Design Workflow: From Hamiltonian to Layout Parameters
#
# The purpose of computing $\chi$ analytically is to **design** a qubit–resonator
# system with a target dispersive shift.  The workflow is:
#
# 1. **Choose target $\chi$** based on readout speed requirements
# 2. **Select $\omega_t$, $\omega_r$, $\alpha$** from qubit design goals
# 3. **Determine $g$** from target $\chi$ using {eq}`eq:dispersive-shift-full`
# 4. **Convert to circuit parameters** ($C_\Sigma$, $L_J$, $C_c$) using qpdk helpers
# 5. **Convert to layout parameters** (resonator length, capacitor geometry)
#
# ### Step 1–3: From target $\chi$ to coupling $g$

# %%
# Design targets
chi_target_mhz = -1.0  # Target dispersive shift in MHz
chi_target = chi_target_mhz * 1e-3  # Convert to GHz

# Qubit and resonator parameters
EJ_design = 20.0  # GHz
EC_design = 0.2  # GHz
omega_r_design = 7.0  # GHz

omega_t_design, alpha_design = ej_ec_to_frequency_and_anharmonicity(
    EJ_design, EC_design
)

# Compute required coupling strength
g_design = dispersive_shift_to_coupling(
    chi_target, omega_t_design, omega_r_design, alpha_design
)

# Verify: compute χ with the found g
chi_verify = dispersive_shift(omega_t_design, omega_r_design, alpha_design, g_design)

display(
    Math(rf"""
\textbf{{Step 1–3: Hamiltonian Design}} \\
\text{{Target:}}\quad \chi = {chi_target_mhz:.1f}\,\mathrm{{MHz}} \\
\text{{Required coupling:}}\quad g = {g_design * 1e3:.1f}\,\mathrm{{MHz}} \\
\text{{Verification:}}\quad \chi(g) = {chi_verify * 1e3:.3f}\,\mathrm{{MHz}}
""")
)

# %% [markdown]
# ### Step 4: Convert to circuit parameters
#
# Using the qpdk helper functions, we convert the Hamiltonian parameters
# to circuit parameters:
# - Total qubit capacitance $C_\Sigma$ from $E_C$ via `ec_to_capacitance`
# - Josephson inductance $L_J$ from $E_J$ via `ej_to_inductance`
# - Coupling capacitance $C_c$ from $g$ via `coupling_strength_to_capacitance`

# %%
# Total qubit capacitance
C_sigma = float(ec_to_capacitance(EC_design))

# Josephson inductance
L_J = float(ej_to_inductance(EJ_design))

# For coupling capacitance, we need the resonator capacitance.
# First, determine resonator length for the target frequency.
resonator_media = cpw_media_skrf(width=10, gap=6)(
    frequency=skrf.Frequency.from_f([omega_r_design], unit="GHz")
)


def _resonator_objective(length: float) -> float:
    """Minimise the squared frequency error."""
    freq = resonator_frequency(
        length=length, media=resonator_media, is_quarter_wave=True
    )
    return (freq - omega_r_design * 1e9) ** 2


result = scipy.optimize.minimize(
    _resonator_objective, 4000.0, bounds=[(1000, 20000)]
)
resonator_length = result.x[0]

# Total resonator capacitance from CPW impedance and phase velocity
C_r = (
    1 / np.real(resonator_media.z0 * resonator_media.v_p).mean()
    * resonator_length
    * 1e-6
)  # F

# Coupling capacitance
C_c = float(
    coupling_strength_to_capacitance(
        g_ghz=g_design,
        c_sigma=C_sigma,
        c_r=C_r,
        f_q_ghz=omega_t_design,
        f_r_ghz=omega_r_design,
    )
)

display(
    Math(rf"""
\textbf{{Step 4: Circuit Parameters}} \\
C_\Sigma = {C_sigma * 1e15:.1f}\,\mathrm{{fF}} \\
L_J = {L_J * 1e9:.2f}\,\mathrm{{nH}} \\
C_r = {C_r * 1e15:.1f}\,\mathrm{{fF}} \\
C_c = {C_c * 1e15:.2f}\,\mathrm{{fF}}
""")
)

# %% [markdown]
# ### Step 5: Layout parameters
#
# The circuit parameters map directly to layout dimensions:

# %%
f_resonator_achieved = resonator_frequency(
    length=resonator_length, media=resonator_media, is_quarter_wave=True
)

display(
    Math(rf"""
\textbf{{Step 5: Layout Parameters}} \\
\text{{Resonator length:}}\quad l = {resonator_length:.0f}\,\mathrm{{\mu m}} \\
\text{{Resonator CPW width:}}\quad w = 10\,\mathrm{{\mu m}}, \quad
\text{{gap}} = 6\,\mathrm{{\mu m}} \\
\text{{Resonator frequency achieved:}}\quad
f_r = {f_resonator_achieved / 1e9:.3f}\,\mathrm{{GHz}}
""")
)

# %% [markdown]
# ## Readout System Design Considerations
#
# A complete readout design also requires considering the resonator
# linewidth $\kappa$, the Purcell decay rate, and the measurement-induced
# dephasing.

# %%
# Resonator quality factor and linewidth
Q_ext = 10_000  # External quality factor
kappa = resonator_linewidth_from_q(omega_r_design, Q_ext)

# Purcell decay rate
gamma_purcell = purcell_decay_rate(
    g_design, omega_t_design, omega_r_design, kappa
)
T_purcell = 1 / (gamma_purcell * 1e9) if gamma_purcell > 0 else float("inf")

# Measurement-induced dephasing
n_bar = 5.0  # Mean photon number during readout
gamma_phi = measurement_induced_dephasing(chi_verify, kappa, n_bar)

# SNR estimate: the signal-to-noise ratio scales as χ/κ
chi_over_kappa = abs(chi_verify) / kappa

display(
    Math(rf"""
\textbf{{Readout System Parameters:}} \\
Q_\text{{ext}} = {Q_ext:,} \\
\kappa = \omega_r / Q_\text{{ext}} = {kappa * 1e6:.1f}\,\mathrm{{kHz}} \\
\gamma_\text{{Purcell}} = \kappa (g/\Delta)^2
  = {gamma_purcell * 1e6:.2f}\,\mathrm{{kHz}}
  \quad \Rightarrow \quad T_\text{{Purcell}}
  = {T_purcell * 1e6:.0f}\,\mathrm{{\mu s}} \\
\Gamma_\phi(\bar{{n}}={n_bar:.0f})
  = 8\chi^2 \bar{{n}} / \kappa
  = {gamma_phi * 1e6:.2f}\,\mathrm{{kHz}} \\
|\chi| / \kappa = {chi_over_kappa:.1f}
  \quad (\text{{want}} \gtrsim 1 \text{{ for high-fidelity readout}})
""")
)

# %% [markdown]
# ## Summary: Complete Design Table
#
# The table below summarises the full design flow from Hamiltonian
# parameters to layout parameters, with the dispersive shift as the
# central design target.

# %%
print("=" * 65)
print("  DESIGN PARAMETER SUMMARY")
print("=" * 65)
print(f"  Target dispersive shift χ:      {chi_target_mhz:.1f} MHz")
print("-" * 65)
print("  Hamiltonian Parameters:")
print(f"    E_J  = {EJ_design:.1f} GHz")
print(f"    E_C  = {EC_design:.1f} GHz")
print(f"    ω_t  = {omega_t_design:.3f} GHz")
print(f"    α    = {alpha_design:.1f} GHz")
print(f"    ω_r  = {omega_r_design:.1f} GHz")
print(f"    g    = {g_design * 1e3:.1f} MHz")
print("-" * 65)
print("  Circuit Parameters:")
print(f"    C_Σ  = {C_sigma * 1e15:.1f} fF")
print(f"    L_J  = {L_J * 1e9:.2f} nH")
print(f"    C_r  = {C_r * 1e15:.1f} fF")
print(f"    C_c  = {C_c * 1e15:.2f} fF")
print("-" * 65)
print("  Layout Parameters:")
print(f"    Resonator length:    {resonator_length:.0f} µm")
print(f"    Resonator CPW:       w=10 µm, gap=6 µm")
print(f"    Resonator freq:      {f_resonator_achieved / 1e9:.3f} GHz")
print("-" * 65)
print("  Readout Parameters:")
print(f"    Q_ext:               {Q_ext:,}")
print(f"    κ:                   {kappa * 1e6:.1f} kHz")
print(f"    T_Purcell:           {T_purcell * 1e6:.0f} µs")
print(f"    |χ|/κ:               {chi_over_kappa:.1f}")
print("=" * 65)


# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
