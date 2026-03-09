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
# # Dispersive Shift of a Transmon–Resonator System with scQubits
#
# This notebook demonstrates how to use
# [scqubits](https://scqubits.readthedocs.io/en/latest/) {cite:p}`groszkowskiScqubitsPythonPackage2021`
# to numerically compute the **dispersive shift** of a readout resonator coupled
# to a transmon qubit, and how to translate the resulting Hamiltonian
# parameters into physical layout parameters using **qpdk**.
#
# This notebook covers the same design workflow as the companion notebook on
# **pymablock-based dispersive shift calculation**, but uses full numerical
# diagonalization instead of closed-form perturbation theory—making it easy to
# compare the two approaches.
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
# :label: eq:transmon-resonator-hamiltonian-scq
# \mathcal{H} = \omega_t\, a_t^\dagger a_t
# + \frac{\alpha}{2}\, a_t^{\dagger 2} a_t^{2}
# + \omega_r\, a_r^\dagger a_r
# + g\,(a_t + a_t^\dagger)(a_r + a_r^\dagger),
# ```
# where $\omega_t$ is the transmon frequency, $\alpha$ its anharmonicity
# ($\alpha < 0$ for a transmon),
# $\omega_r$ the resonator frequency, and $g$ the coupling strength.
#
# scQubits constructs and diagonalizes this Hamiltonian numerically in the
# transmon charge basis and Fock basis {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
# The dispersive shift is then extracted directly from the dressed eigenvalues:
# ```{math}
# :label: eq:dispersive-shift-definition-scq
# \chi = (E_{11} - E_{10}) - (E_{01} - E_{00}),
# ```
# where $E_{ij}$ is the dressed eigenenergy with $i$ transmon excitations and
# $j$ resonator photons.

# %% tags=["hide-input", "hide-output"]
import numpy as np

# Monkeypatch for NumPy 2.0 compatibility with scqubits
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # noqa: NPY201
if not hasattr(np, "float_"):
    np.float_ = np.float64  # noqa: NPY201

import polars as pl
import scipy
import scqubits as scq
import skrf
from IPython.display import Math, display
from matplotlib import pyplot as plt

from qpdk.models.media import cpw_media_skrf
from qpdk.models.perturbation import (
    dispersive_shift,
    dispersive_shift_to_coupling,
    ej_ec_to_frequency_and_anharmonicity,
    measurement_induced_dephasing,
    purcell_decay_rate,
    resonator_linewidth_from_q,
)
from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    ec_to_capacitance,
    ej_to_inductance,
)
from qpdk.models.resonator import resonator_frequency

# %% [markdown]
# ## Building the System
#
# We use the same Hamiltonian parameters as the companion pymablock notebook to
# allow direct numerical comparison.  A transmon qubit in the transmon regime
# ($E_J \gg E_C$) satisfies:
# ```{math}
# \omega_t \approx \sqrt{8 E_J E_C} - E_C, \qquad \alpha \approx -E_C.
# ```

# %% tags=["hide-input", "hide-output"]
# Hamiltonian parameters (matching pymablock notebook for comparison)
EJ = 20.0  # Josephson energy in GHz
EC = 0.2  # Charging energy in GHz
omega_r_val = 7.0  # Resonator frequency in GHz
g_val = 0.1  # Coupling strength in GHz

# %%
transmon = scq.Transmon(
    EJ=EJ,
    EC=EC,
    ng=0.0,  # Offset charge
    ncut=30,  # Charge basis cutoff
    truncated_dim=6,  # Keep 6 transmon levels
)

resonator = scq.Oscillator(
    E_osc=omega_r_val,
    truncated_dim=6,  # Keep 6 Fock levels
)

omega_t_val, alpha_val = ej_ec_to_frequency_and_anharmonicity(EJ, EC)

display(
    Math(rf"""
\textbf{{Hamiltonian Parameters:}} \\
E_J = {EJ:.1f}\,\mathrm{{GHz}}, \quad
E_C = {EC:.1f}\,\mathrm{{GHz}} \\
\omega_t = \sqrt{{8 E_J E_C}} - E_C = {omega_t_val:.3f}\,\mathrm{{GHz}} \\
\alpha \approx -E_C = {alpha_val:.1f}\,\mathrm{{GHz}} \\
\omega_r = {omega_r_val:.1f}\,\mathrm{{GHz}} \\
g = {g_val:.1f}\,\mathrm{{GHz}} \\
\Delta = \omega_t - \omega_r = {omega_t_val - omega_r_val:.3f}\,\mathrm{{GHz}}
""")
)

# %% [markdown]
# ## Transmon Spectrum
#
# scQubits numerically diagonalizes the transmon Hamiltonian in the charge
# basis.  The anharmonicity $\alpha$ is extracted from the first two transition
# frequencies and should satisfy $\alpha \approx -E_C$ in the transmon regime.

# %%
eigenvals_t = transmon.eigenvals(evals_count=5)
f01 = eigenvals_t[1] - eigenvals_t[0]
f12 = eigenvals_t[2] - eigenvals_t[1]
anharmonicity = f12 - f01

display(
    Math(rf"""
\textbf{{Transmon Spectrum (scQubits):}} \\
0\rightarrow 1\ \text{{frequency:}}\ {f01:.3f}\,\mathrm{{GHz}} \\
1\rightarrow 2\ \text{{frequency:}}\ {f12:.3f}\,\mathrm{{GHz}} \\
\text{{Anharmonicity,}}\ \alpha =\ {anharmonicity:.3f}\,\mathrm{{GHz}}
""")
)

# %% [markdown]
# ## Dispersive Shift via Numerical Diagonalization
#
# We build the coupled Hilbert space and extract the dispersive shift from
# the dressed eigenvalues using {eq}`eq:dispersive-shift-definition-scq`.
# In circuit QED the dominant coupling is between the transmon charge operator
# $n_t$ and the resonator electric field $(a_r + a_r^\dagger)$
# {cite:p}`blaisCircuitQuantumElectrodynamics2021`:
# ```{math}
# :label: eq:coupling-term-scq
# \mathcal{H}_\text{int} = g\, n_t\, (a_r + a_r^\dagger),
# ```
# where $n_t \propto (a_t - a_t^\dagger)/(2i)$ is the transmon charge operator.
# This charge-coupling form is equivalent to {eq}`eq:transmon-resonator-hamiltonian-scq`
# when working in the transmon energy eigenbasis.
#
# The dressed states are identified by their maximum overlap with the
# corresponding bare states $|i_t, j_r\rangle$.

# %%
hilbert_space = scq.HilbertSpace([transmon, resonator])

interaction = scq.InteractionTerm(
    g_strength=g_val,
    operator_list=[
        (0, transmon.n_operator),  # (subsystem_index, operator)
        (1, resonator.annihilation_operator() + resonator.creation_operator()),
    ],
)
hilbert_space.interaction_list = [interaction]

# Diagonalize the full Hamiltonian
evals, evecs = hilbert_space.eigensys(evals_count=12)
evecs = np.array(evecs)

dim_t = transmon.truncated_dim
dim_r = resonator.truncated_dim


def bare_state_vec(qt_idx: int, res_idx: int) -> np.ndarray:
    """Return the bare state |qt_idx, res_idx⟩ in the full Hilbert space."""
    vec = np.zeros(dim_t * dim_r)
    vec[qt_idx * dim_r + res_idx] = 1.0
    return vec


def find_dressed_index(bare_vec: np.ndarray) -> int:
    """Return the dressed-state index with maximum overlap with bare_vec."""
    return int(np.argmax(np.abs(evecs.T @ bare_vec) ** 2))


idx_00 = find_dressed_index(bare_state_vec(0, 0))
idx_10 = find_dressed_index(bare_state_vec(1, 0))
idx_01 = find_dressed_index(bare_state_vec(0, 1))
idx_11 = find_dressed_index(bare_state_vec(1, 1))

E_00, E_10, E_01, E_11 = evals[idx_00], evals[idx_10], evals[idx_01], evals[idx_11]
chi_scqubits = (E_11 - E_10) - (E_01 - E_00)

# Compare to the analytical formula from qpdk.models.perturbation
chi_analytical = dispersive_shift(omega_t_val, omega_r_val, alpha_val, g_val)

display(
    Math(rf"""
\chi_{{\mathrm{{scQubits}}}} = {chi_scqubits * 1e3:.3f}\,\mathrm{{MHz}} \\
\chi_{{\mathrm{{analytical}}}} = {chi_analytical * 1e3:.3f}\,\mathrm{{MHz}} \\
\text{{Relative error: }}
{abs(chi_scqubits - chi_analytical) / abs(chi_analytical) * 100:.1f}\,\%
""")
)

# %% [markdown]
# The scQubits result agrees well with the analytical perturbation-theory
# formula; any residual difference reflects higher-order corrections beyond
# the leading $g^2/\Delta$ term.

# %% [markdown]
# ## Dispersive Shift vs. Coupling Strength
#
# We sweep the coupling strength $g$ to compare the scQubits numerical result
# against the analytical formula.  The two approaches agree in the dispersive
# regime ($g \ll |\Delta|$) and diverge as the system approaches the
# resonance condition.

# %%
_a_plus_adag = resonator.annihilation_operator() + resonator.creation_operator()
g_sweep = np.linspace(0.01, 0.3, 200)
chi_sweep_analytical = np.array(
    [dispersive_shift(omega_t_val, omega_r_val, alpha_val, gi) for gi in g_sweep]
)


def chi_scqubits_at_g(g: float) -> float:
    """Numerically compute χ via scQubits at a given coupling g."""
    hs = scq.HilbertSpace([transmon, resonator])
    hs.interaction_list = [
        scq.InteractionTerm(
            g_strength=g,
            operator_list=[(0, transmon.n_operator), (1, _a_plus_adag)],
        )
    ]
    ev, evc = hs.eigensys(evals_count=12)
    evc = np.array(evc)

    def di(qi: int, ri: int) -> int:
        v = np.zeros(dim_t * dim_r)
        v[qi * dim_r + ri] = 1.0
        return int(np.argmax(np.abs(evc.T @ v) ** 2))

    i00, i10, i01, i11 = di(0, 0), di(1, 0), di(0, 1), di(1, 1)
    return float((ev[i11] - ev[i10]) - (ev[i01] - ev[i00]))


g_sweep_coarse = np.linspace(0.01, 0.3, 20)
chi_sweep_scqubits = np.array([chi_scqubits_at_g(gi) for gi in g_sweep_coarse])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    g_sweep * 1e3,
    chi_sweep_analytical * 1e3,
    "-",
    linewidth=2,
    label="Analytical (qpdk)",
)
ax.plot(
    g_sweep_coarse * 1e3,
    chi_sweep_scqubits * 1e3,
    "o",
    markersize=6,
    label="scQubits (numerical)",
)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Coupling strength $g$ (MHz)")
ax.set_ylabel(r"Dispersive shift $\chi$ (MHz)")
ax.set_title(
    rf"Dispersive shift vs. coupling ($\omega_t={omega_t_val:.2f}$ GHz, "
    rf"$\omega_r={omega_r_val:.1f}$ GHz)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Flux-Tunable Transmon
#
# A key advantage of scQubits over the analytical pymablock approach is its
# native support for **flux-tunable transmons** with a SQUID junction.
# The effective Josephson energy varies with external flux $\Phi$:
# ```{math}
# :label: eq:flux-tunable-ej
# E_J(\Phi) = E_{J,\mathrm{max}}
#   \left|\cos\!\left(\pi\frac{\Phi}{\Phi_0}\right)\right|
#   \sqrt{1 + d^2 \tan^2\!\left(\pi\frac{\Phi}{\Phi_0}\right)},
# ```
# where $d$ is the junction asymmetry parameter.  The qubit frequency and
# dispersive shift vary continuously with flux, which scQubits handles without
# any change to the perturbation-theory derivation.
#
# Key parameters of the tunable transmon:
# - $E_{J,\mathrm{max}}$: maximum Josephson energy at zero flux
# - $E_C$: charging energy (sets the anharmonicity)
# - $d$: SQUID junction asymmetry
#
# See the [scqubits documentation](https://scqubits.readthedocs.io/en/latest/guide/qubits/tunable_transmon.html) for more details.

# %%
tunable_transmon = scq.TunableTransmon(
    EJmax=40.0,  # Maximum Josephson energy in GHz
    EC=0.2,  # Charging energy in GHz
    d=0.1,  # Junction asymmetry
    flux=0.0,  # External flux in units of Φ₀
    ng=0.0,
    ncut=30,
)

tunable_transmon.plot_evals_vs_paramvals(
    "flux", np.linspace(-0.5, 0.5, 201), subtract_ground=True, evals_count=5
)
plt.show()

# %% [markdown]
# ## Design Workflow: From Hamiltonian to Layout Parameters
#
# The purpose of computing $\chi$ is to **design** a qubit–resonator system
# with a target dispersive shift.  The workflow mirrors the pymablock notebook,
# using qpdk analytical helpers for speed and scQubits for numerical verification:
#
# 1. **Choose target $\chi$** based on readout speed requirements
# 2. **Select $\omega_t$, $\omega_r$, $\alpha$** from qubit design goals
# 3. **Determine $g$** from target $\chi$
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

# Find g using the qpdk analytical inversion
g_design = dispersive_shift_to_coupling(
    chi_target, omega_t_design, omega_r_design, alpha_design
)

# Verify analytically and with scQubits
chi_verify_analytical = dispersive_shift(
    omega_t_design, omega_r_design, alpha_design, float(g_design)
)
chi_verify_scq = chi_scqubits_at_g(float(g_design))

display(
    Math(rf"""
\textbf{{Step 1–3: Hamiltonian Design}} \\
\text{{Target:}}\quad \chi = {chi_target_mhz:.1f}\,\mathrm{{MHz}} \\
\text{{Required coupling:}}\quad g = {float(g_design) * 1e3:.1f}\,\mathrm{{MHz}} \\
\text{{Verification (analytical):}}\quad \chi = {chi_verify_analytical * 1e3:.3f}\,\mathrm{{MHz}} \\
\text{{Verification (scQubits):}}\quad \chi = {chi_verify_scq * 1e3:.3f}\,\mathrm{{MHz}}
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

# Determine resonator length for the target frequency
resonator_media = cpw_media_skrf(width=10, gap=6)(
    frequency=skrf.Frequency.from_f([omega_r_design], unit="GHz")
)


def _resonator_objective(length: float) -> float:
    """Minimise the squared frequency error."""
    freq = resonator_frequency(
        length=length,
        epsilon_eff=float(np.real(np.mean(resonator_media.ep_r))),
        is_quarter_wave=True,
    )
    return (freq - omega_r_design * 1e9) ** 2


result = scipy.optimize.minimize(_resonator_objective, 4000.0, bounds=[(1000, 20000)])
resonator_length = result.x[0]

# Total resonator capacitance from CPW impedance and phase velocity
# {cite:p}`gopplCoplanarWaveguideResonators2008a`:
# C_r = l / Re(Z_0 * v_p)
C_r = (
    1
    / np.real(resonator_media.z0 * resonator_media.v_p).mean()
    * resonator_length
    * 1e-6
)  # F

# Coupling capacitance
C_c = float(
    coupling_strength_to_capacitance(
        g_ghz=float(g_design),
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
    length=resonator_length,
    epsilon_eff=float(np.real(np.mean(resonator_media.ep_r))),
    is_quarter_wave=True,
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
    float(g_design), omega_t_design, omega_r_design, kappa
)
T_purcell = 1 / (gamma_purcell * 1e9) if gamma_purcell > 0 else float("inf")

# Measurement-induced dephasing
n_bar = 5.0  # Mean photon number during readout
gamma_phi = measurement_induced_dephasing(chi_verify_analytical, kappa, n_bar)

# SNR estimate: the signal-to-noise ratio scales as χ/κ
chi_over_kappa = abs(chi_verify_analytical) / kappa

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
data = [
    ("Target", "Target dispersive shift χ", f"{chi_target_mhz:.1f}", "MHz"),
    ("Hamiltonian", "E_J", f"{EJ_design:.1f}", "GHz"),
    ("Hamiltonian", "E_C", f"{EC_design:.1f}", "GHz"),
    ("Hamiltonian", "ω_t", f"{omega_t_design:.3f}", "GHz"),
    ("Hamiltonian", "α", f"{alpha_design:.1f}", "GHz"),
    ("Hamiltonian", "ω_r", f"{omega_r_design:.1f}", "GHz"),
    ("Hamiltonian", "g", f"{float(g_design) * 1e3:.1f}", "MHz"),
    ("Circuit", "C_Σ", f"{C_sigma * 1e15:.1f}", "fF"),
    ("Circuit", "L_J", f"{L_J * 1e9:.2f}", "nH"),
    ("Circuit", "C_r", f"{C_r * 1e15:.1f}", "fF"),
    ("Circuit", "C_c", f"{C_c * 1e15:.2f}", "fF"),
    ("Layout", "Resonator length", f"{resonator_length:.0f}", "µm"),
    ("Layout", "Resonator CPW width", "10", "µm"),
    ("Layout", "Resonator CPW gap", "6", "µm"),
    ("Layout", "Resonator freq", f"{f_resonator_achieved / 1e9:.3f}", "GHz"),
    ("Readout", "Q_ext", f"{Q_ext:,}", ""),
    ("Readout", "κ", f"{kappa * 1e6:.1f}", "kHz"),
    ("Readout", "T_Purcell", f"{T_purcell * 1e6:.0f}", "µs"),
    ("Readout", "|χ|/κ", f"{chi_over_kappa:.1f}", ""),
]

df = pl.DataFrame(data, schema=["Category", "Parameter", "Value", "Unit"])

with pl.Config(
    tbl_rows=30,
    tbl_formatting="MARKDOWN",
    tbl_hide_column_data_types=True,
    tbl_hide_dataframe_shape=True,
):
    display(df)

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
