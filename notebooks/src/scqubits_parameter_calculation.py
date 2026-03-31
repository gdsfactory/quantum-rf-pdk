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
# # Transmon and Fluxonium Parameter Analysis with scQubits
#
# This notebook demonstrates how to use
# [scqubits](https://scqubits.readthedocs.io/en/latest/) {cite:p}`groszkowskiScqubitsPythonPackage2021`
# to numerically analyse **transmon** and **fluxonium** qubits coupled to a
# readout resonator, and how to translate the resulting Hamiltonian
# parameters into physical layout parameters using **qpdk**.
#
# The two qubit modalities represent complementary design philosophies in
# circuit QED.  The **transmon** {cite:p}`kochChargeinsensitiveQubitDesign2007a`
# achieves charge-noise insensitivity by operating deep in the $E_J \gg E_C$
# regime at the cost of weak anharmonicity ($|\alpha| \sim E_C \sim 200$ MHz).
# The **fluxonium** {cite:p}`manucharyan_fluxonium_2009` adds a large
# superinductance that provides both strong anharmonicity and protection
# against charge noise, while enabling coherence times exceeding one
# millisecond {cite:p}`somoroff_millisecond_2023`.
#
# This notebook covers the same design workflow as the companion notebook on
# **pymablock-based dispersive shift calculation**, but uses full numerical
# diagonalization instead of closed-form perturbation theory—making it easy to
# compare the two approaches.
#
# ## Part I — Transmon
#
# ### Background
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
from IPython.display import Math, display
from matplotlib import pyplot as plt

from qpdk import PDK
from qpdk.helper import display_dataframe
from qpdk.models.constants import c_0
from qpdk.models.cpw import cpw_parameters

PDK.activate()

# ruff: disable[E402]
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
# Stack into (evals_count, hilbert_dim) matrix where each row is an eigenvector
evecs = np.array([np.array(v.full()).flatten() for v in evecs])

dim_t = transmon.truncated_dim
dim_r = resonator.truncated_dim


def bare_state_vec(qt_idx: int, res_idx: int) -> np.ndarray:
    """Return the bare state |qt_idx, res_idx⟩ in the full Hilbert space."""
    vec = np.zeros(dim_t * dim_r)
    vec[qt_idx * dim_r + res_idx] = 1.0
    return vec


def find_dressed_index(bare_vec: np.ndarray) -> int:
    """Return the dressed-state index with maximum overlap with bare_vec."""
    return int(np.argmax(np.abs(evecs @ bare_vec) ** 2))


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
#
# From the quantum model, we can extract physical parameters relevant for PDK design.
#
# The coupling strength $g$ (in the dispersive limit $g\ll \omega_q, \omega_r$) can be related to a coupling capacitance $C_c$ via {cite:p}`Savola2023`:
# ```{math}
# :label: eq:coupling-capacitance
# g &\approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_{\Sigma} \left( C_\text{r} - \frac{C_\text{c}^2}{C_\text{q}} \right) }}  \sqrt{\omega_\text{q}\omega_\text{r}} \\
# &\approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_{\Sigma} C_\text{r}}}  \sqrt{\omega_\text{q}\omega_\text{r}}, \quad \text{for } C_\text{c} \ll C_\text{q}
# ```
# where $C_\Sigma$ is the total qubit capacitance, $C_{\text{q}}$ is the capacitance between the qubit pads, and $C_r$ is the total capacitance of the resonator.

# %%
_a_plus_adag = resonator.annihilation_operator() + resonator.creation_operator()
g_sweep = np.linspace(0.01, 0.3, 200)
chi_sweep_analytical = np.array([
    dispersive_shift(omega_t_val, omega_r_val, alpha_val, gi) for gi in g_sweep
])


def chi_scqubits_at_g(g: float) -> float:
    """Numerically compute χ via scQubits at a given coupling g.

    Returns:
        The dispersive shift value.
    """
    hs = scq.HilbertSpace([transmon, resonator])
    hs.interaction_list = [
        scq.InteractionTerm(
            g_strength=g,
            operator_list=[(0, transmon.n_operator), (1, _a_plus_adag)],
        )
    ]
    ev, evc = hs.eigensys(evals_count=12)
    # Stack into (evals_count, hilbert_dim) matrix where each row is an eigenvector
    evc = np.array([np.array(v.full()).flatten() for v in evc])

    def di(qi: int, ri: int) -> int:
        v = np.zeros(dim_t * dim_r)
        v[qi * dim_r + ri] = 1.0
        return int(np.argmax(np.abs(evc @ v) ** 2))

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
# ## Part II — Fluxonium
#
# ### Background
#
# The **fluxonium** qubit {cite:p}`manucharyan_fluxonium_2009` replaces the
# transmon's simple Josephson junction with a junction *shunted* by a large
# superinductance $L_s$.  Its Hamiltonian reads:
# ```{math}
# :label: eq:fluxonium-hamiltonian
# \mathcal{H}_\text{flx}
#   = 4 E_C\, \hat{n}^2
#   - E_J\, \cos\!\bigl(\hat{\varphi} - 2\pi\Phi_\text{ext}/\Phi_0\bigr)
#   + \tfrac{1}{2} E_L\, \hat{\varphi}^2,
# ```
# where $E_C = e^2 / 2C_\Sigma$ is the charging energy,
# $E_J$ is the Josephson energy, and $E_L = (\Phi_0/2\pi)^2 / L_s$ is the
# inductive energy of the superinductance.  The external flux
# $\Phi_\text{ext}$ threads the loop formed by the junction and superinductor.
#
# Compared with the transmon, the fluxonium has three key advantages:
#
# 1. **Large anharmonicity** — At the half-flux-quantum sweet spot
#    ($\Phi_\text{ext} = \Phi_0/2$) the $0 \to 1$ transition can be as low
#    as $\sim$ 100–500 MHz while higher transitions remain at several GHz,
#    giving anharmonicities of many GHz
#    {cite:p}`manucharyan_fluxonium_2009,nguyen_blueprint_2019`.
# 2. **Charge-noise protection** — Like the transmon, the fluxonium operates
#    in a regime where charge dispersion is exponentially suppressed
#    {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
# 3. **Long coherence times** — Heavy fluxoniums (large $E_L$) biased at the
#    half-flux-quantum sweet spot have demonstrated $T_1$ exceeding 1 ms
#    {cite:p}`somoroff_millisecond_2023`.
#
# The trade-off is that the low qubit frequency makes dispersive readout more
# challenging, since the detuning to a typical readout resonator
# ($\omega_r \sim 7$ GHz) is very large
# {cite:p}`zhu_circuit_2013`.

# %% [markdown]
# ### Fluxonium Spectrum
#
# We construct a fluxonium qubit using parameters representative of a
# high-coherence device {cite:p}`nguyen_blueprint_2019`.  The `cutoff`
# parameter sets the size of the phase-basis truncation.

# %%
# Fluxonium Hamiltonian parameters
EJ_flx = 3.395  # Josephson energy in GHz
EC_flx = 0.479  # Charging energy in GHz
EL_flx = 0.132  # Inductive energy in GHz

fluxonium = scq.Fluxonium(
    EJ=EJ_flx,
    EC=EC_flx,
    EL=EL_flx,
    flux=0.5,  # Half-flux-quantum sweet spot
    cutoff=110,
    truncated_dim=10,
)

eigenvals_f = fluxonium.eigenvals(evals_count=6) - fluxonium.eigenvals(evals_count=1)[0]
f01_flx = eigenvals_f[1]
f12_flx = eigenvals_f[2] - eigenvals_f[1]
anharmonicity_flx = f12_flx - f01_flx

display(
    Math(rf"""
\textbf{{Fluxonium Spectrum (scQubits, \Phi_\text{{ext}} = \Phi_0/2):}} \\
E_J = {EJ_flx:.3f}\,\mathrm{{GHz}}, \quad
E_C = {EC_flx:.3f}\,\mathrm{{GHz}}, \quad
E_L = {EL_flx:.3f}\,\mathrm{{GHz}} \\
0\rightarrow 1\ \text{{frequency:}}\ {f01_flx:.4f}\,\mathrm{{GHz}}
  \ ({f01_flx * 1e3:.1f}\,\mathrm{{MHz}}) \\
1\rightarrow 2\ \text{{frequency:}}\ {f12_flx:.3f}\,\mathrm{{GHz}} \\
\text{{Anharmonicity,}}\ \alpha_\text{{flx}} =\ {anharmonicity_flx:.3f}\,\mathrm{{GHz}}
""")
)

# %% [markdown]
# The fluxonium anharmonicity is **orders of magnitude larger** than a
# transmon's ($\sim E_C \approx 200$ MHz).  This is a direct consequence
# of the double-well potential created by the competition between the
# cosine and quadratic terms in {eq}`eq:fluxonium-hamiltonian`.

# %% [markdown]
# ### Fluxonium Spectrum vs. External Flux
#
# The fluxonium energy levels depend strongly on external flux.
# At half-integer flux quanta the lowest two levels form a
# parity-protected sweet spot against flux noise
# {cite:p}`manucharyan_fluxonium_2009,nguyen_blueprint_2019`.

# %%
fluxonium.plot_evals_vs_paramvals(
    "flux", np.linspace(0.0, 1.0, 201), subtract_ground=True, evals_count=6
)
plt.title("Fluxonium spectrum vs. external flux")
plt.show()

# %% [markdown]
# ### Fluxonium–Resonator Dispersive Shift
#
# We couple the fluxonium to the same readout resonator used for the
# transmon analysis and extract the dispersive shift via numerical
# diagonalization.  The coupling is again of the charge type
# {cite:p}`zhu_circuit_2013`:
# ```{math}
# \mathcal{H}_\text{int} = g\, \hat{n}_\text{flx}\,(a_r + a_r^\dagger).
# ```

# %%
omega_r_flx = 7.0  # Resonator frequency in GHz (same as transmon case)
g_flx = 0.1  # Coupling strength in GHz

resonator_flx = scq.Oscillator(E_osc=omega_r_flx, truncated_dim=6)

hilbert_space_flx = scq.HilbertSpace([fluxonium, resonator_flx])
_a_plus_adag_flx = (
    resonator_flx.annihilation_operator() + resonator_flx.creation_operator()
)

interaction_flx = scq.InteractionTerm(
    g_strength=g_flx,
    operator_list=[
        (0, fluxonium.n_operator),
        (1, _a_plus_adag_flx),
    ],
)
hilbert_space_flx.interaction_list = [interaction_flx]

evals_flx, evecs_flx = hilbert_space_flx.eigensys(evals_count=20)
evecs_flx = np.array([np.array(v.full()).flatten() for v in evecs_flx])

dim_f = fluxonium.truncated_dim
dim_r_flx = resonator_flx.truncated_dim


def bare_state_vec_flx(q_idx: int, r_idx: int) -> np.ndarray:
    """Return the bare state |q_idx, r_idx⟩ in the fluxonium–resonator Hilbert space."""
    vec = np.zeros(dim_f * dim_r_flx)
    vec[q_idx * dim_r_flx + r_idx] = 1.0
    return vec


def find_dressed_index_flx(bare_vec: np.ndarray) -> int:
    """Return the dressed-state index with maximum overlap with bare_vec."""
    return int(np.argmax(np.abs(evecs_flx @ bare_vec) ** 2))


idx_00_f = find_dressed_index_flx(bare_state_vec_flx(0, 0))
idx_10_f = find_dressed_index_flx(bare_state_vec_flx(1, 0))
idx_01_f = find_dressed_index_flx(bare_state_vec_flx(0, 1))
idx_11_f = find_dressed_index_flx(bare_state_vec_flx(1, 1))

E_00_f = evals_flx[idx_00_f]
E_10_f = evals_flx[idx_10_f]
E_01_f = evals_flx[idx_01_f]
E_11_f = evals_flx[idx_11_f]
chi_flx = (E_11_f - E_10_f) - (E_01_f - E_00_f)

display(
    Math(rf"""
\textbf{{Fluxonium–Resonator Dispersive Shift:}} \\
\omega_\text{{flx}} = {f01_flx:.4f}\,\mathrm{{GHz}}, \quad
\omega_r = {omega_r_flx:.1f}\,\mathrm{{GHz}}, \quad
g = {g_flx:.1f}\,\mathrm{{GHz}} \\
\Delta = \omega_\text{{flx}} - \omega_r = {f01_flx - omega_r_flx:.3f}\,\mathrm{{GHz}} \\
\chi_{{\mathrm{{fluxonium}}}} = {chi_flx * 1e3:.4f}\,\mathrm{{MHz}}
""")
)

# %% [markdown]
# The fluxonium dispersive shift is typically **much smaller** than the
# transmon's, because the qubit–resonator detuning $|\Delta|$ is very large
# ($\sim 7$ GHz).  Achieving a useful $\chi$ requires either stronger
# coupling or a lower-frequency resonator
# {cite:p}`zhu_circuit_2013`.

# %% [markdown]
# ## Part III — Transmon vs. Fluxonium Comparison
#
# The table below highlights the key physical differences between the
# two qubit modalities, computed with the parameters used above.

# %%
comparison_data = [
    ("Qubit frequency $\\omega_{01}$", f"{f01:.3f} GHz", f"{f01_flx * 1e3:.1f} MHz"),
    (
        "Anharmonicity $\\alpha$",
        f"{anharmonicity:.3f} GHz",
        f"{anharmonicity_flx:.3f} GHz",
    ),
    (
        "$|\\alpha / \\omega_{01}|$",
        f"{abs(anharmonicity / f01) * 100:.1f}%",
        f"{abs(anharmonicity_flx / f01_flx) * 100:.0f}%",
    ),
    ("$E_J / E_C$", f"{EJ / EC:.0f}", f"{EJ_flx / EC_flx:.1f}"),
    (
        "Dispersive shift $\\chi$ ($g = 100$ MHz)",
        f"{chi_scqubits * 1e3:.3f} MHz",
        f"{chi_flx * 1e3:.4f} MHz",
    ),
    (
        "Detuning $|\\Delta|$ from 7 GHz resonator",
        f"{abs(omega_t_val - omega_r_val):.3f} GHz",
        f"{abs(f01_flx - omega_r_flx):.3f} GHz",
    ),
    ("Flux sweet spot", "Any (fixed-frequency)", "$\\Phi_0/2$"),
    (
        "Dominant dephasing mechanism",
        "Charge noise (mitigated by $E_J/E_C \\gg 1$)",
        "Flux noise (mitigated at $\\Phi_0/2$)",
    ),
]

df_compare = pl.DataFrame(
    comparison_data,
    schema=["Property", "Transmon", "Fluxonium"],
    orient="row",
)
display_dataframe(df_compare)

# %% [markdown]
# ### Anharmonicity Landscape
#
# The most striking difference is the anharmonicity-to-frequency ratio.
# For the transmon, $|\alpha/\omega_{01}| \sim E_C / \sqrt{8 E_J E_C} \sim 4\%$,
# meaning gate pulses must be carefully shaped to avoid leakage to the
# $|2\rangle$ state {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
# The fluxonium at half-flux-quantum has $|\alpha/\omega_{01}| \gg 1$,
# so leakage is naturally suppressed, at the cost of slower gates due to
# the low transition frequency {cite:p}`nguyen_blueprint_2019`.
#
# Below we sweep $E_J/E_C$ for the transmon and $E_L$ for the fluxonium
# to visualise how anharmonicity varies across the design space.

# %%
# Transmon: sweep EJ/EC
ej_ec_ratios = np.linspace(10, 120, 50)
alpha_transmon_sweep = np.array([
    ej_ec_to_frequency_and_anharmonicity(r * EC, EC)[1] for r in ej_ec_ratios
])
freq_transmon_sweep = np.array([
    ej_ec_to_frequency_and_anharmonicity(r * EC, EC)[0] for r in ej_ec_ratios
])

# Fluxonium: sweep EL at half flux
el_sweep = np.linspace(0.05, 1.0, 30)
alpha_flx_sweep = []
freq_flx_sweep = []
for el_val in el_sweep:
    fl_tmp = scq.Fluxonium(
        EJ=EJ_flx, EC=EC_flx, EL=el_val, flux=0.5, cutoff=110, truncated_dim=4
    )
    evals_tmp = fl_tmp.eigenvals(evals_count=3)
    evals_tmp -= evals_tmp[0]
    freq_flx_sweep.append(evals_tmp[1])
    alpha_flx_sweep.append((evals_tmp[2] - evals_tmp[1]) - evals_tmp[1])
alpha_flx_sweep = np.array(alpha_flx_sweep)
freq_flx_sweep = np.array(freq_flx_sweep)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(ej_ec_ratios, freq_transmon_sweep, "C0-", linewidth=2, label="$\\omega_{01}$")
ax.plot(
    ej_ec_ratios,
    alpha_transmon_sweep,
    "C1--",
    linewidth=2,
    label="$|\\alpha|$",
)
ax.set_xlabel("$E_J / E_C$")
ax.set_ylabel("Frequency (GHz)")
ax.set_title("Transmon")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(el_sweep, freq_flx_sweep, "C0-", linewidth=2, label="$\\omega_{01}$")
ax.plot(
    el_sweep,
    np.abs(alpha_flx_sweep),
    "C1--",
    linewidth=2,
    label="$|\\alpha|$",
)
ax.set_xlabel("$E_L$ (GHz)")
ax.set_ylabel("Frequency (GHz)")
ax.set_title("Fluxonium ($\\Phi_\\mathrm{ext} = \\Phi_0/2$)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Qubit frequency and anharmonicity across the design space", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# The left panel shows the well-known transmon trade-off: increasing
# $E_J/E_C$ exponentially suppresses charge dispersion but the
# anharmonicity remains pinned at $\alpha \approx E_C$.
# The right panel shows that the fluxonium frequency decreases with
# smaller $E_L$ (heavier fluxonium), while the anharmonicity remains
# large—a qualitatively different design knob.

# %% [markdown]
# ## Part IV — Design Workflow: From Hamiltonian to Layout Parameters
#
# The purpose of computing $\chi$ is to **design** a qubit–resonator system
# with a target dispersive shift.  The sections below walk through the full
# design flow for both a transmon and a fluxonium.
#
# ### Transmon Design
#
# The transmon workflow mirrors the pymablock notebook,
# using qpdk analytical helpers for speed and scQubits for numerical verification:
#
# 1. **Choose target $\chi$** based on readout speed requirements
# 2. **Select $\omega_t$, $\omega_r$, $\alpha$** from qubit design goals
# 3. **Determine $g$** from target $\chi$
# 4. **Convert to circuit parameters** ($C_\Sigma$, $L_J$, $C_c$) using qpdk helpers
# 5. **Convert to layout parameters** (resonator length, capacitor geometry)
#
# #### Step 1–3: From target $\chi$ to coupling $g$

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
# #### Step 4: Convert to circuit parameters
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
ep_eff, z0 = cpw_parameters(width=10, gap=6)


def _resonator_objective(length: float) -> float:
    """Minimise the squared frequency error.

    Returns:
        The squared error between the calculated and target frequency.
    """
    freq = resonator_frequency(
        length=length,
        epsilon_eff=float(np.real(ep_eff)),
        is_quarter_wave=True,
    )
    return (freq - omega_r_design * 1e9) ** 2


result = scipy.optimize.minimize(_resonator_objective, 4000.0, bounds=[(1000, 20000)])
resonator_length = result.x[0]

# Total resonator capacitance from CPW impedance and phase velocity
# {cite:p}`gopplCoplanarWaveguideResonators2008a`:
# C_r = l / Re(Z_0 * v_p)
C_r = 1 / np.real(z0 * (c_0 / np.sqrt(ep_eff))) * resonator_length * 1e-6  # F

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
# #### Step 5: Layout parameters
#
# The circuit parameters map directly to layout dimensions:

# %%
f_resonator_achieved = resonator_frequency(
    length=resonator_length,
    epsilon_eff=float(np.real(ep_eff)),
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
# ### Fluxonium Design
#
# For the fluxonium, the design workflow differs in two ways:
#
# 1. The qubit frequency and anharmonicity depend on three energies
#    ($E_J$, $E_C$, $E_L$) and the external flux, so numerical
#    diagonalization with scQubits is essential — there is no simple
#    closed-form analog of the transmon approximation
#    $\omega \approx \sqrt{8 E_J E_C} - E_C$.
# 2. The circuit includes a **superinductance** $L_s$ whose value maps
#    directly to the layout meander inductor in the qpdk `fluxonium` cell.
#
# #### Fluxonium circuit parameters

# %%
from qpdk.models.qubit import el_to_inductance

# Fluxonium circuit parameters from the Hamiltonian energies
C_sigma_flx = float(ec_to_capacitance(EC_flx))
L_J_flx = float(ej_to_inductance(EJ_flx))
L_s_flx = float(el_to_inductance(EL_flx))

display(
    Math(rf"""
\textbf{{Fluxonium Circuit Parameters:}} \\
E_J = {EJ_flx:.3f}\,\mathrm{{GHz}} \;\Rightarrow\;
  L_J = {L_J_flx * 1e9:.2f}\,\mathrm{{nH}} \\
E_C = {EC_flx:.3f}\,\mathrm{{GHz}} \;\Rightarrow\;
  C_\Sigma = {C_sigma_flx * 1e15:.1f}\,\mathrm{{fF}} \\
E_L = {EL_flx:.3f}\,\mathrm{{GHz}} \;\Rightarrow\;
  L_s = {L_s_flx * 1e9:.1f}\,\mathrm{{nH}}
""")
)

# %% [markdown]
# The superinductance $L_s \sim 300\,\mathrm{nH}$ is realized by a
# meander inductor made of a high-kinetic-inductance material (e.g. NbTiN
# or granular aluminium) {cite:p}`manucharyan_fluxonium_2009,nguyen_blueprint_2019`.
# The qpdk `fluxonium` cell parameterizes this as `inductor_n_turns`.

# %% [markdown]
# ## Summary: Transmon Design Table
#
# The table below summarises the full transmon design flow from Hamiltonian
# parameters to layout parameters, with the dispersive shift as the
# central design target.

# %%
data = [
    ("Target", "Target dispersive shift $\\chi$", f"{chi_target_mhz:.1f}", "MHz"),
    ("Hamiltonian", "$E_J$", f"{EJ_design:.1f}", "GHz"),
    ("Hamiltonian", "$E_C$", f"{EC_design:.1f}", "GHz"),
    ("Hamiltonian", "$\\omega_t$", f"{omega_t_design:.3f}", "GHz"),
    ("Hamiltonian", "$\\alpha$", f"{alpha_design:.1f}", "GHz"),
    ("Hamiltonian", "$\\omega_r$", f"{omega_r_design:.1f}", "GHz"),
    ("Hamiltonian", "$g$", f"{float(g_design) * 1e3:.1f}", "MHz"),
    ("Circuit", "$C_\\Sigma$", f"{C_sigma * 1e15:.1f}", "fF"),
    ("Circuit", "$L_J$", f"{L_J * 1e9:.2f}", "nH"),
    ("Circuit", "$C_r$", f"{C_r * 1e15:.1f}", "fF"),
    ("Circuit", "$C_c$", f"{C_c * 1e15:.2f}", "fF"),
    ("Layout", "Resonator length", f"{resonator_length:.0f}", "µm"),
    ("Layout", "Resonator CPW width", "10", "µm"),
    ("Layout", "Resonator CPW gap", "6", "µm"),
    ("Layout", "Resonator freq", f"{f_resonator_achieved / 1e9:.3f}", "GHz"),
    ("Readout", "$Q_{\\mathrm{ext}}$", f"{Q_ext:,}", ""),
    ("Readout", "$\\kappa$", f"{kappa * 1e6:.1f}", "kHz"),
    ("Readout", "$T_{\\mathrm{Purcell}}$", f"{T_purcell * 1e6:.0f}", "µs"),
    ("Readout", "$|\\chi|/\\kappa$", f"{chi_over_kappa:.1f}", ""),
]

df = pl.DataFrame(data, schema=["Category", "Parameter", "Value", "Unit"], orient="row")

display_dataframe(df)

# %% [markdown]
# ## Summary: Fluxonium Design Table
#
# The fluxonium design uses the same resonator and readout parameters, but
# the qubit circuit parameters differ significantly due to the
# superinductance.

# %%
data_flx = [
    ("Hamiltonian", "$E_J$", f"{EJ_flx:.3f}", "GHz"),
    ("Hamiltonian", "$E_C$", f"{EC_flx:.3f}", "GHz"),
    ("Hamiltonian", "$E_L$", f"{EL_flx:.3f}", "GHz"),
    ("Hamiltonian", "$\\omega_\\text{flx}$", f"{f01_flx * 1e3:.1f}", "MHz"),
    (
        "Hamiltonian",
        "$\\alpha_\\text{flx}$",
        f"{anharmonicity_flx:.3f}",
        "GHz",
    ),
    ("Hamiltonian", "Flux bias", "$\\Phi_0/2$", ""),
    ("Circuit", "$C_\\Sigma$", f"{C_sigma_flx * 1e15:.1f}", "fF"),
    ("Circuit", "$L_J$", f"{L_J_flx * 1e9:.2f}", "nH"),
    ("Circuit", "$L_s$ (superinductance)", f"{L_s_flx * 1e9:.1f}", "nH"),
    (
        "Dispersive shift",
        "$\\chi$ ($g = 100$ MHz, $\\omega_r = 7$ GHz)",
        f"{chi_flx * 1e3:.4f}",
        "MHz",
    ),
]

df_flx = pl.DataFrame(
    data_flx, schema=["Category", "Parameter", "Value", "Unit"], orient="row"
)

display_dataframe(df_flx)

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
# ruff: enable[E402]
