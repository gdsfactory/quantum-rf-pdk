# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# ruff: noqa: E402

# %% [markdown]
# # Transmon Qubit Design with NetKet
#
# This notebook demonstrates how to use
# [NetKet](https://netket.readthedocs.io/) {cite:p}`netket3:2022,netket2:2019`
# to numerically analyse the transmon qubit Hamiltonian for superconducting
# qubit design, and how to translate the resulting quantum parameters into
# physical layout parameters using **qpdk**.
#
# NetKet is a JAX-based library for many-body quantum physics that provides
# exact diagonalisation and variational solvers for custom Hamiltonians.
# For superconducting qubit design, it serves as a flexible numerical
# tool for computing energy spectra, charge dispersion, and coupling
# parameters directly from the microscopic circuit Hamiltonian.
#
# ## Background
#
# The transmon qubit {cite:p}`kochChargeinsensitiveQubitDesign2007a` is an
# anharmonic oscillator formed by shunting a Josephson junction with a large
# capacitance.  Its Hamiltonian in the charge basis reads:
# ```{math}
# :label: eq:transmon-hamiltonian-netket
# \hat{H} = 4 E_C (\hat{n} - n_g)^2 - E_J \cos\hat{\varphi},
# ```
# where $E_C$ is the charging energy, $E_J$ the Josephson energy, $\hat{n}$
# the Cooper-pair number operator, $n_g$ the offset charge, and
# $\hat{\varphi}$ the superconducting phase.
#
# In the truncated charge basis $\{|n\rangle\}$ with
# $n \in \{-n_\text{cut}, \ldots, +n_\text{cut}\}$:
#
# - **Diagonal elements**: $\langle n | \hat{H} | n \rangle = 4 E_C (n - n_g)^2$
# - **Off-diagonal elements**: $\langle n \pm 1 | \hat{H} | n \rangle = -E_J / 2$
#
# In the transmon regime ($E_J / E_C \gg 1$), the qubit frequency and
# anharmonicity are approximately
# {cite:p}`kochChargeinsensitiveQubitDesign2007a`:
# ```{math}
# :label: eq:transmon-approx-netket
# \omega_q \approx \sqrt{8 E_J E_C} - E_C, \qquad \alpha \approx -E_C.
# ```

# %% tags=["hide-input", "hide-output"]
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
import polars as pl
import scipy
from IPython.display import Math, display
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from qpdk import PDK
from qpdk.models.constants import c_0
from qpdk.models.cpw import cpw_parameters

PDK.activate()

from qpdk.cells.transmon import (
    double_pad_transmon,
    xmon_transmon,
)
from qpdk.models.perturbation import (
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
# ## Building the Transmon Hamiltonian
#
# We construct the transmon Hamiltonian as a matrix in the truncated charge
# basis and wrap it in a NetKet
# [`LocalOperator`](https://netket.readthedocs.io/en/latest/api/operator.html#netket.operator.LocalOperator),
# which can then be passed to NetKet's exact-diagonalisation routines.

# %%
EJ = 20.0  # Josephson energy in GHz
EC = 0.2  # Charging energy in GHz
ng = 0.0  # Offset charge (gate charge)

ncut = 15  # Charge basis truncation: n ∈ {-ncut, ..., +ncut}
n_states = 2 * ncut + 1

# Build the Hamiltonian matrix in the charge basis
charges = jnp.arange(n_states) - ncut  # physical charge quantum numbers
H_mat = jnp.diag(4 * EC * (charges - ng) ** 2)
for n in range(n_states - 1):
    H_mat = H_mat.at[n, n + 1].set(-EJ / 2)  # Josephson tunnelling
    H_mat = H_mat.at[n + 1, n].set(-EJ / 2)

# Wrap as a NetKet operator on a Fock Hilbert space
hi = nk.hilbert.Fock(n_max=n_states - 1, N=1)
H_transmon = nk.operator.LocalOperator(hi, np.asarray(H_mat), acting_on=[0])

# %% [markdown]
# ## Transmon Spectrum via Exact Diagonalisation
#
# NetKet provides
# [`lanczos_ed`](https://netket.readthedocs.io/en/latest/api/exact.html#netket.exact.lanczos_ed)
# for efficient sparse-matrix diagonalisation.  We extract the lowest five
# eigenvalues and compute the qubit frequency $\omega_q = E_1 - E_0$ and
# the anharmonicity $\alpha = (E_2 - E_1) - (E_1 - E_0)$.

# %%
evals_exact = nk.exact.lanczos_ed(H_transmon, k=5, compute_eigenvectors=False)
e0_exact = evals_exact[0]
evals = evals_exact - e0_exact  # shift ground state to zero

f01 = evals[1] - evals[0]
f12 = evals[2] - evals[1]
alpha_nk = f12 - f01  # negative for transmon

# Analytical transmon approximation from qpdk
omega_t_approx, _ = ej_ec_to_frequency_and_anharmonicity(EJ, EC)

display(
    Math(rf"""
\textbf{{Transmon Spectrum (NetKet):}} \\
E_J/E_C = {EJ / EC:.0f} \\
f_{{01}} = {f01:.4f}\,\mathrm{{GHz}} \quad
(\text{{approx.}}\;\sqrt{{8 E_J E_C}} - E_C = {omega_t_approx:.4f}\,\mathrm{{GHz}}) \\
f_{{12}} = {f12:.4f}\,\mathrm{{GHz}} \\
\alpha = f_{{12}} - f_{{01}} = {alpha_nk:.4f}\,\mathrm{{GHz}} \quad
(\text{{approx.}}\;-E_C = -{EC:.4f}\,\mathrm{{GHz}})
""")
)

# %% [markdown]
# ## Variational Energy via VMC (Full Summation)
#
# While exact diagonalisation works for this small Hilbert space, larger
# systems require variational methods. Here we demonstrate a simple
# Variational Monte Carlo (VMC) approach using JAX and Flax, computing
# the energy by summing over all basis states.
#
# This case is trivial to diagonalize exactly, but looking at something
# more complex cannot always be done exactly due to the exponential
# growth of the Hilbert space.
#
# We define a simple feedforward neural network using Flax to serve
# as our variational ansatz for the ground state wavefunction.
# The network takes the charge configuration
# as input and outputs the logarithm of the wavefunction amplitude (log-psi).
# We then compute the energy expectation value by summing over all charge states,
# and use JAX's automatic differentiation to compute gradients for optimization.
#
# For the network architecture, we use two hidden layers with Leaky ReLU {cite:p}`maas2013rectifier` non-linearities,
# constituting a simple multilayer perceptron {cite:p}`rosenblattPerceptronProbabilisticModel1958`.
#
# For details on VMC with NetKet, see the documentation:
# https://netket.readthedocs.io/en/latest/vmc-from-scratch/index.html


# %%
class VariationalTransmon(nn.Module):
    """A simple multilayer perceptron neural network.

    Uses two _fully connected_ linear layers with Leaky ReLU non-linearities
    followed by a final linear layer to output a single scalar value.
    """

    @nn.compact
    def __call__(self, x):
        """Evaluate the model on a set of input configurations."""
        # x has shape (..., 1)
        x = x.astype(jnp.float32)
        x = nn.Dense(features=16)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=1)(x)
        return jnp.squeeze(x, axis=-1)


model = VariationalTransmon()
parameters = model.init(jax.random.key(0), jnp.ones((1, 1)))


def to_array(model, parameters):
    """Compute the normalized wavefunction as an array."""
    all_configs = hi.all_states()
    logpsi = model.apply(parameters, all_configs)
    psi = jnp.exp(logpsi)
    return psi / jnp.linalg.norm(psi)


def compute_energy(model, parameters, H_sparse):
    """Compute the variational energy of the state."""
    psi = to_array(model, parameters)
    psi = psi.astype(H_sparse.dtype)
    return jnp.real(jnp.vdot(psi, H_sparse @ psi))


@partial(jax.jit, static_argnames="model")
def train_step(model, parameters, opt_state, H_sparse):
    """Perform a single variational optimization step."""
    energy, grad = jax.value_and_grad(compute_energy, argnums=1)(
        model, parameters, H_sparse
    )
    updates, opt_state = tx.update(grad, opt_state, parameters)
    parameters = optax.apply_updates(parameters, updates)
    return parameters, opt_state, energy


H_sparse = H_transmon.to_sparse(jax_=True)
tx = optax.adam(0.01)
opt_state = tx.init(parameters)

energies = []
for _ in tqdm(range(200), desc="Variational Optimization"):
    parameters, opt_state, e_val = train_step(model, parameters, opt_state, H_sparse)
    energies.append(e_val)

e_gs_variational = float(energies[-1])

display(
    Math(rf"""
\textbf{{Ground State Energy Comparison:}} \\
E_{{0, \text{{exact}}}} = {e0_exact:.6f}\,\mathrm{{GHz}} \\
E_{{0, \text{{variational}}}} = {e_gs_variational:.6f}\,\mathrm{{GHz}} \\
\text{{Error:}} \quad |E_\text{{exact}} - E_\text{{var}}| = {abs(e0_exact - e_gs_variational):.2e}\,\mathrm{{GHz}}
""")
)


# %% [markdown]
# ## Charge Dispersion
#
# A key design criterion for the transmon is its exponential insensitivity
# to charge noise.  As $E_J/E_C$ increases, the energy levels flatten as a
# function of the offset charge $n_g$, suppressing dephasing from
# charge fluctuations {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
#
# We sweep $n_g$ across one period and compute the first three transition
# energies for several values of $E_J/E_C$.

# %%
ng_sweep = jnp.linspace(-0.5, 0.5, 101)
ej_ec_ratios = [1, 5, 20, 50]

fig, axes = plt.subplots(1, len(ej_ec_ratios), figsize=(14, 3.5), sharey=True)

for ax, ratio in zip(axes, ej_ec_ratios):
    EJ_sweep = ratio * EC
    levels = jnp.zeros((len(ng_sweep), 4))

    for i, ng_val in enumerate(ng_sweep):
        H_sweep = jnp.diag(4 * EC * (charges - ng_val) ** 2)
        for n in range(n_states - 1):
            H_sweep = H_sweep.at[n, n + 1].set(-EJ_sweep / 2)
            H_sweep = H_sweep.at[n + 1, n].set(-EJ_sweep / 2)
        ev = jnp.linalg.eigvalsh(H_sweep)
        levels = levels.at[i].set(ev[:4] - ev[0])

    for j in range(1, 4):
        ax.plot(ng_sweep, levels[:, j], label=rf"$E_{j} - E_0$")
    ax.set_xlabel(r"$n_g$")
    ax.set_title(rf"$E_J/E_C = {ratio}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Energy (GHz)")
axes[-1].legend(fontsize=8)
fig.suptitle("Charge Dispersion: Transmon Regime", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# The plots demonstrate how the energy levels become increasingly flat with
# respect to the offset charge $n_g$ as the ratio $E_J/E_C$ grows.  In the
# transmon regime ($E_J/E_C \gtrsim 50$), the charge dispersion is
# exponentially suppressed, making the qubit insensitive to charge noise—a
# key design advantage {cite:p}`kochChargeinsensitiveQubitDesign2007a`.

# %% [markdown]
# ## Transmon–Resonator Coupling
#
# In circuit QED, the transmon is coupled to a readout resonator via the
# electric-field interaction
# {cite:p}`blaisCircuitQuantumElectrodynamics2021`:
# ```{math}
# :label: eq:coupling-hamiltonian-netket
# \hat{H}_\text{int} = g\, (\hat{a} + \hat{a}^\dagger)\,
#     (\hat{b} + \hat{b}^\dagger),
# ```
# where $\hat{a}$ and $\hat{b}$ are the transmon and resonator ladder
# operators respectively, and $g$ is the coupling strength.
#
# We build the full transmon–resonator Hamiltonian in the energy eigenbasis
# of the transmon (Duffing oscillator approximation) and diagonalise it
# using NetKet to extract the **dispersive shift** $\chi$:
# ```{math}
# :label: eq:dispersive-shift-definition-netket
# \chi = (E_{11} - E_{10}) - (E_{01} - E_{00}),
# ```
# where $E_{ij}$ is the dressed eigenenergy with $i$ transmon excitations
# and $j$ resonator photons.

# %%
omega_r_val = 7.0  # Resonator frequency in GHz
g_val = 0.1  # Coupling strength in GHz
n_levels = 8  # Truncation level for each mode

# Combined Hilbert space: site 0 = transmon, site 1 = resonator
hi_combined = nk.hilbert.Fock(n_max=n_levels - 1, N=2)

# Transmon as Duffing oscillator: f01 * n + (α/2) * n(n-1)
H_qubit = jnp.diag(
    jnp.array([f01 * n + (alpha_nk / 2) * n * (n - 1) for n in range(n_levels)])
)

# Resonator: ω_r * n
H_res = jnp.diag(jnp.array([omega_r_val * n for n in range(n_levels)]))

# (a + a†) ladder operator
a_plus_adag = jnp.zeros((n_levels, n_levels))
for n in range(n_levels - 1):
    a_plus_adag = a_plus_adag.at[n, n + 1].set(jnp.sqrt(n + 1))
    a_plus_adag = a_plus_adag.at[n + 1, n].set(jnp.sqrt(n + 1))

# Coupling operator: g * (a + a†) ⊗ (b + b†)
coupling = g_val * jnp.kron(a_plus_adag, a_plus_adag)

# Build full Hamiltonian as sum of NetKet operators
H_full = (
    nk.operator.LocalOperator(hi_combined, np.asarray(H_qubit), acting_on=[0])
    + nk.operator.LocalOperator(hi_combined, np.asarray(H_res), acting_on=[1])
    + nk.operator.LocalOperator(hi_combined, np.asarray(coupling), acting_on=[0, 1])
)

# Full exact diagonalisation
evals_full, evecs_full = nk.exact.full_ed(H_full, compute_eigenvectors=True)
sort_idx = jnp.argsort(evals_full)
evals_full = evals_full[sort_idx]
evecs_full = evecs_full[:, sort_idx]


# Identify dressed states by maximum overlap with bare states
def _find_dressed(qubit_idx: int, res_idx: int) -> int:
    """Return the index of the dressed state closest to |qubit_idx, res_idx⟩."""
    bare = jnp.zeros(n_levels**2)
    bare = bare.at[qubit_idx * n_levels + res_idx].set(1.0)
    return int(jnp.argmax(jnp.abs(evecs_full.T @ bare) ** 2))


idx_00 = _find_dressed(0, 0)
idx_10 = _find_dressed(1, 0)
idx_01 = _find_dressed(0, 1)
idx_11 = _find_dressed(1, 1)

chi_netket = (evals_full[idx_11] - evals_full[idx_10]) - (
    evals_full[idx_01] - evals_full[idx_00]
)

display(
    Math(rf"""
\textbf{{Dispersive Shift (NetKet):}} \\
g = {g_val * 1e3:.0f}\,\mathrm{{MHz}}, \quad
\omega_r = {omega_r_val:.1f}\,\mathrm{{GHz}} \\
\Delta = \omega_q - \omega_r = {f01 - omega_r_val:.3f}\,\mathrm{{GHz}} \\
\chi = {chi_netket * 1e3:.3f}\,\mathrm{{MHz}}
""")
)

# %% [markdown]
# ## Dispersive Shift vs. Coupling Strength
#
# We sweep the coupling strength $g$ to observe the quadratic scaling of
# the dispersive shift in the dispersive regime ($g \ll |\Delta|$).

# %%
g_sweep = jnp.linspace(0.01, 0.25, 20)
chi_sweep = jnp.zeros_like(g_sweep)

for i, g_i in enumerate(g_sweep):
    H_i = (
        nk.operator.LocalOperator(hi_combined, np.asarray(H_qubit), acting_on=[0])
        + nk.operator.LocalOperator(hi_combined, np.asarray(H_res), acting_on=[1])
        + nk.operator.LocalOperator(
            hi_combined,
            np.asarray(g_i * jnp.kron(a_plus_adag, a_plus_adag)),
            acting_on=[0, 1],
        )
    )
    ev_i, evc_i = nk.exact.full_ed(H_i, compute_eigenvectors=True)
    si = jnp.argsort(ev_i)
    ev_i, evc_i = ev_i[si], evc_i[:, si]

    def _fd(qi: int, ri: int, _evc: jax.Array = evc_i) -> int:
        bv = jnp.zeros(n_levels**2)
        bv = bv.at[qi * n_levels + ri].set(1.0)
        return int(jnp.argmax(jnp.abs(_evc.T @ bv) ** 2))

    j00, j10, j01, j11 = _fd(0, 0), _fd(1, 0), _fd(0, 1), _fd(1, 1)
    chi_sweep = chi_sweep.at[i].set((ev_i[j11] - ev_i[j10]) - (ev_i[j01] - ev_i[j00]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(g_sweep * 1e3, chi_sweep * 1e3, "o-", markersize=5, label="NetKet (numerical)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Coupling strength $g$ (MHz)")
ax.set_ylabel(r"Dispersive shift $\chi$ (MHz)")
ax.set_title(
    rf"Dispersive shift vs. coupling ($\omega_q={f01:.2f}$ GHz, "
    rf"$\omega_r={omega_r_val:.1f}$ GHz)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Design Workflow: From Target $\chi$ to Layout Parameters
#
# The purpose of computing $\chi$ is to **design** a qubit–resonator system
# with a target dispersive shift.  The workflow is:
#
# 1. **Choose target $\chi$** based on readout speed requirements
# 2. **Select $\omega_q$, $\omega_r$, $\alpha$** from qubit design goals
# 3. **Determine $g$** from target $\chi$ using qpdk helpers
# 4. **Convert to circuit parameters** ($C_\Sigma$, $L_J$, $C_c$)
# 5. **Convert to layout parameters** (resonator length, capacitor geometry)
#
# ### Step 1–3: From target $\chi$ to coupling $g$

# %%
# Design targets
chi_target_mhz = -1.0  # Target dispersive shift in MHz
chi_target = chi_target_mhz * 1e-3  # Convert to GHz

# Qubit and resonator parameters (from NetKet calculation above)
EJ_design = EJ
EC_design = EC
omega_r_design = omega_r_val

# Use the analytical transmon approximation for design
omega_t_design, alpha_design = ej_ec_to_frequency_and_anharmonicity(
    EJ_design, EC_design
)

# Find g using the qpdk analytical inversion
g_design = dispersive_shift_to_coupling(
    chi_target, omega_t_design, omega_r_design, alpha_design
)

display(
    Math(rf"""
\textbf{{Step 1–3: Hamiltonian Design}} \\
\text{{Target:}}\quad \chi = {chi_target_mhz:.1f}\,\mathrm{{MHz}} \\
E_J = {EJ_design:.1f}\,\mathrm{{GHz}}, \quad
E_C = {EC_design:.1f}\,\mathrm{{GHz}} \\
\omega_q = {omega_t_design:.3f}\,\mathrm{{GHz}}, \quad
\alpha \approx -E_C = -{EC_design:.1f}\,\mathrm{{GHz}} \\
\omega_r = {omega_r_design:.1f}\,\mathrm{{GHz}} \\
\text{{Required coupling:}}\quad g = {float(g_design) * 1e3:.1f}\,\mathrm{{MHz}}
""")
)

# %% [markdown]
# ### Step 4: Convert to circuit parameters
#
# Using the qpdk helper functions, we convert the Hamiltonian parameters to
# circuit parameters:
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
    """Minimise the squared frequency error."""
    freq = resonator_frequency(
        length=length,
        epsilon_eff=float(jnp.real(ep_eff)),
        is_quarter_wave=True,
    )
    return (freq - omega_r_design * 1e9) ** 2


result = scipy.optimize.minimize(_resonator_objective, 4000.0, bounds=[(1000, 20000)])
resonator_length = result.x[0]

# Total resonator capacitance from CPW impedance and phase velocity
# {cite:p}`gopplCoplanarWaveguideResonators2008a`:
# C_r = l / Re(Z_0 * v_p)
C_r = float(1 / jnp.real(z0 * (c_0 / jnp.sqrt(ep_eff))) * resonator_length * 1e-6)  # F

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
    epsilon_eff=float(jnp.real(ep_eff)),
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
gamma_phi = measurement_induced_dephasing(chi_target, kappa, n_bar)

# SNR estimate: the signal-to-noise ratio scales as χ/κ
chi_over_kappa = abs(chi_target) / kappa

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
# ## QPDK Layout Components
#
# The design parameters computed above map directly to physical layout
# cells in qpdk.  Below we show two transmon geometries that can be
# used for fabrication: the **double-pad transmon** and the **Xmon
# transmon**.
#
# Each cell exposes design parameters (pad dimensions, gap widths,
# junction dimensions) that determine the Hamiltonian parameters
# $E_J$, $E_C$, and $g$ through their effect on the capacitance and
# Josephson energy of the circuit.

# %%
c_dpt = double_pad_transmon()
c_dpt.plot()
plt.title("Double-Pad Transmon")
plt.show()

# %%
c_xmon = xmon_transmon()
c_xmon.plot()
plt.title("Xmon Transmon")
plt.show()

# %% [markdown]
# **Design parameter mapping:**
#
# | Layout parameter | Circuit parameter | Hamiltonian parameter |
# |---|---|---|
# | Pad dimensions, gap width | $C_\Sigma$ (total capacitance) | $E_C = e^2 / (2 C_\Sigma)$ |
# | Junction area, critical current | $L_J$ (Josephson inductance) | $E_J = \Phi_0^2 / (4\pi^2 L_J)$ |
# | Coupling finger length / gap | $C_c$ (coupling capacitance) | $g \propto C_c / \sqrt{C_\Sigma C_r}$ |
# | CPW length, width, gap | $Z_0$, $\varepsilon_\text{eff}$ | $\omega_r$ (resonator frequency) |

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
    ("Hamiltonian", "ω_q (NetKet)", f"{f01:.4f}", "GHz"),
    ("Hamiltonian", "ω_q (approx.)", f"{omega_t_design:.4f}", "GHz"),
    ("Hamiltonian", "α (NetKet)", f"{alpha_nk:.4f}", "GHz"),
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

df = pl.DataFrame(data, schema=["Category", "Parameter", "Value", "Unit"], orient="row")

display(df.to_pandas().style.hide(axis="index"))

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
