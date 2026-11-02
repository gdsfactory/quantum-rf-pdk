# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [raw] tags=["remove-cell"]
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models,netket] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///

# %% [markdown]
# # Transmon Qubit Design with NetKet
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` and `netket` extras:
#
# ```bash
# uv add "qpdk[models,netket]"
# # or, from a checkout of this repository:
# uv sync --extra models --extra netket
# # or with pip:
# pip install "qpdk[models,netket]"
# ```
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# This notebook demonstrates how to use
# [NetKet](https://netket.readthedocs.io/) {cite:p}`netket3:2022,netket2:2019`
# to numerically analyse the transmon qubit Hamiltonian for superconducting
# qubit design, and how to translate the resulting quantum parameters into
# physical layout parameters using **qpdk**.
#
# Every section starts from a qpdk layout cell, takes its circuit parameters
# from an existing qpdk model, and hands NetKet a Hamiltonian built from them:
#
# - a **double-pad transmon**, with $E_\text{C}$ from the Palace capacitance
#   lookup of its pads, solved by exact diagonalisation, by a hand-written
#   variational loop, and by NetKet's own VMC driver;
# - the **transmon–resonator** system, including its real-time dynamics with
#   NetKet's experimental time-dependent variational principle (TDVP);
# - a **Josephson junction array**, a many-body problem whose Hilbert space
#   grows exponentially with its length, solved with Monte Carlo VMC and a
#   custom Flax network, and fed back into the array's SAX model;
# - the **unimon** and **gatemon**, whose existing qpdk Hamiltonians become
#   NetKet operators with one call.
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
#
# ![Transmon circuit with a Josephson junction shunted by capacitance, setting EJ and EC](figures/transmon-energy.svg)
#
# The shunt capacitance sets $E_\text{C}$, while the junction critical current sets $E_\text{J}$.
#
# ```{math}
# :label: eq:transmon-hamiltonian-netket
# \hat{H} = 4 E_\text{C} (\hat{n} - n_\text{g})^2 - E_\text{J} \cos\hat{\varphi},
# ```
# where $E_\text{C}$ is the charging energy, $E_\text{J}$ the Josephson energy, $\hat{n}$
# the Cooper-pair number operator, $n_\text{g}$ the offset charge, and
# $\hat{\varphi}$ the superconducting phase.
#
# In the truncated charge basis $\{|n\rangle\}$ with
# $n \in \{-n_\text{cut}, \ldots, +n_\text{cut}\}$:
#
# - **Diagonal elements**: $\langle n | \hat{H} | n \rangle = 4 E_\text{C} (n - n_\text{g})^2$
# - **Off-diagonal elements**: $\langle n \pm 1 | \hat{H} | n \rangle = -E_\text{J} / 2$
#
# In the transmon regime ($E_\text{J} / E_\text{C} \gg 1$), the qubit frequency and
# anharmonicity are approximately
# {cite:p}`kochChargeinsensitiveQubitDesign2007a`:
# ```{math}
# :label: eq:transmon-approx-netket
# \omega_\text{q} \approx \sqrt{8 E_\text{J} E_\text{C}} - E_\text{C}, \qquad \alpha \approx -E_\text{C}.
# ```

# %% tags=["hide-input", "hide-output"]
import os
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing QPDK...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models,netket] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])
    # On a Colab TPU runtime, installing qpdk replaces the TPU build of JAX
    # with the CPU one, so put the TPU runtime library back at the same JAX
    # version, and run NetKet in single precision, which TPUs compute natively.
    if os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("COLAB_TPU_ADDR"):
        from importlib.metadata import version

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"jax[tpu]=={version('jax')}",
        ])
        os.environ["NETKET_ENABLE_X64"] = "0"

# %% tags=["hide-input", "hide-output"]
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
import optax
import polars as pl
import scipy
from IPython.display import Math, display
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from qpdk import PDK
from qpdk.helper import display_dataframe
from qpdk.models.constants import Φ_0, c_0, e, h
from qpdk.models.cpw import cpw_parameters

PDK.activate()

from qpdk.cells import gatemon, josephson_junction_array
from qpdk.cells.transmon import (
    double_pad_transmon,
    xmon_transmon,
)
from qpdk.models.gatemon import (
    gatemon_capacitances,
    gatemon_frequency_and_anharmonicity,
    gatemon_hamiltonian,
)
from qpdk.models.junction import (
    josephson_junction_array as josephson_junction_array_model,
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
    double_pad_transmon_capacitance,
    ec_to_capacitance,
    ej_to_inductance,
)
from qpdk.models.resonator import resonator_frequency
from qpdk.models.unimon import unimon_frequency_and_anharmonicity, unimon_hamiltonian

# %% [markdown]
# ## Building the Transmon Hamiltonian
#
# The charging energy comes from the layout. We read the pad size and gap
# of a {func}`~qpdk.cells.double_pad_transmon` cell and look up the
# capacitance between its pads with
# {func}`~qpdk.models.qubit.double_pad_transmon_capacitance`, which
# interpolates a stored Palace electrostatic sweep.  The Josephson energy is
# set by the junction, which we treat as a design choice here.
#
# We then construct the transmon Hamiltonian as a matrix in the truncated charge
# basis and wrap it in a NetKet
# [`LocalOperator`](https://netket.readthedocs.io/en/latest/api/operator.html#netket.operator.LocalOperator),
# which can then be passed to NetKet's exact-diagonalisation routines.

# %%
transmon_cell = double_pad_transmon()
pad_width, pad_height = transmon_cell.settings["pad_size"]
C_pads = float(
    double_pad_transmon_capacitance(
        pad_width=pad_width,
        pad_height=pad_height,
        pad_gap=transmon_cell.settings["pad_gap"],
    )
)
# The lookup excludes the junction; add its capacitance for your process here.
C_junction = 0.0

EJ = 20.0  # Josephson energy in GHz, set by the junction
EC = e**2 / (2 * (C_pads + C_junction)) / h / 1e9  # Charging energy in GHz
ng = 0.0  # Offset charge (gate charge)

display(
    Math(rf"""
C_\text{{pads}} = {C_pads * 1e15:.1f}\,\text{{fF}} \quad\Rightarrow\quad
E_\text{{C}} = e^2 / 2C = {EC:.4f}\,\text{{GHz}}
""")
)

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
# eigenvalues and compute the qubit frequency $\omega_\text{q} = E_1 - E_0$ and
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
E_\text{{J}}/E_\text{{C}} = {EJ / EC:.0f} \\
f_{{01}} = {f01:.4f}\,\text{{GHz}} \quad
(\text{{approx.}}\;\sqrt{{8 E_\text{{J}} E_\text{{C}}}} - E_\text{{C}} = {omega_t_approx:.4f}\,\text{{GHz}}) \\
f_{{12}} = {f12:.4f}\,\text{{GHz}} \\
\alpha = f_{{12}} - f_{{01}} = {alpha_nk:.4f}\,\text{{GHz}} \quad
(\text{{approx.}}\;-E_\text{{C}} = -{EC:.4f}\,\text{{GHz}})
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
# For the network architecture, we use a compact multilayer perceptron
# {cite:p}`rosenblattPerceptronProbabilisticModel1958` preceded by a
# sinusoidal input encoding at geometrically spaced frequencies.  This
# encoding lifts the one-dimensional Fock-state index into a
# higher-dimensional feature space of sines and cosines, enabling the
# subsequent :math:`\tanh`-activated hidden layers to approximate the
# smooth, Gaussian-like ground-state wavefunction much more efficiently
# than piecewise-linear activations such as Leaky ReLU.
#
# For details on VMC with NetKet, see the documentation:
# https://netket.readthedocs.io/en/latest/vmc-from-scratch/index.html


# %%
class VariationalTransmon(nn.Module):
    r"""A compact MLP with sinusoidal input encoding for the transmon wavefunction.

    Sinusoidal (Fourier) features at geometrically spaced frequencies lift
    the one-dimensional Fock-state index into a richer feature space, allowing
    the two :math:`\tanh`-activated hidden layers to approximate the smooth,
    Gaussian-like ground state much more efficiently than a plain MLP with
    piecewise-linear activations.
    """

    n_max: int = 30
    n_freqs: int = 4

    @nn.compact
    def __call__(self, x):
        """Evaluate the model on a set of input configurations.

        Returns:
            The model evaluation result as a squeezed array.
        """
        # x has shape (..., 1)
        x = x.astype(jnp.float32)
        # Sinusoidal feature encoding at geometrically spaced frequencies;
        # maps the scalar input to a 2*n_freqs-dimensional feature vector.
        freqs = 2.0 ** jnp.arange(self.n_freqs) * (jnp.pi / self.n_max)
        x = jnp.concatenate([jnp.sin(x * freqs), jnp.cos(x * freqs)], axis=-1)
        x = nn.Dense(features=32)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=32)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=1)(x)
        return jnp.squeeze(x, axis=-1)


model = VariationalTransmon(n_max=n_states - 1)
parameters = model.init(jax.random.key(0), jnp.ones((1, 1)))


def to_array(model, parameters):
    """Compute the normalized wavefunction as an array.

    Returns:
        The normalized wavefunction as a JAX array.
    """
    all_configs = hi.all_states()
    logpsi = model.apply(parameters, all_configs)
    psi = jnp.exp(logpsi)
    return psi / jnp.linalg.norm(psi)


def compute_energy(model, parameters, H_sparse):
    """Compute the variational energy of the state.

    Returns:
        The expected value of the energy (real part).
    """
    psi = to_array(model, parameters)
    psi = psi.astype(H_sparse.dtype)
    return jnp.real(jnp.vdot(psi, H_sparse @ psi))


@partial(jax.jit, static_argnames="model")
def train_step(model, parameters, opt_state, H_sparse):
    """Perform a single variational optimization step.

    Returns:
        A tuple containing (updated parameters, updated opt_state, energy).
    """
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
E_{{0, \text{{exact}}}} = {e0_exact:.6f}\,\text{{GHz}} \\
E_{{0, \text{{variational}}}} = {e_gs_variational:.6f}\,\text{{GHz}} \\
\text{{Error:}} \quad |E_\text{{exact}} - E_\text{{var}}| = {abs(e0_exact - e_gs_variational):.2e}\,\text{{GHz}}
""")
)


# %% [markdown]
# ### The same network with NetKet's VMC driver
#
# The loop above is what NetKet automates.  A variational state wraps any
# Flax module, so the same `VariationalTransmon` network plugs straight into
# [`nk.vqs.FullSumState`](https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.FullSumState.html),
# which still sums over every basis state, and
# [`nk.driver.VMC`](https://netket.readthedocs.io/en/latest/api/_generated/driver/netket.driver.VMC.html)
# runs the optimisation and logs the energy.  For larger systems the only
# change is to swap the full-summation state for a Monte Carlo
# [`MCState`](https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html)
# with a sampler, as in the junction-array section below.

# %%
vs_transmon = nk.vqs.FullSumState(hi, VariationalTransmon(n_max=n_states - 1), seed=0)
vmc_driver = nk.driver.VMC(H_transmon, optax.adam(0.01), variational_state=vs_transmon)
vmc_log = nk.logging.RuntimeLog()
vmc_driver.run(n_iter=200, out=vmc_log, show_progress=False)
e_gs_netket = float(vs_transmon.expect(H_transmon).mean.real)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(
    np.abs(np.asarray(energies) - e0_exact), label="Hand-written loop (Flax + optax)"
)
ax.semilogy(
    np.abs(np.asarray(vmc_log["Energy"]["Mean"].real) - e0_exact),
    "--",
    label="NetKet VMC driver",
)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$|E - E_{0,\text{exact}}|$ (GHz)")
ax.set_title("Same Flax ansatz, two optimisation loops")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()

display(
    Math(rf"""
E_{{0, \text{{NetKet VMC}}}} = {e_gs_netket:.6f}\,\text{{GHz}} \quad
|E_\text{{exact}} - E_\text{{NetKet VMC}}| = {abs(e0_exact - e_gs_netket):.2e}\,\text{{GHz}}
""")
)

# %% [markdown]
# ## Charge Dispersion
#
# A key design criterion for the transmon is its exponential insensitivity
# to charge noise.  As $E_\text{J}/E_\text{C}$ increases, the energy levels flatten as a
# function of the offset charge $n_\text{g}$, suppressing dephasing from
# charge fluctuations {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
#
# We sweep $n_\text{g}$ across one period and compute the first three transition
# energies for several values of $E_\text{J}/E_\text{C}$.

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
    ax.set_xlabel(r"$n_\text{g}$")
    ax.set_title(rf"$E_J/E_C = {ratio}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Energy (GHz)")
axes[-1].legend(fontsize=8)
fig.suptitle("Charge Dispersion: Transmon Regime", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# The plots demonstrate how the energy levels become increasingly flat with
# respect to the offset charge $n_\text{g}$ as the ratio $E_\text{J}/E_\text{C}$ grows.  In the
# transmon regime ($E_\text{J}/E_\text{C} \gtrsim 50$), the charge dispersion is
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
g = {g_val * 1e3:.0f}\,\text{{MHz}}, \quad
\omega_\text{{r}} = {omega_r_val:.1f}\,\text{{GHz}} \\
\Delta = \omega_\text{{q}} - \omega_\text{{r}} = {f01 - omega_r_val:.3f}\,\text{{GHz}} \\
\chi = {chi_netket * 1e3:.3f}\,\text{{MHz}}
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

    def _fd(qi: int, ri: int, _evc: np.ndarray = evc_i) -> int:
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
    rf"Dispersive shift vs. coupling ($\omega_\text{{q}} = {f01:.2f}\,\text{{GHz}}$, "
    rf"$\omega_\text{{r}} = {omega_r_val:.1f}\,\text{{GHz}}$)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Vacuum Rabi Oscillations with TDVP
#
# NetKet also evolves states in time.  The experimental
# [`nkx.TDVP`](https://netket.readthedocs.io/en/latest/api/_generated/experimental/dynamics/netket.experimental.TDVP.html)
# driver integrates the time-dependent variational principle
# {cite:p}`carleoSolvingQuantumManybody2017`: at every step it solves
# $S\,\dot{\theta} = -\mathrm{i}\,F$ for the parameter velocities, with
# $S$ the quantum geometric tensor and $F$ the energy gradient.
#
# Here we tune the transmon from the spectrum above to within $\delta = g$ of
# the readout resonator and start with one excitation in the qubit,
# $|1, 0\rangle$.  In the frame rotating at $\omega_\text{r}$ and within the
# rotating-wave approximation, the excitation swaps partially into the
# resonator and back at the generalised vacuum Rabi frequency
# $\sqrt{4g^2 + \delta^2}$ {cite:p}`blaisCircuitQuantumElectrodynamics2021`.
# The ansatz is
# [`LogStateVector`](https://netket.readthedocs.io/en/latest/api/_generated/models/netket.models.LogStateVector.html),
# one complex parameter $\log\psi(\sigma)$ per basis state, so we can check
# TDVP against direct exponentiation.  Two caveats come with the log
# parametrisation: no amplitude may pass through zero, which is why we detune
# (on resonance the qubit amplitude $\cos(2\pi g t)$ hits zero and the
# evolution diverges), and every unoccupied state needs a small finite
# amplitude to start from.  Swap in any other Flax network and the same driver
# evolves it.

# %%
n_levels_rabi = 4
hi_rabi = nk.hilbert.Fock(n_max=n_levels_rabi - 1, N=2)
a_op = np.diag(np.sqrt(np.arange(1, n_levels_rabi)), k=1)  # lowering operator
detuning_rabi = g_val  # qubit-resonator detuning, in GHz

H_rabi = (
    2
    * np.pi
    * (
        nk.operator.LocalOperator(
            hi_rabi,
            np.diag([
                detuning_rabi * n + float(alpha_nk) / 2 * n * (n - 1)
                for n in range(n_levels_rabi)
            ]),
            acting_on=[0],
        )
        + nk.operator.LocalOperator(
            hi_rabi,
            g_val * (np.kron(a_op, a_op.T) + np.kron(a_op.T, a_op)),
            acting_on=[0, 1],
        )
    )
)  # angular frequency in rad/ns, so time is in ns
n_qubit_op = nk.operator.LocalOperator(
    hi_rabi, np.diag(np.arange(n_levels_rabi, dtype=float)), acting_on=[0]
)

# Start in |1, 0>; a small floor keeps every log-amplitude finite.
psi_initial = np.full(hi_rabi.n_states, 1e-3, dtype=complex)
psi_initial[hi_rabi.states_to_numbers(np.array([[1, 0]]))[0]] = 1.0
psi_initial /= np.linalg.norm(psi_initial)

vs_rabi = nk.vqs.FullSumState(
    hi_rabi, nk.models.LogStateVector(hi_rabi, param_dtype=complex)
)
vs_rabi.parameters = {"logstate": jnp.log(jnp.asarray(psi_initial))}

t_final = 1.5 / g_val  # in ns
t_points = np.linspace(0.0, t_final, 121)
tdvp = nkx.TDVP(
    H_rabi,
    variational_state=vs_rabi,
    ode_solver=nkx.dynamics.RK4(dt=0.005),
    # log ψ is the parameter itself, so the ansatz is holomorphic
    # pyrefly: ignore[bad-argument-type]  # NetKet accepts a QGT constructor here
    qgt=partial(nk.optimizer.qgt.QGTJacobianDense, holomorphic=True),
)
tdvp_log = nk.logging.RuntimeLog()
tdvp.run(
    T=t_final,
    out=tdvp_log,
    obs={"n_qubit": n_qubit_op},
    tstops=t_points,
    show_progress=False,
)

# Reference: exact propagation of the same initial state
H_rabi_dense = H_rabi.to_dense()
n_qubit_dense = n_qubit_op.to_dense()
n_qubit_exact = [
    float(np.real(np.vdot(psi_t, n_qubit_dense @ psi_t)))
    for psi_t in (
        scipy.linalg.expm(-1j * H_rabi_dense * t) @ psi_initial for t in t_points
    )
]

n_qubit_tdvp = np.asarray(tdvp_log["n_qubit"]["Mean"].real)
print(
    f"Max |TDVP - exact| in qubit population: "
    f"{np.max(np.abs(n_qubit_tdvp - n_qubit_exact)):.1e}"
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_points, n_qubit_exact, label="Exact propagation")
ax.plot(
    np.asarray(tdvp_log["n_qubit"].iters),
    n_qubit_tdvp,
    "--",
    label="NetKet TDVP",
)
ax.set_xlabel("Time (ns)")
ax.set_ylabel(r"Qubit population $\langle \hat{n}_\text{q} \rangle$")
ax.set_title(
    rf"Vacuum Rabi oscillations, $g = \delta = {g_val * 1e3:.0f}\,\text{{MHz}}$"
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
# 2. **Select $\omega_\text{q}$, $\omega_\text{r}$, $\alpha$** from qubit design goals
# 3. **Determine $g$** from target $\chi$ using qpdk helpers
# 4. **Convert to circuit parameters** ($C_\Sigma$, $L_\text{J}$, $C_\text{c}$)
# 5. **Convert to layout parameters** (resonator length, capacitor geometry)
#
# ![Design targets give the coupling g, which gives the circuit parameters, which give the layout geometry](figures/qubit-design-workflow.svg)
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
\text{{Target:}}\quad \chi = {chi_target_mhz:.1f}\,\text{{MHz}} \\
E_\text{{J}} = {EJ_design:.1f}\,\text{{GHz}}, \quad
E_\text{{C}} = {EC_design:.3f}\,\text{{GHz}} \\
\omega_\text{{q}} = {omega_t_design:.3f}\,\text{{GHz}}, \quad
\alpha \approx -E_\text{{C}} = -{EC_design:.3f}\,\text{{GHz}} \\
\omega_\text{{r}} = {omega_r_design:.1f}\,\text{{GHz}} \\
\text{{Required coupling:}}\quad g = {float(g_design) * 1e3:.1f}\,\text{{MHz}}
""")
)

# %% [markdown]
# ### Step 4: Convert to circuit parameters
#
# Using the qpdk helper functions, we convert the Hamiltonian parameters to
# circuit parameters:
# - Total qubit capacitance $C_\Sigma$ from $E_\text{C}$ via `ec_to_capacitance`
# - Josephson inductance $L_\text{J}$ from $E_\text{J}$ via `ej_to_inductance`
# - Coupling capacitance $C_\text{c}$ from $g$ via `coupling_strength_to_capacitance`

# %%
# Total qubit capacitance
C_sigma = float(ec_to_capacitance(EC_design))

# Josephson inductance
L_J = float(ej_to_inductance(EJ_design))

# Determine resonator length for the target frequency
ep_eff, z0 = cpw_parameters(width=10, gap=6)


def _resonator_objective(x: np.ndarray) -> float:
    """Minimise the squared frequency error.

    Returns:
        The squared error between the calculated and design frequency.
    """
    freq = resonator_frequency(
        length=float(x[0]),
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
C_\Sigma = {C_sigma * 1e15:.1f}\,\text{{fF}} \\
L_\text{{J}} = {L_J * 1e9:.2f}\,\text{{nH}} \\
C_\text{{r}} = {C_r * 1e15:.1f}\,\text{{fF}} \\
C_\text{{c}} = {C_c * 1e15:.2f}\,\text{{fF}}
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
\text{{Resonator length:}}\quad l = {resonator_length:.0f}\,\text{{µm}} \\
\text{{Resonator CPW width:}}\quad w = 10\,\text{{µm}}, \quad
\text{{gap}} = 6\,\text{{µm}} \\
\text{{Resonator frequency achieved:}}\quad
f_\text{{r}} = {f_resonator_achieved / 1e9:.3f}\,\text{{GHz}}
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
\kappa = \omega_\text{{r}} / Q_\text{{ext}} = {kappa * 1e6:.1f}\,\text{{kHz}} \\
\gamma_\text{{Purcell}} = \kappa (g/\Delta)^2
  = {gamma_purcell * 1e6:.2f}\,\text{{kHz}}
  \quad \Rightarrow \quad T_\text{{Purcell}}
  = {T_purcell * 1e6:.0f}\,\text{{µs}} \\
\Gamma_\phi(\bar{{n}}={n_bar:.0f})
  = 8\chi^2 \bar{{n}} / \kappa
  = {gamma_phi * 1e6:.2f}\,\text{{kHz}} \\
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
# $E_\text{J}$, $E_\text{C}$, and $g$ through their effect on the capacitance and
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
# | Pad dimensions, gap width | $C_\Sigma$ (total capacitance) | $E_\text{C} = e^2 / (2 C_\Sigma)$ |
# | Junction area, critical current | $L_\text{J}$ (Josephson inductance) | $E_\text{J} = \Phi_0^2 / (4\pi^2 L_\text{J})$ |
# | Coupling finger length / gap | $C_\text{c}$ (coupling capacitance) | $g \propto C_\text{c} / \sqrt{C_\Sigma C_\text{r}}$ |
# | CPW length, width, gap | $Z_0$, $\varepsilon_\text{eff}$ | $\omega_\text{r}$ (resonator frequency) |

# %% [markdown]
# ## Summary: Complete Design Table
#
# The table below summarises the full design flow from Hamiltonian
# parameters to layout parameters, with the dispersive shift as the
# central design target.

# %%
data = [
    ("Target", "Target dispersive shift $\\chi$", f"{chi_target_mhz:.1f}", "MHz"),
    ("Hamiltonian", "$E_\\text{J}$", f"{EJ_design:.1f}", "GHz"),
    ("Hamiltonian", "$E_\\text{C}$", f"{EC_design:.3f}", "GHz"),
    ("Hamiltonian", "$\\omega_\\text{q}$ (NetKet)", f"{f01:.4f}", "GHz"),
    ("Hamiltonian", "$\\omega_\\text{q}$ (approx.)", f"{omega_t_design:.4f}", "GHz"),
    ("Hamiltonian", "$\\alpha$ (NetKet)", f"{alpha_nk:.4f}", "GHz"),
    ("Hamiltonian", "$\\omega_\\text{r}$", f"{omega_r_design:.1f}", "GHz"),
    ("Hamiltonian", "$g$", f"{float(g_design) * 1e3:.1f}", "MHz"),
    ("Circuit", "$C_\\Sigma$", f"{C_sigma * 1e15:.1f}", "fF"),
    ("Circuit", "$L_\\text{J}$", f"{L_J * 1e9:.2f}", "nH"),
    ("Circuit", "$C_\\text{r}$", f"{C_r * 1e15:.1f}", "fF"),
    ("Circuit", "$C_\\text{c}$", f"{C_c * 1e15:.2f}", "fF"),
    ("Layout", "Resonator length", f"{resonator_length:.0f}", "µm"),
    ("Layout", "Resonator CPW width", "10", "µm"),
    ("Layout", "Resonator CPW gap", "6", "µm"),
    ("Layout", "Resonator freq", f"{f_resonator_achieved / 1e9:.3f}", "GHz"),
    ("Readout", "$Q_{\\text{ext}}$", f"{Q_ext:,}", ""),
    ("Readout", "$\\kappa$", f"{kappa * 1e6:.1f}", "kHz"),
    ("Readout", "$T_{\\text{Purcell}}$", f"{T_purcell * 1e6:.0f}", "µs"),
    ("Readout", "$|\\chi|/\\kappa$", f"{chi_over_kappa:.1f}", ""),
]

df = pl.DataFrame(data, schema=["Category", "Parameter", "Value", "Unit"], orient="row")

display_dataframe(df)

# %% [markdown]
# ## Josephson Junction Arrays: a Many-Body Problem
#
# A transmon has one degree of freedom, so exact diagonalisation is all it
# needs.  A chain of $N$ junctions, such as the superinductor of a fluxonium,
# has one per island, and the Hilbert space grows as $(2n_\text{cut}+1)^{N-1}$.
# For the 43-junction
# {func}`~qpdk.cells.josephson_junction_array` cell that is $11^{42}$ states,
# far beyond diagonalisation.  This is where NetKet's Monte Carlo variational
# methods earn their place.
#
# With both ends of the chain held at zero phase, the islands
# $i = 1, \dots, N-1$ carry Cooper-pair numbers $\hat{n}_i$ and phases
# $\hat{\varphi}_i$, and
#
# ```{math}
# :label: eq:jja-hamiltonian-netket
# \hat{H} = \hat{\boldsymbol{n}}^\mathsf{T} \mathbf{E}_\text{C}\, \hat{\boldsymbol{n}}
#   - E_\text{J} \sum_{j=1}^{N} \cos\left(\hat{\varphi}_j - \hat{\varphi}_{j-1}\right),
# \qquad
# \mathbf{E}_\text{C} = 2 e^2 \mathbf{C}^{-1},
# ```
#
# with $\hat{\varphi}_0 = \hat{\varphi}_N = 0$ and the tridiagonal capacitance
# matrix $C_{ii} = 2 C_\text{J} + C_\text{g}$, $C_{i,i\pm1} = -C_\text{J}$.
# Because $C_\text{g} \ll C_\text{J}$, $\mathbf{C}^{-1}$ is dense: every island
# talks to every other one through the charging term.  $\cos(\hat{\varphi}_j -
# \hat{\varphi}_{j-1})$ moves one Cooper pair between neighbouring islands.
#
# The helper below reads $N$, the junction area and $I_\text{c}$ straight from
# the cell's `info`, so it works for any variant of the cell.  The cell does not
# report capacitances, so we assume a specific junction capacitance of
# $50\,\text{fF}/\mu\text{m}^2$, typical of Al/AlOx junctions, and take
# $C_\text{g}$ from the default of the array's SAX model
# {func}`~qpdk.models.junction.josephson_junction_array`.  Replace both with
# values for your own process.

# %%
JUNCTION_SPECIFIC_CAPACITANCE = 50e-15  # F/µm², assumed Al/AlOx value
ISLAND_GROUND_CAPACITANCE = 0.04e-15  # F, SAX model default


@dataclass(frozen=True)
class JunctionArrayProblem:
    """Circuit data and NetKet operators of a junction array, energies in GHz."""

    n_junctions: int
    ic: float  # critical current per junction, A
    c_j: float  # junction capacitance, F
    c_g: float  # island ground capacitance, F
    e_j: float
    charging: np.ndarray  # E_C matrix, H_C = nᵀ E_C n
    incidence: np.ndarray  # junction j measures φ_j - φ_{j-1}
    n_cut: int
    hilbert: nk.hilbert.Fock
    hamiltonian: nk.operator.LocalOperator
    cos_ops: list[nk.operator.LocalOperator]


def junction_array_hamiltonian(
    component,
    *,
    specific_capacitance: float = JUNCTION_SPECIFIC_CAPACITANCE,
    ground_capacitance: float = ISLAND_GROUND_CAPACITANCE,
    n_cut: int = 8,
) -> JunctionArrayProblem:
    """Build the charge-basis Hamiltonian of a ``josephson_junction_array`` cell.

    Energies are in GHz.  Each island is a ``Fock`` site whose occupation
    ``k`` encodes the Cooper-pair number ``n = k - n_cut``.  The ground-state
    charge spread per island is about 1.4 Cooper pairs for the default cell,
    so ``n_cut = 8`` keeps the truncation error well below the VMC error bars.

    Returns:
        The array parameters, NetKet Hilbert space, Hamiltonian and per-junction
        ``cos δφ`` operators.
    """
    n_junctions = int(component.info["n_junctions"])
    ic = component.info["critical_current_nA"] * 1e-9
    c_j = specific_capacitance * component.info["junction_area"]
    n_islands = n_junctions - 1

    e_j = ic * Φ_0 / (2 * np.pi) / h / 1e9
    capacitance = (2 * c_j + ground_capacitance) * np.eye(n_islands) - c_j * (
        np.eye(n_islands, k=1) + np.eye(n_islands, k=-1)
    )
    charging = 2 * e**2 * np.linalg.inv(capacitance) / h / 1e9

    hilbert = nk.hilbert.Fock(n_max=2 * n_cut, N=n_islands)
    n_local = np.diag(np.arange(-n_cut, n_cut + 1, dtype=float))
    raise_local = np.eye(2 * n_cut + 1, k=-1)  # |n+1><n|, i.e. exp(i φ)
    cos_ground = (raise_local + raise_local.T) / 2
    cos_pair = (
        np.kron(raise_local, raise_local.T) + np.kron(raise_local.T, raise_local)
    ) / 2

    hamiltonian = nk.operator.LocalOperator(hilbert, dtype=float)
    for i in range(n_islands):
        hamiltonian += nk.operator.LocalOperator(
            hilbert, charging[i, i] * n_local @ n_local, [i]
        )
        for j in range(i + 1, n_islands):
            hamiltonian += nk.operator.LocalOperator(
                hilbert, 2 * charging[i, j] * np.kron(n_local, n_local), [i, j]
            )
    # cos(φ_j - φ_{j-1}) for every junction j, the two end ones to ground
    cos_ops = [nk.operator.LocalOperator(hilbert, cos_ground, [0])]
    cos_ops += [
        nk.operator.LocalOperator(hilbert, cos_pair, [i, i + 1])
        for i in range(n_islands - 1)
    ]
    cos_ops += [nk.operator.LocalOperator(hilbert, cos_ground, [n_islands - 1])]
    for cos_op in cos_ops:
        hamiltonian -= e_j * cos_op

    incidence = np.eye(n_junctions, n_islands) - np.eye(n_junctions, n_islands, k=-1)
    return JunctionArrayProblem(
        n_junctions=n_junctions,
        ic=ic,
        c_j=c_j,
        c_g=ground_capacitance,
        e_j=e_j,
        charging=charging,
        incidence=incidence,
        n_cut=n_cut,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        cos_ops=cos_ops,
    )


def linear_modes(problem: JunctionArrayProblem, cos_mean=1.0):
    """Normal-mode frequencies and Gaussian ground state of the linearised array.

    Linearising ``-E_J cos δφ`` around each junction's ``<cos δφ>`` gives
    ``H = nᵀ E_C n + ½ φᵀ B φ`` with ``B = Dᵀ diag(E_J <cos>) D``.  Returns the
    mode frequencies in GHz and the charge-space precision matrix ``K`` of the
    ground state ``ψ(n) ∝ exp(-½ nᵀ K n)``.

    Returns:
        Mode frequencies in GHz and the charge-space precision matrix ``K``.
    """
    stiffness = (
        problem.incidence.T
        @ np.diag(problem.e_j * np.broadcast_to(cos_mean, problem.n_junctions))
        @ problem.incidence
    )
    charging_sqrt = scipy.linalg.sqrtm(2 * problem.charging).real
    charging_sqrt_inv = np.linalg.inv(charging_sqrt)
    frequencies = np.sqrt(np.linalg.eigvalsh(charging_sqrt @ stiffness @ charging_sqrt))
    # Phase-space precision G solves G (2 E_C) G = B; the charge-space one is G⁻¹
    phase_precision = (
        charging_sqrt_inv
        @ scipy.linalg.sqrtm(charging_sqrt @ stiffness @ charging_sqrt).real
        @ charging_sqrt_inv
    )
    return frequencies, np.linalg.inv(phase_precision)


# %% [markdown]
# ### Checking the Hamiltonian on a short array
#
# A four-junction variant of the cell has $17^3 = 4913$ states, which NetKet
# diagonalises exactly.  Its gap is the lowest normal mode of the chain.
# Linearising every junction as an inductor $L_\text{J} = \Phi_0 / 2\pi
# I_\text{c}$, as the SAX model does, overestimates that gap.  The
# self-consistent harmonic correction replaces $E_\text{J}$ by $E_\text{J}
# \langle \cos\delta\hat{\varphi} \rangle$, the junction's stiffness averaged
# over its zero-point phase fluctuations, and brings the linear model to
# within about 15 MHz of the full Hamiltonian.

# %%
short_array = junction_array_hamiltonian(josephson_junction_array(n_junctions=4))
ed_values, ed_vectors = nk.exact.lanczos_ed(
    short_array.hamiltonian, k=2, compute_eigenvectors=True
)
ed_ground = ed_vectors[:, 0]
cos_exact = np.array([
    ed_ground @ (cos_op.to_sparse() @ ed_ground) for cos_op in short_array.cos_ops
])
f_linear, _ = linear_modes(short_array)
f_renormalised, _ = linear_modes(short_array, cos_exact)

display(
    Math(rf"""
\textbf{{Four-junction array}} \quad (E_\text{{J}} = {short_array.e_j:.1f}\,\text{{GHz}},
\ C_\text{{J}} = {short_array.c_j * 1e15:.0f}\,\text{{fF}}) \\
\text{{Exact gap (NetKet ED):}} \quad {ed_values[1] - ed_values[0]:.3f}\,\text{{GHz}} \\
\text{{Lowest linear mode, }} E_\text{{J}}: \quad {f_linear[0]:.3f}\,\text{{GHz}} \\
\text{{Lowest linear mode, }} E_\text{{J}} \langle\cos\delta\varphi\rangle: \quad
  {f_renormalised[0]:.3f}\,\text{{GHz}}
  \quad (\langle\cos\delta\varphi\rangle = {cos_exact.mean():.3f})
""")
)

# %% [markdown]
# ### Variational Monte Carlo with a custom Flax network
#
# For the full cell we sample instead of summing.  The ansatz is a Flax module
# with two parts:
#
# - a Gaussian $-\tfrac{1}{2}\boldsymbol{n}^\mathsf{T} \mathbf{K} \boldsymbol{n}$
#   whose precision matrix $\mathbf{K}$ starts at the exact ground state of the
#   linearised circuit, the one `linear_modes` returns, so the
#   optimisation begins from the physics the SAX model already contains;
# - a one-hidden-layer network that learns the anharmonic correction, starting
#   from zero.
#
# A cold start from random parameters puts amplitude on states with large net
# island charge, whose energies run to thousands of GHz, and the optimisation
# diverges.
#
# [`MCState`](https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html)
# draws configurations with a Metropolis sampler.  We give it a custom
# transition rule, `ChargeHopRule`, that proposes the moves the junctions
# themselves make: one Cooper pair onto or off an island, or across to the
# next one.
# [`VMC_SR`](https://netket.readthedocs.io/en/latest/api/_generated/driver/netket.driver.VMC_SR.html)
# preconditions the gradient with stochastic reconfiguration, and for more
# parameters than samples it switches to the kernel (NTK) form automatically
# {cite:p}`netket3:2022`.  We first check it against the exact energy of the
# four-junction array.


# %%
class ChargeGaussianNet(nn.Module):
    """log ψ(n) = -½ nᵀ K n + MLP(n), with K initialised to ``precision``."""

    n_cut: int
    precision: tuple[tuple[float, ...], ...]
    hidden: int = 16
    param_dtype: Any = float

    @nn.compact
    def __call__(self, x):
        """Return ``log ψ`` for a batch of island occupations ``x``."""
        n = x - self.n_cut
        k_matrix = self.param(
            "K", lambda _: jnp.asarray(self.precision, dtype=self.param_dtype)
        )
        k_matrix = (k_matrix + k_matrix.T) / 2
        gaussian = -0.5 * jnp.einsum("...i,ij,...j->...", n, k_matrix, n)
        y = nn.Dense(
            self.hidden,
            kernel_init=nn.initializers.normal(1e-2),
            param_dtype=self.param_dtype,
        )(n)
        y = nn.Dense(
            1, kernel_init=nn.initializers.zeros, param_dtype=self.param_dtype
        )(jnp.tanh(y))
        return gaussian + y[..., 0]


class ChargeHopRule(nk.sampler.rules.MetropolisRule):
    """Propose one Cooper pair added to or removed from an island, or moved to its neighbour.

    These are the moves the junctions make.  NetKet's default ``LocalRule``
    redraws a site uniformly from all ``2 n_cut + 1`` charges, which this
    strongly correlated chain almost always rejects.
    """

    def transition(self, sampler, machine, params, sampler_state, key, σ):
        """Return one proposed move per chain; out-of-range moves stay put."""
        key_site, key_step, key_kind = jax.random.split(key, 3)
        n_chains, n_sites = σ.shape
        rows = jnp.arange(n_chains)
        sites = jax.random.randint(key_site, (n_chains,), 0, n_sites)
        steps = jax.random.choice(key_step, jnp.array([-1, 1]), (n_chains,)).astype(
            σ.dtype
        )
        hop = jax.random.bernoulli(key_kind, 0.5, (n_chains,)) & (sites < n_sites - 1)
        σ_new = σ.at[rows, sites].add(steps)
        σ_new = σ_new.at[rows, jnp.minimum(sites + 1, n_sites - 1)].add(-steps * hop)
        # Proposals that leave the truncated charge range stay put
        n_max = sampler.hilbert.shape[0] - 1
        in_range = jnp.all((σ_new >= 0) & (σ_new <= n_max), axis=1, keepdims=True)
        return jnp.where(in_range, σ_new, σ), None


def run_array_vmc(
    problem: JunctionArrayProblem,
    *,
    n_iter: int = 100,
    n_samples: int = 1024,
    n_chains: int = 16,
    hidden: int = 16,
    learning_rate: float = 0.01,
    seed: int = 0,
):
    """Optimise ``ChargeGaussianNet`` with VMC_SR and return the state and log."""
    _, precision = linear_modes(problem)
    model = ChargeGaussianNet(
        n_cut=problem.n_cut,
        precision=tuple(map(tuple, precision.tolist())),
        hidden=hidden,
        param_dtype=jnp.result_type(float),  # float32 unless x64 is enabled
    )
    sampler = nk.sampler.MetropolisSampler(
        problem.hilbert,
        ChargeHopRule(),
        n_chains=n_chains,
        sweep_size=4 * problem.hilbert.size,
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
    driver = nk.driver.VMC_SR(
        problem.hamiltonian.to_jax_operator(),
        optax.sgd(learning_rate),
        variational_state=vstate,
        diag_shift=1e-2,
    )
    log = nk.logging.RuntimeLog()
    driver.run(n_iter, out=log, show_progress=False)
    return vstate, log


vs_short, _ = run_array_vmc(short_array)
vs_short.n_samples = 8192
e_short_vmc = vs_short.expect(short_array.hamiltonian.to_jax_operator())

display(
    Math(rf"""
\textbf{{Four-junction array, ground-state energy}} \\
\text{{NetKet ED:}} \quad {ed_values[0]:.3f}\,\text{{GHz}} \\
\text{{VMC\_SR + Flax:}} \quad {e_short_vmc.mean.real:.3f}
  \pm {e_short_vmc.error_of_mean:.3f}\,\text{{GHz}}
""")
)

# %% [markdown]
# Now the default 43-junction cell, with 42 coupled islands.

# %%
array_cell = josephson_junction_array()
array_problem = junction_array_hamiltonian(array_cell)
vs_array, array_log = run_array_vmc(array_problem)

vs_array.n_samples = 8192
e_array_vmc = vs_array.expect(array_problem.hamiltonian.to_jax_operator())
cos_array = np.array([
    vs_array.expect(cos_op.to_jax_operator()).mean.real
    for cos_op in array_problem.cos_ops
])
f_array_linear, _ = linear_modes(array_problem)
# The network's output layer starts at zero, so the untrained state is the
# harmonic Gaussian; evaluate it on the full cosine Hamiltonian as a baseline.
vs_gaussian, _ = run_array_vmc(array_problem, n_iter=0)
vs_gaussian.n_samples = 8192
e_array_gaussian = vs_gaussian.expect(array_problem.hamiltonian.to_jax_operator())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(np.asarray(array_log["Energy"]["Mean"].real), lw=1)
ax1.axhline(
    e_array_gaussian.mean.real, color="k", ls="--", label="Harmonic Gaussian start"
)
ax1.set_xlabel("VMC_SR iteration")
ax1.set_ylabel("Energy (GHz)")
ax1.set_title(
    f"{array_problem.n_junctions}-junction array, {array_problem.n_junctions - 1} islands"
)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(np.arange(1, array_problem.n_junctions + 1), cos_array, "o-", ms=3)
ax2.set_xlabel("Junction index $j$")
ax2.set_ylabel(r"$\langle \cos\delta\hat{\varphi}_j \rangle$")
ax2.set_title("Phase fluctuations along the chain")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

display(
    Math(rf"""
E_0^\text{{VMC}} = {e_array_vmc.mean.real:.2f} \pm {e_array_vmc.error_of_mean:.2f}\,\text{{GHz}},
\qquad E_\text{{Gaussian}} = {e_array_gaussian.mean.real:.2f}
  \pm {e_array_gaussian.error_of_mean:.2f}\,\text{{GHz}},
\qquad \overline{{\langle\cos\delta\varphi\rangle}} = {cos_array.mean():.3f}
""")
)

# %% [markdown]
# VMC is variational, so the true ground state lies at or below its energy.
# The optimisation starts from the harmonic ground state, and most of the
# work is already done there: the Gaussian alone captures the bulk of the
# energy, and the network and the tuned $K$ lower it by a further fraction of
# a GHz.  Lacking ED at this size, that drop and the four-junction comparison
# above are the evidence the ansatz is adequate.
# $\langle\cos\delta\hat{\varphi}\rangle$
# is nearly uniform along the chain, so we feed its average into the SAX
# model below.
#
# ### Feeding the result back into the SAX model
#
# The array's SAX model {func}`~qpdk.models.junction.josephson_junction_array`
# treats each junction as a linear inductor set by `ic`.  Passing
# $I_\text{c} \langle\cos\delta\hat{\varphi}\rangle$ instead gives the
# self-consistent harmonic model of the same layout: the junctions are softer,
# the chain's inductance rises and its self-resonances move down.

# %%
array_freqs = np.linspace(9e9, 12.5e9, 2001)
s21_bare = josephson_junction_array_model(
    f=array_freqs,
    n_junctions=array_problem.n_junctions,
    ic=array_problem.ic,
    capacitance=array_problem.c_j,
    ground_capacitance=array_problem.c_g,
)["o2", "o1"]
s21_dressed = josephson_junction_array_model(
    f=array_freqs,
    n_junctions=array_problem.n_junctions,
    ic=array_problem.ic * cos_array.mean(),
    capacitance=array_problem.c_j,
    ground_capacitance=array_problem.c_g,
)["o2", "o1"]
f_array_dressed, _ = linear_modes(array_problem, cos_array.mean())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    array_freqs / 1e9,
    20 * np.log10(np.abs(s21_bare)),
    label=r"$I_\text{c}$ (SAX default)",
)
ax.plot(
    array_freqs / 1e9,
    20 * np.log10(np.abs(s21_dressed)),
    label=r"$I_\text{c}\,\langle\cos\delta\varphi\rangle$ (from VMC)",
)
for f_mode in f_array_dressed[:3]:
    ax.axvline(f_mode, color="C1", ls=":", lw=1)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$|S_{21}|$ (dB)")
ax.set_ylim(-60, 5)  # the lossless chain has transmission zeros at -∞ dB
ax.set_title(f"{array_problem.n_junctions}-junction array between 50 Ω ports")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

display(
    Math(rf"""
\text{{Lowest chain mode:}} \quad {f_array_linear[0]:.2f}\,\text{{GHz}}
\ \to\ {f_array_dressed[0]:.2f}\,\text{{GHz}}, \qquad
L_\text{{array}} = {array_cell.info["array_inductance_nH"]:.0f}\,\text{{nH}}
\ \to\ {array_cell.info["array_inductance_nH"] / cos_array.mean():.0f}\,\text{{nH}}
""")
)

# %% [markdown]
# The dotted lines are the first renormalised normal modes of the chain with
# grounded ends, from `linear_modes`.  Between 50 Ω ports the chain is
# loaded, so the $|S_{21}|$ features sit near, not exactly on, these lines.
#
# ## Scaling up on a TPU
#
# ::::{admonition} Running this section on a TPU in Google Colab
# :class: note
#
# The variational calculation above is plain JAX, so NetKet runs it on
# whatever accelerator JAX finds, without code changes.  To try it on a TPU:
#
# 1. Open the notebook in Colab with the button at the top of the page.
# 2. Choose **Runtime → Change runtime type**, select a **TPU** hardware
#    accelerator (`v2-8` or `v5e-1` on the free tier) and save.
# 3. Run all cells.  The install cell detects the TPU, reinstalls the TPU
#    build of JAX that matches qpdk's pinned version, and switches NetKet to
#    single precision with `NETKET_ENABLE_X64=0` before NetKet is imported.
#
# TPUs are available on Colab's **free tier**, but not guaranteed: access
# depends on demand and on your recent usage, and sessions are time-limited.
# If no TPU is available, pick a GPU or keep the CPU runtime; the cell below
# then runs a short version of the same calculation.
#
# TPUs compute in single precision, which is why the install cell disables
# float64.  The rest of the notebook still runs in that mode, but its numbers
# can differ from the documentation in the last few digits.  The TPU path has
# not been verified by the qpdk continuous integration, which only has CPUs.
# ::::
#
# On a TPU the same driver can afford far more samples and a wider network,
# which reduces the Monte Carlo error and the variational bias in the
# 43-junction calculation.

# %%
backend = jax.default_backend()
on_accelerator = backend in {"tpu", "gpu"}
print(f"JAX backend: {backend}, devices: {jax.devices()}")

tpu_settings = (
    {"n_iter": 300, "n_samples": 16384, "n_chains": 1024, "hidden": 64}
    if on_accelerator
    else {"n_iter": 20, "n_samples": 1024, "n_chains": 16, "hidden": 16}
)
start = time.perf_counter()
vs_large, large_log = run_array_vmc(array_problem, **tpu_settings)
elapsed = time.perf_counter() - start

vs_large.n_samples = 65536 if on_accelerator else 8192
e_large = vs_large.expect(array_problem.hamiltonian.to_jax_operator())
print(
    f"{tpu_settings['n_iter']} VMC_SR iterations with {tpu_settings['n_samples']} samples "
    f"on {backend}: {elapsed:.1f} s ({elapsed / tpu_settings['n_iter'] * 1e3:.0f} ms/iteration)"
)
print(f"E0 = {e_large.mean.real:.2f} ± {e_large.error_of_mean:.2f} GHz")
if not on_accelerator:
    print("No accelerator found: ran a short CPU version. See the admonition above.")

# %% [markdown]
# ## Unimon and Gatemon Hamiltonians as NetKet Operators
#
# qpdk's qubit models already build their Hamiltonian matrices:
# {func}`~qpdk.models.unimon.unimon_hamiltonian` in the harmonic-oscillator
# basis and {func}`~qpdk.models.gatemon.gatemon_hamiltonian` in the charge
# basis.  Any such matrix becomes a NetKet operator on a single `Fock` site, so
# it can be diagonalised, coupled to a resonator mode as in the
# transmon–resonator section, or evolved in time with TDVP.


# %%
def matrix_to_netket(matrix) -> nk.operator.LocalOperator:
    """Wrap a qpdk Hamiltonian matrix as a NetKet operator on one Fock site."""
    matrix = np.asarray(matrix)
    hilbert_1 = nk.hilbert.Fock(n_max=matrix.shape[0] - 1, N=1)
    return nk.operator.LocalOperator(hilbert_1, matrix, acting_on=[0])


def netket_frequency_and_anharmonicity(operator) -> tuple[float, float]:
    """f01 and α from NetKet's exact diagonalisation."""
    levels = nk.exact.lanczos_ed(operator, k=3)
    return float(levels[1] - levels[0]), float(levels[2] - 2 * levels[1] + levels[0])


# %% [markdown]
# For the **gatemon**, $C_\Sigma$ comes from the Palace capacitance lookup
# {func}`~qpdk.models.gatemon.gatemon_capacitances` evaluated at the default
# {func}`~qpdk.cells.gatemon` cell's dimensions, and the junction is the
# model's default carbon-nanotube weak link.  The **unimon** has no
# layout-to-$E_\text{C}$, $E_\text{L}$ model in qpdk yet, so it uses the
# model's default energies.

# %%
gatemon_cell = gatemon()
gatemon_settings = gatemon_cell.settings
C_sigma_gatemon, _, _ = gatemon_capacitances(
    island_size=gatemon_settings["island_size"][0],
    pocket_gap=gatemon_settings["pocket_gap"],
    pocket_gap_top=gatemon_settings["pocket_gap_top"],
    gate_tip_gap=gatemon_settings["gate_tip_gap"],
)
EC_gatemon = float(e**2 / (2 * C_sigma_gatemon) / h / 1e9)

qubit_checks = {
    "Gatemon": (
        matrix_to_netket(gatemon_hamiltonian(ec_ghz=EC_gatemon)),
        gatemon_frequency_and_anharmonicity(ec_ghz=EC_gatemon),
    ),
    "Unimon": (
        matrix_to_netket(unimon_hamiltonian()),
        unimon_frequency_and_anharmonicity(),
    ),
}
rows = []
for name, (operator, (f01_model, alpha_model)) in qubit_checks.items():
    f01_netket, alpha_netket = netket_frequency_and_anharmonicity(operator)
    rows.append((
        name,
        f"{float(f01_model):.4f}",
        f"{f01_netket:.4f}",
        f"{float(alpha_model) * 1e3:.1f}",
        f"{alpha_netket * 1e3:.1f}",
    ))

display(
    Math(
        rf"\text{{Gatemon: }} C_\Sigma = {float(C_sigma_gatemon) * 1e15:.1f}\,\text{{fF}}"
        rf" \Rightarrow E_\text{{C}} = {EC_gatemon * 1e3:.0f}\,\text{{MHz}}"
    )
)
display_dataframe(
    pl.DataFrame(
        rows,
        schema=[
            "Qubit",
            "f01 model (GHz)",
            "f01 NetKet (GHz)",
            "α model (MHz)",
            "α NetKet (MHz)",
        ],
        orient="row",
    )
)

# %% [markdown]
# ## Reusing NetKet in Other qpdk Flows
#
# The pattern above is the same for every component:
#
# 1. **Layout cell → circuit parameters.** Read them from the cell's `settings`
#    and `info`, or from a qpdk model of the cell: capacitance lookups such as
#    {func}`~qpdk.models.qubit.double_pad_transmon_capacitance` and
#    {func}`~qpdk.models.gatemon.gatemon_capacitances`, or the critical current
#    the junction cells report.
# 2. **Circuit parameters → NetKet operator.** Either an existing qpdk
#    Hamiltonian matrix through `matrix_to_netket`, or a many-body
#    `LocalOperator` such as `junction_array_hamiltonian`.
# 3. **Solve.** `nk.exact.lanczos_ed` for small spaces, `FullSumState` with
#    `VMC` or `TDVP` when the whole space fits in memory, `MCState` with
#    `VMC_SR` and a Flax network when it does not.
# 4. **Back into the SAX models.** Renormalised parameters, such as
#    $I_\text{c}\langle\cos\delta\hat{\varphi}\rangle$ here, plug into the
#    existing SAX models and from there into any qpdk circuit simulation.

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
