# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pulse-Level Simulation of Superconducting Qubits with QuTiP-QIP
#
# This notebook demonstrates how to use
# [qutip-qip](https://qutip-qip.readthedocs.io/)
# {cite:p}`liBoshlomQutipqipPulselevel2022` to perform
# **pulse-level simulations** of superconducting transmon qubits.
# It bridges the gap between the Hamiltonian-level design covered in the
# companion notebooks
# ([scQubits parameter calculation](scqubits_parameter_calculation)
# and [pymablock dispersive shift](pymablock_dispersive_shift))
# and the actual control pulses that drive quantum gates on hardware.
#
# We use [qutip-jax](https://qutip-jax.readthedocs.io/) as the JAX backend
# for QuTiP, leveraging the existing JAX integration in qpdk.
#
# ## Why Pulse-Level Simulation?
#
# Ideal quantum gate models assume instantaneous, perfect operations.
# In practice, gates on superconducting qubits are realised by shaped
# microwave pulses applied over tens to hundreds of nanoseconds.
# Pulse-level simulation captures the actual time dynamics of the system,
# including:
#
# - **Leakage** to non-computational states (the third level of the transmon)
# - **Gate errors** from finite pulse duration and bandwidth
# - **Decoherence** from $T_1$ relaxation and $T_2$ dephasing
#
# These effects are critical for designing high-fidelity quantum processors
# {cite:p}`krantzQuantumEngineersGuide2019,kjaergaardSuperconductingQubits2020`.
# For some recent examples of `qutip` pulse-level simulation in the context of superconducting
# qubits, see {cite:p}`salmenkiviMitigationCoherentErrors2023a,anderssonPulselevelSimulationsFermionicsimulation2024`.

# %% tags=["hide-input", "hide-output"]
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip
import qutip_jax  # noqa: F401 — registers JAX backend for QuTiP
from IPython.display import Math, display
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits

from qpdk import PDK
from qpdk.cells.transmon import xmon_transmon
from qpdk.models.perturbation import ej_ec_to_frequency_and_anharmonicity

qutip.settings.core["numpy_backend"] = jnp
qutip.settings.core["default_dtype"] = "jax"

PDK.activate()

# Suppress FutureWarnings from qutip internals
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# %% [markdown]
# ## From Layout to Simulation: Design Mapping
#
# In a real quantum PDK (qpdk), the physical layout parameters directly
# determine the Hamiltonian parameters used in the pulse-level simulation.
#
# ### Step 1: Layout Component
#
# We start with an **Xmon transmon** design. The sizes of arms
# and the gap between them determine the total shunt capacitance $C_\Sigma$,
# which sets the charging energy $E_C = e^2 / (2 C_\Sigma)$.

# %%
c_transmon = xmon_transmon(arm_width=[10.0] * 4, arm_lengths=[100.0] * 4)
c_transmon.plot()
plt.title("Xmon Transmon Layout (qpdk)")
plt.show()

# %% [markdown]
# ### Step 2: Parameter Extraction
#
# Changing the `arm_width` and `arm_length` in the layout modifies $E_C$.
# For a transmon, the Josephson energy $E_J$ is determined by the critical
# current of the junction.
#
# Below we see how a change in $E_C$ (e.g., from making the pads larger)
# affects the qubit frequency $\omega_q$ and anharmonicity $\alpha$.

# %%
# Fixed EJ = 15 GHz
EJ_val = 15.0

# EC = 0.3 GHz (standard pads) vs EC = 0.2 GHz (larger pads)
ec_values = [0.3, 0.2]

for ec in ec_values:
    wq_calc, alpha_calc = ej_ec_to_frequency_and_anharmonicity(EJ_val, ec)
    display(
        Math(rf"""
\text{{For }} E_C = {ec:.1f}\,\mathrm{{GHz}} \implies
\omega_q \approx {wq_calc:.3f}\,\mathrm{{GHz}}, \quad
\alpha \approx {alpha_calc:.3f}\,\mathrm{{GHz}}
""")
    )

# %% [markdown]
# **Design Trade-off:**
# - **Larger pads (Lower $E_C$)**: Reduces $\alpha$, which requires longer
#   pulses to avoid leakage to $|2\rangle$ (slower gates).
# - **Smaller pads (Higher $E_C$)**: Increases $\alpha$, allowing faster
#   gates, but makes the qubit more sensitive to charge noise.

# %% [markdown]
# ## Pulse-Level Simulation: Single-Qubit X Gate
#
# We now simulate an $X$ gate on a qubit with the extracted parameters.
# The `SCQubits` processor captures the time-domain pulse envelopes and
# the population dynamics.

# %%
# Using parameters extracted from layout (EC = 0.3 GHz, EJ = 15 GHz)
# QuTiP-QIP SCQubits expects negative anharmonicity for transmons,
# while the qpdk perturbation model returns positive values by convention.
wq0, a0 = ej_ec_to_frequency_and_anharmonicity(EJ_val, 0.3)
wq = [float(wq0), float(wq0) - 0.1]
alpha = [-float(a0), -float(a0)]

g_coupling = 0.1
wr = 6.5

qc_x = QubitCircuit(2)
qc_x.add_gate("X", targets=0)

processor_x = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
processor_x.load_circuit(qc_x)

init_state = qutip.basis([3, 3], [0, 0])
result_x = processor_x.run_state(init_state)

# %% [markdown]
# ### Population Dynamics and Leakage
#
# We track the population in each basis state. The 3-level transmon model
# reveals any **leakage** to the $|2\rangle$ state, which is heavily
# influenced by the anharmonicity $\alpha$ set in the layout.

# %%
# Define projection operators for population tracking
proj_ops = {
    r"$|0,0\rangle$": qutip.tensor(qutip.fock_dm(3, 0), qutip.fock_dm(3, 0)),
    r"$|1,0\rangle$": qutip.tensor(qutip.fock_dm(3, 1), qutip.fock_dm(3, 0)),
    r"$|2,0\rangle$": qutip.tensor(qutip.fock_dm(3, 2), qutip.fock_dm(3, 0)),
}

tlist_x = jnp.asarray(result_x.times)

fig, ax = plt.subplots(figsize=(8, 4))
for label, proj in proj_ops.items():
    pops = [qutip.expect(proj, s) for s in result_x.states]
    ax.plot(tlist_x, pops, linewidth=2, label=label)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Population")
ax.set_title("X gate: population dynamics with leakage tracking")
ax.legend(loc="center right")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Bell State Preparation
#
# Entanglement between two qubits is realized via a CNOT gate. In this
# processor, the CNOT is implemented using **Cross-Resonance (CR)** pulses.
# The efficiency of the CR interaction depends on the frequency detuning
# $\Delta = \omega_{q,1} - \omega_{q,2}$, which is set by the junction
# sizing in the layout.

# %%
qc_bell = QubitCircuit(2)
qc_bell.add_gate("SNOT", targets=0)  # Hadamard
qc_bell.add_gate("CNOT", controls=0, targets=1)

processor_bell = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
processor_bell.load_circuit(qc_bell)

result_bell = processor_bell.run_state(init_state)

# %%
fig, axes = processor_bell.plot_pulses(figsize=(10, 8))
fig.suptitle(
    r"Control pulses for Bell state preparation ($H + \mathrm{CNOT}$)",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Fidelity Analysis
#
# We compute the state fidelity with the ideal Bell state.

# %%
# Ideal Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
bell_ideal = (
    qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 0))
    + qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 1))
).unit()

final_state = result_bell.states[-1]
fidelity_bell = qutip.fidelity(final_state, bell_ideal)

display(
    Math(rf"""
\textbf{{Bell State Fidelity:}} \\
F(|\Phi^+\rangle) = {fidelity_bell:.6f}
""")
)

# %% [markdown]
# ## Decoherence Effects: $T_2$ Sweep
#
# Real qubits are limited by decoherence. If our layout has higher
# loss (e.g., from narrow gaps), the $T_1$ and $T_2$ times decrease,
# directly reducing the gate fidelity.

# %%
# Sweep T2 values (in ns) with fixed T1 = 80 μs
t1_fixed = 80_000.0  # 80 μs in ns
t2_values_us = jnp.array([5, 10, 20, 40, 80, 150])
t2_values_ns = t2_values_us * 1e3

fidelities_t2 = []
for t2 in t2_values_ns:
    proc = SCQubits(
        2, wq=wq, alpha=alpha, g=g_coupling, wr=wr, t1=t1_fixed, t2=float(t2)
    )
    proc.load_circuit(qc_bell)
    res = proc.run_state(init_state)
    f = qutip.fidelity(res.states[-1], bell_ideal)
    fidelities_t2.append(f)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(t2_values_us, fidelities_t2, "o-", linewidth=2, markersize=8)
ax.set_xlabel(r"$T_2$ ($\mu$s)")
ax.set_ylabel(r"Bell state fidelity $F$")
ax.set_title(rf"Bell state fidelity vs. $T_2$ ($T_1 = {t1_fixed / 1e3:.0f}\,\mu$s)")
ax.grid(True, alpha=0.3, which="both")
plt.show()

# %% [markdown]
# ## Summary: The Design Cycle
#
# This notebook demonstrated the connection between physical layout and
# pulse-level performance:
#
# 1. **Layout (qpdk)**: Geometric parameters (pad size, gap) determine $C_\Sigma$ and $L_J$.
# 2. **Hamiltonian (qpdk.models)**: Parameters $E_C, E_J$ set $\omega_q$ and $\alpha$.
# 3. **Simulation (QuTiP)**: Pulse-level dynamics reveal gate fidelity and leakage.
# 4. **Iterate**: If leakage is too high ($|2\rangle$ population), we go back to
#    step 1 and increase $E_C$ in the layout.
#
# For more details on parameter extraction, see the companion notebooks
# on [scQubits parameter calculation](scqubits_parameter_calculation).

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
