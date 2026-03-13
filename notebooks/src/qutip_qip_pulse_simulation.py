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
# - **Cross-talk** between neighbouring qubits
#
# These effects are critical for designing high-fidelity quantum processors
# {cite:p}`krantzQuantumEngineersGuide2019,kjaergaardSuperconductingQubits2020`.

# %% tags=["hide-input", "hide-output"]
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip
import qutip_jax  # noqa: F401 — registers JAX backend for QuTiP
from IPython.display import Math, display
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits

# Suppress FutureWarnings from qutip internals
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# %% [markdown]
# ## System Parameters: From Design to Simulation
#
# We use the same Hamiltonian parameters derived in the companion notebooks
# for a transmon qubit coupled to a readout resonator:
#
# | Parameter | Symbol | Value | Unit |
# |-----------|--------|-------|------|
# | Qubit frequency | $\omega_q$ | 5.15 | GHz |
# | Qubit anharmonicity | $\alpha$ | −0.3 | GHz |
# | Second qubit frequency | $\omega_{q,2}$ | 5.09 | GHz |
# | Resonator frequency | $\omega_r$ | 5.96 | GHz |
# | Qubit–resonator coupling | $g$ | 0.1 | GHz |
#
# The `SCQubits` processor in qutip-qip uses a multi-level Duffing oscillator
# model for each transmon (truncated to 3 levels by default), capturing leakage
# beyond the computational subspace
# {cite:p}`kochChargeinsensitiveQubitDesign2007a`.
# Two-qubit interactions use a cross-resonance (CR) effective Hamiltonian
# {cite:p}`magesanEffectiveHamiltonianModels2020`:
#
# ```{math}
# :label: eq:scqubits-hamiltonian
# H = H_{\rm d}
#   + \sum_j \Omega^x_j (a_j^\dagger + a_j)
#   + \Omega^y_j \, i(a_j^\dagger - a_j)
#   + \sum_j \Omega^{\rm cr}_j \, \sigma^z_j \sigma^x_{j+1}
# ```
# where the drift Hamiltonian is
# $H_{\rm d} = \sum_j \frac{\alpha_j}{2} a_j^{\dagger 2} a_j^{2}$.

# %%
# Physical parameters for a 2-qubit transmon chain
wq = [5.15, 5.09]  # Qubit frequencies (GHz)
alpha = [-0.3, -0.3]  # Anharmonicities (GHz)
g_coupling = 0.1  # Qubit-resonator coupling (GHz)
wr = 5.96  # Resonator frequency (GHz)

display(
    Math(rf"""
\textbf{{Processor Parameters}} \\
\omega_{{q,1}} = {wq[0]}\,\mathrm{{GHz}}, \quad
\omega_{{q,2}} = {wq[1]}\,\mathrm{{GHz}} \\
\alpha_1 = \alpha_2 = {alpha[0]}\,\mathrm{{GHz}} \\
\omega_r = {wr}\,\mathrm{{GHz}}, \quad
g = {g_coupling}\,\mathrm{{GHz}}
""")
)

# %% [markdown]
# ## Single-Qubit Gate: X Gate
#
# We start with a single $X$ gate (bit-flip) on qubit 0, which rotates the
# qubit state from $|0\rangle$ to $|1\rangle$.  The `SCQubits` processor
# decomposes this into calibrated microwave pulses on the
# $\sigma_x$ and $\sigma_y$ quadratures.

# %%
qc_x = QubitCircuit(2)
qc_x.add_gate("X", targets=0)

processor_x = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
processor_x.load_circuit(qc_x)

init_state = qutip.basis([3, 3], [0, 0])
result_x = processor_x.run_state(init_state)

# %% [markdown]
# ### Control Pulses
#
# The processor generates continuous pulse envelopes for each control
# Hamiltonian term.  For a single-qubit $X$ gate, we expect activity on the
# $\sigma_x$ and $\sigma_y$ channels of the target qubit.

# %%
fig, axes = processor_x.plot_pulses(figsize=(10, 6))
fig.suptitle("Control pulses for X gate on qubit 0", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Population Dynamics
#
# We track the population in each computational basis state as the
# pulse is applied.  The 3-level transmon model shows any leakage to the
# $|2\rangle$ state.

# %%
# Define projection operators for population tracking
proj_ops = {
    r"$|0,0\rangle$": qutip.tensor(qutip.fock_dm(3, 0), qutip.fock_dm(3, 0)),
    r"$|1,0\rangle$": qutip.tensor(qutip.fock_dm(3, 1), qutip.fock_dm(3, 0)),
    r"$|2,0\rangle$": qutip.tensor(qutip.fock_dm(3, 2), qutip.fock_dm(3, 0)),
}

n_states = len(result_x.states)
tlist_x = jnp.linspace(0, processor_x.get_full_tlist()[-1], n_states)

fig, ax = plt.subplots(figsize=(8, 4))
for label, proj in proj_ops.items():
    pops = [qutip.expect(proj, s) for s in result_x.states]
    ax.plot(tlist_x, pops, linewidth=2, label=label)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Population")
ax.set_title("X gate: population dynamics with 3-level transmon")
ax.legend(loc="center right")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# Verify final state
p_target = qutip.expect(proj_ops[r"$|1,0\rangle$"], result_x.states[-1])
p_leakage = qutip.expect(proj_ops[r"$|2,0\rangle$"], result_x.states[-1])
display(
    Math(rf"""
\textbf{{X gate result:}} \\
P(|1,0\rangle) = {p_target:.6f} \\
P(|2,0\rangle) = {p_leakage:.2e} \quad \text{{(leakage)}}
""")
)

# %% [markdown]
# ## Two-Qubit Gate: CNOT via Cross-Resonance
#
# The cross-resonance (CR) gate is a widely used two-qubit interaction
# for fixed-frequency transmon qubits
# {cite:p}`magesanEffectiveHamiltonianModels2020`.
# It drives the control qubit at the frequency of the target qubit,
# generating an effective $ZX$ interaction that can be used to implement
# a CNOT gate.
#
# In the `SCQubits` model, the CR gate is realised through the effective
# Hamiltonian:
# ```{math}
# :label: eq:cr-gate-hamiltonian
# H_{\rm CR} = \Omega^{\rm cr} \, \sigma^z_{\rm control}
#              \sigma^x_{\rm target}
# ```

# %%
qc_cnot = QubitCircuit(2)
qc_cnot.add_gate("CNOT", controls=0, targets=1)

processor_cnot = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
processor_cnot.load_circuit(qc_cnot)

# %%
fig, axes = processor_cnot.plot_pulses(figsize=(10, 6))
fig.suptitle("Control pulses for CNOT gate", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Bell State Preparation
#
# A Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ is
# the canonical entangled state.  We prepare it using a Hadamard gate
# followed by a CNOT:
# ```{math}
# :label: eq:bell-state-circuit
# |\Phi^+\rangle = \mathrm{CNOT} \cdot (H \otimes I) |00\rangle
# ```
# This is one of the most important benchmarks for two-qubit gate
# performance {cite:p}`kjaergaardSuperconductingQubits2020`.

# %%
qc_bell = QubitCircuit(2)
qc_bell.add_gate("SNOT", targets=0)  # Hadamard
qc_bell.add_gate("CNOT", controls=0, targets=1)

processor_bell = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
processor_bell.load_circuit(qc_bell)

init_state = qutip.basis([3, 3], [0, 0])
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
# ### Bell State Population Dynamics

# %%
proj_bell = {
    r"$|0,0\rangle$": qutip.tensor(qutip.fock_dm(3, 0), qutip.fock_dm(3, 0)),
    r"$|1,0\rangle$": qutip.tensor(qutip.fock_dm(3, 1), qutip.fock_dm(3, 0)),
    r"$|0,1\rangle$": qutip.tensor(qutip.fock_dm(3, 0), qutip.fock_dm(3, 1)),
    r"$|1,1\rangle$": qutip.tensor(qutip.fock_dm(3, 1), qutip.fock_dm(3, 1)),
}

n_states_bell = len(result_bell.states)
tlist_bell = jnp.linspace(0, processor_bell.get_full_tlist()[-1], n_states_bell)

fig, ax = plt.subplots(figsize=(10, 5))
for label, proj in proj_bell.items():
    pops = [qutip.expect(proj, s) for s in result_bell.states]
    ax.plot(tlist_bell, pops, linewidth=2, label=label)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Population")
ax.set_title(r"Bell state $|\Phi^+\rangle$ preparation: population dynamics")
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Bell State Fidelity with JAX
#
# We use the `qutip-jax` backend to compute the state fidelity with the
# ideal Bell state, demonstrating the JAX integration already established
# in the qpdk ecosystem.

# %%
# Ideal Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 in the 3-level basis
bell_ideal = (
    qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 0))
    + qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 1))
).unit()

# Convert to JAX backend for fidelity computation
final_state_jax = result_bell.states[-1].to("jax")
bell_ideal_jax = bell_ideal.to("jax")
fidelity_bell = qutip.fidelity(final_state_jax, bell_ideal_jax)

# Also compute leakage
leakage_ops = [
    qutip.tensor(qutip.fock_dm(3, i), qutip.fock_dm(3, j))
    for i in range(3)
    for j in range(3)
    if i >= 2 or j >= 2
]
total_leakage = sum(qutip.expect(op, result_bell.states[-1]) for op in leakage_ops)

display(
    Math(rf"""
\textbf{{Bell State Results (with JAX backend):}} \\
F(|\Phi^+\rangle) = {fidelity_bell:.6f} \\
P(|0,0\rangle) = {qutip.expect(proj_bell[r"$|0,0\rangle$"], result_bell.states[-1]):.4f} \\
P(|1,1\rangle) = {qutip.expect(proj_bell[r"$|1,1\rangle$"], result_bell.states[-1]):.4f} \\
\text{{Total leakage}} = {total_leakage:.2e}
""")
)

# %% [markdown]
# ## Decoherence Effects
#
# Real superconducting qubits suffer from energy relaxation ($T_1$) and
# dephasing ($T_2$).  The `SCQubits` processor supports these via
# Lindblad master-equation simulation.  We study how $T_1$ and $T_2$
# affect the Bell state fidelity
# {cite:p}`krantzQuantumEngineersGuide2019`.
#
# Typical state-of-the-art transmon qubits achieve:
# - $T_1 \approx 50$–$300\,\mu\text{s}$
# - $T_2 \approx 30$–$200\,\mu\text{s}$

# %%
# Sweep T2 values (in ns) with fixed T1 = 80 μs
# Note: T2 ≤ 2*T1 is required by the Lindblad master equation
t1_fixed = 80_000.0  # 80 μs in ns
t2_values_us = jnp.array([5, 10, 20, 40, 80, 150])  # μs (must be < 2*T1)
t2_values_ns = t2_values_us * 1e3  # Convert to ns

fidelities_t2 = []
for t2 in t2_values_ns:
    proc = SCQubits(
        2,
        wq=wq,
        alpha=alpha,
        g=g_coupling,
        wr=wr,
        t1=t1_fixed,
        t2=float(t2),
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
ax.set_ylim(0.5, 1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Noisy vs. Ideal Population Dynamics
#
# We compare the population dynamics of the Bell state preparation with
# and without decoherence.

# %%
# Noisy simulation with realistic parameters
t1_realistic = 50_000.0  # 50 μs
t2_realistic = 30_000.0  # 30 μs

processor_noisy = SCQubits(
    2,
    wq=wq,
    alpha=alpha,
    g=g_coupling,
    wr=wr,
    t1=t1_realistic,
    t2=t2_realistic,
)
processor_noisy.load_circuit(qc_bell)
result_noisy = processor_noisy.run_state(init_state)

n_states_noisy = len(result_noisy.states)
tlist_noisy = jnp.linspace(0, processor_noisy.get_full_tlist()[-1], n_states_noisy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Ideal (no noise)
for label, proj in proj_bell.items():
    pops = [qutip.expect(proj, s) for s in result_bell.states]
    ax1.plot(tlist_bell, pops, linewidth=2, label=label)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Population")
ax1.set_title("Ideal (no decoherence)")
ax1.legend(fontsize=9)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# Noisy
for label, proj in proj_bell.items():
    pops = [qutip.expect(proj, s) for s in result_noisy.states]
    ax2.plot(tlist_noisy, pops, linewidth=2, label=label)
ax2.set_xlabel("Time (ns)")
ax2.set_title(
    rf"With decoherence ($T_1 = {t1_realistic / 1e3:.0f}\,\mu$s, "
    rf"$T_2 = {t2_realistic / 1e3:.0f}\,\mu$s)"
)
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

fig.suptitle("Bell state preparation: ideal vs. noisy", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

f_ideal = qutip.fidelity(result_bell.states[-1], bell_ideal)
f_noisy = qutip.fidelity(result_noisy.states[-1], bell_ideal)
display(
    Math(rf"""
\textbf{{Fidelity Comparison:}} \\
F_{{\text{{ideal}}}} = {f_ideal:.6f} \\
F_{{\text{{noisy}}}} = {f_noisy:.6f} \\
\Delta F = {(f_ideal - f_noisy):.2e}
""")
)

# %% [markdown]
# ## Ramsey-Like Experiment
#
# A Ramsey experiment is a standard technique for measuring the qubit's
# dephasing time $T_2^*$ and frequency detuning.  It consists of two
# $\pi/2$ rotations separated by a free-evolution period.
#
# The Ramsey sequence is:
# ```{math}
# :label: eq:ramsey-sequence
# R_y(\pi/2) \;\longrightarrow\; \text{free evolution (time } \tau\text{)}
# \;\longrightarrow\; R_y(\pi/2) \;\longrightarrow\; \text{measure}
# ```
#
# Here we simulate a single Ramsey experiment to show the qubit
# dynamics between the two $\pi/2$ pulses.

# %%
qc_ramsey = QubitCircuit(1)
qc_ramsey.add_gate("RY", targets=0, arg_value=float(jnp.pi / 2))
qc_ramsey.add_gate("IDLE", targets=0, arg_value=100.0)  # 100 ns idle
qc_ramsey.add_gate("RY", targets=0, arg_value=float(jnp.pi / 2))

processor_ramsey = SCQubits(1, wq=[5.15], alpha=[-0.3])
processor_ramsey.load_circuit(qc_ramsey)

init_ramsey = qutip.basis([3], [0])
result_ramsey = processor_ramsey.run_state(init_ramsey)

n_ramsey = len(result_ramsey.states)
tlist_ramsey = jnp.linspace(0, processor_ramsey.get_full_tlist()[-1], n_ramsey)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Populations
for level, label in [(0, r"$|0\rangle$"), (1, r"$|1\rangle$"), (2, r"$|2\rangle$")]:
    proj = qutip.fock_dm(3, level)
    pops = [qutip.expect(proj, s) for s in result_ramsey.states]
    ax1.plot(tlist_ramsey, pops, linewidth=2, label=label)
ax1.set_ylabel("Population")
ax1.set_title("Ramsey sequence: population dynamics")
ax1.legend()
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# Bloch vector components
sx = qutip.Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # σ_x in 3-level
sy = qutip.Qobj([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])  # σ_y in 3-level
sz = qutip.Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])  # σ_z in 3-level

for op, label in [
    (sx, r"$\langle\sigma_x\rangle$"),
    (sy, r"$\langle\sigma_y\rangle$"),
    (sz, r"$\langle\sigma_z\rangle$"),
]:
    exps = [qutip.expect(op, s) for s in result_ramsey.states]
    ax2.plot(tlist_ramsey, jnp.real(jnp.array(exps)), linewidth=2, label=label)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Expectation value")
ax2.set_title("Ramsey sequence: Bloch vector components")
ax2.legend()
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Gate Duration Comparison
#
# The gate duration is a crucial parameter that determines the number of
# operations possible within the coherence time.  We compare the pulse
# durations for different gate types.

# %%
gate_circuits = {
    "X": QubitCircuit(2),
    "Y": QubitCircuit(2),
    "H (SNOT)": QubitCircuit(2),
    "CNOT": QubitCircuit(2),
}
gate_circuits["X"].add_gate("X", targets=0)
gate_circuits["Y"].add_gate("Y", targets=0)
gate_circuits["H (SNOT)"].add_gate("SNOT", targets=0)
gate_circuits["CNOT"].add_gate("CNOT", controls=0, targets=1)

gate_durations = {}
for name, qc in gate_circuits.items():
    proc = SCQubits(2, wq=wq, alpha=alpha, g=g_coupling, wr=wr)
    proc.load_circuit(qc)
    total_time = proc.get_full_tlist()[-1]
    gate_durations[name] = total_time

fig, ax = plt.subplots(figsize=(8, 4))
names = list(gate_durations.keys())
durations = list(gate_durations.values())
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
bars = ax.barh(names, durations, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Gate duration (ns)")
ax.set_title("Pulse-level gate durations (SCQubits processor)")
for bar, d in zip(bars, durations):
    ax.text(
        bar.get_width() + 2,
        bar.get_y() + bar.get_height() / 2,
        f"{d:.1f} ns",
        va="center",
        fontsize=10,
    )
ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## State Tomography: Fidelity Over Time
#
# We compute the instantaneous fidelity of the evolving state with the
# target state throughout the gate sequence.  This reveals the gate dynamics
# and confirms convergence to the target.

# %%
# X gate fidelity over time
target_x = qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 0))

fidelity_evolution_x = [qutip.fidelity(s, target_x) for s in result_x.states]

# Bell state fidelity over time
fidelity_evolution_bell = [qutip.fidelity(s, bell_ideal) for s in result_bell.states]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(tlist_x, fidelity_evolution_x, linewidth=2, color="#2196F3")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Fidelity")
ax1.set_title(r"X gate: $F(|\psi(t)\rangle, |1,0\rangle)$")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

ax2.plot(tlist_bell, fidelity_evolution_bell, linewidth=2, color="#F44336")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Fidelity")
ax2.set_title(r"Bell state: $F(|\psi(t)\rangle, |\Phi^+\rangle)$")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

fig.suptitle("State fidelity evolution during pulse-level gates", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Using JAX for State Analysis
#
# The `qutip-jax` backend allows using JAX's features for quantum state
# analysis.  Here we demonstrate converting simulation results to the JAX
# backend for efficient batch computation of expectation values.

# %%
# Convert all states to JAX for batch processing
proj_00_jax = qutip.tensor(qutip.fock_dm(3, 0), qutip.fock_dm(3, 0)).to("jax")
proj_11_jax = qutip.tensor(qutip.fock_dm(3, 1), qutip.fock_dm(3, 1)).to("jax")

# Compute populations using JAX backend
pops_00_jax = []
pops_11_jax = []
for state in result_bell.states:
    s_jax = state.to("jax")
    pops_00_jax.append(float(qutip.expect(proj_00_jax, s_jax)))
    pops_11_jax.append(float(qutip.expect(proj_11_jax, s_jax)))

# Compute concurrence-like entanglement measure
# For the Bell state, the difference |P(00) - P(11)| should be minimal
imbalance = jnp.abs(jnp.array(pops_00_jax) - jnp.array(pops_11_jax))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(tlist_bell, pops_00_jax, linewidth=2, label=r"$P(|0,0\rangle)$ (JAX)")
ax1.plot(tlist_bell, pops_11_jax, linewidth=2, label=r"$P(|1,1\rangle)$ (JAX)")
ax1.set_ylabel("Population")
ax1.set_title("Bell state populations (computed with qutip-jax backend)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(tlist_bell, imbalance, linewidth=2, color="#9C27B0")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel(r"$|P_{00} - P_{11}|$")
ax2.set_title("Population imbalance (ideally 0 at end)")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

display(
    Math(rf"""
\textbf{{JAX Backend Results:}} \\
\text{{Final }} P(|0,0\rangle) = {pops_00_jax[-1]:.6f} \\
\text{{Final }} P(|1,1\rangle) = {pops_11_jax[-1]:.6f} \\
\text{{Population imbalance}} = {float(imbalance[-1]):.2e}
""")
)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated pulse-level simulation of superconducting
# transmon qubits using **qutip-qip** with the **qutip-jax** backend:
#
# 1. **Single-qubit gates** (X gate): fast (~50 ns), high-fidelity
#    operations with minimal leakage to the transmon's third level
#
# 2. **Two-qubit gates** (CNOT via cross-resonance): longer duration
#    (~400 ns), requiring careful calibration of the CR pulse
#
# 3. **Bell state preparation**: achieves near-unit fidelity in the
#    ideal case, demonstrating proper gate decomposition
#
# 4. **Decoherence effects**: $T_1$ and $T_2$ noise reduce gate fidelity,
#    with $T_2$ being the dominant factor for short gate sequences
#
# 5. **Ramsey experiment**: captures qubit dynamics between $\pi/2$ pulses,
#    a key characterisation technique
#
# 6. **JAX integration**: qutip-jax provides a bridge to the JAX ecosystem
#    already used in qpdk for state analysis and computation
#
# The physical parameters used here are consistent with the companion
# notebooks on
# [scQubits parameter calculation](scqubits_parameter_calculation) and
# [pymablock dispersive shift](pymablock_dispersive_shift), which show
# how to derive these parameters from the Hamiltonian and map them to
# physical layout dimensions.
#
# For designing a transmon qubit with specific gate performance targets,
# the workflow is:
#
# 1. **Hamiltonian design** → choose $\omega_q$, $\alpha$, $g$
#    (see companion notebooks)
# 2. **Pulse-level verification** → simulate gates with `qutip-qip`
#    (this notebook)
# 3. **Layout design** → map circuit parameters to geometry using
#    `qpdk` helpers

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
