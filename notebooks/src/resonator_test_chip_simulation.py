# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # SAX Simulation of a Resonator Test Chip
#
# This notebook demonstrates how to run a circuit-level SAX simulation of the
# `resonator_test_chip_yaml` component, which is defined via a `.pic.yml` netlist
# file and a corresponding gdsfactory+ schematic.
#
# The workflow is:
# 1. Load the component from the YAML netlist with gdsfactory.
# 2. Extract the netlist for circuit simulation.
# 3. Build a SAX circuit using the QPDK model library.
# 4. Evaluate the S-parameters over a frequency range.
# 5. Plot the transmission to observe resonator dips.
# 6. Perform a **Monte Carlo fabrication tolerance analysis** to quantify how
#    CPW width and gap variations affect resonance frequencies and transmission.

# %% tags=["hide-input", "hide-output"]
import warnings

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from matplotlib.lines import Line2D
from sax.models.rf import capacitor as rf_capacitor
from sax.models.rf import electrical_open, electrical_short
from sax.models.rf import tee as rf_tee
from scipy.signal import find_peaks

from qpdk import PATH, PDK
from qpdk.models import models
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.couplers import cpw_cpw_coupling_capacitance_per_length_analytical
from qpdk.models.cpw import (
    cpw_epsilon_eff,
    cpw_parameters,
    cpw_thickness_correction,
    get_cpw_substrate_params,
    propagation_constant,
    transmission_line_s_params,
)

PDK.activate()

# %% [markdown]
# ## Load the component
#
# The resonator test chip is defined in a `.pic.yml` file that lives alongside
# the QPDK sample scripts.  It contains 16 quarter-wave coupled resonators
# (8 per probeline), four launchers, and CPW routing between all elements.

# %%
yaml_path = PATH.samples / "resonator_test_chip_yaml.pic.yml"
chip = gf.read.from_yaml(yaml_path)
chip.plot()

# %% [markdown]
# ## Extract the netlist
#
# `Component.get_netlist()` returns a dictionary with `instances`,
# `nets`, `ports`, and `placements`.  SAX understands this format directly.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    netlist = chip.get_netlist()

print("Instances:")
for name, inst in netlist["instances"].items():
    print(f"  {name}: {inst['component']}")

# %% [markdown]
# ## Build the SAX circuit
#
# QPDK ships a `models` dictionary that maps every component name to a SAX
# model function.  We pass the netlist and models to `sax.circuit`.

# %%
circuit_fn, circuit_info = sax.circuit(
    netlist=netlist,
    models=models,
    on_internal_port="ignore",
)

# %% [markdown]
# ## Simulate
#
# Evaluate the circuit over the 5–9 GHz band.  The four external ports
# correspond to the four launchers:
#
# | Port | Launcher | Probeline |
# |------|----------|-----------|
# | o1   | West top | Top       |
# | o2   | East top | Top       |
# | o3   | West bot | Bottom    |
# | o4   | East bot | Bottom    |
#
# The top probeline has eight resonators with **varying** coupling gaps
# (12–26 µm) and the bottom probeline has eight resonators with a **fixed**
# coupling gap of 16 µm.

# %%
freq = jnp.linspace(5e9, 9e9, 5001)
s_params = circuit_fn(f=freq)

freq_ghz = freq / 1e9

# %% [markdown]
# ## Results
#
# ### Top probeline – variable coupling gap
#
# :math:`S_{21}` (port o1 → o2) shows eight notches, one per resonator.

# %%
s21 = s_params[("o1", "o2")]
s11 = s_params[("o1", "o1")]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="$S_{21}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s11)), label="$S_{11}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Top probeline (variable coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bottom probeline – fixed coupling gap
#
# :math:`S_{43}` (port o3 → o4) shows eight notches with uniform coupling
# depth because all resonators share the same 16 µm coupling gap.

# %%
s43 = s_params[("o3", "o4")]
s33 = s_params[("o3", "o3")]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s43)), label="$S_{43}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s33)), label="$S_{33}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Bottom probeline (fixed coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Both probelines
#
# Overlay both transmission traces to compare the two probelines.

# %%
fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="Top ($S_{21}$)")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s43)), label="Bottom ($S_{43}$)")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Resonator test chip – transmission comparison")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Monte Carlo Fabrication Tolerance Analysis
#
# In superconducting quantum chip fabrication, the CPW centre-conductor width
# and gap to the ground plane are subject to process variations introduced during
# lithography and etching.  Even sub-micrometre deviations from the nominal
# geometry change the characteristic impedance :math:`Z_0`, effective
# permittivity :math:`\varepsilon_{\mathrm{eff}}`, and — most critically — the
# resonance frequencies of the on-chip resonators.
#
# This section performs a **Monte Carlo analysis** inspired by the
# `SAX layout-aware Monte Carlo example <https://flaport.github.io/sax/nbs/examples/07_layout_aware/>`_.
# The approach is:
#
# 1. Build **MC-compatible model functions** whose CPW width and gap are
#    explicit, JAX-traceable parameters (instead of being locked inside a
#    cross-section object).
# 2. Compile the SAX circuit **once** with these models.
# 3. Pass per-instance ``cpw_width`` / ``cpw_gap`` overrides for each Monte
#    Carlo trial, leveraging JAX array broadcasting for efficient batched
#    evaluation.
# 4. Analyse the resulting spread in resonance frequencies and transmission.

# %% [markdown]
# ## CPW impedance sensitivity
#
# Before running the Monte Carlo simulation it is instructive to see how
# :math:`Z_0` and :math:`\varepsilon_{\mathrm{eff}}` depend on the CPW
# dimensions.

# %%
widths_sweep = np.linspace(6, 16, 200)
gaps_sweep = [4.0, 5.0, 6.0, 7.0, 8.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for gap_val in gaps_sweep:
    ep_vals = []
    z0_vals = []
    for w in widths_sweep:
        ep, z0 = cpw_parameters(float(w), float(gap_val))
        ep_vals.append(ep)
        z0_vals.append(z0)
    ax1.plot(widths_sweep, z0_vals, label=f"gap = {gap_val:.0f} µm")
    ax2.plot(widths_sweep, ep_vals, label=f"gap = {gap_val:.0f} µm")

# Nominal point
ep_nom, z0_nom = cpw_parameters(10.0, 6.0)
ax1.axhline(z0_nom, color="k", ls=":", lw=0.8)
ax1.axvline(10.0, color="k", ls=":", lw=0.8)
ax2.axhline(ep_nom, color="k", ls=":", lw=0.8)
ax2.axvline(10.0, color="k", ls=":", lw=0.8)

ax1.set_xlabel("Centre-conductor width [µm]")
ax1.set_ylabel("$Z_0$ [Ω]")
ax1.set_title("Characteristic impedance")
ax1.legend(fontsize=8)
ax1.grid(True)

ax2.set_xlabel("Centre-conductor width [µm]")
ax2.set_ylabel(r"$\varepsilon_{\mathrm{eff}}$")
ax2.set_title("Effective permittivity")
ax2.legend(fontsize=8)
ax2.grid(True)

fig.suptitle("CPW parameter sensitivity to geometry", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# The plots show that :math:`Z_0` and :math:`\varepsilon_{\mathrm{eff}}` are
# sensitive to both width and gap.  A ±0.5 µm width shift at the nominal
# 10 µm / 6 µm geometry translates to a :math:`Z_0` change of several ohms and
# a corresponding shift in resonance frequency.

# %% [markdown]
# ## MC-compatible model functions
#
# The standard QPDK models extract CPW width and gap from a *cross-section
# object*, which is convenient for layout-driven simulation but opaque to JAX's
# automatic differentiation and batching machinery.  Here we define thin wrapper
# models that accept ``cpw_width`` and ``cpw_gap`` as explicit numeric
# parameters while reusing the same analytical CPW theory
# (conformal-mapping :math:`\varepsilon_{\mathrm{eff}}` and :math:`Z_0`,
# conductor-thickness correction).
#
# The substrate constants (height, metal thickness, :math:`\varepsilon_r`) are
# fixed at PDK values since they do not vary across the wafer.

# %%
# Substrate constants (fixed)
_h_m, _t_m, _ep_r = get_cpw_substrate_params()
_h_si = _h_m * 1e-6  # convert µm → m
_t_si = _t_m * 1e-6


def straight_mc(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000.0,
    cpw_width: sax.Float = 10.0,
    cpw_gap: sax.Float = 6.0,
) -> sax.SDict:
    """Straight CPW model with explicit width/gap for Monte Carlo analysis."""
    f = jnp.asarray(f)

    w_m = jnp.asarray(cpw_width) * 1e-6
    s_m = jnp.asarray(cpw_gap) * 1e-6

    ep_eff = cpw_epsilon_eff(w_m, s_m, _h_si, _ep_r)
    ep_eff, z0_val = cpw_thickness_correction(w_m, s_m, _t_si, ep_eff)

    # Do not ravel f: keep shape (n_freq, 1) so it broadcasts
    # against ep_eff of shape (n_trials,) → (n_freq, n_trials).
    gamma = propagation_constant(f, ep_eff, tand=0.0, ep_r=_ep_r)
    length_m = jnp.asarray(length) * 1e-6
    s11, s21 = transmission_line_s_params(gamma, z0_val, length_m)

    return sax.reciprocal(
        {
            ("o1", "o1"): s11,
            ("o1", "o2"): s21,
            ("o2", "o2"): s11,
        }
    )


def bend_mc(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000.0,
    cpw_width: sax.Float = 10.0,
    cpw_gap: sax.Float = 6.0,
) -> sax.SDict:
    """Bend model delegating to ``straight_mc`` (phase-only approximation)."""
    return straight_mc(f=f, length=length, cpw_width=cpw_width, cpw_gap=cpw_gap)


def quarter_wave_resonator_coupled_mc(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 5000.0,
    coupling_gap: float = 0.27,
    coupling_straight_length: float = 20.0,
    cpw_width: float = 10.0,
    cpw_gap: float = 6.0,
) -> sax.SDict:
    """Quarter-wave coupled resonator with explicit CPW width/gap."""
    f_arr = jnp.asarray(f)

    # Coupling capacitance (JAX-traceable analytical model)
    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=coupling_gap, width=cpw_width, cpw_gap=cpw_gap, ep_r=_ep_r
    )
    c_coupling = c_pul * coupling_straight_length * 1e-6

    # Characteristic impedance for the capacitor normalisation
    w_m = jnp.asarray(cpw_width) * 1e-6
    s_m = jnp.asarray(cpw_gap) * 1e-6
    ep_eff = cpw_epsilon_eff(w_m, s_m, _h_si, _ep_r)
    _, z0_val = cpw_thickness_correction(w_m, s_m, _t_si, ep_eff)

    # Internal sub-circuit instances (all JAX-traceable)
    _kw = {"cpw_width": cpw_width, "cpw_gap": cpw_gap}
    instances = {
        "coupling_1": straight_mc(f=f_arr, length=coupling_straight_length / 2, **_kw),
        "coupling_2": straight_mc(f=f_arr, length=coupling_straight_length / 2, **_kw),
        "resonator_1": straight_mc(f=f_arr, length=coupling_straight_length / 2, **_kw),
        "resonator_2": straight_mc(
            f=f_arr, length=length - coupling_straight_length / 2, **_kw
        ),
        "tee_1": rf_tee(f=f_arr),
        "tee_2": rf_tee(f=f_arr),
        "capacitor": rf_capacitor(f=f_arr, capacitance=c_coupling, z0=z0_val),
        "open_start_term": electrical_open(f=f_arr, n_ports=2),
        "short": electrical_short(f=f_arr),
    }

    connections = {
        "coupling_1,o2": "tee_1,o1",
        "coupling_2,o1": "tee_1,o2",
        "resonator_1,o2": "tee_2,o1",
        "resonator_2,o1": "tee_2,o2",
        "tee_1,o3": "capacitor,o1",
        "tee_2,o3": "capacitor,o2",
        "resonator_1,o1": "open_start_term,o1",
        "resonator_2,o2": "short,o1",
    }

    ports = {
        "coupling_o1": "coupling_1,o1",
        "coupling_o2": "coupling_2,o2",
        "resonator_o1": "open_start_term,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


# %%
# Build the MC-compatible models dictionary.
# We replace only the CPW-sensitive components; the launcher model is kept
# as-is because its large pad geometry (200 µm width) is insensitive to the
# small absolute variations we study here.

mc_models = {
    **models,
    "straight": straight_mc,
    "bend_euler": bend_mc,
    "bend_circular": bend_mc,
    "bend_s": bend_mc,
    "rectangle": bend_mc,
    "quarter_wave_resonator_coupled": quarter_wave_resonator_coupled_mc,
}

# %% [markdown]
# ## Build the MC circuit
#
# We compile the circuit **once**.  SAX exposes ``cpw_width`` and ``cpw_gap``
# as tuneable parameters for every instance that uses our MC models.

# %%
mc_circuit_fn, _ = sax.circuit(
    netlist=netlist,
    models=mc_models,
    on_internal_port="ignore",
)

# Verify: the nominal MC result should match the original simulation.
s_nom = mc_circuit_fn(f=freq)
s21_nom = s_nom[("o1", "o2")]
s21_nom_db = 20 * jnp.log10(jnp.abs(s21_nom))

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="Original", lw=2)
ax.plot(freq_ghz, s21_nom_db, label="MC (nominal)", ls="--", lw=1.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("|$S_{21}$| [dB]")
ax.set_title("Sanity check – nominal MC vs original simulation")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Define fabrication tolerances
#
# Typical superconducting CPW fabrication tolerances are on the order of
# hundreds of nanometres to about 1 µm, depending on lithography technology
# :cite:`krantzQuantumEngineersGuide2019`.  We model these as independent
# Gaussian random variables.

# %%
NOMINAL_WIDTH = 10.0  # µm (centre-conductor width)
NOMINAL_GAP = 6.0  # µm (gap to ground plane)
WIDTH_SIGMA = 0.5  # µm (1σ tolerance on width)
GAP_SIGMA = 0.3  # µm (1σ tolerance on gap)
N_TRIALS = 100

rng = np.random.default_rng(42)

# %% [markdown]
# ## Identify MC-tuneable instances
#
# We inspect the circuit settings to find all instances that expose
# ``cpw_width`` as a tuneable parameter.  These are the straight sections,
# bends, and resonators – i.e. every CPW element on the chip.

# %%
mc_settings = sax.get_settings(mc_circuit_fn)
cpw_instance_names = [name for name, s in mc_settings.items() if "cpw_width" in s]
print(f"{len(cpw_instance_names)} CPW instances found for MC perturbation.")

# %% [markdown]
# ## Global (systematic) tolerance simulation
#
# In this scenario **all** CPW sections on the chip receive the **same**
# width and gap perturbation in each trial, modelling a uniform fabrication
# bias across the die.

# %%
dw_global = rng.normal(0, WIDTH_SIGMA, N_TRIALS)
dg_global = rng.normal(0, GAP_SIGMA, N_TRIALS)

dw_jnp = jnp.array(dw_global)
dg_jnp = jnp.array(dg_global)

# All CPW instances get the same perturbation per trial (batched).
global_overrides = {
    name: {
        "cpw_width": NOMINAL_WIDTH + dw_jnp,
        "cpw_gap": NOMINAL_GAP + dg_jnp,
    }
    for name in cpw_instance_names
}

# Evaluate (freq axis broadcasts against MC trial axis)
s_global = mc_circuit_fn(f=freq[:, None], **global_overrides)
s21_global = s_global[("o1", "o2")]  # shape: (n_freq, n_trials)
s21_global_db = 20 * jnp.log10(jnp.abs(s21_global))

# %% [markdown]
# ### Transmission overlay – global tolerance
#
# Each pale blue trace is one MC trial; the orange curve is the nominal
# response.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(freq_ghz, s21_global_db, color="C0", lw=0.4, alpha=0.15, rasterized=True)
ax.plot(freq_ghz, s21_nom_db, color="C1", lw=2, label="Nominal")

mc_line = Line2D([], [], color="C0", lw=1, alpha=0.5, label="MC trials")
ax.legend(handles=[mc_line, ax.get_lines()[-1]], fontsize=10)

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("|$S_{21}$| [dB]")
ax.set_title(
    f"Global tolerance: width = {NOMINAL_WIDTH} ± {WIDTH_SIGMA} µm, "
    f"gap = {NOMINAL_GAP} ± {GAP_SIGMA} µm  ({N_TRIALS} trials)"
)
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Resonance frequency extraction
#
# We locate the dip positions in each :math:`|S_{21}|` trace to track how
# the resonance frequencies shift across Monte Carlo trials.


# %%
def find_resonance_dips(
    freq_hz: np.ndarray,
    s21_db: np.ndarray,
    n_resonators: int = 8,
    prominence: float = 0.3,
    min_distance_idx: int = 30,
) -> np.ndarray:
    """Find resonance dip frequencies by detecting peaks in −|S₂₁| (dB).

    Returns an array of shape ``(n_resonators,)`` with the dip frequencies
    in Hz, sorted in ascending order.  If fewer dips are found, the missing
    entries are filled with ``NaN``.
    """
    s21_np = np.asarray(s21_db)
    peaks, properties = find_peaks(
        -s21_np, prominence=prominence, distance=min_distance_idx
    )

    if len(peaks) == 0:
        return np.full(n_resonators, np.nan)

    # Sort by prominence (deepest dips first) and keep up to n_resonators
    order = np.argsort(properties["prominences"])[::-1][:n_resonators]
    peak_idx = np.sort(peaks[order])

    freqs_out = np.full(n_resonators, np.nan)
    freqs_out[: len(peak_idx)] = np.asarray(freq_hz)[peak_idx]
    return freqs_out


# %%
# Extract resonance frequencies for the nominal and all MC trials
freq_np = np.asarray(freq)
s21_nom_np = np.asarray(s21_nom_db)

nominal_dips = find_resonance_dips(freq_np, s21_nom_np)
n_res = int(np.sum(~np.isnan(nominal_dips)))
print(f"Nominal resonance frequencies ({n_res} detected):")
for i, f_dip in enumerate(nominal_dips):
    if not np.isnan(f_dip):
        print(f"  Resonator {i + 1}: {f_dip / 1e9:.4f} GHz")

# %%
# MC dip extraction
mc_dips = np.zeros((N_TRIALS, n_res))
for trial in range(N_TRIALS):
    mc_dips[trial] = find_resonance_dips(
        freq_np, np.asarray(s21_global_db[:, trial]), n_resonators=n_res
    )

# %% [markdown]
# ### Resonance frequency histograms
#
# Each subplot shows the distribution of a single resonator's frequency
# across all MC trials.  The red dashed line marks the nominal frequency.

# %%
n_cols = min(4, n_res)
n_rows = int(np.ceil(n_res / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes_flat = np.atleast_1d(axes).ravel()

for i in range(n_res):
    ax = axes_flat[i]
    dips_ghz = mc_dips[:, i] / 1e9
    valid = ~np.isnan(dips_ghz)
    ax.hist(dips_ghz[valid], bins=25, color="C0", alpha=0.7, edgecolor="white")
    ax.axvline(nominal_dips[i] / 1e9, color="r", ls="--", lw=1.5, label="Nominal")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Count")
    ax.set_title(f"Resonator {i + 1}", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for j in range(n_res, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    "Resonance frequency distributions (global tolerance)", fontsize=13, y=1.02
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Frequency shift statistics
#
# We summarise the absolute and relative frequency shifts.

# %%
shifts_mhz = (mc_dips - nominal_dips[None, :n_res]) / 1e6

mean_shift = np.nanmean(shifts_mhz, axis=0)
std_shift = np.nanstd(shifts_mhz, axis=0)
max_shift = np.nanmax(np.abs(shifts_mhz), axis=0)

print("Resonator | Nominal [GHz] | Mean shift [MHz] | Std [MHz] | Max |shift| [MHz]")
print("-" * 80)
for i in range(n_res):
    print(
        f"    {i + 1:2d}     |   {nominal_dips[i] / 1e9:8.4f}    |"
        f"    {mean_shift[i]:+7.2f}      |  {std_shift[i]:6.2f}   |    {max_shift[i]:7.2f}"
    )

# %% [markdown]
# ## Per-resonator (independent) tolerance simulation
#
# In practice, fabrication variations are not perfectly uniform across a die.
# Local effects such as proximity-dependent etching cause each resonator to
# experience a **different** random perturbation.  We model this by assigning
# each resonator instance an independent draw from the tolerance distribution,
# while all straight / bend routing sections that connect to a given resonator
# share its perturbation (simulating local correlation).

# %%
# Group instances by resonator: each resonator and its adjacent route segments
# receive the same local perturbation.  For instances not clearly tied to a
# specific resonator (e.g. launcher-adjacent routes), we use the nominal value.

# Resonator instance names
res_names = sorted(
    [n for n in cpw_instance_names if n.startswith("resonator_")],
    key=lambda x: x,
)

# Non-resonator CPW instances keep nominal values in the per-resonator scenario
non_res_names = [n for n in cpw_instance_names if n not in res_names]

# Draw per-resonator perturbations: shape (n_resonator_instances, N_TRIALS)
n_res_instances = len(res_names)
dw_per_res = rng.normal(0, WIDTH_SIGMA, (n_res_instances, N_TRIALS))
dg_per_res = rng.normal(0, GAP_SIGMA, (n_res_instances, N_TRIALS))

local_overrides = {}
for i, name in enumerate(res_names):
    local_overrides[name] = {
        "cpw_width": NOMINAL_WIDTH + jnp.array(dw_per_res[i]),
        "cpw_gap": NOMINAL_GAP + jnp.array(dg_per_res[i]),
    }
# Non-resonator instances at nominal
for name in non_res_names:
    local_overrides[name] = {
        "cpw_width": jnp.full(N_TRIALS, NOMINAL_WIDTH),
        "cpw_gap": jnp.full(N_TRIALS, NOMINAL_GAP),
    }

s_local = mc_circuit_fn(f=freq[:, None], **local_overrides)
s21_local = s_local[("o1", "o2")]
s21_local_db = 20 * jnp.log10(jnp.abs(s21_local))

# %% [markdown]
# ### Transmission overlay – per-resonator tolerance

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(freq_ghz, s21_local_db, color="C0", lw=0.4, alpha=0.15, rasterized=True)
ax.plot(freq_ghz, s21_nom_db, color="C1", lw=2, label="Nominal")

mc_line = Line2D([], [], color="C0", lw=1, alpha=0.5, label="MC trials")
ax.legend(handles=[mc_line, ax.get_lines()[-1]], fontsize=10)

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("|$S_{21}$| [dB]")
ax.set_title(
    f"Per-resonator tolerance: width σ = {WIDTH_SIGMA} µm, "
    f"gap σ = {GAP_SIGMA} µm  ({N_TRIALS} trials)"
)
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Compare global vs per-resonator spread

# %%
mc_dips_local = np.zeros((N_TRIALS, n_res))
for trial in range(N_TRIALS):
    mc_dips_local[trial] = find_resonance_dips(
        freq_np, np.asarray(s21_local_db[:, trial]), n_resonators=n_res
    )

shifts_local = (mc_dips_local - nominal_dips[None, :n_res]) / 1e6
std_local = np.nanstd(shifts_local, axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, n_res + 1)
width_bar = 0.35
ax.bar(x - width_bar / 2, std_shift, width_bar, label="Global", color="C0", alpha=0.8)
ax.bar(
    x + width_bar / 2,
    std_local,
    width_bar,
    label="Per-resonator",
    color="C3",
    alpha=0.8,
)
ax.set_xlabel("Resonator index")
ax.set_ylabel("Frequency spread (1σ) [MHz]")
ax.set_title("Frequency uncertainty: global vs per-resonator tolerance")
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# The global scenario produces **correlated** frequency shifts (all resonators
# move together), whereas the per-resonator scenario produces **uncorrelated**
# shifts that appear as a broadened distribution per resonator.  In practice,
# both mechanisms coexist.

# %% [markdown]
# ## Sensitivity analysis – width vs gap
#
# To understand which geometric parameter dominates the frequency uncertainty
# we run two additional sweeps:
# one varying **only** the width and one varying **only** the gap.

# %%
# Width-only variation
dw_only = rng.normal(0, WIDTH_SIGMA, N_TRIALS)
overrides_w = {
    name: {
        "cpw_width": NOMINAL_WIDTH + jnp.array(dw_only),
        "cpw_gap": jnp.full(N_TRIALS, NOMINAL_GAP),
    }
    for name in cpw_instance_names
}
s_w = mc_circuit_fn(f=freq[:, None], **overrides_w)
s21_w_db = 20 * jnp.log10(jnp.abs(s_w[("o1", "o2")]))

# Gap-only variation
dg_only = rng.normal(0, GAP_SIGMA, N_TRIALS)
overrides_g = {
    name: {
        "cpw_width": jnp.full(N_TRIALS, NOMINAL_WIDTH),
        "cpw_gap": NOMINAL_GAP + jnp.array(dg_only),
    }
    for name in cpw_instance_names
}
s_g = mc_circuit_fn(f=freq[:, None], **overrides_g)
s21_g_db = 20 * jnp.log10(jnp.abs(s_g[("o1", "o2")]))

# %%
mc_dips_w = np.zeros((N_TRIALS, n_res))
mc_dips_g = np.zeros((N_TRIALS, n_res))
for trial in range(N_TRIALS):
    mc_dips_w[trial] = find_resonance_dips(
        freq_np, np.asarray(s21_w_db[:, trial]), n_resonators=n_res
    )
    mc_dips_g[trial] = find_resonance_dips(
        freq_np, np.asarray(s21_g_db[:, trial]), n_resonators=n_res
    )

std_w_only = np.nanstd((mc_dips_w - nominal_dips[None, :n_res]) / 1e6, axis=0)
std_g_only = np.nanstd((mc_dips_g - nominal_dips[None, :n_res]) / 1e6, axis=0)

# %% [markdown]
# ### Sensitivity comparison

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, n_res + 1)
w_bar = 0.25
ax.bar(
    x - w_bar,
    std_w_only,
    w_bar,
    label=f"Width only (σ = {WIDTH_SIGMA} µm)",
    color="C0",
    alpha=0.85,
)
ax.bar(
    x,
    std_g_only,
    w_bar,
    label=f"Gap only (σ = {GAP_SIGMA} µm)",
    color="C2",
    alpha=0.85,
)
ax.bar(
    x + w_bar,
    std_shift,
    w_bar,
    label="Both (combined)",
    color="C3",
    alpha=0.85,
)
ax.set_xlabel("Resonator index")
ax.set_ylabel("Frequency spread (1σ) [MHz]")
ax.set_title("Sensitivity: width vs gap contribution to frequency uncertainty")
ax.set_xticks(x)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Scatter: width perturbation vs frequency shift

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

shifts_w = (mc_dips_w - nominal_dips[None, :n_res]) / 1e6
shifts_g = (mc_dips_g - nominal_dips[None, :n_res]) / 1e6

# Width perturbation scatter
ax = axes[0]
for i in range(n_res):
    valid = ~np.isnan(shifts_w[:, i])
    ax.scatter(
        dw_only[valid], shifts_w[valid, i], s=10, alpha=0.4, label=f"Res {i + 1}"
    )
ax.set_xlabel("Width perturbation δw [µm]")
ax.set_ylabel("Frequency shift [MHz]")
ax.set_title("Width perturbation → frequency shift")
ax.axhline(0, color="k", ls=":", lw=0.8)
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)

# Gap perturbation scatter
ax = axes[1]
for i in range(n_res):
    valid = ~np.isnan(shifts_g[:, i])
    ax.scatter(
        dg_only[valid], shifts_g[valid, i], s=10, alpha=0.4, label=f"Res {i + 1}"
    )
ax.set_xlabel("Gap perturbation δg [µm]")
ax.set_ylabel("Frequency shift [MHz]")
ax.set_title("Gap perturbation → frequency shift")
ax.axhline(0, color="k", ls=":", lw=0.8)
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter plots reveal a nearly **linear** relationship between the
# geometric perturbation and the resulting frequency shift, confirming that
# a first-order sensitivity model is appropriate for these tolerance levels.
# Width variations generally produce a **larger** frequency shift than gap
# variations of similar magnitude, making width control the dominant
# fabrication concern for frequency targeting.

# %% [markdown]
# ## Summary
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 20 20 30
#
#    * - Scenario
#      - Width σ [µm]
#      - Gap σ [µm]
#      - Typical freq. spread (1σ) [MHz]
#    * - Global (systematic bias)
#      - 0.5
#      - 0.3
#      - ~tens of MHz (correlated)
#    * - Per-resonator (local)
#      - 0.5
#      - 0.3
#      - ~tens of MHz (uncorrelated)
#    * - Width only
#      - 0.5
#      - 0.0
#      - dominant contribution
#    * - Gap only
#      - 0.0
#      - 0.3
#      - secondary contribution
#
# **Key takeaways:**
#
# - Fabrication tolerances of ±0.5 µm on CPW width and ±0.3 µm on gap lead to
#   resonance frequency spreads of tens of MHz — well above typical qubit–readout
#   detuning budgets.
# - **Width control** is the single most important parameter for frequency
#   targeting.
# - The Monte Carlo framework developed here can be extended to include spatial
#   correlation across the die (wafer-map approach, as in the
#   `SAX layout-aware example <https://flaport.github.io/sax/nbs/examples/07_layout_aware/>`_)
#   or additional sources of variation (substrate permittivity, metal
#   thickness).
