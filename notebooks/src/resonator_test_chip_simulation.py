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
import os
import warnings
from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import ray
import sax
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from qpdk import PATH, PDK
from qpdk.helper import display_dataframe
from qpdk.models import models
from qpdk.models.cpw import cpw_parameters
from qpdk.tech import coplanar_waveguide

PDK.activate()


def ray_get_with_progress(
    futures: Sequence[ray.ObjectRef],
    desc: str = "Processing",
    unit: str = "sim",
    timeout: float | None = None,
) -> list[Any]:
    """Get results from Ray futures with a tqdm progress bar, preserving order."""
    # Map future ID to its original index to preserve order
    result_map = {f: i for i, f in enumerate(futures)}
    results: list[Any | None] = [None] * len(futures)

    remaining_ids = list(futures)
    with tqdm(
        total=len(futures),
        desc=f"{desc:.<25}",
        unit=unit,
        colour="green",
        dynamic_ncols=True,
    ) as pbar:
        while remaining_ids:
            done_ids, remaining_ids = ray.wait(remaining_ids, timeout=timeout)
            if not done_ids:
                break

            for done_id in done_ids:
                results[result_map[done_id]] = ray.get(done_id)
                pbar.update(1)
    return results  # type: ignore


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
s21 = s_params["o1", "o2"]
s11 = s_params["o1", "o1"]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21)), label="$S_{21}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s11)), label="$S_{11}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Top probeline (variable coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show(block=False)

# %% [markdown]
# ### Bottom probeline – fixed coupling gap
#
# :math:`S_{43}` (port o3 → o4) shows eight notches with uniform coupling
# depth because all resonators share the same 16 µm coupling gap.

# %%
s43 = s_params["o3", "o4"]
s33 = s_params["o3", "o3"]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s43)), label="$S_{43}$")
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s33)), label="$S_{33}$", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Bottom probeline (fixed coupling gap)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show(block=False)

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
plt.show(block=False)

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
# [SAX layout-aware Monte Carlo example](https://flaport.github.io/sax/nbs/examples/07_layout_aware/).
# The approach is:
#
# 1. Reuse the **standard QPDK model functions** (``straight``, ``bend_euler``,
#    ``quarter_wave_resonator_coupled``, etc.) which accept a ``cross_section``
#    parameter.
# 2. Compile the SAX circuit **once** with these models.
# 3. For each Monte Carlo trial, create a ``coplanar_waveguide(width=…, gap=…)``
#    cross-section with perturbed geometry and pass it as an override to every
#    CPW instance in the circuit.
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
plt.show(block=False)

# %% [markdown]
# The plots show that :math:`Z_0` and :math:`\varepsilon_{\mathrm{eff}}` are
# sensitive to both width and gap.  A ±0.5 µm width shift at the nominal
# 10 µm / 6 µm geometry translates to a :math:`Z_0` change of several ohms and
# a corresponding shift in resonance frequency.

# %% [markdown]
# ## Define fabrication tolerances
#
# Typical superconducting CPW fabrication tolerances are on the order of
# hundreds of nanometres to about 1 µm, depending on lithography technology
# {cite:p}`krantzQuantumEngineersGuide2019`.  We model these as independent
# Gaussian random variables.

# %%
NOMINAL_WIDTH = 10.0  # µm (centre-conductor width)
NOMINAL_GAP = 6.0  # µm (gap to ground plane)
WIDTH_SIGMA = 0.5  # µm (1σ tolerance on width)
GAP_SIGMA = 0.3  # µm (1σ tolerance on gap)

# For CI/documentation builds, reduce trials to speed up execution.
IS_CI = any((
    os.environ.get("GITHUB_ACTIONS") == "true",
    os.environ.get("CI") == "true",
))
N_TRIALS = 10 if IS_CI else 100

rng = np.random.default_rng(42)

# %% [markdown]
# ## Initialize Ray for Parallel Execution
#
# We use Ray to parallelize the Monte Carlo trials across multiple CPU cores.
#
# ::::{warning}
# If you are running this notebook inside `uv run` and encounter a `RuntimeError`
# related to `pip` or `uv` environments (e.g., "pip not found"), you may need to set
# `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` in your environment before starting the notebook.
#
# See [Ray issue #50961](https://github.com/ray-project/ray/issues/50961#issuecomment-3953219989)
# for more context on this integration issue.
# ::::
#
# ### **Resource Configuration Guide:**
#
# 1. **Total Parallelism:** By default, Ray uses all available CPU cores. To limit
#    this, use `ray.init(num_cpus=8)`.
# 2. **Cores per Worker:** In the `@ray.remote(num_cpus=1)` decorators below, you can specify
#    `num_cpus=1` (default). If your simulation uses heavy internal multi-threading,
#    increasing this (e.g., `num_cpus=2`) will reduce the number of simultaneous
#    workers to avoid over-subscribing the CPU.
# 3. **GPU Allocation:** If using GPUs, you can specify `num_gpus=0.2` to allow
#    5 workers to share a single GPU.
# 4. **Overall Workers:** The number of concurrent workers is automatically
#    calculated as `Total CPUs / Cores per Worker`.

# %% tags=["hide-output"]
if not ray.is_initialized():
    # 1. Disable Ray's automatic uv-run matching which is causing setup errors.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    runtime_env = {
        "working_dir": str(PATH.repo),
        "excludes": [
            ".git",
            ".venv",
            "build",
            "__pycache__",
            ".ruff_cache",
            ".pytest_cache",
        ],
    }
    print(f"Initializing Ray with working_dir: {PATH.repo}")

    # Initialize Ray using 80% of available CPU cores.
    total_cpus = os.cpu_count() or 1
    num_cpus = max(1, int(total_cpus * 0.8))
    print(f"Initializing Ray with {num_cpus} CPUs (80% of {total_cpus})…")
    ray.init(num_cpus=num_cpus, runtime_env=runtime_env)

# To connect to a remote Ray cluster instead, use:
# ray.init("ray://<head_node_host>:10001", runtime_env=runtime_env)

# %% [markdown]
# ## Identify cross-section-tuneable instances
#
# We inspect the circuit settings to find all instances that expose
# ``cross_section`` as a tuneable parameter.  These are the straight sections,
# bends, and resonators – i.e. every CPW element on the chip.

# %%
settings = sax.get_settings(circuit_fn)
cpw_instance_names = [name for name, s in settings.items() if "cross_section" in s]
# Exclude launcher instances — their large pad geometry (200 µm width)
# is insensitive to the small absolute variations we study here.
cpw_instance_names = [n for n in cpw_instance_names if "launcher" not in n]
print(f"{len(cpw_instance_names)} CPW instances found for MC perturbation.")

# %% [markdown]
# ## Global (systematic) tolerance simulation
#
# In this scenario **all** CPW sections on the chip receive the **same**
# width and gap perturbation in each trial, modelling a uniform fabrication
# bias across the die.  We loop over trials, creating a
# ``coplanar_waveguide(width=…, gap=…)`` cross-section for each and
# passing it as an override to every CPW instance.


# %%
@ray.remote(num_cpus=1)
def simulate_global_tolerance(
    dw: float,
    dg: float,
    freq: jnp.ndarray | ray.ObjectRef,
    netlist: dict[str, Any] | ray.ObjectRef,
    models: dict[str, Any] | ray.ObjectRef,
    instance_names: list[str],
) -> np.ndarray:
    """Ray task for a single global tolerance trial."""
    # ruff: disable[PLC0415]
    import jax.numpy as jnp
    import numpy as np
    import sax

    from qpdk import PDK
    # ruff: enable[PLC0415]

    PDK.activate()
    # Re-compiling the circuit on the worker is very fast in SAX
    circuit_fn, _ = sax.circuit(
        netlist=netlist, models=models, on_internal_port="ignore"
    )

    xs = coplanar_waveguide(width=10.0 + dw, gap=6.0 + dg)
    overrides = {name: {"cross_section": xs} for name in instance_names}
    s_trial = circuit_fn(f=freq, **overrides)
    return np.asarray(20 * jnp.log10(jnp.abs(s_trial["o1", "o2"])))


# %%
dw_global = rng.normal(0, WIDTH_SIGMA, N_TRIALS)
dg_global = rng.normal(0, GAP_SIGMA, N_TRIALS)

# Nominal S₂₁ for reference
s21_nom_db = np.asarray(20 * jnp.log10(jnp.abs(s21)))

# Put large objects in the Ray Object Store to avoid serializing them repeatedly
netlist_ref = ray.put(netlist)
models_ref = ray.put(models)
freq_ref = ray.put(freq)

# Launch tasks in parallel
print(f"Launching {N_TRIALS} parallel global trials with Ray…")
futures = [
    simulate_global_tolerance.remote(
        dw_global[i],
        dg_global[i],
        freq_ref,
        netlist_ref,
        models_ref,
        cpw_instance_names,
    )
    for i in range(N_TRIALS)
]
print("Waiting for global trials to complete…")
results_global = ray_get_with_progress(futures, desc="Global trials", timeout=1200)
print("Global trials completed.")
s21_global_db = np.array(results_global).T

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
plt.show(block=False)

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

    Returns:
        an array of shape ``(n_resonators,)`` with the dip frequencies
        in Hz, sorted in ascending order. If fewer dips are found, the missing
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
@ray.remote(num_cpus=1)
def find_resonance_dips_task(
    freq_hz: np.ndarray, s21_db: np.ndarray, n_resonators: int
) -> np.ndarray:
    """Ray task wrapper for find_resonance_dips."""
    return find_resonance_dips(freq_hz, s21_db, n_resonators=n_resonators)


# %%
# Extract resonance frequencies for the nominal and all MC trials
freq_np = np.asarray(freq)

nominal_dips = find_resonance_dips(freq_np, s21_nom_db)
n_res = int(np.sum(~np.isnan(nominal_dips)))
print(f"Nominal resonance frequencies ({n_res} detected):")
for i, f_dip in enumerate(nominal_dips):
    if not np.isnan(f_dip):
        print(f"  Resonator {i + 1}: {f_dip / 1e9:.4f} GHz")

# %%
# MC dip extraction
print(f"Extracting dips for {N_TRIALS} global trials…")
futures_dips = [
    find_resonance_dips_task.remote(freq_np, s21_global_db[:, trial], n_res)
    for trial in range(N_TRIALS)
]
mc_dips = np.array(
    ray_get_with_progress(futures_dips, desc="Extracting global dips", timeout=1200)
)

# %% [markdown]
# ### Resonance frequency distributions (global tolerance)
#
# This violin plot shows the distribution of frequency shifts for each resonator
# across all Monte Carlo trials.  A shift of 0 MHz corresponds to the nominal
# resonance frequency.

# %%
shifts_mhz = (mc_dips - nominal_dips[None, :n_res]) / 1e6

fig, ax = plt.subplots(figsize=(10, 5))
data_to_plot = [shifts_mhz[~np.isnan(shifts_mhz[:, i]), i] for i in range(n_res)]
vparts = ax.violinplot(data_to_plot, showmeans=True, showmedians=False)

# Customise violin appearance
for pc in vparts["bodies"]:
    pc.set_facecolor("C0")
    pc.set_edgecolor("black")
    pc.set_alpha(0.7)

for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
    vp = vparts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)

ax.set_xticks(np.arange(1, n_res + 1))
ax.set_xticklabels([f"{nominal_dips[i] / 1e9:.3f} GHz" for i in range(n_res)])
ax.set_xlabel("Nominal frequency [GHz]")
ax.set_ylabel("Frequency shift Δf [MHz]")
ax.set_title("Resonance frequency spread (global tolerance)")
ax.axhline(0, color="r", ls="--", lw=1, alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# %% [markdown]
# ### Frequency shift statistics
#
# We summarise the absolute and relative frequency shifts.

# %%
mean_shift = np.nanmean(shifts_mhz, axis=0)
std_shift = np.nanstd(shifts_mhz, axis=0)
max_shift = np.nanmax(np.abs(shifts_mhz), axis=0)

df_stats = pl.DataFrame({
    "Resonator": [f"{i + 1}" for i in range(n_res)],
    "Nominal [GHz]": [float(f) / 1e9 for f in nominal_dips],
    "Mean shift [MHz]": mean_shift,
    "Std [MHz]": std_shift,
    "Max |shift| [MHz]": max_shift,
})
display_dataframe(df_stats)

# %% [markdown]
# ## Per-resonator (independent) tolerance simulation
#
# In practice, fabrication variations are not perfectly uniform across a die.
# Local effects such as proximity-dependent etching cause each resonator to
# experience a **different** random perturbation.  We model this by assigning
# each resonator instance an independent cross-section perturbation.
# Routing sections (straights and bends connecting resonators) are kept at
# nominal dimensions to isolate the effect of per-resonator geometry variation.


# %%
@ray.remote(num_cpus=1)
def simulate_local_tolerance(
    trial_idx: int,
    dw_per_res: np.ndarray | ray.ObjectRef,
    dg_per_res: np.ndarray | ray.ObjectRef,
    freq: jnp.ndarray | ray.ObjectRef,
    netlist: dict[str, Any] | ray.ObjectRef,
    models: dict[str, Any] | ray.ObjectRef,
    res_names: list[str],
    non_res_names: list[str],
) -> np.ndarray:
    """Ray task for a single per-resonator tolerance trial."""
    # ruff: disable[PLC0415]
    import numpy as np
    import sax

    from qpdk import PDK
    # ruff: enable[PLC0415]

    PDK.activate()
    circuit_fn, _ = sax.circuit(
        netlist=netlist, models=models, on_internal_port="ignore"
    )

    overrides = {}
    for i, name in enumerate(res_names):
        xs_res = coplanar_waveguide(
            width=NOMINAL_WIDTH + dw_per_res[i, trial_idx],
            gap=NOMINAL_GAP + dg_per_res[i, trial_idx],
        )
        overrides[name] = {"cross_section": xs_res}

    xs_nominal = coplanar_waveguide(width=NOMINAL_WIDTH, gap=NOMINAL_GAP)
    for name in non_res_names:
        overrides[name] = {"cross_section": xs_nominal}

    s_trial = circuit_fn(f=freq, **overrides)
    return np.asarray(20 * jnp.log10(jnp.abs(s_trial["o1", "o2"])))


# %%
# Resonator instance names get independent perturbations.
# Non-resonator CPW instances (routing) stay at nominal.

# Resonator instance names
res_names = sorted(
    [n for n in cpw_instance_names if n.startswith("resonator_")],
)
non_res_names = [n for n in cpw_instance_names if n not in res_names]
n_res_instances = len(res_names)

# Draw per-resonator perturbations: shape (n_resonator_instances, N_TRIALS)
dw_per_res = rng.normal(0, WIDTH_SIGMA, (n_res_instances, N_TRIALS))
dg_per_res = rng.normal(0, GAP_SIGMA, (n_res_instances, N_TRIALS))

# Put perturbations in object store
dw_per_res_ref = ray.put(dw_per_res)
dg_per_res_ref = ray.put(dg_per_res)

# Launch tasks in parallel
print(f"Launching {N_TRIALS} parallel per-resonator trials with Ray…")
futures_local = [
    simulate_local_tolerance.remote(
        i,
        dw_per_res_ref,
        dg_per_res_ref,
        freq_ref,
        netlist_ref,
        models_ref,
        res_names,
        non_res_names,
    )
    for i in range(N_TRIALS)
]
results_local = ray_get_with_progress(
    futures_local, desc="Per-resonator trials", timeout=1200
)
s21_local_db = np.array(results_local).T

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
plt.show(block=False)

# %% [markdown]
# ### Compare global vs per-resonator spread

# %%
print("Extracting dips for local tolerance trials…")
futures_dips_local = [
    find_resonance_dips_task.remote(freq_np, s21_local_db[:, trial], n_res)
    for trial in range(N_TRIALS)
]
mc_dips_local = np.array(
    ray_get_with_progress(
        futures_dips_local, desc="Extracting local dips", timeout=1200
    )
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
ax.set_xlabel("Nominal frequency [GHz]")
ax.set_ylabel("Frequency spread (1σ) [MHz]")
ax.set_title("Frequency uncertainty: global vs per-resonator tolerance")
ax.set_xticks(x)
ax.set_xticklabels([f"{nominal_dips[i - 1] / 1e9:.3f}" for i in x])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

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
print(f"Launching {N_TRIALS} parallel width-only trials with Ray…")
futures_w = [
    simulate_global_tolerance.remote(
        dw_only[i], 0.0, freq_ref, netlist_ref, models_ref, cpw_instance_names
    )
    for i in range(N_TRIALS)
]
results_w = ray_get_with_progress(futures_w, desc="Width-only trials", timeout=1200)
s21_w_db = np.array(results_w).T

# Gap-only variation
dg_only = rng.normal(0, GAP_SIGMA, N_TRIALS)
print(f"Launching {N_TRIALS} parallel gap-only trials with Ray…")
futures_g = [
    simulate_global_tolerance.remote(
        0.0, dg_only[i], freq_ref, netlist_ref, models_ref, cpw_instance_names
    )
    for i in range(N_TRIALS)
]
results_g = ray_get_with_progress(futures_g, desc="Gap-only trials", timeout=1200)
s21_g_db = np.array(results_g).T

# %%
print("Extracting dips for sensitivity analysis trials…")
futures_dips_w = [
    find_resonance_dips_task.remote(freq_np, s21_w_db[:, trial], n_res)
    for trial in range(N_TRIALS)
]
mc_dips_w = np.array(
    ray_get_with_progress(futures_dips_w, desc="Extracting width dips", timeout=1200)
)

futures_dips_g = [
    find_resonance_dips_task.remote(freq_np, s21_g_db[:, trial], n_res)
    for trial in range(N_TRIALS)
]
mc_dips_g = np.array(
    ray_get_with_progress(futures_dips_g, desc="Extracting gap dips", timeout=1200)
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
ax.set_xlabel("Nominal frequency [GHz]")
ax.set_ylabel("Frequency spread (1σ) [MHz]")
ax.set_title("Sensitivity: width vs gap contribution to frequency uncertainty")
ax.set_xticks(x)
ax.set_xticklabels([f"{nominal_dips[i - 1] / 1e9:.3f}" for i in x])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

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
        dw_only[valid],
        shifts_w[valid, i],
        s=10,
        alpha=0.4,
        label=f"{nominal_dips[i] / 1e9:.3f} GHz",
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
        dg_only[valid],
        shifts_g[valid, i],
        s=10,
        alpha=0.4,
        label=f"{nominal_dips[i] / 1e9:.3f} GHz",
    )
ax.set_xlabel("Gap perturbation δg [µm]")
ax.set_ylabel("Frequency shift [MHz]")
ax.set_title("Gap perturbation → frequency shift")
ax.axhline(0, color="k", ls=":", lw=0.8)
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.show(block=False)

# %% [markdown]
# The scatter plots reveal a nearly **linear** relationship between the
# geometric perturbation and the resulting frequency shift, confirming that
# a first-order sensitivity model is appropriate for these tolerance levels.
# Width variations generally produce a **larger** frequency shift than gap
# variations of similar magnitude, making width control the dominant
# fabrication concern for frequency targeting.

# %% [markdown]
# ## Interactive parallel coordinates plot
#
# The following interactive plot shows how the CPW width and gap perturbations
# propagate through to each resonator's frequency shift.  Each line represents
# one Monte Carlo trial.  Use the axis ranges to filter trials interactively.

# %%
# Build a DataFrame-like dict for parallel coordinates
parcoord_data = {
    "δwidth [µm]": dw_global,
    "δgap [µm]": dg_global,
}
for i in range(n_res):
    parcoord_data[f"{nominal_dips[i] / 1e9:.3f} GHz Δf [MHz]"] = shifts_mhz[:, i]

dimensions = []
for label, values in parcoord_data.items():
    valid = ~np.isnan(values)
    clean = values[valid]
    if len(clean) == 0:
        continue
    dimensions.append(
        dict(
            label=label,
            values=clean,
            range=[float(np.min(clean)), float(np.max(clean))],
        )
    )

fig_pc = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=dw_global,
            colorscale="RdBu",
            showscale=True,
            cmin=float(np.min(dw_global)),
            cmax=float(np.max(dw_global)),
            colorbar=dict(title="δwidth [µm]"),
        ),
        dimensions=dimensions,
    )
)
fig_pc.update_layout(
    title="Monte Carlo: CPW perturbations → resonance frequency shifts",
    width=900,
    height=500,
)
fig_pc.show(block=False)

# %% [markdown]
# ## Summary
#
# | Scenario                 | Width σ [µm] | Gap σ [µm] | Typical freq. spread (1σ) [MHz] |
# | ------------------------ | ------------ | ---------- | ------------------------------- |
# | Global (systematic bias) | 0.5          | 0.3        | ~tens of MHz (correlated)       |
# | Per-resonator (local)    | 0.5          | 0.3        | ~tens of MHz (uncorrelated)     |
# | Width only               | 0.5          | 0.0        | dominant contribution           |
# | Gap only                 | 0.0          | 0.3        | secondary contribution          |
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
#   [SAX layout-aware example](https://flaport.github.io/sax/nbs/examples/07_layout_aware/))
#   or additional sources of variation (substrate permittivity, metal
#   thickness).
