# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [raw] tags=["remove-cell"]
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///

# %% [markdown]
# # JAX Backend Comparison for Quantum Circuit Simulation
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra:
#
# ```bash
# uv add "qpdk[models]"
# # or, from a checkout of this repository:
# uv sync --extra models
# # or with pip:
# pip install "qpdk[models]"
# ```
#
# The GPU benchmark additionally needs a CUDA-capable GPU and a CUDA build of JAX, e.g.
# `pip install "jax[cuda12]"`. It is skipped if no GPU is found.
#
# The Apple Silicon benchmark needs a Mac with an Apple GPU and the experimental
# [jax-mps](https://github.com/tillahoffmann/jax-mps) plugin (`pip install jax-mps`,
# Python ≥ 3.12).  jax-mps is built for `jax`/`jaxlib` 0.11.x only.  It is skipped if
# no MPS device is found.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# This notebook benchmarks a SAX-based quantum circuit simulation on the **CPU**,
# **GPU** (CUDA) and **Apple Silicon GPU** (Metal, through jax-mps) backends of JAX.
#
# JAX provides a unified programming model that can transparently target different
# hardware accelerators.  For large-scale circuit sweeps — e.g. sweeping thousands of
# frequency points or running optimisation loops — hardware acceleration can give
# significant speed-ups.
#
# ## Workflow
#
# 1. Build a coupled resonator circuit with the QPDK model library and SAX.
# 2. Detect which compute backends are available on the current system.
# 3. Benchmark the jit-compiled circuit at a range of frequency-resolution sizes on
#    every available backend.
# 4. Plot and compare performance, and fit a fixed-overhead plus per-point cost model
#    to each backend.
#
# :::{note}
# The documentation build does not execute this notebook, since CI runners have no GPU.
# The page shows the outputs saved by the last run on an accelerator machine (see
# `AGENTS.md`); if no such run has been saved yet, it shows the code only.  Sections
# for accelerators that are not present are skipped.
# :::
#
# :::{note}
# An earlier version of this notebook also benchmarked Intel NPUs through
# [OpenVINO](https://docs.openvino.ai/2026/).  That section was removed: OpenVINO's
# JAX/Flax conversion is experimental, its JAX conversion notebook was deleted from the
# OpenVINO notebooks in the 2026.0 release
# ([release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#previous-2026-releases)),
# and the conversion could not handle the complex-valued SAX circuit.
# :::

# %% tags=["hide-input", "hide-output"]
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
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import datetime
import json
import os
import platform
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from tqdm import tqdm

from qpdk import PDK
from qpdk.config import PATH
from qpdk.models.generic import capacitor, tee
from qpdk.models.waveguides import straight, straight_shorted
from qpdk.tech import coplanar_waveguide

jax.config.update("jax_enable_x64", True)

PDK.activate()

# %% [markdown]
# ## Backend Detection
#
# We probe each backend and record its availability.  The GPU sections are skipped
# when no GPU is found, unless `QPDK_REQUIRE_GPU=1` is set, which turns a missing GPU
# into an error.  Set it when saving outputs for the docs, so a saved run can never
# silently lack the GPU results.  `QPDK_REQUIRE_MPS=1` does the same for the Apple
# Silicon GPU.
#
# Once installed, jax-mps registers itself as a JAX platform and may become the default
# backend.  We pin the default device to the CPU so that only the benchmark sections
# run on an accelerator.

# %%
# --- CPU (always available) ---
cpu_device = jax.devices("cpu")[0]

# --- GPU ---
try:
    gpu_device = jax.devices("gpu")[0]
except RuntimeError:
    gpu_device = None
HAS_GPU = gpu_device is not None

if os.environ.get("QPDK_REQUIRE_GPU") == "1" and not HAS_GPU:
    raise RuntimeError(
        "QPDK_REQUIRE_GPU=1 but JAX found no GPU. Install a CUDA build of JAX, "
        'e.g. `pip install "jax[cuda12]"`, and check `nvidia-smi`.'
    )

# --- Apple Silicon GPU (jax-mps) ---
try:
    mps_device = jax.devices("mps")[0]
except RuntimeError:
    mps_device = None
HAS_MPS = mps_device is not None

if os.environ.get("QPDK_REQUIRE_MPS") == "1" and not HAS_MPS:
    raise RuntimeError(
        "QPDK_REQUIRE_MPS=1 but JAX found no MPS device. Install the jax-mps plugin "
        "(`pip install jax-mps`) on a Mac with Apple Silicon."
    )

jax.config.update("jax_default_device", cpu_device)

print(f"JAX  : {jax.__version__}")
print(f"CPU  : ✓  ({cpu_device.device_kind})")
print(
    f"GPU  : {'✓  (' + gpu_device.device_kind + ')' if HAS_GPU else '✗  (not available)'}"
)
print(
    f"MPS  : {'✓  (' + mps_device.device_kind + ')' if HAS_MPS else '✗  (not available)'}"
)
print(f"\nDefault JAX backend: {jax.default_backend()}")

# %% [markdown]
# ## Circuit Setup
#
# We build a capacitively-coupled quarter-wave resonator on a coplanar waveguide
# (CPW) feed-line.  The topology is:
#
# ```
# port o1 ──[feedline1]──[tee]──[feedline2]── port o2
#                         │
#                        [cap]
#                         │
#                        [res]  (shorted stub ≈ λ/4)
# ```
#
# The resonance frequency of a shorted λ/4 stub of length *L* is approximately
#
# $$
# f_0 = v_\phi / (4 L)
# $$
#
# where $v_\phi$ is the phase velocity on the CPW.
#
# We solve the circuit with the `filipsson_gunnar` backend instead of sax's
# default `klu` backend: the latter routes the solve through
# [klujax](https://github.com/gdsfactory/klujax), whose FFI kernels are only
# registered for the CPU platform, so a circuit built with it cannot run on the
# GPU at all.  `filipsson_gunnar` is pure JAX and yields the same
# S-parameters (differences at the level of machine epsilon).

# %%
cross_section = coplanar_waveguide(width=10, gap=6)

# Component models available to the SAX circuit solver
circuit_models = {
    "straight": straight,
    "capacitor": capacitor,
    "straight_shorted": straight_shorted,
    "tee": tee,
}

# Netlist: component instances with their physical parameters
netlist = {
    "instances": {
        "feedline1": {
            "component": "straight",
            "settings": {"length": 500, "media": cross_section},
        },
        "feedline2": {
            "component": "straight",
            "settings": {"length": 500, "media": cross_section},
        },
        "cap": {
            "component": "capacitor",
            "settings": {"capacitance": 20e-15, "z0": 50},
        },
        "res": {
            "component": "straight_shorted",
            "settings": {"length": 4000, "media": cross_section},
        },
        "tee": "tee",
    },
    "connections": {
        "feedline1,o2": "tee,o1",
        "tee,o2": "feedline2,o1",
        "tee,o3": "cap,o1",
        "cap,o2": "res,o1",
    },
    "ports": {
        "o1": "feedline1,o1",
        "o2": "feedline2,o2",
    },
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    circuit_fn, circuit_info = sax.circuit(
        netlist=netlist, models=circuit_models, backend="filipsson_gunnar"
    )

print("Circuit built successfully.")
print("External ports:", list(netlist["ports"].keys()))

# %% [markdown]
# ### Verify the Simulation
#
# Run the circuit at moderate resolution and plot the transmission to confirm
# the expected resonance dip appears.

# %%
freq_ref = jnp.linspace(2e9, 8e9, 1001)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s_ref = circuit_fn(f=freq_ref)

freq_ghz = freq_ref / 1e9
s21_ref = s_ref["o1", "o2"]

fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * jnp.log10(jnp.abs(s21_ref)), label="$S_{21}$")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Coupled resonator — reference simulation")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Benchmarking Helper
#
# `sax.circuit` returns a plain Python function, so every call re-dispatches each
# primitive from Python.  Timing it directly would mostly measure that dispatch
# overhead, so we wrap it in `jax.jit` first.  The first call for each new input shape
# compiles the function for the target device; we exclude it from the reported times by
# warming up before measuring.
#
# JAX dispatches computation asynchronously.  We call `jax.block_until_ready` to ensure
# all pending work has completed before stopping the clock.  `circuit_fn` returns a
# `sax.SDict`, a plain `dict` mapping port-pair tuples to JAX arrays, and
# `jax.block_until_ready` accepts any such pytree.
#
# With `single_precision=True` the benchmark runs inside `jax.enable_x64(False)`, so
# the frequency array is `float32` and the circuit evaluates in `complex64`.  Metal has
# no 64-bit floating point, so this is the only precision the MPS backend can run.

# %%
circuit_jit = jax.jit(circuit_fn)

_BENCHMARK_SIZES = np.geomspace(100, 100_000, 10, dtype=int).tolist()
_N_REPEATS = 10  # number of timed runs per size


def benchmark_circuit(
    device: jax.Device,
    sizes: list[int] = _BENCHMARK_SIZES,
    n_repeats: int = _N_REPEATS,
    single_precision: bool = False,
) -> list[float]:
    """Return median wall-clock time [s] per circuit evaluation on *device*.

    Args:
        device: JAX device on which to run the circuit.
        sizes: List of frequency-array lengths to sweep.
        n_repeats: Number of timed repetitions per size.
        single_precision: Evaluate in ``float32``/``complex64`` instead of
            ``float64``/``complex128``.

    Returns:
        List of median evaluation times in seconds, one per entry in *sizes*.
    """
    times: list[float] = []
    precision = "float32" if single_precision else "float64"
    for n in tqdm(sizes, desc=f"Benchmarking on {device} ({precision})"):
        with jax.enable_x64(not single_precision), warnings.catch_warnings():
            # Committing the input to the device makes the jitted function run there
            freq = jax.device_put(jnp.linspace(2e9, 8e9, n), device)
            warnings.simplefilter("ignore")
            # Warmup: compiles the function for this shape and device
            jax.block_until_ready(circuit_jit(f=freq))

            run_times: list[float] = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(circuit_jit(f=freq))
                run_times.append(time.perf_counter() - t0)
        # Use the median to reduce noise from OS scheduling
        times.append(float(np.median(run_times)))
    return times


# %% [markdown]
# ## CPU Backend

# %%
cpu_times = benchmark_circuit(cpu_device)
for n, t in zip(_BENCHMARK_SIZES, cpu_times):
    print(f"  n={n:>6d}: {t * 1_000:.3f} ms")

# %% [markdown]
# ## GPU Backend
#
# This section requires a CUDA-capable GPU.  It is skipped automatically when
# no GPU is detected.

# %%
gpu_times: list[float] | None = None

if HAS_GPU:
    gpu_times = benchmark_circuit(gpu_device)
    for n, t in zip(_BENCHMARK_SIZES, gpu_times):
        print(f"  n={n:>6d}: {t * 1_000:.3f} ms")
else:
    print("GPU not available on this system — skipping GPU benchmark.")

# %% [markdown]
# ## Single Precision
#
# Apple GPUs have no 64-bit floating point, and jax-mps narrows `complex128` arrays to
# `complex64` (with a warning).  Before benchmarking the MPS backend we therefore
# check how much accuracy the circuit loses in single precision, by evaluating the
# reference sweep on the CPU at `float32` and comparing it with the `float64` result.

# %%
with jax.enable_x64(False), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s_ref_f32 = circuit_jit(f=jnp.linspace(2e9, 8e9, freq_ref.size))

s21_f64 = np.asarray(s21_ref)
s21_f32 = np.asarray(s_ref_f32["o1", "o2"])
print(f"dtype                 : {s21_f32.dtype} (vs {s21_f64.dtype})")
print(f"max |ΔS21|            : {np.max(np.abs(s21_f32 - s21_f64)):.2e}")
print(
    "max |Δ|S21|| [dB]     : "
    f"{np.max(np.abs(20 * np.log10(np.abs(s21_f32)) - 20 * np.log10(np.abs(s21_f64)))):.2e}"
)

# %% [markdown]
# The MPS timings below are compared with a `float32` CPU baseline, so that
# both sides do the same arithmetic.

# %%
cpu_f32_times = benchmark_circuit(cpu_device, single_precision=True)
for n, t in zip(_BENCHMARK_SIZES, cpu_f32_times):
    print(f"  n={n:>6d}: {t * 1_000:.3f} ms")

# %% [markdown]
# ## Apple Silicon GPU Backend (MPS)
#
# This section requires a Mac with Apple Silicon and the
# [jax-mps](https://github.com/tillahoffmann/jax-mps) plugin.  It is skipped
# automatically when no MPS device is detected.  jax-mps is experimental: it maps
# StableHLO to [MLX](https://github.com/ml-explore/mlx) operations, and not every JAX
# feature is supported.  The `filipsson_gunnar` backend used here needs only complex
# element-wise arithmetic, which jax-mps supports.

# %%
mps_times: list[float] | None = None

if HAS_MPS:
    mps_times = benchmark_circuit(mps_device, single_precision=True)
    for n, t in zip(_BENCHMARK_SIZES, mps_times):
        print(f"  n={n:>6d}: {t * 1_000:.3f} ms")
else:
    print("MPS not available on this system — skipping Apple Silicon benchmark.")

# %% [markdown]
# ## Saved Accelerator Runs
#
# No single machine has both a CUDA GPU and an Apple Silicon GPU, so each accelerator
# run is saved as a JSON file in `notebooks/data/jax_backend_comparison/` (tracked with
# Git LFS) and the comparison below loads every saved run.  A run stores the
# accelerator timings together with the CPU timings measured on the same machine, as
# timings from different machines are not directly comparable.
#
# Set `QPDK_SAVE_BENCHMARKS=1` to save this machine's run.  It is written to
# `cuda.json` or `mps.json`, depending on the accelerator; a machine without one does
# not save anything.  Runs measured in this session replace saved runs for the same
# accelerator.

# %%
RESULTS_DIR = PATH.repo / "notebooks" / "data" / "jax_backend_comparison"

live_times = {"CPU": cpu_times, "CPU (float32)": cpu_f32_times}
if gpu_times is not None:
    live_times["GPU"] = gpu_times
if mps_times is not None:
    live_times["MPS (float32)"] = mps_times

if HAS_GPU:
    live_key, accelerator = "cuda", gpu_device.device_kind
elif HAS_MPS:
    live_key, accelerator = "mps", mps_device.device_kind
else:
    live_key, accelerator = "cpu", None

live_run = {
    "accelerator": accelerator,
    "cpu": platform.processor() or platform.machine(),
    "platform": platform.platform(),
    "jax": jax.__version__,
    "date": datetime.date.today().isoformat(),
    "sizes": _BENCHMARK_SIZES,
    "times": live_times,
}

if os.environ.get("QPDK_SAVE_BENCHMARKS") == "1" and accelerator is not None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{live_key}.json"
    out_path.write_text(json.dumps(live_run, indent=2) + "\n")
    print(f"Saved this run to {out_path.relative_to(PATH.repo)}")

runs: dict[str, dict] = {}
for path in sorted(RESULTS_DIR.glob("*.json")):
    try:
        runs[path.stem] = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        # An un-fetched Git LFS pointer file is not JSON
        print(f"Skipping {path.name}: not valid JSON (run `git lfs pull`?)")
runs[live_key] = live_run

for key, run in runs.items():
    source = "this session" if key == live_key else f"saved {run['date']}"
    print(
        f"{key:>4}: {run['accelerator'] or 'no accelerator'} + {run['cpu']} "
        f"(JAX {run['jax']}, {source})"
    )

# %% [markdown]
# ## Scaling Analysis
#
# The cost of one evaluation has two parts: a fixed overhead per call (dispatch and
# kernel launches) and a cost per frequency point.  We therefore model the median time as
#
# $$
# t(N) = t_0 + c \, N ,
# $$
#
# where $t_0$ is the overhead and $c$ the cost per point.  A single power law
# $t = a N^b$ cannot describe this: the overhead dominates at small $N$, so the
# curve is flat there and only becomes linear once $c N \gg t_0$, near the
# crossover size $N^\ast = t_0 / c$.
#
# The timings span several decades, so an ordinary least-squares fit would only follow
# the largest sizes.  We minimise the *relative* residuals instead, which is still a
# linear problem: dividing $t_0 + c N_i \approx t_i$ by $t_i$ gives
# $t_0 / t_i + c \, N_i / t_i \approx 1$.  Both parameters are physically non-negative;
# if the unconstrained solution makes one of them negative, we drop that term and refit.


# %%
def fit_overhead_model(sizes: list[int], times: list[float]) -> tuple[float, float]:
    """Fit ``t = t0 + c * N`` to timings, minimising the relative residuals.

    Args:
        sizes: Number of frequency points per measurement.
        times: Median evaluation time [s] per measurement.

    Returns:
        Fixed overhead ``t0`` [s] and cost per frequency point ``c`` [s].
    """
    n = np.asarray(sizes, dtype=float)
    t = np.asarray(times, dtype=float)
    design = np.column_stack([1 / t, n / t])
    (t0, c), *_ = np.linalg.lstsq(design, np.ones_like(t), rcond=None)
    if t0 < 0:  # pure per-point cost: minimise sum((c * N / t - 1) ** 2)
        x = n / t
        t0, c = 0.0, x.sum() / (x**2).sum()
    elif c < 0:  # pure overhead
        x = 1 / t
        t0, c = x.sum() / (x**2).sum(), 0.0
    return float(t0), float(c)


fits = {
    key: {
        name: fit_overhead_model(run["sizes"], times)
        for name, times in run["times"].items()
    }
    for key, run in runs.items()
}

for key, run in runs.items():
    print(f"[{key}] {run['accelerator'] or 'no accelerator'} + {run['cpu']}")
    for name, (t0, c) in fits[key].items():
        t_model = t0 + c * np.asarray(run["sizes"])
        max_error = np.max(np.abs(t_model / np.asarray(run["times"][name]) - 1))
        print(
            f"  {name}: t0 = {t0 * 1e6:8.1f} µs, c = {c * 1e9:8.2f} ns/point, "
            f"N* = t0/c ≈ {t0 / c if c > 0 else float('inf'):,.0f} points, "
            f"max relative error {max_error:.0%}"
        )

# %% [markdown]
# ## Performance Comparison
#
# The plot below shows the median evaluation time against the number of frequency
# points, with the fitted model as a dashed line for each backend, in one panel per
# machine.  On log–log axes the model is flat below $N^\ast$ and has slope 1 above it.

# %%
_runtime_labels = {"GPU": "JAX/CUDA", "MPS (float32)": "JAX/jax-mps"}

fig, axes = plt.subplots(
    1, len(runs), figsize=(6.4 * len(runs), 4.8), sharey=True, squeeze=False
)
for ax, (key, run) in zip(axes[0], runs.items()):
    sizes = np.asarray(run["sizes"])
    n_fit = np.geomspace(sizes[0], sizes[-1], 200)
    for (name, times), marker in zip(run["times"].items(), "os^v"):
        t0, c = fits[key][name]
        (line,) = ax.plot(
            sizes,
            np.asarray(times) * 1_000,
            marker,
            label=f"{name} ({_runtime_labels.get(name, 'JAX')})",
        )
        ax.plot(
            n_fit,
            (t0 + c * n_fit) * 1_000,
            "--",
            color=line.get_color(),
            label=f"{name} fit: {t0 * 1e6:.0f} µs + {c * 1e9:.0f} ns · N",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of frequency points")
    ax.set_title(f"{run['accelerator'] or 'CPU only'}\n{run['cpu']}", fontsize="medium")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", ls="--", alpha=0.5)

axes[0, 0].set_ylabel("Median time per evaluation [ms]")
fig.suptitle("JAX backend performance — coupled resonator circuit")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Backend | Notes |
# |---------|-------|
# | **CPU** | Always available; good baseline, with the lowest fixed overhead per call. |
# | **GPU** | Requires a CUDA build of JAX (e.g. `jax[cuda12]`) and a CUDA-capable GPU.  Compare the fitted per-point costs to see whether it pays off: SAX evaluates in `complex128`, and many consumer GPUs run 64-bit floating point at a small fraction of their 32-bit rate. |
# | **CPU (float32)** | Same circuit in `complex64`.  Baseline for the MPS backend; see the single-precision section for the accuracy cost. |
# | **MPS** | Requires Apple Silicon and the experimental jax-mps plugin (`jax`/`jaxlib` 0.11.x).  Single precision only, since Metal has no 64-bit floating point. |
#
# ### SAX / JAX integration notes
#
# * **Jit the circuit**: `sax.circuit` returns an un-jitted function.  Wrap it in
#   `jax.jit` before calling it repeatedly, or Python dispatch overhead dominates the
#   runtime of small and medium sweeps.
# * **Output type**: `sax.circuit` returns an `SDict` (a plain `dict` mapping
#   port-pair tuples to JAX arrays), which `jax.jit` and `jax.block_until_ready`
#   accept as a pytree.
# * **Fixed input shape**: `jax.jit` compiles for a concrete input shape.  Each
#   distinct frequency-array length triggers a new compilation, so reuse one size where
#   you can.
