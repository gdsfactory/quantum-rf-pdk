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
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# This notebook benchmarks a SAX-based quantum circuit simulation on the **CPU** and
# **GPU** (CUDA) backends of JAX.
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
# The page shows the outputs saved by the last `just run-jax-backend-notebook` run on a
# CUDA machine; if no such run has been saved yet, it shows the code only.  On a CPU-only
# machine the GPU sections are skipped.
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
import hashlib
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from tqdm import tqdm

from qpdk import PDK
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
# into an error.  `just run-jax-backend-notebook` sets it, so a saved run can never
# silently lack the GPU results.

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

print(f"JAX  : {jax.__version__}")
print(f"CPU  : ✓  ({cpu_device.device_kind})")
print(
    f"GPU  : {'✓  (' + gpu_device.device_kind + ')' if HAS_GPU else '✗  (not available)'}"
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
    circuit_fn, circuit_info = sax.circuit(netlist=netlist, models=circuit_models)

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

# %%
circuit_jit = jax.jit(circuit_fn)

_BENCHMARK_SIZES = np.geomspace(100, 100_000, 10, dtype=int).tolist()
_N_REPEATS = 10  # number of timed runs per size


def benchmark_circuit(
    device: jax.Device,
    sizes: list[int] = _BENCHMARK_SIZES,
    n_repeats: int = _N_REPEATS,
) -> list[float]:
    """Return median wall-clock time [s] per circuit evaluation on *device*.

    Args:
        device: JAX device on which to run the circuit.
        sizes: List of frequency-array lengths to sweep.
        n_repeats: Number of timed repetitions per size.

    Returns:
        List of median evaluation times in seconds, one per entry in *sizes*.
    """
    times: list[float] = []
    for n in tqdm(sizes, desc=f"Benchmarking on {device}"):
        # Committing the input to the device makes the jitted function run there
        freq = jax.device_put(jnp.linspace(2e9, 8e9, n), device)
        with warnings.catch_warnings():
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


backend_times = {"CPU": cpu_times}
if gpu_times is not None:
    backend_times["GPU"] = gpu_times

fits = {
    name: fit_overhead_model(_BENCHMARK_SIZES, times)
    for name, times in backend_times.items()
}

for name, (t0, c) in fits.items():
    t_model = t0 + c * np.asarray(_BENCHMARK_SIZES)
    max_error = np.max(np.abs(t_model / np.asarray(backend_times[name]) - 1))
    print(
        f"{name}: t0 = {t0 * 1e6:8.1f} µs, c = {c * 1e9:8.2f} ns/point, "
        f"N* = t0/c ≈ {t0 / c if c > 0 else float('inf'):,.0f} points, "
        f"max relative error {max_error:.0%}"
    )

# %% [markdown]
# ## Performance Comparison
#
# The plot below shows the median evaluation time against the number of frequency
# points, with the fitted model as a dashed line for each backend.  On log–log axes the
# model is flat below $N^\ast$ and has slope 1 above it.

# %%
_n_fit = np.geomspace(_BENCHMARK_SIZES[0], _BENCHMARK_SIZES[-1], 200)

fig, ax = plt.subplots()
for (name, times), marker in zip(backend_times.items(), "os"):
    t0, c = fits[name]
    (line,) = ax.plot(
        _BENCHMARK_SIZES,
        np.asarray(times) * 1_000,
        marker,
        label=f"{name} (JAX{'/CUDA' if name == 'GPU' else ''})",
    )
    ax.plot(
        _n_fit,
        (t0 + c * _n_fit) * 1_000,
        "--",
        color=line.get_color(),
        label=f"{name} fit: {t0 * 1e6:.0f} µs + {c * 1e9:.0f} ns · N",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of frequency points")
ax.set_ylabel("Median time per evaluation [ms]")
ax.set_title("JAX backend performance — coupled resonator circuit")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Backend | Notes |
# |---------|-------|
# | **CPU** | Always available; good baseline, with the lowest fixed overhead per call. |
# | **GPU** | Requires a CUDA build of JAX (e.g. `jax[cuda12]`) and a CUDA-capable GPU.  Compare the fitted per-point costs to see whether it pays off: SAX evaluates in `complex128`, and many consumer GPUs run 64-bit floating point at a small fraction of their 32-bit rate. |
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

# %% tags=["hide-input", "hide-output"]
# The documentation shows the outputs of a saved run of this notebook, so the fingerprint below
# ties them to the source: CI recomputes it from notebooks/src/jax_backend_comparison.py and
# rejects the saved outputs when the two have drifted apart.
_source = next(
    (
        path
        # The kernel starts in the directory of the notebook it writes
        for path in (
            Path("src/jax_backend_comparison.py"),
            Path("notebooks/src/jax_backend_comparison.py"),
            Path("jax_backend_comparison.py"),
        )
        if path.is_file()
    ),
    None,
)
if _source is not None:
    print(f"Executed source SHA256: {hashlib.sha256(_source.read_bytes()).hexdigest()}")
else:
    print(
        "Executed source SHA256: unavailable (run from the repository root to record it)"
    )
