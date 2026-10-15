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
# Apple Silicon GPUs use the experimental [jax-mps](https://github.com/tillahoffmann/jax-mps)
# plugin (macOS, Python >=3.12, jaxlib 0.11.x). SAX currently requires JAX <0.10,
# so this needs a separate environment with an explicit dependency override:
#
# ```bash
# uv venv --python 3.13 .venv-mps
# uv pip install --python .venv-mps/bin/python -e ".[models]" \
#   --override .github/jax-mps-overrides.txt jax-mps==0.11.0 jupytext nbconvert ipykernel
# ```
#
# This override is experimental and does not change the project lockfile. Use the
# environment's Python directly; `uv run` would restore the locked JAX version.
# MPS supports single precision only. Compare it with the CPU float32 row, and
# check the accuracy table against the CPU float64 reference before using timings.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# This notebook benchmarks a SAX-based quantum circuit simulation on the **CPU**,
# **GPU** (CUDA), and **Apple Silicon** (MPS) backends of JAX.
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
# The page shows outputs saved by the last accelerator run (see `AGENTS.md`).
# Unavailable accelerator sections are skipped. CUDA and MPS outputs come from
# separate machines; a saved run describes only the machine recorded below.
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
import os
import platform
import subprocess
import time
import warnings
from importlib.metadata import version

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
jax.config.update("jax_default_device", jax.devices("cpu")[0])

PDK.activate()

# %% [markdown]
# ## Backend Detection
#
# We probe each backend and record its availability.  The GPU sections are skipped
# when no GPU is found, unless `QPDK_REQUIRE_GPU=1` is set, which turns a missing GPU
# into an error. `QPDK_REQUIRE_MPS=1` does the same for Apple Silicon. Set the
# appropriate flag when saving outputs for the docs.

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

try:
    mps_device = jax.devices("mps")[0]
except RuntimeError:
    mps_device = None
HAS_MPS = mps_device is not None

if os.environ.get("QPDK_REQUIRE_MPS") == "1" and not HAS_MPS:
    raise RuntimeError(
        "QPDK_REQUIRE_MPS=1 but JAX found no MPS device. "
        "Install jax-mps in the separate environment described above."
    )

print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
if platform.system() == "Darwin":
    chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip()
    print(f"Chip : {chip}; macOS {platform.mac_ver()[0]}")
print(f"Python: {platform.python_version()}")
print(f"JAX  : {jax.__version__}; jaxlib {version('jaxlib')}; SAX {sax.__version__}")
print(f"CPU  : ✓  ({cpu_device.device_kind})")
print(
    f"GPU  : {'✓  (' + gpu_device.device_kind + ')' if HAS_GPU else '✗  (not available)'}"
)
print(
    f"MPS  : {'available (' + mps_device.device_kind + ')' if HAS_MPS else 'not available'}"
)
if HAS_MPS:
    print(f"jax-mps: {version('jax-mps')}")
    print(f"JAX_MPS_ASYNC_DISPATCH={os.environ.get('JAX_MPS_ASYNC_DISPATCH', '0')}")
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
            "settings": {"length": 500, "cross_section": cross_section},
        },
        "feedline2": {
            "component": "straight",
            "settings": {"length": 500, "cross_section": cross_section},
        },
        "cap": {
            "component": "capacitor",
            "settings": {"capacitance": 20e-15, "z0": 50},
        },
        "res": {
            "component": "straight_shorted",
            "settings": {"length": 4000, "cross_section": cross_section},
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
freq_ref = jnp.linspace(2e9, 8e9, 1001, dtype=jnp.float64)
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
# warming up before measuring. Each row reports the median and standard deviation
# of ten timed evaluations.
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
    *,
    single_precision: bool = False,
) -> list[float]:
    """Return median wall-clock time [s] per circuit evaluation on *device*.

    Args:
        device: JAX device on which to run the circuit.
        sizes: List of frequency-array lengths to sweep.
        n_repeats: Number of timed repetitions per size.
        single_precision: Use float32 inputs and complex64 arithmetic.

    Returns:
        List of median evaluation times in seconds, one per entry in *sizes*.

    Raises:
        RuntimeError: If the circuit runs at the wrong precision or on the wrong device.
    """
    times: list[float] = []
    with jax.enable_x64(not single_precision), jax.default_device(device):
        dtype = jnp.float32 if single_precision else jnp.float64
        for n in tqdm(sizes, desc=f"Benchmarking on {device}, {dtype}"):
            freq = jax.device_put(jnp.linspace(2e9, 8e9, n, dtype=dtype), device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = jax.block_until_ready(circuit_jit(f=freq))
                expected_dtype = jnp.complex64 if single_precision else jnp.complex128
                if any(
                    value.dtype != expected_dtype or value.device != device
                    for value in result.values()
                ):
                    raise RuntimeError(
                        "Circuit output precision or device is incorrect."
                    )

                run_times: list[float] = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    jax.block_until_ready(circuit_jit(f=freq))
                    run_times.append(time.perf_counter() - t0)
            median = float(np.median(run_times))
            spread = float(np.std(run_times, ddof=1))
            times.append(median)
            tqdm.write(
                f"  n={n:>6d}: median {median * 1_000:.3f} ms, std {spread * 1_000:.3f} ms"
            )
    return times


# %% [markdown]
# ## CPU Backend

# %%
cpu_times = benchmark_circuit(cpu_device)
cpu_float32_times = benchmark_circuit(cpu_device, single_precision=True)

# %% [markdown]
# ## GPU Backend
#
# This section requires a CUDA-capable GPU.  It is skipped automatically when
# no GPU is detected.

# %%
gpu_times: list[float] | None = None

if HAS_GPU:
    gpu_times = benchmark_circuit(gpu_device)
else:
    print("GPU not available on this system — skipping GPU benchmark.")

# %% [markdown]
# ## Apple Silicon (MPS)
#
# Metal has no float64 arithmetic. We disable x64 while tracing and running this
# section, rather than allowing the plugin to silently downcast an x64 program.

# %%
mps_times: list[float] | None = None
if HAS_MPS:
    mps_times = benchmark_circuit(mps_device, single_precision=True)
else:
    print("MPS not available on this system; skipping Apple Silicon benchmark.")

# %% [markdown]
# ## Accuracy Against the CPU Float64 Reference
#
# All rows use the same 1001-point sweep. The table reports the maximum absolute
# error in transmission magnitude, $\lvert S_{21}\rvert$, and in complex
# transmission. These are absolute linear errors, not dB or relative errors near
# the resonance dip. This check is outside the timed region.

# %%
backend_devices = {
    "CPU float64": (cpu_device, False),
    "CPU float32": (cpu_device, True),
}
if HAS_GPU:
    backend_devices["CUDA float64"] = (gpu_device, False)
if HAS_MPS:
    backend_devices["MPS float32"] = (mps_device, True)

reference = np.asarray(s21_ref)
print(
    f"{'Backend':<16} {'Output dtype':<12} {'Max |S21| error':>17} {'Max complex error':>19}"
)
for name, (device, single_precision) in backend_devices.items():
    with jax.enable_x64(not single_precision), jax.default_device(device):
        freq = jax.device_put(
            jnp.asarray(
                np.asarray(freq_ref),
                dtype=jnp.float32 if single_precision else jnp.float64,
            ),
            device,
        )
        result = np.asarray(jax.block_until_ready(circuit_jit(f=freq))["o1", "o2"])
    if not np.isfinite(result).all():
        raise RuntimeError(f"{name} produced non-finite transmission.")
    magnitude_error = np.max(np.abs(np.abs(result) - np.abs(reference)))
    complex_error = np.max(np.abs(result - reference))
    print(
        f"{name:<16} {result.dtype!s:<12} {magnitude_error:17.6g} {complex_error:19.6g}"
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


backend_times = {"CPU float64": cpu_times, "CPU float32": cpu_float32_times}
if gpu_times is not None:
    backend_times["CUDA float64"] = gpu_times
if mps_times is not None:
    backend_times["MPS float32"] = mps_times

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
for (name, times), marker in zip(backend_times.items(), "os^D"):
    t0, c = fits[name]
    (line,) = ax.plot(
        _BENCHMARK_SIZES,
        np.asarray(times) * 1_000,
        marker,
        label=name,
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
# | **CPU** | Float64 reference and float32 baseline for comparisons at matching precision. |
# | **GPU** | Requires a CUDA build of JAX (e.g. `jax[cuda12]`) and a CUDA-capable GPU.  Compare the fitted per-point costs to see whether it pays off: SAX evaluates in `complex128`, and many consumer GPUs run 64-bit floating point at a small fraction of their 32-bit rate. |
# | **MPS** | Apple Silicon via experimental jax-mps; float32/complex64 only. Requires the separate JAX 0.11 environment above. |
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
