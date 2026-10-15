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
# Unavailable accelerator sections are skipped. Recorded runs in
# `notebooks/data/jax_backend_comparison/` keep CUDA and MPS results together in
# the scaling plot. Each curve identifies its hardware and precision. These runs
# come from different machines, so they compare complete systems, not just backends.
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
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

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

cpu_kind = platform.processor()
if platform.system() == "Linux":
    cpu_kind = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("model name")
    )
print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
if platform.system() == "Darwin":
    chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip()
    cpu_kind = chip
    print(f"Chip : {chip}; macOS {platform.mac_ver()[0]}")
else:
    print(f"CPU model: {cpu_kind}")
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
_FREQUENCY_RANGE = (2e9, 8e9)
freq_ref = jnp.linspace(*_FREQUENCY_RANGE, 1001, dtype=jnp.float64)
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
) -> list[list[float]]:
    """Return raw wall-clock samples [s] per circuit evaluation on *device*.

    Args:
        device: JAX device on which to run the circuit.
        sizes: List of frequency-array lengths to sweep.
        n_repeats: Number of timed repetitions per size.
        single_precision: Use float32 inputs and complex64 arithmetic.

    Returns:
        Timed repetitions, grouped by frequency-array size.

    Raises:
        RuntimeError: If the circuit runs at the wrong precision or on the wrong device.
    """
    samples: list[list[float]] = []
    with jax.enable_x64(not single_precision), jax.default_device(device):
        dtype = jnp.float32 if single_precision else jnp.float64
        for n in tqdm(sizes, desc=f"Benchmarking on {device}, {dtype}"):
            freq = jax.device_put(
                jnp.linspace(*_FREQUENCY_RANGE, n, dtype=dtype), device
            )
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
            samples.append(run_times)
            tqdm.write(
                f"  n={n:>6d}: median {median * 1_000:.3f} ms, std {spread * 1_000:.3f} ms"
            )
    return samples


# %% [markdown]
# ## CPU Backend

# %%
cpu_samples = benchmark_circuit(cpu_device)
cpu_float32_samples = benchmark_circuit(cpu_device, single_precision=True)

# %% [markdown]
# ## GPU Backend
#
# This section requires a CUDA-capable GPU.  It is skipped automatically when
# no GPU is detected. Recorded CUDA measurements from another machine are
# included in the combined table and scaling plot below.

# %%
if HAS_GPU:
    gpu_samples = benchmark_circuit(gpu_device)
    gpu_float32_samples = benchmark_circuit(gpu_device, single_precision=True)
else:
    print("CUDA unavailable locally; see Recorded Runs for saved GPU measurements.")

# %% [markdown]
# ## Apple Silicon (MPS)
#
# Metal has no float64 arithmetic. We disable x64 while tracing and running this
# section, rather than allowing the plugin to silently downcast an x64 program.

# %%
if HAS_MPS:
    mps_samples = benchmark_circuit(mps_device, single_precision=True)
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
    backend_devices["CUDA float32"] = (gpu_device, True)
if HAS_MPS:
    backend_devices["MPS float32"] = (mps_device, True)

reference = np.asarray(s21_ref)
accuracy = {}
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
    accuracy[name] = {
        "output_dtype": str(result.dtype),
        "magnitude_error": float(magnitude_error),
        "complex_error": float(complex_error),
    }

# %% [markdown]
# ## Recorded Runs
#
# Set `QPDK_BENCHMARK_OUTPUT` to a JSON path to record this machine's results.
# Relative paths are rooted at the checkout, independent of the notebook directory.
# Raw repetitions, the reference transmission, workload settings, hardware and
# software versions are saved. Re-run on each machine after changing the workload.
#
# From a checkout, the notebook also reads the recorded runs in
# `notebooks/data/jax_backend_comparison/`. It checks the workload and CPU reference
# before combining them, and prefers live measurements for the current hardware.
# The table and plot include the CPU baseline from each machine. A CUDA float32
# row allows comparison with MPS at the same precision.

# %%
local_samples = {"CPU float64": cpu_samples, "CPU float32": cpu_float32_samples}
if HAS_GPU:
    local_samples["CUDA float64"] = gpu_samples
    local_samples["CUDA float32"] = gpu_float32_samples
if HAS_MPS:
    local_samples["MPS float32"] = mps_samples

workload = {
    "sizes": _BENCHMARK_SIZES,
    "n_repeats": _N_REPEATS,
    "frequency_hz": list(_FREQUENCY_RANGE),
    "solver": "filipsson_gunnar",
    "netlist": json.loads(
        json.dumps(netlist, default=lambda value: value.model_dump(mode="json"))
    ),
    "model_sha256": hashlib.sha256(
        b"".join(
            (PATH.module / name).read_bytes()
            for name in (
                "models/waveguides.py",
                "models/cpw.py",
                "models/generic.py",
                "models/constants.py",
                "tech.py",
            )
        )
    ).hexdigest(),
}
local_run = {
    "recorded_at": datetime.now(UTC).isoformat(),
    "system": platform.system(),
    "cpu": cpu_kind,
    "macos": platform.mac_ver()[0],
    "python": platform.python_version(),
    "versions": {name: version(name) for name in ("jax", "jaxlib", "sax", "jaxellip")},
    "workload": workload,
    "reference_s21": {"real": reference.real.tolist(), "imag": reference.imag.tolist()},
    "results": [],
}
if HAS_MPS:
    local_run["versions"]["jax-mps"] = version("jax-mps")
    local_run["mps_async_dispatch"] = os.environ.get("JAX_MPS_ASYNC_DISPATCH", "0")
for name, samples in local_samples.items():
    device = backend_devices[name][0]
    local_run["results"].append({
        "backend": name,
        "hardware": cpu_kind
        if device.platform == "cpu"
        else (cpu_kind + " GPU" if device.platform == "mps" else device.device_kind),
        "samples_s": samples,
        **accuracy[name],
    })

if output_path := os.environ.get("QPDK_BENCHMARK_OUTPUT"):
    (PATH.repo / output_path).write_text(
        json.dumps(local_run, indent=2) + "\n", encoding="utf-8"
    )

runs = [local_run]
for result_path in sorted(
    (PATH.repo / "notebooks/data/jax_backend_comparison").glob("*.json")
):
    saved_run = json.loads(result_path.read_text(encoding="utf-8"))
    if saved_run["workload"] != workload:
        raise ValueError(f"Recorded workload differs; regenerate {result_path.name}.")
    saved_reference = saved_run["reference_s21"]
    np.testing.assert_allclose(
        np.asarray(saved_reference["real"]) + 1j * np.asarray(saved_reference["imag"]),
        reference,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"CPU reference differs in {result_path.name}",
    )
    runs.append(saved_run)

combined_results = {}
for run in runs:
    for result in run["results"]:
        name = f"{result['hardware']} / {result['backend']}"
        combined_results.setdefault(name, (run, result))

print(
    f"{'Hardware / backend':<60} {'JAX':<8} {'Max |S21| error':>17} {'Max complex error':>19}"
)
for name, (run, result) in combined_results.items():
    samples = np.asarray(result["samples_s"])
    if samples.shape != (len(_BENCHMARK_SIZES), _N_REPEATS) or not np.all(
        np.isfinite(samples) & (samples > 0)
    ):
        raise ValueError(f"Invalid timing samples for {name}.")
    print(
        f"{name:<60} {run['versions']['jax']:<8} {result['magnitude_error']:17.6g} {result['complex_error']:19.6g}"
    )
    print(
        f"  {run['system']}; Python {run['python']}; packages {run['versions']}; recorded {run['recorded_at']}"
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


backend_times = {
    name: np.median(result["samples_s"], axis=1).tolist()
    for name, (_, result) in combined_results.items()
}

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
# model is flat below $N^\ast$ and has slope 1 above it. Hardware and precision
# are shown for every curve; the CPU rows are the baselines for their own machines.
# Points show measured medians. Error bars span the 25th to 75th percentiles
# of the repetitions, while dashed lines show the overhead fit.

# %%
_n_fit = np.geomspace(_BENCHMARK_SIZES[0], _BENCHMARK_SIZES[-1], 200)

fig, ax = plt.subplots(figsize=(11, 7))
for name, times in backend_times.items():
    t0, c = fits[name]
    samples = np.asarray(combined_results[name][1]["samples_s"])
    medians = np.asarray(times)
    lower, upper = np.percentile(samples, [25, 75], axis=1)
    line = ax.errorbar(
        _BENCHMARK_SIZES,
        medians * 1_000,
        yerr=np.asarray([medians - lower, upper - medians]) * 1_000,
        fmt="o",
        capsize=2,
        label=name,
    )[0]
    ax.plot(
        _n_fit,
        (t0 + c * _n_fit) * 1_000,
        "--",
        color=line.get_color(),
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
# | **GPU** | CUDA runs at both float64/complex128 and float32/complex64. Compare float32 against MPS and float64 against the CPU reference; consumer GPUs often have much lower float64 throughput. |
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
