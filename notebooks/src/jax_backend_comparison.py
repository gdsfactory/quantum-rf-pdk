# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # JAX Backend Comparison for Quantum Circuit Simulation
#
# This notebook benchmarks a SAX-based quantum circuit simulation across available
# JAX compute backends: **CPU**, **GPU** (CUDA), and **NPU** via
# [OpenVINO](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-jax.html).
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
# 3. Benchmark the circuit evaluation at a range of frequency-resolution sizes on
#    every available backend.
# 4. For the OpenVINO (NPU) backend, export the circuit as a JAX expression (JAXPR)
#    and compile it with the OpenVINO runtime.
# 5. Plot and compare performance.
#
# > **Note** – GPU and NPU sections are skipped automatically when the corresponding
# > hardware or software is not present, so the notebook runs cleanly on any CI or
# > CPU-only machine.

# %% tags=["hide-input", "hide-output"]
import math
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax

from qpdk import PDK
from qpdk.models.generic import capacitor, tee
from qpdk.models.waveguides import straight, straight_shorted
from qpdk.tech import coplanar_waveguide

PDK.activate()

# %% [markdown]
# ## Backend Detection
#
# We probe each backend and record its availability.  Unavailable backends are
# skipped gracefully throughout the notebook.

# %%
# --- CPU (always available) ---
cpu_device = jax.devices("cpu")[0]
HAS_CPU = True

# --- GPU ---
try:
    _gpu_devices = jax.devices("gpu")
    HAS_GPU = len(_gpu_devices) > 0
    gpu_device = _gpu_devices[0] if HAS_GPU else None
except Exception:
    HAS_GPU = False
    gpu_device = None

# --- OpenVINO (NPU / CPU via OpenVINO runtime) ---
try:
    import openvino as ov

    HAS_OPENVINO = True
    _ov_version_str = f"v{ov.__version__}"
except ImportError:
    HAS_OPENVINO = False
    _ov_version_str = "not installed"

print(f"CPU  : {'✓' if HAS_CPU else '✗'}")
print(f"GPU  : {'✓  (' + str(gpu_device) + ')' if HAS_GPU else '✗  (not available)'}")
print(
    f"OpenVINO: {'✓  (' + _ov_version_str + ')' if HAS_OPENVINO else '✗  (' + _ov_version_str + ')'}"
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
# :math:`f_0 = v_\phi / (4 L)`
#
# where :math:`v_\phi` is the phase velocity on the CPW.

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
s21_ref = s_ref[("o1", "o2")]

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
# JAX dispatches computation asynchronously.  We call `jax.block_until_ready` to
# ensure all pending work has completed before stopping the clock.  The first call
# for each new input shape triggers JIT compilation; we exclude this from the
# reported times by warming up before measuring.
#
# ### Note on SAX output type
#
# `circuit_fn` returns a `sax.SDict`, which is a plain Python `dict` mapping
# port-pair tuples to JAX arrays.  `jax.block_until_ready` accepts any JAX
# *pytree*, including dicts of arrays, so it works directly on the SAX output.

# %%
_BENCHMARK_SIZES = [100, 500, 1_000, 5_000, 10_000]
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
    for n in sizes:
        freq = jnp.linspace(2e9, 8e9, n)
        with jax.default_device(device):
            # Move the frequency array to the target device
            freq_d = jax.device_put(freq, device)
            # JIT warmup — first call compiles the function for this shape
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _warmup = circuit_fn(f=freq_d)
            jax.block_until_ready(_warmup)
            # Timed runs
            run_times: list[float] = []
            for _ in range(n_repeats):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t0 = time.perf_counter()
                    s = circuit_fn(f=freq_d)
                    jax.block_until_ready(s)
                    t1 = time.perf_counter()
                run_times.append(t1 - t0)
        # Use the median to reduce noise from JIT re-traces / OS scheduling
        times.append(sorted(run_times)[n_repeats // 2])
    return times


# %% [markdown]
# ## CPU Backend

# %%
print("Benchmarking on CPU …")
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
    print(f"Benchmarking on GPU ({gpu_device}) …")
    gpu_times = benchmark_circuit(gpu_device)
    for n, t in zip(_BENCHMARK_SIZES, gpu_times):
        print(f"  n={n:>6d}: {t * 1_000:.3f} ms")
else:
    print("GPU not available on this system — skipping GPU benchmark.")

# %% [markdown]
# ## OpenVINO / NPU Backend
#
# [OpenVINO](https://docs.openvino.ai/2025/) is Intel's inference runtime that
# targets CPUs, integrated GPUs, and — most importantly — dedicated Neural
# Processing Units (NPUs) available in recent Intel Core Ultra processors.
#
# ### Conversion workflow
#
# 1. **Wrap** the SAX circuit function so that it accepts a plain array and
#    returns a plain array (OpenVINO cannot consume Python dicts).
# 2. **Trace** the wrapper with `jax.make_jaxpr` to obtain a closed JAXPR.
# 3. **Convert** the JAXPR to an OpenVINO IR model with `ov.convert_model`.
# 4. **Compile** for the preferred device (NPU → GPU → CPU, in that priority
#    order).
# 5. **Benchmark** with the compiled model.
#
# ### Known SAX / JAX-to-OpenVINO considerations
#
# * SAX models use **complex128** arithmetic internally.  The OpenVINO runtime
#   may not expose complex128 directly; `ov.convert_model` translates complex
#   operations into pairs of real operations (real, imag) automatically.
# * The JAXPR is traced for a **fixed input shape** (the reference frequency
#   array).  A separate compiled model is therefore needed for each benchmark
#   size.  For production use, prefer a single representative size or use
#   OpenVINO dynamic shapes.
# * On systems without a physical NPU the OpenVINO runtime falls back to the
#   CPU plugin, which still validates the full conversion and compilation path.

# %%
ov_times: list[float] | None = None
ov_device_name: str = "unknown"

if HAS_OPENVINO:
    import openvino as ov

    core = ov.Core()
    available_ov_devices = core.available_devices
    print("Available OpenVINO devices:", available_ov_devices)

    # Select the best available device: NPU > GPU > CPU
    for _pref in ("NPU", "GPU", "CPU"):
        if _pref in available_ov_devices:
            ov_device_name = _pref
            break

    print(f"Selected OpenVINO device: {ov_device_name}")

    # ------------------------------------------------------------------
    # Wrapper: SAX SDict → plain JAX array
    # SAX returns a dict of complex S-parameters; we expose the magnitude
    # of S21 and S11 as a stacked real-valued array so that OpenVINO can
    # consume the output without encountering Python-dict outputs.
    # ------------------------------------------------------------------
    def _s_magnitudes(freq: jax.Array) -> jax.Array:
        """Return |S21|, |S11| stacked into a real-valued array."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = circuit_fn(f=freq)
        return jnp.stack([jnp.abs(s[("o1", "o2")]), jnp.abs(s[("o1", "o1")])])

    ov_times = []
    for _n in _BENCHMARK_SIZES:
        _freq = jnp.linspace(2e9, 8e9, _n)

        # 1. Trace the function for this specific input shape
        try:
            _jaxpr = jax.make_jaxpr(_s_magnitudes)(_freq)
        except Exception as exc:
            print(f"  n={_n}: make_jaxpr failed — {exc}")
            ov_times.append(float("nan"))
            continue

        # 2. Convert to OpenVINO IR
        try:
            _ov_model = ov.convert_model(_jaxpr)
        except Exception as exc:
            print(f"  n={_n}: ov.convert_model failed — {exc}")
            ov_times.append(float("nan"))
            continue

        # 3. Compile for the selected device
        try:
            _compiled = core.compile_model(_ov_model, ov_device_name)
        except Exception as exc:
            print(f"  n={_n}: compile_model failed — {exc}")
            ov_times.append(float("nan"))
            continue

        # 4. Benchmark: create one infer request and reuse it
        _infer = _compiled.create_infer_request()
        _freq_np = jax.device_get(_freq)  # copy to host as a NumPy array

        # Warmup
        _infer.infer([_freq_np])

        _run_times: list[float] = []
        for _ in range(_N_REPEATS):
            _t0 = time.perf_counter()
            _infer.infer([_freq_np])
            _t1 = time.perf_counter()
            _run_times.append(_t1 - _t0)

        _median_t = sorted(_run_times)[_N_REPEATS // 2]
        ov_times.append(_median_t)
        print(f"  n={_n:>6d}: {_median_t * 1_000:.3f} ms")

else:
    print(
        "OpenVINO is not installed on this system — skipping NPU/OpenVINO benchmark.\n"
        "Install it with:  pip install openvino"
    )

# %% [markdown]
# ## Performance Comparison
#
# The plot below shows median evaluation time as a function of the number of
# frequency points swept.  A linear relationship on a log–log scale indicates
# :math:`O(N)` scaling; super-linear growth suggests memory-bandwidth or
# kernel-launch overhead is significant.

# %%
fig, ax = plt.subplots()
ax.plot(_BENCHMARK_SIZES, [t * 1_000 for t in cpu_times], "o-", label="CPU (JAX)")

if gpu_times is not None:
    ax.plot(
        _BENCHMARK_SIZES, [t * 1_000 for t in gpu_times], "s-", label="GPU (JAX/CUDA)"
    )

if ov_times is not None:
    _valid = [(n, t) for n, t in zip(_BENCHMARK_SIZES, ov_times) if not math.isnan(t)]
    if _valid:
        _ns, _ts = zip(*_valid)
        ax.plot(
            _ns, [t * 1_000 for t in _ts], "^-", label=f"OpenVINO ({ov_device_name})"
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
# ## Scaling Analysis
#
# We fit a power-law model :math:`t = a \cdot N^b` to the CPU timing data to
# characterise the asymptotic scaling exponent *b*.  Perfect vectorised code has
# :math:`b = 1`; values above 1 indicate super-linear overhead (e.g. matrix
# factorisation inside the circuit solver).

# %%
_sizes_arr = np.array(_BENCHMARK_SIZES, dtype=float)
_cpu_arr = np.array(cpu_times, dtype=float)

# Fit log(t) = log(a) + b*log(N) via linear regression
_log_n = np.log(_sizes_arr)
_log_t = np.log(_cpu_arr)
_b, _log_a = np.polyfit(_log_n, _log_t, 1)
_a = np.exp(_log_a)

print(f"CPU scaling fit:  t ≈ {_a * 1e6:.2f} µs · N^{_b:.3f}")

_n_fit = np.logspace(np.log10(_sizes_arr[0]), np.log10(_sizes_arr[-1]), 200)
_t_fit = _a * _n_fit**_b

fig, ax = plt.subplots()
ax.scatter(_sizes_arr, _cpu_arr * 1_000, zorder=5, label="CPU measurements")
ax.plot(_n_fit, _t_fit * 1_000, "--", label=f"fit: $N^{{{_b:.2f}}}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of frequency points")
ax.set_ylabel("Median time per evaluation [ms]")
ax.set_title("CPU scaling behaviour")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Backend | Notes |
# |---------|-------|
# | **CPU** | Always available; good baseline.  JAX JIT provides substantial speed-up over NumPy equivalents. |
# | **GPU** | Requires `jax[cuda]` and a CUDA-capable GPU.  Beneficial for very large frequency sweeps or batched optimisation. |
# | **OpenVINO (NPU)** | Requires `openvino` package.  Converts the JAX trace to an IR model, then compiles for the best available device (NPU > GPU > CPU).  Most beneficial on Intel Core Ultra platforms with a dedicated NPU. |
#
# ### SAX / JAX integration notes
#
# * **Output type**: `sax.circuit` returns an `SDict` (a plain `dict` mapping
#   port-pair tuples to JAX arrays).  To use the circuit with `jax.make_jaxpr`
#   or any system that expects *array* outputs, wrap it in a helper function that
#   extracts or combines the relevant entries.
# * **Complex arithmetic**: SAX models default to `complex128`.  JAX handles this
#   natively; OpenVINO decomposes complex ops into real/imaginary pairs during
#   conversion.  If conversion fails due to dtype issues, consider calling
#   `jax.config.update("jax_enable_x64", True)` early and/or lowering precision
#   with `freq.astype(jnp.complex64)`.
# * **Fixed input shape**: `jax.make_jaxpr` traces for a concrete shape.  Each
#   distinct frequency-array length produces a separate compiled artefact.  Use a
#   single canonical size in production or explore OpenVINO dynamic-shape support.
