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
# This notebook loads recorded CPU, CUDA, and Apple Silicon (MPS) measurements
# from Parquet files in `notebooks/data/jax_backend_comparison/`. Documentation CI
# executes the analysis and renders the plots; no GPU or live benchmark is needed.
# The sweeps cover 100 to 30 million frequency points.
# Each curve identifies its hardware and precision. These are different machines,
# so the plot compares complete systems, with a CPU baseline for each machine.
#
# ## Recording measurements
#
# Install the `models` extra and fetch the data with Git LFS from a checkout:
#
# ```bash
# uv sync --extra models
# git lfs pull --include="notebooks/data/jax_backend_comparison/*.parquet"
# ```
#
# The recorder builds a capacitively coupled quarter-wave resonator with 500 µm
# feedlines, a 20 fF coupling capacitor, and a 4000 µm shorted CPW stub (width
# 10 µm, gap 6 µm). SAX's pure-JAX `filipsson_gunnar` solver runs on every device;
# the default KLU solver's FFI kernels are CPU-only. For each sweep size it warms
# up `jax.jit`, checks output precision and device, then times ten evaluations,
# synchronizing the entire output with `jax.block_until_ready` on each call.
#
# ```bash
# QPDK_REQUIRE_GPU=1 uv run --extra models --with "jax[cuda12]" \
#   python -m qpdk.samples.benchmark_jax_backends \
#   notebooks/data/jax_backend_comparison/cuda.parquet
# ```
#
# Apple Silicon uses experimental [jax-mps](https://github.com/tillahoffmann/jax-mps),
# which requires JAX 0.11.x. Installing a new environment with normal dependency
# resolution requires [SAX to allow that version](https://github.com/gdsfactory/sax/pull/133).
# The saved MPS measurements use an existing experimental JAX 0.11 environment.
# Once compatible SAX is available, install `jax-mps` alongside `qpdk[models]`
# and use `QPDK_REQUIRE_MPS=1` with the same recorder, saving `apple-m5.parquet`.
# MPS supports float32/complex64 only; compare it with CPU and CUDA float32.
# The saved runs retain their original package versions and model source hash.
#
# Parquet stores typed nested records: one machine run per file, with all raw
# repetitions, workload settings, software versions, errors, and real/imaginary
# CPU reference arrays. Commit these files through Git LFS. Notebook outputs are
# stripped; changing plot styling only requires rerendering this analysis.
# See the {ref}`extras reference <notebook-extras>` for the model dependencies.

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

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

import qpdk.colormaps  # ruff: ignore[unused-import] - registers the style colormap
from qpdk.config import PATH

plt.style.use(
    PATH.repo / "docs" / f"{os.environ.get('QPDK_MPL_STYLE', 'qpdk')}.mplstyle"
)

# %% [markdown]
# ## Recorded Runs
#
# Runs must share the same workload and CPU float64 reference before their
# timings can be compared. The reference uses 1001 points from 2 to 8 GHz.
# All errors below are absolute linear transmission errors, not dB errors or
# relative errors near the resonance dip.

# %%
result_paths = sorted(
    (PATH.repo / "notebooks/data/jax_backend_comparison").glob("*.parquet")
)
if not result_paths:
    raise FileNotFoundError(
        "No benchmark records found. Fetch the Parquet files with Git LFS."
    )
runs = [pq.read_table(path).to_pylist()[0] for path in result_paths]
workload = runs[0]["workload"]
_BENCHMARK_SIZES = workload["sizes"]
_N_REPEATS = workload["n_repeats"]
reference = np.asarray(runs[0]["reference_s21"]["real"]) + 1j * np.asarray(
    runs[0]["reference_s21"]["imag"]
)
for result_path, run in zip(result_paths, runs, strict=True):
    if run["workload"] != workload:
        raise ValueError(f"Recorded workload differs in {result_path.name}.")
    trace = run["reference_s21"]
    np.testing.assert_allclose(
        np.asarray(trace["real"]) + 1j * np.asarray(trace["imag"]),
        reference,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"CPU reference differs in {result_path.name}",
    )

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
# ## Reference Transmission
#
# The saved CPU float64 trace shows the resonance of the coupled resonator.

# %%
f_start, f_stop = workload["frequency_hz"]
freq_ghz = np.linspace(f_start, f_stop, len(reference)) / 1e9
fig, ax = plt.subplots()
ax.plot(freq_ghz, 20 * np.log10(np.abs(reference)), label="$S_{21}$")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title("Coupled resonator reference")
ax.legend()
plt.show()

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
#
# The lower panel compares each accelerator with the CPU on its own machine at
# matching precision. Values above 1 mean the GPU is faster. These ratios use
# measured medians; the fitted lines do not establish a crossover on their own.

# %%
_n_fit = np.geomspace(_BENCHMARK_SIZES[0], _BENCHMARK_SIZES[-1], 200)

fig, (ax, speedup_ax) = plt.subplots(
    2, 1, sharex=True, figsize=(11, 10), gridspec_kw={"height_ratios": [2, 1]}
)
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

    run, result = combined_results[name]
    backend, precision = result["backend"].split()
    if backend != "CPU":
        cpu = next(
            row for row in run["results"] if row["backend"] == f"CPU {precision}"
        )
        speedup = np.median(cpu["samples_s"], axis=1) / medians
        speedup_ax.plot(
            _BENCHMARK_SIZES, speedup, "o-", color=line.get_color(), label=name
        )

ax.set_xscale("log")
ax.set_yscale("log")
speedup_ax.set_xlabel("Number of frequency points")
ax.set_ylabel("Median time per evaluation [ms]")
ax.set_title("JAX backend performance — coupled resonator circuit")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
speedup_ax.set_yscale("log")
speedup_ax.set_ylabel("Speedup over matching CPU")
speedup_ax.axhline(1, color="0.4", ls=":", label="CPU baseline")
speedup_ax.legend(fontsize=9)
speedup_ax.grid(True, which="both", ls="--", alpha=0.5)
speedup_ax.set_xlim(_BENCHMARK_SIZES[0] / 1.2, _BENCHMARK_SIZES[-1] * 1.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Backend | Notes |
# |---------|-------|
# | **CPU** | Float64 reference and float32 baseline for comparisons at matching precision. |
# | **GPU** | CUDA runs at both float64/complex128 and float32/complex64. Compare float32 against MPS and float64 against the CPU reference; consumer GPUs often have much lower float64 throughput. |
# | **MPS** | Apple Silicon via experimental jax-mps; float32/complex64 only. New environments await SAX support for JAX 0.11. |
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
