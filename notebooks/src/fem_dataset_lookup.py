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
#   "matplotlib-inline",
# ]
# ///

# %% [markdown]
# # FEM Dataset Lookup in SAX Models
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
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# Circuit models of superconducting layouts need electrostatic quantities such as
# capacitance matrices, which for realistic geometries come from finite-element
# (FEM) solvers {cite:p}`jinFiniteElementMethod2014`. One solve takes minutes to
# hours, far too slow to call inside a circuit simulation or an optimisation loop.
# So a sweep over the design parameters is solved once, stored as a dataset, and
# the circuit model interpolates between the stored points.
#
# ::::{only} html
# :::{mermaid}
# flowchart LR
#     A["FEM solve<br/>per geometry"] --> B["Sweep over<br/>parameter grid"]
#     B --> C["Parquet parts or<br/>Delta table + metadata"]
#     C --> D["Dataset.grid<br/>dense array"]
#     D --> E["GridInterpolator<br/>jittable lookup"]
#     E --> F["SAX model"]
# :::
# ::::
#
# ::::{only} typst or typstpdf
# A FEM solve per geometry is swept over a parameter grid and stored as Parquet parts or a Delta table carrying its metadata; the dataset is arranged into a dense grid, wrapped in a jittable interpolator, and called from a SAX model.
# ::::
#
# {mod}`qpdk.models.datasets` stores the results as a long-format
# [Polars](https://pola.rs) table with one row per matrix entry. Each Parquet
# file (or the Delta table schema) also carries the dataset metadata: units,
# terminal and ground conventions, whether the data is synthetic, and free-form
# provenance such as the layer stack and solver. A file is therefore
# self-describing, and the parameter grid is read off the data rather than
# declared separately. Curated datasets ship in the package as Parquet parts in
# Git LFS; shared, growing datasets can live in a Delta Lake table on a cloud
# bucket.
#
# This notebook follows generation, storage, lazy queries and JAX interpolation,
# then uses the capacitance lookup in a SAX model.
#
# The bundled data comes from Palace electrostatic solves of the actual QPDK
# metal polygons, including their short leads, as perfect conductor sheets on
# silicon. A separate coplanar ground frame surrounds the device. The geometry
# and mesh settings travel with the data. It covers lengths 40–120 µm, widths
# 5–20 µm and gaps 4–10 µm; these results apply to that geometry and stack.
#
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
        "matplotlib-inline",
    ])

# %% tags=["hide-input", "hide-output"]
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sax
import xarray as xr
from matplotlib_inline.backend_inline import set_matplotlib_formats

from qpdk import logger
from qpdk.models.capacitor import plate_capacitor
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.datasets import (
    Dataset,
    GridInterpolator,
    check_maxwell,
    maxwell_to_mutual,
)
from qpdk.models.datasets.generate import write
from qpdk.models.generic import capacitor

set_matplotlib_formats("svg")

# %% [markdown]
# ## Inspect the dataset
#
# The metadata is read from the files and validated when the dataset opens. The
# results table is a plain Polars frame with one row per matrix entry, so it can
# be filtered and aggregated without loading anything into JAX.

# %%
dataset = Dataset("plate_capacitor_palace")
metadata = dataset.metadata
logger.info(f"{dataset!r}: synthetic={metadata.synthetic}")
logger.info(f"solver: {metadata.provenance['solver']}")
logger.info(f"extraction settings: {metadata.provenance['settings']}")
logger.info(f"terminals {metadata.terminals} w.r.t. {metadata.reference_ground!r}")
for axis in metadata.axes:
    logger.info(
        f"{axis.name} [{axis.unit}]: validated domain {axis.validated or 'grid span'}"
    )

# %%
dataset.table.head(8)

# %%
(
    dataset
    .scan()
    .filter(pl.col("row") != pl.col("col"))
    .group_by("gap")
    .agg(c_mutual_fF=(-pl.col("value")).mean() * 1e15)
    .sort("gap")
    .collect()
)

# %% [markdown]
# ## Define a generation script
#
# `datasets/plate_capacitor.py` is a complete experiment: its `GRID`, `SETTINGS`,
# metadata and `solve` function describe what to extract. Its inline dependencies
# let `uv run --script` create the Python 3.12 environment required by gsim.
# Copy the script beside the original in `datasets/` and edit it for another dataset.
#
# Preview the full sweep or run it from a checkout:
#
# ```bash
# uv run --script datasets/plate_capacitor.py --dry-run
# uv run --script datasets/plate_capacitor.py --processes 4
# # A local Palace container:
# uv run --script datasets/plate_capacitor.py --sif /path/to/palace.sif
# ```
#
# The script uses gsim's `ElectrostaticSim` for execution and matrix loading.
# Its sheet mesher preserves the terminal and ground geometry. The output
# carries its metadata; reading it needs neither the script nor Palace. Generator scripts stay outside the installed QPDK package.

# %%
small_grid = {"length": [40.0, 80.0], "width": [10.0], "gap": [4.0, 7.0, 10.0]}
output = Path("build/dataset-example/coarse")
generator_script = Path("datasets/plate_capacitor.py")

# %% [markdown]
# ## Run a small sweep and inspect its output
#
# Copy the generator beside the original, replace `GRID` with `small_grid`, then run it with
# `--output build/dataset-example/coarse`. For example, its grid definition becomes:
#
# ```python
# GRID = {"length": [40.0, 80.0], "width": [10.0], "gap": [4.0, 7.0, 10.0]}
# ```
#
# Set `RUN_PALACE=True` to execute the generator from a checkout. It runs the
# grid defined in `generator_script`; point this path to your edited copy.
# The default selects six existing Palace results and performs no new solves.
# Both paths produce a self-describing table with four entries per matrix.

# %% tags=["keep_output"]
RUN_PALACE = False

if RUN_PALACE:
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            "uv",
            "run",
            "--script",
            str(generator_script),
            "--output",
            str(output),
        ],
        check=True,
    )
    generated = Dataset(output)
else:
    selection = (
        dataset
        .scan()
        .filter(*(pl.col(axis).is_in(values) for axis, values in small_grid.items()))
        .collect()
    )
    generated = write(output, dataset.metadata, selection)
    logger.info("Selected six existing Palace solves; no simulator was run")
coarse = generated.grid("maxwell_capacitance", cross_section="cpw")
logger.info(f"Output: {generated.scan().select(pl.len()).collect().item()} rows")
logger.info(
    f"Dense grid: {coarse.values.shape}, terminals={generated.metadata.terminals}"
)
coarse_lookup = GridInterpolator(coarse)
logger.info(
    f"Small-sweep lookup at 60/10/7 µm [fF]:\n"
    f"{coarse_lookup(length=60.0, width=10.0, gap=7.0) * 1e15}"
)

# %% tags=["keep_output"]
(
    generated
    .scan()
    .group_by("length", "width", "gap", "status")
    .len(name="matrix_entries")
    .sort("length", "gap")
    .collect()
)

# %% [markdown]
# ## Resume and extend a sweep
#
# Rerun the generator after an interruption. Each geometry retains its inputs,
# mesh, config, solver log and output. Completed matching runs are reused;
# failed ones retry. Geometry, mesh, generator or runtime changes get a new run
# directory. The published dataset is replaced only after the sweep succeeds.
#
# Add `120.0` to the small script's length grid and rerun with the same
# `--workdir`: six geometries are reused and three are added. Use `--output`
# and `--workdir` to select locations; paths are relative to your current directory.

# ## Check mesh and domain sensitivity
#
# A denser parameter grid cannot repair a coarse FEM mesh. In a copy of the
# generator, select one geometry and vary the settings:
#
# ```python
# from dataclasses import replace
#
# GRID = {"length": [80.0], "width": [10.0], "gap": [7.0]}
# SETTINGS = replace(SETTINGS, near_mesh=0.38, save_fields=True)
# ```
#
# Run at mesh sizes 0.8, 0.55 and 0.38 µm with separate output directories,
# then increase `domain_pad` to 150 µm. Compare the Maxwell matrices and inspect
# the saved fields in ParaView, especially near metal edges and gaps. Repeat
# representative checks near the ends of any expanded geometry range.
#
# For the bundled extraction at 80/10/7 µm, the final mesh refinement changed
# entries by at most 0.57%; enlarging the domain changed them by 0.25%. Four
# held-out solves at width 10 µm differed from interpolation by at most 2.68%.
# These are checks of the bundled extraction, not a uniform accuracy bound for
# every geometry or a validation of a different mesher.

# ## Query a shared Delta table
#
# Cloud datasets use the same interface. Select the quantity and variant before
# collecting; `grid()` does this internally. Pin a version for reproducible models:
#
# ```python
# shared = Dataset("gs://my-bucket/capacitors", delta=True, version=3)
# rows = shared.scan().filter(
#     pl.col("quantity") == "maxwell_capacitance",
#     pl.col("cross_section") == "cpw",
# ).select("length", "gap", "value").collect()
# lookup = GridInterpolator(shared.grid("maxwell_capacitance", cross_section="cpw"))
# ```
#
# Scans stay lazy until `collect()`, allowing Polars to push filters and column
# selection into storage. Only the selected interpolation grid becomes a dense
# in-memory array. `dataset.table` explicitly loads and validates all rows.

# %% [markdown]
# ## Look up a Maxwell capacitance matrix
#
# For conductors $1, \dots, n$ at potentials $V_j$ above a reference ground, the
# charges are linear in the potentials, $Q_i = \sum_j C_{ij} V_j$. The
# *Maxwell* capacitance matrix $C$ is what an electrostatic FEM solver returns:
# it is symmetric, its diagonal is positive, and its off-diagonal entries are
# non-positive. The *mutual* capacitance between conductors $i \neq j$ is
# $C^{\text{m}}_{ij} = -C_{ij}$, and the capacitance of conductor $i$ to ground
# is the row sum $C^{\text{m}}_{ii} = \sum_j C_{ij}$. The mutual form is what
# appears as branch capacitances in a lumped circuit, and the Maxwell form is the
# capacitance matrix of circuit quantization
# {cite:p}`voolIntroductionQuantumElectromagnetic2017`.
# {func}`~qpdk.models.datasets.maxwell_to_mutual` converts between them, and
# {func}`~qpdk.models.datasets.check_maxwell` raises if any property above is violated.
#
# Between the solved points the lookup is multilinear: inside the grid cell
# containing a query, the result is a weighted mean of the $2^n$ cell corners,
# with weights that are products of the 1D linear weights along each axis
# {cite:p}`weiserNotePiecewiseLinear1988`. It reproduces the data exactly at grid
# points, is continuous everywhere, and needs no tuning. Outside the validated
# domain there is no data to weigh, so the result is NaN.
#
# ![Rectilinear grid of solved points; a query inside a cell is a weighted mean of the cell corners, a query outside the grid returns NaN](figures/fem-dataset-lookup.svg)
#
# {meth}`~qpdk.models.datasets.Dataset.grid` arranges the table into a dense
# `(length, width, gap, 2, 2)` array and raises if a grid point is missing or
# failed. Discrete variants such as the cross-section are chosen explicitly
# here and never interpolated. {class}`~qpdk.models.datasets.GridInterpolator` wraps
# {class}`jax.scipy.interpolate.RegularGridInterpolator` and holds plain JAX
# arrays, so it works inside {func}`jax.jit`, {func}`jax.vmap`, and
# {func}`jax.grad`.

# %%
grid = dataset.grid("maxwell_capacitance", cross_section="cpw")
c_maxwell = GridInterpolator(grid)
logger.info(repr(c_maxwell))

C = c_maxwell(length=100.0, width=10.0, gap=5.0)
logger.info(f"Maxwell matrix [fF]:\n{np.asarray(C) * 1e15}")
logger.info(f"Mutual matrix [fF]:\n{np.asarray(maxwell_to_mutual(C)) * 1e15}")
check_maxwell(C)  # raises NonPhysicalMatrixError if C is not physical

# %% [markdown]
# Outside the validated domain the lookup returns NaN. {func}`sax.interpolate_xarray`
# clamps to the nearest edge instead, which hides an out-of-range design.

# %% tags=["keep_output"]
lengths = jnp.linspace(0.0, 400.0, 401)
ours = -c_maxwell(length=lengths, width=10.0, gap=5.0)[:, 0, 1]

xarr = xr.DataArray(
    grid.values.reshape(*grid.values.shape[:3], 4),
    coords={
        **dict(zip(grid.axis_names, grid.coords, strict=True)),
        "targets": np.array(["c11", "c12", "c21", "c22"], dtype=object),
    },
)
theirs = -jax.vmap(
    lambda length: sax.interpolate_xarray(xarr, length=length, width=10.0, gap=5.0)[
        "c12"
    ]
)(lengths)

fig, ax = plt.subplots()
ax.plot(lengths, theirs * 1e15, "--", label="sax.interpolate_xarray (clamps)")
ax.plot(lengths, ours * 1e15, label="GridInterpolator (NaN outside)")
ax.axvspan(*c_maxwell.domain["length"], alpha=0.1, label="validated domain")
ax.set_xlabel(r"Length $l$ ($\text{µm}$)")
ax.set_ylabel(r"Mutual capacitance $C_{12}$ ($\text{fF}$)")
ax.legend()
plt.show()

# %% [markdown]
# ## Replace an explicit capacitance in a SAX model
#
# {func}`~qpdk.models.capacitor.plate_capacitor` computes its capacitance from a
# closed-form expression. The same circuit can take the mutual capacitance from
# the dataset instead. The analytical formula and FEM model include different
# geometry and ground assumptions, so their disagreement is visible here. The
# lookup uses only the mutual branch; a complete circuit can also include the
# extracted pad-to-ground capacitances.


# %% tags=["keep_output"]
def plate_capacitor_lookup(
    *,
    f=DEFAULT_FREQUENCY,
    length: float = 80.0,
    width: float = 5.0,
    gap: float = 7.0,
    cross_section="cpw",
) -> sax.SDict:
    """Plate capacitor with the mutual capacitance looked up from the dataset."""
    f = jnp.asarray(f)
    c_mutual = maxwell_to_mutual(c_maxwell(length=length, width=width, gap=gap))[0, 1]
    return capacitor(
        f=f, capacitance=c_mutual, z0=cpw_z0_from_cross_section(cross_section, f)
    )


f = jnp.linspace(1e9, 10e9, 201)
s_lookup = jax.jit(
    lambda gap: plate_capacitor_lookup(f=f, length=80.0, width=10.0, gap=gap)
)(7.0)
s_analytical = plate_capacitor(f=f, length=80.0, width=10.0, gap=7.0)

fig, ax = plt.subplots()
ax.plot(f / 1e9, 20 * np.log10(np.abs(s_analytical["o1", "o2"])), label="analytical")
ax.plot(
    f / 1e9, 20 * np.log10(np.abs(s_lookup["o1", "o2"])), "--", label="dataset lookup"
)
ax.set_xlabel(r"Frequency $f$ ($\text{GHz}$)")
ax.set_ylabel(r"$|S_{21}|$ ($\text{dB}$)")
ax.legend()
plt.show()

# %% [markdown]
# The lookup is differentiable, so the gap can be tuned by gradient descent
# towards a target transmission.

# %%
d_s21_d_gap = jax.jit(
    jax.grad(
        lambda gap: jnp.abs(
            plate_capacitor_lookup(f=5e9, length=80.0, width=10.0, gap=gap)["o1", "o2"]
        )
    )
)
logger.info(f"d|S21|/d gap at 5 GHz, gap = 7 µm: {float(d_s21_d_gap(7.0)):.3e} 1/µm")

# %% [markdown]
# ## Compile and query time
#
# Compile time and steady-state query time are measured separately, for a single
# point and for a batch of 10 000 points.


# %%
def benchmark(fn, *args, repeats: int = 50) -> tuple[float, float]:
    """Return (compile time, steady-state time per call) in ms."""
    jitted = jax.jit(fn)
    t0 = time.perf_counter()
    jax.block_until_ready(jitted(*args))
    compile_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(jitted(*args))
    return compile_ms, (time.perf_counter() - t0) / repeats * 1e3


rng = np.random.default_rng(0)
batch = {
    name: jnp.asarray(rng.uniform(lo, hi, 10_000))
    for name, (lo, hi) in c_maxwell.domain.items()
}
single = {name: x[0] for name, x in batch.items()}

rows = []
for label, point in [("1 point", single), ("10 000 points", batch)]:
    for method, fn in [
        ("GridInterpolator", lambda p: c_maxwell(**p)[..., 0, 1]),
        ("sax.interpolate_xarray", lambda p: sax.interpolate_xarray(xarr, **p)["c12"]),
    ]:
        compile_ms, query_ms = benchmark(fn, point)
        rows.append({
            "method": method,
            "query": label,
            "compile_ms": compile_ms,
            "query_ms": query_ms,
        })
pl.DataFrame(rows)

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
