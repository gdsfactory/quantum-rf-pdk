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
# This notebook inspects the bundled dataset, looks up a capacitance matrix inside
# {func}`jax.jit`, and replaces the analytical capacitance of a SAX model with the
# lookup.
#
# ```{warning}
# `plate_capacitor_synthetic` is generated from an analytical formula as a
# placeholder for Palace output. It demonstrates the format, not the physics.
# ```

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
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sax
import xarray as xr

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
from qpdk.models.generic import capacitor

# %% [markdown]
# ## Inspect the dataset
#
# The metadata is read from the files and validated when the dataset opens. The
# results table is a plain Polars frame with one row per matrix entry, so it can
# be filtered and aggregated without loading anything into JAX.

# %%
dataset = Dataset("plate_capacitor_synthetic")
metadata = dataset.metadata
logger.info(f"{dataset!r}: synthetic={metadata.synthetic}")
logger.info(f"provenance: {metadata.provenance}")
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

# %%
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
# the dataset instead. Because the synthetic dataset was generated from that
# expression, both models agree at grid points.


# %%
def plate_capacitor_lookup(
    *,
    f=DEFAULT_FREQUENCY,
    length: float = 26.0,
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
    lambda gap: plate_capacitor_lookup(f=f, length=120.0, width=10.0, gap=gap)
)(7.0)
s_analytical = plate_capacitor(f=f, length=120.0, width=10.0, gap=7.0)

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
            plate_capacitor_lookup(f=5e9, length=120.0, width=10.0, gap=gap)["o1", "o2"]
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
