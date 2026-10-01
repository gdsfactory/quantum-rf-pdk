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
# FEM extractions are expensive, so a sweep is run once and stored as a reusable
# dataset. {mod}`qpdk.datasets` stores each dataset as a directory with
#
# - `manifest.toml`: units, terminal and ground conventions, stack, geometry and
#   solver revisions, accuracy, and the parameter grid, kept in ordinary Git;
# - `results/*.parquet`: one long-format table per batch of solves, kept in Git LFS
#   and read lazily with [Polars](https://pola.rs).
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
from qpdk.datasets import (
    Dataset,
    GridInterpolator,
    maxwell_to_mutual,
    maxwell_violations,
)
from qpdk.models.capacitor import plate_capacitor
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.generic import capacitor

# %% [markdown]
# ## Inspect the dataset
#
# The manifest is validated when the dataset opens. The results table is a plain
# Polars frame with one row per matrix entry, so it can be filtered and
# aggregated without loading anything into JAX.

# %%
dataset = Dataset("plate_capacitor_synthetic")
manifest = dataset.manifest
logger.info(
    f"{dataset!r}: synthetic={manifest.synthetic}, solver={manifest.solver.name}"
)
logger.info(
    f"terminals {manifest.conventions.terminals} w.r.t. {manifest.conventions.reference_ground!r}"
)
for axis in manifest.axes:
    logger.info(f"{axis.name} [{axis.unit}]: {axis.values}")

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
# {meth}`~qpdk.datasets.Dataset.grid` arranges the table into a dense
# `(length, width, gap, 2, 2)` array and raises if a grid point is missing or
# failed. Discrete variants such as the cross-section are chosen explicitly
# here and never interpolated. {class}`~qpdk.datasets.GridInterpolator` holds plain
# JAX arrays, so it works inside {func}`jax.jit`, {func}`jax.vmap`, and
# {func}`jax.grad`.

# %%
grid = dataset.grid("maxwell_capacitance", cross_section="cpw")
c_maxwell = GridInterpolator(grid)
logger.info(repr(c_maxwell))

C = c_maxwell(length=100.0, width=10.0, gap=5.0)
logger.info(f"Maxwell matrix [fF]:\n{np.asarray(C) * 1e15}")
logger.info(f"Mutual matrix [fF]:\n{np.asarray(maxwell_to_mutual(C)) * 1e15}")
logger.info(f"Physical checks: {maxwell_violations(C) or 'ok'}")

# %% [markdown]
# Outside the validated domain the lookup returns NaN. {func}`sax.interpolate_xarray`
# clamps to the nearest edge instead, which hides an out-of-range design.

# %%
lengths = jnp.linspace(0.0, 400.0, 401)
ours = -c_maxwell(length=lengths, width=10.0, gap=5.0)[:, 0, 1]

xarr = xr.DataArray(
    grid.values.reshape(*grid.values.shape[:3], 4),
    coords={
        **{a.name: np.asarray(a.values) for a in grid.axes},
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
batch = {a.name: jnp.asarray(rng.uniform(*a.domain, 10_000)) for a in grid.axes}
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
