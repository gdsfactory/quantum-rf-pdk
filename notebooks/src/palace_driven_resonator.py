# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [raw] tags=["remove-cell"]
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "gsim[meshwell]>=0.7.0",
# ]
# ///

# %% [markdown]
# # Palace Driven Simulation of a Coupled CPW Resonator
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra and [gsim](https://gdsfactory.github.io/gsim/):
#
# ```bash
# uv add "qpdk[models]" "gsim[meshwell]>=0.7.0"
# # or, from a checkout of this repository:
# uv sync --extra models --group palace
# # or with pip:
# pip install "qpdk[models]" "gsim[meshwell]>=0.7.0"
# ```
#
# gsim currently needs Python 3.12. The Palace solve runs on the
# [GDSFactory+](https://gdsfactory.com) cloud, so an account is also required.
#
# See the {ref}`extras reference <notebook-extras>` for what each qpdk extra installs.
# ::::
#
# This notebook builds a compact CPW resonator coupled to a feedline from
# QPDK cells, converts the etch layer to conductor geometry, and runs a
# [Palace](https://awslabs.github.io/palace/) driven simulation to extract the
# S-parameters around the resonance near 7.78 GHz. The loaded quality factor is
# estimated from the S21 dip, and the electric field is plotted at resonance.
#
# Palace is an open-source 3D finite-element electromagnetic solver supporting
# eigenmode, driven (S-parameter), and electrostatic simulations.

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
        "gsim[meshwell]>=0.7.0",
    ])

# %% [markdown]
# ## Layout
#
# The resonator couples to a feedline routed with S-bends. Airbridges tie the ground
# planes together along the feedline, suppressing the slotline mode. A `SIM_AREA` box
# around the layout sets the simulation domain.

# %%
import gdsfactory as gf

from qpdk import PDK, cells
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.tech import LAYER, route_bundle_sbend_cpw

PDK.activate()


@gf.cell
def resonator_compact(coupling_gap: float = 20.0) -> gf.Component:
    """Compact coupled resonator with S-bend CPW routes.

    Args:
        coupling_gap: Gap between the resonator and the feedline in µm.

    Returns:
        Component with feedline ports ``o1`` and ``o2`` and a ``SIM_AREA`` box.
    """
    c = gf.Component()

    res = c << cells.resonator_coupled(
        coupling_straight_length=300, coupling_gap=coupling_gap
    )
    res.movex(-res.size_info.width / 4)

    left = c << cells.straight()
    right = c << cells.straight()

    w = res.size_info.width + 100
    left.move((-w, 0))
    right.move((w, 0))

    route_bundle_sbend_cpw(
        c,
        [left["o2"], right["o1"]],
        [res["coupling_o1"], res["coupling_o2"]],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=250.0, airbridge_padding=20.0
        ),
    )

    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 100))

    c.add_port(name="o1", port=left["o1"])
    c.add_port(name="o2", port=right["o2"])
    return c


component = resonator_compact(coupling_gap=15.0)
_c = component.copy()
_c.draw_ports()
_c.plot()

# %% [markdown]
# ## Convert etch layers to conductor geometry
#
# QPDK draws the CPW as a negative mask: `M1_ETCH` marks where the metal is removed.
# Subtracting it from the simulation area gives the superconductor, which is written
# to a new component together with the substrate and vacuum regions.

# %%
import warnings

import klayout.db as kdb
from gsim.common.polygon_utils import decimate

sim_area_layer = (LAYER.SIM_AREA[0], LAYER.SIM_AREA[1])
etch_layer = (LAYER.M1_ETCH[0], LAYER.M1_ETCH[1])

CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = component.kdb_cell.layout()
sim_region = kdb.Region(
    component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))

etch_polys = decimate(list(etch_region.each()))
etch_region = kdb.Region()
for poly in etch_polys:
    etch_region.insert(poly)

if sim_region.is_empty():
    warnings.warn("No polygons found on SIM_AREA", stacklevel=2)
if etch_region.is_empty():
    warnings.warn("No polygons found on M1_ETCH", stacklevel=2)

conductor_region = sim_region - etch_region

etched = gf.Component("etched_component")
el = etched.kdb_cell.layout()
for name, region in [
    ("SUPERCONDUCTOR", conductor_region),
    ("SUBSTRATE", sim_region),
    ("VACUUM", sim_region),
]:
    idx = el.layer(*CPW_LAYERS[name])
    etched.kdb_cell.shapes(idx).insert(region)

for port in component.ports:
    etched.add_port(name=port.name, port=port)

etched.plot()

# %% [markdown]
# ## Configure the simulation
#
# The layer stack is a 500 µm sapphire substrate under a zero-thickness aluminium
# sheet, with 500 µm of vacuum above. Both feedline ends get a CPW port matching the
# default QPDK cross-section (10 µm centre conductor, 6 µm gaps), and the sweep is
# centred on the resonance with fields saved at 7.78 GHz.

# %%
from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace import DrivenSim

substrate_thickness = 500
vacuum_thickness = 500

stack = LayerStack(pdk_name="qpdk")
stack.layers["SUBSTRATE"] = Layer(
    name="SUBSTRATE",
    gds_layer=(1, 0),
    zmin=0.0,
    zmax=substrate_thickness,
    thickness=substrate_thickness,
    material="sapphire",
    layer_type="dielectric",
)
stack.layers["SUPERCONDUCTOR"] = Layer(
    name="SUPERCONDUCTOR",
    gds_layer=(2, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness,
    thickness=0,
    material="aluminum",
    layer_type="conductor",
)
stack.layers["VACUUM"] = Layer(
    name="VACUUM",
    gds_layer=(3, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness + vacuum_thickness,
    thickness=vacuum_thickness,
    material="vacuum",
    layer_type="dielectric",
)
stack.dielectrics = [
    {
        "name": "substrate",
        "zmin": 0.0,
        "zmax": substrate_thickness,
        "material": "sapphire",
    },
    {
        "name": "vacuum",
        "zmin": substrate_thickness,
        "zmax": substrate_thickness + vacuum_thickness,
        "material": "vacuum",
    },
]
stack.materials = {
    "sapphire": MATERIALS_DB["sapphire"].to_dict(),
    "aluminum": MATERIALS_DB["aluminum"].to_dict(),
    "vacuum": MATERIALS_DB["vacuum"].to_dict(),
}

sim = DrivenSim()
sim.set_geometry(etched)
sim.set_stack(stack)
sim.add_cpw_port("o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.add_cpw_port("o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=300, save_fields_at=[7.78e9])

# %% [markdown]
# ## Mesh and run
#
# The mesh is generated locally with gmsh; the solve itself runs on the GDSFactory+
# cloud. `check_cache=True` reuses a finished job with the same inputs.

# %%
sim.set_output_dir("./sim_qpdk_resonator")
sim.mesh(preset="default")
sim.plot_mesh(show_groups=["superconductor", "P", "sapphire", "vacuum"])

# %%
sim.write_config()
results = sim.run(check_cache=True)

# %%
results.plot_interactive()

# %%
results.plot_interactive(phase=True)

# %% [markdown]
# ## Resonance and quality factor
#
# The resonance is the S21 minimum. The loaded quality factor $Q_L = f_0 / \Delta f$
# is first estimated from the half-depth bandwidth $\Delta f$ of the dip, then refined
# by fitting a Lorentzian to $|S_{21}|$:
#
# $$
# |S_{21}(f)| = b - \frac{d}{1 + \left(2 Q_L (f - f_0) / f_0\right)^2},
# $$
#
# with baseline $b$ and dip depth $d$.

# %%
from pathlib import Path

import numpy as np
import pyvista as pv
from gsim.palace.results import load_fields
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

pv.OFF_SCREEN = True

results_dir = Path(results.files["port-S.csv"]).parent

res_idx = np.argmin(results.s21.db)
freq_ghz = results.freq[res_idx]
print(f"Resonance at {freq_ghz:.4f} GHz (S21 = {results.s21.db[res_idx]:.1f} dB)")

s21_db = results.s21.db
s21_mag = 10 ** (s21_db / 20.0)

# Half-depth bandwidth of the dip
s21_min_mag = s21_mag[res_idx]
baseline = np.median(s21_mag)  # transmission away from resonance
dip_depth = baseline - s21_min_mag
half_depth = s21_min_mag + dip_depth / 2
half_power_db = 20 * np.log10(half_depth)

left_idx = np.argmin(np.abs(s21_db[:res_idx] - half_power_db))
right_idx = res_idx + np.argmin(np.abs(s21_db[res_idx:] - half_power_db))

f_left = results.freq[left_idx]
f_right = results.freq[right_idx]
bw_ghz = f_right - f_left

Q_bw = freq_ghz / bw_ghz if bw_ghz > 0 else np.inf
print(f"Bandwidth: {bw_ghz:.6f} GHz")
print(f"Loaded quality factor Q ~ {Q_bw:.0f}")


def lorentzian_dip(f, f0, Q, dip_depth, baseline):
    """Lorentzian dip in |S21| around a resonance at ``f0``."""
    return baseline - dip_depth / (1 + (2 * Q * (f - f0) / f0) ** 2)


freq = results.freq
mask = np.abs(freq - freq_ghz) < 0.01  # ±10 MHz fit window
if mask.sum() > 5:
    popt, pcov = curve_fit(
        lorentzian_dip,
        freq[mask],
        s21_mag[mask],
        p0=[freq_ghz, Q_bw, dip_depth, baseline],
        bounds=(
            [freq_ghz - 0.01, 100, 0, 0],
            [freq_ghz + 0.01, 1e6, 2 * baseline, 2 * baseline],
        ),
    )
    f0_fit, Q_fit, dip_fit, base_fit = popt
    Q_err = np.sqrt(np.diag(pcov))[1]
    print(f"Lorentzian fit: f0 = {f0_fit:.4f} GHz, Q = {Q_fit:.0f} ± {Q_err:.0f}")
else:
    print("Insufficient points for Lorentzian fit.")

# %% [markdown]
# ## Field at resonance
#
# Palace saves the fields for the last excited port. The volume solution is sliced at
# the conductor plane and interpolated onto a regular grid for a top view of $|E|$.

# %%
import matplotlib.pyplot as plt

field_excitation = 2
vol = load_fields(results_dir, excitation=field_excitation)
print(f"Volume: {vol.n_points:,} points, {vol.n_cells:,} cells")

vol_slice = vol.slice(normal="z", origin=(0, 0, substrate_thickness))

vpts = vol_slice.points
x_pad, y_pad = 5, 5
xi = np.linspace(vpts[:, 0].min() - x_pad, vpts[:, 0].max() + x_pad, 800)
yi = np.linspace(vpts[:, 1].min() - y_pad, vpts[:, 1].max() + y_pad, 400)
Xi, Yi = np.meshgrid(xi, yi)


def plot_topview(pts, data, title, cmap="turbo"):
    """Interpolate point data onto the grid and plot it as a top view."""
    gi = griddata(pts[:, :2], data, (Xi, Yi), method="linear")
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.nanpercentile(gi, 98)
    im = ax.pcolormesh(Xi, Yi, gi, cmap=cmap, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    valid = ~np.isnan(gi)
    if valid.any():
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        ax.set_xlim(xi[cols][0], xi[cols][-1])
        ax.set_ylim(yi[rows][0], yi[rows][-1])
    fig.tight_layout(pad=0.5)
    plt.show()


plot_topview(
    vpts,
    np.linalg.norm(vol_slice.point_data["E_real"], axis=1),
    f"Electric field |E| at {freq_ghz:.4f} GHz (V/m)",
)
