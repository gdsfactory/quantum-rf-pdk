# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # COMSOL transmon capacitance and field
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `comsol` extra (`uv sync --extra comsol`). It installs `MPh`, the Python client for
# COMSOL, and nothing else: not COMSOL and not a license, so the build and solve cells need a local
# installation and cannot run on Colab. See the {ref}`extras reference <notebook-extras>`.
# ::::
#
# This notebook runs a stationary **Electrostatics** solve on a COMSOL sheet model of a QPDK double-pad
# transmon. The figures are **saved cell outputs** of licensed solves; without a license the COMSOL cells are
# skipped.
#
# ## What is modelled, and what is solved
#
# The device is a QPDK {py:func}`~qpdk.cells.transmon.double_pad_transmon_with_bbox`: two pads joined
# **only** through the Josephson junction, with a SQUID loop at the centre, so the qubit frequency follows
# from $E_J$ and $E_C = e^2 / 2C$ {cite:p}`kochChargeinsensitiveQubitDesign2007a`. The sheet model cannot
# represent the junction overlap or its barrier, so the extraction uses an **EM-only copy** with the junction
# layers removed: the SQUID loop and its leads are absent from the solved geometry.
#
# The solve adds **Electrostatics**, not electromagnetic waves, and returns `es.C11`, the driven pad's
# capacitance to the grounded rest of the chip, plus `es.intWe`, which agree through
# $2 W_e / V^2 = C_{11}$ {cite:p}`m.pozarMicrowaveEngineering2012`. It is a **quasi-static extraction**, with
# no Josephson inductance and no resonance solved.
#
# ::::{only} html
# ```{mermaid}
# flowchart LR
#     A["Transmon cell"] --> B["EM-only copy"] --> C["Extracted layout"] --> D["Sheet model"] --> E["Electrostatics"] --> F["Mesh, stationary solve"] --> G["C11, intWe, V, normE"]
# ```
# ::::
#
# ::::{only} typst or typstpdf
# The pipeline: transmon cell, EM-only copy with the junction layers removed, extracted layout of pads and
# ground plane, sheet model with air over silicon, Electrostatics solve driving the left pad, mesh, and the
# capacitance, energy, and field results.
# ::::
#
# ::::{admonition} Reading the numbers on this page
# :class: warning
#
# The values come from licensed electrostatic solves, but they are not a validated device prediction:
# the SQUID loop and leads are absent, and $C_{11}$ is a one-terminal value rather than the two-pad
# differential capacitance that sets $E_C$, so the $f_{LC}$ estimate further down is neither a COMSOL
# eigenmode nor the transmon $f_{01}$ {cite:p}`blaisCircuitQuantumElectrodynamics2021`. The plots below show
# how far it moves with the mesh, the lateral margin, and the air height.
# ::::
#
# **References:**
# - QPDK transmon physics and the double-pad cell: {py:func}`~qpdk.cells.transmon.double_pad_transmon_with_bbox`
# - [COMSOL Electrostatics interface](https://doc.comsol.com/6.3/doc/com.comsol.help.acdc/acdc_ug_electric_fields.07.002.html)
# - [COMSOL capacitance example](https://doc.comsol.com/6.3/doc/com.comsol.help.models.acdc.capacitor_dc/capacitor_dc.html)
# - [COMSOL Reference Manual: Analyzing Model Convergence and Accuracy](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_modeling.19.043.html)
# - [MPh tutorial](https://mph.readthedocs.io/en/stable/tutorial.html)

# %% [markdown]
# ## Setup and imports

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
        "qpdk[comsol] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])
    print(
        "Note: this installs the Python client only. COMSOL itself and its "
        "license are not pip-installable, so the build and solve cells below "
        "cannot run in Colab, and the result cells will report that no exported "
        "results are present."
    )

# %% tags=["hide-input", "hide-output"]
import json
import math
from itertools import pairwise
from operator import itemgetter
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import axes as mpl_axes, font_manager
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon

from qpdk import PDK
from qpdk.cells.transmon import double_pad_transmon_with_bbox
from qpdk.config import PATH
from qpdk.simulation import prepare_comsol_layout
from qpdk.tech import LAYER

try:
    import mph

    from qpdk.simulation import COMSOL
except ImportError:
    mph = None
    COMSOL = None

PDK.activate()

MPH_AVAILABLE = mph is not None


def _outfit_titles() -> None:
    """Draw plot titles in Outfit bold, matching the documentation headings."""
    original_set_title = mpl_axes.Axes.set_title

    def _set_title(self: mpl_axes.Axes, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("fontfamily", "Outfit")
        kwargs.setdefault("fontweight", "bold")
        return original_set_title(self, *args, **kwargs)

    mpl_axes.Axes.set_title = _set_title


def apply_qpdk_style() -> str:
    """Apply the QPDK plot style, falling back to matplotlib's own defaults.

    The style is ``docs/qpdk.mplstyle`` in a checkout and the installed ``qpdk``
    style in a documentation environment; a downloaded notebook outside both
    keeps matplotlib's defaults instead of failing. The documentation fonts are
    used when they are installed, and matplotlib's bundled families otherwise.

    Returns:
        A short description of the style that was applied.
    """
    for source in (PATH.repo / "docs" / "qpdk.mplstyle", "qpdk"):
        try:
            plt.style.use(source)
        except OSError:
            continue
        applied = str(source)
        break
    else:
        applied = "matplotlib defaults"

    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [
        name
        for name in ("Inter", "Outfit", "DejaVu Sans", "Helvetica", "Arial")
        if name in installed
    ] + ["sans-serif"]
    if "Outfit" in installed:
        _outfit_titles()
    return applied


def prefer_svg_figures() -> None:
    """Save every figure as SVG as well as PNG, so stored outputs stay vector.

    The saved cell outputs are what the documentation renders, and both the HTML
    and the Typst PDF build embed the SVG ahead of the PNG. The PNG is kept as a
    fallback for a viewer that cannot render SVG, and text is written as paths so
    the figures carry their own glyphs instead of relying on installed fonts.
    Outside a notebook kernel there is no inline backend to configure, so a
    plain script run keeps matplotlib's PNG default.
    """
    try:
        # Ships with ipykernel, so it is present in a notebook kernel only.
        from matplotlib_inline.backend_inline import (  # ruff: ignore[import-outside-top-level]
            set_matplotlib_formats,
        )
    except ImportError:
        return
    plt.rcParams["svg.fonttype"] = "path"
    set_matplotlib_formats("svg", "png")


def result_file(name: str) -> Path | None:
    """Return the path of an exported solver result, if one is available.

    Args:
        name: File name to look for inside ``RESULTS_DIR``.

    Returns:
        The path, or ``None`` when ``RESULTS_DIR`` is unset or holds no such file.
    """
    if RESULTS_DIR is None:
        return None
    path = RESULTS_DIR / name
    return path if path.exists() else None


def explain_missing_results(name: str) -> None:
    """Print how to supply a result file that is not on disk.

    Args:
        name: File name that was looked for inside ``RESULTS_DIR``.
    """
    print(
        f"No {name} in RESULTS_DIR ({RESULTS_DIR}). The figures on the "
        "documentation page are saved outputs of a licensed solve, and the cells "
        "here replot only from files on disk. To supply them, run with "
        "RUN_COMSOL = True on a licensed machine, which exports into MODEL_DIR, "
        "or set RESULTS_DIR to a directory that already holds an exported "
        f"{name}."
    )


prefer_svg_figures()
STYLE_SOURCE = apply_qpdk_style()
print("Plot style: QPDK" if STYLE_SOURCE != "matplotlib defaults" else STYLE_SOURCE)

# %% [markdown]
# ## Build the transmon cell and an EM-only copy
#
# `prepare_comsol_layout` rejects geometry on layers it does not model, the junction layers among them. The
# cell is copied and `JJ_AREA` and `JJ_PATCH` removed, leaving `M1_DRAW` and the `M1_ETCH` mask.

# %%
component = double_pad_transmon_with_bbox(bbox_extension=200.0)

junction_port = component.ports["junction"]
JUNCTION_CENTER_UM = tuple(junction_port.center)

em_component = component.copy()
em_component.remove_layers(layers=[LAYER.JJ_AREA, LAYER.JJ_PATCH])

print(f"Component: {component.name}")
print(f"Bounding box (µm): {component.bbox()}")
print(f"Junction site: center={JUNCTION_CENTER_UM} µm")
print(f"Layers kept:   {sorted(em_component.layers)}")
print("JJ_AREA/JJ_PATCH removed; M1_DRAW and M1_ETCH remain.")

# %% [markdown]
# ## Extract the COMSOL layout
#
# The pads are not transmission-line ports, so no feed ports are requested. The etched moat is a *hole* in
# the ground-plane polygon; the pads sit inside it as separate polygons.

# %%
layout = prepare_comsol_layout(em_component, feed_ports=None, ground_margin=100.0)

print(f"Metal polygons: {len(layout.polygons)}")
print(f"Feed ports:     {layout.feed_ports}")
print(f"Prepared bbox (µm): {layout.bbox}")

# Three conductor faces: the two pads and the ground plane. Each point below
# sits well inside its metal region so the face selection resolves to exactly
# one face, which the study builder checks before adding the terminals.
LEFT_PAD_POINT = (-132.5, 0.0)
RIGHT_PAD_POINT = (132.5, 0.0)
GROUND_POINT = (-500.0, 0.0)

# The study names one face selection per conductor after these tags, so the mesh
# sizes and the terminal and ground arguments below can name a face by tag.
PAD_L_SELECTION = "pad_l"
PAD_R_SELECTION = "pad_r"
GROUND_SELECTION = "gnd"
CONDUCTORS = (
    (PAD_L_SELECTION, LEFT_PAD_POINT),
    (PAD_R_SELECTION, RIGHT_PAD_POINT),
    (GROUND_SELECTION, GROUND_POINT),
)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
for polygon in sorted(
    layout.polygons,
    key=lambda item: (
        -(
            (max(x for x, _ in item.outline) - min(x for x, _ in item.outline))
            * (max(y for _, y in item.outline) - min(y for _, y in item.outline))
        )
    ),
):
    ax.add_patch(
        MplPolygon(
            polygon.outline,
            closed=True,
            facecolor="0.78",
            edgecolor="0.35",
            linewidth=0.6,
            zorder=1,
        )
    )
    for hole in polygon.holes:
        ax.add_patch(
            MplPolygon(
                hole,
                closed=True,
                facecolor="white",
                edgecolor="0.35",
                linewidth=0.6,
                zorder=1,
            )
        )
for point, label, offset in (
    (LEFT_PAD_POINT, "left pad", (-35, 10)),
    (RIGHT_PAD_POINT, "right pad", (8, 10)),
    (GROUND_POINT, "ground", (8, 8)),
    (JUNCTION_CENTER_UM, "junction site", (8, -15)),
):
    ax.plot(*point, marker="x", color="crimson", markersize=7, zorder=4)
    ax.annotate(
        label,
        point,
        textcoords="offset points",
        xytext=offset,
        fontsize=9,
        color="crimson",
    )
ax.set_aspect("equal")
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("Pads, ground plane, and the conductor points")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build the COMSOL model and capacitance study
#
# Three calls configure the model: {py:meth}`~qpdk.simulation.comsol_model.COMSOL.create_sheet` makes the
# air and silicon blocks meeting at $z = 0$, imprints the layout metal on that interface as faces, and
# assigns materials;
# {py:meth}`~qpdk.simulation.comsol_model.COMSOL.add_capacitance_study` picks the pad and ground faces from
# the points above, drives the left pad, grounds the right pad and the ground plane, and adds the mesh and a
# stationary study; and {py:meth}`~qpdk.simulation.comsol_model.COMSOL.pin_absolute_mesh_sizes` reruns that
# mesh with absolute element sizes in micrometres, so the near-metal resolution no longer moves with the
# domain.
#
# The constants below are the base configuration the saved metrics and field map report: 8000 µm of margin,
# 1600 µm of silicon, 1600 µm of air, and the finest `0.625/1.25` near-metal entry.

# %%
RUN_COMSOL = False
RUN_DOMAIN_STUDY = False  # domain and near-metal mesh series below; needs RUN_COMSOL
MODEL_DIR = Path.home() / "comsol_models"
MODEL_PATH = MODEL_DIR / "comsol_qubit_capacitance.mph"
RESULTS_DIR: Path | None = MODEL_DIR if RUN_COMSOL else None
CORES = 4
VOLTAGE_V = 1.0

# The main solve's domain and element sizes, in µm. The domain series below reads
# these same values as its base, so the two sections describe one configuration.
AIR_HEIGHT_UM = 1600.0
SUBSTRATE_THICKNESS_UM = 1600.0
LATERAL_MARGIN_UM = 8000.0
# Element size COMSOL's physics-controlled build starts from. The pinned sizes
# below replace it, so it only sets the sizing the sequence is materialised with.
BASE_MESH_SIZE = 2
GLOBAL_HMAX_UM = 1000.0
GLOBAL_HMIN_UM = 2.0
HGRAD = 1.4
HCURVE = 0.5
HNARROW = 0.7
# One entry per near-metal resolution: (pad hmax, pad hmin, ground hmax,
# ground hmin) in µm, keyed by the pad and ground hmax. Every hmin is a tenth of
# its hmax, and the insertion order runs from the coarsest setting to the finest,
# which is the order the series is read in. The main solve uses MAIN_NEAR_METAL.
NEAR_METAL_SIZES = {
    "5/10": (5.0, 0.5, 10.0, 1.0),
    "2.5/5": (2.5, 0.25, 5.0, 0.5),
    "1.25/2.5": (1.25, 0.125, 2.5, 0.25),
    "0.8/1.6": (0.8, 0.08, 1.6, 0.16),
    "0.625/1.25": (0.625, 0.0625, 1.25, 0.125),
}
MAIN_NEAR_METAL = "0.625/1.25"

model = None
if RUN_COMSOL and not MPH_AVAILABLE:
    raise RuntimeError("RUN_COMSOL needs MPh and a licensed COMSOL installation")

if RUN_COMSOL and MPH_AVAILABLE:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    client = mph.start(cores=CORES)
    (
        pad_hmax_um,
        pad_hmin_um,
        ground_hmax_um,
        ground_hmin_um,
    ) = NEAR_METAL_SIZES[MAIN_NEAR_METAL]
    model = COMSOL.create_sheet(
        client,
        layout,
        name="QPDK Double-Pad Transmon",
        substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
        air_height_um=AIR_HEIGHT_UM,
        lateral_margin_um=LATERAL_MARGIN_UM,
    ).add_capacitance_study(
        conductors=CONDUCTORS,
        terminal=PAD_L_SELECTION,
        grounds=(PAD_R_SELECTION, GROUND_SELECTION),
        voltage_v=VOLTAGE_V,
        mesh_size=BASE_MESH_SIZE,
    )
    element_count = model.pin_absolute_mesh_sizes(
        global_hmax_um=GLOBAL_HMAX_UM,
        global_hmin_um=GLOBAL_HMIN_UM,
        face_sizes={
            PAD_L_SELECTION: (pad_hmax_um, pad_hmin_um),
            PAD_R_SELECTION: (pad_hmax_um, pad_hmin_um),
            GROUND_SELECTION: (ground_hmax_um, ground_hmin_um),
        },
        hgrad=HGRAD,
        hcurve=HCURVE,
        hnarrow=HNARROW,
    )
    model.java.study("std1").run()
    for problem in model.problems():
        print(f"The main solve reports: {problem}")
    model.save(MODEL_PATH)

    capacitance_f = float(model.evaluate("es.C11"))
    stored_energy_j = float(model.evaluate("es.intWe"))
    print(f"es.C11 = {capacitance_f:.6e} F")
    print(f"es.intWe = {stored_energy_j:.6e} J")
    print(f"2*intWe/V^2 = {2.0 * stored_energy_j / VOLTAGE_V**2:.6e} F")
    print(f"Mesh elements = {element_count}")
    print(f"Saved model to {MODEL_PATH}")

    metrics_path = MODEL_DIR / "comsol_qubit_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "capacitance_f": capacitance_f,
                "stored_energy_j": stored_energy_j,
                "voltage_v": VOLTAGE_V,
                "solver": "COMSOL Multiphysics",
                "study": "Electrostatics stationary",
                "mesh": "absolute element sizes, physics-controlled sizing replaced",
                "base_mesh_size": BASE_MESH_SIZE,
                "element_count": element_count,
                "global_hmax_um": GLOBAL_HMAX_UM,
                "global_hmin_um": GLOBAL_HMIN_UM,
                "hgrad": HGRAD,
                "hcurve": HCURVE,
                "hnarrow": HNARROW,
                "near_metal": MAIN_NEAR_METAL,
                "pad_hmax_um": pad_hmax_um,
                "pad_hmin_um": pad_hmin_um,
                "ground_hmax_um": ground_hmax_um,
                "ground_hmin_um": ground_hmin_um,
                "lateral_margin_um": LATERAL_MARGIN_UM,
                "substrate_thickness_um": SUBSTRATE_THICKNESS_UM,
                "air_height_um": AIR_HEIGHT_UM,
                "silicon_relative_permittivity": 11.7,
            },
            indent=2,
        )
        + "\n"
    )

# %% [markdown]
# ### Exporting the potential and field map
#
# The map below came from COMSOL's Data export on a cut plane at $z = 1$ µm carrying $V$ and `es.normE`; this
# block creates that `CutPlane` dataset and the export into `MODEL_DIR`.

# %%
if RUN_COMSOL and MPH_AVAILABLE and model is not None:
    plane = model.java.result().dataset().create("cutplane", "CutPlane")
    plane.set("planetype", "quick")
    plane.set("quickplane", "xy")
    plane.set("quickz", "1[um]")
    plane.set("data", "dset1")

    field_export = model.java.result().export().create("field", "Data")
    field_export.set("data", "cutplane")
    field_export.set("expr", ["V", "es.normE"])
    field_export.set("filename", str(MODEL_DIR / "comsol_qubit_field.txt"))
    field_export.run()
    print(f"Exported V and es.normE to {MODEL_DIR / 'comsol_qubit_field.txt'}")

# %% [markdown]
# ## Saved capacitance result
#
# This cell replots the exported `C11` and the stored energy checked against it through $2 W_e / V^2$, beside
# the settings the main solve used, since $C_{11}$ moves with both the mesh and the box.

# %%
METRICS_JSON = "comsol_qubit_metrics.json"
metrics_file = result_file(METRICS_JSON)

capacitance_f: float | None = None
stored_energy_j: float | None = None
voltage_v: float | None = None

if metrics_file is None:
    explain_missing_results(METRICS_JSON)
else:
    metrics = json.loads(metrics_file.read_text())
    capacitance_f = float(metrics["capacitance_f"])
    stored_energy_j = float(metrics["stored_energy_j"])
    voltage_v = float(metrics["voltage_v"])

    capacitance_from_energy_f = 2.0 * stored_energy_j / voltage_v**2
    print(
        f"C11                    = {capacitance_f * 1e15:.4f} fF at V = {voltage_v:g} V"
    )
    print(
        f"2 * We / V^2           = {capacitance_from_energy_f * 1e15:.4f} fF "
        f"(relative difference "
        f"{abs(capacitance_from_energy_f - capacitance_f) / capacitance_f:.2e})"
    )

    print(
        f"mesh                   = {int(metrics['element_count']):,} elements, "
        f"{metrics['near_metal']} near-metal sizes"
    )
    print(
        f"box                    = margin {float(metrics['lateral_margin_um']):g} µm, "
        f"silicon {float(metrics['substrate_thickness_um']):g} µm, "
        f"air {float(metrics['air_height_um']):g} µm"
    )

# %% [markdown]
# ### An LC frequency estimate
#
# A chosen $L_J$ turns $C_{11}$ into a rough circuit frequency. **$L_J$ is not in the COMSOL solve** and
# $C_{11}$ is not the two-pad differential capacitance, so this is a scale, not an eigenfrequency.

# %%
LJ_H = 10e-9  # chosen, not solved for

if capacitance_f is None:
    print(f"Skipping the estimate: no {METRICS_JSON} available (see the note above).")
else:
    f_lc_ghz = 1.0 / (2.0 * np.pi * np.sqrt(LJ_H * capacitance_f)) / 1e9
    print(f"Chosen L_J                    = {LJ_H * 1e9:.1f} nH")
    print(f"illustrative f_LC using C11     = {f_lc_ghz:.3f} GHz")

# %% [markdown]
# ## Saved potential and field map
#
# The exported field is $V$ and the field norm `es.normE` on the $z = 1$ µm plane just above the metal sheet.
#
# The export spans ±8.5 mm, where the pads are a dot, so the map is a **close-up**: `FIELD_LIMIT_X_UM` and
# `FIELD_LIMIT_Y_UM` frame both pads, and the remaining nodes are resampled onto a display grid (**display
# only**: it reads the solved field without re-solving it).

# %%
FIELD_TXT = "comsol_qubit_field.txt"
FIELD_LIMIT_X_UM = 300.0
FIELD_LIMIT_Y_UM = 250.0
FIELD_GRID_X = 300
FIELD_GRID_Y = 250
FIELD_LEVELS = 20

field_file = result_file(FIELD_TXT)

if field_file is None:
    explain_missing_results(FIELD_TXT)
else:
    field = np.loadtxt(field_file, comments="%")
    field_x, field_y = field[:, 0], field[:, 1]
    in_view = (np.abs(field_x) <= FIELD_LIMIT_X_UM) & (
        np.abs(field_y) <= FIELD_LIMIT_Y_UM
    )
    if in_view.sum() < 3:
        raise ValueError(
            f"No field nodes lie within ±{FIELD_LIMIT_X_UM:g} by "
            f"±{FIELD_LIMIT_Y_UM:g} µm, so there is nothing to draw a close-up from"
        )
    triangulation = mtri.Triangulation(field_x[in_view], field_y[in_view])
    grid_x = np.linspace(-FIELD_LIMIT_X_UM, FIELD_LIMIT_X_UM, FIELD_GRID_X)
    grid_y = np.linspace(-FIELD_LIMIT_Y_UM, FIELD_LIMIT_Y_UM, FIELD_GRID_Y)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    field_v = mtri.LinearTriInterpolator(triangulation, field[in_view, 3])(
        grid_xx, grid_yy
    ).filled(np.nan)
    field_e = mtri.LinearTriInterpolator(triangulation, field[in_view, 4])(
        grid_xx, grid_yy
    ).filled(np.nan)
    field_e_max = float(np.nanmax(field_e))

    voltage_label = f"at V = {voltage_v:g} V, " if voltage_v is not None else ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    potential = axes[0].contourf(
        grid_x, grid_y, field_v, levels=FIELD_LEVELS, cmap="viridis"
    )
    axes[0].set_title(f"Electric potential {voltage_label}z = 1 µm")
    fig.colorbar(potential, ax=axes[0], label=r"$V$ (V)")

    norm_e = axes[1].contourf(
        grid_x,
        grid_y,
        field_e,
        levels=np.geomspace(1.0, field_e_max, FIELD_LEVELS),
        norm=LogNorm(vmin=1.0, vmax=field_e_max),
        cmap="inferno",
        extend="min",
    )
    axes[1].set_title(f"Electric field norm {voltage_label}z = 1 µm")
    fig.colorbar(norm_e, ax=axes[1], label=r"$|\mathbf{E}|$ (V/m)")

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# The potential holds across the driven pad and falls through the gap to the grounded pad; the field
# concentrates in that gap and at the pad edges, where the pad capacitance mainly lives.

# %% [markdown]
# ## Domain and near-metal mesh study
#
# A physics-controlled mesh scales with the domain, so a domain sweep moves the near-metal resolution with it;
# every case here pins absolute element sizes instead, as the main solve does. Two effects are separated:
#
# - **The finite box.** Air above and silicon below span the layout box grown by `lateral_margin_um`; their
#   outer walls take the Electrostatics default for an exterior boundary, zero charge
#   ($\mathbf{n} \cdot \mathbf{D} = 0$), turning back field that would spread into a larger chip.
# - **The near-metal element size**, which sets how well the field in the pad gap and at the pad edges is
#   resolved.
#
# {py:meth}`~qpdk.simulation.comsol_model.COMSOL.pin_absolute_mesh_sizes` writes absolute sizes into the mesh
# sequence, one per conductor face. `NEAR_METAL_SIZES` runs five settings from `5/10` to `0.625/1.25`; none is a
# converged mesh, and the series measures the step from each setting to the next.
#
# `DOMAIN_CASES` lists the cases, each a fresh model solved from scratch. The curves compare **one parameter
# at a time**, every other length fixed:
#
# - **Lateral margin**, at air 200 µm and silicon 1600 µm, one line per near-metal setting, so the mesh
#   sizes are told apart inside one box.
# - **Air height**, at margin 8000 µm, silicon 1600 µm, and the `2.5/5` near-metal setting.
# - **Mesh**, the near-metal ladder against element count at the base box: margin 8000 µm, air 1600 µm,
#   silicon 1600 µm.
#
# That base box is the main solve's own configuration; run the series with `RUN_DOMAIN_STUDY = True` and
# `RUN_COMSOL = True`.

# %% tags=["hide-input"]
DOMAIN_JSON = "comsol_qubit_domain_convergence.json"


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file, then replace in place.

    The series writes on every case, so an interrupt partway through still
    leaves the cases already solved on disk. The temporary file is a sibling so
    the replace stays a same-filesystem rename, which is what makes it atomic.

    Args:
        path: The JSON file to write.
        payload: The object to serialise.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class DomainCase(NamedTuple):
    """One case of the domain and near-metal series.

    Attributes:
        lateral_margin_um: Margin around the layout box, in µm.
        substrate_thickness_um: Silicon thickness below the interface, in µm.
        air_height_um: Air height above the interface, in µm.
        near_metal: Key of a near-metal element size setting in
            ``NEAR_METAL_SIZES``.
    """

    lateral_margin_um: float
    substrate_thickness_um: float
    air_height_um: float
    near_metal: str


def domain_case_label(case: DomainCase) -> str:
    """Name every case parameter on a single line.

    Args:
        case: One case of the series.

    Returns:
        A label holding the margin, the substrate thickness, the air height, and
        the near-metal setting, so cases sharing a margin and a setting but not
        an air height stay apart.
    """
    return (
        f"margin {case.lateral_margin_um:g} µm, "
        f"substrate {case.substrate_thickness_um:g} µm, "
        f"air {case.air_height_um:g} µm, {case.near_metal} near-metal sizes"
    )


# (lateral margin µm, substrate µm, air µm, near-metal setting). The margin is
# swept at a fixed near-metal setting, the near-metal setting at margins that have
# already been solved and at air the base case does not use, and the substrate
# thickness and air height away from the base case, so every lever can be read
# against the others.
DOMAIN_CASES = (
    DomainCase(2400.0, 1600.0, 200.0, "5/10"),
    DomainCase(8000.0, 1600.0, 200.0, "5/10"),
    DomainCase(16000.0, 1600.0, 200.0, "5/10"),
    DomainCase(8000.0, 1600.0, 200.0, "2.5/5"),
    DomainCase(16000.0, 1600.0, 200.0, "2.5/5"),
    DomainCase(8000.0, 1600.0, 200.0, "1.25/2.5"),
    DomainCase(16000.0, 1600.0, 200.0, "1.25/2.5"),
    DomainCase(8000.0, 1600.0, 400.0, "2.5/5"),
    DomainCase(8000.0, 1600.0, 800.0, "2.5/5"),
    DomainCase(8000.0, 3200.0, 200.0, "2.5/5"),
    DomainCase(8000.0, 1600.0, 1600.0, "2.5/5"),
    DomainCase(8000.0, 1600.0, 3200.0, "2.5/5"),
    DomainCase(16000.0, 1600.0, 1600.0, "2.5/5"),
    DomainCase(8000.0, 1600.0, 1600.0, "1.25/2.5"),
    DomainCase(8000.0, 1600.0, 1600.0, "0.8/1.6"),
    DomainCase(8000.0, 1600.0, 1600.0, "0.625/1.25"),
)


def run_qubit_domain_case(client: Any, case: DomainCase) -> dict[str, Any]:
    """Build, mesh, and solve one domain case, and return its row.

    The model is built from scratch on the substrate thickness and air height the
    case sets, with the base mesh size the main solve uses, and the mesh is then
    pinned to absolute element sizes, so only the case parameters differ between
    cases. The model is removed from the client even when the build, the mesh, or
    the solve raises.

    Args:
        client: The MPh client the licensed branch already started.
        case: Lateral margin, substrate thickness, air height, and the near-metal
            setting name.

    Returns:
        A row holding the case parameters, the element sizes that were pinned,
        the mesh element count, ``es.C11`` in farads, and ``es.intWe`` in joules
        at ``VOLTAGE_V``.
    """
    pad_hmax_um, pad_hmin_um, ground_hmax_um, ground_hmin_um = NEAR_METAL_SIZES[
        case.near_metal
    ]
    temp_model = COMSOL.create_sheet(
        client,
        layout,
        name=f"QPDK transmon domain {domain_case_label(case)}",
        substrate_thickness_um=case.substrate_thickness_um,
        air_height_um=case.air_height_um,
        lateral_margin_um=case.lateral_margin_um,
    ).add_capacitance_study(
        conductors=CONDUCTORS,
        terminal=PAD_L_SELECTION,
        grounds=(PAD_R_SELECTION, GROUND_SELECTION),
        voltage_v=VOLTAGE_V,
        mesh_size=BASE_MESH_SIZE,
    )
    try:
        element_count = temp_model.pin_absolute_mesh_sizes(
            global_hmax_um=GLOBAL_HMAX_UM,
            global_hmin_um=GLOBAL_HMIN_UM,
            face_sizes={
                PAD_L_SELECTION: (pad_hmax_um, pad_hmin_um),
                PAD_R_SELECTION: (pad_hmax_um, pad_hmin_um),
                GROUND_SELECTION: (ground_hmax_um, ground_hmin_um),
            },
            hgrad=HGRAD,
            hcurve=HCURVE,
            hnarrow=HNARROW,
        )
        temp_model.java.study("std1").run()
        for problem in temp_model.problems():
            print(f"  {domain_case_label(case)} reports: {problem}")
        return {
            "lateral_margin_um": case.lateral_margin_um,
            "substrate_thickness_um": case.substrate_thickness_um,
            "air_height_um": case.air_height_um,
            "near_metal": case.near_metal,
            "pad_hmax_um": pad_hmax_um,
            "pad_hmin_um": pad_hmin_um,
            "ground_hmax_um": ground_hmax_um,
            "ground_hmin_um": ground_hmin_um,
            "global_hmax_um": GLOBAL_HMAX_UM,
            "global_hmin_um": GLOBAL_HMIN_UM,
            "hgrad": HGRAD,
            "hcurve": HCURVE,
            "hnarrow": HNARROW,
            "base_mesh_size": BASE_MESH_SIZE,
            "voltage_v": VOLTAGE_V,
            "element_count": element_count,
            "c11_f": float(np.atleast_1d(temp_model.evaluate("es.C11"))[0]),
            "int_we_j": float(np.atleast_1d(temp_model.evaluate("es.intWe"))[0]),
        }
    finally:
        client.remove(temp_model)


if RUN_COMSOL and MPH_AVAILABLE and RUN_DOMAIN_STUDY:
    domain_rows: list[dict[str, Any]] = []
    domain_failures: list[dict[str, Any]] = []
    domain_path = MODEL_DIR / DOMAIN_JSON

    def save_domain() -> None:
        """Write the cases solved so far, so a late failure keeps them."""
        domain_payload: dict[str, Any] = {"rows": domain_rows}
        if domain_failures:
            domain_payload["failures"] = domain_failures
        write_json_atomically(domain_path, domain_payload)

    save_domain()
    for case in DOMAIN_CASES:
        print(f"{domain_case_label(case)}: building, meshing, and solving")
        try:
            row = run_qubit_domain_case(client, case)
        except Exception as error:
            domain_failures.append({
                "label": domain_case_label(case),
                "error": type(error).__name__,
            })
            print(f"{domain_case_label(case)} FAILED: {type(error).__name__}: {error}")
            save_domain()
            continue
        domain_rows.append(row)
        print(
            f"{domain_case_label(case)}: {row['element_count']} elements, "
            f"C11 = {row['c11_f'] * 1e15:.4f} fF, "
            f"intWe = {row['int_we_j']:.6e} J"
        )
        save_domain()

    print(f"Wrote {len(domain_rows)} of {len(DOMAIN_CASES)} rows to {domain_path}")
    if domain_failures:
        failed = ", ".join(item["label"] for item in domain_failures)
        print(f"No row was written for: {failed}")
elif RUN_DOMAIN_STUDY and not RUN_COMSOL:
    print(
        "RUN_DOMAIN_STUDY needs RUN_COMSOL = True: the series reuses the COMSOL "
        "client that the licensed branch starts."
    )

# %% [markdown]
# ### Reading the domain and mesh series
#
# Each line changes one setting at a time. The margin sweep fixes air at 200 µm and silicon at 1600 µm;
# the air sweep fixes margin at 8000 µm, silicon at 1600 µm, and the near-metal mesh at 2.5/5 µm.
# The separate mesh sweep uses the main simulation box. Flat curves suggest lower sensitivity to the
# swept setting; the energy check $2 W_e / V^2$ tests consistency, not convergence.

# %% tags=["hide-input"]
MARGIN_PANEL_AIR_UM = 200.0
AIR_PANEL_NEAR_METAL = "2.5/5"


def domain_near_metal_name(row: dict[str, Any]) -> str:
    """Return the near-metal setting a row was solved at.

    The two pinned conductor sizes identify the setting, so a row written by an
    earlier run is still read as the setting it was solved with, whatever name
    that run recorded.

    Args:
        row: One row read from the domain JSON.

    Returns:
        The setting name, or the pad and ground sizes spelled out when they match
        no entry of ``NEAR_METAL_SIZES``.
    """
    for candidate, (pad_hmax_um, _, ground_hmax_um, _) in NEAR_METAL_SIZES.items():
        if (
            row.get("pad_hmax_um") == pad_hmax_um
            and row.get("ground_hmax_um") == ground_hmax_um
        ):
            return candidate
    if (name := row.get("near_metal")) is not None:
        return str(name)
    return f"pad {row.get('pad_hmax_um')}/gnd {row.get('ground_hmax_um')} µm near-metal"


def ordered_near_metal_names(names: set[str]) -> list[str]:
    """Sort near-metal setting names into refinement order.

    Args:
        names: Setting names present in the rows being read.

    Returns:
        The names ``NEAR_METAL_SIZES`` holds, in its order, then any name it does
        not know.
    """
    known = [name for name in NEAR_METAL_SIZES if name in names]
    return known + sorted(names.difference(NEAR_METAL_SIZES))


def compact_element_count(count: int) -> str:
    """Format an element count for an axis tick.

    Args:
        count: Number of mesh elements.

    Returns:
        The count in millions, or in thousands below a million.
    """
    return f"{count / 1e6:.1f}M" if count >= 1_000_000 else f"{count / 1e3:.0f}k"


def rows_pinning(
    rows: list[dict[str, Any]],
    *,
    lateral_margin_um: float | None = None,
    substrate_thickness_um: float | None = None,
    air_height_um: float | None = None,
    near_metal: str | None = None,
) -> list[dict[str, Any]]:
    """Select the rows that hold the given case parameters.

    A curve may only join rows that differ in the swept length alone, so every
    series reads its points from the rows this picks out.

    Args:
        rows: Rows read from the domain JSON.
        lateral_margin_um: Margin to hold, or ``None`` to leave it free.
        substrate_thickness_um: Substrate thickness to hold, or ``None``.
        air_height_um: Air height to hold, or ``None``.
        near_metal: Near-metal setting to hold, or ``None``.

    Returns:
        The matching rows, in the order they were read.
    """

    def holds(value: Any, target: float | None) -> bool:
        """Return whether a row's value sits on the parameter it is checked against.

        Args:
            value: Value read from a row.
            target: Value to match, or ``None`` to accept any value.

        Returns:
            Whether the value may be joined to the rows being collected.
        """
        return target is None or math.isclose(float(value), target)

    return [
        row
        for row in rows
        if holds(row["lateral_margin_um"], lateral_margin_um)
        and holds(row["substrate_thickness_um"], substrate_thickness_um)
        and holds(row["air_height_um"], air_height_um)
        and (near_metal is None or domain_near_metal_name(row) == near_metal)
    ]


def series_points(
    rows: list[dict[str, Any]], swept_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Turn rows into a ``(swept value, C11)`` series in fF.

    Args:
        rows: Rows holding the swept key and ``c11_f``.
        swept_key: Row key of the value swept along the x axis.

    Returns:
        The swept values and the capacitances, ordered by swept value.
    """
    ordered = sorted(rows, key=itemgetter(swept_key))
    swept = np.array([float(row[swept_key]) for row in ordered], dtype=float)
    capacitance_ff = np.array([float(row["c11_f"]) for row in ordered]) * 1e15
    return swept, capacitance_ff


def log_axis_on_values(
    ax: mpl_axes.Axes,
    values: np.ndarray,
    label: str,
    tick_labels: list[str] | None = None,
) -> None:
    """Put a log x axis on the solved values themselves.

    The solved settings sit between round decades, so matplotlib's own log ticks
    would land off the points; ticking the solved values keeps every tick on a
    curve it belongs to.

    Args:
        ax: Axes to label.
        values: Every value plotted along x.
        label: Axis label.
        tick_labels: One label per distinct value, or the values in µm.
    """
    ticks = np.unique(values)
    ax.set_xscale("log")
    ax.set_xticks(ticks, labels=tick_labels or [f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.15, ticks[-1] * 1.15)
    ax.set_xlabel(label)


domain_file = result_file(DOMAIN_JSON)

if domain_file is None:
    explain_missing_results(DOMAIN_JSON)
else:
    domain_payload = json.loads(domain_file.read_text())
    domain_rows = (
        domain_payload
        if isinstance(domain_payload, list)
        else domain_payload.get("rows") or []
    )
    if isinstance(domain_payload, dict):
        for failure in domain_payload.get("failures") or []:
            print(f"{failure['label']} failed: {failure['error']}")

    if not domain_rows:
        explain_missing_results(DOMAIN_JSON)
        print("The file on disk carries no rows.")
    else:
        voltages_v = {float(row["voltage_v"]) for row in domain_rows}
        if len(voltages_v) != 1:
            raise ValueError("Domain rows have different solved voltages")
        solved_voltage_v = voltages_v.pop()

        margin_rows = rows_pinning(
            domain_rows,
            substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
            air_height_um=MARGIN_PANEL_AIR_UM,
        )
        margin_series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for setting in ordered_near_metal_names({
            domain_near_metal_name(row) for row in margin_rows
        }):
            setting_rows = [
                row for row in margin_rows if domain_near_metal_name(row) == setting
            ]
            if len(setting_rows) < 2:
                continue
            margin_series.append((
                setting,
                *series_points(setting_rows, "lateral_margin_um"),
            ))
        if not margin_series:
            print(
                f"No two solved margins share air {MARGIN_PANEL_AIR_UM:g} µm and "
                f"silicon {SUBSTRATE_THICKNESS_UM:g} µm, so the margin panel is "
                "skipped."
            )

        air_rows = rows_pinning(
            domain_rows,
            lateral_margin_um=LATERAL_MARGIN_UM,
            substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
            near_metal=AIR_PANEL_NEAR_METAL,
        )
        air_heights, air_capacitance_ff = series_points(air_rows, "air_height_um")
        has_air_panel = len(air_heights) >= 2
        if not has_air_panel:
            print(
                f"Fewer than two air heights are on disk at margin "
                f"{LATERAL_MARGIN_UM:g} µm, silicon {SUBSTRATE_THICKNESS_UM:g} µm, "
                f"and {AIR_PANEL_NEAR_METAL} near-metal sizes, so the air panel is "
                "skipped."
            )

        panels: list[tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]] = []
        if margin_series:
            panels.append(("lateral margin (µm)", margin_series))
        if has_air_panel:
            panels.append((
                "air height (µm)",
                [
                    (
                        f"{AIR_PANEL_NEAR_METAL} near-metal sizes",
                        air_heights,
                        air_capacitance_ff,
                    )
                ],
            ))

        if panels:
            fig, axes = plt.subplots(
                1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False
            )
            for index, (ax, (xlabel, series)) in enumerate(
                zip(axes[0], panels, strict=True)
            ):
                for series_label, values, capacitance_ff in series:
                    ax.plot(
                        values,
                        capacitance_ff,
                        marker="o",
                        markersize=4,
                        label=series_label,
                    )
                log_axis_on_values(
                    ax,
                    np.concatenate([values for _, values, _ in series]),
                    f"({chr(ord('a') + index)}) {xlabel}",
                )
                ax.set_ylabel("$C_{11}$ (fF)")
                ax.grid(True, which="both", alpha=0.3)
                if len(series) > 1:
                    ax.legend(fontsize=8)
            fig.suptitle(f"Pad capacitance vs domain size ({solved_voltage_v:g} V)")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            plt.show()

        mesh_rows = rows_pinning(
            domain_rows,
            lateral_margin_um=LATERAL_MARGIN_UM,
            substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
            air_height_um=AIR_HEIGHT_UM,
        )
        mesh_ordered = sorted(mesh_rows, key=itemgetter("element_count"))
        if len(mesh_ordered) >= 2:
            element_counts = np.array([
                float(row["element_count"]) for row in mesh_ordered
            ])
            mesh_capacitance_ff = (
                np.array([float(row["c11_f"]) for row in mesh_ordered]) * 1e15
            )
            ladder = ordered_near_metal_names({
                domain_near_metal_name(row) for row in mesh_ordered
            })
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(element_counts, mesh_capacitance_ff, marker="s", markersize=4)
            log_axis_on_values(
                ax,
                element_counts,
                "mesh elements",
                [
                    compact_element_count(int(count))
                    for count in np.unique(element_counts)
                ],
            )
            ax.set_ylabel("$C_{11}$ (fF)")
            ax.set_title(
                f"Mesh convergence ({ladder[0]} to {ladder[-1]} µm near metal)"
            )
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            plt.show()
        else:
            print(
                f"Fewer than two near-metal settings are on disk at margin "
                f"{LATERAL_MARGIN_UM:g} µm, air {AIR_HEIGHT_UM:g} µm, silicon "
                f"{SUBSTRATE_THICKNESS_UM:g} µm, so the mesh panel is skipped."
            )

        if margin_series:
            print(
                f"\nLateral margin at air {MARGIN_PANEL_AIR_UM:g} µm, silicon "
                f"{SUBSTRATE_THICKNESS_UM:g} µm, C11 in fF"
            )
            for setting, values, capacitance_ff in margin_series:
                print(
                    f"  {setting} near-metal: {len(values)} margins "
                    f"{values[0]:g}-{values[-1]:g} µm, spread "
                    f"{float(np.ptp(capacitance_ff)):.4f}"
                )
        if has_air_panel:
            print(
                f"Air height at margin {LATERAL_MARGIN_UM:g} µm, silicon "
                f"{SUBSTRATE_THICKNESS_UM:g} µm, {AIR_PANEL_NEAR_METAL} near-metal, "
                "C11 in fF"
            )
            print(
                f"  {air_heights[0]:g}-{air_heights[-1]:g} µm: "
                f"{air_capacitance_ff[0]:.4f} to {air_capacitance_ff[-1]:.4f}, spread "
                f"{float(np.ptp(air_capacitance_ff)):.4f}"
            )
        if len(mesh_ordered) >= 2:
            steps_ff = [
                abs(float(later["c11_f"]) - float(earlier["c11_f"])) * 1e15
                for earlier, later in pairwise(mesh_ordered)
            ]
            print(
                f"Near-metal ladder at margin {LATERAL_MARGIN_UM:g} µm, air "
                f"{AIR_HEIGHT_UM:g} µm, silicon {SUBSTRATE_THICKNESS_UM:g} µm, C11 "
                "in fF"
            )
            print(
                f"  {ladder[0]} to {ladder[-1]}: "
                f"{mesh_capacitance_ff[0]:.4f} to {mesh_capacitance_ff[-1]:.4f}, "
                f"largest neighbouring step {max(steps_ff):.4f}"
            )

        print(
            "\nA shift read off a mesh that is still moving is not a converged "
            "shift, in the box or in the metal. C11 here is a one-terminal "
            "capacitance, not the two-pad charging capacitance."
        )

# %% [markdown]
# Read the curves with the printed spreads: a flattening margin or air curve suggests reduced box
# sensitivity, while smaller mesh steps suggest reduced sensitivity to element size.
#
# These observed changes are not error bounds. They do not measure the distance to a mesh-independent answer.

# %% [markdown]
# ## Limitations
#
# The solve is electrostatic and the EM-only copy omits the SQUID loop and its leads, so $C_{11}$ is a
# one-terminal capacitance to the grounded chip rather than the two-pad charging capacitance, and the `f_LC`
# estimate is not a transmon eigenfrequency or $f_{01}$. The observed shifts are not error bounds,
# and the finite zero-charge outer walls act at any mesh, so nothing here is a converged or validated device
# number. The two-pad capacitance matrix, feeding the QPDK Hamiltonian workflow
# ({doc}`/notebooks/scqubits_parameter_calculation`), is the next step.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
