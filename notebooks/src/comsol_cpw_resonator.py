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
# # COMSOL Full-Wave Simulation of a Coupled Quarter-Wave Resonator
#
# ::::{admonition} Required extras
# :class: tip
#
# Needs the `comsol` extra (`uv sync --extra comsol`), which installs `MPh` only. COMSOL, its license,
# the RF Module, and the Design Module CAD kernel that `ProjectToFaces` needs are separate. See the
# {ref}`extras reference <notebook-extras>`.
# ::::
#
# This notebook simulates a QPDK coupled quarter-wave resonator in two stages: a port-free
# eigenfrequency study, then a ported study whose driven $S_{21}$ notch is checked against direct
# solves.
#
# ## What is being modelled
#
# A QPDK {py:func}`~qpdk.cells.quarter_wave_resonator_coupled`: a meandering coplanar-waveguide (CPW)
# resonator beside a straight feedline, the standard hanger geometry for reading out superconducting
# qubits {cite:p}`gopplCoplanarWaveguideResonators2008a`, whose resonance is one of the degrees of
# freedom circuit QED reads a qubit through {cite:p}`blaisCircuitQuantumElectrodynamics2021`. The end
# nearest the feedline is **open** and the far end **shorted**, so the line resonates at an odd multiple
# of $\lambda/4$ {cite:p}`m.pozarMicrowaveEngineering2012`, and the coupling capacitor loads the
# feedline into a **notch** in $|S_{21}|$ whose centre and width give the frequency and loaded quality
# factor {cite:p}`gopplCoplanarWaveguideResonators2008a`.
#
# ## The two-stage model
#
# One sheet model for both stages: metal as faces at the air/silicon interface under **PEC**, a **mesh
# pinned to absolute sizes**, and an **`emw.normE` export** on a cut plane above the metal that scores
# each mode by its field on the meander. Stage 1 adds a plain **eigenfrequency search**; stage 2 keeps
# the ports with their **boundary mode analysis** steps and drives an **adaptive frequency sweep**. Both
# feeds are extended with straight CPW before extraction.
#
# ::::{only} html
# ```{mermaid}
# flowchart TB
#     A["Ported layout"]
#     B["Sheet model: metal faces at z = 0"]
#     C["Stage 1: PEC, no ports"]
#     D["Eigen search,<br>normE scored per mode"]
#     E["Port-free result: one mesh"]
#     F["Stage 2: BMA, both ports kept"]
#     G["Ported series: shifts grow then reverse"]
#     H["Driven window: AWE plus direct solves"]
#     I["Notch verified only at solved points"]
#     A --> B --> C --> D --> E
#     C --> F --> G
#     F --> H --> I
# ```
# ::::
#
# ::::{only} typst or typstpdf
# Ported layout, sheet model, driven sweep checked against direct solves, and a
# separate mesh study.
# ::::
#
# ::::{admonition} Reading the numbers on this page
# :class: warning
#
# Saved outputs of licensed solves, not a validated device prediction, and nothing here is shown to be
# mesh independent: the ported shifts grow then reverse, and the port-free series, both field maps, and
# the driven window each come from one mesh. PEC metal carries no conductor loss or kinetic inductance.
# The port-free and ported sets differ in loading and search and are never merged, and only the four
# directly solved frequencies verify the notch; the rest of the AWE curve is interpolation.
# ::::
#
# **References:**
# - [COMSOL "Coplanar Waveguide Resonator" model](https://www.comsol.com/model/download/953251/models.rf.cpw_resonator.pdf)
# - [COMSOL RF Module User's Guide](https://doc.comsol.com/6.3/doc/com.comsol.help.rf/RFModuleUsersGuide.pdf)
# - [MPh tutorial](https://mph.readthedocs.io/en/stable/tutorial.html)
# - MPh repository: https://github.com/MPh-py/MPh

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
import hashlib
import json
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes as mpl_axes, font_manager
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon

from qpdk import PDK
from qpdk.cells.resonator import quarter_wave_resonator_coupled
from qpdk.config import PATH
from qpdk.simulation import prepare_comsol_layout
from qpdk.tech import coplanar_waveguide

try:
    import mph
    from jpype import JInt

    # The COMSOL class imports MPh eagerly, so it comes in with the same guard:
    # without the extra the licensed branches stay off and the rest of the
    # notebook still reads exported results back from disk.
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


def resolve_record_path(base: Path, name: str) -> Path:
    """Resolve a file name stored in a result record.

    Records store the curve by bare file name so a saved notebook never embeds
    an absolute path from the machine that solved. A bare name resolves under
    ``RESULTS_DIR`` when that directory holds the file, otherwise next to the
    record itself. An absolute path in an older record is still honoured.

    Args:
        base: Directory the record itself lives in.
        name: The stored path or file name.

    Returns:
        The path to read, which may not exist yet.
    """
    stored = Path(name)
    if stored.is_absolute():
        return stored
    under_results = RESULTS_DIR / stored if RESULTS_DIR is not None else None
    if under_results is not None and under_results.exists():
        return under_results
    return base / stored


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


# Frequency units COMSOL may annotate an export header with, to GHz.
FREQUENCY_UNITS_GHZ = {"GHz": 1.0, "MHz": 1.0e-3, "kHz": 1.0e-6, "Hz": 1.0e-9}
# ``@ freq=7.5`` as written by some exports, or a bare ``@ 7.2921 GHz``.
FREQUENCY_ANNOTATION = re.compile(
    r"@\s*freq\s*=\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
FREQUENCY_HEADER = re.compile(
    r"@\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(GHz|MHz|kHz|Hz)\b"
)


def exported_frequency_ghz(file: Path) -> float | None:
    """Read the frequency annotation COMSOL writes into a data export header.

    COMSOL writes either ``@ freq=<value>`` or a bare ``@ <value> <unit>`` line,
    and the live field exports use the second form, so both are parsed and the
    unit is applied explicitly rather than assumed to be GHz.

    Args:
        file: Exported text file to scan.

    Returns:
        The annotated frequency in GHz, or ``None`` when the export carries no
        frequency.
    """
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if (match := FREQUENCY_ANNOTATION.search(line)) is not None:
                return float(match.group(1))
            if (match := FREQUENCY_HEADER.search(line)) is not None:
                return float(match.group(1)) * FREQUENCY_UNITS_GHZ[match.group(2)]
    return None


# COMSOL writes a ported eigenfield export's header with a complex frequency,
# e.g. ``@ 7.3266+5.3458E-4i GHz``. FREQUENCY_HEADER only parses a real value, so
# the complex form gets its own pattern.
PORTED_FIELD_FREQUENCY = re.compile(
    r"@\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    r"\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*i"
    r"\s*(GHz|MHz|kHz|Hz)\b"
)
# The stable file the licensed stage-2 branch writes the selected ported mode's
# field to, and how far the export's annotated real frequency may sit from the
# record's selected mode before the field is refused as belonging to something
# else.
PORTED_FIELD_TXT = "comsol_cpw_ported_field.txt"
PORTED_FIELD_FREQUENCY_RTOL = 1.0e-4


def complex_frequency_ghz(file: Path) -> tuple[float, float] | None:
    """Read a complex frequency annotation from a COMSOL export header.

    Args:
        file: Exported text file to scan.

    Returns:
        ``(real, imag)`` in GHz, or ``None`` when the export carries no complex
        frequency annotation.
    """
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if (match := PORTED_FIELD_FREQUENCY.search(line)) is not None:
                scale = FREQUENCY_UNITS_GHZ[match.group(3)]
                return float(match.group(1)) * scale, float(match.group(2)) * scale
    return None


def field_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a field export's bytes.

    The ported stable record and the stable field are replaced one after the
    other, so an interrupt between the two can leave a record pointing at a
    field it was not solved with. The digest recorded beside the mode is what
    lets the reader catch that instead of plotting the wrong mode.

    Args:
        path: Exported field file to digest.

    Returns:
        The digest as lowercase hex.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


prefer_svg_figures()
STYLE_SOURCE = apply_qpdk_style()
print("Plot style: QPDK" if STYLE_SOURCE != "matplotlib defaults" else STYLE_SOURCE)

# %% [markdown]
# ## Build the ported layout
#
# The resonator is built with an explicit CPW cross-section, since the centre width and gap are reused
# when describing the ports. The ground extends well past the resonator: a larger box weakens the outer
# PEC walls' influence without removing it.

# %%
CPW_WIDTH_UM = 10.0
CPW_GAP_UM = 6.0
LEFT_EXTENSION_UM = 1320.0
RIGHT_EXTENSION_UM = 2000.0
GROUND_MARGIN_UM = 1200.0

cross_section = coplanar_waveguide(width=CPW_WIDTH_UM, gap=CPW_GAP_UM)

resonator = quarter_wave_resonator_coupled(
    length=4000.0,
    meanders=4,
    cross_section=cross_section,
    cross_section_non_resonator=cross_section,
    coupling_straight_length=200.0,
    coupling_gap=20.0,
)

component = gf.Component(name="comsol_cpw_resonator")
component << resonator

left_straight = component << gf.components.straight(
    length=LEFT_EXTENSION_UM, cross_section=cross_section
)
right_straight = component << gf.components.straight(
    length=RIGHT_EXTENSION_UM, cross_section=cross_section
)
left_straight.connect("o2", resonator.ports["coupling_o1"])
right_straight.connect("o1", resonator.ports["coupling_o2"])
component.add_port("input", port=left_straight.ports["o1"])
component.add_port("output", port=right_straight.ports["o2"])

print(f"Extended component: {component.name}")
print(f"Bounding box (µm): {component.bbox()}")
for port in component.ports:
    print(
        f"  {port.name}: center={tuple(round(value, 3) for value in port.center)} µm, "
        f"width={port.width} µm, orientation={port.orientation}°"
    )

# %%
layout = prepare_comsol_layout(
    component,
    feed_ports=("input", "output"),
    ground_margin=GROUND_MARGIN_UM,
    crop_to_feed_ports=True,
)

hole_count = sum(len(polygon.holes) for polygon in layout.polygons)
print(f"Metal polygons: {len(layout.polygons)} ({hole_count} holes total)")
print(f"Prepared bounding box (µm): {layout.bbox}")
for feed in layout.feed_ports:
    print(
        f"Feed {feed.name}: center={feed.center} µm, width={feed.width} µm, "
        f"orientation={feed.orientation}°"
    )

# %% [markdown]
# One ground plane with a single hole: the CPW channel, carrying the centre strip, both etch gaps, and
# the surrounding ground. Keeping the hole is what makes `ProjectToFaces` necessary, so the etched
# region stays open in the sheet.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
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
for feed in layout.feed_ports:
    ax.plot(*feed.center, marker="o", color="crimson", markersize=6, zorder=4)
    ax.annotate(
        feed.name,
        feed.center,
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color="crimson",
    )
ax.set_aspect("equal")
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("Ported metal: ground plane, CPW channel, and feed planes")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build the COMSOL model, physics, and study
#
# {py:class}`~qpdk.simulation.comsol_model.COMSOL` builds the air, silicon, and metal, and
# {py:meth}`~qpdk.simulation.comsol_model.COMSOL.add_cpw_rf_study` adds PEC metal and two CPW ports,
# which the port-free study then removes. Mesh sizes are pinned with
# {py:meth}`~qpdk.simulation.comsol_model.COMSOL.pin_absolute_edge_mesh_sizes`, the local size on the
# meander edges where a meander mode's field concentrates, stopping short of the feedline band.

# %% tags=["hide-input"]
RUN_COMSOL = False
# Stage 1, the port-free series, is its own switch so stage 2 can run without it:
# the ported study does not need stage-1 rows, it only reads them as context. Off
# by default, so enabling RUN_COMSOL alone never starts an expensive solve.
RUN_PORT_FREE_SERIES = False
# Stage 2, the ported eigenmode, the driven sweep, and the mesh-tagged ported
# eigen record the "Ported eigen mesh refinement" chart is assembled from. One
# run per meander edge size builds that series, so it is off by default and a
# licensed run has to ask for each row.
RUN_PORTED_DRIVEN = False
# Stage 2 with the driven sweep skipped, so a mesh-refinement row costs one eigen
# solve instead of a full driven window. It writes only this edge size's tagged
# record and field and the rebuilt series, leaving every stable file to the
# coupled run that solved the driven data. Needs RUN_PORTED_DRIVEN = True.
RUN_PORTED_EIGEN_ONLY = False
MODEL_DIR = Path.home() / "comsol_models"
RESULTS_DIR_ENV = "QPDK_COMSOL_RESULTS_DIR"
# The environment variable wins when it is set and non-empty, so a run without a
# license can still read a licensed run's exports.
_results_from_environment = os.environ.get(RESULTS_DIR_ENV)
RESULTS_DIR: Path | None = (
    Path(_results_from_environment).expanduser()
    if _results_from_environment
    else MODEL_DIR
    if RUN_COMSOL
    else None
)
CORES = 4

# Stage 1 is port-free: PEC only, no numeric ports. These two are the search the
# coarser stage-1 row and the stage-2 ported study share.
EIGEN_SHIFT_GHZ = 7.5
EIGEN_MODE_COUNT = 16
# Modes outside this window are not exported or scored; a mode below it is not a
# candidate for a quarter-wave resonance at this meander length. This is the
# stage-1 window; the ported selection has its own, narrower one below.
EIGEN_MODE_WINDOW_GHZ = (3.0, 25.0)
# Rows above this element count record their mesh and skip the solve.
MAX_ELEMENTS = 1_600_000

# Absolute element sizes of the pinned sequence, in µm, the same for every row.
GLOBAL_HMAX_UM = 100.0
GLOBAL_HMIN_UM = 2.0
# The enclosure the sheet model is built with, in µm, the same for every row.
SUBSTRATE_THICKNESS_UM = 200.0
AIR_HEIGHT_UM = 200.0


@dataclass(frozen=True, slots=True)
class EdgeMeshCase:
    """One stage-1 row: its meander-edge sizes and its eigenfrequency search.

    ``eigwhich`` is COMSOL's eigenvalue-selection property. ``None`` leaves the
    solver's own default in place, which is what the coarser row keeps.
    """

    edge_hmax_um: float
    edge_hmin_um: float
    shift_ghz: float
    neigs: int
    eigwhich: str | None = None


# One row per meander-edge mesh, each with its own search. The coarser row keeps
# the original sixteen modes near a 7.5 GHz shift and COMSOL's default selection.
# The finer row searches four modes near a 7.0 GHz shift, largest real part
# first: the same sixteen-mode search at 7.5 GHz did not finish on that mesh
# inside the time it was given, so the finer row trades span of the spectrum for
# a smaller solve.
EDGE_MESH_CONFIGS: tuple[EdgeMeshCase, ...] = (
    EdgeMeshCase(2.0, 0.2, shift_ghz=EIGEN_SHIFT_GHZ, neigs=EIGEN_MODE_COUNT),
    EdgeMeshCase(1.0, 0.1, shift_ghz=7.0, neigs=4, eigwhich="lr"),
)

# Stage 2's ported search is deliberately its own, not a reuse of the stage-1 row
# above. The ported study runs on its own mesh with its own search, and reading a
# stage-1 constant into it would quietly tie the two solves together. Four modes
# at a 7.0 GHz shift with the largest-real-part rule is what the ported solve
# used: the port-free sixteen-mode search at a 7.5 GHz shift left the default
# selection and spent most of its slots on a near-zero cluster, so the shift sits
# under the meander mode and ``lr`` asks for the largest real part relative to
# it instead.
# The meander edge sizes of this ported solve. The default pair is the 4 µm row
# the saved ported eigen and driven results were solved on; solving another pair
# and rerunning adds, or replaces, that pair's row of the ported mesh series.
PORTED_EDGE_HMAX_UM = 4.0
PORTED_EDGE_HMIN_UM = 0.4
PORTED_SHIFT_GHZ = 7.0
PORTED_NEIGS = 4
PORTED_EIGWHICH = "lr"
# The ported mode selection window, in GHz. It is narrower than stage 1's 3 to
# 25 GHz window and sits around the 7.0 GHz ported shift, so the ported scoring
# considers only the candidates the saved batch output selected from. Stage 1
# keeps its own window.
PORTED_MODE_WINDOW_GHZ = (5.0, 9.0)
# A ported mode is called identified only at or above this meander-to-feed p95
# ratio, and only once its field export has been tied to the selected frequency.
PORTED_MIN_MEANDER_FEED_RATIO = 1.0

# The named edge selection and the box behind it.
MEANDER_EDGE_SELECTION = "meander_edges"
MEANDER_EDGE_BOX_UM = {
    "x": (-150.0, 900.0),
    "y": (-900.0, -60.0),
    "z": (-0.02, 0.02),
}
# Feedline band in y, used both to check the edge selection and to score a
# mode's field. The feed runs along x at y = 0.
FEED_Y_UM = (-30.0, 30.0)
# Meander footprint on the cut plane, as (xmin, xmax, ymin, ymax) in µm.
MEANDER_BOX_UM = (-100.0, 850.0, -850.0, -60.0)

# Field cut plane and the mode-evaluation window.
FIELD_CUT_Z = "1[um]"
FIELD_TXT = "comsol_cpw_field.txt"
EIGEN_JSON = "comsol_cpw_port_free_eigen.json"

# Stage 2 tags, kept so the removal and the sweep setups are named explicitly.
PORT_FEATURES = ("port1", "port2")
BOUNDARY_MODE_STEPS = ("bma1", "bma2")
FREQUENCY_STEP = "freq"


def mesh_element_count(model: Any) -> int:
    """Return the number of 3D mesh elements in a model's mesh.

    Args:
        model: A built and meshed MPh model.

    Returns:
        The number of mesh elements.
    """
    return int(model.java.component("comp1").mesh("mesh1").getNumElem())


def enforce_element_budget(label: str, element_count: int) -> None:
    """Refuse to start a stage-2 solve on a mesh over the element limit.

    Stage 2 solves one row at a time and keeps nothing to fall back on, so an
    unexpectedly large mesh is stopped here rather than left to run for hours.
    Stage 1 has the same limit but records a mesh-only row instead of raising,
    because its series is built to keep partial rows.

    Args:
        label: Name of the solve, for the error message.
        element_count: The element count the mesh was pinned to.

    Raises:
        RuntimeError: If the mesh produced no elements, or more than
            ``MAX_ELEMENTS``.
    """
    if element_count == 0:
        raise RuntimeError(f"The {label} mesh produced no elements")
    if element_count > MAX_ELEMENTS:
        raise RuntimeError(
            f"The {label} mesh has {element_count} elements, above the "
            f"{MAX_ELEMENTS} limit, so the solve was not started"
        )


def default_dataset(model: Any) -> tuple[Any, str]:
    """Return MPh's default evaluation dataset node and its Java tag.

    MPh resolves a node path by label rather than by tag, so ``evaluate`` is
    handed the dataset node itself; a Java export still needs the tag. The
    default is whatever COMSOL's own throwaway evaluation node reports. A study
    that carries boundary mode analysis steps leaves more than one dataset on
    the model, and the first tag is not the solution just solved; taking the
    default is.

    Returns:
        The dataset node to pass to ``evaluate``, and its tag.

    Raises:
        RuntimeError: If COMSOL will not name the default dataset, or names one
            that is not on the model. Both mean the numbers read next would come
            off a solution other than the one just solved, so neither is worth
            guessing past.
    """
    evaluation = (model / "evaluations").create("Eval")
    try:
        tag = str(evaluation.property("data"))
    finally:
        with suppress(Exception):
            evaluation.remove()
    for dataset in model / "datasets":
        if dataset.tag() == tag:
            return dataset, tag
    raise RuntimeError(f"the model holds no dataset tagged {tag!r}")


def eigenfrequencies(model: Any) -> list[complex]:
    """Read a solved eigenfrequency model's modes from its default dataset.

    The ported model's study keeps boundary mode analysis steps alongside the
    eigenfrequency step, so the model holds more than one dataset and the first
    tag is not the eigen solution; the default MPh would evaluate on is.

    Args:
        model: A solved eigenfrequency model.

    Returns:
        The complex eigenfrequencies, real part the frequency in Hz.
    """
    dataset, _ = default_dataset(model)
    return [
        complex(value)
        for value in np.atleast_1d(model.evaluate("freq", dataset=dataset)).ravel()
    ]


def frequency_solution(model: Any) -> dict[str, Any]:
    """Read the current frequency solution's S-parameters.

    All evaluations use the dataset node MPh reports as the default, so the
    numbers come off the frequency solution just solved rather than a boundary
    mode analysis dataset.

    Returns:
        Frequency, complex S21 and S11, their dB levels, the power sum, and the
        dataset tag.

    Raises:
        RuntimeError: If the vectors disagree in length, or a frequency or
            complex amplitude is not finite. A dB level of negative infinity is
            allowed only where its own amplitude is exactly zero, which is a real
            S-parameter of 0 and nothing else.
    """
    dataset, tag = default_dataset(model)
    values = {
        "frequency_ghz": np
        .atleast_1d(model.evaluate("freq", dataset=dataset))
        .ravel()
        .real
        / 1e9,
        "s21": np.atleast_1d(model.evaluate("emw.S21", dataset=dataset)).ravel(),
        "s11": np.atleast_1d(model.evaluate("emw.S11", dataset=dataset)).ravel(),
        "s21_db": np
        .atleast_1d(model.evaluate("emw.S21dB", dataset=dataset))
        .ravel()
        .real,
        "s11_db": np
        .atleast_1d(model.evaluate("emw.S11dB", dataset=dataset))
        .ravel()
        .real,
    }
    sizes = {vector.size for vector in values.values()}
    if len(sizes) != 1:
        raise RuntimeError(f"solution vectors of different lengths: {sorted(sizes)}")
    if not np.all(np.isfinite(values["frequency_ghz"])):
        raise RuntimeError("the solution returned a non-finite frequency")
    for amplitude_name in ("s21", "s11"):
        if not np.all(np.isfinite(values[amplitude_name])):
            raise RuntimeError(f"the solution returned a non-finite {amplitude_name}")
    # S = 0 gives a dB level of -inf and is physically possible, so it is only
    # the amplitude that has to be finite; any other non-finite dB is refused.
    for db_name, amplitude_name in (("s21_db", "s21"), ("s11_db", "s11")):
        allowed = np.isfinite(values[db_name]) | (
            np.isneginf(values[db_name]) & (np.abs(values[amplitude_name]) <= 0.0)
        )
        if not np.all(allowed):
            raise RuntimeError(
                f"the solution returned a non-finite {db_name} where the "
                "amplitude is not zero"
            )
    values["dataset"] = tag
    values["power_sum"] = np.abs(values["s11"]) ** 2 + np.abs(values["s21"]) ** 2
    return values


def create_meander_edge_selection(model: Any) -> tuple[int, ...]:
    """Create the named meander edge selection on ``comp1`` and check it.

    The box takes edges with a vertex inside it, which is what reaches the trace
    and gap outlines that cross the meander stripe. A second box over the
    feedline band checks the selection does not also take feed edges; the local
    size must not thin the mesh along the whole feed.

    Args:
        model: A model whose geometry is built.

    Returns:
        The edge entities the meander selection resolved to.

    Raises:
        RuntimeError: If the meander box resolves to no edge, or if it shares an
            edge with the feedline band.
    """
    component = model.java.component("comp1")
    component.selection().create(MEANDER_EDGE_SELECTION, "Box")
    meander = component.selection(MEANDER_EDGE_SELECTION)
    # entitydim is a string: an int makes JPype pick the numeric set() overload.
    meander.set("entitydim", "1")
    meander.set("condition", "somevertex")
    for axis, (low, high) in MEANDER_EDGE_BOX_UM.items():
        meander.set(f"{axis}min", f"{low:g}")
        meander.set(f"{axis}max", f"{high:g}")
    edges = {int(entity) for entity in meander.entities()}
    if not edges:
        raise RuntimeError(
            f"the {MEANDER_EDGE_SELECTION} box resolved to no edge, so the local "
            "size would have nothing to go on"
        )

    component.selection().create("feed_band_edges", "Box")
    feed = component.selection("feed_band_edges")
    feed.set("entitydim", "1")
    feed.set("condition", "intersects")
    feed.set("xmin", f"{layout.bbox.xmin - 1:g}")
    feed.set("xmax", f"{layout.bbox.xmax + 1:g}")
    feed.set("ymin", f"{FEED_Y_UM[0]:g}")
    feed.set("ymax", f"{FEED_Y_UM[1]:g}")
    feed.set("zmin", f"{MEANDER_EDGE_BOX_UM['z'][0]:g}")
    feed.set("zmax", f"{MEANDER_EDGE_BOX_UM['z'][1]:g}")
    feed_edges = {int(entity) for entity in feed.entities()}
    overlap = sorted(edges & feed_edges)
    if not feed_edges:
        raise RuntimeError(
            "the feedline probe resolved to no edge, so the meander box could "
            "not be checked against the feed at all"
        )
    if overlap:
        raise RuntimeError(
            f"the meander edge box also selects the feed edges {overlap}; the "
            "local size would thin the mesh along the whole feed"
        )
    return tuple(sorted(edges))


def configure_port_free_eigen_study(model: Any, case: EdgeMeshCase) -> None:
    """Replace a CPW study with a port-free eigenfrequency search.

    The study steps are removed before the physics they name, so no boundary
    mode analysis is ever left pointing at a deleted port. The shift, the mode
    count, and the eigenvalue selection come from the row, so each edge mesh
    searches the spectrum its own way.

    Args:
        model: A model that
            :func:`~qpdk.simulation.comsol_rf.add_cpw_rf_study` has already
            given a frequency study to.
        case: The row's edge sizes and its eigenfrequency search settings.
    """
    component = model.java.component("comp1")
    study = model.java.study("std1")
    for tag in (*BOUNDARY_MODE_STEPS, FREQUENCY_STEP):
        study.feature().remove(tag)
    for tag in PORT_FEATURES:
        component.physics("emw").feature().remove(tag)
    study.create("eig", "Eigenfrequency")
    eigen = study.feature("eig")
    eigen.set("shiftactive", "on")
    eigen.set("shift", f"{case.shift_ghz:g}[GHz]")
    eigen.set("neigsactive", "on")
    # neigs needs a Java int: JPype cannot pick between COMSOL's overloaded
    # setters for a plain Python int.
    eigen.set("neigs", JInt(case.neigs))
    # A row that names no selection keeps COMSOL's default, which is what the
    # coarser row is recorded as.
    if case.eigwhich is not None:
        eigen.set("eigwhich", case.eigwhich)


def configure_ported_eigen_study(model: Any) -> None:
    """Replace a CPW study's frequency step with a ported eigenfrequency search.

    Unlike the port-free version this keeps ``bma1`` and ``bma2``: the ports
    need their boundary mode fields, and those steps are what supplies them. The
    shift, the mode count, and the eigenvalue selection are the ported constants
    rather than the stage-1 pair.

    Args:
        model: A model that
            :func:`~qpdk.simulation.comsol_rf.add_cpw_rf_study` has already
            given a frequency study to.
    """
    study = model.java.study("std1")
    study.feature().remove(FREQUENCY_STEP)
    study.create("eig", "Eigenfrequency")
    eigen = study.feature("eig")
    eigen.set("shiftactive", "on")
    eigen.set("shift", f"{PORTED_SHIFT_GHZ:g}[GHz]")
    eigen.set("neigsactive", "on")
    eigen.set("neigs", JInt(PORTED_NEIGS))
    eigen.set("eigwhich", PORTED_EIGWHICH)


def field_localization_ratio(path: Path) -> float | None:
    """Score one exported field by how much of it sits on the meander.

    Args:
        path: A ``emw.normE`` export on the cut plane, columns x, y, z, E.

    Returns:
        The 95th percentile of the field inside the meander footprint divided by
        the same percentile over the feedline band, or ``None`` when either
        region holds no exported node or the feed level is not usable as a
        denominator.
    """
    data = np.loadtxt(path, comments="%")
    feed = data[(data[:, 1] > FEED_Y_UM[0]) & (data[:, 1] < FEED_Y_UM[1]), 3]
    xmin, xmax, ymin, ymax = MEANDER_BOX_UM
    meander = data[
        (data[:, 0] > xmin)
        & (data[:, 0] < xmax)
        & (data[:, 1] > ymin)
        & (data[:, 1] < ymax),
        3,
    ]
    if feed.size == 0 or meander.size == 0:
        return None
    feed_p95 = float(np.percentile(feed, 95))
    # A zero, negative, or non-finite denominator makes the ratio meaningless
    # rather than large.
    if not np.isfinite(feed_p95) or feed_p95 <= 0.0:
        return None
    return float(np.percentile(meander, 95)) / feed_p95


def export_mode_field(model: Any, solution_index: int, path: Path) -> None:
    """Export ``emw.normE`` on the cut plane for one eigenmode solution.

    The dataset is created once and reused, and points at the dataset MPh reports
    as the default, which is the eigen solution; a model with boundary mode
    analysis steps holds other datasets first, so the tag is not guessed.

    Args:
        model: A solved eigenfrequency model.
        solution_index: One-based index into the model's solutions.
        path: File to write the export to.
    """
    result = model.java.result()
    if "fieldcut" not in {str(tag) for tag in result.dataset().tags()}:
        _, dataset_tag = default_dataset(model)
        cut = result.dataset().create("fieldcut", "CutPlane")
        cut.set("data", dataset_tag)
        cut.set("planetype", "quick")
        cut.set("quickplane", "xy")
        cut.set("quickz", FIELD_CUT_Z)
    export = result.export().create(f"field{solution_index}", "Data")
    export.set("data", "fieldcut")
    export.set("expr", ["emw.normE"])
    export.set("innerinput", "manual")
    export.set("solnum", str(solution_index))
    export.set("filename", str(path))
    export.run()


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file, then replace in place.

    The series writes on every row, so an interrupt partway through still leaves
    the rows already solved on disk. The temporary file is a sibling so the
    replace stays a same-filesystem rename, which is what makes it atomic.

    Args:
        path: The JSON file to write.
        payload: The object to serialise.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def edge_mesh_label(edge_hmax_um: float, edge_hmin_um: float) -> str:
    """Name one row after the absolute sizes that produced it.

    Args:
        edge_hmax_um: Meander-edge ``hmax`` in µm.
        edge_hmin_um: Meander-edge ``hmin`` in µm.

    Returns:
        A label naming the global sizes and the edge sizes.
    """
    return (
        f"global {GLOBAL_HMAX_UM:g}/{GLOBAL_HMIN_UM:g} µm, "
        f"edges {edge_hmax_um:g}/{edge_hmin_um:g} µm"
    )


def solve_port_free_row(
    client: Any, out_dir: Path, case: EdgeMeshCase
) -> dict[str, Any]:
    """Build, mesh, solve, and score one port-free edge-mesh row.

    A fresh model is built per row so nothing carries over, and it is removed
    from the client even when the mesh or the solve raises. The physics and the
    mesh controls other than the edge sizes match the other rows, so every row
    measures the same thing at a different edge resolution, but the
    eigenfrequency search is the row's own, so the rows do not all cover the
    same span of the spectrum.

    Args:
        client: The MPh client the licensed branch already started.
        out_dir: Directory to write the per-mode field exports into.
        case: The row's edge sizes and its eigenfrequency search settings.

    Returns:
        A row holding the label, both size pairs, the search settings, the
        element count, the mesh element count guard outcome, the full mode
        spectrum, the localisation ratio per mode, and the selected mode.

    Raises:
        RuntimeError: If the mesh produced no elements.
    """
    label = edge_mesh_label(case.edge_hmax_um, case.edge_hmin_um)
    model = COMSOL.create_sheet(
        client,
        layout,
        name=f"QPDK CPW port-free eigen, {label}",
        substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
        air_height_um=AIR_HEIGHT_UM,
    )
    try:
        model.add_cpw_rf_study(
            cpw_gap_um=CPW_GAP_UM,
            frequency_ghz=case.shift_ghz,
            mesh_size=2,
        )
        configure_port_free_eigen_study(model, case)
        edges = create_meander_edge_selection(model)
        element_count = model.pin_absolute_edge_mesh_sizes(
            edge_selection=MEANDER_EDGE_SELECTION,
            global_hmax_um=GLOBAL_HMAX_UM,
            global_hmin_um=GLOBAL_HMIN_UM,
            edge_hmax_um=case.edge_hmax_um,
            edge_hmin_um=case.edge_hmin_um,
        )
        if element_count == 0:
            raise RuntimeError("The mesh produced no elements")
        row: dict[str, Any] = {
            "label": label,
            "edge_hmax_um": case.edge_hmax_um,
            "edge_hmin_um": case.edge_hmin_um,
            "global_hmax_um": GLOBAL_HMAX_UM,
            "global_hmin_um": GLOBAL_HMIN_UM,
            "meander_edges": list(edges),
            "element_count": element_count,
            "shift_ghz": case.shift_ghz,
            "neigs": case.neigs,
            # No selection set means COMSOL's own default was used, so the record
            # says so rather than naming one that was never passed.
            "eigwhich": case.eigwhich if case.eigwhich is not None else "default",
            "mode_window_ghz": list(EIGEN_MODE_WINDOW_GHZ),
        }
        if element_count > MAX_ELEMENTS:
            row["event"] = "mesh_only"
            row["note"] = (
                f"{element_count} elements is above the {MAX_ELEMENTS} limit, so "
                "the eigen solve was not started"
            )
            return row

        model.java.study("std1").run()
        for problem in model.problems():
            print(f"  {label} reports: {problem}")
        # Eigenfrequencies come back complex: f' is the mode frequency and f''
        # its damping. With no ports there is no matched-load decay, so any
        # imaginary part here is numerical.
        modes = [
            complex(value) for value in np.atleast_1d(model.evaluate("freq")).ravel()
        ]
        mode_rows: list[dict[str, Any]] = []
        for index, mode in enumerate(modes, start=1):
            entry: dict[str, Any] = {
                "solution_index": index,
                "real_ghz": mode.real / 1e9,
                "imag_hz": mode.imag,
            }
            if (
                EIGEN_MODE_WINDOW_GHZ[0]
                <= entry["real_ghz"]
                <= EIGEN_MODE_WINDOW_GHZ[1]
            ):
                path = out_dir / f"field_{case.edge_hmax_um:g}um_mode{index}.txt"
                try:
                    export_mode_field(model, index, path)
                    entry["field_file"] = str(path)
                    entry["meander_to_feed_p95"] = field_localization_ratio(path)
                except Exception as error:
                    entry["error"] = repr(error)
            mode_rows.append(entry)
        row["modes"] = mode_rows
        scored = [entry for entry in mode_rows if entry.get("meander_to_feed_p95")]
        if scored:
            selected = max(scored, key=itemgetter("meander_to_feed_p95"))
            row["selected_mode"] = {
                "solution_index": selected["solution_index"],
                "real_ghz": selected["real_ghz"],
                "imag_hz": selected["imag_hz"],
                "meander_to_feed_p95": selected["meander_to_feed_p95"],
                "field_file": selected.get("field_file"),
            }
        row["event"] = "solved"
        return row
    finally:
        client.remove(model)


model = None
if RUN_COMSOL and not MPH_AVAILABLE:
    raise RuntimeError("RUN_COMSOL needs MPh and a licensed COMSOL installation")

# %% [markdown]
# ### Stage 1: the port-free eigenfrequency series
#
# Each row runs its own eigenfrequency search and scores every mode in the window by the
# meander-to-feed 95th percentile ratio of `emw.normE`, taking the highest ratio as the selected mode.
#
# The two rows search different parts of the spectrum with different selection rules, and the tighter
# mesh's wider search never finished, so only one row carries a result and the tighter one stops at the
# element guard as mesh-only. Modes are therefore compared by field rather than by mode number.

# %%
if RUN_COMSOL and MPH_AVAILABLE:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # One MPh client per Python process, shared by every stage, so the ported
    # branch can run on this client with the port-free series skipped.
    client = mph.start(cores=CORES)
else:
    client = None

if RUN_PORT_FREE_SERIES and client is not None:
    eigen_path = MODEL_DIR / EIGEN_JSON
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def save_rows() -> None:
        """Write the rows solved so far, so a late failure keeps them."""
        write_json_atomically(eigen_path, {"rows": rows, "failures": failures})

    save_rows()
    for case in EDGE_MESH_CONFIGS:
        label = edge_mesh_label(case.edge_hmax_um, case.edge_hmin_um)
        print(f"{label}: building, meshing, and solving")
        try:
            row = solve_port_free_row(client, MODEL_DIR, case)
        except Exception as error:
            failures.append({
                "label": label,
                "edge_hmax_um": case.edge_hmax_um,
                "edge_hmin_um": case.edge_hmin_um,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"{label} FAILED: {type(error).__name__}: {error}")
            save_rows()
            continue
        rows.append(row)
        print(f"{label}: {row['element_count']} elements, {row['event']}")
        save_rows()

    print(f"Wrote {len(rows)} of {len(EDGE_MESH_CONFIGS)} rows to {eigen_path}")

    # The field map cell reads a stable file name, so the first solved row's
    # selected mode gets its field copied to one.
    for row in rows:
        if row.get("selected_mode", {}).get("field_file"):
            source = Path(row["selected_mode"]["field_file"])
            (MODEL_DIR / FIELD_TXT).write_bytes(source.read_bytes())
            break

# %% [markdown]
# ## Mode spectrum and localisation
#
# Mode frequency against localisation ratio, so a mode living on the meander separates from the others.
# The two-mesh delta is printed only once both rows exist.

# %%
eigen_file = result_file(EIGEN_JSON)

if eigen_file is None:
    explain_missing_results(EIGEN_JSON)
else:
    eigen_payload = json.loads(eigen_file.read_text())
    eigen_rows = eigen_payload.get("rows") or []
    for failure in eigen_payload.get("failures") or []:
        print(f"{failure['label']} failed: {failure['error']}")
    if not eigen_rows:
        explain_missing_results(EIGEN_JSON)
        print("The file on disk carries no rows.")
    else:
        for row in eigen_rows:
            print(f"\n{row['label']}: {row['element_count']:,} elements")
            print(
                f"  search: shift {row.get('shift_ghz')} GHz, "
                f"{row.get('neigs')} modes, "
                f"eigwhich {row.get('eigwhich', 'not recorded')}"
            )
            if row.get("event") != "solved":
                print(f"  {row.get('note', 'no solve recorded')}")
                continue
            print(
                f"  {'mode':>5} {'frequency (GHz)':>16} {'imag (Hz)':>12} "
                f"{'meander/feed p95':>17}"
            )
            for entry in row.get("modes", []):
                if "real_ghz" not in entry:
                    continue
                ratio = entry.get("meander_to_feed_p95")
                ratio_text = f"{ratio:>17.3f}" if ratio else f"{'not scored':>17}"
                print(
                    f"  {entry['solution_index']:>5} {entry['real_ghz']:>16.6f} "
                    f"{entry['imag_hz']:>+12.3f} {ratio_text}"
                )
            selected = row.get("selected_mode")
            if selected is None:
                print("  No mode inside the window had a usable field export.")
            else:
                print(
                    f"  selected: mode {selected['solution_index']} at "
                    f"{selected['real_ghz']:.6f} GHz, meander/feed p95 "
                    f"{selected['meander_to_feed_p95']:.3f}"
                )

        # Only a row that solved and carries a finite selected frequency can be
        # compared: a mesh-only row has no mode, and a solved row whose field
        # scoring found nothing has no selection to follow.
        solved_rows = [
            row
            for row in eigen_rows
            if row.get("event") == "solved"
            and (row.get("selected_mode") or {}).get("real_ghz") is not None
            and np.isfinite(row["selected_mode"]["real_ghz"])
        ]
        solved_ids = {id(row) for row in solved_rows}
        incomplete_reasons = []
        for row in eigen_rows:
            if id(row) in solved_ids:
                continue
            reason = row.get("note") or row.get("event") or "no result recorded"
            if row.get("event") == "solved":
                reason = "solved but no finite selected mode"
            incomplete_reasons.append(f"{row['label']}: {reason}")
        if incomplete_reasons:
            print("\nRows not usable for the comparison:")
            for reason in incomplete_reasons:
                print(f"  {reason}")

        primary = min(solved_rows, key=itemgetter("element_count"), default=None)
        if primary and primary.get("selected_mode"):
            scored = [
                (entry["real_ghz"], entry["meander_to_feed_p95"])
                for entry in primary["modes"]
                if entry.get("meander_to_feed_p95")
            ]
            frequencies_ghz = np.array([item[0] for item in scored])
            ratios = np.array([item[1] for item in scored])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                frequencies_ghz,
                ratios,
                marker="o",
                linestyle="none",
                label="mode, scored by field",
            )
            selected = primary["selected_mode"]
            ax.plot(
                selected["real_ghz"],
                selected["meander_to_feed_p95"],
                marker="*",
                markersize=14,
                color="crimson",
                linestyle="none",
                label="selected mode",
            )
            ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Mode frequency (GHz)")
            ax.set_ylabel("Meander / feed $|\\mathbf{E}|$ p95")
            ax.set_title(f"Mode localisation, {primary['label']}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.show()

        solved_sizes = {
            (row["edge_hmax_um"], row["edge_hmin_um"]) for row in solved_rows
        }
        missing = [
            edge_mesh_label(case.edge_hmax_um, case.edge_hmin_um)
            for case in EDGE_MESH_CONFIGS
            if (case.edge_hmax_um, case.edge_hmin_um) not in solved_sizes
        ]
        if missing:
            print(
                "\nNo numerical delta yet: still waiting on "
                + ", ".join(missing)
                + ". One solved mesh is not a convergence result."
            )
        else:
            ordered = sorted(solved_rows, key=itemgetter("element_count"))
            coarser, finer = ordered[0], ordered[-1]
            delta_hz = (
                finer["selected_mode"]["real_ghz"]
                - coarser["selected_mode"]["real_ghz"]
            ) * 1e9
            print(
                f"\nNumerical delta between the two edge meshes: "
                f"{delta_hz / 1e6:+.3f} MHz, "
                f"{coarser['element_count']:,} to {finer['element_count']:,} elements, "
                f"ratio {coarser['selected_mode']['meander_to_feed_p95']:.3f} to "
                f"{finer['selected_mode']['meander_to_feed_p95']:.3f}."
            )
            print(
                "Two meshes bound the movement, they do not show convergence; a "
                "third mesh would be needed before calling either settled."
            )
            print(
                "The two rows searched different spans, so this delta is between "
                "each row's best-localised mode. Check that both selections are "
                "the same physical mode before reading it as one mode moving."
            )

            # Only genuinely solved rows with a finite selection reach here, so a
            # missing row is reported above instead of being drawn as a break.
            frequencies_ghz = np.array(
                [row["selected_mode"]["real_ghz"] for row in ordered], dtype=float
            )
            elements = np.array([row["element_count"] for row in ordered], dtype=float)
            has_series = len(ordered) >= 3
            figure, axes = plt.subplots(
                1,
                2 if has_series else 1,
                figsize=(11, 4) if has_series else (7, 4),
                squeeze=False,
            )
            frequency_axis = axes[0][0]
            frequency_axis.plot(
                elements,
                frequencies_ghz,
                marker="o",
                linewidth=1.0 if has_series else 0.0,
                linestyle="-" if has_series else "none",
                label="selected mode",
            )
            for row in ordered:
                frequency_axis.annotate(
                    f"{row['edge_hmax_um']:g}/{row['edge_hmin_um']:g} µm",
                    (row["element_count"], row["selected_mode"]["real_ghz"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )
            frequency_axis.set_xscale("log")
            frequency_axis.set_xlabel("Mesh elements")
            frequency_axis.set_ylabel("Selected mode (GHz)")
            frequency_axis.set_title("Selected mode against meander edge mesh")
            frequency_axis.grid(True, which="both", alpha=0.3)
            frequency_axis.annotate(
                f"final delta {delta_hz / 1e6:+.3f} MHz",
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                va="top",
                fontsize=9,
            )
            if has_series:
                shift_axis = axes[0][1]
                shifts_mhz = np.diff(frequencies_ghz) * 1e3
                shift_axis.plot(elements[1:], shifts_mhz, marker="s", linewidth=1.0)
                for element, shift in zip(elements[1:], shifts_mhz, strict=False):
                    shift_axis.annotate(
                        f"{shift:+.3f}",
                        (element, shift),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                    )
                shift_axis.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
                shift_axis.set_xscale("log")
                shift_axis.set_xlabel("Mesh elements")
                shift_axis.set_ylabel("Incremental shift (MHz)")
                shift_axis.set_title("Change from the previous mesh")
                shift_axis.grid(True, which="both", alpha=0.3)
            figure.suptitle(
                (
                    "Three or more meshes bound the movement, still not an asymptote"
                    if has_series
                    else "Two meshes measure the change, not asymptotic convergence"
                ),
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ## Field map
#
# `emw.normE` on a cut plane just above the metal for the selected port-free mode, read from the
# export's own frequency annotation. The amplitude follows the solver's eigenvector normalization, so
# the colour scale shows relative shape rather than a field strength at a specified drive power. The
# figure crops to the coupling section, the meander, and the feedline, and for **display only** draws a
# fixed-stride subset of the cropped nodes.

# %%
# xmin, xmax, ymin, ymax: the coupling section, the meander, and the feedline.
FIELD_VIEW_UM = (-200.0, 900.0, -950.0, 150.0)
# Drawing stride over the cropped nodes, and contour levels. Display only.
FIELD_DISPLAY_STRIDE = 4
FIELD_CONTOUR_LEVELS = 30
field_file = result_file(FIELD_TXT)

if field_file is None:
    explain_missing_results(FIELD_TXT)
else:
    field = np.loadtxt(field_file, comments="%")
    field_x, field_y, field_e = field[:, 0], field[:, 1], field[:, 3]

    # Crop before triangulating: only the nodes in the window reach the figure,
    # and with them the SVG the documentation embeds.
    field_view = (
        (field_x >= FIELD_VIEW_UM[0])
        & (field_x <= FIELD_VIEW_UM[1])
        & (field_y >= FIELD_VIEW_UM[2])
        & (field_y <= FIELD_VIEW_UM[3])
    )
    view_x, view_y, view_e = (
        field_x[field_view],
        field_y[field_view],
        field_e[field_view],
    )
    if view_e.size == 0:
        raise ValueError(f"No exported field nodes inside {FIELD_VIEW_UM}")

    # Colour limits from the data in the window rather than the full-domain
    # maximum, which sits far outside it and flattens everything on the scale.
    color_min, color_max = (float(value) for value in np.percentile(view_e, [1, 99]))
    draw_x, draw_y, draw_e = view_x, view_y, view_e
    if FIELD_DISPLAY_STRIDE > 1 and view_e.size // FIELD_DISPLAY_STRIDE >= 10:
        draw_x = view_x[::FIELD_DISPLAY_STRIDE]
        draw_y = view_y[::FIELD_DISPLAY_STRIDE]
        draw_e = view_e[::FIELD_DISPLAY_STRIDE]
    print(
        f"Field nodes: {view_e.size} of {field_e.size} inside the view; "
        f"{draw_e.size} drawn at stride {FIELD_DISPLAY_STRIDE}; "
        f"range {view_e.min():.3g} to {view_e.max():.3g} V/m, "
        f"1st to 99th percentile {color_min:.3g} to {color_max:.3g} V/m"
    )

    field_frequency_ghz = exported_frequency_ghz(field_file)
    frequency_label = (
        f"{field_frequency_ghz:g} GHz, " if field_frequency_ghz is not None else ""
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.tricontourf(
        draw_x,
        draw_y,
        draw_e,
        levels=np.geomspace(color_min, color_max, FIELD_CONTOUR_LEVELS),
        norm=LogNorm(vmin=color_min, vmax=color_max),
        cmap="inferno",
        extend="both",
    )
    ax.set_xlim(FIELD_VIEW_UM[0], FIELD_VIEW_UM[1])
    ax.set_ylim(FIELD_VIEW_UM[2], FIELD_VIEW_UM[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Electric field norm at {frequency_label}z = 1 µm")
    fig.colorbar(contour, ax=ax, label=r"$|\mathbf{E}|$ (V/m)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# The field is concentrated on the meander, which is what a meander-localised mode should look like; the
# ratio printed alongside the spectrum is the numerical version of the same statement. It does not show
# that this is the quarter-wave resonance: localisation is not the quarter-wave condition, and this is
# the port-free mode.

# %% [markdown]
# ## Stage 2: ported eigenmode and a driven curve verified by direct solves
#
# Stage 2 keeps both numeric TEM ports and their boundary mode analysis steps, and solves a ported
# eigenfrequency search followed by a driven window centred on the selected loaded mode. Each run adds
# one row to the mesh series below, tagged with its meander edge sizes.
#
# A direct solve takes minutes, so the window uses COMSOL's **adaptive frequency sweep** (AWE): a
# rational fit to a handful of solved points fills the rest of the curve, so **its rows are
# interpolation, not independent solves**. The notch evidence sits outside that curve: direct solves
# with AWE off at the two flanks, the centre, and the curve's minimum; a notch verdict at the
# **directly solved minimum only**, since a loaded mode can sit off the ported eigenfrequency; and a
# two-port power balance near unity on the direct solves only. A run that fails those checks is reported
# **unverified**.

# %% tags=["hide-input"]
PORTED_EIGEN_JSON = "comsol_cpw_ported_eigen.json"
# One mesh-tagged copy of the ported eigen record per meander edge mesh, written
# beside the stable record so successive edge sizes accumulate instead of
# overwriting each other, and the series the chart cell reads, rebuilt from those
# copies rather than appended to.
PORTED_EIGEN_TAGGED_PREFIX = "comsol_cpw_ported_eigen_edge"
PORTED_FIELD_TAGGED_PREFIX = "comsol_cpw_ported_field_edge"
PORTED_MESH_SERIES_JSON = "comsol_cpw_ported_mesh_series.json"
# The curve and the verification record are named after a per-run ID, so a reader
# never pairs a fresh curve with a stale verification record.
AWE_CURVE_PREFIX = "comsol_cpw_awe_curve"
DIRECT_PREFIX = "comsol_cpw_direct_points"
# The driven window around the ported mode. Its span comes from the notch width
# implied by the mode's damping: a complex eigenfrequency f' + i f'' implies a
# notch 2|f''| wide, so the span is a few of those widths. It is floored so a
# window sized from an under-estimated damping cannot sit inside the notch, and
# capped so the adaptive stage stays bounded.
AWE_HALF_SPAN_FWHM = 5.0
AWE_HALF_SPAN_GHZ_MIN = 1.0e-3
AWE_HALF_SPAN_GHZ_MAX = 5.0e-3
FWHM_GUESS_GHZ = 1.0e-4
# Requested curve rows per notch width, and a ceiling on the request. These are
# curve rows, not solves. The window is a fixed half-span at the loaded damping
# (AWE_HALF_SPAN_FWHM of the implied width, floored and capped above), not that
# many widths of the raw FWHM, so the request is the span the range actually
# sweeps divided by this fraction of the implied notch width.
AWE_POINTS_PER_FWHM = 10.0
MAX_SWEEP_POINTS = 1001
# How far an AWE row may differ from a direct solve at the same frequency.
AWE_AGREEMENT_DB = 0.5
# A directly solved minimum has to sit this far below the lower direct flank.
NOTCH_MIN_DEPTH_DB = 1.0
# Two-port power balance band. PEC metal and a lossless dielectric carry no loss,
# so |S11|^2 + |S21|^2 should be 1; at a direct solve a deviation this large means
# power is leaving through a channel this two-port record does not see, and the run
# is unverified. The band is applied to the AWE curve only as a diagnostic: its
# fitted rows can sit a little outside it without that being physical.
POWER_SUM_MIN = 0.99
POWER_SUM_MAX = 1.001
# A direct point and the AWE curve are both built from the same formatted grid
# tokens, but they can still differ by a fraction of a hertz. A frequency this
# far outside the reconstructed rows is clamped to the nearest endpoint; further
# out, the comparison is refused rather than accepted at a distant row.
CURVE_ENDPOINT_TOLERANCE_GHZ = 1.0e-7
# How far the saved direct "AWE minimum" frequency may sit from the minimum of
# the curve as loaded. One kilohertz is well under one row here (~106 kHz) and
# well over the grid's rounding, so it accepts the writer's own result while
# catching a minimum that moved by a row.
DIRECT_MINIMUM_FREQUENCY_TOLERANCE_GHZ = 1.0e-6


def ported_mode_field_path(solution_index: int) -> Path:
    """Return the per-mode field export path for one ported solution.

    Args:
        solution_index: One-based index into the ported model's solutions.

    Returns:
        The path the field export is written to.
    """
    return MODEL_DIR / f"comsol_cpw_ported_mode{solution_index}.txt"


def select_ported_mode(model: Any, modes: list[complex]) -> tuple[complex, Path, float]:
    """Pick the ported mode whose field sits on the meander.

    The ported selection repeats stage 1's field scoring rather than reusing
    stage 1's answer: the ports load the mode, so the localised ported mode has
    to be identified from the ported solve. Only modes inside
    ``PORTED_MODE_WINDOW_GHZ`` are candidates, which is a narrower band than the
    stage-1 window.

    Args:
        model: A solved ported eigenfrequency model.
        modes: The complex eigenfrequencies, real part the frequency in Hz.

    Returns:
        The selected mode, its field export path, and its meander-to-feed ratio.

    Raises:
        ValueError: If no mode inside the search window has a usable field
            export.
    """
    scored: list[tuple[float, complex, Path]] = []
    for index, mode in enumerate(modes, start=1):
        real_ghz = mode.real / 1e9
        if not PORTED_MODE_WINDOW_GHZ[0] <= real_ghz <= PORTED_MODE_WINDOW_GHZ[1]:
            continue
        path = ported_mode_field_path(index)
        export_mode_field(model, index, path)
        ratio = field_localization_ratio(path)
        if ratio is not None:
            scored.append((ratio, mode, path))
    if not scored:
        raise ValueError(
            "No ported mode inside "
            f"{PORTED_MODE_WINDOW_GHZ[0]:g} to {PORTED_MODE_WINDOW_GHZ[1]:g} GHz "
            "had a usable field export: "
            f"{[mode.real / 1e9 for mode in modes]} GHz"
        )
    ratio, mode, path = max(scored, key=itemgetter(0))
    return mode, path, ratio


def identify_ported_mode(
    field_file: Path, selected: complex, ratio: float | None
) -> dict[str, Any]:
    """Check a ported field export against the selected mode's frequency.

    The export's own header carries the complex eigenfrequency of the solution it
    was written from, so its real part is what ties the field to the mode. A
    field whose header disagrees, or is missing, is not the selected mode's field
    and must not be presented as identified, whatever ratio it scored.

    Args:
        field_file: The stable ported field export.
        selected: The selected ported eigenfrequency, real part the frequency in
            Hz.
        ratio: The export's meander-to-feed p95 ratio, or ``None``.

    Returns:
        The identified flag, the header's real frequency, the relative
        difference, and the reason.
    """
    selected_ghz = selected.real / 1e9
    annotation = complex_frequency_ghz(field_file)
    if annotation is None:
        return {
            "selected_mode_identified": False,
            "field_annotated_ghz": None,
            "relative_difference": None,
            "reason": (
                f"{field_file.name} carries no complex frequency annotation, so "
                "the field cannot be tied to the selected ported mode"
            ),
        }
    field_real_ghz, _ = annotation
    relative = abs(field_real_ghz - selected_ghz) / abs(selected_ghz)
    header_agrees = relative <= PORTED_FIELD_FREQUENCY_RTOL
    ratio_ok = ratio is not None and ratio >= PORTED_MIN_MEANDER_FEED_RATIO
    return {
        "selected_mode_identified": bool(header_agrees and ratio_ok),
        "field_annotated_ghz": field_real_ghz,
        "relative_difference": relative,
        "reason": (
            f"the field export is annotated {field_real_ghz:.9f} GHz against the "
            f"selected {selected_ghz:.9f} GHz (relative difference {relative:.3e}, "
            f"tolerance {PORTED_FIELD_FREQUENCY_RTOL:g}) "
            + (
                "and its meander/feed p95 ratio reaches "
                f"{PORTED_MIN_MEANDER_FEED_RATIO:g}, so the field is the selected "
                "mode's field"
                if header_agrees and ratio_ok
                else "so the field is not identified as the selected mode's field"
            )
        ),
    }


def positive_number(value: Any) -> float | None:
    """Return ``value`` as a float when it is a finite positive number.

    Args:
        value: A value read back from a result record.

    Returns:
        The value as a float, or ``None`` when it is not a number, not finite, or
        not positive. A ``bool`` is not a number here: ``True`` would otherwise
        pass as 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    number = float(value)
    return number if np.isfinite(number) and number > 0.0 else None


def rounded_coordinates(points: Any) -> list[list[float]]:
    """Round a polygon coordinate sequence to JSON-stable floats.

    Args:
        points: A sequence of ``(x, y)`` pairs, as a prepared layout stores them.

    Returns:
        The coordinates as plain floats rounded to 6 decimals of a micrometre,
        which is a picometre: far finer than the geometry is built at, and coarse
        enough that two runs of the same layout produce the same numbers.
    """
    return [[round(float(x), 6), round(float(y), 6)] for x, y in points]


def ported_layout_signature(layout: Any) -> str:
    """Return a deterministic signature of the ported layout and its RF setup.

    Two ported eigen solves belong beside each other in one mesh series only when
    they discretise the same device: the same prepared metal polygons and feed
    planes, the same enclosure, under the same mesh-independent RF settings. The
    signature makes that check mechanical, so a record solved on a different
    layout, in a different enclosure, with a different port setup, or with a
    different search is left out of the series and reported rather than charted as
    if the mesh were the only thing that changed. The meander edge box and the
    field cut plane are in it as well, because both change which edges carry the
    local size and which plane the mode is scored on, and a ratio from another
    selection or another cut is not this row's. Only the meander-edge sizes are
    outside it, because refining them is what the series measures.

    Args:
        layout: The prepared layout the ported model is built from.

    Returns:
        A short hex digest of the rounded geometry and the settings.
    """
    geometry = {
        "bbox_um": [
            round(float(value), 6)
            for value in (
                layout.bbox.xmin,
                layout.bbox.ymin,
                layout.bbox.xmax,
                layout.bbox.ymax,
            )
        ],
        # Sorted serialised entries rather than lists: a layout that hands its
        # polygons or feeds back in another order is still the same layout.
        "polygons": sorted(
            json.dumps(
                {
                    "outline": rounded_coordinates(polygon.outline),
                    "holes": [rounded_coordinates(hole) for hole in polygon.holes],
                },
                sort_keys=True,
            )
            for polygon in layout.polygons
        ),
        "feed_ports": sorted(
            json.dumps(
                {
                    "name": str(feed.name),
                    "center_um": [round(float(value), 6) for value in feed.center],
                    "width_um": round(float(feed.width), 6),
                    "orientation_deg": round(float(feed.orientation), 6),
                },
                sort_keys=True,
            )
            for feed in layout.feed_ports
        ),
    }
    settings = {
        "cpw_width_um": CPW_WIDTH_UM,
        "cpw_gap_um": CPW_GAP_UM,
        "global_hmax_um": GLOBAL_HMAX_UM,
        "global_hmin_um": GLOBAL_HMIN_UM,
        "substrate_thickness_um": SUBSTRATE_THICKNESS_UM,
        "air_height_um": AIR_HEIGHT_UM,
        "ported_shift_ghz": PORTED_SHIFT_GHZ,
        "ported_neigs": PORTED_NEIGS,
        "ported_eigwhich": PORTED_EIGWHICH,
        "min_meander_feed_ratio": PORTED_MIN_MEANDER_FEED_RATIO,
        "mode_window_ghz": list(PORTED_MODE_WINDOW_GHZ),
        "feed_y_um": list(FEED_Y_UM),
        "meander_box_um": MEANDER_BOX_UM,
        # Both change what a row measures, not just how finely it discretises.
        "meander_edge_box_um": MEANDER_EDGE_BOX_UM,
        "field_cut_z": FIELD_CUT_Z,
    }
    payload = json.dumps({"geometry": geometry, "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ported_eigen_record_path(edge_hmax_um: float, edge_hmin_um: float) -> Path:
    """Return the mesh-tagged ported eigen record path for one edge mesh.

    The tag is the edge sizes, so re-solving a mesh replaces its own record and
    solving another mesh adds one, which is what lets the series be rebuilt from
    the directory on every run instead of appended to.

    Args:
        edge_hmax_um: Meander-edge ``hmax`` in µm.
        edge_hmin_um: Meander-edge ``hmin`` in µm.

    Returns:
        The path of this mesh's ported eigen record.
    """
    return MODEL_DIR / (
        f"{PORTED_EIGEN_TAGGED_PREFIX}_{edge_hmax_um:g}um_{edge_hmin_um:g}um.json"
    )


def ported_field_record_path(edge_hmax_um: float, edge_hmin_um: float) -> Path:
    """Return the mesh-tagged copy of one edge mesh's selected field export.

    The tagged eigen record points at this copy rather than at ``PORTED_FIELD_TXT``,
    which only a completed coupled run publishes, so each tagged record keeps the
    field it was actually solved with.

    Args:
        edge_hmax_um: Meander-edge ``hmax`` in µm.
        edge_hmin_um: Meander-edge ``hmin`` in µm.

    Returns:
        The path of this mesh's selected field export.
    """
    return MODEL_DIR / (
        f"{PORTED_FIELD_TAGGED_PREFIX}_{edge_hmax_um:g}um_{edge_hmin_um:g}um.txt"
    )


def publish_ported_stable(record: dict[str, Any], field_source: Path) -> None:
    """Publish the stable ported eigen record and field of a coupled run.

    Both are written only once the driven window has finished and its model is
    saved, so a driven solve that fails leaves the stable pair on the mesh its
    driven data came from.

    Args:
        record: The ported eigen record to publish, naming the stable field file.
        field_source: The selected mode's field export this run solved.
    """
    field_path = MODEL_DIR / PORTED_FIELD_TXT
    temporary = field_path.with_name(field_path.name + ".tmp")
    temporary.write_bytes(field_source.read_bytes())
    temporary.replace(field_path)
    write_json_atomically(MODEL_DIR / PORTED_EIGEN_JSON, record)


def driven_pairing_problems(direct: dict[str, Any]) -> list[str]:
    """Check a driven record is paired with the stable ported eigen row it was solved with.

    The driven record is checkpointed before the stable row is published, so a save
    that fails leaves the two from different runs.

    Args:
        direct: The driven verification record that was read.

    Returns:
        A list of reasons the record is not paired, empty when it is.
    """
    record_file = result_file(PORTED_EIGEN_JSON)
    if record_file is None:
        return [
            (
                f"the stable {PORTED_EIGEN_JSON} is not in RESULTS_DIR "
                f"({RESULTS_DIR}), so the driven record cannot be paired with the "
                "ported eigen row it was solved with"
            )
        ]
    try:
        stable = json.loads(record_file.read_text())
    except (OSError, ValueError) as error:
        return [f"{PORTED_EIGEN_JSON} is unreadable ({error})"]
    if not isinstance(stable, dict):
        return [f"{PORTED_EIGEN_JSON} does not hold a record object"]
    problems: list[str] = []
    for key in (
        "run_id",
        "edge_hmax_um",
        "edge_hmin_um",
        "element_count",
        "layout_signature",
    ):
        stable_value = stable.get(key)
        direct_value = direct.get(key)
        # A value missing from either record is not agreement.
        if stable_value is None or direct_value is None:
            problems.append(
                f"{key} is missing from {PORTED_EIGEN_JSON} ({stable_value!r}) or "
                f"from the driven record ({direct_value!r})"
            )
        elif stable_value != direct_value:
            problems.append(
                f"{key} is {stable_value!r} in {PORTED_EIGEN_JSON} against "
                f"{direct_value!r} in the driven record"
            )
    return problems


def ported_series_row(
    record: dict[str, Any], signature: str
) -> tuple[dict[str, Any] | None, str]:
    """Build one chart row from a solved ported eigen record.

    Every value comes from the record itself: nothing is interpolated, corrected,
    or filled in from another row. A record is refused when it was not solved for
    this layout and RF setup, when one of the fields the chart needs is not
    finite and positive, or when the selected mode was not identified as the
    meander mode, and the reason is returned so the read side can report it.

    Args:
        record: A mesh-tagged ported eigen record.
        signature: The signature of the layout and RF setup the series is for.

    Returns:
        The row, or ``None`` and the reason the record was refused.
    """
    if record.get("layout_signature") != signature:
        return None, (
            "solved for a different layout or RF setup (signature "
            f"{record.get('layout_signature', 'not recorded')!r}, this run "
            f"{signature!r})"
        )
    if record.get("selected_mode_identified") is not True:
        return None, (
            "the selected ported mode was not identified as the meander mode: "
            + str(record.get("selected_mode_identified_reason", "no reason recorded"))
        )
    selected = record.get("selected_mode_hz") or {}
    fields = {
        "edge_hmax_um": record.get("edge_hmax_um"),
        "edge_hmin_um": record.get("edge_hmin_um"),
        "element_count": record.get("element_count"),
        "meander_to_feed_p95": selected.get("meander_to_feed_p95"),
    }
    checked: dict[str, float] = {}
    for key, value in fields.items():
        number = positive_number(value)
        if number is None:
            return None, f"{key} is {value!r}, not finite and positive"
        checked[key] = number
    # The frequency is read in Hz and converted once, so the loaded ratio is
    # formed from the solver's own units. The imaginary part is signed by the
    # solver's own convention, so that ratio uses its magnitude; a zero damping
    # would leave it undefined.
    real_hz = positive_number(selected.get("real"))
    if real_hz is None:
        return None, (
            f"the selected mode's real part is {selected.get('real')!r}, not finite "
            "and positive"
        )
    imag_hz = selected.get("imag")
    if (
        isinstance(imag_hz, bool)
        or not isinstance(imag_hz, (float, int))
        or not np.isfinite(imag_hz)
        or not imag_hz
    ):
        return None, (
            f"the selected mode's imaginary part is {imag_hz!r}, so its loaded "
            "damping ratio is undefined"
        )
    return (
        {
            "edge_hmax_um": checked["edge_hmax_um"],
            "edge_hmin_um": checked["edge_hmin_um"],
            "element_count": int(checked["element_count"]),
            "frequency_ghz": real_hz / 1e9,
            "imag_hz": float(imag_hz),
            "q": abs(real_hz / (2.0 * float(imag_hz))),
            "meander_to_feed_p95": checked["meander_to_feed_p95"],
            "selected_mode_identified": True,
        },
        "",
    )


def update_ported_mesh_series(out_dir: Path, signature: str) -> dict[str, Any]:
    """Rebuild the ported mesh series from every valid tagged eigen record.

    The file is rewritten from the records on disk rather than appended to, so
    re-solving one edge size replaces that row instead of duplicating it, and a
    record from another layout or RF setup is reported and left out. Rows are the
    records' own numbers, sorted coarse to fine by element count.

    Args:
        out_dir: The directory holding the mesh-tagged ported eigen records.
        signature: The signature of the layout and RF setup the series is for.

    Returns:
        The payload written to ``PORTED_MESH_SERIES_JSON``.
    """
    rows: list[dict[str, Any]] = []
    rejected: list[str] = []
    for path in sorted(out_dir.glob(f"{PORTED_EIGEN_TAGGED_PREFIX}_*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            rejected.append(f"{path.name}: unreadable ({error})")
            continue
        if not isinstance(record, dict):
            rejected.append(f"{path.name}: not a record object")
            continue
        row, reason = ported_series_row(record, signature)
        if row is None:
            rejected.append(f"{path.name}: {reason}")
            continue
        rows.append(row)
    rows.sort(key=itemgetter("element_count"))
    payload = {"signature": signature, "rows": rows, "rejected": rejected}
    write_json_atomically(out_dir / PORTED_MESH_SERIES_JSON, payload)
    return payload


def sweep_fwhm_ghz(imag_hz: float) -> float:
    """Return the notch width implied by a mode's damping.

    A complex eigenfrequency ``f' + i f''`` implies a notch ``2|f''|`` wide.

    Args:
        imag_hz: Imaginary part of the mode's eigenfrequency, in Hz.

    Returns:
        ``2|f''|`` in GHz, or the fallback guess when the damping is zero.
    """
    fwhm_ghz = 2.0 * abs(imag_hz) / 1e9 if imag_hz else 0.0
    return fwhm_ghz if fwhm_ghz > 0.0 else FWHM_GUESS_GHZ


def sweep_half_span_ghz(imag_hz: float) -> float:
    """Return the half-span of the driven window around a mode.

    Args:
        imag_hz: Imaginary part of the mode's eigenfrequency, in Hz.

    Returns:
        The half-span in GHz, floored and capped as described above.
    """
    fwhm_ghz = sweep_fwhm_ghz(imag_hz)
    return min(
        max(AWE_HALF_SPAN_FWHM * fwhm_ghz, AWE_HALF_SPAN_GHZ_MIN),
        AWE_HALF_SPAN_GHZ_MAX,
    )


def sweep_requested_points(imag_hz: float) -> int:
    """Return the requested adaptive-curve row count for a mode's window.

    The request is the span the range actually sweeps divided by a fraction of
    the implied notch width, so ten rows land inside each loading width whether
    the span came from the damping or from the floor. The count is floored at
    five rows and capped so the stage stays inside a wall-clock budget.

    Args:
        imag_hz: Imaginary part of the mode's eigenfrequency, in Hz.

    Returns:
        The number of curve rows to request.
    """
    fwhm_ghz = sweep_fwhm_ghz(imag_hz)
    points = (
        round(2.0 * sweep_half_span_ghz(imag_hz) / (fwhm_ghz / AWE_POINTS_PER_FWHM)) + 1
    )
    points = max(min(points, MAX_SWEEP_POINTS), 5)
    return points + (points % 2 == 0)


def requested_frequency_grid(
    low_ghz: float, high_ghz: float, points: int
) -> tuple[str, np.ndarray]:
    """Return the COMSOL ``range`` for a requested grid, and that exact grid.

    The start and step are formatted once and the grid is rebuilt from those
    formatted tokens, so the grid checked here is the grid COMSOL will build.
    Formatting all three tokens independently is what goes wrong: a rounded step
    can make ``start + (points - 1) * step`` exceed a separately rounded stop by a
    few millihertz, and an inclusive range then returns only ``points - 1`` rows.
    The stop here is the intended last frequency plus half a step, so rounding
    cannot push the last point past it, while the point one step further is still
    beyond it, so the range returns exactly ``points`` frequencies.

    Args:
        low_ghz: Low end of the physical window, in GHz.
        high_ghz: High end of the physical window, in GHz.
        points: Number of grid points.

    Returns:
        The ``range`` expression, and the ``points``-long grid it should return.
    """
    step_ghz = (high_ghz - low_ghz) / (points - 1)
    start_token = f"{low_ghz:.12g}"
    step_token = f"{step_ghz:.12g}"
    start_ghz = float(start_token)
    step_rounded_ghz = float(step_token)
    grid_ghz = start_ghz + step_rounded_ghz * np.arange(points)
    stop_ghz = grid_ghz[-1] + 0.5 * step_rounded_ghz
    expression = f"range({start_token}[GHz],{step_token}[GHz],{stop_ghz:.12g}[GHz])"
    return expression, grid_ghz


def power_balance(power_sum: float) -> dict[str, Any]:
    """Judge one frequency's two-port power balance.

    Args:
        power_sum: ``|S11|^2 + |S21|^2`` at that frequency.

    Returns:
        The sum, its deficit from unity, and whether it sits inside the band.
    """
    deficit = 1.0 - power_sum
    return {
        "power_sum": float(power_sum),
        "power_deficit": float(deficit),
        "power_within_band": bool(POWER_SUM_MIN <= power_sum <= POWER_SUM_MAX),
    }


def solve_direct_point(model: Any, label: str, frequency_ghz: float) -> dict[str, Any]:
    """Solve one frequency with the adaptive sweep off and read it back.

    A direct solve is the only number a notch verdict rests on: part of an AWE
    curve is COMSOL's rational fit rather than solved points.

    Args:
        model: The built and meshed driven model.
        label: Name of this point for the record.
        frequency_ghz: The frequency to solve, in GHz.

    Returns:
        The point's frequency, S-parameters, and power balance.

    Raises:
        RuntimeError: If the one-point solve does not come back at the frequency
            it was asked for.
    """
    study = model.java.study("std1")
    feature = study.feature(FREQUENCY_STEP)
    # The same finite precision the requested grid is formatted with, so a point
    # asked for at a grid frequency comes back at that grid frequency.
    feature.set("plist", f"{frequency_ghz:.12g}[GHz]")
    feature.set("awe", "off")
    study.run()
    values = frequency_solution(model)
    if values["frequency_ghz"].size != 1:
        raise RuntimeError(
            f"a one-point solve returned {values['frequency_ghz'].size} frequencies"
        )
    solved_ghz = float(values["frequency_ghz"][0])
    if abs(solved_ghz - frequency_ghz) / frequency_ghz > 1e-9:
        raise RuntimeError(
            f"a solve asked for {frequency_ghz:.9f} GHz came back at "
            f"{solved_ghz:.9f} GHz"
        )
    return {
        "label": label,
        "frequency_ghz": solved_ghz,
        "s21_db": float(values["s21_db"][0]),
        "s11_db": float(values["s11_db"][0]),
        "dataset": values["dataset"],
    } | power_balance(float(values["power_sum"][0]))


def solve_awe_curve(
    model: Any, low_ghz: float, high_ghz: float, points: int, path: Path
) -> dict[str, Any]:
    """Solve the adaptive sweep and save its reconstructed curve.

    The curve's rows are a mix of solved points and values from COMSOL's rational
    fit, so it locates a feature but does not resolve it. The returned rows are
    refused unless they cover the requested grid, which is what makes the curve
    dense rather than a handful of rows.

    Args:
        model: The built and meshed driven model.
        low_ghz: Low end of the window, in GHz.
        high_ghz: High end of the window, in GHz.
        points: Requested number of curve rows.
        path: CSV file to write the curve to.

    Returns:
        The curve's frequency, S21 (dB), and S11 (dB) arrays, the requested grid
        it was checked against, and the curve's own power balance.

    Raises:
        RuntimeError: If the curve does not contain every point of the exact
            requested grid, or returns a row past the requested last frequency,
            so it is not the dense curve it was asked for.

    A reconstructed row that reports a little more power than it is driven with
    is not refused here. The rational fit can put a row slightly outside the
    passive band, and that fitted surplus is a property of the fit, not proof
    that power leaves the model. It is reported as a diagnostic instead, and the
    passivity judgement stays on the direct solves.
    """
    expression, requested_ghz = requested_frequency_grid(low_ghz, high_ghz, points)
    study = model.java.study("std1")
    feature = study.feature(FREQUENCY_STEP)
    feature.set("plist", expression)
    feature.set("awe", "on")
    feature.set("awefunc", ["abs(comp1.emw.S11)"])
    study.run()
    values = frequency_solution(model)
    order = np.argsort(values["frequency_ghz"])
    frequencies_ghz = values["frequency_ghz"][order]
    s21_db = values["s21_db"][order]
    s11_db = values["s11_db"][order]
    power_sum = values["power_sum"][order]
    step_ghz = float(requested_ghz[1] - requested_ghz[0])
    atol_ghz = max(1e-8, 1e-3 * step_ghz)
    # The returned rows must contain every point of the exact requested grid. A
    # count check alone would pass a grid that dropped an interior point and
    # gained another, and an interpolated identity cannot see a gap at all, so
    # each requested frequency is matched to its nearest returned row. The stop
    # sits half a step past the last frequency, so a row beyond that is a range
    # that ran one point too far.
    if (
        frequencies_ghz.size < 2
        or not np.all(np.isfinite(frequencies_ghz))
        or not np.all(np.diff(frequencies_ghz) > 0.0)
    ):
        raise RuntimeError(
            "the adaptive sweep returned frequencies that are not a finite, "
            "strictly increasing series"
        )
    if frequencies_ghz.size < requested_ghz.size:
        raise RuntimeError(
            f"the adaptive sweep returned {frequencies_ghz.size} rows for a "
            f"{requested_ghz.size}-point requested grid"
        )
    if frequencies_ghz[-1] > requested_ghz[-1] + 0.5 * step_ghz + atol_ghz:
        raise RuntimeError(
            f"the adaptive sweep returned a row at {frequencies_ghz[-1]:.9f} GHz, "
            f"past the requested last frequency {requested_ghz[-1]:.9f} GHz"
        )
    # A requested point outside the returned range must not be clipped onto an
    # endpoint and pass as zero distance, so each side is clamped to a valid row
    # and the gap is the absolute distance to the nearest of the two.
    insertion = np.searchsorted(frequencies_ghz, requested_ghz)
    left = np.clip(insertion - 1, 0, frequencies_ghz.size - 1)
    right = np.clip(insertion, 0, frequencies_ghz.size - 1)
    gaps_ghz = np.minimum(
        np.abs(requested_ghz - frequencies_ghz[left]),
        np.abs(frequencies_ghz[right] - requested_ghz),
    )
    if gaps_ghz.max() > atol_ghz:
        missing = int(np.count_nonzero(gaps_ghz > atol_ghz))
        raise RuntimeError(
            f"the adaptive sweep omitted {missing} point(s) of the requested grid: "
            f"largest gap {gaps_ghz.max():.3e} GHz against a {atol_ghz:.3e} GHz "
            "tolerance"
        )
    np.savetxt(
        path,
        np.column_stack([frequencies_ghz, s21_db, s11_db]),
        delimiter=",",
        header="frequency_ghz,s21_db,s11_db",
        comments="",
    )
    # Worst deviation from unit power is the largest magnitude, not the smallest
    # power sum: a slight fitted surplus gives a negative deficit that must not be
    # mistaken for the best row.
    deficits = 1.0 - power_sum
    worst_index = int(np.argmax(np.abs(deficits)))
    within_band = bool(
        power_sum.min() >= POWER_SUM_MIN and power_sum.max() <= POWER_SUM_MAX
    )
    caveat = (
        ""
        if within_band
        else (
            "reconstructed AWE rows are not fully passive: the rational fit "
            f"reaches |S11|^2 + |S21|^2 = {power_sum.max():.6f}, "
            f"{power_sum.max() - POWER_SUM_MAX:+.2e} above the {POWER_SUM_MAX:g} "
            "band edge. This fitted surplus is not physical loss or gain, and the "
            "passivity verdict rests on the direct solves."
        )
    )
    print(
        f"Curve power balance (diagnostic, reconstructed rows): minimum "
        f"{power_sum.min():.6f}, maximum {power_sum.max():.6f}, worst unit-power "
        f"deviation {deficits[worst_index]:+.2e}, band {POWER_SUM_MIN:g} to "
        f"{POWER_SUM_MAX:g} "
        f"({'in band' if within_band else 'OUT OF BAND'})"
    )
    if caveat:
        print(f"Caveat: {caveat}")
    return {
        "frequencies_ghz": frequencies_ghz,
        "s21_db": s21_db,
        "s11_db": s11_db,
        "requested_points": int(requested_ghz.size),
        "requested_grid_low_ghz": float(requested_ghz[0]),
        "requested_grid_high_ghz": float(requested_ghz[-1]),
        "requested_grid_step_ghz": step_ghz,
        # The physical window is kept separately from the grid: the grid is the
        # formatted range the solver was handed, not the unrounded window.
        "physical_window_ghz": [float(low_ghz), float(high_ghz)],
        "solved_rows": int(frequencies_ghz.size),
        "refinement_rows": int(frequencies_ghz.size - requested_ghz.size),
        "worst_power_sum": float(power_sum[worst_index]),
        "worst_power_deficit": float(deficits[worst_index]),
        "power_sum_min": float(power_sum.min()),
        "power_sum_max": float(power_sum.max()),
        "power_within_band": within_band,
        "power_balance_caveat": caveat,
        "dataset": values["dataset"],
    }


def curve_value_db(curve: dict[str, Any], at_ghz: float) -> float | None:
    """Interpolate the adaptive curve at one directly solved frequency.

    A direct point is asked for at the same finite precision as the grid tokens,
    but the two still differ by a fraction of a hertz from formatting, so a
    frequency a hair outside the reconstructed rows is clamped to the nearest
    endpoint. Materially outside the window it is ``None``, so an out-of-window
    direct point is never accepted silently.

    Args:
        curve: The dict returned by :func:`solve_awe_curve`.
        at_ghz: The frequency to read, in GHz.

    Returns:
        The interpolated ``S21`` level in dB, or ``None`` when the frequency is
        outside the curve by more than the endpoint tolerance.
    """
    frequencies_ghz = curve["frequencies_ghz"]
    if frequencies_ghz.size == 0:
        return None
    if (
        at_ghz < frequencies_ghz[0] - CURVE_ENDPOINT_TOLERANCE_GHZ
        or at_ghz > frequencies_ghz[-1] + CURVE_ENDPOINT_TOLERANCE_GHZ
    ):
        return None
    clamped = float(np.clip(at_ghz, frequencies_ghz[0], frequencies_ghz[-1]))
    return float(np.interp(clamped, frequencies_ghz, curve["s21_db"]))


def notch_verdict(centre_db: float, flank_levels_db: list[float]) -> dict[str, Any]:
    """Decide from directly solved points whether the centre is a notch.

    Only direct solves count: the minimum of an adaptive curve says where to
    look, not that a notch is there.

    Args:
        centre_db: Directly solved ``S21`` at the centre, in dB.
        flank_levels_db: Directly solved ``S21`` at the two flanks, in dB.

    Returns:
        The depth below the lower flank and whether that clears the threshold.
    """
    lower = min(flank_levels_db)
    depth = lower - centre_db
    return {
        "depth_below_lower_flank_db": depth,
        "minimum_depth_db": NOTCH_MIN_DEPTH_DB,
        "is_a_notch": bool(depth >= NOTCH_MIN_DEPTH_DB),
    }


if RUN_PORTED_DRIVEN and not RUN_COMSOL:
    print(
        "RUN_PORTED_DRIVEN needs RUN_COMSOL = True: stage 2 uses the COMSOL "
        "client that the licensed branch starts."
    )
elif RUN_PORTED_DRIVEN and client is not None:
    # One ID for the whole stage-2 run, created here before either solve, so the
    # ported eigen record, the driven record, and the driven curve all carry it.
    # A reader can then tell whether a fresh driven result is paired with the
    # stable ported eigen row solved in the same run or with an older one.
    run_id = uuid.uuid4().hex[:12]
    ported_model = COMSOL.create_sheet(
        client,
        layout,
        name="QPDK Coupled Quarter-Wave Resonator ported eigenmodes",
        substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
        air_height_um=AIR_HEIGHT_UM,
    )
    try:
        ported_model.add_cpw_rf_study(
            cpw_gap_um=CPW_GAP_UM,
            frequency_ghz=PORTED_SHIFT_GHZ,
            mesh_size=2,
        )
        # Ported eigen keeps bma1 and bma2: the ports need their boundary mode
        # fields, and only the frequency step is replaced.
        configure_ported_eigen_study(ported_model)
        create_meander_edge_selection(ported_model)
        ported_elements = ported_model.pin_absolute_edge_mesh_sizes(
            edge_selection=MEANDER_EDGE_SELECTION,
            global_hmax_um=GLOBAL_HMAX_UM,
            global_hmin_um=GLOBAL_HMIN_UM,
            edge_hmax_um=PORTED_EDGE_HMAX_UM,
            edge_hmin_um=PORTED_EDGE_HMIN_UM,
        )
        enforce_element_budget("ported eigen", ported_elements)
        ported_model.java.study("std1").run()
        for problem in ported_model.problems():
            print(f"  ported eigen model reports: {problem}")
        ported_modes = eigenfrequencies(ported_model)
        ported_selected, ported_field_source, ported_ratio = select_ported_mode(
            ported_model, ported_modes
        )
        # Every run writes its own tagged copy; the stable name is published only
        # by a coupled run that reaches the end of its driven window.
        ported_field_path = MODEL_DIR / PORTED_FIELD_TXT
        tagged_field_path = ported_field_record_path(
            PORTED_EDGE_HMAX_UM, PORTED_EDGE_HMIN_UM
        )
        tagged_field_path.write_bytes(ported_field_source.read_bytes())
        # Identified from the tagged copy, since the stable name still holds the
        # last published row's field.
        ported_identification = identify_ported_mode(
            tagged_field_path, ported_selected, ported_ratio
        )
        ported_record = {
            "run_id": run_id,
            "element_count": ported_elements,
            "edge_hmax_um": PORTED_EDGE_HMAX_UM,
            "edge_hmin_um": PORTED_EDGE_HMIN_UM,
            "global_hmax_um": GLOBAL_HMAX_UM,
            "global_hmin_um": GLOBAL_HMIN_UM,
            "shift_ghz": PORTED_SHIFT_GHZ,
            "neigs": PORTED_NEIGS,
            "eigwhich": PORTED_EIGWHICH,
            "modes_hz": [
                {"real": mode.real, "imag": mode.imag} for mode in ported_modes
            ],
            "selected_mode_hz": {
                "real": ported_selected.real,
                "imag": ported_selected.imag,
                "field_file": ported_field_path.name,
                # Digest of the exact field bytes both the tagged and the stable
                # copy hold, so a reader can tell an old record from a newer field.
                "field_sha256": field_sha256(ported_field_source),
                "meander_to_feed_p95": ported_ratio,
            },
            "selected_mode_identified": ported_identification[
                "selected_mode_identified"
            ],
            "selected_mode_identified_reason": ported_identification["reason"],
            # Names the device and the RF setup this row was solved for, so a run
            # at another edge size can tell whether it belongs beside this record
            # in the mesh series.
            "layout_signature": ported_layout_signature(layout),
        }
        # The tagged copy overrides field_file with its own field, so a later run
        # cannot leave it describing a field that has since been overwritten.
        write_json_atomically(
            ported_eigen_record_path(PORTED_EDGE_HMAX_UM, PORTED_EDGE_HMIN_UM),
            ported_record
            | {
                "selected_mode_hz": ported_record["selected_mode_hz"]
                | {"field_file": tagged_field_path.name}
            },
        )
        ported_series = update_ported_mesh_series(
            MODEL_DIR, ported_record["layout_signature"]
        )
        print(
            f"Ported eigen: {ported_elements} elements, selected "
            f"{ported_selected.real / 1e9:.9f} GHz, meander/feed p95 "
            f"{ported_ratio:.3f}, identified "
            f"{ported_identification['selected_mode_identified']}"
        )
        print(
            f"Ported mesh series: {len(ported_series['rows'])} row(s) rebuilt into "
            f"{PORTED_MESH_SERIES_JSON} from the mesh-tagged records in {MODEL_DIR}; "
            "rerun this stage at another PORTED_EDGE_HMAX_UM/PORTED_EDGE_HMIN_UM to "
            "add a row, with RUN_PORTED_EIGEN_ONLY = True to pay for the eigen solve "
            "only"
        )
        for reason in ported_series["rejected"]:
            print(f"  left out of the series: {reason}")
    finally:
        client.remove(ported_model)

    if RUN_PORTED_EIGEN_ONLY:
        print(
            "RUN_PORTED_EIGEN_ONLY is set, so this run stops after the eigen solve. "
            f"The stable {PORTED_EIGEN_JSON}, the stable field {PORTED_FIELD_TXT}, "
            "the saved driven model, and the driven curve and verification record "
            "are all left as they were, so the driven plot on this page still "
            "belongs to the mesh it was solved on. Clear RUN_PORTED_EIGEN_ONLY to "
            "solve the driven window again."
        )
    else:
        center_ghz = ported_selected.real / 1e9
        half_span_ghz = sweep_half_span_ghz(ported_selected.imag)
        low_ghz, high_ghz = center_ghz - half_span_ghz, center_ghz + half_span_ghz
        # Ten requested rows per loading width over the span the range actually
        # sweeps, so a 1.07 MHz notch inside the +/-5 MHz cap asks for ~95 rows.
        awe_points = sweep_requested_points(ported_selected.imag)

        model = COMSOL.create_sheet(
            client,
            layout,
            name="QPDK Coupled Quarter-Wave Resonator driven",
            substrate_thickness_um=SUBSTRATE_THICKNESS_UM,
            air_height_um=AIR_HEIGHT_UM,
        )
        try:
            model.add_cpw_rf_study(
                cpw_gap_um=CPW_GAP_UM,
                frequency_ghz=PORTED_SHIFT_GHZ,
                mesh_size=2,
            )
            create_meander_edge_selection(model)
            driven_elements = model.pin_absolute_edge_mesh_sizes(
                edge_selection=MEANDER_EDGE_SELECTION,
                global_hmax_um=GLOBAL_HMAX_UM,
                global_hmin_um=GLOBAL_HMIN_UM,
                edge_hmax_um=PORTED_EDGE_HMAX_UM,
                edge_hmin_um=PORTED_EDGE_HMIN_UM,
            )
            enforce_element_budget("driven sweep", driven_elements)
            # The stage-2 run ID names the curve and the record, so a reader can
            # tell a fresh curve from a stale verification record, and can pair
            # the driven result with the stable ported eigen row of the same run.
            direct_path = MODEL_DIR / f"{DIRECT_PREFIX}-{run_id}.json"
            awe_path = MODEL_DIR / f"{AWE_CURVE_PREFIX}-{run_id}.csv"
            record: dict[str, Any] = {
                "run_id": run_id,
                # The driven mesh and device this window was solved on, so the
                # reading side can require them to match the stable ported eigen
                # row before it reports a verdict or plots the curve.
                "edge_hmax_um": PORTED_EDGE_HMAX_UM,
                "edge_hmin_um": PORTED_EDGE_HMIN_UM,
                "element_count": driven_elements,
                "layout_signature": ported_layout_signature(layout),
                "window": {
                    "centre_ghz": center_ghz,
                    "low_ghz": low_ghz,
                    "high_ghz": high_ghz,
                    "half_span_ghz": half_span_ghz,
                    "fwhm_ghz": sweep_fwhm_ghz(ported_selected.imag),
                },
                "awe_points_requested": awe_points,
                "awe_points_per_fwhm": AWE_POINTS_PER_FWHM,
                "awe_agreement_db": AWE_AGREEMENT_DB,
                "notch_min_depth_db": NOTCH_MIN_DEPTH_DB,
                "power_sum_band": [POWER_SUM_MIN, POWER_SUM_MAX],
                # Stored as a bare file name, so a saved record carries no absolute path;
                # the reader resolves it under RESULTS_DIR or next to the record.
                "awe_curve_csv": awe_path.name,
                "awe_curve_rows_are_reconstructed": True,
                "direct_points": [],
                "verified": False,
                "verified_reason": "run started; nothing solved yet",
            }

            def checkpoint() -> None:
                """Write the record with whatever has been solved so far."""
                write_json_atomically(direct_path, record)

            checkpoint()
            # The flanks are the requested grid's own endpoints, at the same formatted
            # precision the AWE curve is built from, so a flank sits on a curve row
            # rather than a fraction of a hertz outside it.
            _, driven_grid_ghz = requested_frequency_grid(low_ghz, high_ghz, awe_points)
            # Both flanks and the centre, solved directly. Each point is solved and
            # checkpointed on its own, so an interrupted run keeps every point already
            # solved rather than losing the batch, and stays marked unverified until the
            # whole check passes.
            for label, frequency in (
                ("low flank", float(driven_grid_ghz[0])),
                ("centre", center_ghz),
                ("high flank", float(driven_grid_ghz[-1])),
            ):
                record["direct_points"].append(
                    solve_direct_point(model, label, frequency)
                )
                record["verified_reason"] = (
                    f"direct point {label!r} solved; curve and checks pending"
                )
                checkpoint()
            direct_points = list(record["direct_points"])

            flank_levels_db = [direct_points[0]["s21_db"], direct_points[2]["s21_db"]]
            direct_notch_at_centre = notch_verdict(
                direct_points[1]["s21_db"], flank_levels_db
            )

            curve = solve_awe_curve(model, low_ghz, high_ghz, awe_points, awe_path)
            minimum_index = int(np.argmin(curve["s21_db"]))
            awe_minimum_ghz = float(curve["frequencies_ghz"][minimum_index])
            record |= {
                "awe_curve_rows": curve["solved_rows"],
                "awe_curve_requested_points": curve["requested_points"],
                "awe_curve_requested_grid_low_ghz": curve["requested_grid_low_ghz"],
                "awe_curve_requested_grid_high_ghz": curve["requested_grid_high_ghz"],
                "awe_curve_requested_grid_step_ghz": curve["requested_grid_step_ghz"],
                "awe_curve_physical_window_ghz": curve["physical_window_ghz"],
                "awe_curve_refinement_rows": curve["refinement_rows"],
                "awe_curve_worst_power_sum": curve["worst_power_sum"],
                "awe_curve_worst_power_deficit": curve["worst_power_deficit"],
                "awe_curve_power_sum_min": curve["power_sum_min"],
                "awe_curve_power_sum_max": curve["power_sum_max"],
                "awe_curve_power_within_band": curve["power_within_band"],
                "awe_curve_power_balance_caveat": curve["power_balance_caveat"],
                "awe_minimum_ghz": awe_minimum_ghz,
            }
            record["verified_reason"] = "curve solved; checks pending"
            checkpoint()

            # The fit can put its minimum between the direct points, so that frequency
            # gets its own direct solve before any depth is believed.
            direct_minimum = solve_direct_point(model, "AWE minimum", awe_minimum_ghz)
            record["direct_points"] = [*direct_points, direct_minimum]
            checkpoint()

            # The notch is required at the directly solved AWE minimum, relative to both
            # flanks. The centre is kept as a comparison point: a loaded mode can sit off
            # the ported eigenfrequency, so the true minimum need not fall on the centre.
            direct_notch_at_minimum = notch_verdict(
                direct_minimum["s21_db"], flank_levels_db
            )

            # A direct frequency outside the curve cannot be compared, so it is a failure
            # rather than a skipped comparison. Every direct frequency is compared, not
            # only the minimum and the edges, so a flat curve that missed the minimum
            # fails too.
            comparisons = []
            for point in record["direct_points"]:
                curve_db = curve_value_db(curve, point["frequency_ghz"])
                # Distance to the nearest curve row, so an endpoint comparison says how
                # far the direct point sat from the row it was read at.
                nearest_gap_ghz = (
                    float(
                        np.min(
                            np.abs(curve["frequencies_ghz"] - point["frequency_ghz"])
                        )
                    )
                    if curve["frequencies_ghz"].size
                    else None
                )
                comparisons.append({
                    "label": point["label"],
                    "frequency_ghz": point["frequency_ghz"],
                    "direct_db": point["s21_db"],
                    "awe_curve_db": curve_db,
                    "curve_gap_ghz": nearest_gap_ghz,
                    "difference_db": (
                        None if curve_db is None else curve_db - point["s21_db"]
                    ),
                })
            outside_curve = [
                item["label"] for item in comparisons if item["difference_db"] is None
            ]
            differences = [
                abs(item["difference_db"])
                for item in comparisons
                if item["difference_db"] is not None
            ]
            curve_agrees = bool(
                not outside_curve
                and len(differences) == len(comparisons)
                and max(differences) <= AWE_AGREEMENT_DB
            )
            # Only the direct solves carry the passivity verdict. The curve's fitted rows
            # can sit a little outside the band, and that surplus is a property of the
            # rational fit rather than power leaving the model, so it stays a separate
            # diagnostic and never flips the verdict on its own.
            power_ok = bool(
                all(point["power_within_band"] for point in record["direct_points"])
            )
            verified = bool(
                curve_agrees and direct_notch_at_minimum["is_a_notch"] and power_ok
            )

            # The worst power deviation over the direct solves is the largest magnitude
            # from unity, so a slight surplus (a negative deficit) is not mistaken for the
            # best point.
            direct_deficits = [
                point["power_deficit"] for point in record["direct_points"]
            ]
            worst_deficit = max(direct_deficits, key=abs)

            record |= {
                "comparisons": comparisons,
                "direct_points_outside_curve": outside_curve,
                "max_absolute_difference_db": max(differences) if differences else None,
                "curve_agrees": curve_agrees,
                "power_within_band": power_ok,
                "worst_power_deficit": worst_deficit,
                "direct_notch_at_centre": direct_notch_at_centre,
                "direct_notch_at_minimum": direct_notch_at_minimum,
                "verified": verified,
                "verified_reason": (
                    "the direct solve at the AWE minimum is a notch below both flanks, "
                    "the AWE curve reproduces every direct level at all four directly "
                    "solved frequencies within tolerance, and the directly solved "
                    "two-port power balance is near unity"
                    if verified
                    else "run did not pass every check"
                ),
                "convergence_claim": {
                    "claimed": False,
                    "reason": (
                        "one mesh, one enclosure, one centre frequency; no convergence "
                        "verdict is claimed for any frequency, depth or width"
                    ),
                },
            }
            # The last checkpoint goes before model.save, so a save that does not finish
            # cannot lose the verification record.
            checkpoint()

            for point in record["direct_points"]:
                print(
                    f"direct {point['label']}: {point['frequency_ghz']:.9f} GHz, "
                    f"S21 {point['s21_db']:+.3f} dB, power sum {point['power_sum']:.6f} "
                    f"(unit-power deviation {point['power_deficit']:+.2e}, "
                    f"{'in band' if point['power_within_band'] else 'OUT OF BAND'})"
                )
            print(
                f"AWE curve: {curve['solved_rows']} rows of "
                f"{curve['requested_points']} requested, minimum "
                f"{curve['s21_db'][minimum_index]:+.3f} dB at {awe_minimum_ghz:.9f} GHz "
                "(reconstructed)"
            )
            print(
                f"Direct power balance: worst deviation from unity over the directly "
                f"solved points {worst_deficit:+.2e}, band {POWER_SUM_MIN:g} to "
                f"{POWER_SUM_MAX:g}"
            )
            print(
                f"AWE curve power balance (diagnostic, not a passivity verdict): rows "
                f"{curve['power_sum_min']:.6f} to {curve['power_sum_max']:.6f}, worst "
                f"unit-power deviation {curve['worst_power_deficit']:+.2e}"
                + (
                    ""
                    if curve["power_within_band"]
                    else "; reconstructed rows are not fully passive, which is a fitted "
                    "surplus and not physical loss"
                )
            )
            print(
                f"Direct notch at the AWE minimum: depth "
                f"{direct_notch_at_minimum['depth_below_lower_flank_db']:+.3f} dB against "
                f"the {NOTCH_MIN_DEPTH_DB:g} dB threshold"
            )
            print(
                f"Direct centre (comparison point, not required to be the notch): depth "
                f"{direct_notch_at_centre['depth_below_lower_flank_db']:+.3f} dB"
            )
            print(
                f"Curve against direct solves: largest difference "
                f"{max(differences) if differences else float('nan'):.3f} dB against the "
                f"{AWE_AGREEMENT_DB:g} dB tolerance"
                + (
                    f"; {len(outside_curve)} direct point(s) outside the curve: "
                    + ", ".join(outside_curve)
                    if outside_curve
                    else ""
                )
            )
            if verified:
                print(
                    "Verified: the direct solve at the AWE minimum is a notch below both "
                    "flanks, the AWE curve reproduces the direct levels at all four "
                    "directly solved frequencies within tolerance, and the directly solved "
                    "two-port power balance is near unity. No stage failed or was skipped, "
                    "but only those four frequencies are independent solves; the rest of "
                    "the curve is reconstructed."
                )
            else:
                reasons = []
                if outside_curve:
                    reasons.append(
                        "direct points outside the curve ("
                        + ", ".join(outside_curve)
                        + ")"
                    )
                elif not curve_agrees:
                    reasons.append("the curve does not reproduce the direct levels")
                if not direct_notch_at_minimum["is_a_notch"]:
                    reasons.append("no direct notch at the AWE minimum")
                if not power_ok:
                    reasons.append(
                        "the directly solved two-port power balance is outside the band, so "
                        "power leaves through a channel this record does not see"
                    )
                print(
                    "UNVERIFIED: "
                    + "; ".join(reasons)
                    + ". No resonance is inferred from the reconstructed curve; see the "
                    "direct points and comparisons in "
                    f"{direct_path.name}."
                )

            model.save(MODEL_DIR / "comsol_cpw_resonator_driven.mph")
            # Last, so a driven solve that fails leaves the old stable pair.
            publish_ported_stable(ported_record, ported_field_source)
        finally:
            client.remove(model)
elif not RUN_PORTED_DRIVEN:
    print(
        "The stage-2 solve is off by default. Set RUN_PORTED_DRIVEN = True with "
        "RUN_COMSOL = True to solve the ported eigenmode and the driven window; "
        "it runs on its own, and reads the port-free series only as context when "
        "RUN_PORT_FREE_SERIES is also True. Every run also writes a mesh-tagged "
        "copy of its ported eigen record and rebuilds the ported mesh series, so "
        "rerunning it at tighter PORTED_EDGE_HMAX_UM / PORTED_EDGE_HMIN_UM pairs "
        "is what produces the chart in 'Ported eigen mesh refinement'. Set "
        "RUN_PORTED_EIGEN_ONLY = True alongside it to add a mesh-refinement row "
        "without repeating the driven sweep; on its own that switch does nothing."
    )

# %% [markdown]
# ## Driven curve and its direct checks (stage 2 output)
#
# The curve as a line with the directly solved frequencies as markers: **the line is AWE
# interpolation**, and the markers are the only independent solves. The cell refuses a curve or eigen
# row that does not match this notebook's layout signature, mesh, and element count, and recomputes the
# verdict from the loaded curve and saved points rather than trusting the record's own flag. The notch
# is therefore verified at the directly solved frequencies only: the depth comes from the coupling plus
# whatever loss the model carries, and PEC adds no conductor or dielectric loss.

# %%
direct_files = (
    sorted(
        RESULTS_DIR.glob(f"{DIRECT_PREFIX}-*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if RESULTS_DIR is not None
    else []
)

if not direct_files:
    explain_missing_results(f"{DIRECT_PREFIX}-<run-id>.json")
else:
    direct_file = direct_files[-1]
    direct = json.loads(direct_file.read_text())
    curve_name = direct.get("awe_curve_csv")
    # A record is only readable once every key this cell reads is present. The
    # verdict keys the write side stores are deliberately not required: the
    # comparisons and the verdict are recomputed below, so a record whose flags
    # are stale or absent still gets a verdict drawn from the data it carries. A
    # checkpoint written mid-run is caught by the missing curve file, or by the
    # direct-point checks below.
    required_keys = (
        "run_id",
        "window",
        "awe_points_requested",
        "direct_points",
    )
    missing_keys = [key for key in required_keys if key not in direct]
    ported_signature = ported_layout_signature(layout)
    pairing_problems = driven_pairing_problems(direct)
    if direct.get("layout_signature") != ported_signature:
        print(
            f"{direct_file.name} was solved for a different layout or RF setup "
            f"(signature {direct.get('layout_signature', 'not recorded')!r} against "
            f"this run's {ported_signature!r}), so its driven result does not "
            "describe this device. Nothing is verified or plotted from it."
        )
    elif pairing_problems:
        print(
            "UNVERIFIED: the driven record is not paired with the stable ported "
            "eigen row it was solved with ("
            + "; ".join(pairing_problems)
            + "). No resonance is inferred and the curve is not plotted."
        )
    elif not isinstance(curve_name, str):
        print(
            f"{direct_file.name} holds no curve file name yet, so the driven "
            "window has not produced a curve."
        )
    elif missing_keys:
        print(
            f"{direct_file.name} is a partial record, still missing "
            + ", ".join(missing_keys)
            + "; the verification is pending and no curve is plotted."
        )
    else:
        awe_file = resolve_record_path(direct_file.parent, curve_name)
        if not awe_file.exists():
            print(
                f"The record {direct_file.name} names {curve_name}, but that curve "
                f"is not in {awe_file.parent}. The sweep is missing or incomplete, "
                "so there is nothing to plot yet."
            )
        elif direct["run_id"] not in awe_file.name:
            print(
                "The curve file does not carry this record's run ID, so the two are "
                "from different runs. Refusing to plot a curve against a verification "
                "record it does not belong to."
            )
        else:
            # Bind the arrays only once every column is present. A CSV with, say,
            # frequency_ghz and s21_db but no s11_db would otherwise leave the
            # frequency array bound while a later column is undefined, and the
            # plot gate below would open on an incomplete read.
            curve_arrays = None
            try:
                curve_rows = np.atleast_1d(
                    np.genfromtxt(awe_file, delimiter=",", names=True)
                )
                curve_arrays = (
                    curve_rows["frequency_ghz"],
                    curve_rows["s21_db"],
                    curve_rows["s11_db"],
                )
            except Exception as error:
                print(
                    f"The curve {awe_file.name} could not be read ({error!r}), so it "
                    "is treated as incomplete and nothing is plotted."
                )
            if curve_arrays is not None:
                curve_ghz, curve_s21_db, curve_s11_db = curve_arrays
                # Interpolation, and so every comparison below, needs a finite
                # frequency column that strictly increases. A curve swapped in
                # after the solve must not inherit the record's verdict, so this
                # is checked here rather than taken from the record.
                finite_curve = bool(
                    curve_ghz.size >= 2
                    and np.all(np.isfinite(curve_ghz))
                    and np.all(np.isfinite(curve_s21_db))
                    and np.all(np.isfinite(curve_s11_db))
                )
                increasing = bool(finite_curve and np.all(np.diff(curve_ghz) > 0.0))
                if not finite_curve:
                    print(
                        f"UNVERIFIED: the curve {awe_file.name} holds "
                        f"{curve_ghz.size} row(s), and a comparison needs at least "
                        "two rows with a finite frequency and S-parameter entry on "
                        "each. It cannot be interpolated against the direct solves, "
                        "so no resonance is inferred and nothing is plotted."
                    )
                elif not increasing:
                    print(
                        f"UNVERIFIED: the frequency column of {awe_file.name} is not "
                        "strictly increasing, so the curve cannot be interpolated "
                        "against the direct solves. No resonance is inferred and "
                        "nothing is plotted."
                    )
                else:
                    curve = {"frequencies_ghz": curve_ghz, "s21_db": curve_s21_db}
                    power_sum = 10 ** (curve_s21_db / 10) + 10 ** (curve_s11_db / 10)
                    # The curve's own power sum is a diagnostic, not a passivity
                    # verdict: the rational fit can put a row slightly outside the
                    # band, and that fitted surplus is not power leaving the model.
                    # The curve is still drawn, and the verdict rests on the direct
                    # solves below.
                    banded = (power_sum.min() >= POWER_SUM_MIN) and (
                        power_sum.max() <= POWER_SUM_MAX
                    )
                    # Worst deviation is the largest magnitude from unity, so a
                    # slight fitted surplus is not reported as the best row.
                    curve_deficits = 1.0 - power_sum
                    worst_index = int(np.argmax(np.abs(curve_deficits)))
                    centre_ghz = direct["window"]["centre_ghz"]
                    offset_khz = (curve_ghz - centre_ghz) * 1e6

                    # Exactly one direct solve per role, and finite numbers on
                    # each. Anything less is a record that cannot support a
                    # verdict, whatever flags it carries.
                    problems: list[str] = []
                    required_labels = (
                        "low flank",
                        "centre",
                        "high flank",
                        "AWE minimum",
                    )
                    stored_points = direct["direct_points"]
                    stored_points = (
                        stored_points if isinstance(stored_points, list) else []
                    )
                    direct_values: dict[str, dict[str, float]] = {}
                    for label in required_labels:
                        matches = [
                            point
                            for point in stored_points
                            if isinstance(point, dict) and point.get("label") == label
                        ]
                        if len(matches) != 1:
                            problems.append(
                                f"{len(matches)} direct point(s) are labelled "
                                f"{label!r}, and exactly one is required"
                            )
                            continue
                        values: dict[str, float] = {}
                        for key in ("frequency_ghz", "s21_db", "s11_db"):
                            value = matches[0].get(key)
                            numeric = not isinstance(value, bool) and isinstance(
                                value, (int, float)
                            )
                            if not numeric or not np.isfinite(value):
                                problems.append(
                                    f"direct point {label!r} has {key} = {value!r}, "
                                    "which is not a finite number"
                                )
                            else:
                                values[key] = float(value)
                        if len(values) == 3:
                            direct_values[label] = values

                    # Every comparison is recomputed against the curve as loaded,
                    # at the frequency each direct point was solved at. A direct
                    # frequency outside the curve fails rather than being skipped.
                    comparisons = []
                    for label in required_labels:
                        if label not in direct_values:
                            continue
                        values = direct_values[label]
                        curve_db = curve_value_db(curve, values["frequency_ghz"])
                        comparisons.append({
                            "label": label,
                            "frequency_ghz": values["frequency_ghz"],
                            "direct_db": values["s21_db"],
                            "awe_curve_db": curve_db,
                            "difference_db": (
                                None
                                if curve_db is None
                                else curve_db - values["s21_db"]
                            ),
                        })
                    outside_curve = [
                        item["label"]
                        for item in comparisons
                        if item["difference_db"] is None
                    ]
                    differences = [
                        abs(item["difference_db"])
                        for item in comparisons
                        if item["difference_db"] is not None
                    ]
                    if outside_curve:
                        problems.append(
                            "direct point(s) outside the current curve: "
                            + ", ".join(outside_curve)
                        )
                    elif len(differences) == len(required_labels) and (
                        max(differences) > AWE_AGREEMENT_DB
                    ):
                        problems.append(
                            "the current curve does not reproduce the direct levels: "
                            f"worst difference {max(differences):.3f} dB against the "
                            f"{AWE_AGREEMENT_DB:g} dB tolerance"
                        )

                    # The notch is required only at the directly solved AWE
                    # minimum, relative to both direct flanks.
                    direct_notch = None
                    # The label is not evidence: recompute where this curve
                    # bottoms out, then check the saved point is still there.
                    loaded_minimum_ghz = float(curve_ghz[int(np.argmin(curve_s21_db))])
                    minimum_offset_ghz = None
                    if len(direct_values) == len(required_labels):
                        direct_notch = notch_verdict(
                            direct_values["AWE minimum"]["s21_db"],
                            [
                                direct_values["low flank"]["s21_db"],
                                direct_values["high flank"]["s21_db"],
                            ],
                        )
                        if not direct_notch["is_a_notch"]:
                            problems.append(
                                "no direct notch at the AWE minimum: depth "
                                f"{direct_notch['depth_below_lower_flank_db']:+.3f} dB "
                                f"against the {NOTCH_MIN_DEPTH_DB:g} dB threshold"
                            )
                        low_flank_ghz = direct_values["low flank"]["frequency_ghz"]
                        high_flank_ghz = direct_values["high flank"]["frequency_ghz"]
                        minimum_offset_ghz = abs(
                            direct_values["AWE minimum"]["frequency_ghz"]
                            - loaded_minimum_ghz
                        )
                        if minimum_offset_ghz > DIRECT_MINIMUM_FREQUENCY_TOLERANCE_GHZ:
                            problems.append(
                                "the direct point labelled 'AWE minimum' is at "
                                f"{direct_values['AWE minimum']['frequency_ghz']:.9f} "
                                f"GHz, but the current curve's minimum is at "
                                f"{loaded_minimum_ghz:.9f} GHz: {minimum_offset_ghz:.3e} "
                                "GHz apart against the "
                                f"{DIRECT_MINIMUM_FREQUENCY_TOLERANCE_GHZ:g} GHz "
                                "tolerance, so that point was not solved at this "
                                "curve's minimum"
                            )
                        if not low_flank_ghz < loaded_minimum_ghz < high_flank_ghz:
                            problems.append(
                                "the current curve's minimum at "
                                f"{loaded_minimum_ghz:.9f} GHz does not sit between "
                                "the directly solved flanks at "
                                f"{low_flank_ghz:.9f} and {high_flank_ghz:.9f} GHz"
                            )
                        if not (
                            low_flank_ghz
                            < direct_values["centre"]["frequency_ghz"]
                            < high_flank_ghz
                        ):
                            problems.append(
                                "the directly solved centre at "
                                f"{direct_values['centre']['frequency_ghz']:.9f} GHz "
                                "does not sit between the directly solved flanks at "
                                f"{low_flank_ghz:.9f} and {high_flank_ghz:.9f} GHz"
                            )

                    # Passivity is recomputed from the saved S-parameters too, so
                    # a stored power flag cannot stand in for the numbers.
                    direct_power = {
                        label: power_balance(
                            10 ** (values["s21_db"] / 10)
                            + 10 ** (values["s11_db"] / 10)
                        )
                        for label, values in direct_values.items()
                    }
                    if any(
                        not entry["power_within_band"]
                        for entry in direct_power.values()
                    ):
                        problems.append(
                            "the directly solved two-port power balance is outside "
                            f"the {POWER_SUM_MIN:g} to {POWER_SUM_MAX:g} band, so "
                            "power leaves through a channel this record does not see"
                        )

                    print(f"Run ID: {direct['run_id']}")
                    print(
                        f"AWE curve: {curve_ghz.size} reconstructed rows of "
                        f"{direct['awe_points_requested']} requested, "
                        f"{curve_ghz.min():.6f} to {curve_ghz.max():.6f} GHz"
                    )
                    print(
                        f"Curve power balance (diagnostic, reconstructed rows): "
                        f"|S11|² + |S21|² from {power_sum.min():.6f} to "
                        f"{power_sum.max():.6f}, worst deviation "
                        f"{curve_deficits[worst_index]:+.2e}, band {POWER_SUM_MIN:g} to "
                        f"{POWER_SUM_MAX:g}, "
                        f"{'in band' if banded else 'OUT OF BAND'}"
                    )
                    if not banded:
                        print(
                            "Caveat: reconstructed AWE rows are not fully passive; the "
                            f"fitted surplus above the {POWER_SUM_MAX:g} band edge is a "
                            "property of the rational fit, not physical loss or gain, "
                            "and it does not by itself unverify the resonance."
                        )
                    if direct_power:
                        worst_power = max(
                            direct_power.values(),
                            key=lambda entry: abs(entry["power_deficit"]),
                        )
                        print(
                            f"Direct points (recomputed from the saved S-parameters): "
                            f"{len(direct_values)} of {len(required_labels)} usable, "
                            "power balance in band: "
                            f"{all(entry['power_within_band'] for entry in direct_power.values())}, "
                            "worst unit-power deviation "
                            f"{worst_power['power_deficit']:+.2e}"
                        )
                    for item in comparisons:
                        curve_db = item["awe_curve_db"]
                        curve_text = (
                            "outside curve"
                            if curve_db is None
                            else f"{curve_db:+.3f} dB"
                        )
                        difference = item["difference_db"]
                        difference_text = (
                            "n/a" if difference is None else f"{difference:+.3f} dB"
                        )
                        print(
                            f"  {item['label']}: direct "
                            f"{item['direct_db']:+.3f} dB, "
                            f"AWE curve {curve_text}, difference {difference_text}"
                        )
                    if differences:
                        print(
                            f"Curve against direct solves (recomputed at the direct "
                            f"frequencies): largest difference "
                            f"{max(differences):.3f} dB against the "
                            f"{AWE_AGREEMENT_DB:g} dB tolerance"
                        )

                    minimum_index = int(np.argmin(curve_s21_db))
                    print(
                        f"AWE curve minimum (reconstructed, not a direct solve): "
                        f"{curve_s21_db[minimum_index]:+.3f} dB at "
                        f"{curve_ghz[minimum_index]:.9f} GHz"
                    )
                    if minimum_offset_ghz is not None:
                        print(
                            "Minimum check: the saved direct 'AWE minimum' at "
                            f"{direct_values['AWE minimum']['frequency_ghz']:.9f} GHz "
                            f"against the minimum of the curve as loaded at "
                            f"{loaded_minimum_ghz:.9f} GHz, "
                            f"{minimum_offset_ghz:.3e} GHz apart against the "
                            f"{DIRECT_MINIMUM_FREQUENCY_TOLERANCE_GHZ:g} GHz tolerance"
                        )
                    if direct_notch is not None:
                        print(
                            "Direct notch at the AWE minimum: depth "
                            f"{direct_notch['depth_below_lower_flank_db']:+.3f} dB "
                            f"against the {NOTCH_MIN_DEPTH_DB:g} dB threshold"
                        )
                    if problems:
                        print(
                            "UNVERIFIED: "
                            + "; ".join(problems)
                            + ". No resonance is inferred, and no quality factor is "
                            "read from the reconstructed curve."
                        )
                    else:
                        print(
                            "Verified: the direct solve at the AWE minimum is a notch "
                            "below both flanks, that point still sits at the minimum "
                            "of the curve as loaded, the current curve reproduces the "
                            "direct levels at all four directly solved frequencies "
                            "within tolerance, and the directly solved two-port power "
                            "balance is near unity. Only those four frequencies are "
                            "independent solves."
                        )

                    # The figure is drawn from the loaded curve whatever the
                    # verdict, and the verdict is on it, because the figure is
                    # what a person looks at to see why a run failed.
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(
                        offset_khz,
                        curve_s21_db,
                        color="C0",
                        label=r"AWE interpolation of $|S_{21}|$ (reconstructed)",
                    )
                    direct_markers = [
                        (
                            label,
                            (direct_values[label]["frequency_ghz"] - centre_ghz) * 1e6,
                            direct_values[label]["s21_db"],
                        )
                        for label in required_labels
                        if label in direct_values
                    ]
                    ax.plot(
                        [offset for _, offset, _ in direct_markers],
                        [level for _, _, level in direct_markers],
                        marker="x",
                        markersize=8,
                        linestyle="none",
                        color="crimson",
                        label="direct solve",
                    )
                    for label, offset, level in direct_markers:
                        ax.annotate(
                            label,
                            (offset, level),
                            textcoords="offset points",
                            xytext=(6, 6),
                            fontsize=8,
                            color="crimson",
                        )
                    ax.set_xlabel("Frequency offset from the ported mode (kHz)")
                    ax.set_ylabel(r"$|S_{21}|$ (dB)")
                    ax.set_title(
                        f"Driven window around {centre_ghz:.6f} GHz: AWE curve and "
                        "direct solves ("
                        + ("UNVERIFIED" if problems else "VERIFIED")
                        + ")"
                    )
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

# %% [markdown]
# ## Ported eigenfrequency (stage 2 output)
#
# The ported eigenvalues, under their own heading. The numeric TEM ports are matched terminations, so
# these modes are **loaded** and their imaginary part carries the port loading on top of any numerical
# error; the printed ratio $f'/(2|f''|)$ is that loaded eigenvalue damping, not a $Q$ from a linewidth.

# %%
ported_file = result_file(PORTED_EIGEN_JSON)

if ported_file is None:
    print(
        f"No {PORTED_EIGEN_JSON} in RESULTS_DIR ({RESULTS_DIR}), so there is no "
        "ported eigen result to show yet. It is written by the licensed stage-2 "
        "branch."
    )
else:
    ported = json.loads(ported_file.read_text())
    ported_signature = ported_layout_signature(layout)
    if ported.get("layout_signature") != ported_signature:
        print(
            f"{PORTED_EIGEN_JSON} was solved for a different layout or RF setup "
            f"(signature {ported.get('layout_signature', 'not recorded')!r} "
            f"against this run's {ported_signature!r}), so it does not describe "
            "this device. It is refused and not shown."
        )
    else:
        selected_hz = ported.get("selected_mode_hz") or {}
        print(
            "Ported eigen solve: loaded modes, not comparable one to one with the "
            "port-free modes above."
        )
        print(f"  elements: {ported.get('element_count')}")
        print(
            f"  shift: {ported.get('shift_ghz')} GHz, "
            f"modes requested: {ported.get('neigs')}"
        )
        modes_hz = ported.get("modes_hz") or []
        reals_ghz = [
            mode["real"] / 1e9
            for mode in modes_hz
            if mode.get("real") is not None and np.isfinite(mode["real"])
        ]
        if reals_ghz:
            print(
                f"  {len(reals_ghz)} solved modes (GHz): "
                + ", ".join(f"{value:.6f}" for value in reals_ghz)
            )
        if selected_hz.get("real") is not None and np.isfinite(selected_hz["real"]):
            imag_hz = selected_hz.get("imag")
            damping = (
                f"{imag_hz:+.4f} Hz"
                if imag_hz is not None and np.isfinite(imag_hz)
                else "not recorded"
            )
            print(
                f"  selected ported mode: {selected_hz['real'] / 1e9:.6f} GHz, "
                f"imaginary {damping}"
            )
            if imag_hz:
                print(
                    "  its implied loss ratio f'/(2|f''|), port loading included: "
                    f"{abs(selected_hz['real'] / (2 * imag_hz)):.3g}"
                )
        else:
            print("  no selected ported mode is recorded in the file")

# %% [markdown]
# ## Ported eigen field map (stage 2 output)
#
# The same cut-plane field for the selected ported mode, labelled **PORTED**: matched terminations make
# this a loaded mode's field, not the port-free one, with the same eigenvector normalization as above.
# Nothing is drawn unless the record's SHA-256 digest matches the bytes on disk and the export header's
# real frequency matches the selected mode, so a stale field cannot be shown.

# %%
# PORTED_FIELD_TXT, PORTED_FIELD_FREQUENCY, PORTED_FIELD_FREQUENCY_RTOL and
# complex_frequency_ghz are defined in the setup imports above, because the
# licensed stage-2 branch validates the export against them before it writes the
# record this cell reads.


def draw_cropped_field_map(
    file: Path,
    title: str,
    *,
    view_um: tuple[float, float, float, float] = FIELD_VIEW_UM,
    stride: int = FIELD_DISPLAY_STRIDE,
    contour_levels: int = FIELD_CONTOUR_LEVELS,
) -> None:
    """Draw a cropped ``emw.normE`` map from a COMSOL cut-plane export.

    The export covers the whole prepared box, so the nodes are cropped to
    ``view_um`` and, for display only, drawn on a fixed-stride subset with
    percentiles of the cropped data as the colour limits, the same treatment the
    port-free map uses. The stride changes what is drawn, not what was solved.

    Args:
        file: A ``emw.normE`` export on the cut plane, columns x, y, z, E.
        title: Figure title.
        view_um: ``(xmin, xmax, ymin, ymax)`` crop window, in µm.
        stride: Drawing stride over the cropped nodes.
        contour_levels: Number of contour levels.

    Raises:
        ValueError: If the crop window holds no exported node.
    """
    field = np.loadtxt(file, comments="%")
    field_x, field_y, field_e = field[:, 0], field[:, 1], field[:, 3]
    inside = (
        (field_x >= view_um[0])
        & (field_x <= view_um[1])
        & (field_y >= view_um[2])
        & (field_y <= view_um[3])
    )
    view_x, view_y, view_e = field_x[inside], field_y[inside], field_e[inside]
    if view_e.size == 0:
        raise ValueError(f"No exported field nodes inside {view_um}")

    color_min, color_max = (float(value) for value in np.percentile(view_e, [1, 99]))
    draw_x, draw_y, draw_e = view_x, view_y, view_e
    if stride > 1 and view_e.size // stride >= 10:
        draw_x = view_x[::stride]
        draw_y = view_y[::stride]
        draw_e = view_e[::stride]
    print(
        f"Field nodes: {view_e.size} of {field_e.size} inside the view; "
        f"{draw_e.size} drawn at stride {stride}; "
        f"range {view_e.min():.3g} to {view_e.max():.3g} V/m, "
        f"1st to 99th percentile {color_min:.3g} to {color_max:.3g} V/m"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.tricontourf(
        draw_x,
        draw_y,
        draw_e,
        levels=np.geomspace(color_min, color_max, contour_levels),
        norm=LogNorm(vmin=color_min, vmax=color_max),
        cmap="inferno",
        extend="both",
    )
    ax.set_xlim(view_um[0], view_um[1])
    ax.set_ylim(view_um[2], view_um[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label=r"$|\mathbf{E}|$ (V/m)")
    plt.tight_layout()
    plt.show()


ported_field_file = result_file(PORTED_FIELD_TXT)

if ported_field_file is None:
    print(
        f"No {PORTED_FIELD_TXT} in RESULTS_DIR ({RESULTS_DIR}), so there is no "
        "ported field map to show."
    )
else:
    ported_record_file = result_file(PORTED_EIGEN_JSON)
    if ported_record_file is None:
        print(
            f"{PORTED_FIELD_TXT} is present but {PORTED_EIGEN_JSON} is not, so the "
            "field cannot be checked against the ported mode it belongs to and is "
            "not plotted."
        )
    else:
        ported_record = json.loads(ported_record_file.read_text())
        ported_signature = ported_layout_signature(layout)
        selected_hz = ported_record.get("selected_mode_hz") or {}
        selected_real_hz = selected_hz.get("real")
        if ported_record.get("layout_signature") != ported_signature:
            print(
                f"{PORTED_EIGEN_JSON} was solved for a different layout or RF "
                f"setup (signature "
                f"{ported_record.get('layout_signature', 'not recorded')!r} against "
                f"this run's {ported_signature!r}), so the field cannot be checked "
                "against this device's ported mode and is not plotted."
            )
        elif selected_real_hz is None or not np.isfinite(selected_real_hz):
            print(
                f"{PORTED_EIGEN_JSON} records no finite selected ported mode, so "
                "there is no mode to check the field against and it is not plotted."
            )
        elif not (
            isinstance(recorded_digest := selected_hz.get("field_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", recorded_digest)
        ):
            print(
                f"{PORTED_EIGEN_JSON} records no valid sha256 digest of the field it "
                f"was solved with (selected_mode_hz.field_sha256 is "
                f"{selected_hz.get('field_sha256')!r}), so it cannot be checked "
                f"against {ported_field_file.name} and the field is not plotted."
            )
        elif field_sha256(ported_field_file) != recorded_digest:
            print(
                f"{ported_field_file.name} does not match the sha256 digest "
                f"recorded in {PORTED_EIGEN_JSON}, so the record and the field are "
                "not from the same solve and the field is not plotted."
            )
        else:
            annotation = complex_frequency_ghz(ported_field_file)
            if annotation is None:
                raise ValueError(
                    f"{ported_field_file.name} carries no complex frequency "
                    "annotation, so it cannot be checked against the ported record "
                    "and will not be plotted"
                )
            field_real_ghz, field_imag_ghz = annotation
            selected_ghz = selected_real_hz / 1e9
            relative = abs(field_real_ghz - selected_ghz) / abs(selected_ghz)
            if relative > PORTED_FIELD_FREQUENCY_RTOL:
                raise ValueError(
                    f"{ported_field_file.name} is annotated at "
                    f"{field_real_ghz:.9f} GHz but {PORTED_EIGEN_JSON} selects "
                    f"{selected_ghz:.9f} GHz, a relative difference of "
                    f"{relative:.2e} against the {PORTED_FIELD_FREQUENCY_RTOL:g} "
                    "tolerance; the field is not the selected mode's field and is "
                    "not plotted"
                )

            print(f"Ported field export: {ported_field_file.name}")
            print(
                f"  annotated complex frequency: {field_real_ghz:.9f} "
                f"{field_imag_ghz:+.3e}j GHz"
            )
            print(
                f"  selected ported mode: {selected_ghz:.9f} GHz, relative "
                f"difference {relative:.2e} (tolerance "
                f"{PORTED_FIELD_FREQUENCY_RTOL:g})"
            )
            identified = ported_record.get("selected_mode_identified")
            if identified is not None:
                print(f"  identified as: {identified}")
            ported_p95 = selected_hz.get("meander_to_feed_p95")
            if ported_p95 is not None and np.isfinite(ported_p95):
                print(
                    f"  measured meander/feed |E| p95 for this loaded mode: "
                    f"{ported_p95:.3f}"
                )
            else:
                print("  no meander/feed p95 is recorded for this ported mode")
            print(
                "One cut plane from one ported solve is a consistency check on the "
                "field, not a convergence result; nothing here is called settled."
            )
            draw_cropped_field_map(
                ported_field_file,
                f"PORTED eigen field: $|\\mathbf{{E}}|$ at {field_real_ghz:g} GHz, "
                "z = 1 µm",
            )

# %% [markdown]
# ## Ported eigen mesh refinement (stage 2 output)
#
# The selected loaded frequency against the ported mesh element count, coarse to fine; only the meander
# edge sizes differ between rows, and the series is rebuilt from the mesh-tagged records the licensed
# study writes. The signed shifts grow then reverse, so the series does not demonstrate convergence and
# no limit is extrapolated. A frequency that moves between meshes is discretisation error, not the
# device changing: a meander mode sits at the sharp PEC edges, where the mesh sets the effective
# inductance and capacitance the discrete mode sees.

# %%
# Read side only: one row per meander edge mesh, coarse to fine, and nothing
# here solves or interpolates a row that is not in the file. The series file name
# is defined with the writer above, so the two cannot drift apart.
SERIES_SIZES = ("edge_hmax_um", "edge_hmin_um")
SERIES_FIELDS = ("element_count", "frequency_ghz", "q", "meander_to_feed_p95")

ported_series_file = result_file(PORTED_MESH_SERIES_JSON)
if ported_series_file is None:
    print(
        f"No {PORTED_MESH_SERIES_JSON} in RESULTS_DIR ({RESULTS_DIR}): no ported mesh "
        "series to read yet. This cell does not solve the tighter meshes; a licensed "
        "stage-2 run writes one row per PORTED_EDGE_HMAX_UM / PORTED_EDGE_HMIN_UM pair "
        "it was run at, and an externally produced series JSON can also be read here "
        "through QPDK_COMSOL_RESULTS_DIR."
    )
else:
    payload = json.loads(ported_series_file.read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else None
    rows = rows if isinstance(rows, list) else []
    # Only a series rebuilt for this layout and RF setup is charted, so rows from
    # another device cannot pass as a mesh-only change.
    series_signature = ported_layout_signature(layout)
    refused = None
    if not isinstance(payload, dict):
        refused = f"{PORTED_MESH_SERIES_JSON} does not hold a record object"
    elif payload.get("signature") != series_signature:
        refused = (
            f"{PORTED_MESH_SERIES_JSON} was built for a different layout or RF "
            f"setup: signature {payload.get('signature', 'not recorded')!r} against "
            f"this run's {series_signature!r}"
        )
    if refused is not None:
        problems = [refused]
    else:
        problems = [] if len(rows) >= 2 else [f"{len(rows)} row(s): a delta needs two"]
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                problems.append(f"row {index} is not an object")
                continue
            if row.get("selected_mode_identified") is not True:
                problems.append(f"row {index}: selected_mode_identified is not True")
            for key in SERIES_SIZES + SERIES_FIELDS:
                value = row.get(key)
                numeric = not isinstance(value, bool) and isinstance(
                    value, (int, float)
                )
                if not numeric or not np.isfinite(value) or value <= 0.0:
                    problems.append(
                        f"row {index}/{key}: {value!r} not finite and positive"
                    )
        if not problems:
            counts = np.array([row["element_count"] for row in rows], dtype=float)
            if np.any(np.diff(counts) <= 0.0):
                problems.append("element counts are not increasing coarse to fine")
            for key in SERIES_SIZES:
                sizes = np.array([row[key] for row in rows], dtype=float)
                if np.any(np.diff(sizes) >= 0.0):
                    problems.append(f"{key} is not strictly decreasing coarse to fine")

    if problems:
        print(
            f"{PORTED_MESH_SERIES_JSON} cannot be charted, and no row is corrected or "
            "filled in here:"
        )
        print("\n".join(f"  {problem}" for problem in problems))
    else:
        frequencies_ghz = np.array([row["frequency_ghz"] for row in rows], dtype=float)
        deltas_mhz = np.diff(frequencies_ghz) * 1e3
        print("Ported eigen mesh series, coarse to fine; the deltas are in MHz:")
        print(
            f"  {'meander edges (µm)':>19} {'elements':>9} {'f (GHz)':>15} {'Q':>10} "
            f"{'meander/feed p95':>17} {'|delta|':>7} {'signed':>8}"
        )
        steps = ["n/a"] + [f"{abs(v):>7.3f} {v:>+8.3f}" for v in deltas_mhz]
        for index, row in enumerate(rows):
            print(
                f"  {row['edge_hmax_um']:>9g}/{row['edge_hmin_um']:<9g} "
                f"{int(row['element_count']):>9,d} {row['frequency_ghz']:>15.6f} "
                f"{row['q']:>10.0f} {row['meander_to_feed_p95']:>17.3f} {steps[index]}"
            )

        # Only validated rows reach here, so a bad row was reported above instead of
        # being drawn as a break. Two deltas are what the delta panel needs.
        has_deltas = len(rows) > 2
        figure, axes = plt.subplots(
            1, 2 if has_deltas else 1, figsize=(11, 4), squeeze=False
        )
        panels = [(counts, frequencies_ghz, "o", "Selected loaded mode (GHz)")]
        if has_deltas:
            panels.append((counts[1:], deltas_mhz, "s", "Change from previous (MHz)"))
        for axis, (x, y, marker, label) in zip(axes[0], panels, strict=True):
            axis.plot(x, y, marker=marker)
            axis.set(xscale="log", xlabel="Mesh elements", ylabel=label)
            axis.grid(True, which="both", alpha=0.3)
        frequency_axis = axes[0][0]
        for row, count, frequency in zip(rows, counts, frequencies_ghz, strict=True):
            frequency_axis.annotate(
                f"{row['edge_hmax_um']:g}/{row['edge_hmin_um']:g} µm",
                (count, frequency),
            )
        frequency_axis.set_title("Ported eigenfrequency against mesh, not an asymptote")
        if has_deltas:
            axes[0][1].axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        plt.tight_layout()
        plt.show()

        print(
            "The first row has no delta above it. Q is the loaded eigenvalue's own damping "
            "ratio f'/(2|f''|), not a notch linewidth, and numerical error may fall with "
            "refinement without falling monotonically: no order is fitted here, and the "
            "driven notch depth and linewidth, from a driven sweep on its own mesh, stay "
            "open regardless."
        )

# %% [markdown]
# ## Summary
#
# A scripted path from layout through meshing to a port-free localised eigenmode, and then to a ported
# eigenmode whose driven notch is verified at directly solved frequencies. The ported series does not
# settle with refinement, so this is not a converged model and not a device prediction.
#
# ### Next steps
#
# - Refine the ported eigen mesh further, watching the shift rather than the element count.
# - Extend the driven check beyond the selected mesh, and fit a quality factor from the notch width.
# - Replace the outer PEC walls with scattering boundaries, and PEC with a surface-impedance condition.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
