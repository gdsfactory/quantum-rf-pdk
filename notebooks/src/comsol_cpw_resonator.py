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
# This notebook needs the `comsol` extra:
#
# ```bash
# uv add "qpdk[comsol]"
# # or, from a checkout of this repository:
# uv sync --extra comsol
# # or with pip:
# pip install "qpdk[comsol]"
# ```
#
# Installing the extra installs `MPh`, the Python client for COMSOL, and nothing
# else. **It does not install COMSOL and it does not grant a license.** Running
# the model build and the full-wave solve needs a local COMSOL installation, a
# license, and the RF Module. The metal here has etched holes, so the sheet model
# is imprinted with the Design Module's `ProjectToFaces`, which needs the Design
# Module CAD kernel as well. Google Colab has none of them, so the build cells
# cannot run there.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# This notebook solves a QPDK coupled quarter-wave resonator in COMSOL in two
# stages. **Stage 1** is a port-free eigenfrequency study of the sheet model:
# PEC on the metal, no numeric ports, a mesh whose element sizes are pinned to
# absolute values including a local size on the meander edges, and an
# `Eigenfrequency` step whose search is set per edge mesh. **Stage 2** is the
# ported study: the same sheet model with the two numeric TEM ports and their
# boundary mode analysis steps kept, a ported eigenfrequency search, and a
# driven $S_{21}$ window whose notch is checked against direct single-frequency
# solves. Both stages have been solved on a licensed machine, and the saved
# figures and numbers on this page are their exported output.
#
# ## What the saved output is
#
# It is a record of a COMSOL workflow, not an experimentally validated QPDK
# resonator prediction. The ported and driven result is a single edge mesh: a
# meander edge mesh pinned at 4 µm / 0.4 µm over a global 100 µm / 2 µm mesh,
# **659682 elements**, four modes searched near a 7.0 GHz shift with the
# largest-real-part rule, with the boundary mode analysis steps and both numeric
# ports kept. The meander-localised selected mode sits at
# **7.326615894917222 GHz** with an imaginary part of **+534576.3883 Hz**, which
# puts the loaded eigen $Q = f_r / (2|f''|)$ at **6852.73055**, and the 95th
# percentile of $|\mathbf{E}|$ inside the meander region is **2.823515744 times**
# the same percentile over the feed region. The field export's own
# complex-frequency annotation matches the selected mode, so the map below
# belongs to the mode it is shown against.
#
# The driven $S_{21}$ was solved directly at three frequencies across the window
# and at the minimum the adaptive curve reported: **-0.169357 dB** at
# 7.321615895 GHz, **-17.311564 dB** at 7.326615895 GHz, **-0.002041 dB** at
# 7.331615895 GHz, and **-24.192816 dB** at 7.32650951194 GHz. The notch at that
# directly solved minimum is **24.02346 dB** below the lower flank, and the
# directly solved two-port power sums run from 0.999939 to 1.000030, which is
# what PEC metal and a lossless dielectric should give. Those four frequencies
# are the only points the driven solver solved one at a time; everything else on
# the curve is the adaptive fit.
#
# Stage 1 supplies port-free context, not the driven answer, and the two are
# different objects. Its one completed row is a *port-free* mode at
# **7.292084525308305 GHz** with a **5.398** meander-to-feed 95th percentile
# ratio, the sixteenth of the sixteen modes that row searched. The numeric ports
# in stage 2 are matched terminations, so they load the mode: the ported
# eigenvalue is not expected to sit at the port-free frequency, and the
# 7.292 GHz row does not predict the 7.3266 GHz ported mode.
#
# No convergence result is claimed. Every ported number above comes from that
# one edge mesh, and tighter ported eigen-only meshes at 3 µm / 0.3 µm and
# 2 µm / 0.2 µm are being solved without results recorded here. The finer
# stage-1 edge mesh at 1 µm / 0.1 µm never finished, so the port-free series has
# one row as well. Nothing on this page is mesh independent, and no quality
# factor read from the notch width is independently verified.
#
# ## How this page is published
#
# The figures and numbers on this page are **saved cell outputs**. The numerical
# data comes from licensed COMSOL solves; the plotting cells were rerun against
# those exports and saved in the committed notebook. The documentation renders
# that copy rather than running the cells again.
#
# The whole path is scripted, so a licensed machine reproduces it end to end by
# setting `RUN_COMSOL = True` and, for the ported eigenmode and the driven
# window, `RUN_PORTED_DRIVEN = True`. The port-free series is its own switch,
# `RUN_PORT_FREE_SERIES`, so the ported study can run without it. Without a
# license the notebook still runs from top to bottom: the COMSOL cells are
# skipped, and every cell that reads results prints how to supply them instead
# of plotting. On a fresh machine with neither a license nor exported results you
# will therefore see the stored figures in the documentation, but a local run
# prints skip messages rather than plots. To replot locally, either export the
# files with the licensed switches on, or point `RESULTS_DIR` at a directory that
# already holds an exported copy. The `QPDK_COMSOL_RESULTS_DIR` environment
# variable does the same without editing the notebook, which is how the saved
# outputs can be regenerated from a licensed run's exports on a machine that has
# no license.
#
# ## What is being modelled
#
# The device is a QPDK
# {py:func}`~qpdk.cells.quarter_wave_resonator_coupled`: a meandering
# coplanar-waveguide (CPW) resonator placed alongside a straight feedline,
# separated by a coupling gap. This is the standard hanger geometry used to read
# out superconducting qubits {cite:p}`gopplCoplanarWaveguideResonators2008a`, and
# its resonance is one of the degrees of freedom that circuit QED uses to
# dispersively read a qubit {cite:p}`blaisCircuitQuantumElectrodynamics2021`.
#
# Two terminations define a **quarter-wave** resonator:
#
# - The end nearest the feedline is **open**, where the voltage has an antinode.
# - The far end is **shorted**, where the current has an antinode.
#
# A line with one open and one shorted end resonates when its electrical length
# is an odd multiple of $\lambda/4$ {cite:p}`m.pozarMicrowaveEngineering2012`.
# Close to resonance the coupling capacitor loads the feedline and the
# transmission $|S_{21}|$ shows a **notch**: at the resonant frequency, power
# that would travel from `coupling_o1` to `coupling_o2` is largely reflected. The
# centre of the notch gives $f_r$ and its width the loaded quality factor, which
# for a hanger is set by the coupling to the feedline together with any loss the
# model carries {cite:p}`gopplCoplanarWaveguideResonators2008a`.
#
# That is the textbook behaviour. The driven solve below does return a notch, but
# whether the mode it belongs to is the quarter-wave resonance, and how much of
# the notch width is coupling rather than loss, are questions the checks below
# answer only partly.
#
# ## The two-stage model
#
# Stage 1, port-free, answers one question: does the sheet geometry support an
# eigenmode whose field actually lives on the meander? Removing the ports removes
# the matched-load terminations as well, so the eigenfrequencies come back closer
# to real and any damping is numerical rather than partly the external decay of a
# driven line. Stage 2, ported, is what turns an eigenmode into an S-parameter
# statement. It has been run on its own edge mesh and search, and it is where the
# driven notch comes from.
#
# Both use the same five ingredients:
#
# 1. A silicon block under the metal and an air region above, meeting at the
#    metal plane.
# 2. **PEC on the metal sheets.** The metal is drawn as faces on the
#    silicon/air interface rather than as a thin extruded solid, which meshes far
#    more reliably for a metal that is three orders of magnitude thinner than the
#    substrate.
# 3. A **mesh pinned to absolute sizes**, so the resolution near the meander does
#    not follow the size of the enclosing box. The bulk runs at 100 µm / 2 µm in
#    every stage, the stage-1 meander edges at 2 µm / 0.2 µm, and the stage-2
#    meander edges at 4 µm / 0.4 µm.
# 4. In stage 1, a plain **eigenfrequency search**, with the shift, the mode
#    count, and the eigenvalue selection set per edge-mesh row. In stage 2,
#    **boundary mode analysis** steps and **numeric TEM ports** with voltage
#    integration lines across the CPW gap are kept, four modes are searched near
#    a 7.0 GHz shift with the largest-real-part rule, and the driven window
#    carries an **AWE** curve plus direct single-frequency solves at the two
#    flanks, the centre, and the AWE minimum.
# 5. An **`emw.normE` export** on a cut plane just above the metal, used to score
#    every mode by how much of its field sits on the meander.
#
# ::::{only} html
# ```{mermaid}
# flowchart TB
#     A["Ported layout:<br>metal polygons and two open feed planes"]
#     B["Sheet model:<br>air above, silicon below, metal faces at z = 0"]
#     C["Stage 1 physics:<br>PEC on the metal and no ports,<br>so no matched loads"]
#     D["Mesh:<br>absolute global 100/2 um,<br>meander edges 2/0.2 um stage 1,<br>4/0.4 um stage 2"]
#     E["Stage 1 study:<br>Eigenfrequency, 16 modes<br>at a 7.5 GHz shift,<br>then emw.normE per mode"]
#     F["Stage 1 result:<br>port-free meander mode,<br>7.292 GHz, ratio 5.398"]
#     G["Stage 2 study:<br>BMA and two numeric ports kept,<br>4 ported modes at a<br>7.0 GHz shift (eigwhich=lr)"]
#     H["Driven S21 window:<br>AWE curve plus direct solves<br>at both flanks, the centre<br>and the AWE minimum"]
#     I["Stage 2 result:<br>loaded mode 7.32662 GHz,<br>Q 6853 from the damping,<br>direct notch -24.19 dB"]
#     J["Pending:<br>ported eigen-only meshes<br>at 3/0.3 and 2/0.2 um<br>solving, no results yet"]
#     A --> B --> C --> D
#     D --> E --> F
#     D --> G --> H --> I
#     I -.-> J
# ```
# ::::
#
# ::::{only} typst or typstpdf
# The pipeline: ported layout, then the sheet model with air above and silicon
# below, then PEC on the metal with no ports and an absolutely pinned mesh whose
# meander edges carry a local element size, then an eigenfrequency search with a
# field export per mode, sixteen modes near a 7.5 GHz shift, then the port-free
# mode spectrum and localisation ratio. Separately, on the same sheet model with
# the boundary mode analysis steps and both numeric ports kept, four ported modes
# near a 7.0 GHz shift with the largest-real-part rule, then an AWE curve over
# the driven window with direct solves at the flanks, centre and AWE minimum.
# Two tighter ported eigen-only meshes are still solving and carry no results
# here.
# ::::
#
# The outer air and silicon walls use COMSOL's default PEC boundary. This is a
# finite conducting enclosure rather than an open radiating one, so it is a
# candidate explanation for any feature that moves with the enclosure size.
# Neither stage tests the enclosure; swapping the outer walls for scattering
# boundaries is a separate model change and appears under "Next steps".
#
# ### What PEC leaves out
#
# The QPDK metal is a superconductor, but this model treats it as a perfect
# electric conductor. PEC has zero surface resistance, so it predicts no
# conductor loss, and it ignores the **kinetic inductance** of the film. It is a
# useful first approximation for checking geometry, meshing, and mode
# localisation, but a measured quality factor cannot be predicted from it. The
# kinetic inductance also shifts the resonance. Replacing PEC with a
# surface-impedance or transition boundary condition is the next modelling step
# after the mesh series.
#
# ### The ported layout
#
# `prepare_comsol_layout` inverts the M1 etch mask into a ground plane, and the
# ground plane then *surrounds* the source feed ports, which sit at $x = 0$ and
# $x = 200$ µm inside the prepared bounding box. A port face there would be
# buried in ground metal, so those planes cannot be turned into TEM ports.
#
# To expose them, this notebook extends each feed with a straight CPW section to
# a plane clear of the whole resonator, then extracts with
# `crop_to_feed_ports=True` so both external faces are open CPW cross sections.
# Stage 1 does not use the ports, but the extended layout is what stage 2 drives,
# and the meander edge selection is taken from this geometry, so the extension is
# built either way. **The extension is part of the modelled device**: it
# lengthens the feedline and changes the coupling geometry, so nothing solved
# here belongs to the unextended reference cell.
#
# ### Why the driven window is checked at solved points
#
# A hanger resonator can reach $Q \sim 10^4$ to $10^6$, so its fractional
# linewidth can be $10^{-4}$ or smaller. A uniform driven sweep at a convenient
# spacing samples the band but **cannot resolve a notch** that narrow: the
# feature can fall between two points and nothing in the exported curve shows
# it. That is why the driven sweep is centred on a mode rather than scanned
# across a band.
#
# Two things have to be read from the ported solve itself, and neither one can be
# replaced by a reconstruction:
#
# - **A loaded ported eigenmode.** Adding numeric TEM ports makes them matched
#   terminations, so the ported eigenvalues are not the port-free ones. Whether
#   the meander-localised mode survives as a ported eigenmode, and where its
#   loaded frequency lands, comes from the ported eigen solve. Here it does
#   survive: the selected ported mode sits at 7.326615894917222 GHz with a
#   meander-to-feed 95th percentile ratio of 2.823515744, against the port-free
#   7.292084525308305 GHz. The models differ in both port loading and edge
#   mesh, so their 34 MHz difference cannot be assigned to either change alone.
# - **A driven $S_{21}$ notch.** The notch is what a measurement would see, and it
#   has to be read at frequencies the solver actually solved. The four direct
#   solves give **-0.169357 dB**, **-17.311564 dB**, **-0.002041 dB** and
#   **-24.192816 dB**; the last of those, at 7.32650951194 GHz, is 24.02346 dB
#   below the lower flank.
#
# COMSOL's **adaptive frequency sweep** (AWE) fits a rational model to a handful
# of solved points and fills the rest of the curve from that fit. It is the only
# affordable way to draw a dense curve on a model this size, where one direct
# solve costs minutes. It is also why the curve cannot be evidence on its own:
# its fitted rows are **not** independent direct solutions, so a narrow feature
# that appears only in an AWE-reconstructed curve is not something the solver
# resolved. On this run the fit produced 95 reconstructed rows, its minimum is
# -23.984659 dB at 7.326509512 GHz, and every one of the four direct solves
# agrees with the fitted curve at its own frequency to within 0.208157 dB. The
# fitted curve's own two-port power sum runs from 0.999923 to 1.002911, and that
# 0.291 percent apparent surplus is a property of the rational fit, not power
# gained by the device.
#
# ::::{admonition} Reading the numbers on this page
# :class: warning
#
# The values come from the licensed solves described below, and both stages have
# run, but they are not a validated device prediction. All of them come from a
# single edge mesh, so no frequency, ratio, or quality factor here has been shown
# to be mesh independent; two tighter ported eigen-only meshes are still solving.
# The metal is a perfect conductor with no surface resistance or kinetic
# inductance, so conductor loss and its frequency shift are missing. The feed
# extension changes the layout relative to the reference cell. The
# **7.292084525308305 GHz** row is the *port-free* mode and is not the ported one
# at 7.326615894917222 GHz. The driven notch is verified at the four directly
# solved frequencies only: the curve between them is a fit, so any quality factor
# read from its linewidth is not independently verified. Treat the 24.192816 dB
# direct notch depth as one mesh's result, not as what a fabricated device would
# show.
# ::::
#
# **References:**
# - [COMSOL "Coplanar Waveguide Resonator" model](https://www.comsol.com/model/download/953251/models.rf.cpw_resonator.pdf)
# - [COMSOL RF Module User's Guide](https://doc.comsol.com/6.3/doc/com.comsol.help.rf/RFModuleUsersGuide.pdf)
# - [COMSOL Reference Manual: Analyzing Model Convergence and Accuracy](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_modeling.19.043.html)
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
from qpdk.simulation import (
    add_cpw_rf_study,
    build_comsol_sheet_model,
    pin_absolute_edge_mesh_sizes,
    prepare_comsol_layout,
)
from qpdk.tech import coplanar_waveguide

try:
    import mph
    from jpype import JInt
except ImportError:
    mph = None

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


prefer_svg_figures()
STYLE_SOURCE = apply_qpdk_style()
print("Plot style: QPDK" if STYLE_SOURCE != "matplotlib defaults" else STYLE_SOURCE)

# %% [markdown]
# ## Build the ported layout
#
# The resonator is created with an explicit CPW cross-section so that the
# centre-conductor width and the gap are known constants we reuse when
# describing the ports. On the source cell the feed ports `coupling_o1` and
# `coupling_o2` sit at $x = 0$ and $x = 200$ µm, well inside the prepared ground
# plane. Each is extended to the left and right with a straight CPW of the same
# cross-section, so the new feed planes at $x = -1320$ µm and $x = 2200$ µm are
# clear of the resonator.
#
# The ground reaches at least 1200 µm beyond the resonator in every lateral
# direction. The prepared box spans $x=-1320\ldots2200$ µm and
# $y=-2041\ldots1211$ µm. The larger box reduces the influence of the outer
# PEC walls, but does not remove it; testing that influence means replacing
# those walls with scattering boundaries, which is not done on this page.

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
# The result is three metal polygons with one hole. One polygon is the ground
# plane, and the hole is the CPW channel cut through it: the centre-conductor
# strip, both etch gaps, and the surrounding ground all come from that one
# outline-plus-hole shape. Keeping the hole is what makes the corrected sheet
# model necessary: the builder imprints these outlines with `ProjectToFaces`, so
# the etched region stays open in the sheet instead of being filled, and that is
# the step that needs the Design Module CAD kernel.

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
# Three calls turn the layout into a configured model:
#
# - {py:func}`~qpdk.simulation.comsol_sheet.build_comsol_sheet_model` creates the
#   air and silicon blocks meeting at $z = 0$, imprints the layout metal on that
#   interface as faces while preserving the etch hole, and assigns materials
#   ($\epsilon_r = 1$ air, $\epsilon_r = 11.7$ silicon).
# - {py:func}`~qpdk.simulation.comsol_rf.add_cpw_rf_study` selects the metal
#   faces and applies PEC, adds two numeric TEM ports with voltage integration
#   lines spanning the CPW gap, adds the mesh sequence, and adds a study with one
#   boundary mode analysis step per port plus a frequency step. Every selection
#   is derived from the layout geometry, so no face or edge ID is hard-coded.
# - Stage 1 then *removes* the port and study-step features it does not want and
#   installs its own eigenfrequency step, and the last calls mesh and solve.
#
# The order matters when the RF features come out. A boundary mode analysis step
# points at a port by name, so the study steps `bma1`, `bma2`, and `freq` are
# removed **before** the physics features `port1` and `port2`, and the model is
# never left with a step referring to a deleted port.
#
# The mesh is not left to the physics. `add_cpw_rf_study` sets COMSOL's
# automatic mesh size, which scales its element sizes with the longest dimension
# of the domain, so a change to the enclosing box would change the resolution on
# the meander. Stage 1 replaces that with absolute sizes through
# {py:func}`~qpdk.simulation.comsol_mesh.pin_absolute_edge_mesh_sizes`: the
# default `Size` feature takes 100 µm / 2 µm globally, a second `Size` feature
# takes the per-case edge sizes on a named meander edge selection, and one
# `FreeTet` follows both.
#
# ### The meander edge selection
#
# The local size goes on the edges that bound the CPW trace and gap in the
# meander, because that is where the field of a meander mode concentrates and
# where a coarse mesh smears it. The selection is a COMSOL Box over edges
# (`entitydim` 1, `condition` `somevertex`) spanning the meander stripe below the
# feedline, in a thin $z$ slab around the metal sheet. `somevertex` takes any
# edge with a vertex in the box, which reaches edges crossing the box without
# needing them to lie entirely inside it.
#
# The box stops short of the feedline band at $y = 0$ so the whole feedline is
# not refined along its length, and a probe over the feed band on either side of
# the meander checks that the selection and the feed edges share nothing. If they
# did, the local size would thin the mesh along the entire feed.
#
# `RUN_COMSOL` gates the licensed branch. `mph.start(cores=...)` launches a local
# COMSOL process and attaches to it. Only one MPh client can exist per Python
# process, and the call needs a COMSOL installation and a license, so the whole
# block is off by default; set `RUN_COMSOL = True` on a licensed machine to
# build, mesh, solve, and save.
#
# `RESULTS_DIR` is where the cells that read results look for exported files. It
# defaults to `None` so that a run without a license skips those cells, and to
# `MODEL_DIR` when `RUN_COMSOL` is `True`, because the licensed branch exports
# into `MODEL_DIR`. The optional environment variable
# `QPDK_COMSOL_RESULTS_DIR` overrides both: when it is set, `RESULTS_DIR` points
# at it, so the notebook can replot real exported solver output on a machine with
# no license. That is how the saved figures on a documentation build can be
# produced from a licensed run's exports without starting COMSOL here. The
# variable is read once, `~` is expanded, and an empty value is ignored. Point it
# at any directory of exported files to replot an existing run, or set
# `RESULTS_DIR = Path("exports")` in the notebook directly. It is meant for the
# read side, on a machine with no license; a licensed run still writes into
# `MODEL_DIR`.

# %%
RUN_COMSOL = False
# Stage 1, the port-free series, is its own switch so stage 2 can run without it:
# the ported study does not need stage-1 rows, it only reads them as context. Off
# by default, so enabling RUN_COMSOL alone never starts an expensive solve.
RUN_PORT_FREE_SERIES = False
# The ported eigenmode and driven sweep are shown in "Stage 2".
RUN_PORTED_DRIVEN = False
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
# candidate for a quarter-wave resonance at this meander length.
EIGEN_MODE_WINDOW_GHZ = (3.0, 25.0)
# Element count above which a row records its mesh and stops instead of solving,
# so an unexpectedly large mesh cannot take the job down.
MAX_ELEMENTS = 3_500_000

# Absolute element sizes of the pinned sequence, in µm, the same for every row.
GLOBAL_HMAX_UM = 100.0
GLOBAL_HMIN_UM = 2.0


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
PORTED_EDGE_HMAX_UM = 4.0
PORTED_EDGE_HMIN_UM = 0.4
PORTED_SHIFT_GHZ = 7.0
PORTED_NEIGS = 4
PORTED_EIGWHICH = "lr"
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
    model = build_comsol_sheet_model(
        client,
        layout,
        name=f"QPDK CPW port-free eigen, {label}",
        substrate_thickness_um=200.0,
        air_height_um=200.0,
    )
    try:
        add_cpw_rf_study(
            model,
            layout,
            cpw_gap_um=CPW_GAP_UM,
            frequency_ghz=case.shift_ghz,
            mesh_size=2,
        )
        configure_port_free_eigen_study(model, case)
        edges = create_meander_edge_selection(model)
        element_count = pin_absolute_edge_mesh_sizes(
            model,
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
# Both edge meshes run in one process behind one MPh client, because only one
# client can exist per Python process. Each row builds a fresh model, pins the
# global sizes at 100 µm / 2 µm and its own meander-edge sizes, runs its own
# eigenfrequency search, exports `emw.normE` for every mode inside the 3 to
# 25 GHz window, and scores each mode by the meander-to-feed 95th percentile
# ratio. The row's selected mode is the one with the highest ratio, which is the
# mode whose field actually sits on the meander.
#
# The two rows do **not** search the same span of the spectrum. The coarser
# 2 µm / 0.2 µm row keeps sixteen modes near a 7.5 GHz shift with COMSOL's own
# eigenvalue selection, and it is the row that returned. The finer 1 µm / 0.1 µm
# row searches four modes near a 7.0 GHz shift with `eigwhich="lr"`, the
# largest-real-part selection: the same sixteen-mode search at 7.5 GHz was run on
# the finer mesh and did not finish inside the time it was given, so the finer
# search was retargeted to fewer modes at a lower shift. That row never
# completed, so only one row carries a result and the port-free series has no
# second point.
#
# A different search is why the rows are compared by field and not by mode number
# or by frequency. Which eigenvalues come back, and in what order, depends on the
# shift and the selection rule, so the nth mode of one row is not the nth mode of
# the other, and the row with fewer modes has fewer candidates to offer. The
# meander-to-feed 95th percentile ratio identifies a mode by where its field
# sits, which is a property of the field pattern rather than of the search, so it
# is what carries a mode across a change of search settings. Reading a frequency
# difference between two rows as one physical mode moving is only sound once both
# selections are confirmed to be the same physical mode, and with one row solved
# there is no such difference to read.
#
# The JSON is rewritten after every row, so interrupting a long series keeps the
# rows already solved. A row that fails to mesh or solve is printed in full and
# left out. Above `MAX_ELEMENTS` a row records only its mesh and stops, so an
# unexpectedly large mesh cannot take the job down.
#
# The coarser row, 2 µm / 0.2 µm on the meander edges, is the one solved: about
# 1.294 million elements and the mode at 7.292084525308305 GHz with a ratio of
# 5.398, the sixteenth of its sixteen modes. This is a port-free result. It is
# **not** the ported mode from stage 2, which sits at 7.326615894917222 GHz under
# the matched port terminations; the two are reported separately and never
# merged. Nothing below is a convergence result, because the port-free series has
# one solved mesh.

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
# This cell reads the port-free eigenfrequency rows back and prints, per row, the
# element count, that row's eigenfrequency search settings, every mode inside the
# window with its localisation ratio, and the selected mode. The figure plots the
# selected row's mode frequencies against their localisation ratio, so a mode
# that lives on the meander is separated from the ones that do not.
#
# The **numerical delta between the two edge meshes is reported only once both
# rows exist**. One solved mesh says nothing about convergence: the whole point of
# the second row is to see whether the selected frequency and ratio move when the
# meander edges are refined, and a single number cannot show that. Only the
# 2 µm / 0.2 µm row is solved, and the 1 µm / 0.1 µm row never completed, so the
# cell lists the rows it cannot compare instead of printing a delta. The two rows
# carry different eigenfrequency searches, so a delta would be between each row's
# best-localised mode, and the cell says so: the same physical mode has to be
# confirmed by field localisation before a delta could be read as one mode
# moving.
#
# Once at least two rows are genuinely solved and carry a finite, localised
# selected mode, a compact figure plots that mode's frequency against the mesh
# element count, labels each point with its meander edge `hmax`/`hmin`, and shows
# the final delta in MHz. With three or more rows the points are joined in
# increasing element-count order and a second panel shows each incremental shift.
# Rows that are mesh-only or carry no finite selection are left out, so a missing
# row is reported rather than plotted. Two meshes measure a change, they are **not
# proof of asymptotic convergence**; a third mesh, or a Richardson-style estimate,
# would be needed before either frequency could be called settled.
#
# With `RESULTS_DIR` unset, or set to a directory without the export, the cell
# prints how to supply the file and draws nothing.

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
            print(f"\n{row['label']}: {row['element_count']} elements")
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
                ratio_text = f"{ratio:>17.3g}" if ratio else f"{'not scored':>17}"
                print(
                    f"  {entry['solution_index']:>5} {entry['real_ghz']:>16.9f} "
                    f"{entry['imag_hz']:>+12.3f} {ratio_text}"
                )
            selected = row.get("selected_mode")
            if selected is None:
                print("  No mode inside the window had a usable field export.")
            else:
                print(
                    f"  selected: mode {selected['solution_index']} at "
                    f"{selected['real_ghz']:.9f} GHz, meander/feed p95 "
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
                f"{coarser['element_count']} to {finer['element_count']} elements, "
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
# The exported field is the electric-field norm on the $z = 1$ µm plane for the
# selected mode of the coarser row, read from the export's own frequency
# annotation rather than assumed. The metal sheet lies at $z = 0$ and the plane
# sits just above it, so the map shows the field in the CPW gaps of whatever
# metal the plane cuts through.
#
# The export covers the whole prepared box, and outside the device the field
# falls to values far below the ones near the metal, so plotting all of it on a
# colour scale wide enough to hold the whole range washes the device out. The
# figure below therefore crops the nodes to a window around the coupling section,
# the meander, and the feedline, and takes its colour limits from percentiles of
# the cropped data rather than from the full-domain maximum.
#
# The crop still holds far more nodes than the figure needs to show, and drawing
# every one of them writes a very large SVG for no visible gain. For **display
# only**, the cell draws a fixed-stride subset of the cropped nodes with fewer
# contour levels. This changes what is drawn, not what was solved: the simulation,
# the export, and the meander-to-feed ratio all use the full-resolution field, and
# no raster image is produced. The saved SVG is still true vector. The cell prints
# the full cropped node count and the drawn count, and falls back to the full view
# if the stride would leave too few nodes to triangulate.

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
# What this map supports is narrow. It shows that for the selected mode the field
# is concentrated on the meander, which is what a meander-localised mode should
# look like and what a coarse mesh on the meander edges would smear away. The
# ratio printed by the spectrum cell is the numerical version of the same
# statement. A single cut plane at one frequency cannot establish localisation on
# its own, so this is a consistency check on the selection, not proof of the
# quarter-wave resonance.
#
# What the map does **not** show:
#
# - **That this is the quarter-wave resonance.** Localisation is not the same as
#   the quarter-wave condition, and this is the port-free mode, not the loaded
#   one that the driven solve below actually excites. The driven notch shows that
#   the ported mode is resonant; it does not turn this port-free field map into
#   the quarter-wave mode's field.
# - **A converged field.** This is the coarser of the stage-1 edge meshes, and the
#   finer one never completed, so whether the localisation survives refinement is
#   still open.
#
# The figure is also a strided display of the cropped nodes, not every exported
# node. That stride is a drawing choice only; the field values, the export, and
# the numerical ratio are full resolution.

# %% [markdown]
# ## Stage 2: ported eigenmode and a driven curve verified by direct solves
#
# Stage 2 builds the same sheet model from the same layout behind
# `RUN_PORTED_DRIVEN`, which is off by default so a licensed run has to ask for
# the ported solve explicitly. It keeps the two numeric TEM ports with their
# voltage integration lines, so the **boundary mode analysis steps stay**. It
# solves a ported eigenfrequency search, then a driven window centred on the
# ported selected mode.
#
# The ported solve here ran on its own edge mesh, pinned at 4 µm / 0.4 µm on the
# meander edges over the same 100 µm / 2 µm global mesh, and came back with
# **659682 elements**. Four modes were searched near a 7.0 GHz shift with
# `eigwhich="lr"`. The selected mode is the meander-localised one, at
# **7.326615894917222 GHz** with an imaginary part of **+534576.3883 Hz**, so the
# loaded eigen $Q = f_r / (2|f''|)$ is **6852.73055**, and its meander-to-feed
# 95th percentile field ratio is **2.823515744**. That $Q$ comes from the
# eigenvalue's damping, which carries the port loading and the numerical error of
# this one mesh; it is not read from the notch width and it is not a
# converged quality factor.
#
# ### A dense curve without a many-hour sweep
#
# A direct solve on a model this size costs minutes, so a dense direct sweep over
# the window is not affordable and a sparse one would step straight over a notch
# whose width is a small fraction of the window. So the window is covered by
# COMSOL's **adaptive frequency sweep** (AWE), which fits a rational model to a
# handful of solved points and fills the rest of the curve from that fit. The
# curve is dense at the cost of a few solves, and **its rows are AWE
# interpolation, not independent solves**. Nothing on this page may call a row of
# it a direct solve. On this run the fit produced 95 reconstructed rows over the
# roughly 10 MHz window.
#
# The evidence for the notch is separate from the curve that locates it:
#
# 1. Three direct single-frequency solves, at the low flank, the centre, and the
#    high flank, with AWE off, each one frequency the solver actually solved.
#    Each is checkpointed as soon as it returns, so an interrupted run keeps the
#    points it solved. Here they are **-0.169357 dB** at 7.321615895 GHz,
#    **-17.311564 dB** at 7.326615895 GHz, and **-0.002041 dB** at
#    7.331615895 GHz.
# 2. One more direct solve at the frequency where the AWE curve reaches its
#    minimum, since a fit can place a minimum between the direct points. Here
#    that is **-24.192816 dB** at 7.32650951194 GHz, and the fitted curve's own
#    minimum is -23.984659 dB at 7.326509512 GHz.
# 3. A notch verdict at the **directly solved AWE minimum only**: it has to sit
#    at least `NOTCH_MIN_DEPTH_DB` below both flanks. The centre is solved as an
#    AWE comparison point, not as a notch requirement, because a loaded mode can
#    sit off the ported eigenfrequency and the true minimum need not fall on it.
#    Here the direct minimum is **24.02346 dB** below the lower flank, so the
#    notch holds at a frequency the solver solved.
# 4. A comparison of the AWE curve against every one of those direct levels, at
#    the direct frequencies, within `AWE_AGREEMENT_DB`. A direct frequency that
#    falls outside the curve fails the comparison rather than being skipped. The
#    curve itself is checked against the requested grid it was built from. All
#    four comparisons here pass, the worst of them off by 0.208157 dB.
# 5. A **two-port power balance** check near unity, on the direct solves only.
#    The metal is PEC and the dielectric is lossless, so
#    $|S_{11}|^2 + |S_{21}|^2$ should be very close to 1 at every directly solved
#    frequency. A large deficit there means power is leaving through a channel
#    this two-port record does not see, and the result is then labelled unverified
#    rather than accepted with the deficit ignored. The four direct solves here
#    run from 0.999939 to 1.000030. The reconstructed AWE rows are checked against
#    the same band as a **separate diagnostic**: the rational fit put those rows
#    between 0.999923 and 1.002911, and that 0.291 percent fitted surplus is not
#    physical loss or gain, so it never unverifies a resonance on its own.
#
# Every evaluation reads the dataset MPh reports as the default, not the first
# dataset on the model, because a study with boundary mode analysis steps holds
# more than one and the first is not the frequency solution. That reproduces
# MPh's own semantics, which pick the first compatible dataset rather than
# provably the latest, so each read-back frequency is checked against the
# frequency that was requested.
#
# If the direct points do not support a notch at the minimum, the curve does not
# reproduce them, or the directly solved power balance is off, the run is reported
# as **unverified**. A reconstructed minimum is not evidence of a resonance, so a
# failed check reports the failure rather than inferring one. A reconstructed
# row's fitted surplus does not fail any of these on its own.
#
# Two things this stage must not do, both of which the earlier single-stage
# version of this page got wrong:
#
# - **Do not assume the port-free frequency.** The ports are matched
#   terminations, so a loaded ported eigenmode will not sit at the port-free
#   frequency, and the meander-localised mode may not survive loading at all.
#   Both are read from the ported solve: the mode does survive, at 7.3266 GHz
#   against the port-free 7.2921 GHz, and the two are never merged.
# - **Do not quote a quality factor from one mesh.** The notch width is the
#   quantity a $Q$ comes from, and a width from a single edge mesh is not settled
#   any more than the frequency is. That is why the 6852.73055 figure above is
#   labelled as the eigenvalue's own loaded damping on one mesh, and no $Q$ is
#   read from the fitted curve.

# %%
PORTED_EIGEN_JSON = "comsol_cpw_ported_eigen.json"
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
    to be identified from the ported solve.

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
        if not EIGEN_MODE_WINDOW_GHZ[0] <= real_ghz <= EIGEN_MODE_WINDOW_GHZ[1]:
            continue
        path = ported_mode_field_path(index)
        export_mode_field(model, index, path)
        ratio = field_localization_ratio(path)
        if ratio is not None:
            scored.append((ratio, mode, path))
    if not scored:
        raise ValueError(
            "No ported mode inside the window had a usable field export: "
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
    return max(min(points, MAX_SWEEP_POINTS), 5)


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
    if frequencies_ghz.size < 2 or not np.all(np.diff(frequencies_ghz) > 0.0):
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
    insertion = np.clip(
        np.searchsorted(frequencies_ghz, requested_ghz), 1, frequencies_ghz.size - 1
    )
    gaps_ghz = np.minimum(
        requested_ghz - frequencies_ghz[insertion - 1],
        frequencies_ghz[insertion] - requested_ghz,
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
    ported_model = build_comsol_sheet_model(
        client,
        layout,
        name="QPDK Coupled Quarter-Wave Resonator ported eigenmodes",
        substrate_thickness_um=200.0,
        air_height_um=200.0,
    )
    try:
        add_cpw_rf_study(
            ported_model,
            layout,
            cpw_gap_um=CPW_GAP_UM,
            frequency_ghz=PORTED_SHIFT_GHZ,
            mesh_size=2,
        )
        # Ported eigen keeps bma1 and bma2: the ports need their boundary mode
        # fields, and only the frequency step is replaced.
        configure_ported_eigen_study(ported_model)
        create_meander_edge_selection(ported_model)
        ported_elements = pin_absolute_edge_mesh_sizes(
            ported_model,
            edge_selection=MEANDER_EDGE_SELECTION,
            global_hmax_um=GLOBAL_HMAX_UM,
            global_hmin_um=GLOBAL_HMIN_UM,
            edge_hmax_um=PORTED_EDGE_HMAX_UM,
            edge_hmin_um=PORTED_EDGE_HMIN_UM,
        )
        ported_model.java.study("std1").run()
        for problem in ported_model.problems():
            print(f"  ported eigen model reports: {problem}")
        ported_modes = eigenfrequencies(ported_model)
        ported_selected, ported_field_source, ported_ratio = select_ported_mode(
            ported_model, ported_modes
        )
        # The read side looks for one stable name, so the selected mode's field
        # is copied there from the per-mode export that was already scored.
        ported_field_path = MODEL_DIR / PORTED_FIELD_TXT
        ported_field_path.write_bytes(ported_field_source.read_bytes())
        ported_identification = identify_ported_mode(
            ported_field_path, ported_selected, ported_ratio
        )
        (MODEL_DIR / PORTED_EIGEN_JSON).write_text(
            json.dumps(
                {
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
                        "meander_to_feed_p95": ported_ratio,
                    },
                    "selected_mode_identified": ported_identification[
                        "selected_mode_identified"
                    ],
                    "selected_mode_identified_reason": ported_identification["reason"],
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Ported eigen: {ported_elements} elements, selected "
            f"{ported_selected.real / 1e9:.9f} GHz, meander/feed p95 "
            f"{ported_ratio:.3f}, identified "
            f"{ported_identification['selected_mode_identified']}"
        )
    finally:
        client.remove(ported_model)

    center_ghz = ported_selected.real / 1e9
    half_span_ghz = sweep_half_span_ghz(ported_selected.imag)
    low_ghz, high_ghz = center_ghz - half_span_ghz, center_ghz + half_span_ghz
    # Ten requested rows per loading width over the span the range actually
    # sweeps, so a 1.07 MHz notch inside the +/-5 MHz cap asks for ~95 rows.
    awe_points = sweep_requested_points(ported_selected.imag)

    model = build_comsol_sheet_model(
        client,
        layout,
        name="QPDK Coupled Quarter-Wave Resonator driven",
        substrate_thickness_um=200.0,
        air_height_um=200.0,
    )
    add_cpw_rf_study(
        model,
        layout,
        cpw_gap_um=CPW_GAP_UM,
        frequency_ghz=PORTED_SHIFT_GHZ,
        mesh_size=2,
    )
    create_meander_edge_selection(model)
    pin_absolute_edge_mesh_sizes(
        model,
        edge_selection=MEANDER_EDGE_SELECTION,
        global_hmax_um=GLOBAL_HMAX_UM,
        global_hmin_um=GLOBAL_HMIN_UM,
        edge_hmax_um=PORTED_EDGE_HMAX_UM,
        edge_hmin_um=PORTED_EDGE_HMIN_UM,
    )

    # The run's own ID names the curve and the record, so a reader can tell a
    # fresh curve from a stale verification record.
    run_id = uuid.uuid4().hex[:12]
    direct_path = MODEL_DIR / f"{DIRECT_PREFIX}-{run_id}.json"
    awe_path = MODEL_DIR / f"{AWE_CURVE_PREFIX}-{run_id}.csv"
    record: dict[str, Any] = {
        "run_id": run_id,
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
        record["direct_points"].append(solve_direct_point(model, label, frequency))
        record["verified_reason"] = (
            f"direct point {label!r} solved; curve and checks pending"
        )
        checkpoint()
    direct_points = list(record["direct_points"])

    flank_levels_db = [direct_points[0]["s21_db"], direct_points[2]["s21_db"]]
    direct_notch_at_centre = notch_verdict(direct_points[1]["s21_db"], flank_levels_db)

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
    direct_notch_at_minimum = notch_verdict(direct_minimum["s21_db"], flank_levels_db)

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
            float(np.min(np.abs(curve["frequencies_ghz"] - point["frequency_ghz"])))
            if curve["frequencies_ghz"].size
            else None
        )
        comparisons.append({
            "label": point["label"],
            "frequency_ghz": point["frequency_ghz"],
            "direct_db": point["s21_db"],
            "awe_curve_db": curve_db,
            "curve_gap_ghz": nearest_gap_ghz,
            "difference_db": (None if curve_db is None else curve_db - point["s21_db"]),
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
    verified = bool(curve_agrees and direct_notch_at_minimum["is_a_notch"] and power_ok)

    # The worst power deviation over the direct solves is the largest magnitude
    # from unity, so a slight surplus (a negative deficit) is not mistaken for the
    # best point.
    direct_deficits = [point["power_deficit"] for point in record["direct_points"]]
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
                "direct points outside the curve (" + ", ".join(outside_curve) + ")"
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
elif not RUN_PORTED_DRIVEN:
    print(
        "The stage-2 solve is off by default. Set RUN_PORTED_DRIVEN = True with "
        "RUN_COMSOL = True to solve the ported eigenmode and the driven window; "
        "it runs on its own, and reads the port-free series only as context when "
        "RUN_PORT_FREE_SERIES is also True."
    )

# %% [markdown]
# ## Driven curve and its direct checks (stage 2 output)
#
# This cell reads the newest verification record and the adaptive curve that
# record names, and plots them together: the curve as a line, the directly solved
# frequencies as distinct markers. **The line is AWE interpolation.** Its rows are
# a mix of solved points and values from COMSOL's rational fit, and none of them
# is called an independent solve here. The markers are the only frequencies the
# solver solved one by one.
#
# The record stores the curve by bare file name, which the cell resolves under
# `RESULTS_DIR`, so a saved notebook never carries an absolute path from the
# machine that solved. It carries a per-run ID that also appears in the curve's
# file name, and the cell refuses to plot a curve that does not carry the
# record's ID, so a fresh curve is never shown against a stale verification
# record. A record that is still partial, or whose curve is missing or
# unreadable, prints that and plots nothing rather than reading an unbound array.
# The record is checkpointed after each direct solve and again after the curve,
# before the model is saved, so an interrupted run keeps what it solved and stays
# marked unverified.
#
# The verdict printed below is read from the direct points, the comparisons, and
# the two-port power balance, not from the curve's own minimum. A notch is
# required only at the directly solved AWE minimum, relative to both flanks; the
# centre is a comparison point, because a loaded mode can sit off the ported
# eigenfrequency. The directly solved power balance is checked **near unity**, not
# merely below one: with PEC metal and a lossless dielectric, a large deficit
# means power is leaving through a channel this two-port record does not see, and
# that leaves the run unverified. A reconstructed row's power sum is reported as a
# separate diagnostic instead, and a fitted surplus above the band edge does not
# unverify the resonance. When the checks do fail, no resonance is inferred and
# the figure is still drawn, because it is what a person would look at to see
# why. With no verification record in `RESULTS_DIR` the cell prints how to supply
# the files instead.
#
# What this run's record holds: **95 reconstructed curve rows**, the curve
# minimum at **-23.984659 dB** at 7.326509512 GHz, and four direct points, which
# are **-0.169357 dB** at 7.321615895 GHz, **-17.311564 dB** at 7.326615895 GHz,
# **-0.002041 dB** at 7.331615895 GHz, and **-24.192816 dB** at
# 7.32650951194 GHz. The direct minimum is **24.02346 dB** below the lower
# flank. Every direct point agrees with the fitted curve at its own frequency
# within 0.208157 dB, and the direct power sums span 0.999939 to 1.000030. The
# curve's own power sums span 0.999923 to 1.002911; that 0.291 percent surplus
# sits on fitted rows, is an interpolation artifact rather than power gained, and
# the four direct solves are what the verdict rests on.
#
# So the notch is verified **at those four solved frequencies**, and nowhere
# else. The curve between them, and any quality factor taken from the width of a
# fitted line, are not independently verified. The depth is set by the coupling
# and by whatever loss the model carries: the PEC metal adds no conductor or
# dielectric loss, so a lossless model still shows an external decay from the
# matched feed ports, with numerical error on top, and the depth is not a
# fabricated-device prediction. No quality factor is read from this curve while
# it comes from one edge mesh.

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
    # A record is only readable once every key the checks below need is present;
    # a checkpoint written mid-run has some of them missing.
    required_keys = (
        "run_id",
        "window",
        "awe_points_requested",
        "direct_points",
        "comparisons",
        "curve_agrees",
        "direct_notch_at_minimum",
    )
    missing_keys = [key for key in required_keys if key not in direct]
    if not isinstance(curve_name, str):
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
            if curve_arrays is not None and curve_arrays[0].size >= 2:
                curve_ghz, curve_s21_db, curve_s11_db = curve_arrays
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
                points = direct["direct_points"]
                point_offsets_khz = [
                    (point["frequency_ghz"] - centre_ghz) * 1e6 for point in points
                ]
                point_s21_db = [point["s21_db"] for point in points]

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
                print(
                    f"Direct points: {len(points)}, power balance in band: "
                    f"{all(point.get('power_within_band', False) for point in points)}; "
                    f"worst unit-power deviation "
                    f"{max((point.get('power_deficit', float('nan')) for point in points), key=abs, default=float('nan')):+.2e}"
                )
                for comparison in direct["comparisons"]:
                    curve_db = comparison["awe_curve_db"]
                    curve_text = (
                        "outside curve" if curve_db is None else f"{curve_db:+.3f} dB"
                    )
                    difference = comparison["difference_db"]
                    difference_text = (
                        "n/a" if difference is None else f"{difference:+.3f} dB"
                    )
                    print(
                        f"  {comparison['label']}: direct "
                        f"{comparison['direct_db']:+.3f} dB, "
                        f"AWE curve {curve_text}, difference {difference_text}"
                    )

                minimum_index = int(np.argmin(curve_s21_db))
                print(
                    f"AWE curve minimum (reconstructed, not a direct solve): "
                    f"{curve_s21_db[minimum_index]:+.3f} dB at "
                    f"{curve_ghz[minimum_index]:.9f} GHz"
                )
                if direct["verified"]:
                    print(
                        "Verified: the direct solve at the AWE minimum is a notch "
                        "below both flanks, the AWE curve reproduces the direct "
                        "levels at all four directly solved frequencies within "
                        "tolerance, and the directly solved two-port power balance "
                        "is near unity. No stage failed or was skipped, but only "
                        "those four frequencies are independent solves."
                    )
                else:
                    outside = direct.get("direct_points_outside_curve") or []
                    reasons = []
                    if outside:
                        reasons.append(
                            "direct points outside the curve: " + ", ".join(outside)
                        )
                    elif not direct["curve_agrees"]:
                        reasons.append("the curve does not reproduce the direct levels")
                    if not direct["direct_notch_at_minimum"]["is_a_notch"]:
                        reasons.append("no direct notch at the AWE minimum")
                    if not direct.get("power_within_band", True):
                        reasons.append(
                            "the directly solved two-port power balance is outside "
                            "the band, so power leaves through a channel this record "
                            "does not see"
                        )
                    print(
                        "UNVERIFIED: "
                        + ("; ".join(reasons) if reasons else "the checks did not pass")
                        + ". No resonance is inferred, and no quality factor is read "
                        "from the reconstructed curve."
                    )

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(
                    offset_khz,
                    curve_s21_db,
                    color="C0",
                    label=r"AWE interpolation of $|S_{21}|$ (reconstructed)",
                )
                ax.plot(
                    point_offsets_khz,
                    point_s21_db,
                    marker="x",
                    markersize=8,
                    linestyle="none",
                    color="crimson",
                    label="direct solve",
                )
                for point in points:
                    ax.annotate(
                        point["label"],
                        ((point["frequency_ghz"] - centre_ghz) * 1e6, point["s21_db"]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                        color="crimson",
                    )
                ax.set_xlabel("Frequency offset from the ported mode (kHz)")
                ax.set_ylabel(r"$|S_{21}|$ (dB)")
                ax.set_title(
                    f"Driven window around {centre_ghz:.6f} GHz: AWE curve and "
                    "direct solves"
                )
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()

# %% [markdown]
# ## Ported eigenfrequency (stage 2 output)
#
# When `comsol_cpw_ported_eigen.json` is present, this cell prints the ported
# eigen solve under its own heading. The ported eigenvalues are a different
# object from the port-free ones above: the numeric TEM ports are matched
# terminations, so these modes are **loaded**, and their imaginary part carries
# the port loading on top of any numerical error. The two sets are therefore
# printed separately and never merged or compared one to one.
#
# For this run the cell prints **659682 elements**, a 7.0 GHz shift with four
# modes requested under `eigwhich="lr"`, and the selected mode at
# **7.326615894917222 GHz** with an imaginary part of **+534576.3883 Hz**. The
# loss ratio the cell derives from that pair, $f'/(2|f''|)$, is **6852.73055**,
# and it is labelled as an eigenvalue damping ratio with the port loading
# included, not as a Q read from a measured or simulated linewidth. One mesh,
# one mode, so neither the frequency nor that ratio is shown to be mesh
# independent; tighter ported eigen-only meshes at 3 µm / 0.3 µm and
# 2 µm / 0.2 µm are solving and are not in this record.
#
# When the file is absent, the cell says so and infers nothing.

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
            f"  selected ported mode: {selected_hz['real'] / 1e9:.9f} GHz, "
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
# The ported field export is the electric-field norm on the same $z = 1$ µm cut
# plane the port-free map above uses, written for the selected ported mode. It is
# labelled **PORTED** wherever it appears, because the numeric TEM ports are
# matched terminations: this is a loaded mode's field, not the port-free one, and
# the two are not compared here.
#
# Before anything is drawn, the cell checks the export belongs to the mode it is
# shown against. COMSOL annotates the export header with the complex frequency of
# the solution it wrote, and the real part of that annotation has to match
# `selected_mode_hz.real` in `comsol_cpw_ported_eigen.json` within a relative
# $10^{-4}$. A mismatch is refused rather than plotted, so a stale field file
# cannot be presented as the selected mode's field. On this run the annotation
# does match the selected 7.326615894917222 GHz mode, so the map drawn here is
# that mode's field.
#
# The loaded mode's meander-to-feed 95th percentile field ratio is
# **2.823515744**, against the port-free row's 5.398. Those are two different
# modes under two different port conditions on two different edge meshes, so the
# drop is not read as the loading moving the field; it is reported as the ported
# number on its own.
#
# One cut plane, one ported mode, one edge mesh. That is a consistency check on
# where the loaded mode's field sits, not a convergence result, and no frequency
# here is called settled.

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
        selected_hz = ported_record.get("selected_mode_hz") or {}
        selected_real_hz = selected_hz.get("real")
        if selected_real_hz is None or not np.isfinite(selected_real_hz):
            print(
                f"{PORTED_EIGEN_JSON} records no finite selected ported mode, so "
                "there is no mode to check the field against and it is not plotted."
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
# ## Summary
#
# 1. Built a coupled quarter-wave resonator, extended both feeds with straight
#    CPW to planes clear of the resonator, and extracted a ported layout with
#    `crop_to_feed_ports=True`. The extension is part of the modelled device and
#    changes the coupling geometry relative to the unextended cell.
# 2. Built the COMSOL sheet model (air, silicon, metal faces on the interface
#    with the etch hole preserved) and configured the PEC, the mesh, and the
#    study, all from layout geometry.
# 3. Replaced the physics-controlled mesh with absolute sizes: 100 µm / 2 µm
#    globally, 2 µm / 0.2 µm on a named meander edge selection for the port-free
#    series, and 4 µm / 0.4 µm on the same selection for the ported solve,
#    excluding the feedline edges.
# 4. Removed the port study steps and the port physics features, in that order,
#    and ran the port-free eigenfrequency search. The coarser row searched
#    sixteen modes near a 7.5 GHz shift and is the one solved: about 1.294
#    million elements and a mode at 7.292084525308305 GHz whose meander-to-feed
#    95th percentile ratio is 5.398, the sixteenth of the sixteen modes. The
#    finer row at 1 µm / 0.1 µm, estimated at about 3.009 million elements,
#    searched four modes near a 7.0 GHz shift with the largest-real-part
#    selection, because the sixteen-mode search at 7.5 GHz did not finish on that
#    mesh inside the time it was given, and that row never completed.
# 5. Replotted the mode spectrum and localisation ratios and the field map for the
#    selected port-free mode. This is a port-free result and no convergence
#    result: one mesh is solved and the second never returned.
# 6. Ran stage 2 on its own 4 µm / 0.4 µm edge mesh with the boundary mode
#    analysis steps and both numeric TEM ports kept: 659682 elements, four modes
#    searched near a 7.0 GHz shift with `eigwhich="lr"`. The meander-localised
#    loaded mode comes back at 7.326615894917222 GHz with an imaginary part of
#    +534576.3883 Hz, giving a loaded eigen ratio $f_r/(2|f''|)$ of 6852.73055,
#    and a meander-to-feed 95th percentile field ratio of 2.823515744. The field
#    export's complex-frequency annotation matches that mode, so the ported map
#    belongs to it. This loaded mode is a different object from the port-free
#    7.292084525308305 GHz row and the two are never merged.
# 7. Drove the window with an AWE curve over 95 reconstructed rows and direct
#    solves at the two flanks, the centre, and the AWE minimum. The direct points
#    are -0.169357 dB at 7.321615895 GHz, -17.311564 dB at 7.326615895 GHz,
#    -0.002041 dB at 7.331615895 GHz, and -24.192816 dB at 7.32650951194 GHz.
#    The direct minimum is 24.02346 dB below the lower flank, all four direct
#    points agree with the fitted curve within 0.208157 dB, and the direct
#    two-port power sums run from 0.999939 to 1.000030. The fitted curve's own
#    power sums run from 0.999923 to 1.002911, and that 0.291 percent surplus
#    sits on fitted rows rather than on solved ones. The curve's 95 rows are
#    interpolation throughout and are never called independent solves.
#
# What the page establishes is a scripted, reproducible path from layout through
# meshing to a port-free localised eigenmode and then to a ported eigenmode with a
# driven notch verified at four directly solved frequencies. The ported
# eigenvalue, the localisation ratios, and the notch depth all come from a single
# edge mesh, so this is not a converged model and not a device prediction.
#
# ### Limitations
#
# - **Not mesh independent.** Every ported number comes from one edge mesh, pinned
#   at 4 µm / 0.4 µm on the meander edges, and the port-free series has one solved
#   mesh with its finer one never completing. No frequency, localisation ratio, or
#   quality factor here is shown to be mesh independent, and tighter ported
#   eigen-only meshes at 3 µm / 0.3 µm and 2 µm / 0.2 µm are solving without
#   results recorded here.
# - **Not an experimentally validated prediction.** These are simulated
#   S-parameters for a PEC model with a lengthened feedline, on one mesh. Nothing
#   here has been compared against a measurement, and the agreement of four direct
#   solves with one fitted curve is an internal consistency check, not validation.
# - Metal is PEC: no surface resistance and no kinetic inductance, so conductor
#   loss and the kinetic-inductance frequency shift are both missing. A measured
#   quality factor cannot be predicted from this model.
# - The driven curve is AWE interpolation. Only the two flanks, the centre, and
#   the AWE minimum are solved points, 95 rows come from the rational fit, so the
#   curve's shape between those points is a fit and is not a set of independent
#   solves.
# - No quality factor is verified from the notch width. The 6852.73055 figure is
#   the loaded eigenvalue's own damping ratio, which carries the port loading and
#   the numerical error of one mesh; a $Q$ from the fitted linewidth is not
#   independently checked.
# - The notch is required at the directly solved AWE minimum, not at the ported
#   eigenfrequency. A loaded mode can sit off the eigenfrequency, so a centre that
#   is not a notch is not by itself a failure; the minimum is what is tested. Here
#   the centre sits 17.311564 dB down while the minimum is 24.192816 dB down, so
#   the true minimum is indeed off the centre.
# - Two-port power balance is a check, not a guarantee: it only sees the two
#   simulated ports, so a deficit means power left through a channel the record
#   does not model, and the run is then reported unverified.
# - The port-free rows and the ported mode do not search the same modes: sixteen
#   near a 7.5 GHz shift with COMSOL's default selection, four near a 7.0 GHz
#   shift with `eigwhich="lr"` on a different edge mesh. A mode outside either span
#   would not be found, and the selection rule decides which of the modes inside
#   it come back and in what order, so a mode is carried across a change of search
#   only by its field localisation.
# - The feed extension is not the reference device, so the solved coupling is
#   not the reference coupling.
# - The outer boundary is still PEC. A mode that moves with the enclosure would
#   look stable on the meander mesh alone, so neither stage tests the enclosure.
#
# ### Next steps
#
# - **Read the tighter ported eigen-only meshes when they land.** The 3 µm / 0.3 µm
#   and 2 µm / 0.2 µm rows are solving now and carry no result on this page. Check
#   by field localisation that each selects the same physical mode as the
#   4 µm / 0.4 µm row, then read the numerical deltas. Until then the 7.3266 GHz
#   frequency and the 6852.73055 ratio are one-mesh numbers.
# - **Verify a quality factor from the notch width, not from the damping.** Once
#   more than one ported mesh is in hand, fit the linewidth where the direct solves
#   actually constrain it and check that the width stops moving, so any $Q$ comes
#   from solved points rather than from the fitted curve.
# - **Test the enclosure.** Replace the four non-port outer PEC walls with
#   scattering boundaries and check the selected mode survives at the same
#   frequency.
# - **Replace PEC** with a surface-impedance or transition boundary condition
#   using the superconductor's surface resistance and kinetic inductance for a
#   realistic $Q$.
# - Compare the ported section against the QPDK analytical and SAX models in
#   {doc}`/notebooks/all_models` as a sanity check on the coupling.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
