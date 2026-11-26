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
#   "gsim @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///

# %% [markdown]
# # Palace Driven Simulation of a Coupled CPW Resonator
#
# This notebook runs a driven (S-parameter) [Palace](https://awslabs.github.io/palace/)
# simulation of a quarter-wave CPW resonator side-coupled to a feedline, through
# [gsim](https://gdsfactory.github.io/gsim/). The solve sweeps $S_{21}$ across the
# resonance, estimates the loaded quality factor $Q_\text{L}$ two ways, and maps
# $|E|$ on the conductor plane. It ports the *QPDK Resonator* template of the
# [GDSFactory+ simulation templates](https://gdsfactory.com/plus/v2/simulation-templates/notebooks/fem_palace_qpdk_resonator/)
# to the qpdk cells and the {func}`~qpdk.simulation.to_fem_regions` conversion.
#
# ````{only} html
# ```{mermaid}
# flowchart TB
#     A["Layout<br>(resonator_coupled, S-bend feeds)"]
#     B["FEM regions<br>(M1_ETCH mask to conductor)"]
#     C["Sapphire / aluminium / vacuum stack"]
#     D["Driven setup and mesh<br>(gsim, Gmsh)"]
#     E["Palace solve<br>(GDSFactory+ cloud, guarded)"]
#     F["S21 sweep"]
#     G["Q from the 3 dB bandwidth and a Lorentzian fit"]
#     H["|E| on the conductor plane"]
#     A --> B --> C --> D --> E
#     E --> F --> G
#     E --> H
# ```
# ````
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra:
#
# ```bash
# uv add "qpdk[models]"
# # or, from a checkout of this repository:
# uv sync --python 3.12 --extra models --group palace
# # or with pip:
# pip install "qpdk[models]"
# ```
#
# The layout, stack, mesh and configuration cells also need
# [gsim](https://gdsfactory.github.io/gsim/). The `palace` dependency group
# installs the pinned version used here. The standalone script and Colab setup
# install that same revision. These cells run on a laptop and need no account.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# ```{warning}
# The Palace solve runs on the GDSFactory+ cloud and needs a GDSFactory+ account,
# so the docs build does not execute this notebook. The cloud cell is guarded:
# it runs only when you set `QPDK_RUN_RESONATOR_CLOUD=1` before starting the
# kernel (see {ref}`resonator-cloud-run`). Without the flag, the $S_{21}$, $Q$ and
# field cells log that no solver result is available and skip their analysis.
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
        "gsim @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
    ])

# %% tags=["hide-input", "hide-output"]
import os
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace import DrivenSim, SParams, load_sparams, plot_fields_2d
from matplotlib import font_manager
from scipy.optimize import curve_fit

import qpdk.colormaps  # ruff: ignore[unused-import]
from qpdk import PDK, logger
from qpdk.cells import resonator_coupled, straight
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.config import PATH
from qpdk.models.constants import c_0
from qpdk.models.cpw import cpw_epsilon_eff, get_cpw_dimensions
from qpdk.simulation import FEM_LAYERS, to_fem_regions
from qpdk.simulation.palace_run import verify_port_connectivity
from qpdk.tech import LAYER, route_bundle_sbend_cpw

PDK.activate()

for style in (PATH.repo / "docs" / "qpdk.mplstyle", "qpdk"):
    try:
        plt.style.use(style)
    except OSError:
        continue
    break

for font_path in (PATH.repo / "build" / "docs-fonts").glob("*"):
    if font_path.suffix.lower() in {".otf", ".ttf"}:
        font_manager.fontManager.addfont(str(font_path))

installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [
    name
    for name in ("Inter", "Outfit", "DejaVu Sans", "Helvetica", "Arial")
    if name in installed_fonts
] + ["sans-serif"]

# Keep figures raster: a field map as SVG is megabytes for no visual gain.
matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# %% [markdown]
# ## Sizing the resonator
#
# The resonator is open at the coupled end and shorted to ground at the far end,
# so it is a quarter-wave line: $f_\text{r} = v_\text{p} / 4L$ with
# $v_\text{p} = c_0 / \sqrt{\varepsilon_\text{eff}}$
# {cite:p}`simonsCoplanarWaveguideCircuits2001,m.pozarMicrowaveEngineering2012`.
# The template targets a resonance near 7.78 GHz. That value comes from the
# template and has not been re-derived here.
#
# The substrate is sapphire, which gsim stores as a uniaxial tensor
# (in-plane $\varepsilon_\perp$, normal $\varepsilon_\parallel$). The conformal-mapping
# model ``qpdk.models.cpw.cpw_epsilon_eff`` takes one scalar, so the
# estimate uses the geometric mean $\sqrt{\varepsilon_\perp \varepsilon_\parallel}$,
# a heuristic for an anisotropic substrate. Palace solves the full tensor, and
# the sweep window below is wide enough to absorb the difference.

# %%
F_TARGET = 7.78e9  # Hz, resonance of the source template
SUBSTRATE_THICKNESS = 500.0  # µm
VACUUM_THICKNESS = 500.0  # µm

sapphire = MATERIALS_DB["sapphire"].to_dict()
eps_in_plane, _, eps_normal = sapphire["permittivity"]
eps_substrate = float(np.sqrt(eps_in_plane * eps_normal))

cpw_width, cpw_gap = get_cpw_dimensions("cpw")
eps_eff = float(
    cpw_epsilon_eff(
        cpw_width * 1e-6, cpw_gap * 1e-6, SUBSTRATE_THICKNESS * 1e-6, eps_substrate
    )
)
RESONATOR_LENGTH = c_0 / np.sqrt(eps_eff) / (4 * F_TARGET) * 1e6  # µm

logger.info(f"CPW width {cpw_width} µm, gap {cpw_gap} µm")
logger.info(f"sapphire ε = ({eps_in_plane}, {eps_in_plane}, {eps_normal})")
logger.info(f"ε_eff = {eps_eff:.3f}, quarter-wave length {RESONATOR_LENGTH:.1f} µm")

# %% [markdown]
# ## Simulation layout
#
# The layout is the {func}`~qpdk.cells.resonator_coupled` cell: a meandering
# quarter-wave resonator below a short coupling section of the feedline. Two
# S-bends (``qpdk.tech.route_bundle_sbend_cpw``) lift each feed end clear of the
# resonator, and the CPW cross-section carries airbridges every 150 µm
# (``qpdk.cells.airbridge.cpw_with_airbridges``). Each feed ends in a
# straight stub with an etched cap, where the CPW port goes. A ``SIM_AREA``
# rectangle bounds the substrate and vacuum volumes.
#
# - The far end of the meander is not etched, so the resonator trace meets the
#   ground plane there. This is the short.
# - ``feed_o1`` and ``feed_o2`` are the two ports.
# - The resonator, the S-bends and the feed stubs all sit inside one box, which
#   keeps the model compact.

# %%
RESONATOR_MEANDERS = 4
COUPLING_LENGTH = 200.0  # µm
COUPLING_GAP = 10.0  # µm, edge to edge
FEED_DX = 150.0  # µm, horizontal run of each S-bend
FEED_DY = 60.0  # µm, vertical offset of each S-bend
FEED_STUB = 100.0  # µm, straight CPW at each port
ETCH_CAP = 10.0  # µm of etch behind each port
GROUND_MARGIN = 300.0  # µm of ground plane around the layout
AIRBRIDGE_SPACING = 150.0  # µm
AIRBRIDGE_PADDING = 20.0  # µm


@gf.cell
def resonator_sim_component(
    length: float = RESONATOR_LENGTH,
    meanders: int = RESONATOR_MEANDERS,
    coupling_gap: float = COUPLING_GAP,
    ground_margin: float = GROUND_MARGIN,
) -> gf.Component:
    """Coupled quarter-wave resonator with S-bend feeds and a simulation area.

    Args:
        length: Resonator length in µm.
        meanders: Number of meander sections.
        coupling_gap: Edge-to-edge gap between resonator and feedline in µm.
        ground_margin: Ground clearance around the layout in µm.

    Returns:
        Component with the layout and ports ``feed_o1``, ``feed_o2``.
    """
    c = gf.Component()
    res = c << resonator_coupled(
        length=length,
        meanders=meanders,
        open_start=True,
        open_end=False,
        coupling_straight_length=COUPLING_LENGTH,
        coupling_gap=coupling_gap,
    )
    left_coupling = res.ports["coupling_o1"]
    right_coupling = res.ports["coupling_o2"]

    left = c << straight(length=FEED_STUB, cross_section="cpw")
    right = c << straight(length=FEED_STUB, cross_section="cpw")
    left.dmove((
        left_coupling.x - FEED_DX - FEED_STUB,
        left_coupling.y + FEED_DY,
    ))
    right.dmove((right_coupling.x + FEED_DX, right_coupling.y + FEED_DY))

    route_bundle_sbend_cpw(
        c,
        [left.ports["o2"], right.ports["o1"]],
        [left_coupling, right_coupling],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=AIRBRIDGE_SPACING, airbridge_padding=AIRBRIDGE_PADDING
        ),
    )

    for name, port in (("feed_o1", left.ports["o1"]), ("feed_o2", right.ports["o2"])):
        c.add_port(name, port=port)
        cap = c << straight(length=ETCH_CAP, cross_section="etch")
        cap.connect("o1", port, allow_layer_mismatch=True)

    c.kdb_cell.shapes(gf.get_layer(LAYER.SIM_AREA)).insert(
        c.bbox().enlarged(ground_margin, ground_margin)
    )
    short = res.ports["resonator_o2"]
    c.info["resonator_length"] = res.cell.info["length"]
    c.info["short_x"] = short.x
    c.info["short_y"] = short.y
    return c


component = resonator_sim_component()
logger.info(f"ports: {[p.name for p in component.ports]}")
logger.info(
    f"resonator length {component.info['resonator_length']:.1f} µm; "
    f"simulation footprint {component.dbbox().width():.0f} × "
    f"{component.dbbox().height():.0f} µm"
)
component.plot()

# %% [markdown]
# ## From the etch mask to conductor regions
#
# qpdk draws metal subtractively: ``M1_ETCH`` marks where metal is removed.
# A volumetric solver needs the conductor itself, so
# {func}`~qpdk.simulation.to_fem_regions` builds it as
# ``SIM_AREA - (M1_ETCH - M1_DRAW)`` and copies the sapphire and vacuum regions
# onto gsim's layer numbers. {ref}`subtractive-to-positive` shows the conversion
# with a picture.
#
# The converter rejects fabrication layers other than the metal and junction
# masks, so the airbridge layers (``AB_DRAW`` and ``AB_VIA``) are removed first.
# **The solve therefore does not include the airbridges**: they appear in the
# layout, but the model has no bridge metal. The ground plane is one connected
# island, so the model does not need the bridges to join the two ground halves,
# but it also cannot show what they do to slot modes.
#
# After the conversion, checks guard the geometry: the merged metal has exactly
# two islands (ground and feedline), both feed ports sit on the feedline, and the
# resonator's far end sits on the ground island.

# %%
without_bridges = component.remove_layers([LAYER.AB_DRAW, LAYER.AB_VIA])
regions = to_fem_regions(without_bridges)

metal = kdb.Region(
    regions.kdb_cell.begin_shapes_rec(
        regions.kdb_cell.layout().layer(*FEM_LAYERS["SUPERCONDUCTOR"])
    )
).merged()
islands = list(metal)
assert len(islands) == 2, f"expected ground and feedline, got {len(islands)} islands"


def metal_island(x: float, y: float) -> int:
    """Return the index of the single metal island containing the point (µm)."""
    point = kdb.Point(round(x / regions.kcl.dbu), round(y / regions.kcl.dbu))
    matches = [i for i, polygon in enumerate(islands) if polygon.inside(point)]
    assert len(matches) == 1, f"expected one conductor at ({x}, {y}), got {matches}"
    return matches[0]


bounds = component.dbbox()
ground = metal_island(bounds.left + 5, bounds.top - 5)
feed1 = component.ports["feed_o1"]
feed2 = component.ports["feed_o2"]
feedline = metal_island(feed1.x + 5, feed1.y)
assert metal_island(feed2.x - 5, feed2.y) == feedline, "feeds must share one conductor"
assert feedline != ground, "the feedline must not touch ground"
# The trace ends at the short: a point 5 µm back from its end must be ground.
assert (
    metal_island(component.info["short_x"] - 5, component.info["short_y"]) == ground
), "the far end of the resonator must meet ground"

regions.plot()

# %% [markdown]
# ## Layer stack
#
# The stack has three layers, as in the template: 500 µm of sapphire, a
# zero-thickness aluminium sheet on top of it, and 500 µm of vacuum. The
# material entries come from gsim's database. The conductor has no thickness, so
# Palace treats it as a perfect electric conductor: its conductivity only
# names the layer and adds no loss or kinetic inductance. All loss in the
# model comes from the sapphire loss tangent and from the open boundaries.

# %%
stack = LayerStack(pdk_name="qpdk")
stack.layers["SUBSTRATE"] = Layer(
    name="SUBSTRATE",
    gds_layer=FEM_LAYERS["SUBSTRATE"],
    zmin=0.0,
    zmax=SUBSTRATE_THICKNESS,
    thickness=SUBSTRATE_THICKNESS,
    material="sapphire",
    layer_type="dielectric",
)
stack.layers["SUPERCONDUCTOR"] = Layer(
    name="SUPERCONDUCTOR",
    gds_layer=FEM_LAYERS["SUPERCONDUCTOR"],
    zmin=SUBSTRATE_THICKNESS,
    zmax=SUBSTRATE_THICKNESS,
    thickness=0,
    material="aluminum",
    layer_type="conductor",
)
stack.layers["VACUUM"] = Layer(
    name="VACUUM",
    gds_layer=FEM_LAYERS["VACUUM"],
    zmin=SUBSTRATE_THICKNESS,
    zmax=SUBSTRATE_THICKNESS + VACUUM_THICKNESS,
    thickness=VACUUM_THICKNESS,
    material="vacuum",
    layer_type="dielectric",
)
stack.materials = {
    name: MATERIALS_DB[name].to_dict() for name in ("sapphire", "aluminum", "vacuum")
}
for layer in stack.layers.values():
    logger.info(
        f"{layer.name:>14}: z {layer.zmin:.0f} to {layer.zmax:.0f} µm, {layer.material}"
    )

# %% [markdown]
# ## Driven setup
#
# [`DrivenSim`](https://gdsfactory.github.io/gsim/) takes the regions, the
# stack and two 50 Ω lumped CPW ports. The port geometry matches the ``cpw``
# cross-section: a 10 µm centre conductor between 6 µm gaps.
#
# The sweep covers 7.5 to 8.1 GHz at 0.25 MHz spacing with Palace's adaptive
# frequency sampling, which builds a reduced-order model from a few full solves
# and interpolates the rest. The window is wide because the analytical estimate
# above is approximate, and the spacing is fine so that several points fall
# inside a resonance linewidth of a few megahertz. These sampling settings are
# a starting point that has not been checked against a converged solve: if the
# solved $S_{21}$ has spikes, tighten ``ADAPTIVE_TOL`` or raise
# ``ADAPTIVE_MAX_SAMPLES``. Only ``feed_o1`` is excited, which is enough for
# $S_{21}$ and halves the cost of the sweep.
#
# Palace also saves the fields at one frequency, ``F_FIELD``. The resonance is
# not known before the sweep, so ``F_FIELD`` defaults to the target. The field
# cell below checks how far the saved frequency is from the fitted resonance.

# %%
F_MIN, F_MAX = 7.5e9, 8.1e9  # Hz
NUM_POINTS = 2401  # 0.25 MHz spacing
ADAPTIVE_TOL = 1e-3
ADAPTIVE_MAX_SAMPLES = 40
F_FIELD = F_TARGET  # Hz, field snapshot; see the field cell below
ELEMENT_ORDER = 2
NEAR_MESH = 5.0  # µm, element size near the conductors
SIM_DIR = Path("build/sim_palace_resonator")

sim = DrivenSim()
sim.set_geometry(regions)
sim.set_stack(stack)
sim.set_solver(order=ELEMENT_ORDER)
for port_name in ("feed_o1", "feed_o2"):
    sim.add_cpw_port(
        port_name,
        layer="SUPERCONDUCTOR",
        s_width=cpw_width,
        gap_width=cpw_gap,
        length=5.0,
        excited=port_name == "feed_o1",
    )
sim.set_driven(
    fmin=F_MIN,
    fmax=F_MAX,
    num_points=NUM_POINTS,
    adaptive_tol=ADAPTIVE_TOL,
    adaptive_max_samples=ADAPTIVE_MAX_SAMPLES,
    save_fields_at=[F_FIELD],
)
sim.set_output_dir(SIM_DIR)
validation = sim.validate_config()
assert validation.valid, validation.errors

# %% [markdown]
# ### Mesh and configuration
#
# Meshing and writing the configuration run locally. The coarse preset with a
# 5 µm element size near the metal puts about one element across each 6 µm gap;
# the second-order field basis carries the rest. Refine ``NEAR_MESH`` for a
# converged study. {func}`~qpdk.simulation.palace_run.verify_port_connectivity`
# then checks that both ports share mesh nodes with the conductor.

# %%
sim.mesh(preset="coarse", refined_mesh_size=NEAR_MESH, auto_size=False, verbose=False)
config_path = sim.write_config()
shared_nodes = verify_port_connectivity(SIM_DIR)
logger.info(f"config: {config_path}")
logger.info(f"nodes shared between each port and the conductor: {shared_nodes}")
sim.print_mesh_stats()

# %% [markdown]
# (resonator-cloud-run)=
# ## Running Palace on the GDSFactory+ cloud
#
# One job produces everything the next two sections need. The cell below uploads
# the mesh and configuration, starts the job, waits, and downloads the results.
# It runs only when `QPDK_RUN_RESONATOR_CLOUD=1`. Set it before launching the
# kernel, with a GDSFactory+ account signed in as described in the
# [gsim documentation](https://gdsfactory.github.io/gsim/):
#
# ```bash
# QPDK_RUN_RESONATOR_CLOUD=1 jupyter lab
# ```
#
# The $S_{21}$ sweep is saved to ``s_params.npz`` in the simulation directory, so
# later sessions load it without a new job.

# %%
RUN_CLOUD = os.environ.get("QPDK_RUN_RESONATOR_CLOUD") == "1"
S_PARAMS_FILE = SIM_DIR / "s_params.npz"

results = None
sparams = None
if RUN_CLOUD:
    job_id = sim.upload()
    logger.info(f"uploaded job {job_id}")
    sim.start()
    parsed = sim.wait_for_results(parent_dir=SIM_DIR / "cloud")
    # The cloud client returns the parsed S-parameters; the downloaded run
    # folder (with the field output) is the newest sim-data-* under cloud/.
    sparams = parsed if isinstance(parsed, SParams) else load_sparams(parsed)
    results = max((SIM_DIR / "cloud").glob("sim-data-*"), key=lambda p: p.stat().st_mtime) / "output"
    sparams.save_npz(S_PARAMS_FILE)
elif S_PARAMS_FILE.is_file():
    sparams = SParams.from_file(S_PARAMS_FILE)
    logger.info(f"loaded saved S-parameters from {S_PARAMS_FILE}")
else:
    logger.info(
        "No solver result: set QPDK_RUN_RESONATOR_CLOUD=1 to run the cloud job "
        "(needs a GDSFactory+ account). The analysis cells below are skipped."
    )

# %% [markdown]
# ## $S_{21}$ across the resonance
#
# A resonator side-coupled to a through line notches the transmission at
# resonance. For a symmetric notch with a constant background the dip in power
# is a Lorentzian:
#
# ```{math}
# \lvert S_{21}\rvert^2 = B\left[1 - \frac{D}{1 + 4 Q_\text{L}^2 (f - f_\text{r})^2 / f_\text{r}^2}\right],
# ```
#
# with baseline $B$, depth $D$, resonance $f_\text{r}$ and loaded quality factor
# $Q_\text{L}$. The two estimates of $Q_\text{L}$ are:
#
# 1. **3 dB bandwidth.** The width of the dip at the level halfway between its
#    floor and the baseline, $Q_\text{L} = f_\text{r} / \Delta f$. For a dip that
#    reaches zero this level is the usual $-3$ dB point. Between the same two
#    levels the width equals the Lorentzian full width at half depth for any
#    depth. The estimate is limited by the frequency spacing and needs several
#    points across $\Delta f$.
# 2. **Lorentzian fit** of the equation above to $\lvert S_{21}\rvert^2$, which does
#    not depend on the spacing.
#
# The fit also separates the loss channels. With $a = 1 - \sqrt{1 - D}$ the
# coupling and internal quality factors are $Q_\text{c} = Q_\text{L}/a$ and
# $Q_\text{i} = Q_\text{L}/\sqrt{1 - D}$, so that
# $1/Q_\text{L} = 1/Q_\text{c} + 1/Q_\text{i}$ {cite:p}`besedinQualityFactorTransmission2018a`.
# This assumes a real $Q_\text{c}$, so a feedline mismatch that skews the dip
# would bias $Q_\text{c}$ and $Q_\text{i}$. Open boundaries and the sapphire
# loss tangent both feed $Q_\text{i}$.


# %%
def half_depth_bandwidth(
    f: np.ndarray, power: np.ndarray
) -> tuple[float, float, float, int]:
    """Measure a power dip at the level halfway between its floor and baseline.

    Args:
        f: Frequencies in ascending order.
        power: Power transmission at ``f``.

    Returns:
        The frequency of the minimum, the width of the dip at half depth in the
        units of ``f`` (NaN if the dip leaves the band), the baseline power and
        the number of points inside the dip.
    """
    edge = max(len(f) // 20, 1)
    baseline = float(np.median(np.r_[power[:edge], power[-edge:]]))
    i0 = int(np.argmin(power))
    level = (baseline + power[i0]) / 2
    lo = i0
    while lo > 0 and power[lo] < level:
        lo -= 1
    hi = i0
    while hi < len(f) - 1 and power[hi] < level:
        hi += 1
    if power[lo] < level or power[hi] < level:
        return float(f[i0]), float("nan"), baseline, 0
    # np.interp needs ascending x values, so list the lower point first on this side.
    f_lo = np.interp(level, [power[lo + 1], power[lo]], [f[lo + 1], f[lo]])
    f_hi = np.interp(level, [power[hi - 1], power[hi]], [f[hi - 1], f[hi]])
    return float(f[i0]), float(f_hi - f_lo), baseline, hi - lo - 1


def lorentzian_dip(
    f: np.ndarray, f_r: float, q_loaded: float, depth: float, baseline: float
) -> np.ndarray:
    """Return the Lorentzian power dip of a side-coupled resonator."""
    return baseline * (1 - depth / (1 + (2 * q_loaded * (f - f_r) / f_r) ** 2))


def fit_dip(f: np.ndarray, power: np.ndarray) -> tuple[float, float, float, float]:
    """Fit :func:`lorentzian_dip`, starting from the half-depth bandwidth.

    Args:
        f: Frequencies in ascending order.
        power: Power transmission at ``f``.

    Returns:
        The fitted resonance, loaded quality factor, depth and baseline.
    """
    f_r, width, baseline, _ = half_depth_bandwidth(f, power)
    q0 = f_r / width if np.isfinite(width) else f_r / (10 * np.median(np.diff(f)))
    depth0 = float(np.clip(1 - power.min() / baseline, 1e-3, 1.0))
    popt, _ = curve_fit(
        lorentzian_dip,
        f,
        power,
        p0=[f_r, q0, depth0, baseline],
        bounds=([f.min(), 1.0, 0.0, 0.0], [f.max(), 1e9, 1.0, 2.0]),
        x_scale=[f_r / q0, q0, 0.1, 0.1],
    )
    return tuple(float(p) for p in popt)


if sparams is not None:
    freq = np.asarray(sparams.freq)  # GHz
    s21 = np.asarray(sparams.s21.complex)
    power = np.abs(s21) ** 2

    f_bw, width, baseline, points = half_depth_bandwidth(freq, power)
    f_fit, q_fit, depth, baseline_fit = fit_dip(freq, power)
    q_bw = f_bw / width
    q_c = q_fit / (1 - np.sqrt(1 - depth))
    q_i = q_fit / np.sqrt(1 - depth)

    logger.info(f"analytical estimate      f_r = {F_TARGET / 1e9:.4f} GHz")
    logger.info(f"fitted resonance         f_r = {f_fit:.4f} GHz")
    logger.info(
        f"half-depth bandwidth     Δf = {width * 1e3:.3f} MHz, Q_L = {q_bw:.0f}"
    )
    logger.info(f"Lorentzian fit           Q_L = {q_fit:.0f}, depth {depth:.3f}")
    logger.info(f"fit: Q_c = {q_c:.0f}, Q_i = {q_i:.0f}")
    if points < 5:
        logger.warning(
            f"only {points} sweep points inside the dip: the bandwidth estimate "
            "is limited by the frequency spacing; prefer the fit"
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq, 10 * np.log10(power), ".", ms=3, label="Palace")
    ax.plot(
        freq,
        10 * np.log10(lorentzian_dip(freq, f_fit, q_fit, depth, baseline_fit)),
        label=rf"fit, $Q_\text{{L}}$ = {q_fit:.0f}",
    )
    ax.axvspan(
        f_bw - width / 2,
        f_bw + width / 2,
        alpha=0.15,
        label=rf"half-depth width, $Q_\text{{L}}$ = {q_bw:.0f}",
    )
    ax.set(xlabel="frequency (GHz)", ylabel=r"$|S_{21}|^2$ (dB)")
    ax.legend()
    plt.show()
    plt.close(fig)
else:
    logger.info("No solver result: skipping the S21 analysis.")

# %% [markdown]
# ## $\lvert E\rvert$ on the conductor plane
#
# Palace wrote the fields at ``F_FIELD`` during the sweep. The slice sits
# 0.5 µm below the conductor sheet, because a cut exactly on the sheet is
# degenerate in the mesh. The field is strongest in the gaps of the feedline and
# in the resonator gap. Only a map at the resonance shows the resonator
# excited. The cell therefore reports how far ``F_FIELD`` is from the fitted
# resonance, measured in half-linewidths $f_\text{r} / 2Q_\text{L}$.
#
# If the saved frequency is more than about one half-linewidth from the
# resonance, set ``F_FIELD`` to the fitted $f_\text{r}$ and repeat the job with
# `set_driven(f=F_FIELD)`, which needs one frequency instead of a sweep.

# %%
if results is not None and sparams is not None:
    offset = abs(F_FIELD / 1e9 - f_fit) / (f_fit / (2 * q_fit))
    logger.info(
        f"saved field at {F_FIELD / 1e9:.4f} GHz, {offset:.1f} half-linewidths "
        f"from the fitted resonance at {f_fit:.4f} GHz"
    )
    if offset > 1:
        logger.warning("the saved field is not at resonance: see the note above")
    plot_fields_2d(
        results,
        field="E_real",
        normal="z",
        origin=SUBSTRATE_THICKNESS - 0.5,
        title=f"|E| at {F_FIELD / 1e9:.4f} GHz",
        cmap="turbo",
        mode="static",
    )
else:
    logger.info("No solver result in this session: skipping the field map.")

# %% [markdown]
# ## Summary
#
# - {func}`~qpdk.cells.resonator_coupled`, S-bend feeds with airbridges and a
#   ``SIM_AREA`` box give a compact layout, and
#   {func}`~qpdk.simulation.to_fem_regions` turns its ``M1_ETCH`` mask into
#   conductor geometry. The airbridges are dropped in the conversion.
# - A sapphire / aluminium / vacuum stack and two CPW ports configure
#   `gsim.palace.DrivenSim`. The mesh, configuration and port check run locally.
# - One guarded cloud job sweeps $S_{21}$. $Q_\text{L}$ comes from the half-depth
#   bandwidth and from a Lorentzian fit, which also splits $Q_\text{c}$ and
#   $Q_\text{i}$.
# - Not modelled: airbridge metal, conductor loss and kinetic inductance (the
#   aluminium is a perfect conductor), and radiation beyond the absorbing
#   boundary.

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
