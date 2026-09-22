# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Palace Eigenmode Simulation of a Transmon with a Readout Resonator
#
# This notebook demonstrates a full-wave eigenmode simulation of a **double-pad
# transmon qubit** capacitively coupled to a **quarter-wave CPW readout
# resonator** and read through a **probeline**, using
# [Palace](https://awslabs.github.io/palace/), an open-source parallel 3-D
# finite-element electromagnetic solver, driven through
# [gsim](https://gdsfactory.github.io/gsim/). It is the open-source counterpart
# of the {doc}`hfss_eigenmode_resonator` notebook: same physics, no commercial
# solver license.
#
# The workflow goes from layout to nearby qubit and readout eigenmodes, with an
# analytical CPW estimate as a cross-check:
#
# ````{only} html
# ```{mermaid}
# flowchart TB
#     A["Layout<br>(qpdk cells)"]
#     B["FEM regions<br>(subtractive mask to explicit conductor)"]
#     C["Analytical estimate<br>(CPW cross-section)"]
#     D["Eigenmode setup and mesh<br>(gsim, Gmsh)"]
#     E["Palace solve<br>(cluster)"]
#     F["Eigenmodes vs. estimate"]
#     G["Check participation, fields, and mesh sensitivity"]
#     A --> B --> D --> E --> F --> G
#     C --> F
# ```
# ````
#
# Eigenmode analysis is the workhorse behind quantitative qubit design: the
# linearized eigenfrequencies and eigenfields of the Josephson circuit, with the
# junction treated as a lumped inductor, determine the system Hamiltonian once
# completed with the junction's energy participation
# {cite:p}`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`.
# Here we stop at the classical part: eigenfrequencies and quality factors.
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
# The `models` extra includes gsim. Palace itself is external; mesh generation
# and the solve are shown as commands for a cluster. The results are embedded
# below so the analysis runs without Palace.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# The results shown below were produced with Palace on an HPC cluster; the
# solver data is embedded in hidden cells so the rendered docs show the
# analysis, not the raw tables.

# %% tags=["hide-input", "hide-output"]
import importlib
import sys
from functools import partial
from pathlib import Path

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
import gdsfactory as gf
import klayout.db as kdb
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from qpdk import PDK, logger
from qpdk.cells import (
    double_pad_transmon_with_bbox,
    plate_capacitor_single,
    straight,
    transmon_with_resonator_and_probeline,
)
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.resonator import resonator_frequency
from qpdk.simulation import FEM_LAYERS, to_fem_regions
from qpdk.tech import LAYER

PDK.activate()

for style in (Path("docs/qpdk.mplstyle"), Path("../docs/qpdk.mplstyle"), "qpdk"):
    try:
        plt.style.use(style)
        break
    except OSError:
        continue

# This notebook ships with its outputs committed, so keep figures raster:
# matplotlib's SVG embeds a DOCTYPE URL that link checkers follow, and a dense
# field map as SVG is megabytes for no visual gain.
matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# %% [markdown]
# ## Simulation layout
#
# The device is the
# {func}`~qpdk.cells.transmon_with_resonator_and_probeline` cell: a double-pad
# transmon, a meandered quarter-wave CPW resonator coupled to one pad, and a
# probeline feed coupled along the readout line. It exposes exactly
# the ports a 3-D solve needs:
#
# - ``junction``: the position of the Josephson junction between the two qubit
#   pads, which becomes the lumped inductive port.
# - ``coupling_o1`` / ``coupling_o2``: the two ends of the probeline feed, which
#   become 50 Ω CPW lumped ports.
#
# The cell's ``resonator_length`` includes the meander, probeline coupling
# section, and qubit-side route. The conductor is shorted at the far end of the
# meander and open at the plate facing the qubit; its full routed length sets
# the first quarter-wave estimate below.
#
# Two details need care when the cell is used standalone:
#
# - The meander's far end must reach ground. The qubit-side plate stays open,
#   outside the qubit etch, and separated from the pads by a capacitive gap.
# - Bare probeline port ends touch ground at their end faces, so the wrapper
#   extends the gap etch past both feed ends.
#
# A ``SIM_AREA`` rectangle around the device bounds the simulation domain.

# %%
# The cell uses this total to size the meander after laying out the separate
# coupling section and qubit-side arm.
RESONATOR_LENGTH = 6900.0  # µm
RESONATOR_MEANDERS = 5
RESONATOR_MEANDER_START = (-1100.0, -1200.0)  # µm
QUBIT_PAD_SIZE = (55.0, 110.0)  # µm
QUBIT_PAD_GAP = 15.0  # µm
QUBIT_BBOX_EXTENSION = 20.0  # µm
COUPLER_WIDTH = 10.0  # µm
COUPLER_LENGTH = 120.0  # µm
COUPLER_OFFSET = (-26.0, 0.0)  # µm
COUPLER_ETCH_MARGIN = 2.0  # µm


@gf.cell
def qubit_resonator_sim_component(
    resonator_length: float = RESONATOR_LENGTH,
    meanders: int = RESONATOR_MEANDERS,
) -> gf.Component:
    """Transmon, resonator and probeline wrapped with a simulation area.

    Args:
        resonator_length: Layout length budget used to size the meander in µm.
        meanders: Number of meander sections.

    Returns:
        Component with the simulation layout and ports
        ``junction``, ``coupling_o1``, ``coupling_o2``.
    """
    c = gf.Component()

    ref = c << transmon_with_resonator_and_probeline(
        qubit=partial(
            double_pad_transmon_with_bbox,
            pad_size=QUBIT_PAD_SIZE,
            pad_gap=QUBIT_PAD_GAP,
            bbox_extension=QUBIT_BBOX_EXTENSION,
        ),
        coupler=partial(
            plate_capacitor_single,
            width=COUPLER_WIDTH,
            length=COUPLER_LENGTH,
            etch_bbox_margin=COUPLER_ETCH_MARGIN,
        ),
        coupler_offset=COUPLER_OFFSET,
        resonator_length=resonator_length,
        resonator_meanders=meanders,
        resonator_meander_start=RESONATOR_MEANDER_START,
        # The open end is the plate at the qubit; this bare meander end meets ground.
        resonator_open_end=False,
        qubit_rotation=90,
    )
    c.add_ports(ref.ports)
    routed = ref.cell.info["length"]
    assert abs(routed - resonator_length) < 1.0, (
        f"cell routed {routed} µm, expected {resonator_length} µm"
    )
    c.info["routed_length"] = routed
    meander_ref = next(
        inst
        for inst in ref.cell.insts
        if inst.cell.info.get("resonator_type") == "quarter_wave"
    )
    c.info["meander_length"] = meander_ref.cell.info["length"]
    qubit_ref, coupler_ref = list(ref.cell.insts)[:2]
    assert (
        coupler_ref.dbbox().top - COUPLER_ETCH_MARGIN
        < qubit_ref.dbbox().bottom
        < coupler_ref.dbbox().top
    ), "open-end metal must stay outside the qubit etch while the etch windows meet"
    assert (
        qubit_ref.ports["left_pad"].y - (coupler_ref.dbbox().top - COUPLER_ETCH_MARGIN)
        >= 20
    ), "the open end is too close to the qubit pad"
    assert meander_ref.dbbox().right < coupler_ref.ports["o1"].x - 30, (
        "the meander crosses the qubit-side route"
    )
    # Open the probeline feed ends: extend the gap etch past both port faces.
    for name in ("coupling_o1", "coupling_o2"):
        ext = c << straight(length=10.0, cross_section="etch")
        ext.connect("o1", ref.ports[name], allow_layer_mismatch=True)

    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(100, 100))
    return c


component = qubit_resonator_sim_component()
component.plot()
logger.info(f"Ports: {[p.name for p in component.ports]}")

# %% [markdown]
# The close view shows the 10 µm wide readout plate below the qubit's etched
# box. Its own etch joins that box, but the metal stays outside it and at least
# 20 µm from the qubit pad. The trace continues down to the meander, whose
# opposite end terminates on the ground plane.

# %%
qubit_close_view = gf.Component()
qubit_close_view << component
qubit_close_view.trim(left=-150, bottom=-180, right=150, top=120, flatten=True)
qubit_close_view.plot(show_labels=False, show_ruler=False)

# %% [markdown]
# At the opposite end, the meander etch stops at the trace end. The conductor
# therefore joins the surrounding ground plane there, providing the short of
# the quarter-wave resonator.

# %%
device_ref = next(iter(component.insts))
meander_ref = next(
    inst
    for inst in device_ref.cell.insts
    if inst.cell.info.get("resonator_type") == "quarter_wave"
)
short_x, short_y = meander_ref.ports["o2"].center
short_close_view = gf.Component()
short_close_view << component
short_close_view.trim(
    left=short_x - 80,
    bottom=short_y - 65,
    right=short_x + 80,
    top=short_y + 65,
    flatten=True,
)
short_close_view.plot(show_labels=False, show_ruler=False)

# %% [markdown]
# ## From etch layers to simulation regions
#
# qpdk draws metal subtractively, a volumetric solver needs explicit conductor
# and dielectric bodies, and {func}`~qpdk.simulation.to_fem_regions` converts
# between the two with ``SIM_AREA - (M1_ETCH - M1_DRAW)``. The conversion, and
# what it does to an additive shape sitting inside an etched gap, is worked
# through with a picture in {ref}`subtractive-to-positive`.

# %%
etched = to_fem_regions(component)
metal = kdb.Region(
    etched.kdb_cell.begin_shapes_rec(
        etched.kdb_cell.layout().layer(*FEM_LAYERS["SUPERCONDUCTOR"])
    )
).merged()
assert len(metal) == 4, "expected the grounded readout, feed, and two qubit pads"
islands = list(metal)


def _metal_island(x: float, y: float) -> int:
    point = kdb.Point(round(x / etched.kcl.dbu), round(y / etched.kcl.dbu))
    matches = [index for index, polygon in enumerate(islands) if polygon.inside(point)]
    assert len(matches) == 1, f"expected one conductor at ({x}, {y}), got {matches}"
    return matches[0]


ground = _metal_island(component.dbbox().left + 5, component.dbbox().top - 5)
readout = _metal_island(*component.ports["o1"].center)
feed1 = component.ports["coupling_o1"].center
feed2 = component.ports["coupling_o2"].center
feed = _metal_island((feed1[0] + feed2[0]) / 2, (feed1[1] + feed2[1]) / 2)
pad_centres = (QUBIT_PAD_SIZE[0] + QUBIT_PAD_GAP) / 2
pad1 = _metal_island(0, -pad_centres)
pad2 = _metal_island(0, pad_centres)
assert readout == ground, "the far end of the readout must meet ground"
assert len({readout, feed, pad1, pad2}) == 4, (
    "the readout must stay separate from the feed and both qubit pads"
)

# The cell's ``junction`` port is a *placement* marker: it points along the
# junction wire, which the transmon cells build perpendicular to the pad-to-pad
# axis (``left_pad_inner`` and ``right_pad_inner`` face each other across the
# gap, while the junction port sits at the gap centre at 90° to them). A lumped
# port has to bridge pad to pad, so rotate by that 90°. Deriving it this way
# rather than hard-coding an angle keeps it correct for any ``qubit_rotation``.
etched.ports["junction"].orientation = (
    etched.ports["junction"].orientation + 90.0
) % 360.0

etched.plot()

# %% [markdown]
# ## Analytical resonator estimate
#
# Before meshing anything, the CPW cross-section model gives a first estimate
# of the readout frequency. A quarter-wave resonator of length $L$
# resonates at $f_r = v_p / 4L$ with phase velocity
# $v_p = c_0 / \sqrt{\varepsilon_{\mathrm{eff}}}$, where
# $\varepsilon_{\mathrm{eff}}$ comes from the conformal-mapping CPW model
# in {func}`~qpdk.models.cpw_parameters`
# {cite:p}`simonsCoplanarWaveguideCircuits2001,m.pozarMicrowaveEngineering2012`.

# %%
RESONATOR_CROSS_SECTION = "cpw"

width, gap = get_cpw_dimensions(RESONATOR_CROSS_SECTION)
epsilon_eff, z0 = cpw_parameters(width, gap)
logger.info(f"CPW width {width} µm, gap {gap} µm")
logger.info(
    f"ε_eff = {float(np.real(epsilon_eff)):.3f}, Z0 = {float(np.real(z0)):.1f} Ω"
)

routed_length = component.info["routed_length"]
analytical_freq = resonator_frequency(
    length=routed_length,
    cross_section=RESONATOR_CROSS_SECTION,
    is_quarter_wave=True,
)
logger.info(
    f"Routed length {routed_length:.1f} µm, analytical quarter-wave estimate "
    f"{analytical_freq / 1e9:.4f} GHz"
)

# %% [markdown]
# This is an estimate for the full routed quarter-wave line with this CPW cross-section.
# The grounded meander end, probeline coupling, turns, and open-end fringing
# are all part of the actual 3-D layout. The eigenmode solve measures
# their combined frequency offset, which can then calibrate a nearby length
# change. The sign of that offset has to come from the solve.

# %% [markdown]
# ## Eigenmode setup with gsim
#
# gsim turns the converted layout into a 3-D model: a `LayerStack` assigns each
# GDS layer a material and a z-extent, and `EigenmodeSim` configures the ports
# and the eigenmode search. The setup runs here; the fine mesh is generated on
# a cluster.

# %%
from gsim.palace import EigenmodeSim

from qpdk.simulation import single_chip_stack

stack = single_chip_stack(substrate_thickness=500.0, vacuum_thickness=500.0)
sim = EigenmodeSim()
sim.set_geometry(etched)
sim.set_stack(stack)
sim.set_numerical(order=2, solver_type="MUMPS")

# The junction port spans the pad gap and overlaps both pads; resistance=0
# keeps the linearized Josephson inductor lossless.
sim.add_port(
    "junction",
    layer="SUPERCONDUCTOR",
    length=QUBIT_PAD_GAP + 10.0,
    inductance=10e-9,
    resistance=0.0,
)
sim.add_cpw_port(
    "coupling_o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)
sim.add_cpw_port(
    "coupling_o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)
sim.set_eigenmode(target=4e9, num_modes=2, save=2)
sim.set_output_dir("./sim_palace_qubit_resonator")

# %% [markdown]
# Generate the 0.70 µm near-conductor mesh and Palace config on a cluster:
#
# ```python
# import json
#
# sim.mesh(preset="default", refined_mesh_size=0.70, auto_size=False)
# sim.write_config()
# config_path = Path("sim_palace_qubit_resonator/config.json")
# config = json.loads(config_path.read_text())
# config["Solver"]["Linear"].update({
#     "Tol": 1e-5,
#     "MaxIts": 20,
#     "MGMaxLevels": 2,
#     "PCMatReal": False,
#     "PCMatShifted": False,
#     "ComplexCoarseSolve": True,
#     "ColumnOrdering": "ParMETIS",
# })
# config["Solver"]["Eigenmode"]["Tol"] = 1e-5
# config_path.write_text(json.dumps(config, indent=2))
# ```
#
# The mesh is graded around metal edges and the junction gap. At this
# resolution a full-domain wireframe obscures the device, so the saved field
# slices below show the spatial mode patterns instead.

# %% [markdown]
# A few points worth noting:
#
# - The substrate carries the qpdk microwave-silicon loss tangent
#   ($\tan\delta = 2.7 \times 10^{-6}$
#   {cite:p}`checchinMeasurementLowTemperatureLoss2022`), which sets the
#   dielectric-limited quality factors seen below.
# - The junction port makes the qubit participate in the linear spectrum: the
#   capacitance between the pads comes from the field solution, the inductive
#   part from the port. Deep in the transmon regime this fixes the qubit-like
#   mode to roughly $f_q \approx 1 / (2\pi\sqrt{L_J C_\Sigma})$
#   {cite:p}`kochChargeinsensitiveQubitDesign2007a,krantzQuantumEngineersGuide2019`.
# - Meshing invokes [Gmsh](https://gmsh.info/); on some cluster login nodes it
#   needs `libGLU` at import time (for example `module load mesa-glu`).
# - `sim.run()` submits the job to
#   [GDSFactory+](https://gdsfactory.com/), a commercial hosted service. It is
#   entirely optional and nothing here needs it: `sim.mesh()` +
#   `sim.write_config()` write a self-contained directory (`palace.msh` +
#   `config.json`) that any local Palace binary executes directly, which is
#   what the next section does.

# %% [markdown]
# ## Solving with Palace
#
# `sim.mesh()` + `sim.write_config()` leave a self-contained directory
# (`palace.msh` + `config.json`) that any Palace build can execute, so the
# solve is just a matter of putting it somewhere with enough cores. For a
# single-node allocation:
#
# ```bash
# cd <simulation directory>  # config.json + palace.msh
# apptainer exec --cleanenv /path/to/palace.sif mpirun -np 24 palace-x86_64.bin config.json
# ```
#
# The saved result used eight MPI ranks across two regular Triton nodes, with
# Palace launched inside the same Apptainer image on every node.
#
# `refined_mesh_size` is the knob that matters for field quality: it sets the
# element size near the conductors, where the field varies fastest. The
# comparison below shows the size of the discretisation error before
# interpreting the mode spacing. The saved 0.70 µm run uses Palace's two-level
# multigrid preconditioner with MUMPS only on the coarse level and 32 GiB
# requested per node.
#
# Palace writes one row per mode to `output/palace/eig.csv`: mode index, the
# real and imaginary parts of the eigenfrequency in GHz, and the quality
# factor. Two companion tables make mode identification quantitative,
# `output/palace/port-EPR.csv` for the energy-participation ratio of each
# lumped port in each mode and `output/palace/port-Q.csv` for the external
# quality factor per port. The solved rows are embedded below so the analysis
# runs without the output files.

# %% [markdown]
# ## Results: eigenmodes versus the analytical estimate
#
# The meander, feed coupling section, and qubit-side route form a single
# quarter-wave line with a 6900 µm routed length. The two returned modes
# should be the qubit-like mode (set by the junction inductance and pad
# capacitance) and the dressed readout resonance near the analytical estimate.

# %% tags=["hide-input"]
# Palace output of the 6900 µm run, embedded so the analysis
# runs without the cluster: eig.csv (mode index, Re{f} (GHz), Q),
# port-EPR.csv (junction participation p[1]) and port-Q.csv (external quality
# factors of the two probeline ports).
eig_m = np.arange(1, 3)
eig_f_re = np.array([4.067865919254, 4.269576094402])
eig_q = np.array([7.443209161838e04, 3.161356339253e03])
junction_epr = np.array([-1.119464678462e-01, -4.310283878317e-03])
q_ext_feed1 = np.array([1.820860918095e05, 6.411466172706e03])
q_ext_feed2 = np.array([1.818846892897e05, 6.406471679158e03])
q_ext_feed = 1 / (1 / q_ext_feed1 + 1 / q_ext_feed2)

logger.info(f"{'m':>3} {'Re{f} (GHz)':>12} {'Q_raw':>10} {'p_J':>11} {'Q_ext':>11}")
for m, f_re, q, p1, qe in zip(
    eig_m, eig_f_re, eig_q, junction_epr, q_ext_feed, strict=True
):
    logger.info(f"{m:>3} {f_re:>12.4f} {q:>10.2e} {p1:>11.2e} {qe:>11.2e}")

# %% [markdown]
# Three features identify the modes, and they must agree:
#
# 1. **Frequency**: the qubit and readout form a nearby pair; the analytical
#    quarter-wave estimate is only a starting point for the readout.
# 2. **Junction participation**: `port-EPR.csv` reports the fraction
#    $p_{m,J}$ of each mode's energy stored in the junction port
#    {cite:p}`minevEnergyParticipationQuantization2021`. The qubit-like mode
#    carries by far the largest participation, while a resonator-like mode
#    keeps most of its inductive energy in the CPW trace, much less.
# 3. **Port dissipation**: `port-Q.csv` gives the external quality factor of
#    each mode through the 50 Ω probeline ports. The readout resonance couples
#    to the feedline and gets a finite $Q_{\mathrm{ext}}$ (with both feed
#    ports adding in parallel, $1/Q = 1/Q_{\mathrm{diel}} + \sum 1/Q_{\mathrm{ext}}$).
#    The nearby qubit-like mode couples more weakly to the feedline, though its
#    external loss need not be negligible at this small detuning. Silicon loss
#    alone would give $Q_{\mathrm{diel}} \approx 1 / (p_{\mathrm{diel}} \tan\delta)$
#    with $\tan\delta = 2.7 \times 10^{-6}$.
#
# No single feature is sufficient on its own: a heavily feed-coupled spurious
# mode can have the lowest Q of the spectrum without being the readout
# resonance. Participation-based identification is the standard in
# quantitative qubit design: the same EPRs that label the modes also turn the
# eigenmode data into a quantum Hamiltonian with anharmonicities and couplings
# {cite:p}`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`.

# %%
analytical_ghz = analytical_freq / 1e9
# A higher harmonic can carry more junction energy than the fundamental.
JUNCTION_PARTICIPATION_FLOOR = 0.05
_candidates = np.flatnonzero(np.abs(junction_epr) >= JUNCTION_PARTICIPATION_FLOOR)
i_qubit = int(_candidates[np.argmin(eig_f_re[_candidates])])
_readout_candidates = np.flatnonzero(np.abs(junction_epr) < 1e-2)
assert _readout_candidates.size, "no mode with low junction participation"
i_res = int(
    _readout_candidates[
        np.argmin(np.abs(eig_f_re[_readout_candidates] - analytical_ghz))
    ]
)
resonator_fem_ghz = eig_f_re[i_res]
resonator_q = eig_q[i_res]
assert abs(junction_epr[i_res]) < 1e-2, "identified readout mode stores junction energy"
logger.info(f"Analytical estimate:          {analytical_ghz:.4f} GHz")
logger.info(
    f"Palace readout resonance:     {resonator_fem_ghz:.4f} GHz "
    f"(mode {eig_m[i_res]}, p_J = {junction_epr[i_res]:.2e})"
)
logger.info(
    f"Relative offset:              "
    f"{(resonator_fem_ghz - analytical_ghz) / analytical_ghz:+.2%}"
)
logger.info(
    f"Readout raw quality factor:   {resonator_q:.3e} "
    f"(Q_ext = {q_ext_feed[i_res]:.3e} through both feed ports)"
)
logger.info(
    f"Qubit-like mode:              {eig_f_re[i_qubit]:.4f} GHz "
    f"(mode {eig_m[i_qubit]}, p_J = {junction_epr[i_qubit]:.2e})"
)
logger.info(
    f"Qubit-readout separation:    "
    f"{abs(resonator_fem_ghz - eig_f_re[i_qubit]) * 1e3:.1f} MHz"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].scatter(eig_f_re, eig_m, color="0.55", s=36)
axes[0].scatter(eig_f_re[i_qubit], eig_m[i_qubit], color="darkorange", s=80)
axes[0].scatter(eig_f_re[i_res], eig_m[i_res], color="steelblue", s=80)
axes[0].axvline(
    analytical_ghz,
    color="crimson",
    linestyle="--",
    label=f"analytical $f_r$ = {analytical_ghz:.3f} GHz",
)
axes[0].set_xlabel("frequency (GHz)")
axes[0].set_ylabel("Palace mode")
axes[0].set_yticks(eig_m)
axes[0].set_xlim(min(analytical_ghz, eig_f_re.min()) - 0.2, eig_f_re.max() + 0.2)
axes[0].legend(loc="center left")
axes[1].scatter(
    [eig_f_re[i_qubit], resonator_fem_ghz],
    [1, 0],
    color=["darkorange", "steelblue"],
    s=90,
)
axes[1].set_yticks([0, 1], ["readout", "qubit"])
axes[1].set_xlim(
    min(eig_f_re[i_qubit], resonator_fem_ghz) - 0.12,
    max(eig_f_re[i_qubit], resonator_fem_ghz) + 0.12,
)
axes[1].set_ylim(-0.4, 1.4)
axes[1].set_xlabel("frequency (GHz)")
axes[1].set_title(
    f"separation: {abs(resonator_fem_ghz - eig_f_re[i_qubit]) * 1e3:.0f} MHz"
)
fig.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ### Mesh sensitivity
#
# Re-solving the 55 × 110 µm pads and 6900 µm resonator budget with a smaller
# near-conductor element size changes the mode spacing:
#
# | mesh size (µm) | qubit (mode, GHz) | readout (mode, GHz) | separation (MHz) |
# | ---: | ---: | ---: | ---: |
# | 0.90 | 1, 4.1486 | 2, 4.2736 | 125.0 |
# | 0.70 | 1, 4.0679 | 2, 4.2696 | 201.7 |
#
# Palace reports backward eigenpair errors of $5.7\times10^{-9}$ and
# $1.3\times10^{-8}$ on the saved solve. Those test the algebraic solve, not
# the geometry resolution. Its absolute estimates are $3.1\times10^{-3}$ and
# $7.3\times10^{-3}$.
#
# The readout shifts by 4.0 MHz, while the qubit-like frequency shifts by
# 80.7 MHz (1.9%). Its junction participation changes from 0.1086 to 0.1119.
# The 201.7 MHz separation is the finer-mesh result, but its remaining qubit
# frequency error is not bounded by this two-point check. This is an
# illustrative operating point, not a converged design tolerance.
#
# The qubit mode's raw complex-frequency $Q$ changes from
# $2.85\times10^4$ to $7.44\times10^4$, so it is not a reliable lifetime
# estimate at this resolution. The readout's raw $Q\approx3.16\times10^3$
# is close to its combined feed-port $Q_{\mathrm{ext}}\approx3.20\times10^3$.

# %% [markdown]
# The FEM readout differs from the ideal-line estimate because the model includes
# the grounded end, probeline coupling, meander corners, and open-end fringing.
# The solve determines the sign and size of that offset.
#
# Note on signs: Palace reports the junction participation with the sign of
# the port current relative to the mode's field orientation, so the same
# physical mode can come out negative in one solve and positive in another.
# Identification therefore compares magnitudes, never the raw signed value.
#
# Magnitude alone is not enough either: a higher harmonic can have a larger
# junction participation. The qubit mode is the lowest one above the stated
# participation floor.
#
# The qubit-like frequency depends on the junction inductance and the pad
# capacitance the field solution sees. Inverting
# $f_q = 1/(2\pi\sqrt{L_J C})$ with $L_J = 10$ nH gives the capacitance a pure
# LC mode would need at this frequency. It is an effective capacitance for the
# coupled mode, not a separate extraction of the pad shunt capacitance
# $C_\Sigma$. A Hamiltonian extraction needs the full energy-participation
# treatment {cite:p}`minevEnergyParticipationQuantization2021`.

# %%
c_mode = 1 / (2 * np.pi * eig_f_re[i_qubit] * 1e9) ** 2 / 10e-9
logger.info(
    f"Effective mode capacitance: {c_mode * 1e15:.0f} fF"
    " (from f_q and L_J, assuming a pure LC mode)"
)

# %% [markdown]
# A single solve provides a first length correction: with
# $\delta = 1 - f_{\mathrm{FEM}} / f_{\mathrm{est}}$ the corrected model
# $f \approx (1 - \delta)\, v_p / 4L$ inverts to
# $L^{*} = (1 - \delta)\, v_p / (4 f_{\mathrm{target}})$, which is enough to
# estimate the routed length for a target readout frequency. The offset need
# not remain constant when the layout changes, so the retuned geometry needs
# its own solve.

# %% [markdown]
# ## Field visualization
#
# Palace saved the first two eigenfields under
# `output/palace/paraview/eigenmode/Cycle00000{1,2}/data.pvtu`. The images
# sample their electric-field magnitude 1 µm above the metal plane at model
# coordinate $z=501$ µm. The qubit view zooms in on the pads; the readout view
# shows the full meander. [Palace's visualization guide](https://awslabs.github.io/palace/stable/guide/postprocessing/#visualization-field-spaces-and-interface-values)
# explains that its electric-field export preserves values local to each mesh
# element. Only the field tangent must agree across a tetrahedron face; the
# normal component can jump. A 2-D cut through that 3-D mesh can therefore show
# triangular seams, especially on a logarithmic colour scale. Matplotlib is
# displaying those sampled values, not inventing the facets. We clip the lower
# range and apply a small display filter:
# $\sigma=0.4$ µm for the pads and 1.5 µm for the full meander. Sampling closer
# to the metal keeps the physical edge fields sharper than a higher 3-D slice.
# Frequencies and participations use the unsmoothed solver output. Gray areas
# lie outside the solved domain. The images are embedded in the notebook
# outputs so they survive without rerunning Palace. Eigenmode amplitudes have
# arbitrary normalisation, so compare each map's spatial pattern rather than
# its absolute colour scale. The cross-section below marks the slice above the
# qubit pads; the metal segments are schematic.

# %%
fig, ax = plt.subplots(figsize=(6.5, 2.4))
ax.axhspan(496, 500, color="steelblue", alpha=0.12)
ax.axhspan(500, 506, color="darkorange", alpha=0.08)
ax.hlines([500, 500], [-62.5, 7.5], [-7.5, 62.5], color="0.2", linewidth=5)
ax.axhline(501, color="crimson", linestyle="--", label="field sampling plane")
ax.text(-72, 497.2, "silicon")
ax.text(-72, 503.5, "air")
ax.set_xlim(-75, 75)
ax.set_ylim(496, 506)
ax.set_xlabel("y across the qubit pad gap (µm)")
ax.set_ylabel("z (µm)")
ax.grid(False)
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()
plt.close(fig)

# %%
field_root = Path("sim_palace_qubit_resonator/output/palace/paraview/eigenmode")


def _plot_eigenfield(mode: int, extent: tuple[float, float, float, float]) -> None:
    pv = importlib.import_module("pyvista")

    x = np.linspace(extent[0], extent[1], 700)
    y = np.linspace(extent[2], extent[3], 700)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack((
        grid_x.ravel(),
        grid_y.ravel(),
        np.full(grid_x.size, 501.0),
    ))
    path = field_root / f"Cycle{mode:06d}" / "data.pvtu"
    sampled = pv.PolyData(points).sample(pv.read(path))
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool).reshape(
        grid_x.shape
    )
    e_real = np.asarray(sampled.point_data["E_real"], dtype=float)
    e_imag = np.asarray(sampled.point_data["E_imag"], dtype=float)
    magnitude = np.sqrt(np.sum(e_real**2 + e_imag**2, axis=1)).reshape(grid_x.shape)
    positive = magnitude[valid & (magnitude > 0)]
    vmax = float(np.percentile(positive, 99.5))
    vmin = max(vmax / 150, float(np.percentile(positive, 25)))
    sigma_um = 0.4 if x[-1] - x[0] < 500 else 1.5
    sigma = (sigma_um / (y[1] - y[0]), sigma_um / (x[1] - x[0]))
    log_field = np.log(np.where(valid, np.maximum(magnitude, vmin), vmin))
    weight = valid.astype(float)
    smooth_weight = gaussian_filter(weight, sigma)
    shown = np.where(
        valid,
        np.exp(
            gaussian_filter(log_field * weight, sigma)
            / np.maximum(smooth_weight, 1e-12)
        ),
        np.nan,
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.94")

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(
        shown,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="none",
    )
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Mode {mode}: {eig_f_re[mode - 1]:.3f} GHz")
    fig.colorbar(image, ax=ax, label=r"$|\mathbf{E}|$ (V/m)")
    plt.show()
    plt.close(fig)


if field_root.exists():
    _plot_eigenfield(int(eig_m[i_qubit]), (-120, 120, -120, 120))
    _plot_eigenfield(int(eig_m[i_res]), (-1400, 200, -1320, 200))
else:
    logger.info("Palace field files are needed to regenerate the embedded plots")

# %% [markdown]
#
# The qubit-like mode concentrates field across the pad gap and along the pad
# edges. The readout mode follows the meander as a quarter-wave standing wave.
# Their small detuning leaves some field on both parts of the layout, consistent
# with the nonzero junction and feed-port participation of each mode.

# %% [markdown]
# ## Summary
#
# A qpdk layout goes to a full-wave eigenmode spectrum with no commercial
# solver license. Enlarged qubit pads and a longer resonator bring their modes
# close near 4 GHz. Junction participation, feed-port coupling and the saved
# electric fields distinguish the two. The CPW estimate gives a starting
# length; the Palace solve checks the retuned layout.
#
# The SAX circuit models in {doc}`all_models` and {doc}`circuit_simulation_demo`
# miss the geometry-dependent effects captured here (fringing, loading, meander
# parasitics) but cost almost nothing, so the two complement each other:
# circuit models to explore, FEM to verify and calibrate. The next step,
# extracting the Hamiltonian, anharmonicities and dispersive couplings from
# these eigenmodes via energy participation
# {cite:p}`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`,
# is what tools such as [pyEPR](https://github.com/zlatko-minev/pyEPR) do.

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
