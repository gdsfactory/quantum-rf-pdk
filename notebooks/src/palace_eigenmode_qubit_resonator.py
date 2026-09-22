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
# The workflow runs layout to calibrated geometry, with the analytical CPW
# model as the cross-check at every step:
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
#     G["Calibrate, retune to target"]
#     A --> B --> D --> E --> F --> G
#     C --> F
#     G -.->|re-solve to verify| D
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
# The simulation cells additionally need the [Palace](https://awslabs.github.io/palace/)
# solver itself, which is external to qpdk; the cells that invoke it are fenced,
# and the Palace results are embedded below so the analysis runs anywhere.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# The results shown below were produced with Palace on an HPC cluster; the
# solver data is embedded in hidden cells so the rendered docs show the
# analysis, not the raw tables.

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
import gdsfactory as gf
import klayout.db as kdb
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np

from qpdk import PDK, logger
from qpdk.cells import straight, transmon_with_resonator_and_probeline
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.resonator import resonator_frequency
from qpdk.simulation import to_fem_regions
from qpdk.tech import LAYER

PDK.activate()

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
# probeline feed coupled to the resonator's shorted end. It exposes exactly
# the ports a 3-D solve needs:
#
# - ``junction``: the position of the Josephson junction between the two qubit
#   pads, which becomes the lumped inductive port.
# - ``coupling_o1`` / ``coupling_o2``: the two ends of the probeline feed, which
#   become 50 Ω CPW lumped ports.
#
# The cell's ``resonator_length`` is the *total* routed length: meander plus
# the probeline coupling section plus the feed arm up to the qubit. All three
# are one continuous quarter-wave line between the short and the open end, so
# that total is what sets the resonance, and it is the length the analytical
# estimate below is evaluated at. (Sizing the *drawn meander* to a target
# instead, and comparing against an estimate for the meander alone, detunes the
# resonator by the length of the coupling section and feed arm.)
#
# One detail needs care when the cell is used standalone, and the wrapper below
# patches it:
#
# - The cell attaches the probeline coupling arm at the resonator's start port,
#   so the start never merges into the ground plane as the quarter-wave short
#   requires, and the bare probeline port ends touch the ground plane at their
#   end faces. The wrapper bridges the start bend to ground (realizing the
#   short) and extends the gap etch past both port faces (opening the feed
#   ends).
#
# A ``SIM_AREA`` rectangle around the device bounds the simulation domain.

# %%
# Total routed length of the quarter-wave line, short to open. This is the
# length the resonance is set by, and so the length the analytical estimate
# below is evaluated at: the meander, the probeline coupling section and the
# feed arm up to the qubit are all part of the same resonator.
RESONATOR_LENGTH = 5000.0  # µm
RESONATOR_MEANDERS = 5


@gf.cell
def qubit_resonator_sim_component(
    resonator_length: float = RESONATOR_LENGTH,
    meanders: int = RESONATOR_MEANDERS,
) -> gf.Component:
    """Transmon, resonator and probeline wrapped with a simulation area.

    Args:
        resonator_length: Total routed length of the quarter-wave line in µm.
        meanders: Number of meander sections.

    Returns:
        Component with the simulation layout and ports
        ``junction``, ``coupling_o1``, ``coupling_o2``.
    """
    c = gf.Component()

    ref = c << transmon_with_resonator_and_probeline(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=resonator_length,
        resonator_meanders=meanders,
        qubit_rotation=90,
    )
    c.add_ports(ref.ports)
    # The analytical estimate is evaluated at `resonator_length`, so the cell
    # must actually route that length; meander quantisation could otherwise
    # reintroduce the drift this notebook exists to avoid.
    routed = ref.cell.info["length"]
    assert abs(routed - resonator_length) < 1.0, (
        f"cell routed {routed} µm, expected {resonator_length} µm"
    )

    # Quarter-wave short: bridge the resonator's start bend to the ground
    # plane just west of it (see the notes above).
    c.kdb_cell.shapes(LAYER.M1_DRAW).insert(kdb.DBox(-975.0, -1201.0, -920.0, -1197.0))

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
# ## From etch layers to simulation regions
#
# qpdk draws metal subtractively, a volumetric solver needs explicit conductor
# and dielectric bodies, and {func}`~qpdk.simulation.to_fem_regions` converts
# between the two with ``SIM_AREA - (M1_ETCH - M1_DRAW)``. The conversion, and
# what it does to an additive shape sitting inside an etched gap, is worked
# through with a picture in {ref}`subtractive-to-positive`.

# %%
etched = to_fem_regions(component)

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

analytical_freq = resonator_frequency(
    length=RESONATOR_LENGTH,
    cross_section=RESONATOR_CROSS_SECTION,
    is_quarter_wave=True,
)
logger.info(f"Analytical quarter-wave estimate: {analytical_freq / 1e9:.4f} GHz")

# %% [markdown]
# This estimate systematically runs *high* relative to a full-wave solve:
# fringing fields extend the open end electrically, the resonator is
# capacitively loaded by the qubit pad and the probeline coupling section, and
# every meander bend adds parasitic capacitance. How far the resonance lands
# below the estimate depends on the geometry, from a few tenths of a percent
# to a few percent, and the offset is nearly constant for small geometry
# changes. The eigenmode solve measures that offset, and the retuning section
# below reuses it as a correction.

# %% [markdown]
# ## Eigenmode setup with gsim
#
# gsim turns the converted layout into a 3-D model: a `LayerStack` assigns each
# GDS layer a material and a z-extent, and `EigenmodeSim` configures the ports
# and the eigenmode search. gsim is part of the `models` extra, so the setup
# below runs as-is; only the Palace solver itself stays external.

# %%
from gsim.palace import EigenmodeSim

from qpdk.simulation import single_chip_stack

# 500 µm of microwave silicon and 500 µm of air above it; the substrate
# uses the qpdk material properties (eps_r = 11.45, tan d = 2.7e-6), so the
# FEM models the same chip as the analytical CPW models of the previous
# section.
stack = single_chip_stack(substrate_thickness=500.0, vacuum_thickness=500.0)

sim = EigenmodeSim()
sim.set_geometry(etched)
sim.set_stack(stack)
# A direct factorization makes each shift-and-invert apply cheap, which
# matters on this roughly 1.2M-unknown first-order model.
sim.set_numerical(order=1, solver_type="MUMPS")

# Josephson junction as a linear lumped inductor, L_J = 10 nH, a typical
# transmon value. resistance=0 overrides the R = 50 Ω that gsim's default
# port impedance would emit: the linearized junction is purely reactive, so
# it shifts the qubit-like mode but adds no dissipation, the same convention
# as Palace's own transmon example. The port length spans the 15 µm pad gap
# and overlaps both pads; a shorter rectangle would sit entirely in the
# vacuum gap and couple nothing.
sim.add_port(
    "junction",
    layer="SUPERCONDUCTOR",
    length=25.0,
    inductance=10e-9,
    resistance=0.0,
)

# Probeline feeds as 50 Ω CPW lumped ports, one per end.
sim.add_cpw_port(
    "coupling_o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)
sim.add_cpw_port(
    "coupling_o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)

# Target sets the lower edge of the eigenvalue search: the solver returns
# the N lowest modes above it. Keep it well below the expected band so the
# qubit-like mode (a few GHz below the readout) is included.
sim.set_eigenmode(target=2e9, num_modes=10)

sim.set_output_dir("./sim_palace_qubit_resonator")
sim.mesh(preset="default", refined_mesh_size=1.5, auto_size=False)
sim.write_config()

# %% [markdown]
# The mesh is graded, which is the whole game in FEM cost: elements cluster on
# the metal edges and the junction gap, where the fields vary fastest, and
# coarsen into the substrate bulk and the vacuum above, where they do not.

# %% tags=["hide-input"]
import gsim.viz

gsim.viz.plot_mesh(
    "./sim_palace_qubit_resonator/palace.msh", style="wireframe", mode="static"
)

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
# solve is just a matter of putting it somewhere with enough cores:
#
# ```bash
# cd <simulation directory>          # config.json + palace.msh
# mpirun -np 16 palace config.json   # Palace parallelizes with MPI
# ```
#
# `refined_mesh_size` is the knob that matters for field quality: it sets the
# element size near the conductors, where the field varies fastest. At 1.5 µm
# the model is about 900k first-order tetrahedra (190k nodes), roughly four
# times the default preset, which is what keeps the field maps below smooth
# rather than faceted. That size is also why the direct MUMPS factorization
# configured above is worth it, and why this one runs on a cluster: a few
# dozen ranks finish the multi-mode solve in minutes. Nothing about the
# workflow changes with the machine.
#
# Palace writes one row per mode to `output/palace/eig.csv`: mode index, the
# real and imaginary parts of the eigenfrequency in GHz, and the quality
# factor. Two companion tables make mode identification quantitative,
# `output/palace/port-EPR.csv` for the energy-participation ratio of each
# lumped port in each mode and `output/palace/port-Q.csv` for the external
# quality factor per port. The rows of the two runs used here are embedded
# below so the analysis runs without the output files.

# %% [markdown]
# ## Results: eigenmodes versus the analytical estimate
#
# The first run is the baseline geometry ($L = 5000$ µm). Of the modes
# returned, we expect to find the qubit-like mode (set by the junction
# inductance and the pad capacitance), the dressed readout resonance near the
# analytical estimate, and higher modes of the feed and meander.

# %% tags=["hide-input"]
# Palace output of the baseline run (L = 5000 µm), embedded so the analysis
# runs without the cluster: eig.csv (mode index, Re{f} (GHz), Q),
# port-EPR.csv (junction participation p[1]) and port-Q.csv (external quality
# factor of the first probeline port).
eig_m = np.arange(1, 11)
eig_f_re = np.array([
    2.146766,
    6.024779,
    10.331287,
    13.092997,
    15.265808,
    19.560870,
    23.142545,
    25.610757,
    30.337317,
    30.522750,
])
eig_q = np.array([
    3.970334e05,
    5.644419e03,
    5.742178e03,
    5.089113e05,
    2.328773e02,
    5.685445e02,
    3.010411e05,
    4.201441e03,
    2.494473e05,
    4.138391e05,
])
junction_epr = np.array([
    1.854248e-01,
    -1.536861e-05,
    -4.332592e-06,
    -3.190882e-06,
    4.472507e-06,
    7.845207e-08,
    -7.600931e-05,
    5.747219e-06,
    -6.635207e-01,
    1.457564e-02,
])
q_ext_feed1 = np.array([
    5.796148e07,
    1.094878e04,
    1.306650e04,
    2.840414e08,
    5.261121e02,
    1.260579e03,
    4.829728e09,
    7.776510e03,
    1.396160e06,
    5.362442e07,
])

logger.info(f"{'m':>3} {'Re{f} (GHz)':>12} {'Q':>10} {'p_J':>11} {'Q_ext':>11}")
for m, f_re, q, p1, qe in zip(
    eig_m, eig_f_re, eig_q, junction_epr, q_ext_feed1, strict=True
):
    logger.info(f"{m:>3} {f_re:>12.4f} {q:>10.2e} {p1:>11.2e} {qe:>11.2e}")

# %% [markdown]
# Three features identify the modes, and they must agree:
#
# 1. **Frequency**: the readout resonance lies closest to the analytical
#    quarter-wave estimate.
# 2. **Junction participation**: `port-EPR.csv` reports the fraction
#    $p_{m,J}$ of each mode's energy stored in the junction port
#    {cite:p}`minevEnergyParticipationQuantization2021`. The qubit-like mode
#    carries by far the largest participation, while a resonator-like mode
#    keeps its inductive energy in the CPW trace, orders of magnitude less.
# 3. **Port dissipation**: `port-Q.csv` gives the external quality factor of
#    each mode through the 50 Ω probeline ports. The readout resonance couples
#    to the feedline and gets a finite $Q_{\mathrm{ext}}$ (with both feed
#    ports adding in parallel, $1/Q = 1/Q_{\mathrm{diel}} + \sum 1/Q_{\mathrm{ext}}$); the qubit-like mode barely sees the feedline,
#    so its Q is limited instead by the dielectric loss of the silicon,
#    $Q \approx 1 / (p_{\mathrm{diel}} \tan\delta)$ with
#    $\tan\delta = 2.7 \times 10^{-6}$.
#
# No single feature is sufficient on its own: a heavily feed-coupled spurious
# mode can have the lowest Q of the spectrum without being the readout
# resonance. Participation-based identification is the standard in
# quantitative qubit design: the same EPRs that label the modes also turn the
# eigenmode data into a quantum Hamiltonian with anharmonicities and couplings
# {cite:p}`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`.

# %%
analytical_ghz = analytical_freq / 1e9
i_res = int(np.argmin(np.abs(eig_f_re - analytical_ghz)))
# The qubit mode is the *lowest* mode the junction takes part in, not the one
# with the largest participation. In this spectrum mode 9 at 30.3 GHz carries
# |p_J| = 0.66 against the qubit mode's 0.185, so picking the maximum returns a
# harmonic. The floor separates the two modes the junction actually takes part
# in (0.185 and 0.66) from the rest of the spectrum, whose largest is mode 10
# at 1.5e-02, an order of magnitude below the floor.
JUNCTION_PARTICIPATION_FLOOR = 0.1
_candidates = np.flatnonzero(np.abs(junction_epr) >= JUNCTION_PARTICIPATION_FLOOR)
i_qubit = int(_candidates[np.argmin(eig_f_re[_candidates])])
resonator_fem_ghz = eig_f_re[i_res]
resonator_q = eig_q[i_res]
# The cross-checks from the text: the identified readout mode barely touches
# the junction, yet couples to the feedline (finite Q_ext well below the
# dielectric-limited qubit mode's Q).
assert abs(junction_epr[i_res]) < 1e-2, "identified readout mode stores junction energy"
assert q_ext_feed1[i_res] < 1e6, "identified mode is not coupled to the feedline"
assert resonator_q < eig_q[i_qubit], (
    "readout mode should be lossier than the dielectric-limited qubit mode"
)

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
    f"Readout quality factor:       {resonator_q:.3e} "
    f"(Q_ext = {q_ext_feed1[i_res]:.3e} through the feedline)"
)
logger.info(
    f"Qubit-like mode:              {eig_f_re[i_qubit]:.4f} GHz "
    f"(mode {eig_m[i_qubit]}, p_J = {junction_epr[i_qubit]:.2e})"
)

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["lightsteelblue"] * len(eig_m)
colors[i_qubit] = "darkorange"
colors[i_res] = "steelblue"
ax.barh(
    [f"mode {m}" for m in eig_m],
    eig_f_re,
    color=colors,
    label="Palace eigenmode",
)
ax.axvline(
    analytical_ghz,
    color="crimson",
    linestyle="--",
    label=f"analytical $f_r$ = {analytical_ghz:.3f} GHz",
)
ax.axvline(
    resonator_fem_ghz,
    color="steelblue",
    linestyle=":",
    label=f"FEM resonance = {resonator_fem_ghz:.3f} GHz",
)
ax.set_xlabel("Frequency (GHz)")
ax.legend(loc="lower right")
fig.tight_layout()

# %% [markdown]
# The FEM resonance sits about one percent below the estimate, in the expected
# direction: the analytical model knows nothing about the open-end fringing,
# the qubit and probeline loading, or the meander corners.
#
# Note on signs: Palace reports the junction participation with the sign of
# the port current relative to the mode's field orientation, so the same
# physical mode can come out negative in one solve and positive in another.
# Identification therefore compares magnitudes, never the raw signed value.
#
# Magnitude alone is not enough either. Mode 9 at 30.3 GHz carries
# $|p_J| = 0.66$ against the qubit mode's $0.185$, so taking the largest
# participation returns a harmonic. The qubit mode is the *lowest* mode the
# junction takes part in at all.
#
# The qubit-like mode at 2.15 GHz is the one whose frequency is controlled by
# the junction inductance shunted by the pad capacitance the field solution
# sees: inverting $f_q = 1/(2\pi\sqrt{L_J C_\Sigma})$ with the
# simulated frequency and $L_J = 10$ nH gives the total capacitance
# the mode sees, computed in the next cell. Its junction participation is 0.185
# rather than nearly one because the mode shares its inductive energy with
# the CPW network the pads couple into. Sweeping the junction inductance in
# the lumped port would move this mode as
# $f_q \propto 1/\sqrt{L_J}$, while the readout mode stays put, which is
# the standard numerical check that the port really is the junction
# {cite:p}`minevEnergyParticipationQuantization2021`.

# %% [markdown]
# Inverting $f_q = 1/(2\pi\sqrt{L_J C}) $ for $C$ gives the capacitance a
# *pure* LC mode at this frequency would need. With $p_J = 0.185$ the junction
# holds only about a fifth of the mode's inductive energy, so the number below
# is an effective mode capacitance, not the qubit's shunt capacitance
# $C_\Sigma$; extracting the latter needs the full energy-participation
# treatment {cite:p}`minevEnergyParticipationQuantization2021`.

# %%
c_sigma_qubit = 1 / (2 * np.pi * eig_f_re[i_qubit] * 1e9) ** 2 / 10e-9
logger.info(
    f"Effective mode capacitance: {c_sigma_qubit * 1e15:.0f} fF"
    " (from f_q and L_J, assuming a pure LC mode)"
)

# %% [markdown]
# Because the analytical model tracks the FEM result to within a nearly
# constant offset, a single solve also calibrates it: with
# $\delta = 1 - f_{\mathrm{FEM}} / f_{\mathrm{est}}$ the corrected model
# $f \approx (1 - \delta)\, v_p / 4L$ inverts to
# $L^{*} = (1 - \delta)\, v_p / (4 f_{\mathrm{target}})$, which is enough to
# retune the meander for a target readout frequency. The offset is not perfectly
# constant, though: the coupler and the qubit load the open end, so a retune
# that moves the resonator relative to the pads needs its calibration redone
# against a fresh solve.

# %% [markdown]
# ## Field visualization
#
# `set_eigenmode(..., save=N)` makes Palace write the eigenfields of the first
# $N$ modes as ParaView collections under `output/palace/paraview/`.
# gsim slices them straight onto the mesh with
# [pyvista](https://docs.pyvista.org):
#

# %% tags=["hide-input"]
import os
from pathlib import Path

import matplotlib_inline
import pyvista as pv

# A dense field map is a raster image whichever way it is stored: as SVG this
# figure is ~4 MB of vector cells and renders no better than a 200 kB PNG.
matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# Point QPDK_PALACE_FIELDS at a solve made with `save=N` to regenerate this.
field_file = Path(
    os.environ.get(
        "QPDK_PALACE_FIELDS",
        "sim_palace_qubit_resonator/output/palace/paraview"
        "/eigenmode/Cycle000001/data.pvtu",
    )
)
if field_file.exists():
    # Log scale, because the field spans several decades between the junction
    # gap and the far side of the chip: on a linear scale everything except
    # the pads reads as black.
    gsim.viz.plot_cross_section(
        pv.read(field_file),
        normal="z",
        origin=500.0,  # the metal plane, atop the 500 µm substrate
        field="E_real",
        log=True,
        quiver=False,
        cmap="inferno",
        title="|E| of the qubit mode at the metal plane",
    )
else:
    print(f"no saved eigenfields at {field_file}")

# %% [markdown]
# The qubit-like mode is unmistakable: the field piles up across the junction
# gap between the two pads and along the pad edges, while the resonator
# meander stays dark. The readout mode does the opposite, forming the
# quarter-wave standing wave along the meander, strongest at the open end and
# with a voltage node at the shorted end where the probeline couples.
#
# The same call renders as an interactive
# [trame](https://kitware.github.io/trame/) widget once live views are
# enabled, which is the better way to look at a mode locally:
#
# ```python
# import gsim.viz
#
# gsim.viz.set_interactive_mode(True)  # then re-run the plot
# ```

# %% [markdown]
# ## Summary
#
# A qpdk layout goes to a full-wave eigenmode spectrum with no commercial
# solver license, and the readout resonance lands within about a percent of the
# semi-analytical CPW model. Calibration transfers in one round for
# perturbative retunes, and the same offset inverts to give the length for a
# target frequency.
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
