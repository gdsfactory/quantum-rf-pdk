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
# The workflow:
#
# 1. Build the layout with qpdk cells.
# 2. Convert the etch-based mask into explicit conductor and dielectric regions.
# 3. Estimate the readout resonance analytically from the CPW cross-section.
# 4. Configure the eigenmode simulation with the Josephson junction as a
#    linear lumped inductor and mesh with Gmsh.
# 5. Solve with Palace on a cluster and compare the eigenmodes against the
#    analytical estimate.
# 6. Calibrate the analytical model against the FEM result and retune the
#    resonator length to a target frequency, iterating the calibration when the
#    retune changes the resonator's surroundings.
#
# Eigenmode analysis is the workhorse behind quantitative qubit design: the
# linearized eigenfrequencies and eigenfields of the Josephson circuit, with the
# junction treated as a lumped inductor, determine the system Hamiltonian once
# completed with the junction's energy participation
# :cite:`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`.
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
# The simulation cells additionally need [gsim](https://gdsfactory.github.io/gsim/)
# (`pip install "gsim @ git+https://github.com/gdsfactory/gsim.git"`, the PyPI
# release lags the repository) and a Palace installation; both are external to
# qpdk. The sections that call them are fenced so the notebook renders without
# them, and the Palace results are embedded below so the analysis runs anywhere.
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
import numpy as np

from qpdk import PDK, logger
from qpdk.cells import straight, transmon_with_resonator_and_probeline
from qpdk.models.constants import c_0
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.resonator import resonator_frequency
from qpdk.simulation import to_fem_regions
from qpdk.tech import LAYER

PDK.activate()

# %% [markdown]
# ## Simulation layout
#
# The device under test is the
# :func:`~qpdk.cells.transmon_with_resonator_and_probeline` cell: a double-pad
# transmon, a meandered quarter-wave CPW resonator
# capacitively coupled to one pad, and a probeline feed capacitively coupled to
# the resonator's shorted end. The cell exposes exactly the ports a
# point-to-point 3-D solve needs:
#
# - ``junction``: the position of the Josephson junction between the two qubit
#   pads, which becomes the lumped inductive port.
# - ``coupling_o1`` / ``coupling_o2``: the two ends of the probeline feed, which
#   become 50 Ω CPW lumped ports.
#
# Two details need care when the cell is used standalone, and the wrapper
# below patches both:
#
# - The cell's ``resonator_length`` parameter is the *total* routed length
#   (meander + feed route + coupling arm); the drawn meander is what sets the
#   quarter-wave frequency, so the wrapper solves for the parameter that draws
#   the requested meander length.
# - The cell attaches the probeline coupling arm at the resonator's start port,
#   so the start never merges into the ground plane as the quarter-wave short
#   requires, and the bare probeline port ends touch the ground plane at their
#   end faces. The wrapper bridges the start bend to ground (realizing the
#   short) and extends the gap etch past both port faces (opening the feed
#   ends).
#
# A ``SIM_AREA`` rectangle around the device bounds the simulation domain.

# %%
MEANDER_LENGTH = 5000.0  # µm, the drawn quarter-wave meander
RESONATOR_MEANDERS = 5


def _drawn_meander_length(resonator_length: float, meanders: int) -> float:
    """Length of the meander the cell actually draws for a given parameter."""
    probe = transmon_with_resonator_and_probeline(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=resonator_length,
        resonator_meanders=meanders,
        qubit_rotation=90,
    )
    resonator = next(
        inst.cell for inst in probe.insts if inst.cell.name.startswith("resonator_")
    )
    return resonator.info["length"]


@gf.cell
def qubit_resonator_sim_component(
    meander_length: float = MEANDER_LENGTH,
    meanders: int = RESONATOR_MEANDERS,
) -> gf.Component:
    """Transmon, resonator and probeline wrapped with a simulation area.

    Args:
        meander_length: Drawn length of the quarter-wave meander in µm.
        meanders: Number of meander sections.

    Returns:
        Component with the simulation layout and ports
        ``junction``, ``coupling_o1``, ``coupling_o2``.
    """
    c = gf.Component()

    # The parameter is the total routed length; the overhead (feed route +
    # coupling arm) is fixed by the layout constants, so one probe call
    # calibrates the parameter for the requested meander.
    probe_length = 5000.0
    overhead = probe_length - _drawn_meander_length(probe_length, meanders)
    ref = c << transmon_with_resonator_and_probeline(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=meander_length + overhead,
        resonator_meanders=meanders,
        qubit_rotation=90,
    )
    c.add_ports(ref.ports)

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
# qpdk draws metal subtractively: ``M1_ETCH`` describes where metal is
# *removed* from a full sheet, and additive shapes on ``M1_DRAW`` punch through
# the etch mask (the transmon pads are drawn this way). A volumetric FEM solver
# needs the opposite convention, explicit conductor and dielectric regions.
# :func:`~qpdk.simulation.to_fem_regions` applies the additive metals, takes
# the conductor as ``SIM_AREA - (M1_ETCH - M1_DRAW)`` (matching
# ``get_layer_stack()``'s M1 definition) and copies the conductor, substrate
# and vacuum regions onto dedicated GDS layers that gsim maps through the
# layer stack. The zero-thickness `SUPERCONDUCTOR` sheet later becomes a
# perfect electric conductor boundary in Palace.

# %%
etched = to_fem_regions(component)

# The cell's ``junction`` port is a placement marker oriented along the
# junction wire, not along the axis connecting the two pads (which separate
# along y after the 90° qubit rotation). Point it across the 15 µm pad gap so
# the lumped port of the simulation bridges pad to pad.
etched.ports["junction"].orientation = 270.0

etched.plot()

# %% [markdown]
# ## Analytical resonator estimate
#
# Before meshing anything, the CPW cross-section model gives a first estimate
# of the readout frequency. A quarter-wave resonator of length :math:`L`
# resonates at :math:`f_r = v_p / 4L` with phase velocity
# :math:`v_p = c_0 / \sqrt{\varepsilon_{\mathrm{eff}}}`, where
# :math:`\varepsilon_{\mathrm{eff}}` comes from the conformal-mapping CPW model
# in :func:`~qpdk.models.cpw_parameters`
# :cite:`simonsCoplanarWaveguideCircuits2001,m.pozarMicrowaveEngineering2012`.

# %%
RESONATOR_CROSS_SECTION = "cpw"

width, gap = get_cpw_dimensions(RESONATOR_CROSS_SECTION)
epsilon_eff, z0 = cpw_parameters(width, gap)
logger.info(f"CPW width {width} µm, gap {gap} µm")
logger.info(
    f"ε_eff = {float(np.real(epsilon_eff)):.3f}, Z0 = {float(np.real(z0)):.1f} Ω"
)

analytical_freq = resonator_frequency(
    length=MEANDER_LENGTH,
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
# and the eigenmode search. This section is fenced because gsim and Palace are
# not qpdk dependencies, but it is the exact code that produced the mesh and
# configuration used below.
#
# ```python
# from gsim.palace import EigenmodeSim
#
# from qpdk.simulation import single_chip_stack
#
# # 500 µm of microwave silicon and 500 µm of air above it; the substrate
# # uses the qpdk material properties (eps_r = 11.45, tan d = 2.7e-6), so the
# # FEM models the same chip as the analytical CPW models of the previous
# # section.
# stack = single_chip_stack(substrate_thickness=500.0, vacuum_thickness=500.0)
#
# sim = EigenmodeSim()
# sim.set_geometry(etched)
# sim.set_stack(stack)
# # A direct factorization makes each shift-and-invert apply cheap, which
# # matters on this roughly 400k-unknown first-order model.
# sim.set_numerical(order=1, solver_type="MUMPS")
#
# # Josephson junction as a linear lumped inductor, L_J = 10 nH, a typical
# # transmon value. resistance=0 overrides the R = 50 Ω that gsim's default
# # port impedance would emit: the linearized junction is purely reactive, so
# # it shifts the qubit-like mode but adds no dissipation, the same convention
# # as Palace's own transmon example. The port length spans the 15 µm pad gap
# # and overlaps both pads; a shorter rectangle would sit entirely in the
# # vacuum gap and couple nothing.
# sim.add_port(
#     "junction",
#     layer="SUPERCONDUCTOR",
#     length=25.0,
#     inductance=10e-9,
#     resistance=0.0,
# )
#
# # Probeline feeds as 50 Ω CPW lumped ports, one per end.
# sim.add_cpw_port("coupling_o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0)
# sim.add_cpw_port("coupling_o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0)
#
# # Target sets the lower edge of the eigenvalue search: the solver returns
# # the N lowest modes above it. Keep it well below the expected band so the
# # qubit-like mode (a few GHz below the readout) is included.
# sim.set_eigenmode(target=2e9, num_modes=10)
#
# sim.set_output_dir("./sim_palace_qubit_resonator")
# sim.mesh(preset="default")
# sim.write_config()
# ```
#
# A few points worth noting:
#
# - The substrate carries the qpdk microwave-silicon loss tangent
#   (:math:`\tan\delta = 2.7 \times 10^{-6}`
#   :cite:`checchinMeasurementLowTemperatureLoss2022`), which sets the
#   dielectric-limited quality factors seen below.
# - The junction port makes the qubit participate in the linear spectrum: the
#   capacitance between the pads comes from the field solution, the inductive
#   part from the port. Deep in the transmon regime this fixes the qubit-like
#   mode to roughly :math:`f_q \approx 1 / (2\pi\sqrt{L_J C_\Sigma})`
#   :cite:`kochChargeinsensitiveQubitDesign2007a,krantzQuantumEngineersGuide2019`.
# - Meshing invokes Gmsh; on some cluster login nodes it needs `libGLU` at
#   import time (for example `module load mesa-glu`).
# - `sim.run()` submits to the GDSFactory+ cloud service. To keep everything
#   local, `sim.mesh()` + `sim.write_config()` produce a self-contained
#   directory (`palace.msh` + `config.json`) that a Palace binary can execute
#   directly, which is what the next section does.

# %% [markdown]
# ## Running Palace on a cluster
#
# The self-contained simulation directory is copied to an HPC cluster and
# solved inside a precompiled Palace Apptainer image with the container's own
# MPI. With the default mesh preset the model has about 260k first-order
# tetrahedra, on the order of 400k unknowns, so the direct MUMPS
# factorization configured above matters: with it, a handful of MPI ranks
# finish the multi-mode solve in minutes. The memory request
# leaves headroom for the factorization, and naming several partitions lets
# Slurm start the job wherever it can begin earliest. If the cluster mixes CPU
# generations, pick a container that matches the node it lands on (an
# AVX-512 build on newer Xeons, AVX2 elsewhere).
#
# ```bash
# #!/bin/bash
# #SBATCH --job-name=palace-eig-qres
# #SBATCH --partition=cpu            # several partitions: earliest-start wins
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=64G
# #SBATCH --time=02:00:00
# #SBATCH --output=slurm-logs/%x-%j.out
#
# set -euo pipefail
# # One rank per core, Palace parallelizes with MPI: pin BLAS to one thread.
# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
#
# SIF=/path/to/palace.sif
#
# dir=$1   # the self-contained simulation directory: config.json + palace.msh
# cd "$dir"
# # --cleanenv keeps host modules and SLURM_* variables out of the container;
# # the palace binary is invoked directly instead of the `palace` wrapper,
# # which mis-parses Slurm's environment inside an allocation.
# apptainer exec --cleanenv "$SIF" \
#   mpirun --oversubscribe -np "$SLURM_CPUS_PER_TASK" \
#   palace config.json > run.log 2>&1
#
# cat output/palace/eig.csv
# ```
#
# Palace writes one row per mode to `output/palace/eig.csv`: the mode index, the
# real and imaginary parts of the eigenfrequency in GHz, and the quality factor.
# Two companion tables make mode identification quantitative:
# `output/palace/port-EPR.csv` reports the energy-participation ratio of each
# lumped port in each mode, and `output/palace/port-Q.csv` the external quality
# factor per port. The rows of the two runs used in this notebook are embedded
# below so the analysis runs without the output files.

# %% [markdown]
# ## Results: eigenmodes versus the analytical estimate
#
# The first run is the baseline geometry (:math:`L = 5000` µm). Of the modes
# returned, we expect to find the qubit-like mode (set by the junction
# inductance and the pad capacitance), the dressed readout resonance near the
# analytical estimate, and higher modes of the feed and meander.

# %% tags=["hide-input"]
# Palace output of the baseline run (L = 5000 µm), embedded so the analysis
# runs without the cluster: eig.csv (mode index, Re{f} (GHz), Q),
# port-EPR.csv (junction participation p[1]) and port-Q.csv (external quality
# factor of the first probeline port).
eig_m = np.arange(1, 12)
eig_f_re = np.array([
    2.476604,
    6.061797,
    12.768789,
    12.833730,
    15.800796,
    16.461020,
    19.657819,
    21.109731,
    23.390055,
    23.515003,
    24.439024,
])
eig_q = np.array([
    3.970572e05,
    4.216235e04,
    5.565953e03,
    1.290870e05,
    3.442681e02,
    6.782414e03,
    2.663997e05,
    2.384206e05,
    3.257438e05,
    1.818605e03,
    3.611636e05,
])
junction_epr = np.array([
    2.668547e-01,
    4.154147e-05,
    1.859882e-05,
    2.324306e-03,
    1.746947e-08,
    9.477185e-08,
    -2.469515e-04,
    1.673788e-01,
    2.424873e-01,
    -5.857359e-04,
    7.040723e-02,
])
q_ext_feed1 = np.array([
    5.612473e07,
    1.073698e05,
    1.064549e04,
    3.554587e05,
    9.325590e02,
    1.980757e04,
    1.721685e11,
    2.875506e07,
    2.943128e06,
    3.349264e03,
    1.821795e07,
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
#    :math:`p_{m,J}` of each mode's energy stored in the junction port
#    :cite:`minevEnergyParticipationQuantization2021`. The qubit-like mode
#    carries by far the largest participation, while a resonator-like mode
#    keeps its inductive energy in the CPW trace, orders of magnitude less.
# 3. **Port dissipation**: `port-Q.csv` gives the external quality factor of
#    each mode through the 50 Ω probeline ports. The readout resonance couples
#    to the feedline and gets a finite :math:`Q_{\mathrm{ext}}` (with both feed
#    ports adding in parallel, :math:`1/Q = 1/Q_{\mathrm{diel}} +
#    \sum 1/Q_{\mathrm{ext}}`); the qubit-like mode barely sees the feedline,
#    so its Q is limited instead by the dielectric loss of the silicon,
#    :math:`Q \approx 1 / (p_{\mathrm{diel}} \tan\delta)` with
#    :math:`\tan\delta = 2.7 \times 10^{-6}`.
#
# No single feature is sufficient on its own: a heavily feed-coupled spurious
# mode can have the lowest Q of the spectrum without being the readout
# resonance. Participation-based identification is the standard in
# quantitative qubit design: the same EPRs that label the modes also turn the
# eigenmode data into a quantum Hamiltonian with anharmonicities and couplings
# :cite:`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`.

# %%
analytical_ghz = analytical_freq / 1e9
i_res = int(np.argmin(np.abs(eig_f_re - analytical_ghz)))
i_qubit = int(np.argmax(np.abs(junction_epr)))
resonator_fem_ghz = eig_f_re[i_res]
resonator_q = eig_q[i_res]
# The cross-checks from the text: the identified readout mode barely touches
# the junction, yet couples to the feedline (finite Q_ext well below the
# dielectric-limited qubit mode's Q).
assert junction_epr[i_res] < 1e-2, "identified mode stores junction energy"
assert q_ext_feed1[i_res] < 1e6, "identified mode is not coupled to the feedline"
assert resonator_q < eig_q[i_qubit], "identified mode is not feed-coupled"

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
# The FEM resonance sits half a percent below the estimate, in the expected
# direction: the analytical model knows nothing about the open-end fringing,
# the qubit and probeline loading, or the meander corners.
#
# Note on signs: Palace reports the junction participation with the sign of
# the port current relative to the mode's field orientation, so the same
# physical mode can come out negative in one solve and positive in another
# (the retuned runs below report it negative). Every identification therefore
# compares magnitudes, :code:`argmax(abs(p_J))`, never the raw signed value.
#
# The qubit-like mode at 2.48 GHz is the one whose frequency is controlled by
# the junction inductance shunted by the pad capacitance the field solution
# sees: inverting :math:`f_q = 1/(2\pi\sqrt{L_J C_\Sigma})` with the
# simulated frequency and :math:`L_J = 10` nH gives the total capacitance
# the mode sees, computed in the next cell. Its junction participation is 0.27
# rather than nearly one because the mode shares its inductive energy with
# the CPW network the pads couple into. Sweeping the junction inductance in
# the lumped port would move this mode as
# :math:`f_q \propto 1/\sqrt{L_J}`, while the readout mode stays put, which is
# the standard numerical check that the port really is the junction
# :cite:`minevEnergyParticipationQuantization2021`.

# %%
c_sigma_qubit = 1 / (2 * np.pi * eig_f_re[i_qubit] * 1e9) ** 2 / 10e-9
logger.info(
    f"C_sigma from the qubit mode: {c_sigma_qubit * 1e15:.0f} fF"
    " (pads + surroundings, from f_q and L_J)"
)

# %% [markdown]
# ## Retuning the resonator to a target frequency
#
# In a design flow the resonator length is chosen for a target readout
# frequency. Since the analytical model tracks the FEM result to within a
# nearly constant multiplicative offset, one eigenmode solve is enough to
# calibrate it: with :math:`\delta = 1 - f_{\mathrm{FEM}} / f_{\mathrm{est}}`,
# the corrected model :math:`f \approx (1 - \delta) v_p / 4L` inverts in closed
# form to
#
# ```{math}
# :label: eq-retuned-length
#
# L^* = \frac{(1 - \delta) \, v_p}{4 f_{\mathrm{target}}}.
# ```
#
# This replaces a blind optimization sweep of full-wave solves (as
# {doc}`optimize_capacitor_optuna` demonstrates for a case without a good
# analytical model): the analytical model does the design, the FEM verifies it.

# %%
TARGET_FREQUENCY = 5.5e9  # Hz

delta = 1.0 - resonator_fem_ghz / analytical_ghz
v_p = c_0 / np.sqrt(float(np.real(epsilon_eff)))
retuned_length = (1.0 - delta) * v_p / (4.0 * TARGET_FREQUENCY) * 1e6

logger.info(f"FEM-vs-analytical offset δ = {delta:.4f}")
logger.info(
    f"Retuned resonator length for {TARGET_FREQUENCY / 1e9:.1f} GHz: "
    f"{retuned_length:.1f} µm (was {MEANDER_LENGTH:.1f} µm)"
)

# %% [markdown]
# ### First retune: the calibration does not transfer unchanged
#
# Regenerating the layout, mesh and configuration with the retuned length (the
# fenced gsim and cluster cells above, with `MEANDER_LENGTH = retuned_length`)
# and solving again gives:

# %% tags=["hide-input"]
# Palace output of the first retuned run (L = 5510.7 µm): the same tables as
# the baseline run; eight of the requested ten modes converged.
eig2_m = np.arange(1, 9)
eig2_f_re = np.array([
    2.160040,
    5.229892,
    12.005100,
    12.710218,
    13.947923,
    16.010887,
    19.637672,
    20.996209,
])
eig2_q = np.array([
    3.679088e05,
    5.734358e04,
    2.781834e04,
    3.948990e05,
    2.750690e03,
    3.451842e02,
    2.819167e05,
    2.533410e05,
])
junction_epr2 = np.array([
    -2.638310e-01,
    3.759043e-03,
    5.922547e-05,
    1.023296e-04,
    -2.940621e-05,
    2.648443e-07,
    3.244723e-02,
    5.762075e-02,
])
q_ext_feed2 = np.array([
    8.631160e06,
    1.432432e05,
    4.347032e04,
    5.644779e07,
    6.963579e03,
    9.050798e02,
    7.642660e06,
    3.227743e06,
])

analytical2_ghz = (
    resonator_frequency(
        length=retuned_length,
        cross_section=RESONATOR_CROSS_SECTION,
        is_quarter_wave=True,
    )
    / 1e9
)
# Same identification as the baseline run: closest to this run's analytical
# estimate, with the participation and feed-coupling cross-checks.
i_res2 = int(np.argmin(np.abs(eig2_f_re - analytical2_ghz)))
assert junction_epr2[i_res2] < 1e-2, "identified mode stores junction energy"
assert q_ext_feed2[i_res2] < 1e6, "identified mode is not coupled to the feedline"
assert eig2_q[i_res2] < eig2_q[int(np.argmax(np.abs(junction_epr2)))], (
    "not feed-coupled"
)
resonator2_fem_ghz = eig2_f_re[i_res2]

logger.info(f"Target frequency:             {TARGET_FREQUENCY / 1e9:.4f} GHz")
logger.info(f"Retuned analytical estimate:   {analytical2_ghz:.4f} GHz")
logger.info(f"Retuned Palace resonance:      {resonator2_fem_ghz:.4f} GHz")
logger.info(
    f"Distance to target:            "
    f"{abs(resonator2_fem_ghz - TARGET_FREQUENCY / 1e9) / (TARGET_FREQUENCY / 1e9):.3%}"
)

# %% [markdown]
# The retuned resonance lands 4.9% *below* the target: the baseline calibration
# did not transfer, and the data says why. The cell pins the meander's start
# and lets it grow eastward, so lengthening it slides the meander *into the
# qubit pads' span*: its east edge advances from about 100 µm to about 200 µm
# inside the pads' footprint. The capacitive loading grows with the coupled
# length, and the smoking gun is the junction participation of the resonator
# mode, which jumped by two orders of magnitude
# (:math:`p_{J} = 4 \times 10^{-5} \to 3.8 \times 10^{-3}`): the same geometry
# change that detuned the resonance also strengthened the qubit-readout
# coupling.
#
# The offset is therefore invariant only under retunes that preserve the
# resonator's surroundings. When the change moves the resonator relative to
# its neighbours, the honest and still cheap workflow is to *iterate*: each
# solve recalibrates the model with fresh data.

# %%
# Recalibrate at the first retuned geometry and invert again.
delta2 = 1.0 - resonator2_fem_ghz / analytical2_ghz
retuned_length2 = (1.0 - delta2) * v_p / (4.0 * TARGET_FREQUENCY) * 1e6

logger.info(f"Recalibrated offset δ =        {delta2:.4f}")
logger.info(
    f"Second retuned length:         {retuned_length2:.1f} µm "
    f"(was {retuned_length:.1f} µm)"
)

# %% [markdown]
# The solve at that length (its resonance identified the same way) puts the
# resonance at 5.4111 GHz, 1.6% below the target: the iteration is
# contracting (4.9% then 1.6%), because each recalibration absorbs most of
# the loading change at the current geometry, but the loading keeps drifting
# with the length. One more round from that solve:

# %%
# Identified resonance of the second retuned solve (L = 5240.1 µm), from the
# same mode-identification procedure (table omitted for brevity).
resonator_fem_2nd_ghz = 5.411140800221

delta3 = 1.0 - resonator_fem_2nd_ghz / (
    resonator_frequency(
        length=retuned_length2,
        cross_section=RESONATOR_CROSS_SECTION,
        is_quarter_wave=True,
    )
    / 1e9
)
retuned_length3 = (1.0 - delta3) * v_p / (4.0 * TARGET_FREQUENCY) * 1e6

logger.info(f"Recalibrated offset δ =        {delta3:.4f}")
logger.info(
    f"Third retuned length:          {retuned_length3:.1f} µm "
    f"(was {retuned_length2:.1f} µm)"
)

# %% [markdown]
# Regenerating and solving once more at the third length:

# %% tags=["hide-input"]
# Palace output of the third retuned run (L = 5155.4 µm).
eig3_m = np.arange(1, 11)
eig3_f_re = np.array([
    2.160023,
    5.450175,
    11.929411,
    12.726322,
    14.646837,
    15.845500,
    19.079637,
    21.015047,
    21.640726,
    22.036610,
])
eig3_q = np.array([
    3.683081e05,
    4.550164e04,
    1.359456e04,
    3.825926e05,
    1.805980e03,
    3.879726e02,
    3.511768e06,
    1.050866e05,
    1.052249e05,
    1.084466e04,
])
junction_epr3 = np.array([
    -2.637066e-01,
    3.422069e-03,
    3.413442e-05,
    1.867641e-06,
    -1.678876e-06,
    1.628997e-08,
    4.965509e-04,
    1.158460e-01,
    2.412692e-01,
    -5.474370e-02,
])
q_ext_feed3 = np.array([
    8.768860e06,
    1.151446e05,
    2.465282e04,
    6.186214e07,
    4.316667e03,
    1.049501e03,
    1.180199e09,
    2.140736e06,
    1.381890e05,
    1.604208e04,
])

analytical3_ghz = (
    resonator_frequency(
        length=retuned_length3,
        cross_section=RESONATOR_CROSS_SECTION,
        is_quarter_wave=True,
    )
    / 1e9
)
i_res3 = int(np.argmin(np.abs(eig3_f_re - analytical3_ghz)))
assert junction_epr3[i_res3] < 1e-2, "identified mode stores junction energy"
assert q_ext_feed3[i_res3] < 1e6, "identified mode is not coupled to the feedline"
assert eig3_q[i_res3] < eig3_q[int(np.argmax(np.abs(junction_epr3)))], (
    "not feed-coupled"
)
resonator3_fem_ghz = eig3_f_re[i_res3]

logger.info(f"Target frequency:             {TARGET_FREQUENCY / 1e9:.4f} GHz")
logger.info(f"Retuned Palace resonance:      {resonator3_fem_ghz:.4f} GHz")
logger.info(
    f"Distance to target:            "
    f"{abs(resonator3_fem_ghz - TARGET_FREQUENCY / 1e9) / (TARGET_FREQUENCY / 1e9):.3%}"
)

# %% [markdown]
# The third iteration brings the resonance to 5.450 GHz, 0.9% below the
# 5.5 GHz target, from 4.9% and 1.6% in the earlier rounds: each round
# recalibrates against a fresh solve, so the calibration loop is a
# contraction towards the fixed point where the analytical model, corrected
# by the latest offset, *is* the FEM.
# For perturbative retunes that keep the resonator's surroundings fixed, a
# single round suffices; for design sweeps that reflow the meander relative to
# the qubit, budget two or three.
#
# The drift that forces the extra rounds is the lumped part of the offset,
# which does not scale with :math:`L`: the fringe and loading capacitances act
# as an effective length extension :math:`\Delta L \approx v_p Z_0
# C_{\mathrm{load}}`, whose size here changes with the meander's position
# relative to the pads. The refined alternative is to calibrate the two
# parameters separately (:math:`v_p` and :math:`\Delta L`, from two solves at
# different lengths) and invert :math:`L^* = v_p / (4 f_{\mathrm{target}}) -
# \Delta L` instead.

# %% [markdown]
# ## Field visualization
#
# With ``set_eigenmode(..., save=N)`` Palace writes the eigenfields of the
# first :math:`N` modes as ParaView data collections under
# ``output/palace/paraview/``, which [pyvista](https://docs.pyvista.org) reads
# back for inspection:
#
# ```python
# from pathlib import Path
#
# import pyvista as pv
#
# field_files = sorted(Path("output/palace/paraview").glob("*.pvd"))
# mode = pv.read(field_files[0])  # the first saved mode
# slice = mode.slice(normal="z", origin=(0.0, 0.0, 500.0))  # at the metal plane
# ```
#
# The resonator mode's electric field forms the quarter-wave standing wave
# along the meander: strongest at the open end near the qubit pad, with a
# voltage node at the shorted end where the probeline couples. The qubit-like
# mode instead concentrates its field between the two transmon pads, across
# the junction port.

# %% [markdown]
# ## Summary
#
# Starting from a pure qpdk layout, the gsim meshing pipeline plus a parallel
# Palace solve yields the full-wave eigenmode spectrum of a transmon-resonator
# system, including the microwave silicon substrate, the linearized junction,
# and realistic probeline terminations, without any commercial solver license.
# The readout resonance agrees with the semi-analytical CPW model to within
# half a percent here. Calibration transfers in one round for perturbative
# retunes, and a second round handles retunes that move the resonator relative
# to the qubit, each round verified by a full-wave solve.
#
# Compared to the SAX circuit models in {doc}`all_models` and
# {doc}`circuit_simulation_demo`, this FEM workflow captures
# geometry-dependent effects, such as fringing, loading and meander parasitics,
# at far higher computational cost, so the two approaches complement each
# other: circuit models to explore the design space, FEM to verify and
# calibrate the final geometry. The natural next step, extracting the quantum
# Hamiltonian, anharmonicities and dispersive couplings from these eigenmodes
# via energy participation, is covered by
# :cite:`minevEnergyParticipationQuantization2021,niggBlackboxSuperconductingCircuit2012`
# and implemented by tools such as [pyEPR](https://github.com/zlatko-minev/pyEPR).
