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
#   "qpdk[models,optimization] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "gsim @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///

# %% [markdown]
# # Batched Layout Optimisation of a Transmon with Optuna
#
# This notebook extends {doc}`palace_eigenmode_qubit_resonator` to a geometry
# sweep. [Optuna](https://optuna.readthedocs.io/) proposes pad dimensions and
# gaps; Palace supplies the linearised qubit frequency and dielectric energy
# participation. The study keeps frequency error, quality factor, and pad area
# as separate Pareto objectives.
#
# Following {cite:p}`Savola2023`, the included dielectric loss gives
#
# $$
# Q_\text{diel}=\left(\sum_i p_i\tan\delta_i\right)^{-1},
# \qquad T_1\approx\frac{Q_\text{diel}}{2\pi f_\text{lin}}.
# $$
#
# The electromagnetic mode needs an anharmonic correction to estimate the
# 0-1 transition. $T_1$ is a derived check; its $1/f$ factor would otherwise
# favor low frequencies even when the field loss is unchanged.
#
# ::::{admonition} Required extras
# :class: tip
#
# Install `models` and `optimization`. The optional Ray backend also needs `ray`:
#
# ```bash
# uv add "qpdk[models,optimization]"
# # in this repository:
# uv sync --extra models --extra optimization --group palace
# # with pip:
# pip install "qpdk[models,optimization]"
# ```
#
# Palace is external to qpdk. Saved trial results let the analysis render
# without running the solver.
#
# The standalone script and Colab setup also install the pinned gsim revision.
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# ::::{note}
# The {ref}`launcher <slurm-launcher>` shows how to repeat the study on Slurm.
# ::::

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
        "qpdk[models,optimization] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
        "gsim @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
    ])

# %%
import io
import json
import os
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

# The same source file is the cluster driver, where there is no display and a
# blocking figure would hang the batch. Rendering inside a kernel keeps the
# interactive backend so the figures appear inline.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import gdsfactory as gf
import klayout.db as kdb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib import font_manager

import qpdk.colormaps  # ruff: ignore[unused-import] - registers the style colormap
from qpdk import PDK, logger
from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.config import PATH
from qpdk.logger import PLAIN_FORMAT, configure_logger
from qpdk.simulation import SlurmCluster
from qpdk.simulation.fem import FEM_LAYERS, single_chip_stack, to_fem_regions
from qpdk.simulation.palace_run import (
    add_domain_energy_postprocessing,
    domain_loss_tangents,
    solve,
    verify_port_connectivity,
)
from qpdk.simulation.study import RayRunner, SlurmRunner, run_study

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

if "ipykernel" in sys.modules:
    import matplotlib_inline

    # Raster outputs avoid Matplotlib SVG links that docs link checkers follow.
    matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# %% [markdown]
# ## The device and its knobs
#
# The {func}`~qpdk.cells.double_pad_transmon_with_bbox` has two pads and a
# junction across their gap. We vary:
#
# - pad width, along the junction axis;
# - pad height, perpendicular to it;
# - pad gap, which the junction bridges.
#
# The junction inductance stays fixed, so geometry changes the capacitance
# and the linearised frequency $f_\text{lin}\approx
# 1/(2\pi\sqrt{L_\text{J}C_\Sigma})$.

# %%
PAD_WIDTH_RANGE = (80.0, 420.0)  # µm, along the junction gap
PAD_HEIGHT_RANGE = (150.0, 650.0)  # µm, perpendicular to it
PAD_GAP_RANGE = (10.0, 160.0)  # µm, island-island separation
LAYOUT_GRID_UM = 0.01

# A fixed inductor, so the geometry moves the linearised frequency through capacitance.
L_JUNCTION = 10e-9  # H
JUNCTION_PARTICIPATION_FLOOR = 0.5
F_TARGET = 4.5e9  # Hz, target linearised mode; the physical transition is lower
# How close to the target counts as on-target when picking a geometry.
TOLERANCE_BAND = 0.02
# Accept small Q differences when choosing a smaller pad footprint.
Q_SLACK = 0.01

# The study size follows the allocation rather than the other way round, so the
# Slurm launcher at the end of the notebook sets these from the job request.
# One switch for every cell that needs a solver, so a half-set value cannot
# run the demos and then silently skip the study.
RUN_OPTIMIZATION = os.environ.get("QPDK_RUN_OPTIMIZATION") == "1"
RUN_DEMO_SOLVE = os.environ.get("QPDK_RUN_DEMO_SOLVE") == "1"
if RUN_OPTIMIZATION:
    # The driver's stdout is a Slurm .out file someone will `tail -f`, not a
    # terminal, so drop the colour escapes and the decoration.
    configure_logger(log_format=PLAIN_FORMAT, colorize=False)
MAX_IN_FLIGHT = int(os.environ.get("QPDK_MAX_IN_FLIGHT") or 6)  # trials at once
N_TRIALS = int(os.environ.get("QPDK_N_TRIALS") or 150)  # total trials

# %% [markdown]
# ## From a layout to a simulation directory
#
# Each trial converts the etch masks into conductor and dielectric regions
# with {func}`~qpdk.simulation.to_fem_regions`, applies
# {func}`~qpdk.simulation.single_chip_stack`, and writes a Palace mesh and
# config. The junction port is turned to face across the pad gap. The config
# requests domain energies so the loss objective can be reconstructed.
#
# The junction sheet is 2 µm wide and spans the snapped pad gap exactly.
# Overhanging sheets remove PEC coverage inside the pads during meshing,
# as shown in {doc}`palace_eigenmode_qubit_resonator`. The checks below reject
# overlap and require contact with both pad edges.


# %%
@dataclass(slots=True, frozen=True, kw_only=True)
class Layout:
    """One candidate double-pad transmon geometry.

    Attributes:
        pad_width: Pad extent along the junction gap, in µm.
        pad_height: Pad extent perpendicular to the gap, in µm.
        pad_gap: Island-island separation the junction bridges, in µm.
    """

    pad_width: float
    pad_height: float
    pad_gap: float

    @property
    def pad_size(self) -> tuple[float, float]:
        """Pad size in µm, as the cell expects it."""
        return (self.pad_width, self.pad_height)

    @property
    def footprint_mm2(self) -> float:
        """Metal area of the two pads in mm².

        The qubit's footprint, which is what a chip pays for in real estate and
        what stands in for the coupling and crosstalk a larger qubit invites.
        Only the pads are counted; the etch around them belongs to the ground
        plane and scales with the gap.
        """
        return 2.0 * self.pad_width * self.pad_height / 1e6


def transmon_sim_layout(
    pad_size: tuple[float, float], pad_gap: float, margin: float = 250.0
) -> gf.Component:
    """Transmon pads wrapped in the area the solver meshes.

    Args:
        pad_size: Pad width and height in µm, snapped to a 10 nm grid.
        pad_gap: Island-island gap in µm, snapped to the same grid.
        margin: Space around the etched box to the simulation boundary in µm.

    Returns:
        Component carrying the layout, its ports, and a ``SIM_AREA`` rectangle
        bounding the simulation domain.
    """
    # Resolve the layer at call time so the notebook evaluator can be sent to workers.
    from qpdk.tech import LAYER

    grid_factor = round(LAYOUT_GRID_UM * 1000)
    pad_size = (
        gf.snap.snap_to_grid(pad_size[0], grid_factor=grid_factor),
        gf.snap.snap_to_grid(pad_size[1], grid_factor=grid_factor),
    )
    pad_gap = gf.snap.snap_to_grid(pad_gap, grid_factor=round(LAYOUT_GRID_UM * 1000))

    c = gf.Component()
    ref = c << double_pad_transmon_with_bbox(pad_size=pad_size, pad_gap=pad_gap)
    junction = ref.ports["junction"]
    c.add_port(
        "junction",
        center=junction.center,
        width=2.0,
        orientation=junction.orientation,
        layer=junction.layer,
        port_type=junction.port_type,
    )
    c.add_ports(ref.ports.filter(regex=r"^(?!junction$)"))
    c.kdb_cell.shapes(gf.get_layer(LAYER.SIM_AREA)).insert(
        c.bbox().enlarged(margin, margin)
    )
    return c


def build_simulation(
    layout: Layout,
    sim_dir: Path,
    *,
    junction_inductance: float = L_JUNCTION,
    num_modes: int = 3,
    substrate_thickness: float = 500.0,
    vacuum_thickness: float = 500.0,
    lateral_margin: float = 250.0,
    refined_mesh_size: float = 0.35,
    linear_max_its: int = 40,
    mesh_preset: Literal["coarse", "default", "fine"] = "coarse",
) -> Path:
    """Write a self-contained Palace eigenmode directory for ``layout``.

    The result is ``config.json`` plus ``palace.msh``, which is everything a
    Palace binary needs, so the directory can be handed to a worker on another
    node without any further state.

    Args:
        layout: Candidate geometry.
        sim_dir: Directory to write into; created if missing.
        junction_inductance: Linearised junction inductance in H.
        num_modes: Number of eigenmodes to solve for.
        substrate_thickness: Silicon thickness in µm.
        vacuum_thickness: Air height above the chip in µm.
        lateral_margin: Distance from the etch-box edge to the outer boundary in µm.
        refined_mesh_size: Near-conductor mesh size in µm.
        linear_max_its: Maximum iterations for the Palace linear solve.
        mesh_preset: Gmsh mesh preset.

    Returns:
        The simulation directory.
    """
    from gsim.palace import EigenmodeSim, resolve_physical_groups

    from qpdk import PDK

    # Worker imports need the PDK active to resolve cell layers.
    PDK.activate()

    sim_dir.mkdir(parents=True, exist_ok=True)
    regions = to_fem_regions(
        transmon_sim_layout(layout.pad_size, layout.pad_gap, lateral_margin)
    )
    # The cell's junction marker is perpendicular to the pad-to-pad axis.
    regions.ports["junction"].orientation = 0.0
    port_length = regions.ports["right_pad_inner"].x - regions.ports["left_pad_inner"].x
    metal = kdb.Region(
        regions.kdb_cell.begin_shapes_rec(
            regions.kdb_cell.layout().layer(*FEM_LAYERS["SUPERCONDUCTOR"])
        )
    ).merged()
    junction_sheet = kdb.Region(
        kdb.DBox(-port_length / 2, -1.0, port_length / 2, 1.0).to_itype(regions.kcl.dbu)
    )
    assert (junction_sheet & metal).is_empty(), "junction sheet overlaps pad metal"
    assert len(metal.interacting(junction_sheet.sized(1))) == 2, (
        "junction sheet must touch both pad edges"
    )

    sim = EigenmodeSim()
    sim.set_geometry(regions)
    sim.set_stack(
        single_chip_stack(
            substrate_thickness=substrate_thickness,
            vacuum_thickness=vacuum_thickness,
        )
    )
    sim.set_numerical(order=2, solver_type="MUMPS")
    # Gsim's default 50 Ohm port would add loss to the linearised junction.
    sim.add_port(
        "junction",
        layer="SUPERCONDUCTOR",
        length=port_length,
        inductance=junction_inductance,
        resistance=0.0,
    )
    sim.set_eigenmode(target=1e9, num_modes=num_modes)
    sim.set_output_dir(sim_dir)
    sim.mesh(preset=mesh_preset, refined_mesh_size=refined_mesh_size, auto_size=False)
    sim.write_config()
    config_path = sim_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["Solver"]["Linear"].update({
        "MaxIts": linear_max_its,
        "Tol": 1e-5,
        "MGMaxLevels": 2,
        "PCMatReal": False,
        "PCMatShifted": False,
        "ComplexCoarseSolve": True,
        "ColumnOrdering": "ParMETIS",
    })
    config["Solver"]["Eigenmode"]["Tol"] = 1e-5
    surfaces = (
        ("SA", "SUBSTRATE__VACUUM"),
        ("MS", "SUPERCONDUCTOR_pec"),
        ("MA", "SUPERCONDUCTOR_pec"),
    )
    config["Boundaries"]["Postprocessing"] = {
        "Dielectric": [
            {
                "Index": index,
                "Attributes": resolve_physical_groups(sim_dir, [group]),
                "Type": kind,
                "Thickness": 3e-9 / config["Model"]["L0"],
                "Permittivity": 10.0,
                "LossTan": 1e-3,
            }
            for index, (kind, group) in enumerate(surfaces, start=1)
        ]
    }
    config_path.write_text(json.dumps(config, indent=2))
    add_domain_energy_postprocessing(sim_dir)
    verify_port_connectivity(sim_dir)
    return sim_dir


def evaluate_simulation(sim_dir: Path) -> dict[str, Any]:
    """Extract the transmon mode and its dielectric loss from a Palace solve.

    ``f_linear`` is the LC mode, so a target for the physical 0-1 transition
    needs the anharmonic correction from the inferred capacitance.

    Args:
        sim_dir: Simulation directory Palace wrote its ``output/`` into.

    Returns:
        A row of results: ``f_linear`` in Hz, ``T1`` in seconds, dielectric
        ``quality_factor`` from domain participations, the raw
        ``eigenmode_quality_factor``, junction participation, and per-domain
        electric-energy participations.

    Raises:
        FileNotFoundError: If the solve produced no eigenmode table.
        ValueError: If no mode carries junction energy.
    """
    from gsim.palace.results import load_text_results

    output = sim_dir / "output" / "palace"
    eig_path = output / "eig.csv"
    if not eig_path.exists():
        msg = f"Palace produced no eig.csv in {output}; see {sim_dir}/run.log"
        raise FileNotFoundError(msg)

    tables = load_text_results(output).csv_tables

    def table(name: str) -> list[dict[str, str]]:
        return [
            {key.strip(): value for key, value in row.items()} for row in tables[name]
        ]

    eig = table("eig.csv")
    junction = table("port-EPR.csv")
    domains = table("domain-E.csv")

    if not len(eig) == len(junction) == len(domains):
        msg = (
            f"Palace tables disagree on mode count in {output}: "
            f"eig={len(eig)}, port-EPR={len(junction)}, domain-E={len(domains)}"
        )
        raise ValueError(msg)

    participations = np.array([abs(float(row["p[1]"])) for row in junction])
    frequencies = np.array([float(row["Re{f} (GHz)"]) for row in eig])

    # A higher harmonic can have more junction energy, so choose the lowest
    # participating mode above the floor that excludes packaging modes.
    candidates = np.flatnonzero(participations >= JUNCTION_PARTICIPATION_FLOOR)
    if candidates.size == 0:
        msg = (
            "no mode carries junction energy; the lumped port is probably not "
            "spanning the pad gap"
        )
        raise ValueError(msg)
    index = int(candidates[np.argmin(frequencies[candidates])])

    f_linear = float(eig[index]["Re{f} (GHz)"]) * 1e9
    loss_tangents = domain_loss_tangents(sim_dir)
    domain_participations = {
        int(key[len("p_elec[") : -1]): float(value)
        for key, value in domains[index].items()
        if key.startswith("p_elec[")
    }
    if domain_participations.keys() != loss_tangents.keys():
        msg = f"domain-E.csv is missing a configured dielectric domain in {output}"
        raise ValueError(msg)
    inverse_q = sum(
        participation * loss_tangents[domain]
        for domain, participation in domain_participations.items()
    )
    quality_factor = 1.0 / inverse_q if inverse_q > 0 else float("inf")

    row: dict[str, Any] = {
        "f_linear": f_linear,
        "quality_factor": quality_factor,
        "eigenmode_quality_factor": float(eig[index]["Q"]),
        "T1": quality_factor / (2.0 * np.pi * f_linear),
        "junction_participation": float(participations[index]),
        "num_modes": len(eig),
        "mode_index": index + 1,
    }
    for domain, participation in domain_participations.items():
        row[f"participation_domain{domain}"] = participation
    if "surface-Q.csv" in tables:
        surfaces = table("surface-Q.csv")
        if len(surfaces) != len(eig):
            msg = f"surface-Q.csv disagrees on mode count in {output}"
            raise ValueError(msg)
        for surface_index, kind in enumerate(("sa", "ms", "ma"), start=1):
            row[f"participation_{kind}"] = float(
                surfaces[index][f"p_surf[{surface_index}]"]
            )
    return row


def trial_dir_name(layout: Layout) -> str:
    """Return the directory name a trial of ``layout`` is written under.

    Full float precision keeps distinct inputs distinct even when a serial
    caller passes values off the sampling grid.

    Args:
        layout: Candidate geometry.

    Returns:
        A directory name unique to these geometry values.
    """
    return f"w{layout.pad_width:.17g}_h{layout.pad_height:.17g}_g{layout.pad_gap:.17g}"


def evaluate_layout(
    params: dict[str, float],
    run_root: str | Path,
    *,
    ranks: int = 2,
    sim_dir: str | Path | None = None,
    timeout: float = 6000.0,
    **build_settings: Any,
) -> dict[str, Any]:
    """Build, solve and post-process one candidate geometry.

    This is the unit of work a sweep repeats, so it takes and returns plain
    data: everything it learns is either in the return value or on disk under
    ``run_root``, and a caller that dies loses only its own trial.

    Args:
        params: Keyword arguments for :class:`Layout`.
        run_root: Parent directory for trial output.
        ranks: MPI ranks to give each solve.
        sim_dir: Directory for this trial. Defaults to a geometry-named
            directory under ``run_root``, which is only safe when trials do not
            run concurrently.
        timeout: Seconds before a solve is abandoned. Size it under the trial
            job's wall clock, or Slurm kills the task first and the trial
            reports nothing.
        **build_settings: Forwarded to :func:`build_simulation`.

    Returns:
        The row from :func:`evaluate_simulation`, plus the geometry parameters.
    """
    layout = Layout(**params)
    # Geometry-named output is safe only when callers run serially.
    sim_dir = Path(sim_dir) if sim_dir else Path(run_root) / trial_dir_name(layout)
    build_settings.setdefault("junction_inductance", L_JUNCTION)
    build_simulation(layout, sim_dir, **build_settings)
    solve(sim_dir, ranks=ranks, timeout=timeout)
    return (
        evaluate_simulation(sim_dir) | params | {"footprint_mm2": layout.footprint_mm2}
    )


# %%
demo_layout = Layout(pad_width=250.0, pad_height=400.0, pad_gap=15.0)
demo_dir = Path("build/palace_optimization_demo")
# Meshing needs Gmsh and a large allocation, so the demo runs separately from
# the one-core sweep driver.
if RUN_DEMO_SOLVE:
    build_simulation(demo_layout, demo_dir)
    logger.info(f"wrote {demo_dir}")

# %% [markdown]
# ## Running Palace
#
# {func}`~qpdk.simulation.palace_run.solve` launches Palace with MPI. Each
# trial can use several nodes.
#
# ## Reading a solve back into physics
#
# The evaluator reads `eig.csv` for frequency, `port-EPR.csv` to identify
# the lowest junction-coupled mode, and `domain-E.csv` for dielectric energy.
# It calculates the loss objective from those energies and the stack's loss
# tangents. The raw eigenfrequency $Q$ remains a solver diagnostic.

# %%
# Run the demo inside an allocation sized like one trial before launching a sweep.
demo_result: dict[str, float] | None = None
if RUN_DEMO_SOLVE:
    solve(demo_dir, ranks=int(os.environ.get("QPDK_SOLVER_CORES") or 24))
    demo_result = evaluate_simulation(demo_dir)
    logger.info(
        f"demo layout: f_lin = {demo_result['f_linear'] / 1e9:.4f} GHz, "
        f"T1 = {demo_result['T1'] * 1e6:.2f} µs, "
        f"Q = {demo_result['quality_factor']:.3e}"
    )

# %% [markdown]
# ### Loss model
#
# Vacuum is lossless in this model. Silicon has the PDK loss tangent, so the
# reported $Q_\text{diel}$ is limited by bulk silicon. Nanometre scale
# metal-air, metal-substrate, and substrate-air layers add loss in a fabricated
# device {cite:p}`Savola2023`. Their participations are examined separately
# below; no interface loss tangent is assumed for the Pareto objective.

# %%
if demo_result is not None:
    tangents = domain_loss_tangents(demo_dir)
    budget = sum(
        demo_result[f"participation_domain{index}"] * tangent
        for index, tangent in tangents.items()
    )
    logger.info(
        f"1/Q from participations {budget:.4e}; "
        f"raw eigenmode 1/Q {1 / demo_result['eigenmode_quality_factor']:.4e}"
    )


# %% [markdown]
# ## Checking that the mode is real
#
# A junction port can be rectangular yet electrically detached from its
# conductor. `build_simulation` checks mesh-node contact before submission.
# Doubling the junction inductance in a separate solve moves the identified
# mode close to the expected $1/\sqrt{2}$ frequency ratio.

# %% [markdown]
# ### Mesh sensitivity
#
# The screening mesh is not a converged frequency limit. A same-geometry
# refinement check is needed before selection; the figure below shows that the
# frequency can move enough to change which designs meet the target band.
# Bulk-silicon $Q$ changes much less, while interface participation needs its
# own refinement check near the metal edges.


# %%
def site(name: str, default: str) -> str:
    """Return a site setting, letting the environment override the default."""
    return os.environ.get(f"QPDK_{name.upper()}") or default


DEFAULT_SETUP = """
module load python 2>/dev/null || true
export UV_CACHE_DIR="$QPDK_SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$QPDK_SCRATCH/uv-python"
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
"""

PALACE_SIF = site("palace_sif", site("sif", "/path/to/palace.sif"))

cluster = SlurmCluster(
    partition=site("partition", "batch"),
    scratch=site("scratch", "/scratch/$USER/palace-optimization"),
    setup=site("setup", DEFAULT_SETUP),
    solver_cores=int(site("solver_cores", "8")),
    trial_nodes=int(site("trial_nodes", "4")),
    mem_per_task=site("mem_per_task", "32G"),
    time_limit=site("time_limit", "02:00:00"),
    nodes=int(site("nodes", "4")),
    cores_per_node=int(site("cores_per_node", "8")),
    mem_per_node=site("mem_per_node", "96G"),
    job_name=site("job_name", "qpdk-opt"),
    log_dir=site("log_dir", "slurm-logs"),
    account=site("account", "") or None,
    driver_time_limit=site("driver_time_limit", "24:00:00"),
)

# %% [markdown]
# One measured eight-rank screening trial used about 63 GiB. The defaults
# reserve 96 GiB for each Ray node and 32 GiB per node for a four-node Slurm
# trial. Check a representative layout before scaling the study.

# %% [markdown]
# ## Running the study
#
# Optuna proposes layouts and receives each Palace result as soon as its trial
# finishes. The runner chooses where each independent solve runs:
#
# - {class}`~qpdk.simulation.study.SlurmRunner` submits one job per trial.
# - {class}`~qpdk.simulation.study.RayRunner` uses a reserved Ray allocation.
#
# The runner serializes this notebook's `evaluate_layout` function for each
# worker. `max_in_flight` limits concurrent solves.

# %%
RUN_ROOT = (
    Path(os.path.expandvars(site("scratch", "build"))) / "palace_optimization_trials"
)


def suggest_layout(trial: optuna.Trial) -> None:
    """Ask Optuna for a candidate geometry, recording it on the trial.

    Suggestions land on the layout grid, so a trial's recorded parameters are
    exactly the geometry that was simulated.
    """
    for name, (low, high) in (
        ("pad_width", PAD_WIDTH_RANGE),
        ("pad_height", PAD_HEIGHT_RANGE),
        ("pad_gap", PAD_GAP_RANGE),
    ):
        trial.suggest_float(name, low, high, step=LAYOUT_GRID_UM)


def objectives(row: dict[str, Any]) -> list[float]:
    """Map one solved trial to the three values the study is ranking.

    Args:
        row: A result row from ``evaluate_layout``.

    Returns:
        Relative frequency error, substrate-limited quality factor, and pad footprint.
    """
    return [
        abs(row["f_linear"] - F_TARGET) / F_TARGET,
        row["quality_factor"],
        row["footprint_mm2"],
    ]


def make_study() -> optuna.Study:
    """Return the multi-objective study this sweep optimises.

    Returns:
        A study ranking frequency error, quality factor and footprint together.
    """
    return optuna.create_study(
        study_name="transmon-layout",
        # Frequency error, then substrate-limited Q, then footprint.
        directions=["minimize", "maximize", "minimize"],
        # NSGA-II keeps a population of non-dominated trials instead of
        # collapsing the objectives into one score.
        sampler=optuna.samplers.NSGAIISampler(seed=0),
    )


# An importable module:function name can still override the notebook evaluator.
EVALUATOR = os.environ.get("QPDK_EVALUATOR") or evaluate_layout


def make_runner() -> SlurmRunner | RayRunner:
    """Return the runner the environment selects.

    Returns:
        A Ray runner when ``QPDK_EXECUTOR=ray``, otherwise a Slurm one.

    Raises:
        ValueError: If a Ray trial needs more cores than one node provides.
    """
    if os.environ.get("QPDK_EXECUTOR") == "ray":
        if cluster.solver_cores > cluster.cores_per_node:
            raise ValueError("Ray trials must fit on one node")
        return RayRunner(
            run_root=RUN_ROOT, ranks=cluster.solver_cores, evaluator=EVALUATOR
        )
    return SlurmRunner(
        cluster=cluster,
        run_root=RUN_ROOT,
        ranks=cluster.solver_cores,
        evaluator=EVALUATOR,
        timeout=6000.0,
    )


# %% [markdown]
# ## Results
#
# The embedded example replays 150 stored candidate geometries with the
# exact-gap junction sheet. The live workflow above uses Optuna to generate
# new candidates. The analysis below reads the saved table without starting
# a solver.
#
# The recorded bulk screen solves one isolated qubit mode with order-2
# elements, a 0.35 µm conductor mesh and a 10 nH junction. Linear-solver and
# eigenmode tolerances are both $10^{-5}$. Separate finer, order and domain
# checks below test the near-target layouts.

# %% tags=["hide-input"]
# Palace sweep results, embedded so the analysis runs without the solver:
# one row per solved candidate, with geometry, objectives, and loss diagnostics.
TRIALS_CSV = """\
trial,pad_width,pad_height,pad_gap,f_linear,T1,quality_factor,eigenmode_quality_factor,junction_participation,footprint_mm2,participation_domain1,participation_domain2
0,266.6,507.6,100.42,5604681430.967,1.1433753879957159e-05,402642.52439989755,404907.879799,0.9805118711886,0.27065232,0.9198491165889,0.08015088341112
1,265.26,361.83000000000004,106.89,6396640715.247,1.001771375996961e-05,402624.7282417888,403530.2982186,0.9801575552678,0.1919580516,0.9198897742514,0.08011022574858
2,228.78,595.89,154.55,5635902126.191,1.1370255314719237e-05,402636.8572577744,405535.8036664,0.9730620443918,0.27265542839999996,0.91986206353,0.08013793646999
3,210.37,545.87,89.33,5664882710.805,1.1312819839774154e-05,402662.94138150365,401748.1329338,0.9829068051945,0.2296693438,0.9198024757373,0.08019752426272
4,273.14,612.8,20.65,4513739217.626,1.4198455021742636e-05,402677.55444035673,403220.7482366,0.9922345648208,0.33476038399999997,0.9197690963558,0.08023090364422
5,109.62,160.1,134.9,10682526476.970001,5.998265767959566e-06,402605.358258545,403883.1219075,0.9808466875303,0.035100324,0.9199340316095,0.08006596839054
6,344.58,585.01,156.8,5193629038.979,1.2337944803082816e-05,402618.3991885957,405035.2357956,0.9705197728227,0.4031654915999999,0.9199042346718,0.08009576532816
7,351.72,380.74,127.08,5935962532.233,1.079497794378613e-05,402617.6213301614,394732.8233722,0.9753689415943,0.2678277456,0.9199060119295,0.0800939880705
8,120.21000000000001,469.96,31.5,6060026256.11,1.0576013116866993e-05,402695.10910617415,401500.5616782,0.9927921368614,0.11298778320000001,0.919729000912,0.08027099908801
9,401.19,410.92,72.2,5389700897.084001,1.1889442734197483e-05,402629.90847436775,402802.1986391,0.9823862590999,0.32971398960000003,0.9198779389583,0.08012206104165
10,169.95,537.12,78.42,5880027073.091,1.0898893734214868e-05,402662.8955339605,405339.5730109,0.9851618479968,0.182567088,0.9198025804668,0.08019741953324
11,273.27,159.39,102.65,8239267028.679001,7.776905411911133e-06,402601.383913772,411154.0784684,0.9802148510404,0.08711301059999998,0.9199431128873,0.08005688711269
12,288.11,458.47,151.57,5907090974.215,1.0847669442919643e-05,402615.0178731178,400454.7089616,0.9730562642121,0.26417958340000003,0.9199119603807,0.08008803961928
13,311.82,329.75,75.55,6202855202.940001,1.0330980228810132e-05,402636.40713153844,403156.1678921,0.9836827807643,0.20564529,0.9198630918872,0.08013690811278
14,317.20000000000005,180.11,110.02,7642286532.799,8.384345982775834e-06,402598.7075594517,400426.7856522,0.9781159400401,0.11426178400000003,0.9199492283906,0.08005077160936
15,308.02,255.19,29.34,6378945482.129,1.0046161377872857e-05,402651.1177671632,395520.770792,0.9906021471594,0.15720724760000002,0.9198294851985,0.08017051480148
16,187.24,331.85,95.53,7100448099.877,9.024976914102151e-06,402635.18641811906,400164.4460559,0.983478725985,0.124271188,0.9198658807374,0.08013411926262
17,229.12,644.19,25.3,4629257155.935,1.3844628346241866e-05,402691.5130502433,402951.2335268,0.9918037433728,0.2951936256,0.9197372141393,0.08026278586072
18,151.01999999999998,230.65,107.97,8598766602.445,7.4520247303445515e-06,402615.33903400326,382210.0392653,0.9829933191971,0.06966552599999999,0.9199112265792,0.08008877342083
19,166.12,383.15999999999997,46.660000000000004,6468784950.121,9.907376759434491e-06,402681.1133324579,403890.327289,0.9903800227881,0.1273010784,0.9197609674447,0.08023903255535
20,134.05,205.18,108.45,9223218724.462,6.9478086586165736e-06,402633.7961563118,406863.0380942,0.9834403908991,0.05500875800000001,0.9198690569596,0.08013094304045
21,126.98,248.29000000000002,65.31,8371758194.891999,7.6551833003266e-06,402672.6552307323,402839.8490651,0.9890782351653,0.06305572840000001,0.9197802869384,0.08021971306163
22,359.14,198.55,135.7,7238655711.142,8.85186558641588e-06,402598.9148472358,403264.4834779,0.9735233187723,0.142614494,0.9199487547325,0.08005124526751
23,112.67,638.23,80.3,5916874324.589,1.0831645846055847e-05,402686.1243170056,401742.7043059,0.9851909423483,0.1438187482,0.9197495220342,0.08025047796581
24,412.1,452.42,120.89,5362165111.088,1.1950404970356786e-05,402626.7946853543,410981.7338595,0.9748549690235,0.37288456400000003,0.919885053005,0.08011494699496
25,93.32,291.4,28.03,7707419152.36,8.315425940409148e-06,402692.3192460009,403242.3076123,0.9943828613694,0.05438689599999999,0.9197353728123,0.0802646271877
26,180.69,209.36,57.7,8133771493.09,7.878773351650235e-06,402652.53999159764,410863.017802,0.9889891592606,0.07565851679999999,0.9198262362336,0.08017376376645
27,220.85,182.07,113.87,8459188521.327001,7.575022316630992e-06,402617.3525294913,413602.919381,0.980361592613,0.08042031899999999,0.9199066260892,0.08009337391079
28,272.65,282.69,88.49,6835789818.525,9.37410582447408e-06,402622.85234657564,402123.2561494,0.9826744387405,0.154150857,0.91989406019,0.08010593981
29,111.94,437.97,149.4,7390810300.563999,8.670792785058628e-06,402652.80728896614,406966.6136532,0.9772536469982,0.0980527236,0.9198256256154,0.08017437438463
30,188.31,483.71,29.77,5498856257.194,1.1655394194020415e-05,402697.74880222057,403107.39482,0.9921483388248,0.18217486019999998,0.9197229720603,0.08027702793969
31,323.55,294.70000000000005,37.480000000000004,6093578907.889,1.0516471910201933e-05,402645.0587865966,402208.0645986,0.9891271746725,0.19070037000000004,0.9198433267417,0.0801566732583
32,279.42,160.05,134.34,8276600534.331999,7.741462043332565e-06,402582.4623452741,398691.2142753,0.9757516934258,0.08944234200000001,0.9199863506541,0.08001364934595
33,81.59,488.91,50.5,6712509882.228,9.54800306702043e-06,402696.0375708344,402358.1644532,0.9905732250218,0.0797803338,0.919726880365,0.08027311963501
34,329.97,631.1,47.31,4618155830.577,1.3876939261081517e-05,402663.3839586346,402300.3865801,0.9869917033212,0.41648813400000007,0.91980146476,0.08019853523996
35,275.89,446.02,95.84,5821364932.143,1.1007984259700298e-05,402635.89869151014,400240.7488727,0.9812420599033,0.24610491559999997,0.9198642534707,0.08013574652926
36,155.84,626.38,77.07000000000001,5590149455.151,1.146465253783085e-05,402683.8242833959,403097.032131,0.9850812473074,0.1952301184,0.9197547754233,0.08024522457674
37,367.78000000000003,499.74,54.61,5013914664.712,1.2781776832473682e-05,402668.85248051956,402722.462117,0.9856191298827,0.36758875440000005,0.9197889732191,0.08021102678093
38,356.69,348.25,142.17000000000002,6134276429.386001,1.0445734695808056e-05,402607.8165546961,403303.3108767,0.9731604455059,0.24843458499999999,0.9199284145544,0.08007158544561
39,277.63,590.87,113.88,5290006963.948999,1.2113978767258665e-05,402645.5653527984,399460.5467791,0.9779694541732,0.3280864762,0.9198421694918,0.08015783050819
40,326.59000000000003,400.65999999999997,153.42000000000002,6015030927.988999,1.0652705376390626e-05,402603.59534511546,402332.7526427,0.9721771321186,0.26170309880000003,0.9199380598002,0.08006194019977
41,298.96000000000004,361.93,100.96000000000001,6190046064.349,1.0352015395547926e-05,402623.0722868125,400835.2874202,0.9802756254364,0.21640518560000002,0.919893557681,0.08010644231896
42,86.52,300.78999999999996,109.03,8823380591.37,7.262970803660771e-06,402651.3684071103,400196.1126727,0.9840503135287,0.05204870159999999,0.9198289126287,0.08017108737134
43,178.62,459.01,74.31,6181052757.644,1.0368006138391107e-05,402659.14264422044,402988.0907974,0.985996381816,0.16397673240000002,0.9198111532702,0.08018884672984
44,126.06,299.14,95.5,8068618045.560999,7.942215897628307e-06,402643.5200089521,400324.0043758,0.9849377587221,0.0754191768,0.9198468420953,0.08015315790468
45,280.9,437.16,107.98,5895882086.449,1.0868816902680637e-05,402634.4493768481,400954.8086997,0.979449162244,0.24559648799999997,0.9198675645951,0.08013243540488
46,301.72,365.71000000000004,144.49,6304990998.587,1.0163105362947017e-05,402615.7566094174,396796.2243432,0.9740921677462,0.22068404240000003,0.9199102724876,0.08008972751237
47,204.97,367.93,143.79,6906290082.832,9.278417669861997e-06,402623.02064185287,412115.6177013,0.9763863852503,0.1508292242,0.9198936756769,0.08010632432314
48,354.11,501.95,25.03,4754457653.791,1.3479778560338113e-05,402683.2915305851,400412.3555719,0.9907315113644,0.35549102899999996,0.9197559922653,0.08024400773465
49,392.63,507.12,159.83,5313117215.272,1.2060803484293618e-05,402629.42122641567,398175.6603468,0.9694461335481,0.3982210512,0.9198790521622,0.08012094783778
50,401.19,160.05,72.2,7219356313.277,8.87602078615982e-06,402621.21586975636,403040.764237,0.9807666007844,0.12842091900000002,0.9198977991517,0.08010220084834
51,112.67,638.23,32.41,5357254520.3289995,1.1963403242934543e-05,402695.6050288921,406951.9683282,0.9919549813613,0.1438187482,0.9197278682587,0.08027213174131
52,111.94,446.02,140.22,7295364952.121,8.78389268893935e-06,402637.213907202,405278.7653758,0.9784096544924,0.09985495759999999,0.9198612487313,0.08013875126874
53,273.27,159.39,34.370000000000005,7754724014.165,8.263390493669747e-06,402628.47923412523,405196.8330556,0.9899096679679,0.08711301059999998,0.9198812043174,0.08011879568256
54,289.28999999999996,211.91,137.2,7568993615.128,8.465678730230679e-06,402605.5805244035,390016.4899786,0.9753971804751,0.12260688779999998,0.9199335237429,0.08006647625708
55,273.14,612.8,20.65,4513739178.809,1.4198455145367798e-05,402677.55448352406,401165.6934637,0.9922345964301,0.33476038399999997,0.9197690962572,0.08023090374281
56,359.14,553.6600000000001,29.34,4613585845.605,1.3890708944841093e-05,402664.0764549466,404184.3871109,0.9898100721892,0.3976829048,0.9197998828977,0.08020011710227
57,111.94,437.97,95.37,7094664574.799,9.032906174320718e-06,402660.69150886324,398214.2907557,0.9844337683365,0.0980527236,0.9198076151474,0.0801923848526
58,218.44,483.71,20.37,5180377801.865,1.2372144328880247e-05,402704.311896832,394849.9815932,0.9932753048357,0.2113232248,0.919707982827,0.08029201717302
59,298.96000000000004,498.72,78.03,5392180684.768001,1.1884449177438354e-05,402645.975417105,403534.6898125,0.9832255739062,0.29819466240000003,0.9198412327025,0.08015876729751
60,93.32,511.03000000000003,28.03,6019410230.72,1.0647646865643595e-05,402705.3965841453,403530.0732975,0.9933955405358,0.0953786392,0.9197055055928,0.08029449440721
61,210.37,255.19,89.33,7554108347.19,8.482787436452495e-06,402625.8571444106,404573.196878,0.9840037470212,0.1073686406,0.9198871950182,0.08011280498184
62,180.69,209.36,139.96,8639687122.0,7.416534611950236e-06,402604.7657202593,400658.8732528,0.9780461667411,0.07565851679999999,0.9199353855332,0.08006461446682
63,411.68,577.9100000000001,65.31,4694936409.768001,1.3649596489365097e-05,402651.5689152041,402853.1955782,0.9830336349188,0.4758279776000001,0.9198284545822,0.08017154541779
64,412.1,294.70000000000005,120.89,6164441249.073,1.0394397820772227e-05,402599.2120687438,401286.5338267,0.9747579436978,0.24289174000000005,0.9199480755743,0.08005192442568
65,83.98,488.91,64.0,6856682929.147,9.347185612092875e-06,402693.66930689674,398697.6447917,0.9887510315111,0.0821173236,0.9197322893301,0.08026771066993
66,328.20000000000005,638.23,80.3,4810883386.866,1.3320606557558008e-05,402650.92334413,401641.2709825,0.981852366925,0.4189341720000001,0.9198299293451,0.08017007065491
67,301.72,365.71000000000004,144.49,6304990998.585,1.0163105362950241e-05,402615.7566094174,396796.2519649,0.9740921677465,0.22068404240000003,0.9199102724876,0.08008972751237
68,109.62,160.1,134.9,10682533179.89,5.99826200168109e-06,402605.35808576236,401284.5807166,0.9808454108434,0.035100324,0.9199340320043,0.08006596799571
69,93.32,235.81,28.03,8419108690.999,7.61247710364795e-06,402691.05626425886,397754.8442051,0.9945928422424,0.0440115784,0.9197382574281,0.08026174257194
70,367.78000000000003,410.52,18.15,4991128625.553,1.2840125958807304e-05,402668.738926355,401300.650188,0.9918735314833,0.3019620912,0.9197892326032,0.08021076739682
71,272.65,282.69,88.49,6835789818.521001,9.374105824479566e-06,402622.85234657564,402123.2856905,0.9826744387422,0.154150857,0.91989406019,0.08010593981
72,229.12,507.6,40.0,5336338676.869,1.2009698020360916e-05,402675.62413418107,402404.6630477,0.9899685079324,0.232602624,0.919773505453,0.080226494547
73,86.29,469.96,134.34,7487219635.745,8.559158985343012e-06,402653.55241603433,406351.7365231,0.979537388731,0.0811056968,0.91982392344,0.08017607656001
74,86.52,546.85,109.03,6869455997.914,9.329231611377075e-06,402668.901158161,386690.8502529,0.9823106462689,0.094626924,0.9197888620281,0.08021113797189
75,166.12,383.15999999999997,75.55,6760248838.705999,9.479635015699114e-06,402655.9927154412,400884.1123166,0.9863976851995,0.1273010784,0.9198183488408,0.08018165115917
76,392.63,507.12,95.84,5125506655.607,1.2502548131003705e-05,402638.41268508864,401553.9394227,0.9788429447677,0.3982210512,0.9198585100226,0.08014148997743
77,156.13,322.67,80.3,7365670484.558,8.700415070270189e-06,402654.1007242905,405189.6452915,0.9862120467634,0.10075693420000001,0.9198226708834,0.08017732911658
78,120.21000000000001,469.96,149.22,7075833094.924,9.056786731300016e-06,402653.60309920897,404296.7950872,0.9768607861858,0.11298778320000001,0.9198238076591,0.08017619234095
79,187.24,331.85,28.03,6399110720.826,1.0015093935830358e-05,402674.863436994,394785.3877434,0.9928904776411,0.124271188,0.9197752430065,0.08022475699353
80,319.5,182.07,113.87,7615246550.951,8.413808900950524e-06,402583.9723767091,392818.3723373,0.9775388508335,0.11634272999999999,0.9199828999248,0.0800171000752
81,155.84,205.18,14.77,7613215185.1050005,8.417850758888374e-06,402669.9264157515,401942.5647785,0.9955256136839,0.06395050240000001,0.9197865201087,0.08021347989126
82,326.59000000000003,400.65999999999997,153.42000000000002,6015030927.986,1.0652705376395937e-05,402603.59534511546,402332.8258816,0.9721771321196,0.26170309880000003,0.9199380598002,0.08006194019977
83,135.99,460.74,143.79,6932762145.127999,9.243285790481405e-06,402635.9507278932,405706.942764,0.9773235486276,0.12531206520000002,0.9198641345881,0.08013586541187
84,277.63,438.62,45.68,5512641891.063,1.1625323538958762e-05,402665.756356821,402822.5010661,0.9885953287291,0.2435481412,0.9197960455375,0.08020395446248
85,397.64,469.96,134.9,5377312929.286,1.1916545183108096e-05,402620.18408406613,395633.0268813,0.9730773524188,0.3737497888,0.9199001565531,0.08009984344691
86,288.75,294.70000000000005,90.35000000000001,6651924221.589,9.633494265711336e-06,402634.51828589285,401165.336461,0.9820535883037,0.17018925000000004,0.919867407164,0.08013259283603
87,93.32,444.96,28.03,6407053093.473001,1.0003557903620075e-05,402710.248057568,403179.0764208,0.9937051367723,0.08304733439999999,0.9196944258479,0.0803055741521
88,328.24,361.93,149.4,6206256950.526,1.0324436472346682e-05,402602.04535254976,399584.1451742,0.9727926354256,0.2375998064,0.9199416015039,0.0800583984961
89,311.82,329.75,75.55,6202855202.934,1.0330980228820127e-05,402636.40713153844,403156.1120133,0.9836827807662,0.20564529,0.9198630918872,0.08013690811278
90,308.02,255.19,56.79,6661917184.945,9.618990810073449e-06,402632.29614210787,399526.6641328,0.986377996189,0.15720724760000002,0.9198724839491,0.08012751605092
91,323.55,294.70000000000005,69.73,6354062830.25,1.0084843560068542e-05,402624.8155982077,399998.0388654,0.9842464987306,0.19070037000000004,0.9198895746654,0.08011042533465
92,317.20000000000005,254.92000000000002,88.49,6801600453.626,9.421142055704905e-06,402619.2516229838,403593.0932767,0.9815801003006,0.161721248,0.9199022870302,0.08009771296983
93,120.21000000000001,243.09,31.5,7998072829.427,8.012889274275935e-06,402674.71901761455,407551.0648699,0.9936699007657,0.0584436978,0.9197755728841,0.08022442711591
94,401.09000000000003,291.4,120.94,6232690251.845,1.0280725140538346e-05,402604.99686983426,403649.6638082,0.9750126796944,0.233755252,0.9199348573662,0.08006514263382
95,323.55,626.38,77.07000000000001,4845289969.1380005,1.3226333035161807e-05,402660.5617064965,403249.6162434,0.9824699535089,0.40533049800000004,0.9198079116582,0.0801920883418
96,246.76,160.1,44.11,8074438474.000001,7.936616998303289e-06,402649.9233988656,401166.4847122,0.9891286416761,0.079012552,0.9198322136609,0.08016778633907
97,210.37,282.69,89.33,7288005004.909,8.792644389954199e-06,402631.7692399366,399611.1479363,0.9839659419896,0.1189389906,0.9198736877359,0.08012631226407
98,166.48000000000002,179.01,75.16,8925215266.358,7.179671808014993e-06,402627.2460248752,403814.8051579,0.986979028721,0.0596031696,0.9198840218267,0.08011597817335
99,186.01,458.47,114.45,6374216522.431001,1.005354793645803e-05,402648.4513812515,398832.1376655,0.9803165440171,0.17056000940000002,0.9198355764187,0.08016442358133
100,208.43,638.23,134.9,5525803709.417,1.159722520160919e-05,402651.58463218174,407786.0932134,0.9759412878784,0.26605255780000003,0.9198284186779,0.08017158132208
101,359.14,553.6600000000001,29.34,4613585845.606,1.3890708944838079e-05,402664.0764549466,404184.4409427,0.9898100721888,0.3976829048,0.9197998828977,0.08020011710227
102,141.06,230.65,28.03,7823979379.301,8.19134727406308e-06,402682.6369065919,406821.3717977,0.9938244062572,0.065070978,0.9197574874734,0.0802425125266
103,109.62,577.9100000000001,13.7,5144330099.033,1.245898330215945e-05,402708.9674974238,401036.1750228,0.9951213806308,0.12670098840000002,0.919697350352,0.08030264964805
104,102.86,459.01,111.91,7182329653.451,8.922765763886998e-06,402665.7538356182,405142.0698192,0.9822658728968,0.0944275372,0.9197960512966,0.08020394870336
105,155.84,507.6,100.42,6272319972.648,1.0217052817955603e-05,402655.5707708072,396928.8832884,0.9825255454367,0.158208768,0.9198193127227,0.08018068727733
106,166.12,383.15999999999997,46.660000000000004,6468784950.123,9.907376759431428e-06,402681.1133324579,403890.3176659,0.9903800227875,0.1273010784,0.9197609674447,0.08023903255534
107,155.84,438.62,78.05,6504356524.566,9.853087482685238e-06,402676.74135712645,401555.5514884,0.9859748703488,0.1367090816,0.9197709535498,0.08022904645017
108,93.32,235.81,28.03,8419108691.056,7.61247710359641e-06,402691.05626425886,397754.6633615,0.9945928422286,0.0440115784,0.9197382574281,0.08026174257194
109,218.44,418.29,144.5,6492107342.41,9.870277907964194e-06,402619.62608929264,402451.3965741,0.9757338744561,0.1827425352,0.9199014314524,0.08009856854761
110,126.98,248.29000000000002,75.16,8472562112.188,7.563709374372632e-06,402651.6313471266,403166.6171847,0.9877942812994,0.06305572840000001,0.919828311961,0.08017168803905
111,416.72,258.45,109.46000000000001,6360135249.189,1.0074622978373454e-05,402601.16011631076,404751.2451725,0.9761031904447,0.215402568,0.9199436242642,0.08005637573579
112,169.53,444.96,13.09,5391759529.2699995,1.188676952648612e-05,402693.13426012883,402825.3224702,0.9951431290125,0.15086813759999998,0.9197335113519,0.08026648864807
113,188.31,529.19,29.77,5290864833.462,1.2113550619750836e-05,402696.6284383362,403036.6109769,0.9919810289568,0.1993035378,0.9197255308709,0.08027446912907
114,188.8,444.96,28.03,5662916012.18,1.1317284282298011e-05,402681.9973766938,403178.044618,0.9925442111741,0.168016896,0.9197589482102,0.08024105178983
115,401.19,410.92,67.52000000000001,5364918373.097,1.1944637662280395e-05,402639.1188953134,403224.980097,0.9831061113192,0.32971398960000003,0.9198568966337,0.08014310336633
116,280.03,507.12,29.34,5008037890.158999,1.279684870501263e-05,402671.14513893396,407472.3244397,0.9909577341673,0.2840176272,0.9197837362858,0.0802162637142
117,188.31,483.71,29.77,5498856257.192,1.1655394194024654e-05,402697.74880222057,403107.403914,0.9921483388255,0.18217486019999998,0.9197229720603,0.08027702793969
118,362.56,464.49,109.03,5439975533.756,1.1779565877372242e-05,402629.9713114924,404222.9739068,0.977581887527,0.3368109888,0.919877795396,0.08012220460397
119,376.7,286.77,28.03,5827325888.049,1.0997211021498691e-05,402653.736270793,401092.024222,0.9896510364212,0.21605251799999997,0.9198235034414,0.08017649655858
120,351.34000000000003,626.38,37.84,4495926326.158,1.4254336643861172e-05,402667.0245614105,408755.148865,0.9882722904005,0.44014469840000003,0.9197931486289,0.08020685137112
121,401.19,626.4000000000001,134.9,4839431330.851,1.324123195918011e-05,402626.7205871342,398448.7643412,0.9724198445318,0.502610832,0.9198852222979,0.08011477770212
122,313.75,257.75,134.9,6965834247.227,9.198630638397486e-06,402602.2383203235,401238.7276799,0.9751831755549,0.161738125,0.9199411605747,0.08005883942533
123,120.21000000000001,499.74,54.61,6247341578.478001,1.0258658041361729e-05,402685.2052228244,407873.1617796,0.9894331227412,0.12014749080000002,0.919751621283,0.08024837871701
124,120.21000000000001,501.95,31.5,5886133558.859,1.0888520007134486e-05,402697.4077914497,406199.3497064,0.9926491655939,0.120678819,0.9197237508968,0.08027624910319
125,93.32,469.96,28.03,6251102179.14,1.0252746539727793e-05,402695.41738114075,404988.5450321,0.9935868646512,0.0877133344,0.9197282968329,0.08027170316709
126,402.11,515.4300000000001,48.09,4816570743.073,1.3305136927142404e-05,402658.76768182963,402872.8249248,0.986085514674,0.41451911460000007,0.9198120098133,0.08018799018667
127,229.12,507.6,41.99,5359596284.136,1.1957352180539623e-05,402667.8598221254,406616.0402016,0.989667909079,0.232602624,0.9197912406865,0.08020875931351
128,354.11,160.1,25.03,7104407341.692,9.020180663312879e-06,402645.60109092545,402366.9373492,0.9894411043433,0.113386022,0.9198420878482,0.08015791215182
129,83.98,488.91,64.0,6856682929.133,9.347185612111959e-06,402693.66930689674,398697.7465334,0.9887510315115,0.0821173236,0.9197322893301,0.08026771066993
130,256.19,638.23,134.34,5290496468.004,1.2112740873249794e-05,402641.67460243334,402871.3926304,0.9751507709517,0.32701628740000005,0.9198510579812,0.08014894201876
131,88.72,507.12,28.03,6092022653.215,1.052095758311119e-05,402713.92571715463,405877.6787129,0.9934581705645,0.0899833728,0.9196860270248,0.0803139729752
132,180.69,209.36,41.120000000000005,7929032572.916,8.082312755507719e-06,402657.43468559964,409705.1467905,0.9912695250203,0.07565851679999999,0.9198150548482,0.08018494515178
133,93.32,444.96,28.03,6407053093.5,1.0003557903577922e-05,402710.248057568,403179.2276257,0.9937051367624,0.08304733439999999,0.9196944258479,0.0803055741521
134,86.52,362.34000000000003,66.12,7790928327.341,8.226117133085053e-06,402683.6223313072,404958.7349346,0.989189871764,0.0626993136,0.9197552366946,0.08024476330539
135,367.78000000000003,381.78999999999996,51.64,5515672310.367,1.1618488479935168e-05,402650.23474699556,400709.7508311,0.9861867794259,0.2808294524,0.9198315024033,0.08016849759668
136,279.51,294.70000000000005,18.15,5984097399.3880005,1.0709430124523772e-05,402665.92863588966,401885.2430535,0.9930912124486,0.164743194,0.9197956520063,0.0802043479937
137,266.6,507.6,50.5,5288287377.393,1.2118719918544896e-05,402672.215706165,399188.394848,0.9878339750811,0.27065232,0.9197812908965,0.08021870910347
138,373.71,208.76,29.34,6507036236.242,9.848376013164013e-06,402650.01099458843,402162.3138351,0.9889024433164,0.15603139919999998,0.9198320135532,0.08016798644681
139,93.32,408.69,29.81,6697684945.358999,9.569009562051993e-06,402690.67388467194,399371.2050253,0.9936278046985,0.0762779016,0.9197391307762,0.08026086922376
140,308.02,577.9100000000001,29.34,4664486465.051,1.3739888075901991e-05,402686.34273178346,402704.0430397,0.9904131164482,0.35601567640000004,0.9197490231673,0.08025097683265
141,93.32,243.09,31.5,8410032754.29,7.620651631253712e-06,402688.90543268895,401734.4238857,0.9941224779252,0.045370317599999994,0.9197431699103,0.0802568300897
142,411.68,248.29000000000002,65.31,6282030847.166,1.0200226662105606e-05,402614.81858238037,405981.557848,0.9826616371424,0.20443205440000004,0.9199124157289,0.08008758427114
143,218.44,483.71,117.53,6041570583.403,1.060706244905343e-05,402647.3524703431,405247.7198497,0.9791213017237,0.2113232248,0.9198380868471,0.08016191315291
144,359.14,483.71,69.41,5194481633.709,1.2336594732634688e-05,402640.430238986,402356.2249636,0.9835034474986,0.34743921879999995,0.9198539007882,0.08014609921177
145,367.78000000000003,159.39,94.81,7511372649.281,8.530555499035754e-06,402602.52062440495,400452.691825,0.9784927449298,0.1172409084,0.9199405155138,0.0800594844862
146,210.37,545.87,37.49,5228791712.727,1.225674299468766e-05,402676.5047356616,405249.1300804,0.9904594536798,0.2296693438,0.9197714940272,0.08022850597278
147,218.44,209.36,20.37,7216196853.717,8.881409044740254e-06,402689.32123724895,404035.1089818,0.9935833983385,0.09146519680000001,0.9197422202119,0.08025777978812
148,120.21000000000001,612.8,31.5,5381962031.4539995,1.1908584285626026e-05,402699.0756837685,402414.8374913,0.9921363793075,0.14732937599999998,0.9197199416003,0.08028005839967
149,308.02,222.42000000000002,20.37,6536798520.754,9.803965048845953e-06,402667.6330864374,403860.7139866,0.9919568033399,0.1370196168,0.9197917586062,0.08020824139384
"""


recorded_trials = pd.read_csv(io.StringIO(TRIALS_CSV))
recorded_trials["f_linear_ghz"] = recorded_trials["f_linear"] / 1e9
if RUN_OPTIMIZATION:
    study = make_study()
    rows = run_study(
        study,
        make_runner(),
        suggest=suggest_layout,
        objectives=objectives,
        n_trials=N_TRIALS,
        max_in_flight=MAX_IN_FLIGHT,
    )
    trials = pd.DataFrame(rows)
    trials.to_csv(os.environ.get("QPDK_TRIALS_CSV", "trials.csv"), index=False)
else:
    trials = recorded_trials.copy()
    study = None

trials["f_linear_ghz"] = trials["f_linear"] / 1e9
trials["T1_us"] = trials["T1"] * 1e6
logger.info(f"{len(trials)} successful trials")

# %% [markdown]
# ### Quality factor across the geometry sweep
#
# These plots show bulk-silicon $Q$ against each layout parameter. Color
# marks distance from the target frequency. The narrow $Q$ axis makes small
# differences visible; the absolute scale appears in the labels.

# %%
frames = {
    "pad_width": "pad width (µm)",
    "pad_height": "pad height (µm)",
    "pad_gap": "pad gap (µm)",
}
PARAMETERS = list(frames)
target_ghz = F_TARGET / 1e9
frequency_error_mhz = np.abs(trials["f_linear_ghz"] - target_ghz) * 1e3

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
for ax, name in zip(axes, PARAMETERS, strict=True):
    points = ax.scatter(
        trials[name],
        trials["quality_factor"] / 1e3,
        c=frequency_error_mhz,
        cmap="viridis_r",
        s=25,
    )
    ax.set_xlabel(frames[name])
axes[0].set_ylabel("substrate-limited $Q$ (thousands)")
fig.colorbar(points, ax=axes, label="frequency error (MHz)")
fig.suptitle("Quality factor versus pad geometry")
plt.show()
plt.close(fig)


# %% [markdown]
# ### The Pareto frontier
#
# The highlighted designs are not beaten on all three objectives at once:
# frequency error, bulk-silicon $Q$, and pad area. Interface loss could
# change this ordering.


# %%
def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return a mask of rows not dominated when every column is minimised."""
    keep = np.ones(len(values), dtype=bool)
    for i, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(
            values < candidate, axis=1
        )
        if dominated.any():
            keep[i] = False
    return keep


objective_values = np.column_stack([
    frequency_error_mhz,
    -trials["quality_factor"],
    trials["footprint_mm2"],
])
on_front = pareto_mask(objective_values)
front = trials[on_front].sort_values("f_linear_ghz")
logger.info(f"{len(front)} of {len(trials)} trials are on the Pareto frontier")

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
points = axes[0].scatter(
    trials["f_linear_ghz"],
    trials["quality_factor"] / 1e3,
    c=trials["footprint_mm2"],
    cmap="viridis",
    s=24,
)
axes[0].scatter(
    front["f_linear_ghz"],
    front["quality_factor"] / 1e3,
    facecolors="none",
    edgecolors="crimson",
    s=70,
    label="Pareto frontier",
)
axes[0].axvspan(
    target_ghz * (1 - TOLERANCE_BAND),
    target_ghz * (1 + TOLERANCE_BAND),
    color="crimson",
    alpha=0.1,
    label=f"target ±{TOLERANCE_BAND:.0%}",
)
axes[0].set_xlabel("linearised qubit frequency (GHz)")
axes[0].set_ylabel("substrate-limited $Q$ (thousands)")
axes[0].legend()
fig.colorbar(points, ax=axes[0], label="pad footprint (mm²)")

points = axes[1].scatter(
    trials["footprint_mm2"],
    trials["quality_factor"] / 1e3,
    c=frequency_error_mhz,
    cmap="viridis_r",
    s=24,
)
axes[1].scatter(
    front["footprint_mm2"],
    front["quality_factor"] / 1e3,
    facecolors="none",
    edgecolors="crimson",
    s=70,
)
axes[1].set_xlabel("pad footprint (mm²)")
axes[1].set_ylabel("substrate-limited $Q$ (thousands)")
fig.colorbar(points, ax=axes[1], label="frequency error (MHz)")
fig.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ### Seeing several dimensions at once
#
# Each line is a trial. Axes are independently scaled to expose trends;
# the colorbar gives the actual $Q$ range.

# %%
parallel_axes = [*PARAMETERS, "f_linear_ghz", "quality_factor", "footprint_mm2"]
axis_limits = {
    name: (float(trials[name].min()), float(trials[name].max()))
    for name in parallel_axes
}
normaliser = plt.Normalize(
    float(trials["quality_factor"].min()),
    float(trials["quality_factor"].max()),
)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
colors = plt.cm.viridis(normaliser(trials["quality_factor"].to_numpy(dtype=float)))
for row, color in zip(trials.itertuples(), colors, strict=True):
    values = []
    for name in parallel_axes:
        lower, upper = axis_limits[name]
        values.append(
            (getattr(row, name) - lower) / (upper - lower) if upper > lower else 0.5
        )
    ax.plot(
        range(len(parallel_axes)),
        values,
        color=color,
        alpha=0.35,
        lw=0.8,
    )
for index in range(len(parallel_axes)):
    ax.axvline(index, color="0.85", lw=0.8, zorder=0)
ax.set_xticks(range(len(parallel_axes)))
ax.set_xticklabels([
    *[frames[name] for name in PARAMETERS],
    r"$f_\text{lin}$ (GHz)",
    "$Q$",
    "footprint (mm²)",
])
ax.set_ylabel("position within observed range")
ax.set_title("Parallel coordinates, coloured by substrate-limited $Q$")
fig.colorbar(
    plt.cm.ScalarMappable(normaliser, plt.cm.viridis),
    ax=ax,
    label="substrate-limited $Q$",
)
fig.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ### Frequency and relaxation time
#
# The median-$Q$ line isolates the $1/f_\text{lin}$ factor in
# $T_1\approx Q/(2\pi f_\text{lin})$. Color shows each trial's actual $Q$.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8))
points = ax.scatter(
    trials["f_linear_ghz"],
    trials["T1_us"],
    c=trials["quality_factor"] / 1e3,
    cmap="viridis",
    s=25,
)
frequencies = np.linspace(
    trials["f_linear_ghz"].min(), trials["f_linear_ghz"].max(), 200
)
median_q = trials["quality_factor"].median()
ax.plot(
    frequencies,
    median_q / (2 * np.pi * frequencies * 1e3),
    color="crimson",
    label=r"median $Q/(2\pi f_\text{lin})$",
)
ax.axvline(target_ghz, color="0.45", ls="--", lw=1)
ax.set_xlabel("linearised qubit frequency (GHz)")
ax.set_ylabel("$T_1$ (µs)")
ax.legend()
fig.colorbar(points, ax=ax, label="$Q$ (thousands)")
fig.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ### Screened geometry and finer checks
#
# The screen keeps candidates near the target frequency, then favors a small
# footprint among those with similar bulk-silicon $Q$. Candidate frequencies
# are checked on finer meshes before selecting a layout.

# %%
on_target = trials[
    np.abs(trials["f_linear_ghz"] - target_ghz) < TOLERANCE_BAND * target_ghz
]
if on_target.empty:
    logger.warning(
        f"no trial within {TOLERANCE_BAND:.0%} of {target_ghz} GHz; "
        "falling back to the closest trial"
    )
    on_target = trials.iloc[[int(np.argmin(frequency_error_mhz))]]
best_q = float(on_target["quality_factor"].max())
affordable = on_target[on_target["quality_factor"] >= best_q * (1 - Q_SLACK)]
screen_best = affordable.sort_values("footprint_mm2").iloc[0]
logger.info(
    f"coarse-screen trial {int(screen_best['trial'])}: pads "
    f"{screen_best['pad_width']:.2f} × {screen_best['pad_height']:.2f} µm, "
    f"gap {screen_best['pad_gap']:.2f} µm"
)
logger.info(
    f"f_lin = {screen_best['f_linear_ghz']:.4f} GHz, "
    f"Q = {screen_best['quality_factor']:.1f}, "
    f"T1 = {screen_best['T1_us']:.2f} µs, "
    f"footprint = {screen_best['footprint_mm2']:.3f} mm²"
)

# %% [markdown]
# The 0.35 µm, order-2 screen is a first pass. Separate solves check three
# near-target layouts at 0.25 and 0.18 µm, then change the element order.
# Trials 4 and 120 also have lateral-margin and vacuum-height checks.
# All use the same exact-gap junction sheet and 10 nH inductance.
# These stored checks refer to the recorded replay. A live study needs
# independent follow-up solves for its own candidate geometries.

# %% tags=["hide-input"]
VALIDATION_CSV = """\
trial,mesh_um,order,lateral_margin,vacuum_thickness,f_linear_ghz,p_silicon,raw_eigenmode_q
4,0.25,2,500.0,500.0,4.509242673014,0.9197589545697,404019.6521584
4,0.18,2,250.0,500.0,4.51537207245,0.9197637731133,401326.4318183
4,0.25,2,250.0,500.0,4.51441236528,0.9197587374942,402470.9432577
4,0.18,3,250.0,500.0,4.5273511895,0.9197577021821,402368.273222
4,0.25,3,250.0,500.0,4.526966035373,0.9197568357319,402465.9706221
4,0.25,2,250.0,1000.0,4.513722370968,0.9194605202065,403098.3935339
56,0.18,2,250.0,500.0,4.615568464954,0.9197944116966,400962.3817425
56,0.25,2,250.0,500.0,4.614651819947,0.9197751793862,403109.4133622
56,0.25,3,250.0,500.0,4.625479286504,0.9197917487072,403904.5165871
120,0.25,2,500.0,500.0,4.486454799212,0.9197906151886,401047.5065955
120,0.18,2,250.0,500.0,4.497419856893,0.9197960505837,405084.7673214
120,0.25,2,250.0,500.0,4.497067697159999,0.9197901494788,401205.2406622
120,0.18,3,250.0,500.0,4.50691040646,0.9197953577544,405182.9634816
120,0.25,3,250.0,500.0,4.506666625873,0.9197938579773,403712.4140725
120,0.35,3,250.0,500.0,4.506246717762,0.9197939602154,403883.7044948
120,0.25,2,250.0,1000.0,4.495667651331001,0.9191283634783,402487.6503235
"""
checks = pd.read_csv(io.StringIO(VALIDATION_CSV)).merge(
    recorded_trials[["trial", "pad_width", "pad_height", "pad_gap", "footprint_mm2"]],
    on="trial",
    validate="many_to_one",
)
checks["candidate"] = "trial " + checks["trial"].astype(str)
checks["dielectric_q"] = 1 / (checks["p_silicon"] * 2.7e-6)
checks["target_error_mhz"] = (checks["f_linear_ghz"] - target_ghz) * 1e3
mesh_checks = checks[
    (checks["lateral_margin"] == 250) & (checks["vacuum_thickness"] == 500)
]

# %%
coarse_checks = recorded_trials.loc[
    recorded_trials["trial"].isin(checks["trial"]), ["trial", "f_linear_ghz"]
].assign(mesh_um=0.35, order=2)
coarse_checks["candidate"] = "trial " + coarse_checks["trial"].astype(str)
plotted_checks = pd.concat(
    [coarse_checks, mesh_checks[["candidate", "mesh_um", "order", "f_linear_ghz"]]],
    ignore_index=True,
)
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.axhspan(
    target_ghz * (1 - TOLERANCE_BAND),
    target_ghz * (1 + TOLERANCE_BAND),
    color="crimson",
    alpha=0.12,
    label=f"target ±{TOLERANCE_BAND:.0%}",
)
for (candidate, order), group in plotted_checks.groupby(["candidate", "order"]):
    ordered = group.sort_values("mesh_um")
    ax.plot(
        ordered["mesh_um"],
        ordered["f_linear_ghz"],
        "o-" if order == 2 else "s--",
        label=f"{candidate}, order {order}",
    )
ax.set_xlabel("conductor mesh size (µm)")
ax.set_ylabel("linearised qubit frequency (GHz)")
ax.set_title("Independent mesh and element-order checks")
ax.legend(ncol=2, fontsize="small")
fig.tight_layout()
plt.show()
plt.close(fig)

# %%
checks.loc[
    checks["trial"].isin([4, 120]),
    ["trial", "mesh_um", "order", "lateral_margin", "vacuum_thickness", "f_linear_ghz"],
].sort_values(["trial", "order", "mesh_um", "lateral_margin", "vacuum_thickness"])

# %% [markdown]
# Selection uses the finest order-3 result for each checked layout and
# requires every saved frequency check to stay inside the target band.
# This supports screening at that band, not a converged mask specification
# or an absolute lifetime prediction. Small differences in bulk-silicon
# quality factor should not be read as resolved device-loss improvements.

# %%
SELECTION_MESH_UM = 0.18
selected_resolution = mesh_checks[
    (mesh_checks["mesh_um"] == SELECTION_MESH_UM) & (mesh_checks["order"] == 3)
]
band_mhz = TOLERANCE_BAND * target_ghz * 1e3
stays_in_band = checks.groupby("candidate")["target_error_mhz"].apply(
    lambda errors: errors.abs().lt(band_mhz).all()
)
verified_on_target = selected_resolution[
    selected_resolution["candidate"].isin(stays_in_band[stays_in_band].index)
]
assert not verified_on_target.empty, "no candidate stays in the target band"
best_q = float(verified_on_target["dielectric_q"].max())
affordable = verified_on_target[
    verified_on_target["dielectric_q"] >= best_q * (1 - Q_SLACK)
]
best = affordable.sort_values("footprint_mm2").iloc[0]
best_layout = Layout(
    pad_width=float(best["pad_width"]),
    pad_height=float(best["pad_height"]),
    pad_gap=float(best["pad_gap"]),
)
logger.info(
    f"selected recorded candidate at working mesh {best['candidate']}: {best_layout}, "
    f"f_lin = {best['f_linear_ghz']:.4f} GHz at {best['mesh_um']:.3f} µm, "
    f"order {int(best['order'])}, "
    f"substrate Q = {best['dielectric_q']:.1f}, "
    f"footprint = {best['footprint_mm2']:.3f} mm²"
)
best_layout

# %% [markdown]
# ### Interface participation
#
# Separate Palace solves add 3 nm substrate-air (SA), metal-substrate (MS),
# and metal-air (MA) dielectric sheets with relative permittivity 10. These
# first-order solves reuse the refined layout meshes but have lower basis order
# than the main sweep to stay within moderate memory. Their participations are
# screening diagnostics, not inputs to the bulk-loss Pareto frontier.

# %% tags=["hide-input"]
INTERFACE_CSV = """\
trial,f_linear_ghz,participation_sa,participation_ms,participation_ma,junction_participation
11,7.833728954447,6.824467356723e-05,6.692614247426e-05,5.10212508188e-07,0.9812917316874
30,5.216741516243,9.189196337016e-05,9.021975851803e-05,6.797717026997e-07,0.9926316806468
32,7.881224870809,6.541355670948e-05,6.400417825216e-05,4.857020683876e-07,0.9770923476422
33,6.40324198803,0.0001017501755525,9.86110755754e-05,7.481073613076e-07,0.9911127770621
42,8.411984768571,8.905261441131e-05,8.625352563279e-05,6.678377903853e-07,0.9849566253396
53,7.39317301994,8.83117320028e-05,8.59492536728e-05,6.501279615843e-07,0.9904070918337
62,8.230347962552,6.98400028415e-05,6.854546775824e-05,5.224693956796e-07,0.9792980481013
81,7.227199592737,0.0001445131334379,0.0001328293061745,1.022589302004e-06,0.9957970390669
87,6.095508771705,0.0001215786407533,0.0001174100334911,8.923626246758e-07,0.9940693625285
111,6.082339802312,5.158340715175e-05,5.191453269884e-05,3.941907444876e-07,0.9774147071611
112,5.104240206624,0.0001440598684397,0.0001315160924288,1.003758898654e-06,0.9954935007389
134,7.415852720799,9.589360744654e-05,9.372244464328e-05,7.156105452623e-07,0.9898072329492
"""
interfaces = pd.read_csv(io.StringIO(INTERFACE_CSV)).merge(
    recorded_trials[["trial", *PARAMETERS]], on="trial", validate="one_to_one"
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
for ax, name in zip(axes, PARAMETERS, strict=True):
    for kind, label, color, marker in (
        ("sa", "substrate-air", "#009E73", "o"),
        ("ms", "metal-substrate", "#0072B2", "s"),
        ("ma", "metal-air", "#CC79A7", "^"),
    ):
        ax.scatter(
            interfaces[name],
            interfaces[f"participation_{kind}"],
            label=label,
            color=color,
            marker=marker,
            s=34,
        )
    ax.set_xlabel(frames[name])
axes[0].set_yscale("log")
axes[0].set_ylabel("3 nm interface participation")
axes[-1].legend(loc="upper right", fontsize="small")
fig.suptitle("Interface electric-energy participation")
plt.show()
plt.close(fig)

# %% [markdown]
# These values depend on near-metal mesh resolution and sheet assumptions.
# A loss tangent for each interface is needed to turn them into an interface
# quality factor.

# %% [markdown]
# (slurm-launcher)=
# ## Running the study on a Slurm cluster
#
# `SlurmCluster` writes one trial job per geometry and a driver that keeps a
# bounded number of jobs in flight. Site settings come from the `QPDK_*`
# environment variables in `site()` above. A Ray script is available for
# allocations that reserve their nodes together.

# %%
driver = (
    f"export QPDK_RUN_OPTIMIZATION=1 "
    f"QPDK_N_TRIALS={N_TRIALS} QPDK_MAX_IN_FLIGHT={MAX_IN_FLIGHT} "
    f"QPDK_PALACE_SIF={shlex.quote(PALACE_SIF)} QPDK_TRIALS_CSV=trials.csv\n"
    "uv run --frozen python notebooks/src/palace_batched_qubit_optimization.py"
)

driver_path = cluster.write_driver_script("build/submit_driver.sbatch", driver)
logger.info(f"wrote {driver_path}")

# %% [markdown]
# Ray uses one allocation for its driver and workers. Use it to group short
# trials into hour-plus batches. Its walltime covers the full study.

# %%
if cluster.solver_cores <= cluster.cores_per_node:
    ray_driver = (
        f"export QPDK_RUN_OPTIMIZATION=1 QPDK_EXECUTOR=ray "
        f"QPDK_N_TRIALS={N_TRIALS} QPDK_MAX_IN_FLIGHT={MAX_IN_FLIGHT} "
        f"QPDK_PALACE_SIF={shlex.quote(PALACE_SIF)} QPDK_TRIALS_CSV=trials.csv\n"
        "uv run --frozen python notebooks/src/palace_batched_qubit_optimization.py"
    )
    ray_cluster = replace(cluster, time_limit=cluster.driver_time_limit)
    ray_path = ray_cluster.write_sbatch_script("build/submit_ray.sbatch", ray_driver)
    logger.info(f"wrote {ray_path}")
else:
    logger.info("Ray needs each trial to fit on one node")

# %% [markdown]
# Submit the driver and follow its log:
#
# ```bash
# sbatch build/submit_driver.sbatch
# tail -f slurm-logs/*-driver-*.out
# ```
#
# Check a known geometry and size the allocation from its measured memory use
# before launching the full sweep.
#
# ## Summary
#
# - The Pareto frontier keeps frequency error, bulk-loss $Q$, and footprint
#   separate. The parallel-coordinates plot exposes their geometry dependence.
# - $T_1=Q/(2\pi f_\text{lin})$ falls with frequency even when $Q$ barely
#   changes. Thin-interface participation is examined separately above and
#   has no assumed loss tangent in the Pareto objective.
# - These are simulated screening results, not absolute device lifetimes.

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
