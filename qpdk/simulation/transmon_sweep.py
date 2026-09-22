"""Evaluate double-pad transmon geometries with Palace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gdsfactory as gf

from qpdk import PDK
from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.simulation.fem import single_chip_stack, to_fem_regions
from qpdk.simulation.palace_run import (
    add_domain_energy_postprocessing,
    evaluate_simulation,
    solve,
    verify_port_connectivity,
)
from qpdk.tech import LAYER

LAYOUT_GRID_UM = 0.01


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


@gf.cell
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
    pad_size = tuple(  # ty: ignore[invalid-assignment]
        gf.snap.snap_to_grid(size, grid_factor=LAYOUT_GRID_UM * 1000)
        for size in pad_size
    )
    pad_gap = gf.snap.snap_to_grid(pad_gap, grid_factor=LAYOUT_GRID_UM * 1000)

    c = gf.Component()
    ref = c << double_pad_transmon_with_bbox(pad_size=pad_size, pad_gap=pad_gap)
    c.add_ports(ref.ports)
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(margin, margin))
    return c


def build_simulation(
    layout: Layout,
    sim_dir: Path,
    *,
    junction_inductance: float = 10e-9,
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
    from gsim.palace import EigenmodeSim

    # Worker imports need the PDK active to resolve cell layers.
    PDK.activate()

    sim_dir.mkdir(parents=True, exist_ok=True)
    regions = to_fem_regions(
        transmon_sim_layout(layout.pad_size, layout.pad_gap, lateral_margin)
    )
    # The cell's junction marker is perpendicular to the pad-to-pad axis.
    regions.ports["junction"].orientation = 0.0

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
        length=layout.pad_gap + 10.0,
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
        "Tol": 1e-4,
        "MGMaxLevels": 2,
        "PCMatReal": False,
        "PCMatShifted": False,
        "ComplexCoarseSolve": True,
        "ColumnOrdering": "ParMETIS",
    })
    config["Solver"]["Eigenmode"]["Tol"] = 1e-3
    config_path.write_text(json.dumps(config, indent=2))
    add_domain_energy_postprocessing(sim_dir)
    verify_port_connectivity(sim_dir)
    return sim_dir


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
    build_simulation(layout, sim_dir, **build_settings)
    solve(sim_dir, ranks=ranks, timeout=timeout)
    return (
        evaluate_simulation(sim_dir) | params | {"footprint_mm2": layout.footprint_mm2}
    )
