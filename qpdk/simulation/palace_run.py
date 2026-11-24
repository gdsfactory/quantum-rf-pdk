r"""Add Palace domain-energy output and run meshed directories with MPI.

gsim already provides single-node ``run_local()`` and Palace text-result
loading. This module covers Slurm multi-node MPI, domain-energy output,
port-connectivity checks and junction-port alignment. The domain-energy output
and connectivity checks could move to gsim later.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import gdsfactory as gf
import klayout.db as kdb

from qpdk.simulation.fem import FEM_LAYERS

__all__ = [
    "add_domain_energy_postprocessing",
    "align_junction_port",
    "domain_loss_tangents",
    "palace_command",
    "solve",
    "verify_lumped_port_sheet",
    "verify_port_connectivity",
]


_MPI_THREAD_LIMITS = (
    "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1"
)


def _domain_attributes(config: dict[str, Any]) -> list[int]:
    """Return the domain attribute integers in the order Palace indexes them.

    gsim emits the materials in whatever order the mesh produced them, which is
    not attribute order, so both the postprocessing request and the loss-tangent
    lookup have to sort to agree on what ``p_elec[n]`` refers to.
    """
    return sorted(
        attribute
        for material in config["Domains"]["Materials"]
        for attribute in material["Attributes"]
    )


def add_domain_energy_postprocessing(sim_dir: Path) -> list[int]:
    """Ask Palace to report the field energy in every dielectric domain.

    gsim writes the domain materials but leaves the postprocessing list empty,
    so the domain attributes are read back out of the written config and
    requested one by one. Palace then reports the electric and magnetic energy
    in each, which is what the participation ratios come from.

    Args:
        sim_dir: Directory holding the written ``config.json``.

    Returns:
        The domain attribute integers, in the order Palace indexes them.
    """
    config_path = sim_dir / "config.json"
    config = json.loads(config_path.read_text())
    attributes = _domain_attributes(config)
    config["Domains"]["Postprocessing"]["Energy"] = [
        {"Attributes": [attribute], "Index": index + 1}
        for index, attribute in enumerate(attributes)
    ]
    config_path.write_text(json.dumps(config, indent=2))
    return attributes


def _conductor(regions: gf.Component) -> kdb.Region:
    """Return the merged positive metal of a converted layout."""
    layer = regions.kdb_cell.layout().layer(*FEM_LAYERS["SUPERCONDUCTOR"])
    return kdb.Region(regions.kdb_cell.begin_shapes_rec(layer)).merged()


def _along_x(orientation: float) -> bool:
    """Return whether gsim lays an in-plane lumped port sheet along x."""
    angle = orientation % 360
    return not (45 <= angle < 135 or 225 <= angle < 315)


def verify_lumped_port_sheet(regions: gf.Component, port: str, length: float) -> None:
    """Check that an in-plane lumped port sheet bridges two conductors.

    gsim draws an in-plane lumped port as a sheet centred on the port,
    ``length`` along its orientation and the port width across it. The sheet
    has to end on two separate conductors without covering either. A sheet
    that runs over the pads removes PEC coverage from the overlap in the mesh,
    and a sheet that misses a pad drives nothing. Palace solves both without
    complaint, and the junction mode it returns no longer tracks the junction
    inductance.

    Args:
        regions: Positive-metal layout from :func:`~qpdk.simulation.to_fem_regions`.
        port: Name of the port the sheet is built on.
        length: Sheet extent along the port orientation in µm, as passed to
            gsim's ``add_port``.

    Raises:
        ValueError: If the sheet overlaps conductor by more than one database
            unit at either end or does not touch exactly two separate
            conductors.
    """
    p = regions.ports[port]
    half_length, half_width = length / 2, p.width / 2
    along_x = _along_x(p.orientation)
    if not along_x:
        half_length, half_width = half_width, half_length
    x, y = p.dcenter
    box = kdb.DBox(
        x - half_length, y - half_width, x + half_length, y + half_width
    ).to_itype(regions.kcl.dbu)
    sheet = kdb.Region(box)
    metal = _conductor(regions)
    # Snapping the centre and ends to the grid can put an end one dbu onto a pad.
    core = kdb.Region(box.enlarged(kdb.Vector(-1, 0) if along_x else kdb.Vector(0, -1)))
    if not (core & metal).is_empty():
        msg = (
            f"lumped port {port!r} sheet ({length} µm long) overlaps conductor, "
            "so the mesh loses PEC coverage under it"
        )
        raise ValueError(msg)
    touched = metal.interacting(sheet.sized(1)).count()
    if touched != 2:
        msg = (
            f"lumped port {port!r} sheet ({length} µm long) touches {touched} "
            "conductor(s) instead of bridging two"
        )
        raise ValueError(msg)


def align_junction_port(regions: gf.Component, port: str = "junction") -> float:
    """Turn a junction placement marker into a lumped port across the pad gap.

    A qubit cell's ``junction`` port is a placement marker pointing along the
    junction wire, perpendicular to the pad-to-pad axis. A lumped port has to
    bridge pad to pad, so this rotates the port by 90°, measures the gap from
    the conductor along the rotated axis and moves the port to the midpoint of
    the gap on that axis, snapped to the database grid. Its offset
    perpendicular to that axis and its width are kept. Pass the returned
    length to gsim's ``add_port``. The length is stored in the port's
    ``info["lumped_port_length"]``, and a repeat call returns it unchanged.

    This holds for the two-pad transmons, such as
    :func:`~qpdk.cells.double_pad_transmon_with_bbox`. The ``xmon_transmon``
    and ``unimon`` markers already point across their junction gap, so the
    rotated sheet fails the check and this raises.

    Args:
        regions: Positive-metal layout from :func:`~qpdk.simulation.to_fem_regions`.
            Its port is modified in place.
        port: Name of the junction placement port.

    Returns:
        The gap between the pads along the rotated port, in µm.

    Raises:
        ValueError: If the port is not axis-aligned, does not sit in a gap
            between conductors along its rotated axis, or the resulting sheet
            fails :func:`verify_lumped_port_sheet`.
    """
    p = regions.ports[port]
    if "lumped_port_length" in p.info:
        length = float(p.info["lumped_port_length"])
        verify_lumped_port_sheet(regions, port, length)
        return length
    if p.orientation % 90:
        msg = (
            f"port {port!r} is at {p.orientation:g}°, but must be axis-aligned "
            "because gsim snaps an in-plane lumped port sheet to the x or y axis"
        )
        raise ValueError(msg)
    p.orientation = (p.orientation + 90.0) % 360.0
    along_x = _along_x(p.orientation)
    dbu = regions.kcl.dbu
    x, y = (round(v / dbu) for v in p.dcenter)
    half_width = round(p.width / dbu / 2)
    metal = _conductor(regions)
    bounds = metal.bbox()

    def _strip(lo: int, hi: int) -> kdb.Region:
        """Return the conductor under a port-wide strip from ``lo`` to ``hi``."""
        if along_x:
            return metal & kdb.Region(kdb.Box(lo, y - half_width, hi, y + half_width))
        return metal & kdb.Region(kdb.Box(x - half_width, lo, x + half_width, hi))

    if along_x:
        below, above = _strip(bounds.left, x), _strip(x, bounds.right)
    else:
        below, above = _strip(bounds.bottom, y), _strip(y, bounds.top)
    msg = (
        f"port {port!r} does not sit in a gap between conductors along its rotated axis"
    )
    if below.is_empty() or above.is_empty():
        raise ValueError(msg)
    start = below.bbox().right if along_x else below.bbox().top
    end = above.bbox().left if along_x else above.bbox().bottom
    if end <= start:
        raise ValueError(msg)
    mid = (start + end) / 2 * dbu
    p.dcenter = (mid, y * dbu) if along_x else (x * dbu, mid)
    length = (end - start) * dbu
    verify_lumped_port_sheet(regions, port, length)
    p.info["lumped_port_length"] = length
    return length


def verify_port_connectivity(sim_dir: Path) -> dict[str, int]:
    """Check that every lumped port is attached to metal in the mesh.

    A port rectangle that overhangs thin conductor features makes gsim's
    boolean pipeline delete them, which can leave the port with nothing to
    drive. Nothing downstream complains on its own: the mesh validates, Palace
    runs, and the modes it returns are port-local artifacts whose frequency
    tracks the port inductance rather than the geometry. A sweep would inherit
    that for every trial, so this runs once per built directory.

    The check is deliberately narrow: it verifies each port shares mesh nodes
    with *some* conductor. It cannot tell which conductor, so a port that
    reaches the ground plane but has lost the island it was meant to drive
    still passes. Treat it as a floor, not a proof.

    Args:
        sim_dir: Directory holding the written ``config.json`` and ``palace.msh``.

    Returns:
        Shared node counts keyed by port group name.

    Raises:
        ValueError: If the config has no lumped port or no conductor surfaces
            at all, if a port's mesh group is missing, or if a port shares no
            nodes with any conductor.
    """
    config = json.loads((sim_dir / "config.json").read_text())
    boundaries = config.get("Boundaries", {})
    # gsim omits PEC entirely when no conductor reduces to a planar surface,
    # and omits a port whose surface the boolean pipeline consumed. Both are
    # the failure this function exists to report, so neither may KeyError.
    pec = set(boundaries.get("PEC", {}).get("Attributes", []))
    ports = boundaries.get("LumpedPort", [])
    if not ports:
        msg = (
            f"{sim_dir}/config.json declares no lumped port: its surface was "
            "most likely consumed when the port overhung the conductor"
        )
        raise ValueError(msg)
    if not pec:
        msg = f"{sim_dir}/config.json declares no PEC conductor surfaces"
        raise ValueError(msg)

    # Imported only once the config checks pass: they are pure JSON, and gmsh
    # needs GL system libraries that a bare test runner may not have.
    import gmsh

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.open(str(sim_dir / "palace.msh"))
        # Palace's port Index is not the Gmsh tag; the mesh names the group
        # "P<index>", so resolve by name rather than assuming they match.
        groups = {
            gmsh.model.getPhysicalName(dim, tag): tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
        }

        def nodes(tag: int) -> set[int]:
            found: set[int] = set()
            for entity in gmsh.model.getEntitiesForPhysicalGroup(2, tag):
                _, _, element_nodes = gmsh.model.mesh.getElements(2, int(entity))
                for block in element_nodes:
                    found.update(int(n) for n in block)
            return found

        conductor_nodes: set[int] = set()
        for name, tag in groups.items():
            if tag in pec or name.endswith("_pec"):
                conductor_nodes |= nodes(tag)

        shared = {}
        for port in ports:
            name = f"P{port['Index']}"
            # A CPW port is split into elements, so its surfaces come through
            # as "P2_E0"/"P2_E1" rather than a single "P2".
            tags = [
                tag
                for group, tag in groups.items()
                if group == name or group.startswith(f"{name}_")
            ]
            if not tags:
                msg = (
                    f"{sim_dir}/palace.msh has no {name} surface for the lumped "
                    "port of the same index: it was dropped during meshing"
                )
                raise ValueError(msg)
            port_nodes: set[int] = set()
            for tag in tags:
                port_nodes |= nodes(tag)
            shared[name] = len(port_nodes & conductor_nodes)
    finally:
        if started:
            gmsh.finalize()

    orphaned = [name for name, count in shared.items() if count == 0]
    if orphaned:
        msg = (
            f"lumped port(s) {', '.join(orphaned)} in {sim_dir} share no mesh "
            "nodes with any conductor, so they are orphaned and every mode "
            "Palace returns would be a port-local artifact"
        )
        raise ValueError(msg)
    return shared


def palace_command(*, ranks: int, launcher_args: list[str] | None = None) -> list[str]:
    """Return the command that solves a simulation directory in place.

    The native Palace launcher starts MPI with the requested ranks. Set
    ``QPDK_PALACE_SIF`` to an Apptainer image to run the solver from a
    container, which is how a compute node usually has it; otherwise ``palace``
    is taken from ``PATH``.

    Args:
        ranks: MPI ranks to give Palace.
        launcher_args: Site-specific MPI options, such as a Slurm hostfile.

    Returns:
        The argument vector, ready for ``subprocess.run``.
    """
    sif = os.environ.get("QPDK_PALACE_SIF")
    if not sif:
        command = ["palace", "-np", str(ranks)]
        if launcher_args:
            command += ["-launcher-args", shlex.join(launcher_args)]
        return [*command, "config.json"]
    command = [
        "mpirun",
        *(launcher_args or []),
        "-np",
        str(ranks),
        "palace-x86_64.bin",
        "config.json",
    ]
    # --cleanenv keeps host modules and SLURM_* variables out of the image.
    return [
        "apptainer",
        "exec",
        "--cleanenv",
        "--env",
        _MPI_THREAD_LIMITS,
        sif,
        *command,
    ]


def _slurm_mpi_launcher(sim_dir: Path, ranks: int) -> list[str]:
    """Configure Open MPI across the allocation using the site's SSH host-key checks."""
    sim_dir = sim_dir.resolve()
    nodes = int(os.environ.get("QPDK_MPI_NODES", "1"))
    if nodes == 1:
        return []
    sif = os.environ.get("QPDK_PALACE_SIF")
    if not sif:
        raise ValueError("multi-node Palace needs QPDK_PALACE_SIF")
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    hostnames = subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["scontrol", "show", "hostnames", nodelist], text=True
    ).splitlines()
    if len(hostnames) != nodes or ranks % nodes:
        raise ValueError("MPI ranks must divide evenly across the allocated nodes")

    slots = ranks // nodes
    hostfile = sim_dir / "palace.hosts"
    hostfile.write_text("".join(f"{host} slots={slots}\n" for host in hostnames))
    agent = sim_dir / "palace-mpi-agent"
    agent.write_text(
        "#!/bin/bash\n"
        f"run_dir={shlex.quote(str(sim_dir.resolve()))}\n"
        f"sif={shlex.quote(sif)}\n"
        'node="$1"; shift\n'
        'cmd="$*"\n'
        "printf -v quoted_dir '%q' \"$run_dir\"\n"
        "printf -v quoted_sif '%q' \"$sif\"\n"
        "printf -v quoted_cmd '%q' \"$cmd\"\n"
        'exec ssh -o BatchMode=yes "$node" '
        f'"cd $quoted_dir && apptainer exec --cleanenv --env {_MPI_THREAD_LIMITS} '
        '$quoted_sif /bin/sh -c $quoted_cmd"\n'
    )
    agent.chmod(0o700)
    return [
        "--prtemca",
        "plm_ssh_agent",
        str(agent),
        "--map-by",
        f"ppr:{slots}:node",
        "--bind-to",
        "none",
        "--oversubscribe",
        "--hostfile",
        str(hostfile),
    ]


def solve(sim_dir: Path, *, ranks: int, timeout: float = 1800.0) -> None:
    """Run the solver over an already-meshed simulation directory.

    Args:
        sim_dir: Directory holding ``config.json`` and ``palace.msh``.
        ranks: MPI ranks to give Palace.
        timeout: Seconds before the solve is abandoned.

    Raises:
        RuntimeError: If Palace exits non-zero.
    """
    launcher_args = _slurm_mpi_launcher(sim_dir, ranks)
    with (
        (sim_dir / "run.log").open("w") as log,
        subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            palace_command(ranks=ranks, launcher_args=launcher_args),
            cwd=sim_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if returncode != 0:
        msg = f"Palace exited {returncode}; see {sim_dir}/run.log"
        raise RuntimeError(msg)


def domain_loss_tangents(sim_dir: Path) -> dict[int, float]:
    """Map each domain's postprocessing index to its loss tangent.

    Args:
        sim_dir: Simulation directory holding the written ``config.json``.

    Returns:
        Loss tangent keyed by the index Palace uses in ``domain-E.csv``.
    """
    config = json.loads((sim_dir / "config.json").read_text())
    by_attribute = {
        attribute: material.get("LossTan", 0.0)
        for material in config["Domains"]["Materials"]
        for attribute in material["Attributes"]
    }
    return {
        index + 1: by_attribute[attribute]
        for index, attribute in enumerate(_domain_attributes(config))
    }
