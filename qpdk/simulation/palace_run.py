r"""Add Palace domain-energy output and run meshed directories with MPI.

gsim already provides single-node ``run_local()`` and Palace text-result
loading. This module covers Slurm multi-node MPI, domain-energy output, and
port-connectivity checks. The latter two could move to gsim later.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "add_domain_energy_postprocessing",
    "domain_loss_tangents",
    "palace_command",
    "solve",
    "verify_port_connectivity",
]


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

    Palace is MPI-parallel, so the ranks go to ``mpirun``. Set
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
    binary = "palace-x86_64.bin" if sif else "palace"
    command = [
        "mpirun",
        *(launcher_args or []),
        "-np",
        str(ranks),
        binary,
        "config.json",
    ]
    if sif:
        # --cleanenv keeps host modules and SLURM_* variables out of the image.
        return ["apptainer", "exec", "--cleanenv", sif, *command]
    return command


def _slurm_mpi_launcher(sim_dir: Path, ranks: int) -> list[str]:
    """Configure Open MPI to launch Palace across the Slurm allocation."""
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
        "exec ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
        '-o UserKnownHostsFile=/dev/null "$node" '
        '"cd $quoted_dir && apptainer exec --cleanenv $quoted_sif /bin/sh -c $quoted_cmd"\n'
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
    with (sim_dir / "run.log").open("w") as log:
        # The command comes from :func:`palace_command`, which reads a
        # site-level environment variable rather than anything a trial supplies.
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            palace_command(ranks=ranks, launcher_args=launcher_args),
            cwd=sim_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if result.returncode != 0:
        msg = f"Palace exited {result.returncode}; see {sim_dir}/run.log"
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
