r"""Run a Palace simulation directory and read the result back.

These are the pieces a design sweep needs and gsim does not provide: its
``sim.run()`` submits to a hosted service rather than a local binary or a
scheduler, and its ``write_config()`` leaves the domain-energy postprocessing
empty, so anything wanting participation ratios has to ask for them itself.

Nothing here knows what is being simulated. What is being swept, and how a
layout turns into a simulation, belongs to the study that uses these; see the
``palace_batched_qubit_optimization`` notebook for a worked one.

Note:
    Several of these are arguably gsim's job and would be better upstream than
    vendored here:

    - :func:`add_domain_energy_postprocessing` fills in a config block gsim
      writes empty.
    - :func:`verify_port_connectivity` is a mesh sanity check ``write_config``
      could make itself, and the failure it catches is silent otherwise.
    - :func:`palace_command` and :func:`solve` are the local and cluster
      counterpart of ``sim.run()``.
    - :func:`read_palace_csv` duplicates header handling gsim already does
      internally in ``gsim.palace.results``.

    This module requires the optional ``models`` extra for gsim, and the Palace
    solver itself to actually run a simulation.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "JUNCTION_PARTICIPATION_FLOOR",
    "add_domain_energy_postprocessing",
    "domain_loss_tangents",
    "evaluate_simulation",
    "palace_command",
    "read_palace_csv",
    "solve",
    "verify_port_connectivity",
]


# Junction participation a mode must reach to be a candidate qubit mode.
JUNCTION_PARTICIPATION_FLOOR = 0.05


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


def read_palace_csv(path: Path) -> list[dict[str, str]]:
    """Read a Palace CSV, normalising its padded column headers."""
    table = pd.read_csv(path, skipinitialspace=True)
    table.columns = [column.strip() for column in table.columns]
    return table.to_dict(orient="records")


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


def evaluate_simulation(sim_dir: Path) -> dict[str, Any]:
    """Extract the objectives and loss diagnostics from a finished solve.

    The qubit-like mode is identified as the one storing energy in the junction
    port, which separates it from the packaging and ground-plane modes the solve
    also returns.

    Args:
        sim_dir: Simulation directory Palace wrote its ``output/`` into.

    Returns:
        A row of results: ``f_linear`` in Hz, ``T1`` in seconds, dielectric
        ``quality_factor`` from domain participations, the raw
        ``eigenmode_quality_factor``, junction participation, and per-domain
        electric-energy participations.

    Note:
        ``f_linear`` is the mode frequency from a junction-as-inductor solve.
        The real 0-1 transition
        sits about :math:`E_C/h` below it (of order 0.1-0.2 GHz for these
        geometries), so a study targeting a transition frequency should either
        offset its target or add the anharmonic correction from the extracted
        capacitance.

    Raises:
        FileNotFoundError: If the solve produced no eigenmode table.
        ValueError: If no mode carries junction energy.
    """
    output = sim_dir / "output" / "palace"
    eig_path = output / "eig.csv"
    if not eig_path.exists():
        msg = f"Palace produced no eig.csv in {output}; see {sim_dir}/run.log"
        raise FileNotFoundError(msg)

    eig = read_palace_csv(eig_path)
    junction = read_palace_csv(output / "port-EPR.csv")
    domains = read_palace_csv(output / "domain-E.csv")

    if not len(eig) == len(junction) == len(domains):
        msg = (
            f"Palace tables disagree on mode count in {output}: "
            f"eig={len(eig)}, port-EPR={len(junction)}, domain-E={len(domains)}"
        )
        raise ValueError(msg)

    participations = np.array([abs(float(row["p[1]"])) for row in junction])
    frequencies = np.array([float(row["Re{f} (GHz)"]) for row in eig])

    # The qubit mode is the *lowest* mode the junction takes part in. Picking
    # the largest participation instead looks equivalent and is not: a
    # higher-order mode can carry marginally more junction energy than the
    # fundamental, and on a real sweep that near-tie silently returned harmonic
    # modes at several times the qubit frequency. The floor sits far above the
    # ~1e-6 a spurious mode shows and below the ~0.1 a weak junction mode can show.
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
        # The ring-down time of the mode, which is what T1 is for a linearised
        # transmon limited by dielectric loss.
        "T1": quality_factor / (2.0 * np.pi * f_linear),
        # Magnitude: the sign Palace reports is the port current's
        # orientation relative to the mode, not a negative participation.
        "junction_participation": float(participations[index]),
        "num_modes": len(eig),
        "mode_index": index + 1,
    }
    for domain, participation in domain_participations.items():
        row[f"participation_domain{domain}"] = participation
    return row
