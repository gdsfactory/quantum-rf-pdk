"""Sheet meshing and Palace execution shared by the dataset experiments."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import logging
import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh
import numpy as np
from gsim.common.stack import LayerStack
from gsim.palace import ElectrostaticSim
from gsim.palace.capacitance import load_capacitance
from gsim.palace.mesh.generator import MeshResult, write_config
from gsim.palace.models.ports import TerminalConfig
from gsim.palace.runtime import resolve_palace_binary
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.resolution import ThresholdField
from pydantic import PrivateAttr
from shapely import Polygon

from qpdk import logger
from qpdk.models.datasets.capacitance import check_maxwell


class SheetSimulation(ElectrostaticSim):
    """Use gsim execution with an externally generated mesh."""

    # gsim initializes this attribute only in its native mesh() implementation.
    _last_mesh_result: MeshResult | None = PrivateAttr(default=None)


def runtime(
    workdir: Path, executable: Path | None, sif: Path | None, binary: str
) -> tuple[Path, dict]:
    """Resolve a runtime, recording its content hash rather than its host path."""
    if executable is not None and sif is not None:
        raise ValueError("Choose an executable or a container, not both")
    if executable is None and sif is None:
        executable = resolve_palace_binary()
        if executable is None:
            raise FileNotFoundError("No Palace runtime; pass --executable or --sif")
    selected = sif if sif is not None else executable
    with selected.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if sif is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        executable = workdir.resolve() / "palace-container"
        wrapper = (
            '#!/bin/sh\nset -eu\nranks=1\nwhile [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n    -np) ranks="$2"; shift 2 ;;\n'
            "    -nt) shift 2 ;;\n    *) break ;;\n  esac\ndone\n"
            f"exec apptainer exec --cleanenv {shlex.quote(str(sif.resolve()))} "
            f'mpirun --oversubscribe -np "$ranks" {shlex.quote(binary)} "$@"\n'
        )
        # Replace the inode; a completed launcher can still be open on another node.
        with TemporaryDirectory(dir=workdir) as staging:
            staged = Path(staging) / executable.name
            staged.write_text(wrapper, encoding="utf-8")
            staged.chmod(0o755)
            staged.replace(executable)
    return executable.resolve(), {
        "runtime_sha256": digest,
        "container_binary": binary if sif else None,
    }


def fingerprint(script: Path) -> dict:
    """Invalidate cached physics changes while allowing a grid to be extended."""
    source = script.read_text(encoding="utf-8")
    nodes = [
        node
        for node in ast.parse(source).body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GRID"
                for target in node.targets
            )
        )
    ]
    installed = importlib.metadata.distribution("gsim").read_text("direct_url.json")
    return {
        "solver": "Palace",
        "generator_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(node) for node in nodes).encode()
        ).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gsim", "meshwell", "gmsh", "gdsfactory", "qpdk")
        },
        "gsim_revision": json
        .loads(installed or "{}")
        .get("vcs_info", {})
        .get("commit_id"),
    }


def mesh_sheets(
    sheets: dict[str, Polygon],
    footprint: Polygon,
    *,
    height: float,
    near_mesh: float,
    far_mesh: float,
    path: Path,
) -> MeshResult:
    """Mesh zero-thickness conductors between equal-height air and substrate volumes."""
    if gmsh.isInitialized():
        raise RuntimeError(
            "Meshwell requires its own Gmsh session; finalize the existing session first"
        )
    model = ModelManager(n_threads=1, filename=str(path.with_suffix("")))
    entities = [
        PolyPrism(footprint, {0: 0, height: 0}, physical_name="air"),
        PolyPrism(footprint, {-height: 0, 0: 0}, physical_name="silicon"),
        *(PolySurface(polygon, physical_name=name) for name, polygon in sheets.items()),
    ]
    try:
        model.cad.process_entities(entities, interface_delimiter="___")
        field = ThresholdField(
            apply_to="surfaces",
            sizemin=near_mesh,
            sizemax=far_mesh,
            distmin=1,
            distmax=25,
        )
        model.mesh.process_geometry(
            dim=3,
            default_characteristic_length=far_mesh,
            resolution_specs={name: [field] for name in sheets if name != "ground"},
            verbosity=0,
        )
        groups = {
            "volumes": {},
            "pec_surfaces": {},
            "conductor_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        }
        # Palace rejects faces exported twice as conductor and dielectric interface.
        gmsh.model.removePhysicalGroups([
            (dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups(2)
            if gmsh.model.getPhysicalName(dim, tag) not in sheets
        ])
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            if dim == 3:
                groups["volumes"][name] = {"phys_group": tag}
            elif name in sheets:
                groups["pec_surfaces"][name] = {"phys_group": tag}
        if set(groups["pec_surfaces"]) != set(sheets) or set(groups["volumes"]) != {
            "air",
            "silicon",
        }:
            raise ValueError(
                "Mesh physical groups do not match the conductors and dielectric domains"
            )
        count = sum(len(elements) for elements in gmsh.model.mesh.getElements(3)[1])
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        model.save_to_mesh(path)
        return MeshResult(
            mesh_path=path,
            output_dir=path.parent,
            groups=groups,
            mesh_stats={"tetrahedra": count},
        )
    finally:
        model.finalize()


def extract(
    sheets: dict[str, Polygon],
    footprint: Polygon,
    *,
    height: float,
    inputs: dict,
    workdir: Path,
    executable: Path,
    processes: int,
    near_mesh: float,
    far_mesh: float,
    permittivity: float,
    order: int,
    tolerance: float,
    save_fields: bool = False,
) -> np.ndarray:
    """Resume a matching solve, or mesh and solve with gsim's public APIs."""
    terminals = tuple(name for name in sheets if name != "ground")
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    run = workdir.resolve() / key
    complete = run / "complete.json"
    if complete.is_file():
        logger.info(f"Reusing Palace result: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        (run / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        mesh = mesh_sheets(
            sheets,
            footprint,
            height=height,
            near_mesh=near_mesh,
            far_mesh=far_mesh,
            path=run / "palace.msh",
        )
        sim = SheetSimulation()
        sim.set_output_dir(run)
        sim.set_solver(
            order=order,
            tolerance=tolerance,
            max_iterations=1000,
            preconditioner="BoomerAMG",
        )
        sim.set_electrostatic(save_fields=len(terminals) if save_fields else 0)
        write_config(
            mesh,
            LayerStack(materials={"silicon": {"permittivity": permittivity}}),
            [],
            simulation_type="electrostatic",
            numerical_config=sim.solver,
            electrostatic_config=sim.solver.electrostatic,
            terminals=[TerminalConfig(name=name, layer=name) for name in terminals],
            absorbing_boundary=False,
        )
        shutil.rmtree(run / "output", ignore_errors=True)
        handler = logging.FileHandler(run / "solver.log", mode="w")
        palace_logger = logging.getLogger("gsim.palace.base")
        palace_logger.addHandler(handler)
        try:
            sim.run_local(
                palace_executable=executable,
                use_apptainer=False,
                num_processes=processes,
                num_threads=1,
                verbose=True,
            )
        finally:
            palace_logger.removeHandler(handler)
            handler.close()
    if (run / "solver.log").read_text().count("solver converged in") != len(terminals):
        raise RuntimeError(
            f"Palace did not converge for every terminal; see {run / 'solver.log'}"
        )
    matrices = load_capacitance(run, terminal_names=terminals)
    check_maxwell(matrices.maxwell)
    if problems := matrices.problems(rtol=1e-8):
        raise ValueError(f"Inconsistent Palace results in {run}: {problems}")
    complete.write_text(json.dumps({"terminals": terminals}), encoding="utf-8")
    return matrices.maxwell
