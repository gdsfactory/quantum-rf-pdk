"""Sheet meshing and Palace execution shared by the dataset experiments."""

# Native mesh imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import re
import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from gsim.palace.mesh.generator import MeshResult
    from meshwell.geometry_entity import GeometryEntity
    from meshwell.model import ModelManager
    from shapely import Polygon

from qpdk import logger
from qpdk.models.datasets.capacitance import check_maxwell


def runtime(
    workdir: Path, executable: Path | None, sif: Path | None, binary: str
) -> tuple[Path, dict]:
    """Resolve a runtime, recording its content hash rather than its host path."""
    if executable is not None and sif is not None:
        raise ValueError("Choose an executable or a container, not both")
    if executable is None and sif is None:
        from gsim.palace.runtime import (
            resolve_palace_binary,
        )

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
                isinstance(target, ast.Name)
                and (target.id == "GRID" or target.id.endswith("_GRID"))
                for target in node.targets
            )
        )
    ]
    installed = importlib.metadata.distribution("gsim").read_text("direct_url.json")
    qpdk_url = importlib.metadata.distribution("qpdk").read_text("direct_url.json")
    return {
        "solver": "Palace",
        "linear_initial_guess": False,
        "generator_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(node) for node in nodes).encode()
        ).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostics": {
            "fem_error_indicator_norm": "Palace energy-normalized recovered-flux estimator norm; not a capacitance error bound",
            "solver_relative_residual": "Maximum final/initial KSP residual norm over terminal solves",
            "solver_iterations": "Maximum KSP iteration count over terminal solves",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "gsim",
                "meshwell",
                "gmsh",
                "gdsfactory",
                "qpdk",
                "jax",
                "jaxlib",
                "jaxellip",
                "sax",
                "numpy",
                "polars",
            )
        },
        "qpdk_revision": json
        .loads(qpdk_url or "{}")
        .get("vcs_info", {})
        .get("commit_id"),
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
    import gmsh
    from meshwell.model import ModelManager
    from meshwell.polyprism import PolyPrism
    from meshwell.polysurface import (
        PolySurface,
    )

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
    return _mesh_entities(
        model,
        entities,
        sheets,
        dim=3,
        near_mesh=near_mesh,
        far_mesh=far_mesh,
        path=path,
    )


def mesh_cross_section(
    sheets: dict[str, Polygon],
    footprint: Polygon,
    *,
    height: float,
    near_mesh: float,
    far_mesh: float,
    path: Path,
) -> MeshResult:
    """Mesh the transverse section of a uniform line with meshwell curves."""
    import gmsh
    from meshwell.model import ModelManager
    from meshwell.polyline import PolyLine
    from meshwell.polysurface import PolySurface
    from shapely import LineString, box

    if gmsh.isInitialized():
        raise RuntimeError(
            "Meshwell requires its own Gmsh session; finalize the existing session first"
        )
    lower, upper = footprint.bounds[1], footprint.bounds[3]
    model = ModelManager(n_threads=1, filename=str(path.with_suffix("")))
    # Different snapping grids leave conductor edges outside the dielectric mesh.
    entities = [
        PolySurface(
            box(lower, 0, upper, height), physical_name="air", point_tolerance=1e-8
        ),
        PolySurface(
            box(lower, -height, upper, 0),
            physical_name="silicon",
            point_tolerance=1e-8,
        ),
    ]
    for name, polygon in sheets.items():
        polygons = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        lines = [
            LineString([(part.bounds[1], 0), (part.bounds[3], 0)]) for part in polygons
        ]
        entities.append(PolyLine(lines, physical_name=name, point_tolerance=1e-8))
    return _mesh_entities(
        model,
        entities,
        sheets,
        dim=2,
        near_mesh=near_mesh,
        far_mesh=far_mesh,
        path=path,
    )


def _mesh_entities(
    model: ModelManager,
    entities: list[GeometryEntity],
    sheets: dict[str, Polygon],
    *,
    dim: Literal[2, 3],
    near_mesh: float,
    far_mesh: float,
    path: Path,
) -> MeshResult:
    """Keep meshing, physical groups and Palace export consistent across dimensions."""
    import gmsh
    from gsim.palace.mesh.generator import MeshResult
    from meshwell.resolution import ThresholdField

    try:
        model.cad.process_entities(entities, interface_delimiter="___")
        field = ThresholdField(
            apply_to="surfaces" if dim == 3 else "curves",
            sizemin=near_mesh,
            sizemax=far_mesh,
            distmin=1,
            distmax=25,
        )
        resolutions = {name: [field] for name in sheets if name != "ground"}
        ground_parts = (
            list(sheets["ground"].geoms)
            if hasattr(sheets["ground"], "geoms")
            else [sheets["ground"]]
        )
        if dim == 2 and len(ground_parts) == 3:
            strip = min(part.bounds[3] - part.bounds[1] for part in ground_parts)
            # Refine the inner strip without refining the long outer ground rails.
            resolutions["ground"] = [
                ThresholdField(
                    apply_to="curves",
                    max_mass=strip * 1.001,
                    sizemin=min(near_mesh, strip * near_mesh / (4 * 0.14)),
                    sizemax=far_mesh,
                    distmin=min(1, strip / 4),
                    distmax=25,
                )
            ]
        model.mesh.process_geometry(
            dim=dim,
            default_characteristic_length=far_mesh,
            resolution_specs=resolutions,
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
            (group_dim, tag)
            for group_dim, tag in gmsh.model.getPhysicalGroups(dim - 1)
            if gmsh.model.getPhysicalName(group_dim, tag) not in sheets
        ])
        for group_dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(group_dim, tag)
            if group_dim == dim:
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
        count = sum(len(elements) for elements in gmsh.model.mesh.getElements(dim)[1])
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        model.save_to_mesh(path)
        return MeshResult(
            mesh_path=path,
            output_dir=path.parent,
            groups=groups,
            mesh_stats={"tetrahedra" if dim == 3 else "triangles": count},
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
    normalization_depth_um: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Resume a matching solve and return capacitance with numerical diagnostics."""
    from gsim.common.stack import LayerStack
    from gsim.palace import ElectrostaticSim
    from gsim.palace.capacitance import (
        load_capacitance,
    )
    from gsim.palace.mesh.generator import (
        write_config,
    )
    from gsim.palace.models.ports import (
        TerminalConfig,
    )
    from pydantic import PrivateAttr

    class SheetSimulation(ElectrostaticSim):
        """Use gsim execution with an externally generated mesh."""

        _last_mesh_result: MeshResult | None = PrivateAttr(default=None)

    terminals = tuple(name for name in sheets if name != "ground")
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    run = workdir.resolve() / key
    complete = run / "complete.json"
    if complete.is_file():
        logger.info(f"Reusing Palace result: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        (run / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        mesh_function = (
            mesh_sheets if normalization_depth_um is None else mesh_cross_section
        )
        mesh = mesh_function(
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
        config_path = write_config(
            mesh,
            LayerStack(materials={"silicon": {"permittivity": permittivity}}),
            [],
            simulation_type="electrostatic",
            numerical_config=sim.solver,
            electrostatic_config=sim.solver.electrostatic,
            terminals=[TerminalConfig(name=name, layer=name) for name in terminals],
            absorbing_boundary=False,
        )
        # Warm starts change the norm used to normalize the logged residuals.
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["Solver"]["Linear"]["InitialGuess"] = False
        if normalization_depth_um is not None:
            # Palace reports 2D capacitance for an implicit depth equal to Model.Lc.
            config["Model"]["Lc"] = normalization_depth_um
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
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
    quality = diagnostics(run, terminals=len(terminals), tolerance=tolerance)
    matrices = load_capacitance(run, terminal_names=terminals)
    check_maxwell(matrices.maxwell)
    if problems := matrices.problems(rtol=1e-8):
        raise ValueError(f"Inconsistent Palace results in {run}: {problems}")
    complete.write_text(json.dumps({"terminals": terminals}), encoding="utf-8")
    return {"maxwell_capacitance": matrices.maxwell, **quality}


def diagnostics(run: Path, *, terminals: int, tolerance: float) -> dict[str, float]:
    """Read Palace convergence and error indicators, rejecting incomplete solves.

    With zero initial guesses, the residual is normalized by its initial KSP norm.
    The FEM indicator is Palace's energy-normalized flux recovery estimate;
    it is not a relative error bound on individual capacitance entries.

    Returns:
        Worst terminal residual and iteration count, and the global FEM indicator.

    Raises:
        RuntimeError: If a terminal failed to converge or diagnostics are invalid or missing.
    """
    from gsim.palace.results import (
        load_text_results,
    )

    try:
        log = (run / "solver.log").read_text(encoding="utf-8")
        indicators = load_text_results({
            "error-indicators.csv": run / "output/palace/error-indicators.csv"
        }).error_indicators
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing Palace diagnostics in {run}") from error
    quality = solver_diagnostics(
        log,
        terminals=terminals,
        tolerance=tolerance,
    )
    if (
        indicators is None
        or not math.isfinite(indicators["norm"])
        or indicators["norm"] < 0
    ):
        raise RuntimeError(f"Invalid FEM error indicator in {run}")
    return {"fem_error_indicator_norm": indicators["norm"], **quality}


def solver_diagnostics(
    log: str, *, terminals: int, tolerance: float
) -> dict[str, float]:
    """Check every electrostatic terminal's linear solve and return the worst metrics.

    Returns:
        Final-to-initial KSP residual ratio and maximum terminal iteration count.

    Raises:
        RuntimeError: If terminal solves are missing or fail the requested tolerance.
    """
    blocks = re.split(r"\nIt \d+/\d+: Index = \d+[^\n]*\n", log)[1:]
    ratios, iterations = [], []
    if len(blocks) != terminals:
        raise RuntimeError("Missing terminal solves in Palace log")
    for block in blocks:
        solver_log = block.split("Updating solution error estimates")[0]
        residuals = [
            float(value) for value in re.findall(r"KSP residual norm (\S+)", solver_log)
        ]
        converged = re.findall(r"solver converged in (\d+) iterations?", solver_log)
        if (
            len(residuals) < 2
            or len(converged) != 1
            or not all(math.isfinite(v) and v >= 0 for v in residuals)
            or residuals[0] <= 0
        ):
            raise RuntimeError("Missing solver convergence in Palace log")
        ratio = residuals[-1] / residuals[0]
        if not math.isfinite(ratio) or ratio > tolerance * 1.00001:
            raise RuntimeError(
                f"Solver residual {ratio:g} exceeds tolerance {tolerance:g}"
            )
        ratios.append(ratio)
        iterations.append(int(converged[-1]))
    return {
        "solver_relative_residual": max(ratios),
        "solver_iterations": float(max(iterations)),
    }
