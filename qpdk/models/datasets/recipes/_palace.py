"""Palace provenance, resumable extraction and quality checks for dataset experiments."""

# Native mesh imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from gdsfactory import Component
    from gsim.palace import ElectrostaticSim

from qpdk import logger
from qpdk.models.datasets.capacitance import check_maxwell


def runtime(
    executable: Path | None, sif: Path | None, binary: str
) -> tuple[dict, dict]:
    """Resolve a runtime, recording its content hash rather than its host path."""
    if executable is not None and sif is not None:
        raise ValueError("Choose an executable or a container, not both")
    library_dir = None
    if executable is None and sif is None:
        from gsim.palace.runtime import (
            resolve_palace_binary,
            resolve_palace_library_dir,
        )

        executable = resolve_palace_binary()
        if executable is None:
            raise FileNotFoundError("No Palace runtime; pass --executable or --sif")
        resolved_libraries = resolve_palace_library_dir()
        # The two SDK resolvers can select different installations.
        if (
            resolved_libraries is not None
            and executable.resolve().parent
            == resolved_libraries.resolve().parent / "bin"
        ):
            library_dir = resolved_libraries.resolve()
    selected = sif if sif is not None else executable
    with selected.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "palace_executable": executable.resolve() if executable else None,
        "palace_sif_path": sif.resolve() if sif else None,
        "use_apptainer": sif is not None,
        "container_binary": binary if sif else None,
        **({"palace_library_dir": library_dir} if library_dir is not None else {}),
    }, {"runtime_sha256": digest, "container_binary": binary if sif else None}


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


def extract(
    component: Component,
    domain_bounds: tuple[float, float, float, float],
    terminal_ports: dict[str, str],
    *,
    height: float,
    inputs: dict,
    workdir: Path,
    execution: dict,
    processes: int,
    near_mesh: float,
    far_mesh: float,
    permittivity: float,
    order: int,
    tolerance: float,
    save_fields: bool = False,
    normalization_depth_um: float | None = None,
    minimum_feature_elements: float = 4,
    mesh: Callable[[ElectrostaticSim], None] | None = None,
) -> dict[str, np.ndarray | float]:
    """Resume a matching solve and return capacitance with numerical diagnostics."""
    import gdsfactory as gf
    from gsim.palace import ElectrostaticSim
    from gsim.palace.capacitance import load_capacitance

    from qpdk.tech import LAYER

    terminals = tuple(terminal_ports)
    inputs = {
        **inputs,
        "terminal_order": terminals,
        "minimum_feature_elements": minimum_feature_elements,
        "mesh_threads": processes,
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    run = workdir.resolve() / key
    complete = run / "complete.json"
    if complete.is_file():
        logger.info(f"Reusing Palace result: {run}")
    else:
        run.mkdir(parents=True, exist_ok=True)
        (run / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        sim = ElectrostaticSim()
        sim.set_geometry(component)
        sim.set_output_dir(run)
        if mesh is not None:
            mesh(sim)
            actual = [terminal.name for terminal in sim.terminals]
            if actual != list(terminals):
                raise ValueError(
                    f"Mesh callback terminals {actual} != {list(terminals)}"
                )
        else:
            sim.mesh_sheets(
                conductor_layer=gf.get_layer_tuple(LAYER.M1_DRAW),
                terminal_ports=terminal_ports,
                domain_bounds=domain_bounds,
                height=height,
                near_mesh=near_mesh,
                far_mesh=far_mesh,
                permittivity=permittivity,
                normalization_depth_um=normalization_depth_um,
                minimum_feature_elements=minimum_feature_elements,
                verbose=True,
                num_threads=processes,
            )
        # Warm starts change the norm used to normalize the logged residuals.
        sim.set_solver(
            order=order,
            tolerance=tolerance,
            max_iterations=1000,
            solver_type="Default",
            preconditioner="BoomerAMG",
        )
        sim.set_electrostatic(save_fields=len(terminals) if save_fields else 0)
        path = sim.write_config()
        config = json.loads(path.read_text())
        # Residual ratios need zero starts for every terminal, across gsim versions.
        config["Solver"]["Linear"]["InitialGuess"] = False
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        shutil.rmtree(run / "output", ignore_errors=True)
        _run_palace(execution, run, processes)
    quality = diagnostics(run, terminals=len(terminals), tolerance=tolerance)
    matrices = load_capacitance(run, terminal_names=terminals)
    check_maxwell(matrices.maxwell)
    if problems := matrices.problems(rtol=1e-8):
        raise ValueError(f"Inconsistent Palace results in {run}: {problems}")
    complete.write_text(json.dumps({"terminals": terminals}), encoding="utf-8")
    return {"maxwell_capacitance": matrices.maxwell, **quality}


def _run_palace(execution: dict, run: Path, processes: int) -> None:
    """Keep the selected binary, MPI ranks and failed solver output explicit."""
    environment = None
    if execution["use_apptainer"]:
        command = [
            "apptainer",
            "exec",
            "--cleanenv",
            "--env",
            "OMP_NUM_THREADS=1",
            str(execution["palace_sif_path"]),
            "mpirun",
            "--oversubscribe",
            "-np",
            str(processes),
            execution["container_binary"],
            "config.json",
        ]
    else:
        executable = Path(execution["palace_executable"])
        environment = os.environ.copy()
        if library_dir := execution.get("palace_library_dir"):
            prior = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = (
                f"{library_dir}{os.pathsep}{prior}" if prior else str(library_dir)
            )
            if shutil.which("mpirun", path=str(executable.parent)):
                prior = environment.get("PATH", "")
                environment["PATH"] = (
                    f"{executable.parent}{os.pathsep}{prior}"
                    if prior
                    else str(executable.parent)
                )
        with executable.open("rb") as stream:
            header = stream.read(4)
        # The Palace script accepts MPI flags; the native executable does not.
        if header in {
            b"\x7fELF",
            b"\xfe\xed\xfa\xce",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
        }:
            launcher = shutil.which("mpirun", path=str(executable.parent)) or "mpirun"
            command = [launcher, "-np", str(processes), str(executable), "config.json"]
            environment["OMP_NUM_THREADS"] = "1"
        else:
            command = [
                str(executable),
                "-np",
                str(processes),
                "-nt",
                "1",
                "config.json",
            ]
    logger.info(f"Running Palace in {run}")
    with (run / "solver.log").open("w", encoding="utf-8") as log:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            cwd=run,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )


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
    from gsim.palace.results import load_static_solver_metrics

    try:
        metrics = load_static_solver_metrics(log)
    except ValueError as error:
        raise RuntimeError("Missing solver convergence in Palace log") from error
    if [item["index"] for item in metrics] != list(range(1, terminals + 1)):
        raise RuntimeError("Missing terminal solves in Palace log")
    ratio = max(item["relative_residual"] for item in metrics)
    if ratio > tolerance * 1.00001:
        raise RuntimeError(f"Solver residual {ratio:g} exceeds tolerance {tolerance:g}")
    return {
        "solver_relative_residual": ratio,
        "solver_iterations": float(max(item["iterations"] for item in metrics)),
    }
