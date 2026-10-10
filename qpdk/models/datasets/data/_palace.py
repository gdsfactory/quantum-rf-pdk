"""Palace provenance, resumable extraction and quality checks for dataset experiments."""

# Native mesh imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gdsfactory import Component

from qpdk import logger
from qpdk.models.datasets.capacitance import check_maxwell


def runtime(
    executable: Path | None, sif: Path | None, binary: str
) -> tuple[dict, dict]:
    """Resolve a runtime, recording its content hash rather than its host path."""
    if executable is not None and sif is not None:
        raise ValueError("Choose an executable or a container, not both")
    if executable is None and sif is None:
        from gsim.palace.runtime import resolve_palace_binary

        executable = resolve_palace_binary()
        if executable is None:
            raise FileNotFoundError("No Palace runtime; pass --executable or --sif")
    selected = sif if sif is not None else executable
    with selected.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "palace_executable": executable.resolve() if executable else None,
        "palace_sif_path": sif.resolve() if sif else None,
        "use_apptainer": sif is not None,
        "container_binary": binary if sif else None,
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
) -> dict[str, np.ndarray | float]:
    """Resume a matching solve and return capacitance with numerical diagnostics."""
    import gdsfactory as gf
    from gsim.palace import ElectrostaticSim
    from gsim.palace.capacitance import load_capacitance

    from qpdk.tech import LAYER

    terminals = tuple(terminal_ports)
    inputs = {**inputs, "minimum_feature_elements": minimum_feature_elements}
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
        )
        # Warm starts change the norm used to normalize the logged residuals.
        sim.set_solver(
            order=order,
            tolerance=tolerance,
            max_iterations=1000,
            preconditioner="BoomerAMG",
            initial_guess=False,
        )
        sim.set_electrostatic(save_fields=len(terminals) if save_fields else 0)
        sim.write_config()
        shutil.rmtree(run / "output", ignore_errors=True)
        sim.run_local(
            **execution,
            num_processes=processes,
            num_threads=1,
            verbose=True,
            log_path=run / "solver.log",
        )
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
