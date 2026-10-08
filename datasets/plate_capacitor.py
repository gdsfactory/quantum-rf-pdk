# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models]",
#   "gsim @ git+https://github.com/gdsfactory/gsim.git@05c6c93cc14522f8a6a78a08b28116e242cdf1c0",
# ]
# [tool.uv.sources]
# qpdk = { path = "..", editable = true }
# ///
"""Generate Maxwell capacitance data with gsim and Palace.

Run ``uv run --script datasets/plate_capacitor.py --help`` from a checkout.
Edit GRID and SETTINGS to define your experiment. The electrodes are perfect
conductor sheets on silicon with a separate coplanar ground frame. Mesh and
domain sensitivity must be checked before relying on small differences.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import os
import shlex
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import starmap
from pathlib import Path
from uuid import uuid4

import gdsfactory as gf
import numpy as np
from gsim.palace import ElectrostaticSim
from gsim.palace.capacitance import load_capacitance
from gsim.palace.runtime import resolve_palace_binary
from pydantic import PrivateAttr
from shapely import Point, Polygon

from qpdk import PDK, __version__, logger
from qpdk.cells import plate_capacitor
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.capacitance import check_maxwell
from qpdk.models.datasets.generate import sweep, write
from qpdk.tech import LAYER, material_properties

NAME = "plate_capacitor_palace"
GRID = {
    "length": [40.0, 80.0, 120.0],
    "width": [5.0, 10.0, 20.0],
    "gap": [4.0, 7.0, 10.0],
}
TERMINALS = ("o1", "o2")


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Lengths in µm; planar perfect conductors at the silicon interface."""

    near_mesh: float = 0.55
    far_mesh: float = 25.0
    domain_pad: float = 100.0
    ground_clearance: float = 10.0
    ground_pad: float = 45.0
    permittivity: float = material_properties["Si"]["relative_permittivity"]
    order: int = 2
    tolerance: float = 1e-9
    save_fields: bool = False

    def __post_init__(self) -> None:
        """Reject an invalid ground frame, mesh or numerical setting."""
        if not 0 < self.ground_clearance < self.ground_pad < self.domain_pad:
            raise ValueError("Require 0 < ground_clearance < ground_pad < domain_pad")
        if not 0 < self.near_mesh <= self.far_mesh or self.permittivity <= 0:
            raise ValueError("Mesh sizes and permittivity must be positive")
        if self.order < 1 or not 0 < self.tolerance < 1:
            raise ValueError("Require order >= 1 and 0 < tolerance < 1")


SETTINGS = Settings()


class SheetSimulation(ElectrostaticSim):
    """Run an external sheet mesh with gsim's execution and result APIs."""

    # gsim currently initializes this field only through its own mesh() call.
    _last_mesh_result: object = PrivateAttr(default=None)


def _terminal_polygons(
    component: gf.Component, terminals: tuple[str, ...]
) -> list[np.ndarray]:
    """Return metal polygons in port order, rejecting floating or shorted terminals."""
    polygons = component.get_polygons_points(by="tuple")[
        gf.get_layer_tuple(LAYER.M1_DRAW)
    ]
    ordered = []
    indices = []
    for name in terminals:
        point = Point(component.ports[name].center)
        matches = [
            i
            for i, p in enumerate(polygons)
            if Polygon(p).distance(point) <= component.kcl.dbu
        ]
        if len(matches) != 1:
            raise ValueError(f"Terminal {name!r} must touch exactly one metal polygon")
        indices.append(matches[0])
        ordered.append(polygons[matches[0]])
    if len(set(indices)) != len(polygons) or len(indices) != len(polygons):
        raise ValueError("Every metal polygon must have exactly one terminal")
    return ordered


def _mesh(polygons: list[np.ndarray], settings: Settings, path: Path) -> int:
    """Mesh sheets and dielectric volumes, preserving a caller-owned Gmsh model.

    Returns:
        Number of tetrahedra.
    """
    # gmsh loads native OpenGL libraries; import it only when meshing.
    import gmsh  # ruff: ignore[import-outside-top-level]

    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    options = {
        "General.Terminal": 0,
        "Mesh.MeshSizeFromPoints": 0,
        "Mesh.MeshSizeFromCurvature": 0,
        "Mesh.MeshSizeExtendFromBoundary": 0,
        "Mesh.MshFileVersion": 2.2,
    }
    saved = {key: gmsh.option.getNumber(key) for key in options}
    gmsh.model.add(f"qpdk-{uuid4().hex}")
    try:
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        occ = gmsh.model.occ
        bounds = np.concatenate(polygons)
        left, bottom = bounds.min(axis=0)
        right, top = bounds.max(axis=0)
        pad = settings.domain_pad
        air = (
            3,
            occ.addBox(
                left - pad,
                bottom - pad,
                0,
                right - left + 2 * pad,
                top - bottom + 2 * pad,
                pad,
            ),
        )
        silicon = (
            3,
            occ.addBox(
                left - pad,
                bottom - pad,
                -pad,
                right - left + 2 * pad,
                top - bottom + 2 * pad,
                pad,
            ),
        )
        sheets = []
        for polygon in polygons:
            vertices = [occ.addPoint(float(x), float(y), 0) for x, y in polygon]
            lines = list(
                starmap(
                    occ.addLine, zip(vertices, vertices[1:] + vertices[:1], strict=True)
                )
            )
            sheets.append((2, occ.addPlaneSurface([occ.addWire(lines)])))
        outer = settings.ground_pad
        inner = settings.ground_clearance
        plane = (
            2,
            occ.addRectangle(
                left - outer,
                bottom - outer,
                0,
                right - left + 2 * outer,
                top - bottom + 2 * outer,
            ),
        )
        opening = (
            2,
            occ.addRectangle(
                left - inner,
                bottom - inner,
                0,
                right - left + 2 * inner,
                top - bottom + 2 * inner,
            ),
        )
        # The frame leaves the truncated CPW leads clear of ground.
        ground, _ = occ.cut([plane], [opening])
        _, mapping = occ.fragment([air, silicon], [*sheets, *ground])
        occ.synchronize()
        for index, entities in enumerate(mapping, start=1):
            dimension = 3 if index <= 2 else 2
            tags = sorted({tag for dim, tag in entities if dim == dimension})
            gmsh.model.addPhysicalGroup(dimension, tags, index)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            distance,
            "SurfacesList",
            [
                tag
                for entities in mapping[2 : 2 + len(sheets)]
                for dim, tag in entities
                if dim == 2
            ],
        )
        threshold = gmsh.model.mesh.field.add("Threshold")
        for key, value in {
            "InField": distance,
            "SizeMin": settings.near_mesh,
            "SizeMax": settings.far_mesh,
            "DistMin": 1,
            "DistMax": 25,
        }.items():
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
        return sum(len(elements) for elements in gmsh.model.mesh.getElements(3)[1])
    finally:
        gmsh.model.remove()
        for key, value in saved.items():
            gmsh.option.setNumber(key, value)
        if owned:
            gmsh.finalize()
        elif previous:
            gmsh.model.setCurrent(previous)


def generate(
    *,
    grid: Mapping[str, Sequence[float]] = GRID,
    output: Path = Path("build/datasets") / NAME,
    workdir: Path = Path("build/palace/plate-capacitor"),
    settings: Settings = SETTINGS,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace",
) -> Dataset:
    """Solve the grid, resume completed matching runs, and publish validated data."""
    PDK.activate()
    if executable is None and sif is None:
        executable = resolve_palace_binary()
        if executable is None:
            raise FileNotFoundError("No Palace runtime; pass --executable or --sif")
    fingerprints = {}
    for path in (executable, sif):
        if path is not None:
            with path.open("rb") as stream:
                fingerprints[path.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    source = Path(__file__).read_text(encoding="utf-8")
    # Extending GRID must not invalidate already solved geometries.
    extraction = [
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
    provenance = {
        "solver": "Palace",
        "generator_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(node) for node in extraction).encode()
        ).hexdigest(),
        "container_binary": container_binary if sif is not None else None,
        "gsim_version": importlib.metadata.version("gsim"),
        "gsim_revision": json
        .loads(installed or "{}")
        .get("vcs_info", {})
        .get("commit_id"),
        "gmsh_version": importlib.metadata.version("gmsh"),
        "gdsfactory_version": gf.__version__,
        "qpdk_version": __version__,
        "settings": asdict(settings),
        "files_sha256": fingerprints,
        "processes": processes,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "APPTAINERENV_OMP_NUM_THREADS",
            )
        },
        "metal": "zero-thickness perfect conductor sheets",
        "outer_boundary": "natural zero normal electric displacement",
    }
    if sif is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        executable = workdir.resolve() / "palace-container"
        executable.write_text(
            "#!/bin/sh\nset -eu\nranks=1\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n'
            '    -np) ranks="$2"; shift 2 ;;\n'
            "    -nt) shift 2 ;;\n"
            "    *) break ;;\n"
            "  esac\ndone\n"
            f"exec apptainer exec --cleanenv {shlex.quote(str(sif.resolve()))} "
            f'mpirun --oversubscribe -np "$ranks" {shlex.quote(container_binary)} "$@"\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)
    metadata = DatasetMetadata(
        name=NAME,
        description="Palace Maxwell capacitance of QPDK plate-capacitor sheets on silicon with a separate coplanar ground frame.",
        axes=tuple(Axis(name=name, unit="um") for name in GRID),
        variants=("cross_section",),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=TERMINALS,
        reference_ground="coplanar_ground_frame",
        provenance=provenance,
    )

    def solve(**point: float | str) -> dict:
        component = plate_capacitor(**point)
        inputs = {
            "point": point,
            "terminals": TERMINALS,
            "polygons_um": [
                p.tolist()
                for p in component.get_polygons_points(by="tuple")[
                    gf.get_layer_tuple(LAYER.M1_DRAW)
                ]
            ],
            "provenance": {
                name: value
                for name, value in provenance.items()
                if name != "generator_sha256"
            },
        }
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        run = workdir.resolve() / key
        complete = run / "complete.json"
        if complete.is_file():
            logger.info(f"Reusing Palace result: {run}")
            matrices = load_capacitance(run, terminal_names=TERMINALS)
        else:
            run.mkdir(parents=True, exist_ok=True)
            (run / "inputs.json").write_text(
                json.dumps(inputs, indent=2), encoding="utf-8"
            )
            _mesh(
                _terminal_polygons(component, TERMINALS), settings, run / "palace.msh"
            )
            sim = SheetSimulation()
            sim.set_output_dir(run)
            sim.set_solver(
                order=settings.order,
                tolerance=settings.tolerance,
                max_iterations=1000,
                preconditioner="BoomerAMG",
            )
            sim.set_electrostatic(
                save_fields=len(TERMINALS) if settings.save_fields else 0
            )
            config = {
                "Problem": {
                    "Type": "Electrostatic",
                    "Verbose": 2,
                    "Output": "output/palace",
                },
                "Model": {"Mesh": "palace.msh", "L0": 1e-6},
                "Domains": {
                    "Materials": [
                        {"Attributes": [1], "Permittivity": 1.0},
                        {"Attributes": [2], "Permittivity": settings.permittivity},
                    ]
                },
                "Boundaries": {
                    "Ground": {"Attributes": [len(TERMINALS) + 3]},
                    "Terminal": [
                        {"Index": index + 1, "Attributes": [index + 3]}
                        for index in range(len(TERMINALS))
                    ],
                },
                "Solver": sim.solver.to_palace_config(),
            }
            (run / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
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
            matrices = load_capacitance(run, terminal_names=TERMINALS)
            if (run / "solver.log").read_text().count("solver converged in") != len(
                TERMINALS
            ):
                raise RuntimeError(
                    f"Palace did not converge for every terminal; see {run / 'solver.log'}"
                )
        check_maxwell(matrices.maxwell)
        if problems := matrices.problems(rtol=1e-8):
            raise ValueError(f"Inconsistent Palace results in {run}: {problems}")
        complete.write_text(json.dumps({"terminals": TERMINALS}), encoding="utf-8")
        return {"maxwell_capacitance": matrices.maxwell}

    logger.info(
        f"Generating {NAME}: {np.prod([len(values) for values in grid.values()])} geometries"
    )
    result = write(
        output, metadata, sweep(metadata, solve, grid, {"cross_section": ["cpw"]})
    )
    result.grid("maxwell_capacitance", cross_section="cpw")
    return result


def main() -> None:
    """Preview or run this experiment; edit the constants for another sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("build/datasets") / NAME)
    parser.add_argument(
        "--workdir", type=Path, default=Path("build/palace/plate-capacitor")
    )
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--executable", type=Path)
    runtime.add_argument("--sif", type=Path)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--container-binary", default="palace")
    args = parser.parse_args()
    logger.info(
        f"Grid: {GRID}; settings: {SETTINGS}; output: {args.output}; runs: {args.workdir}"
    )
    if not args.dry_run:
        generate(
            output=args.output,
            workdir=args.workdir,
            executable=args.executable,
            sif=args.sif,
            processes=args.processes,
            container_binary=args.container_binary,
        )


if __name__ == "__main__":
    main()
