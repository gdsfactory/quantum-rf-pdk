"""Palace electrostatic extraction of planar conductors on a dielectric substrate.

Metal is modelled as perfect conductor sheets at the air/substrate interface.
A separate coplanar ground frame surrounds the component. The exterior uses
Palace's natural zero-normal-displacement boundary condition. Mesh and domain
convergence must be checked for the geometries being used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from functools import cached_property
from itertools import starmap
from pathlib import Path
from typing import Any
from uuid import uuid4

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from numpy.typing import NDArray
from shapely import Point, Polygon

from qpdk import logger
from qpdk.models.datasets.capacitance import check_maxwell, maxwell_to_mutual
from qpdk.tech import LAYER, material_properties


@dataclass(frozen=True, slots=True, kw_only=True)
class ElectrostaticSettings:
    """Electrostatic recipe, with lengths in µm and permittivity relative to vacuum."""

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
        """Reject settings that cannot describe the surrounding ground and domain."""
        if not 0 < self.ground_clearance < self.ground_pad < self.domain_pad:
            raise ValueError("Require 0 < ground_clearance < ground_pad < domain_pad")
        if not 0 < self.near_mesh <= self.far_mesh or self.permittivity <= 0:
            raise ValueError("Mesh sizes and permittivity must be positive")
        if self.order < 1 or not 0 < self.tolerance < 1:
            raise ValueError("Require order >= 1 and 0 < tolerance < 1")


def _terminal_polygons(
    component: Component, terminals: tuple[str, ...]
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


def _mesh(
    polygons: list[np.ndarray], settings: ElectrostaticSettings, path: Path
) -> int:
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


def _matrix(path: Path, count: int) -> NDArray[np.float64]:
    """Read a Palace matrix with explicit SI units and terminal index checks.

    Returns:
        Matrix in farads, in terminal index order.

    Raises:
        ValueError: If the units, shape or terminal indices do not match.
    """
    if "(F)" not in path.read_text(encoding="utf-8").splitlines()[0]:
        raise ValueError(f"Expected capacitance in farads in {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape != (count, count + 1) or not np.array_equal(
        values[:, 0], np.arange(1, count + 1)
    ):
        raise ValueError(f"Unexpected terminal order or matrix shape in {path}")
    return values[:, 1:]


@dataclass(frozen=True, kw_only=True)
class Palace:
    """Run and resume electrostatic extractions, keeping inputs and logs per geometry.

    ``command`` is an argument vector, e.g. ``('palace', '-np', '4')`` or an
    Apptainer/MPI prefix ending in the Palace binary. The config filename is
    appended; no shell is used. Use one writer per work directory.
    """

    workdir: Path
    command: tuple[str, ...] = ("palace", "-np", "4")
    settings: ElectrostaticSettings = field(default_factory=ElectrostaticSettings)

    @cached_property
    def provenance(self) -> dict[str, Any]:
        """Describe the solver and recipe used by the dataset and resume keys.

        Host paths are reduced to file names, since the provenance ships with
        the dataset. Some Palace builds report ``UNKNOWN`` as their version, so
        the SHA-256 of each file named in ``command`` is what identifies a build.

        Returns:
            Solver version, command, file fingerprints, mesh settings and recipe hash.
        """
        import gmsh  # ruff: ignore[import-outside-top-level]

        version = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [*self.command, "--version"], capture_output=True, text=True, check=True
        )
        files = {}
        for argument in self.command:
            path = Path(shutil.which(argument) or argument)
            if path.is_file():
                with path.open("rb") as stream:
                    files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "solver": "Palace",
            "version": (version.stdout + version.stderr).strip(),
            "command": [
                Path(arg).name if Path(arg).is_absolute() else arg
                for arg in self.command
            ],
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "APPTAINERENV_OMP_NUM_THREADS",
                )
            },
            "files_sha256": files,
            "settings": asdict(self.settings),
            "gmsh_version": gmsh.__version__,
            "gdsfactory_version": gf.__version__,
            "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metal": "zero-thickness perfect conductor sheets",
            "outer_boundary": "natural zero normal electric displacement",
        }

    def capacitance(
        self, component: Component, *, terminals: tuple[str, ...] = ("o1", "o2")
    ) -> NDArray[np.float64]:
        """Extract a Maxwell matrix in F, ordered by ``terminals``.

        Completed runs with identical geometry, recipe and solver are reused.
        A failed solve raises with the path to its log and never becomes data.

        Returns:
            Maxwell capacitance matrix.

        Raises:
            RuntimeError: If Palace fails or does not converge for every terminal.
        """
        polygons = _terminal_polygons(component, terminals)
        inputs = {
            "polygons_um": [p.tolist() for p in polygons],
            "terminals": terminals,
            "provenance": self.provenance,
        }
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        run = self.workdir.resolve() / key
        result_file = run / "result.json"
        if result_file.is_file():
            matrix = np.asarray(
                json.loads(result_file.read_text())["matrix_F"], dtype=float
            )
            check_maxwell(matrix)
            logger.info(f"Reusing Palace result: {run}")
            return matrix
        run.mkdir(parents=True, exist_ok=True)
        (run / "inputs.json").write_text(
            json.dumps(
                {
                    **inputs,
                    "component": component.name,
                    "settings": component.settings.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        tetrahedra = _mesh(polygons, self.settings, run / "mesh.msh")
        count = len(terminals)
        config = {
            "Problem": {"Type": "Electrostatic", "Verbose": 2, "Output": "postpro"},
            "Model": {"Mesh": "mesh.msh", "L0": 1e-6},
            "Domains": {
                "Materials": [
                    {"Attributes": [1], "Permittivity": 1.0},
                    {"Attributes": [2], "Permittivity": self.settings.permittivity},
                ]
            },
            "Boundaries": {
                "Ground": {"Attributes": [count + 3]},
                "Terminal": [
                    {"Index": i + 1, "Attributes": [i + 3]} for i in range(count)
                ],
            },
            "Solver": {
                "Order": self.settings.order,
                "Device": "CPU",
                "Electrostatic": {"Save": count if self.settings.save_fields else 0},
                "Linear": {
                    "Type": "BoomerAMG",
                    "KSPType": "CG",
                    "Tol": self.settings.tolerance,
                    "MaxIts": 1000,
                },
            },
        }
        (run / "config.json").write_text(json.dumps(config, indent=2))
        shutil.rmtree(run / "postpro", ignore_errors=True)
        log_path = run / "solver.log"
        with log_path.open("w") as log:
            status = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                [*self.command, "config.json"],
                cwd=run,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if (
            status.returncode
            or log_path.read_text().count("PCG solver converged in") != count
        ):
            raise RuntimeError(f"Palace extraction failed; see {log_path}")
        matrix = _matrix(run / "postpro/terminal-C.csv", count)
        check_maxwell(matrix)
        np.testing.assert_allclose(
            maxwell_to_mutual(matrix),
            _matrix(run / "postpro/terminal-Cm.csv", count),
            rtol=1e-9,
            atol=1e-24,
        )
        staging = run / "result.tmp"
        staging.write_text(
            json.dumps(
                {"matrix_F": matrix.tolist(), "tetrahedra": tetrahedra}, indent=2
            )
        )
        staging.replace(result_file)
        logger.info(
            f"Solved {component.name}: {tetrahedra} tetrahedra; results in {run}"
        )
        return matrix
