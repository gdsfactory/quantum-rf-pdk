"""Fresh Palace solves, interpolation and cache reuse.

Run with ``uv run --group palace --python 3.12 pytest -m palace``.
PALACE_SIF selects a container; otherwise gsim resolves a native runtime.

These are smoke tests of the pipeline, not of the physics: every solve uses a
coarse mesh, a loose solver tolerance and a small grid so the suite stays fast
in CI. The shipped datasets record their own production settings.
"""

import ast
import hashlib
import os
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np
import polars as pl
import pytest

from qpdk.models.datasets import GridInterpolator
from qpdk.tech import LAYER

pytestmark = pytest.mark.palace


def coarse_plate_settings(generator: dict[str, Any]) -> Any:
    """Coarse first-order 3D mesh; enough to exercise the pipeline only."""
    return replace(
        generator["SETTINGS"], near_mesh=2.0, far_mesh=50.0, order=1, tolerance=1e-6
    )


def coarse_cpw_settings(generator: dict[str, Any]) -> Any:
    """Small 2D domain with a coarse mesh; enough to exercise the pipeline only."""
    return replace(
        generator["SETTINGS"],
        domain_pad=300.0,
        near_mesh=0.5,
        far_mesh=30.0,
        tolerance=1e-6,
    )


@pytest.fixture
def palace_processes() -> int:
    """Fit the solver ranks within the CPUs provided by a test runner."""
    return min(2, os.cpu_count() or 1)


@pytest.mark.parametrize(
    ("width", "cpw_gap", "gap", "topology"),
    [
        (2.5, 1.25, 1.125, "fully-etched"),
        (2.450, 1.225, 2.462, "as-drawn"),
    ],
)
def test_fractional_cross_section_conductors_belong_to_domain_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: float,
    cpw_gap: float,
    gap: float,
    topology: str,
) -> None:
    """Detached boundary edges make the Palace mesh reader crash."""
    import gmsh  # ruff: ignore[import-outside-top-level]

    source = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(source))
    generator = runpy.run_path(str(source / "cpw_coupling.py"))
    settings = generator["SETTINGS"]
    component, domain_bounds = generator["geometry"](
        width=width,
        cpw_gap=cpw_gap,
        gap=gap,
        topology=generator["Topology"](topology),
        settings=settings,
    )
    from gsim.palace.mesh.meshwell import (  # ruff: ignore[import-outside-top-level]
        mesh_sheets,
    )

    mesh = mesh_sheets(
        component,
        conductor_layer=gf.get_layer_tuple(LAYER.M1_DRAW),
        terminal_ports={name: name for name in generator["TERMINALS"]},
        domain_bounds=domain_bounds,
        height=settings.domain_pad,
        near_mesh=settings.near_mesh,
        far_mesh=settings.far_mesh,
        cross_section=True,
        path=tmp_path / "section.msh",
    )
    gmsh.initialize()
    try:
        gmsh.open(str(mesh.mesh_path))
        triangles = gmsh.model.mesh.getElementsByType(2)[1].reshape(-1, 3)
        edges = {
            tuple(sorted((int(a), int(b))))
            for triangle in triangles
            for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True)
        }
        segments = gmsh.model.mesh.getElementsByType(1)[1].reshape(-1, 2)
        assert len(segments) > 0
        assert all(tuple(sorted(map(int, segment))) in edges for segment in segments)
        if topology == "as-drawn" and gap > 2 * cpw_gap:
            strip = gap - 2 * cpw_gap
            nodes, coordinates, _ = gmsh.model.mesh.getNodes()
            positions = dict(zip(nodes, coordinates.reshape(-1, 3), strict=True))
            inner = [
                segment
                for segment in segments
                if all(abs(positions[node][0]) <= strip / 2 + 1e-8 for node in segment)
            ]
            assert len(inner) >= 4
    finally:
        gmsh.finalize()


def test_cpw_cross_section_normalization_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The implicit 2D depth changes total capacitance, never capacitance per length.

    One process keeps both depths on the same mesh: multithreaded Gmsh meshing
    is not reproducible. Identical meshes give identical solves.
    """
    source = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(source))
    generator = runpy.run_path(str(source / "cpw_coupling.py"))
    runtime = {
        "workdir": tmp_path / "runs",
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
        "processes": 1,
    }
    settings = coarse_cpw_settings(generator)
    matrices = []
    for depth in (1.0, 4.0):
        data = generator["generate"](
            grid={"width": [10.0], "cpw_gap": [6.0], "gap": [8.0]},
            settings=replace(settings, slice_length_um=depth),
            output=tmp_path / f"depth-{depth}",
            mesh_tolerance=0.2,
            min_refinements=1,
            max_refinements=2,
            **runtime,
        )
        assert data.metadata.provenance["mesh_dimension"] == 2
        assert data.metadata.provenance["mesh_convergence"]["minimum_refinements"] == 1
        assert data.grid("mesh_refinement_level").values.item() >= 1
        assert data.grid("mesh_relative_change").values.item() <= 0.2
        matrices.append(data.grid("maxwell_capacitance").values / depth)
    np.testing.assert_allclose(matrices[0], matrices[1], rtol=1e-7, atol=0)


def test_changing_ground_strip_resolution_invalidates_cached_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, palace_processes: int
) -> None:
    """Changing the feature resolution must solve a new mesh at the same point."""
    source = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(source))
    generator = runpy.run_path(str(source / "cpw_coupling.py"))
    helper = runpy.run_path(str(source / "_palace.py"))
    settings = replace(coarse_cpw_settings(generator), domain_pad=100, far_mesh=10)
    point = {"width": 2.45, "cpw_gap": 1.225, "gap": 2.462}
    component, bounds = generator["geometry"](
        **point, settings=settings, topology=generator["Topology"].AS_DRAWN
    )
    execution, _ = helper["runtime"](
        None,
        Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
    )
    for resolution in (4, 8):
        helper["extract"](
            component,
            bounds,
            {name: name for name in generator["TERMINALS"]},
            height=settings.domain_pad,
            inputs={"point": point},
            workdir=tmp_path / "runs",
            execution=execution,
            processes=palace_processes,
            near_mesh=settings.near_mesh,
            far_mesh=settings.far_mesh,
            permittivity=settings.permittivity,
            order=settings.order,
            tolerance=settings.tolerance,
            normalization_depth_um=settings.slice_length_um,
            minimum_feature_elements=resolution,
        )
    meshes = list((tmp_path / "runs").glob("*/palace.msh"))
    assert len(meshes) == 2
    assert len({hashlib.sha256(path.read_bytes()).digest() for path in meshes}) == 2


def test_container_runtime_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data")
    )
    helper = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data/_palace.py"
        )
    )
    image = tmp_path / "palace.sif"
    image.write_bytes(b"runtime test")
    options, provenance = helper["runtime"](None, image, "palace-custom")
    assert options == {
        "palace_executable": None,
        "palace_sif_path": image,
        "use_apptainer": True,
        "container_binary": "palace-custom",
    }
    assert len(provenance["runtime_sha256"]) == 64
    assert not (tmp_path / "palace-container").exists()


def test_interpolation_against_fresh_palace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, palace_processes: int
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data")
    )
    generator = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "qpdk/models/datasets/data/plate_capacitor.py"
        )
    )
    generate = generator["generate"]
    runtime = {
        "workdir": tmp_path / "runs",
        "processes": palace_processes,
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
    }
    settings = coarse_plate_settings(generator)
    grid = {"length": [40.0, 80.0, 120.0], "width": [10.0], "gap": [4.0, 7.0, 10.0]}
    source = generate(grid=grid, output=tmp_path / "grid", settings=settings, **runtime)
    lookup = GridInterpolator(source.grid("maxwell_capacitance", cross_section="cpw"))
    heldout = generate(
        grid={"length": [60.0], "width": [10.0], "gap": [5.5]},
        output=tmp_path / "heldout",
        settings=settings,
        **runtime,
    ).grid("maxwell_capacitance", cross_section="cpw")
    # Coarse meshes differ between geometries, so this checks plumbing, not accuracy.
    np.testing.assert_allclose(
        lookup(length=60.0, width=10.0, gap=5.5),
        heldout.values[0, 0, 0],
        rtol=0.15,
        atol=1e-24,
    )
    logs = {
        path: path.stat().st_mtime_ns
        for path in runtime["workdir"].glob("*/solver.log")
    }
    generate(grid=grid, output=tmp_path / "grid", settings=settings, **runtime)
    assert logs == {
        path: path.stat().st_mtime_ns
        for path in runtime["workdir"].glob("*/solver.log")
    }


def test_editing_grid_reuses_completed_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, palace_processes: int
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data")
    )
    original = (
        Path(__file__).resolve().parents[2]
        / "qpdk/models/datasets/data/plate_capacitor.py"
    )
    source = original.read_text(encoding="utf-8")
    copied = tmp_path / "plate_capacitor.py"
    runtime = {
        "workdir": tmp_path / "runs",
        "output": tmp_path / "data",
        "processes": palace_processes,
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
    }
    before: dict[Path, int] = {}
    settings = coarse_plate_settings(runpy.run_path(str(original)))
    for lengths, expected in [([40.0, 80.0], 4), ([40.0, 80.0, 120.0], 6)]:
        grid = {"length": lengths, "width": [10.0], "gap": [4.0, 10.0]}
        tree = ast.parse(source)
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GRID"
                for target in node.targets
            )
        )
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1 : node.end_lineno] = [f"GRID = {grid!r}\n"]
        copied.write_text("".join(lines), encoding="utf-8")
        dataset = runpy.run_path(str(copied))["generate"](settings=settings, **runtime)
        rows = (
            dataset
            .scan()
            .filter(pl.col("quantity") == "maxwell_capacitance")
            .select(pl.len())
            .collect()
            .item()
        )
        assert rows == expected * 4
        current = {
            path: path.stat().st_mtime_ns
            for path in runtime["workdir"].glob("*/solver.log")
        }
        assert len(current) == expected
        if expected == 4:
            before = current
        else:
            assert all(current[path] == stamp for path, stamp in before.items())
