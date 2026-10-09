"""Opt-in comparison of interpolation with fresh Palace solves.

Run under Python 3.12 with gsim installed and QPDK_RUN_PALACE=1. PALACE_SIF
selects a container; PALACE_CONTAINER_BINARY defaults to palace. All outputs
are self-describing datasets and solver logs, with no validation JSON fixture.
"""

import ast
import os
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qpdk.models.datasets import GridInterpolator


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1",
    reason="Requires the experiment dependencies",
)
@pytest.mark.parametrize(
    ("width", "cpw_gap", "gap", "topology"),
    [
        (2.5, 1.25, 1.125, "fully-etched"),
        (2.449489743, 1.224744871, 2.461737191, "as-drawn"),
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
    helper = runpy.run_path(str(source / "_palace.py"))
    settings = generator["SETTINGS"]
    sheets, footprint = generator["geometry"](
        width=width,
        cpw_gap=cpw_gap,
        gap=gap,
        topology=generator["Topology"](topology),
        settings=settings,
    )
    mesh = helper["mesh_cross_section"](
        sheets,
        footprint,
        height=settings.domain_pad,
        near_mesh=settings.near_mesh,
        far_mesh=settings.far_mesh,
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
    finally:
        gmsh.finalize()


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1", reason="Requires a real Palace runtime"
)
def test_cpw_cross_section_normalization_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The implicit 2D depth changes total capacitance, never capacitance per length."""
    source = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(source))
    generator = runpy.run_path(str(source / "cpw_coupling.py"))
    runtime = {
        "workdir": tmp_path / "runs",
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
        "processes": 4,
    }
    matrices = []
    for depth in (1.0, 4.0):
        data = generator["generate"](
            grid={"width": [10.0], "cpw_gap": [6.0], "gap": [8.0]},
            settings=replace(generator["SETTINGS"], slice_length_um=depth),
            output=tmp_path / f"depth-{depth}",
            **runtime,
        )
        assert data.metadata.provenance["mesh_dimension"] == 2
        assert data.grid("mesh_relative_change").values.item() <= 0.01
        matrices.append(data.grid("maxwell_capacitance").values / depth)
    np.testing.assert_allclose(matrices[0], matrices[1], rtol=1e-7, atol=0)


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1",
    reason="Requires the experiment dependencies",
)
def test_container_launcher_replacement(
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
    image.write_bytes(b"launcher test")
    wrapper = tmp_path / "palace-container"
    wrapper.write_text("old launcher", encoding="utf-8")
    with wrapper.open(encoding="utf-8") as original:
        executable, _ = helper["runtime"](tmp_path, None, image, "palace")
        assert original.read() == "old launcher"
    assert executable == wrapper
    assert "apptainer exec --cleanenv" in wrapper.read_text(encoding="utf-8")
    assert os.access(wrapper, os.X_OK)


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1", reason="Requires a real Palace runtime"
)
def test_interpolation_against_fresh_palace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
    }
    source = generate(output=tmp_path / "grid", **runtime)
    lookup = GridInterpolator(source.grid("maxwell_capacitance", cross_section="cpw"))
    heldout = generate(
        grid={"length": [60.0, 100.0], "width": [10.0], "gap": [5.5, 8.5]},
        output=tmp_path / "heldout",
        **runtime,
    ).grid("maxwell_capacitance", cross_section="cpw")
    for i, length in enumerate(heldout.coords[0]):
        for j, gap in enumerate(heldout.coords[2]):
            np.testing.assert_allclose(
                lookup(length=length, width=10.0, gap=gap),
                heldout.values[i, 0, j],
                rtol=0.06,
                atol=1e-24,
            )
    logs = {
        path: path.stat().st_mtime_ns
        for path in runtime["workdir"].glob("*/solver.log")
    }
    generate(output=tmp_path / "grid", **runtime)
    assert logs == {
        path: path.stat().st_mtime_ns
        for path in runtime["workdir"].glob("*/solver.log")
    }


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1", reason="Requires a real Palace runtime"
)
def test_editing_grid_reuses_completed_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "sif": Path(value) if (value := os.environ.get("PALACE_SIF")) else None,
        "container_binary": os.environ.get("PALACE_CONTAINER_BINARY", "palace"),
    }
    before: dict[Path, int] = {}
    for lengths, expected in [([40.0, 80.0], 6), ([40.0, 80.0, 120.0], 9)]:
        grid = {"length": lengths, "width": [10.0], "gap": [4.0, 7.0, 10.0]}
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
        dataset = runpy.run_path(str(copied))["generate"](**runtime)
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
        if expected == 6:
            before = current
        else:
            assert all(current[path] == stamp for path, stamp in before.items())
