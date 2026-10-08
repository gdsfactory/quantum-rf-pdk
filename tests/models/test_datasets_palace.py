"""Opt-in comparison of interpolation with fresh Palace solves.

Run under Python 3.12 with gsim installed and QPDK_RUN_PALACE=1. PALACE_SIF
selects a container; PALACE_CONTAINER_BINARY defaults to palace. All outputs
are self-describing datasets and solver logs, with no validation JSON fixture.
"""

import ast
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from qpdk.models.datasets import GridInterpolator


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1",
    reason="Requires the experiment dependencies",
)
def test_container_launcher_replacement(tmp_path: Path) -> None:
    generator = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "datasets/plate_capacitor.py")
    )
    image = tmp_path / "palace.sif"
    image.write_bytes(b"launcher test")
    wrapper = tmp_path / "palace-container"
    wrapper.write_text("old launcher", encoding="utf-8")
    with wrapper.open(encoding="utf-8") as original:
        executable, _ = generator["runtime"](tmp_path, None, image, "palace")
        assert original.read() == "old launcher"
    assert executable == wrapper
    assert "apptainer exec --cleanenv" in wrapper.read_text(encoding="utf-8")
    assert os.access(wrapper, os.X_OK)


@pytest.mark.skipif(
    os.environ.get("QPDK_RUN_PALACE") != "1", reason="Requires a real Palace runtime"
)
def test_interpolation_against_fresh_palace(tmp_path: Path) -> None:
    generator = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "datasets/plate_capacitor.py")
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
def test_editing_grid_reuses_completed_solves(tmp_path: Path) -> None:
    original = Path(__file__).resolve().parents[2] / "datasets/plate_capacitor.py"
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
        assert dataset.table.height == expected * 4
        current = {
            path: path.stat().st_mtime_ns
            for path in runtime["workdir"].glob("*/solver.log")
        }
        assert len(current) == expected
        if expected == 6:
            before = current
        else:
            assert all(current[path] == stamp for path, stamp in before.items())
