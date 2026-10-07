"""Palace process boundary, resume keys and terminal assignment."""

import dataclasses
import subprocess
from pathlib import Path

import numpy as np
import pytest

try:
    import gmsh
except OSError:  # gmsh loads native OpenGL libraries such as libGLU
    pytest.skip("gmsh's native libraries are missing", allow_module_level=True)

from qpdk import PDK
from qpdk.cells import plate_capacitor
from qpdk.simulation import palace
from qpdk.simulation.palace import ElectrostaticSettings, Palace


@pytest.fixture
def solver(monkeypatch: pytest.MonkeyPatch) -> dict:
    state = {"returncode": 0, "converged": True, "calls": 0}

    def run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout="Palace fixture", stderr=""
            )
        state["calls"] += 1
        output = kwargs["cwd"] / "postpro"
        output.mkdir()
        (output / "terminal-C.csv").write_text(
            "i, C1 (F), C2 (F)\n1,2e-15,-5e-16\n2,-5e-16,3e-15\n", encoding="utf-8"
        )
        (output / "terminal-Cm.csv").write_text(
            "i, C1 (F), C2 (F)\n1,1.5e-15,5e-16\n2,5e-16,2.5e-15\n", encoding="utf-8"
        )
        if state["converged"]:
            kwargs["stdout"].write("PCG solver converged in 12 iterations\n" * 2)
        return subprocess.CompletedProcess(command, state["returncode"])

    monkeypatch.setattr(palace.subprocess, "run", run)
    monkeypatch.setattr(palace, "_mesh", lambda *_args: 100)
    PDK.activate()
    return state


def test_resume_and_changed_recipe(tmp_path: Path, solver: dict) -> None:
    runner = Palace(workdir=tmp_path, command=("palace-fixture",))
    component = plate_capacitor(length=40, width=10, gap=7)
    matrix = runner.capacitance(component)
    np.testing.assert_array_equal(runner.capacitance(component), matrix)
    assert solver["calls"] == 1
    refined = dataclasses.replace(
        runner, settings=ElectrostaticSettings(near_mesh=0.38)
    )
    refined.capacitance(component)
    assert solver["calls"] == 2
    assert len(list(tmp_path.glob("*/result.json"))) == 2


@pytest.mark.parametrize("failure", ["exit", "convergence"])
def test_failed_solve_is_not_cached(tmp_path: Path, solver: dict, failure: str) -> None:
    runner = Palace(workdir=tmp_path, command=("palace-fixture",))
    component = plate_capacitor(length=40, width=10, gap=7)
    solver["returncode"] = 1 if failure == "exit" else 0
    solver["converged"] = failure == "exit"
    with pytest.raises(RuntimeError, match=r"solver\.log"):
        runner.capacitance(component)
    assert not list(tmp_path.glob("*/result.json"))
    solver.update(returncode=0, converged=True)
    runner.capacitance(component)
    assert solver["calls"] == 2


def test_terminal_order_and_short_rejection() -> None:
    PDK.activate()
    component = plate_capacitor(length=40, width=10, gap=7)
    ordered = palace._terminal_polygons(component, ("o1", "o2"))
    reversed_order = palace._terminal_polygons(component, ("o2", "o1"))
    np.testing.assert_array_equal(ordered[0], reversed_order[1])
    with pytest.raises(ValueError, match="exactly one terminal"):
        palace._terminal_polygons(component, ("o1", "o1"))


@pytest.mark.parametrize("kind", ["units", "order"])
def test_matrix_reader_refuses_wrong_conventions(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "terminal-C.csv"
    output.write_text(
        "i, C1 (pF), C2 (pF)\n1,2,-0.5\n2,-0.5,3\n"
        if kind == "units"
        else "i, C1 (F), C2 (F)\n2,2,-0.5\n1,-0.5,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"farads|terminal order"):
        palace._matrix(output, 2)


def test_thread_environment_changes_resume_key(
    tmp_path: Path, solver: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = plate_capacitor(length=40, width=10, gap=7)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    Palace(workdir=tmp_path, command=("palace-fixture",)).capacitance(component)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    Palace(workdir=tmp_path, command=("palace-fixture",)).capacitance(component)
    assert solver["calls"] == 2


def test_mesh_preserves_caller_model(tmp_path: Path) -> None:
    PDK.activate()
    component = plate_capacitor(length=40, width=10, gap=7)
    gmsh.initialize()
    try:
        gmsh.model.add("caller")
        version = gmsh.option.getNumber("Mesh.MshFileVersion")
        mesh = tmp_path / "mesh.msh"
        count = palace._mesh(
            palace._terminal_polygons(component, ("o1", "o2")),
            ElectrostaticSettings(near_mesh=3),
            mesh,
        )
        assert count > 0
        assert gmsh.model.getCurrent() == "caller"
        assert gmsh.option.getNumber("Mesh.MshFileVersion") == version
        assert mesh.read_text(encoding="utf-8").splitlines()[1] == "2.2 0 8"
    finally:
        gmsh.finalize()
