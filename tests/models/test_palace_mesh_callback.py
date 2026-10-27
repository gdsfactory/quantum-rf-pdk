"""Custom native meshes preserve extraction failure and cache behavior."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("mesh_fails", [False, True])
@pytest.mark.parametrize("terminal_order", [("a", "b"), ("b", "a"), ("a",)])
def test_custom_mesh_controls_solver_and_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mesh_fails: bool,
    terminal_order: tuple[str, ...],
) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"
    monkeypatch.syspath_prepend(str(scripts))
    helper = importlib.import_module("_palace")
    sim = Mock()
    sim.terminals = [SimpleNamespace(name=name) for name in terminal_order]
    constructor = Mock(return_value=sim)
    matrix = jnp.array([[3.0, -1.0], [-1.0, 3.0]]) * 1e-15
    gsim = ModuleType("gsim")
    palace = ModuleType("gsim.palace")
    capacitance = ModuleType("gsim.palace.capacitance")
    monkeypatch.setattr(palace, "ElectrostaticSim", constructor, raising=False)
    monkeypatch.setattr(
        capacitance,
        "load_capacitance",
        lambda *_args, **_kwargs: SimpleNamespace(
            maxwell=matrix, problems=lambda **_kwargs: []
        ),
        raising=False,
    )
    for name, module in (
        ("gsim", gsim),
        ("gsim.palace", palace),
        ("gsim.palace.capacitance", capacitance),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(helper, "diagnostics", lambda *_args, **_kwargs: {})

    def prepare_mesh(selected: object) -> None:
        assert selected is sim
        sim.set_geometry.assert_called_once()
        sim.set_output_dir.assert_called_once()
        sim.run_local.assert_not_called()
        if mesh_fails:
            raise RuntimeError("mesh rejected")

    mesh = Mock(side_effect=prepare_mesh)
    kwargs = dict(
        component=object(),
        domain_bounds=(0.0, 0.0, 10.0, 10.0),
        terminal_ports={"a": "o1", "b": "o2"},
        height=10.0,
        inputs={"custom_mesh": True},
        workdir=tmp_path,
        execution={},
        processes=1,
        near_mesh=1.0,
        far_mesh=10.0,
        permittivity=11.45,
        order=2,
        tolerance=1e-9,
        mesh=mesh,
    )
    if mesh_fails:
        with pytest.raises(RuntimeError, match="mesh rejected"):
            helper.extract(**kwargs)
        assert not list(tmp_path.rglob("complete.json"))
        sim.run_local.assert_not_called()
    elif terminal_order != ("a", "b"):
        with pytest.raises(ValueError, match="Mesh callback terminals"):
            helper.extract(**kwargs)
        assert not list(tmp_path.rglob("complete.json"))
        sim.run_local.assert_not_called()
    else:
        for _ in range(2):
            result = helper.extract(**kwargs)
            np.testing.assert_allclose(result["maxwell_capacitance"], matrix)
        assert len(list(tmp_path.rglob("complete.json"))) == 1
        constructor.assert_called_once()
        sim.run_local.assert_called_once()
        sim.reset_mock()
        kwargs["terminal_ports"] = {"b": "o2", "a": "o1"}
        sim.terminals.reverse()
        helper.extract(**kwargs)
        assert constructor.call_count == 2
        assert len(list(tmp_path.rglob("complete.json"))) == 2
    assert mesh.call_count == (
        2 if not mesh_fails and terminal_order == ("a", "b") else 1
    )
    sim.mesh_sheets.assert_not_called()
