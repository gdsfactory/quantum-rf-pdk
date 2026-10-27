"""Driven winding ports span the return gap and retain the film assumption."""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qpdk.tech import LAYER, material_properties


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("meander_inductor")


@pytest.mark.parametrize("turns", [3, 4, 5, 7])
def test_wire_ends_and_ground_frame(recipe, turns: int) -> None:
    component = recipe.geometry(
        dict(n_turns=str(turns), turn_length=100.0, wire_width=1.0, wire_gap=2.0)
    )
    regions = list(component.get_region(LAYER.M1_DRAW).merged().each())
    winding, frame = sorted(regions, key=lambda polygon: polygon.bbox().width())
    assert frame.holes() == 1
    dbu = component.kcl.dbu
    assert (winding.bbox().left - frame.bbox().left) * dbu == pytest.approx(120.0)
    assert tuple(port.name for port in component.ports) == ("o1", "o2")
    assert component.ports["o1"].orientation == 180
    assert component.ports["o2"].orientation == (0 if turns % 2 else 180)
    assert {port.width for port in component.ports} == {1.0}


def test_film_config_and_independent_excitations(recipe, tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = {
        "Boundaries": {
            "PEC": {"Attributes": [2, 3]},
            "Absorbing": {"Attributes": [9]},
            "LumpedPort": [
                {"Index": 1, "Excitation": 1},
                {"Index": 2, "Excitation": 2},
            ],
        }
    }
    path.write_text(json.dumps(config))
    sim = Mock(write_config=Mock(return_value=path))
    recipe.write_config(sim, recipe.SETTINGS)
    boundaries = json.loads(path.read_text())["Boundaries"]
    assert "PEC" not in boundaries
    assert boundaries["Impedance"] == [{"Attributes": [2, 3], "Ls": 0.4e-12}]
    path.write_text(json.dumps(config))
    recipe.write_config(sim, replace(recipe.SETTINGS, sheet_inductance=0.0))
    assert json.loads(path.read_text())["Boundaries"]["PEC"] == {"Attributes": [2, 3]}
    config["Boundaries"]["LumpedPort"][1]["Excitation"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="independently"):
        recipe.write_config(sim, recipe.SETTINGS)


@pytest.mark.parametrize(
    "settings",
    [
        {"near_mesh": 0.0},
        {"near_mesh": 2.0, "far_mesh": 1.0},
        {"domain_pad": 0.0},
        {"sheet_inductance": -1e-12},
        {"order": 0},
        {"tolerance": 1.0},
        {"fmin": 0.0},
        {"fmin": 2e9, "fmax": 1e9},
        {"points": 1},
    ],
)
def test_invalid_settings(recipe, settings: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"Require|Invalid"):
        replace(recipe.SETTINGS, **settings)


@pytest.mark.parametrize(
    "point", [{"n_turns": "0"}, {"turn_length": 0.0}, {"wire_gap": -1.0}]
)
def test_invalid_winding(recipe, point: dict[str, float | str]) -> None:
    with pytest.raises(ValueError, match="positive"):
        recipe.geometry(
            dict(n_turns="3", turn_length=100.0, wire_width=1.0, wire_gap=2.0) | point
        )


@pytest.mark.palace
def test_lossless_materials_and_mesh_budget(
    recipe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    palace = pytest.importorskip("gsim.palace")
    mesh_options = {}
    tetrahedra = 100_000

    def mesh(sim, **options) -> None:
        mesh_options.update(options)
        sim._last_mesh_result = SimpleNamespace(mesh_stats={"tetrahedra": tetrahedra})

    monkeypatch.setattr(palace.DrivenSim, "mesh", mesh)
    point = dict(n_turns="3", turn_length=100.0, wire_width=0.6, wire_gap=2.0)
    sim = recipe.simulation(point, recipe.SETTINGS, tmp_path, 8)
    silicon = sim.stack.materials["coupon_silicon"]
    assert silicon["permittivity"] == material_properties["Si"]["relative_permittivity"]
    assert silicon["loss_tangent"] == pytest.approx(0.0)
    assert silicon["conductivity"] == pytest.approx(0.0)
    assert sim.solver.order == 2
    assert sim.solver.linear.initial_guess is False
    assert sim.solver.driven.adaptive_tol == pytest.approx(0.0)
    assert mesh_options["refined_mesh_size"] == pytest.approx(0.3)
    assert mesh_options["planar_conductors"] is True
    for order, maximum in ((2, 1_500_000), (3, 800_000)):
        tetrahedra = maximum + 1
        with pytest.raises(ValueError, match="mesh budget"):
            recipe.simulation(point, replace(recipe.SETTINGS, order=order), tmp_path, 8)


@pytest.mark.parametrize(
    "boundary", [{"Conductivity": [{"Attributes": [2]}]}, {"Absorbing": None}]
)
def test_invalid_film_boundaries(recipe, tmp_path: Path, boundary: dict) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"Boundaries": {"Absorbing": {"Attributes": [9]}} | boundary})
    )
    with pytest.raises(ValueError, match="planar metal"):
        recipe.write_config(Mock(write_config=Mock(return_value=path)), recipe.SETTINGS)
