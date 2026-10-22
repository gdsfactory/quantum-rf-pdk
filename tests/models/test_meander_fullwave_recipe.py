"""Driven winding ports span the return gap and retain the film assumption."""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from qpdk.tech import LAYER


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
