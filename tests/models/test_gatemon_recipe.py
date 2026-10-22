"""Every gatemon recipe point separates island, gate, readout and ground."""

import itertools
import runpy
from pathlib import Path

import pytest

from qpdk.tech import LAYER

SCRIPTS = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "gatemon.py"))["RECIPE"]


@pytest.mark.parametrize("corner", list(itertools.product((0, -1), repeat=4)))
def test_gatemon_nets(recipe, corner: tuple[int, ...]) -> None:
    point = {
        name: values[i]
        for (name, values), i in zip(recipe.grid.items(), corner, strict=True)
    }
    geometry = recipe.geometry(point, recipe.settings, 1)
    assert geometry.component.get_region(LAYER.M1_DRAW).merged().count() == 4
    assert tuple(geometry.terminal_ports) == ("island", "gate", "readout")
    ports = geometry.component.ports
    assert ports["gate"].dy < ports["island"].dy < ports["readout"].dy


def test_gatemon_rejects_nonpositive(recipe) -> None:
    point = {name: values[0] for name, values in recipe.grid.items()}
    with pytest.raises(ValueError, match="positive"):
        recipe.geometry({**point, "gate_tip_gap": 0.0}, recipe.settings, 1)
