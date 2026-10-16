"""Connected winding ends cannot become independent electrostatic terminals."""

import runpy
from pathlib import Path

import pytest

from qpdk.tech import LAYER


@pytest.mark.parametrize("turns", ["3", "5", "7"])
def test_one_winding_terminal(turns: str, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = runpy.run_path(str(scripts / "meander_inductor.py"))["RECIPE"]
    point = {name: values[0] for name, values in recipe.grid.items()} | {
        "n_turns": turns
    }
    geometry = recipe.geometry(point, recipe.settings, 1)
    assert geometry.component.get_region(LAYER.M1_DRAW).merged().count() == 2
    assert tuple(geometry.terminal_ports) == ("winding",)
    assert recipe.variants == {"n_turns": ["3", "5", "7"]}
