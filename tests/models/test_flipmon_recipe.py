"""A flip-chip coupon keeps opposing metal and chip grounds separate."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("index", [0, -1])
def test_flipmon_stack_electrodes(index: int, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gsim")
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = runpy.run_path(str(scripts / "flipmon.py"))["RECIPE"]
    point = {name: values[index] for name, values in recipe.grid.items()}
    geometry = recipe.geometry(point, recipe.settings, 1)
    assert geometry.mesh is not None
    assert tuple(geometry.terminal_ports) == ("inner_pad", "outer_ring", "top_pad")
    for layer in geometry.component.layers:
        assert geometry.component.get_region(layer).merged().count() == 1
    assert len(geometry.component.layers) == 5
