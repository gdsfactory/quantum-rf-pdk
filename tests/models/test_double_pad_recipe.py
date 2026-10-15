"""Both floating pads survive the boxed-device and ground construction."""

import runpy
from pathlib import Path

import pytest

from qpdk.tech import LAYER


@pytest.mark.parametrize("index", [0, 1, 2])
def test_double_pad_islands(index: int, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = runpy.run_path(str(scripts / "double_pad_transmon.py"))["RECIPE"]
    point = {name: values[index] for name, values in recipe.grid.items()}
    geometry = recipe.geometry(point, recipe.settings, 1)
    assert geometry.component.get_region(LAYER.M1_DRAW).merged().count() == 3
    assert tuple(geometry.terminal_ports) == ("left_pad", "right_pad")
    assert (
        geometry.component.ports["left_pad"].dx
        < geometry.component.ports["right_pad"].dx
    )
