"""Facing pads retain two isolated electrodes and the specified ground frame."""

import importlib
import runpy
from itertools import product
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest

from qpdk import PDK
from qpdk.cells import coupling_pad_pair, plate_capacitor_single
from qpdk.tech import LAYER


@pytest.mark.parametrize("index", [0, 1, 2])
def test_coupling_pad_electrodes(index: int, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = runpy.run_path(str(scripts / "coupling_pad.py"))["RECIPE"]
    point = {name: values[index] for name, values in recipe.grid.items()}
    geometry = recipe.geometry(point, recipe.settings, 1)
    assert geometry.component.get_region(LAYER.M1_DRAW).merged().count() == 3
    assert tuple(geometry.terminal_ports) == ("o1", "o2")


@pytest.mark.parametrize(
    ("length", "width", "gap", "offset"),
    tuple(
        product(
            (40.0, 80.0, 120.0), (5.0, 10.0, 20.0), (4.0, 7.0, 10.0), (-12.0, 0.0, 12.0)
        )
    ),
)
def test_cell_preserves_simulated_geometry(
    length: float,
    width: float,
    gap: float,
    offset: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    ground_frame = importlib.import_module("_experiment").ground_frame

    point = dict(length=length, width=width, gap=gap, offset=offset)
    cell = coupling_pad_pair(**point)
    original = gf.Component()
    pad = plate_capacitor_single(length=length, width=width)
    first = original.add_ref(pad)
    second = original.add_ref(pad)
    second.drotate(180)
    second.dmove((width + gap, offset))
    bounds = ground_frame(original, clearance=10.0, width=45.0, domain_pad=200.0)
    assert (
        cell.get_region(LAYER.M1_DRAW) ^ original.get_region(LAYER.M1_DRAW)
    ).is_empty()
    assert (cell.get_region(LAYER.M1_DRAW) & cell.get_region(LAYER.M1_ETCH)).is_empty()
    for name, port in (("o1", first.ports["o1"]), ("o2", second.ports["o1"])):
        np.testing.assert_array_equal(cell.ports[name].center, port.center)
        assert cell.ports[name].orientation == port.orientation
        assert cell.ports[name].width == port.width
    recipe = runpy.run_path(str(scripts / "coupling_pad.py"))["RECIPE"]
    assert recipe.geometry(point, recipe.settings, 1).bounds == bounds


def test_paired_cell_uses_dataset_model() -> None:
    assert PDK.cells["coupling_pad_pair"] is coupling_pad_pair
    assert PDK.models["coupling_pad_pair"].__module__ == "qpdk.models.capacitor"
