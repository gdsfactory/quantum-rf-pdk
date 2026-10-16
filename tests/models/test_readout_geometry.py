"""Coupon reference cuts leave distinct electrodes at every grid corner."""

import importlib
import sys
from itertools import product
from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk.tech import LAYER


@pytest.mark.parametrize("flip", [False, True])
def test_readout_corners(flip: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gsim.common", None)
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    helper = importlib.import_module("_readout")
    settings = importlib.import_module("_experiment").Settings()
    axes = {
        "pad_length": [48.0, 68.0] if flip else [300.0, 394.0],
        "pad_offset": [8.0, 16.0] if flip else [35.0, 55.0],
        "feedline_gap": [12.0, 20.0],
        "coupling_length": [500.0, 800.0],
    }
    for coordinates in product(*axes.values()):
        point = dict(zip(axes, coordinates, strict=True))
        geometry = helper.geometry(point, settings, 1, flip=flip)
        terminals = (*helper.TERMINALS, "top_pad") if flip else helper.TERMINALS
        assert tuple(geometry.terminal_ports) == terminals
        assert len(geometry.component.ports) == len(terminals)
        if flip:
            assert geometry.mesh_inputs


def test_invalid_geometry_and_terminal_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    helper = importlib.import_module("_readout")
    settings = importlib.import_module("_experiment").Settings()
    with pytest.raises(ValueError, match="dimensions must be positive"):
        helper.geometry(
            dict(
                pad_length=0.0,
                pad_offset=35.0,
                feedline_gap=12.0,
                coupling_length=500.0,
            ),
            settings,
            1,
        )
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_DRAW)
    component.add_port(
        "missing", center=(20, 20), width=1, orientation=0, layer=LAYER.M1_DRAW
    )
    with pytest.raises(ValueError, match="exactly one electrode"):
        helper.select_metal(
            component.get_region(LAYER.M1_DRAW),
            component.ports["missing"],
            component.kcl.dbu,
        )
