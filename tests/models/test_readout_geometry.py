"""Coupon reference cuts leave distinct electrodes at every grid corner."""

import importlib
from itertools import product
from pathlib import Path

import pytest


@pytest.mark.parametrize("flip", [False, True])
def test_readout_corners(flip: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if flip:
        pytest.importorskip("gsim")
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
