"""Every gatemon recipe point separates island, gate, readout and ground."""

import itertools
import runpy
from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk.cells import gatemon
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


def test_nominal_recipe_matches_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lookup's default point must represent the default layout."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = runpy.run_path(str(SCRIPTS / "gatemon.py"))
    point = dict(
        island_size=150.0, pocket_gap=60.0, pocket_gap_top=40.0, gate_tip_gap=5.0
    )
    device = gatemon(**module["device_settings"](point))
    expected = gatemon()
    for layer in (
        LAYER.M1_DRAW,
        LAYER.M1_ETCH,
        LAYER.CNT_CONTACT,
        LAYER.HBN,
        LAYER.CNT,
        LAYER.TOP_GATE,
        LAYER.JJ_PATCH,
    ):
        assert (device.get_region(layer) ^ expected.get_region(layer)).is_empty(), layer


def test_recipe_matches_chip_readout_reference_plane(recipe) -> None:
    """The default coupon uses the chip's 600 um line and 200 um coupling point."""
    point = dict(
        island_size=150.0, pocket_gap=60.0, pocket_gap_top=40.0, gate_tip_gap=5.0
    )
    geometry = recipe.geometry(point, recipe.settings, 1)
    c = geometry.component
    port = c.ports["readout"]
    x, y = (round(value / c.kcl.dbu) for value in port.center)
    metal = c.get_region(LAYER.M1_DRAW).merged()
    readout = metal.interacting(gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2)))
    bounds = readout.bbox().to_dtype(c.kcl.dbu)
    assert bounds.left == pytest.approx(-200.0)
    assert bounds.right == pytest.approx(400.0)
    assert bounds.bottom == pytest.approx(115.0)
    assert bounds.top == pytest.approx(125.0)
