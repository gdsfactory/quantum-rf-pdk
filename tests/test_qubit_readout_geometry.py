"""The simulated boundary must retain the same metal as the layout."""

import importlib
from pathlib import Path
from typing import Any

import gdsfactory as gf
import pytest

from qpdk import cells
from qpdk.tech import LAYER


@pytest.mark.parametrize("name", ["xmon_readout"])
def test_layout_coupon_parity(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(__file__).resolve().parents[1] / "qpdk/models/datasets/recipes"
    monkeypatch.syspath_prepend(str(source))
    recipe = importlib.import_module(name).RECIPE
    select_metal = importlib.import_module("_readout").select_metal
    point = recipe.provenance.get("nominal_layout") or {
        key: values[len(values) // 2] for key, values in recipe.grid.items()
    }
    factory = getattr(cells, name)
    layout = factory(**point)
    coupon = recipe.geometry(point, recipe.settings, 4)
    metal = coupon.component.get_region(LAYER.M1_DRAW)
    electrodes = gf.kdb.Region()
    for port in coupon.component.ports:
        electrodes |= select_metal(metal, port, coupon.component.kcl.dbu)
    assert (electrodes ^ layout.get_region(LAYER.M1_DRAW)).is_empty()
    assert [p.name for p in layout.ports] == ["readout"]
    assert layout.ports["readout"].port_type == "optical"
    assert layout.ports["readout"].center == coupon.component.ports["readout"].center
    schematic: Any = gf.kcl.factories[factory.__name__].get_schematic().info
    assert [p["name"] for p in schematic["ports"]] == ["readout"]
    assert schematic["models"][0]["port_order"] == ["readout"]
    assert schematic["models"][0]["name"] == name


@pytest.mark.parametrize("name", ["xmon_readout"])
def test_connect_to_cpw(name: str) -> None:
    c = gf.Component()
    qubit = c << getattr(cells, name)()
    line = c << cells.straight(length=100)
    line.connect("o2", qubit.ports["readout"])
    c.add_port("o1", port=line.ports["o1"])
    nets = c.get_netlist()["nets"]
    assert len(nets) == 1
