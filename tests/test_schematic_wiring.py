"""Visible Mosaic wiring of the resonator test chip."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qpdk.read import read_gsch

GSCH_PATH = Path(__file__).parents[1] / "qpdk/samples/resonator_test_chip_yaml.gsch"


def _mosaic_pin_position(item: dict[str, Any], port: str) -> tuple[float, float]:
    """Use the viewer's pin pattern after GFP reverses same-side left ports."""
    if item["type"] == "port":
        return item["x"], item["y"]
    if item["model"].endswith("quarter_wave_resonator_coupled"):
        # GFP's generated library lists resonator_o1 before coupling_o1.
        px, py = {
            "resonator_o1": (0, 1),
            "coupling_o1": (0, 3),
            "coupling_o2": (2, 2),
        }[port]
        midpoint = 2
    else:
        px, py = (0, 1) if port in {"o1", "waveport"} else (2, 1)
        if item["model"].endswith("launcher"):
            px = 0 if port == "waveport" else 2
        midpoint = 1
    a, b, c, d, e, f = item.get("transform", [1, 0, 0, 1, 0, 0])
    return (
        item["x"] + a * (px - midpoint) + c * (py - midpoint) + e + midpoint,
        item["y"] + b * (px - midpoint) + d * (py - midpoint) + f + midpoint,
    )


def test_resonator_chip_shared_nets_have_visible_wires() -> None:
    """Draw every connected net instead of relying on invisible net IDs."""
    document = json.loads(GSCH_PATH.read_text())
    nets = defaultdict(list)
    for name, item in document.items():
        if item["type"] == "wire":
            continue
        for port, net in item.get("nets", {}).items():
            nets[net].append((name, port))

    wires = [item for item in document.values() if item["type"] == "wire"]
    assert len(wires) == 44
    assert {wire["name"] for wire in wires} == {
        net for net, terminals in nets.items() if len(terminals) > 1
    }
    for wire in wires:
        assert all(float(wire[key]).is_integer() for key in ("x", "y", "rx", "ry"))
        assert wire["rx"] != 0 or wire["ry"] != 0
        assert {
            (wire["x"], wire["y"]),
            (wire["x"] + wire["rx"], wire["y"] + wire["ry"]),
        } == {
            _mosaic_pin_position(document[name], port)
            for name, port in nets[wire["name"]]
        }

    for name in ("o1", "o2", "o3", "o4"):
        port = document[name]
        attached = port["attached_port"]
        assert (
            port["nets"]["P"]
            == document[attached["component_id"]]["nets"][attached["port_name"]]
        )


def test_resonator_chip_open_ends_remain_unconnected() -> None:
    """Keep virtual open ends separate from the connected feedline."""
    document = json.loads(GSCH_PATH.read_text())
    netlist = read_gsch(GSCH_PATH)
    assert len(netlist["instances"]) == 42
    assert len(netlist["ports"]) == 4
    assert len(netlist.get("routes", {})) == 0
    resonators = [
        item for name, item in document.items() if name.startswith("resonator_")
    ]
    assert len(resonators) == 16
    for resonator in resonators:
        open_net = resonator["nets"]["resonator_o1"]
        open_pin = _mosaic_pin_position(resonator, "resonator_o1")
        for wire in document.values():
            if wire["type"] != "wire":
                continue
            x1, y1 = wire["x"], wire["y"]
            x2, y2 = x1 + wire["rx"], y1 + wire["ry"]
            # Every fixture wire uses the viewer's horizontal-then-vertical bend.
            assert not (min(x1, x2) <= open_pin[0] <= max(x1, x2) and open_pin[1] == y1)
            assert not (open_pin[0] == x2 and min(y1, y2) <= open_pin[1] <= max(y1, y2))
        assert (
            sum(open_net in item.get("nets", {}).values() for item in document.values())
            == 1
        )
