"""Keep sample schematic pins and physical routes consistent."""

import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import gdsfactory as gf
import pytest
import yaml
from kfactory.schematic import DSchematic

SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
# Sample stem -> number of distinct nets wired in the Mosaic view.
SAMPLES = {
    "gatemon_test_chip": 22,
    "tsv_flip_chip_test_chip": 22,
    "vortex_neck_test_chip": 11,
}


def _slots(count: int, size: int) -> list[int]:
    """Mirror ``spread-ports`` in Mosaic: place ``count`` ports in ``size`` slots."""
    if size <= count:
        return list(range(1, count + 1))
    half = count // 2
    middle = [(size + 1) // 2] if count % 2 else []
    return [*range(1, half + 1), *middle, *range(size - half + 1, size + 1)]


def _pin(device: dict, name: str) -> tuple[float, float]:
    """Grid position of a pin; a symbol cell spans two grid units."""
    if device["type"] == "port":
        local, mid = (0, 0), (0, 0)
    else:
        factory = gf.kcl.factories[device["model"].rsplit(".", 1)[-1]]
        schematic = factory.get_schematic()
        assert isinstance(schematic, DSchematic)
        ports = schematic.info["ports"]
        assert isinstance(ports, list)
        sides = [port["side"] for port in ports if isinstance(port, dict)]
        n = {side: sides.count(side) for side in ("left", "right", "top", "bottom")}
        # Box size in cells, as ``port-perimeter`` sizes it.
        height = max(1, n["left"], n["right"])
        height += n["left"] % 2 != n["right"] % 2 and height % 2 == 0
        width = max(1, n["top"], n["bottom"])
        width += n["top"] % 2 != n["bottom"] % 2 and width % 2 == 0
        port = next(
            port for port in ports if isinstance(port, dict) and port["name"] == name
        )
        side = port["side"]
        assert isinstance(side, str)
        same_side = [
            p["name"] for p in ports if isinstance(p, dict) and p["side"] == side
        ]
        size = height if side in {"left", "right"} else width
        slot = 2 * _slots(len(same_side), size)[same_side.index(name)] - 1
        local = {
            "left": (0, slot),
            "right": (2 * width, slot),
            "top": (slot, 0),
            "bottom": (slot, 2 * height),
        }[side]
        mid = (width, height)
    a, b, c, d, e, f = device["transform"]
    x, y = local[0] - mid[0], local[1] - mid[1]
    return (
        device["x"] + a * x + c * y + e + mid[0],
        device["y"] + b * x + d * y + f + mid[1],
    )


@pytest.mark.parametrize("stem", SAMPLES)
def test_mosaic_wires_reach_all_declared_pins(stem: str) -> None:
    document = json.loads((SAMPLE_DIR / f"{stem}.gsch").read_text())
    pins = defaultdict(set)
    graph = defaultdict(lambda: defaultdict(set))
    for entry in document.values():
        assert isinstance(entry["name"], str)
        assert all(math.isfinite(entry[key]) for key in ("x", "y"))
        if entry["type"] in {"ckt", "port"}:
            for name, net in entry.get("nets", {}).items():
                pins[net].add(_pin(entry, name))
        elif entry["type"] == "wire":
            start = (entry["x"], entry["y"])
            finish = (entry["x"] + entry["rx"], entry["y"] + entry["ry"])
            assert start[0] == finish[0] or start[1] == finish[1]
            graph[entry["net"]][start].add(finish)
            graph[entry["net"]][finish].add(start)
    assert len(pins) == SAMPLES[stem]
    for net, endpoints in pins.items():
        assert len(endpoints) == 2, net
        visited = set()
        pending = [next(iter(endpoints))]
        while pending:
            point = pending.pop()
            if point not in visited:
                visited.add(point)
                pending.extend(graph[net][point] - visited)
        assert endpoints <= visited, net


@pytest.mark.parametrize("stem", SAMPLES)
def test_livewire_routes_and_ports_match_layout(stem: str) -> None:
    document = json.loads((SAMPLE_DIR / f"{stem}.gsch").read_text())
    netlist = yaml.safe_load((SAMPLE_DIR / f"{stem}.pic.yml").read_text())
    layout = gf.get_component(stem)
    for name, spec in netlist["instances"].items():
        cell = gf.get_component(spec["component"], **spec["settings"])
        bbox = cell.dbbox()
        center = bbox.center()
        actual = document[name]
        assert actual["bbox_width"] == pytest.approx(bbox.width())
        assert actual["bbox_height"] == pytest.approx(bbox.height())
        assert actual["bbox_xmin"] == pytest.approx(bbox.left)
        assert actual["bbox_ymin"] == pytest.approx(bbox.bottom)
        for metadata in actual["ports"]:
            port = cell.ports[metadata["name"]]
            assert metadata["irl_rx"] == pytest.approx(port.x - center.x)
            assert metadata["irl_ry"] == pytest.approx(center.y - port.y)
    for name in netlist["routes"]:
        route = document[name]
        points = route["vertices"]
        for side, endpoint in (("start", points[0]), ("finish", points[-1])):
            terminal = route["terminals"][side]
            port = layout.insts[terminal["component_id"]].ports[terminal["port_name"]]
            assert endpoint == pytest.approx((port.x, -port.y))
        for start, finish in pairwise(points):
            assert abs(start[0] - finish[0]) < 0.01 or abs(start[1] - finish[1]) < 0.01
