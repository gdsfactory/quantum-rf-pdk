"""Keep the gatemon's schematic pins and physical routes consistent."""

import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import gdsfactory as gf
import pytest
import yaml
from kfactory.schematic import DSchematic

SAMPLE = Path(__file__).parents[1] / "qpdk/samples/gatemon_test_chip"


def _pin(device: dict, name: str) -> tuple[float, float]:
    if device["type"] == "port":
        local = (0, 0)
    else:
        factory = gf.kcl.factories[device["model"].rsplit(".", 1)[-1]]
        schematic = factory.get_schematic()
        assert isinstance(schematic, DSchematic)
        ports = schematic.info["ports"]
        assert isinstance(ports, list)
        port = next(
            port for port in ports if isinstance(port, dict) and port["name"] == name
        )
        side = port["side"]
        assert isinstance(side, str)
        local = {"left": (0, 1), "right": (2, 1), "top": (1, 0), "bottom": (1, 2)}[side]
    a, b, c, d, e, f = device["transform"]
    mid = 0 if device["type"] == "port" else 1
    x, y = local[0] - mid, local[1] - mid
    return (
        device["x"] + a * x + c * y + e + mid,
        device["y"] + b * x + d * y + f + mid,
    )


def test_mosaic_wires_reach_all_declared_pins() -> None:
    document = json.loads(SAMPLE.with_suffix(".gsch").read_text())
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
    assert len(pins) == 22
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


def test_livewire_routes_and_ports_match_layout() -> None:
    document = json.loads(SAMPLE.with_suffix(".gsch").read_text())
    netlist = yaml.safe_load(SAMPLE.with_suffix(".pic.yml").read_text())
    layout = gf.get_component("gatemon_test_chip")
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
