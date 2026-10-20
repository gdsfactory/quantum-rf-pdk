"""Connectivity checks for the resonator test-chip schematic view."""

import json
from pathlib import Path

SCHEMATIC_SAMPLE = (
    Path(__file__).parents[1] / "qpdk/samples/resonator_test_chip_yaml.gsch"
)


def _schematic_pin_position(device: dict, port_name: str) -> tuple[int, int]:
    """Return the Mosaic grid position of a test-chip symbol port."""
    x, y = device["x"], device["y"]
    if port_name == "resonator_o1":
        return x, y + 1
    if port_name == "coupling_o1":
        return x, y + 3
    if port_name == "coupling_o2":
        return x + 2, y + 2

    port = next(port for port in device["ports"] if port["name"] == port_name)
    is_left = port["orientation"] == 180
    if device.get("transform") == [-1, 0, 0, -1, 0, 0]:
        is_left = not is_left
    return x if is_left else x + 2, y + 1


def test_schematic_wires_connect_two_complete_probelines() -> None:
    """Each visible wire reaches its symbols, forming two complete rows."""
    schematic = json.loads(SCHEMATIC_SAMPLE.read_text())
    components = {
        name: device
        for name, device in schematic.items()
        if device["type"] not in {"wire", "port"}
    }
    neighbors = {name: set() for name in components}

    for wire in (device for device in schematic.values() if device["type"] == "wire"):
        endpoints = {
            (wire["x"], wire["y"]),
            (wire["x"] + wire["rx"], wire["y"] + wire["ry"]),
        }
        terminals = [
            (name, _schematic_pin_position(device, port_name))
            for name, device in components.items()
            for port_name, net in device["nets"].items()
            if net == wire["net"]
        ]
        assert len(terminals) == 2, wire["name"]
        assert {position for _, position in terminals} == endpoints, wire["name"]
        first_name, second_name = (name for name, _ in terminals)
        neighbors[first_name].add(second_name)
        neighbors[second_name].add(first_name)

    for line in ("top", "bot"):
        expected = {
            name
            for name in components
            if name.endswith(f"_{line}") or f"_{line}_" in name
        }
        visited = set()
        pending = [f"probe_west_{line}"]
        while pending:
            name = pending.pop()
            if name not in visited:
                visited.add(name)
                pending.extend(neighbors[name] - visited)
        assert len(expected) == 21
        assert visited == expected
