"""Shared circuit builders for the complete gatemon chip checks."""

from typing import Literal

import gdsfactory as gf
import sax
import yaml

from qpdk import PDK
from qpdk.config import PATH
from qpdk.models.gatemon import gatemon_island
from qpdk.models.generic import short


def _netlist(source: Literal["schematic", "layout"]) -> dict:
    """Read logical YAML connections or extract every physical CPW route."""
    PDK.activate()
    if source == "layout":
        # Preserve capacitances in F; the default rounding would erase them.
        netlist = gf.get_component("gatemon_test_chip").get_netlist(
            serialization_max_digits=18, on_dangling_port="ignore"
        )
        return {
            "instances": netlist["instances"],
            "connections": {net["p1"]: net["p2"] for net in netlist["nets"]},
            "ports": netlist["ports"],
        }
    document = yaml.safe_load((PATH.samples / "gatemon_test_chip.pic.yml").read_text())
    connections = dict(document["connections"])
    for route in document["routes"].values():
        connections.update(route["links"])
    return {
        "instances": document["instances"],
        "connections": connections,
        "ports": document["ports"],
    }


def _circuit(
    source: Literal["schematic", "layout"],
    gate_load: Literal["matched", "short"] = "matched",
) -> tuple[sax.SDictModel, dict]:
    """Bind the layout's island adapter and optional external RF gate loads."""
    netlist = _netlist(source)
    if gate_load == "short":
        for port in ("gate_a", "gate_b"):
            name = f"load_{port}"
            netlist["instances"][name] = {"component": "short"}
            netlist["connections"][netlist["ports"].pop(port)] = f"{name},o1"
    # The legacy gatemon model has a ground port; the layout uses readout/gate.
    models = dict(PDK.models) | {"gatemon": gatemon_island, "short": short}
    circuit, _ = sax.circuit(
        netlist, models=models, ignore_impossible_connections=False
    )
    return circuit, netlist
