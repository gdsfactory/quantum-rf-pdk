"""Parity checks for Mosaic and pic.yml sample sources."""

import json
from collections import defaultdict
from pathlib import Path

import gdsfactory as gf
import pytest
import yaml

from qpdk import PDK

SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
SAMPLE_STEMS = tuple(path.stem for path in sorted(SAMPLE_DIR.glob("*.gsch")))


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_gsch_matches_pic_yml(stem: str) -> None:
    """Keep each Mosaic schematic in sync with its declarative netlist."""
    yaml_document = yaml.safe_load((SAMPLE_DIR / f"{stem}.pic.yml").read_text())
    gsch_document = json.loads((SAMPLE_DIR / f"{stem}.gsch").read_text())
    gsch_instances = {
        name: instance
        for name, instance in gsch_document.items()
        if isinstance(instance, dict) and instance.get("model")
    }

    assert set(gsch_instances) == set(yaml_document["instances"])
    for name, yaml_instance in yaml_document["instances"].items():
        gsch_instance = gsch_instances[name]
        assert gsch_instance["model"].rsplit(".", 1)[-1] == yaml_instance["component"]
        assert (gsch_instance.get("props") or {}) == yaml_instance["settings"]

    gsch_nets = defaultdict(set)
    for name, instance in gsch_instances.items():
        for port, net in instance["nets"].items():
            gsch_nets[net].add(f"{name},{port}")
    connected_gsch_nets = {
        frozenset(endpoints) for endpoints in gsch_nets.values() if len(endpoints) > 1
    }

    yaml_nets = {
        frozenset((left, right))
        for left, right in yaml_document.get("connections", {}).items()
    }
    for route in yaml_document.get("routes", {}).values():
        yaml_nets.update(
            frozenset((left, right)) for left, right in route["links"].items()
        )
    assert connected_gsch_nets == yaml_nets

    gsch_ports = {
        name: item
        for name, item in gsch_document.items()
        if isinstance(item, dict) and item.get("type") == "port"
    }
    assert set(gsch_ports) == set(yaml_document["ports"])
    for name, attached_port in yaml_document["ports"].items():
        instance, port = attached_port.split(",")
        assert gsch_ports[name]["attached_port"] == {
            "component_id": instance,
            "port_name": port,
        }


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_sample_resolves_from_pdk(stem: str) -> None:
    """Resolve every checked-in schematic sample by its short name."""
    PDK.activate()

    assert gf.get_component(stem).name == stem
