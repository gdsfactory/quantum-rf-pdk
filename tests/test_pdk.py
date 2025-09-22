"""Description: Test netlists for all cells in the PDK."""

from __future__ import annotations

import pathlib
from typing import cast

import gdsfactory as gf
import jsondiff
import pytest
from gdsfactory.difftest import difftest
from gdsfactory.technology import LayerViews
from gdsfactory.typings import ComponentFactory
from kfactory import LayerEnum
from pytest_regressions.data_regression import DataRegressionFixture

import qpdk.samples
from qpdk import PDK
from qpdk.config import PATH
from qpdk.helper import denest_layerviews_to_layer_tuples
from qpdk.tech import LAYER

cells = PDK.cells
skip_test_netlist = {
    "wire_corner",
    "pack_doe",
    "pack_doe_grid",
    "add_pads_top",
    "add_pads_bot",
    "transmon_circular",
    "resonator_lumped",
    "coupler_symmetric",
    "die_with_pads",
    "launcher",
    "indium_bump",
    "flipmon_with_resonator",  # Skip due to complex routing causing netlist extraction issues
}
# Skip default gdsfactory cells
skip_test = {
    "pack_doe",
    "pack_doe_grid",
    "add_pads_top",
    "add_pads_bot",
    "die_with_pads",
    "transform_component",
    "transmon_with_resonator",
}
cell_names = cells.keys() - skip_test
cell_names = [name for name in cell_names if not name.startswith("_")]
cell_names.sort()  # Fix running in parallel with pytest-xdist
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds").parent / "gds_ref"
dirpath.mkdir(exist_ok=True, parents=True)


def get_minimal_netlist(comp: gf.Component):
    """Get minimal netlist from a component."""
    net = comp.get_netlist()

    def _get_instance(inst):
        return {
            "component": inst["component"],
            "settings": inst["settings"],
        }

    return {"instances": {i: _get_instance(c) for i, c in net["instances"].items()}}


def instances_without_info(net):
    """Get instances without info."""
    return {
        k: {
            "component": v.get("component", ""),
            "settings": v.get("settings", {}),
        }
        for k, v in net.get("instances", {}).items()
    }


@pytest.mark.parametrize("name", cell_names)
def test_cell_in_pdk(name):
    """Test that cell is in the PDK."""
    c1 = gf.Component()
    component = gf.get_component(name)
    if isinstance(component, gf.Component):
        c1.add_ref(gf.get_component(name))
    elif isinstance(component, gf.ComponentAllAngle):
        c1.add_ref_off_grid(component)
    net1 = get_minimal_netlist(c1)

    c2 = gf.read.from_yaml(net1)
    net2 = get_minimal_netlist(c2)

    instances1 = instances_without_info(net1)
    instances2 = instances_without_info(net2)
    assert instances1 == instances2


@pytest.mark.parametrize("component_name", cell_names)
def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = cells[component_name]()
    difftest(component, test_name=component_name, dirpath=dirpath)


@pytest.mark.parametrize("component_name", cell_names)
def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())


@pytest.mark.parametrize("component_type", cell_names)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
) -> None:
    """Write netlists for hierarchical circuits.

    Checks that both netlists are the same jsondiff does a hierarchical diff.

    Component -> netlist -> Component -> netlist

    """
    if component_type in skip_test_netlist:
        pytest.skip(f"Skipping {component_type} netlist test")
    c = cells[component_type]()
    n = c.get_netlist()
    data_regression.check(n)

    n.pop("connections", None)
    n.pop("warnings", None)
    yaml_str = c.write_netlist(n)

    cis = list(c.kcl.each_cell_top_down())
    for ci in cis:
        gf.kcl.dkcells[ci].delete()

    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("ports", None)
    assert len(d) == 0, d

    cis = list(c.kcl.each_cell_top_down())
    for ci in cis:
        gf.kcl.dkcells[ci].delete()


def test_yaml_matches_layers():
    """Test that the YAML LayerView matches defined layers."""
    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    LAYERS_ACCORDING_TO_YAML = denest_layerviews_to_layer_tuples(LAYER_VIEWS)
    LAYERS_DEFINED = {
        str(layer_enum): (layer_enum.layer, layer_enum.datatype)
        for layer_enum in cast(LayerEnum, LAYER)
    }
    assert LAYERS_ACCORDING_TO_YAML == LAYERS_DEFINED


@pytest.mark.parametrize(
    "sample",
    list(qpdk.sample_functions.values()),
    ids=list(qpdk.sample_functions.keys()),
)
def test_sample_generates(sample: ComponentFactory):
    """Test that all sample cells generate without errors."""
    result = gf.get_component(sample)
    assert result
    print(f"Successfully ran {sample!r}")


if __name__ == "__main__":
    component_type = "coupler_symmetric"
    c = cells[component_type]()
    n = c.get_netlist()
    n.pop("connections", None)
    print(n)
    c2 = gf.read.from_yaml(n)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    assert len(d) == 0, d
