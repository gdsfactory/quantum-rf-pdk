"""Tests for the qpdk.simulation FEM helpers."""

import sys
from types import ModuleType, SimpleNamespace

import gdsfactory as gf
import klayout.db as kdb
import pytest

from qpdk.cells import coupler_straight
from qpdk.simulation import FEM_LAYERS, single_chip_stack, to_fem_regions
from qpdk.tech import LAYER


def _sim_layout() -> gf.Component:
    """Return a small layout with a simulation area and feed ports."""

    @gf.cell
    def _layout() -> gf.Component:
        c = gf.Component()
        ref = c << coupler_straight(gap=16.0, length=100.0)
        c.add_ports(ref.ports)
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    return _layout()


def test_to_fem_regions_ports_and_layers():
    c = to_fem_regions(_sim_layout())

    assert sorted(p.name for p in c.ports) == ["o1", "o2", "o3", "o4"]
    layout = c.kdb_cell.layout()
    for name, layer in FEM_LAYERS.items():
        region = c.kdb_cell.begin_shapes_rec(layout.layer(*layer))
        assert not region.at_end(), f"{name} empty"


def test_to_fem_regions_conductor_excludes_etch():
    component = _sim_layout()
    c = to_fem_regions(component)

    layout_in = component.kdb_cell.layout()
    etch = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout_in.layer(*LAYER.M1_ETCH))
    ).merged()
    layout_out = c.kdb_cell.layout()
    conductor = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout_out.layer(*FEM_LAYERS["SUPERCONDUCTOR"]))
    ).merged()
    sim_area = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout_out.layer(*FEM_LAYERS["SUBSTRATE"]))
    ).merged()
    assert (conductor & etch).is_empty()
    assert not conductor.is_empty()
    assert conductor.area() < sim_area.area()


def test_to_fem_regions_draw_bridges_etched_gap():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(LAYER.M1_ETCH).insert(kdb.DBox(48, 0, 52, 100))
    component.kdb_cell.shapes(LAYER.M1_DRAW).insert(kdb.DBox(45, 48, 55, 52))

    converted = to_fem_regions(component)
    layout = converted.kdb_cell.layout()
    conductor = kdb.Region(
        converted.kdb_cell.begin_shapes_rec(layout.layer(*FEM_LAYERS["SUPERCONDUCTOR"]))
    ).merged()

    assert len(list(conductor.each())) == 1
    assert (conductor & kdb.Region(kdb.Box(49000, 49000, 51000, 51000))).area() > 0
    assert (conductor & kdb.Region(kdb.Box(49000, 10000, 51000, 20000))).is_empty()
    source_layout = component.kdb_cell.layout()
    assert not component.kdb_cell.begin_shapes_rec(
        source_layout.layer(*LAYER.M1_DRAW)
    ).at_end()


def test_single_chip_stack_matches_material_properties():
    pytest.importorskip("gsim")

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["SUPERCONDUCTOR"].thickness == 0
    assert stack.materials["silicon"]["permittivity"] == pytest.approx(11.45)
    assert stack.materials["silicon"]["loss_tangent"] == pytest.approx(2.7e-6)


def test_single_chip_stack_without_optional_gsim(monkeypatch):
    """Check the exported layer stack when gsim is not installed."""
    gsim = ModuleType("gsim")
    common = ModuleType("gsim.common")
    stack_module = ModuleType("gsim.common.stack")
    materials = ModuleType("gsim.common.stack.materials")

    class Layer(SimpleNamespace):
        pass

    class LayerStack:
        def __init__(self, pdk_name):
            self.pdk_name = pdk_name
            self.layers = {}
            self.materials = {}

    stack_module.__dict__.update(Layer=Layer, LayerStack=LayerStack)
    materials.__dict__["MATERIALS_DB"] = {
        "vacuum": SimpleNamespace(to_dict=lambda: {"permittivity": 1.0})
    }
    for name, module in (
        ("gsim", gsim),
        ("gsim.common", common),
        ("gsim.common.stack", stack_module),
        ("gsim.common.stack.materials", materials),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.pdk_name == "qpdk"
    assert stack.layers["SUBSTRATE"].gds_layer == FEM_LAYERS["SUBSTRATE"]
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["SUPERCONDUCTOR"].zmin == pytest.approx(200.0)
    assert stack.layers["SUPERCONDUCTOR"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.materials["silicon"] == {
        "permittivity": pytest.approx(11.45),
        "loss_tangent": pytest.approx(2.7e-6),
    }
    assert stack.materials["vacuum"] == {"permittivity": 1.0}


def test_to_fem_regions_requires_sim_area():
    """A component without SIM_AREA must fail loudly, not model empty space."""
    with pytest.raises(ValueError, match="SIM_AREA"):
        to_fem_regions(coupler_straight(gap=16.0, length=100.0))


def test_to_fem_regions_rejects_fully_etched_area():
    component = gf.Component()
    for layer in (LAYER.SIM_AREA, LAYER.M1_ETCH):
        component.kdb_cell.shapes(layer).insert(kdb.DBox(0, 0, 100, 100))

    with pytest.raises(ValueError, match="fully etched away"):
        to_fem_regions(component)


def test_fem_layers_do_not_collide_with_mask_layers():
    """FEM regions must not land on real mask layers (M1_DRAW is (1,0) ...)."""
    mask_layers = {
        tuple(getattr(LAYER, name))
        for name in dir(LAYER)
        if not name.startswith("_") and isinstance(getattr(LAYER, name), tuple)
    }
    for name, layer in FEM_LAYERS.items():
        assert layer not in mask_layers, f"{name} collides with a mask layer"
