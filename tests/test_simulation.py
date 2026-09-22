"""Tests for the qpdk.simulation FEM helpers."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import gdsfactory as gf
import klayout.db as kdb
import pytest

from qpdk.cells import coupler_straight, flipmon_with_bbox
from qpdk.simulation import (
    FEM_LAYERS,
    FLIP_CHIP_FEM_LAYERS,
    single_chip_stack,
    to_fem_regions,
    to_flip_chip_regions,
)
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


BUMP_RADIUS = 7.5  # µm, matches the cell's own indium bump


def _bump(x: float, y: float) -> kdb.DPolygon:
    """Return a circular bump polygon of the standard radius at (x, y)."""
    return kdb.DPolygon.ellipse(
        kdb.DBox(x - BUMP_RADIUS, y - BUMP_RADIUS, x + BUMP_RADIUS, y + BUMP_RADIUS), 64
    )


def _flip_chip_layout() -> gf.Component:
    """Return a flipmon layout with corner ground bumps and a simulation area."""

    @gf.cell
    def _flipmon_layout() -> gf.Component:
        c = gf.Component()
        ref = c << flipmon_with_bbox()
        c.add_ports(ref.ports)
        # Corner bumps tie the two chips' ground planes into a single node.
        for x, y in ((-150, -150), (150, -150), (-150, 150), (150, 150)):
            c.kdb_cell.shapes(LAYER.IND).insert(_bump(x, y))
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    return _flipmon_layout()


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


def _require_gsim() -> None:
    """Skip unless gsim imports; gmsh can raise OSError, not just ImportError."""
    try:
        importlib.import_module("gsim")
    except (ImportError, OSError):
        pytest.skip("gsim unavailable (missing package or GL system libraries)")


def test_single_chip_stack_matches_material_properties():
    _require_gsim()

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["SUPERCONDUCTOR"].thickness == 0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)
    assert stack.materials["qpdk-silicon"]["loss_tangent"] == pytest.approx(2.7e-6)


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
    assert stack.materials["qpdk-silicon"] == {
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
    for name, layer in {**FEM_LAYERS, **FLIP_CHIP_FEM_LAYERS}.items():
        assert layer not in mask_layers, f"{name} collides with a mask layer"


def test_to_flip_chip_regions_topology():
    c = to_flip_chip_regions(_flip_chip_layout())

    assert sorted(p.name for p in c.ports) == [
        "center",
        "inner_ring_near_junction",
        "junction",
        "outer_ring_near_junction",
        "outer_ring_outside",
    ]
    layout = c.kdb_cell.layout()
    regions = {
        name: kdb.Region(
            c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS[name]))
        ).merged()
        for name in FLIP_CHIP_FEM_LAYERS
    }
    # Both chips follow the subtractive convention inside their etched
    # bounding circles, so each metal splits into isolated islands plus
    # the surrounding ground plane.
    assert len(regions["M1"]) == 3  # ground plane, outer ring, inner circle
    assert len(regions["M2"]) == 2  # ground plane, top circle

    def islands(region: kdb.Region) -> kdb.Region:
        """Return the isolated islands (everything but the ground frame)."""
        ground = max(region.each(), key=lambda poly: poly.area())
        return region - kdb.Region(ground)

    # The center bump must bridge the two isolated islands (the inner
    # circle on M1 and the top circle on M2), not merely land anywhere on
    # the near-ubiquitous ground planes.
    center_bump = kdb.Region(kdb.DBox(-8, -8, 8, 8).to_itype(c.kcl.dbu))
    assert not (center_bump & islands(regions["M1"])).is_empty()
    assert not (center_bump & islands(regions["M2"])).is_empty()
    # Every bump must land on conductor of both chips to actually connect.
    assert (regions["BUMP"] - regions["M1"]).is_empty()
    assert (regions["BUMP"] - regions["M2"]).is_empty()


def test_to_flip_chip_regions_bump_off_metal_detected():
    """A bump in the etched lead gap is visible in the converted regions.

    The topology test asserts every bump lands on conductor of both chips;
    this is the negative case proving that assertion has teeth: a bump
    placed in the etched junction gap (no bottom-chip metal under it — the
    M2 top circle above does not help) shows up outside the M1 conductor.
    """

    @gf.cell
    def _bad_bump_layout() -> gf.Component:
        c = gf.Component()
        ref = c << flipmon_with_bbox()
        c.add_ports(ref.ports)
        # In the 12 um etched lead gap between the inner circle and the ring.
        c.kdb_cell.shapes(LAYER.IND).insert(_bump(66.0, 0.0))
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    c = to_flip_chip_regions(_bad_bump_layout())
    layout = c.kdb_cell.layout()
    bumps = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS["BUMP"]))
    ).merged()
    m1 = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS["M1"]))
    ).merged()
    assert not (bumps - m1).is_empty(), "off-metal bump not detected"


def test_flip_chip_stack_matches_conventions():
    _require_gsim()

    from qpdk.simulation import (  # ruff: ignore[import-outside-top-level]
        flip_chip_stack,
    )

    stack = flip_chip_stack(substrate_thickness=200.0, bump_thickness=10.0)
    assert stack.layers["SUBSTRATE"].zmin == pytest.approx(-200.0)
    assert stack.layers["M1"].zmax == pytest.approx(0.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(10.0)
    assert stack.layers["M2"].zmin == pytest.approx(10.0)
    assert stack.layers["SUBSTRATE_TOP"].zmax == pytest.approx(210.0)
    assert stack.layers["BUMP"].layer_type == "via"
    # The bump must be a real 3-D volume spanning the gap: a zero-height via
    # would degrade to a 2-D PEC sheet and connect nothing.
    assert stack.layers["BUMP"].zmin == pytest.approx(0.0)
    assert stack.layers["BUMP"].zmax == pytest.approx(10.0)
    assert stack.layers["BUMP"].thickness > 0
    # Every layer's material must resolve in the stack's own materials dict;
    # gsim has no builtin fallback and silently emits Permittivity 1.0 for
    # unknown materials (this caught the top substrate modeled as vacuum).
    stack.validate_stack()
    # Without a conductivity gsim demotes each bump to a 2-D PEC sheet at
    # its base, so the two chips would no longer connect.
    assert stack.materials["qpdk-indium"]["conductivity"] > 0.0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)
