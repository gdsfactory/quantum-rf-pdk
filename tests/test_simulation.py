"""Tests for the qpdk.simulation FEM helpers."""

import gdsfactory as gf
import klayout.db as kdb
import pytest

from qpdk.cells import coupler_straight
from qpdk.simulation import FEM_LAYERS, to_fem_regions
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

    # The conductor is the simulation area minus the etch mask, so no point
    # of the input etch mask may be conductor. The etch region must come from
    # the INPUT component: the output never carries M1_ETCH, so reading it
    # from the output would make the assertion vacuous.
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


def test_single_chip_stack_matches_material_properties():
    pytest.importorskip("gsim")

    from qpdk.simulation import (  # ruff: ignore[import-outside-top-level]
        single_chip_stack,
    )

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["SUPERCONDUCTOR"].thickness == 0
    assert stack.materials["silicon"]["permittivity"] == pytest.approx(11.45)
    assert stack.materials["silicon"]["loss_tangent"] == pytest.approx(2.7e-6)


def test_to_fem_regions_requires_sim_area():
    """A component without SIM_AREA must fail loudly, not model empty space."""
    with pytest.raises(ValueError, match="SIM_AREA"):
        to_fem_regions(coupler_straight(gap=16.0, length=100.0))


def test_fem_layers_do_not_collide_with_mask_layers():
    """FEM regions must not land on real mask layers (M1_DRAW is (1,0) ...)."""
    mask_layers = {
        tuple(getattr(LAYER, name))
        for name in dir(LAYER)
        if not name.startswith("_") and isinstance(getattr(LAYER, name), tuple)
    }
    for name, layer in FEM_LAYERS.items():
        assert layer not in mask_layers, f"{name} collides with a mask layer"
