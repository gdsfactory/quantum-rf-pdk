"""Geometry checks for the flipmon's configurable indium bump."""

from collections.abc import Callable

import gdsfactory as gf
import kfactory as kf
import klayout.db as kdb
import pytest

from qpdk.cells import flipmon, flipmon_with_bbox
from qpdk.simulation import FLIP_CHIP_FEM_LAYERS, to_flip_chip_regions
from qpdk.tech import LAYER


@pytest.mark.parametrize("factory", [flipmon, flipmon_with_bbox])
@pytest.mark.parametrize("diameter", [10.0, 30.0])
def test_flipmon_bump_diameter(
    factory: Callable[..., gf.Component], diameter: float
) -> None:
    component = factory(bump_diameter=diameter)
    bump = kdb.Region(component.kdb_cell.begin_shapes_rec(gf.get_layer(LAYER.IND)))
    box = bump.bbox()
    assert box.width() * component.kcl.dbu == pytest.approx(diameter)
    assert box.height() * component.kcl.dbu == pytest.approx(diameter)
    assert component.ports["center"].width == pytest.approx(diameter)


@pytest.mark.parametrize("dbu", [0.0005, 0.01])
def test_flip_chip_conversion_preserves_physical_dimensions(dbu: float) -> None:
    kcl = kf.KCLayout(f"flip-chip-dbu-{dbu}")
    kcl.layout.dbu = dbu
    component = gf.Component(kcl=kcl)
    layout = component.kdb_cell.layout()
    for layer in (LAYER.SIM_AREA, LAYER.M1_DRAW, LAYER.M2_DRAW):
        component.kdb_cell.shapes(layout.layer(*layer)).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(layout.layer(*LAYER.IND)).insert(kdb.DBox(40, 40, 60, 60))
    regions = to_flip_chip_regions(component)
    assert regions.dbbox().width() == pytest.approx(100)
    assert regions.dbbox().height() == pytest.approx(100)
    result_layout = regions.kdb_cell.layout()
    bump = kdb.Region(
        regions.kdb_cell.begin_shapes_rec(
            result_layout.layer(*FLIP_CHIP_FEM_LAYERS["BUMP"])
        )
    )
    assert bump.bbox().width() * result_layout.dbu == pytest.approx(20)


def test_flip_chip_fem_rejects_airbridges() -> None:
    component = gf.Component()
    component.kdb_cell.shapes(component.kdb_cell.layout().layer(*LAYER.AB_DRAW)).insert(
        kdb.DBox(0, 0, 100, 100)
    )
    with pytest.raises(ValueError, match="AB_DRAW"):
        to_flip_chip_regions(component)


@pytest.mark.parametrize("level", ["M1", "M2"])
def test_flip_chip_fem_rejects_fully_etched_metal(level: str) -> None:
    component = gf.Component()
    layout = component.kdb_cell.layout()
    etch = LAYER.M1_ETCH if level == "M1" else LAYER.M2_ETCH
    other_draw = LAYER.M2_DRAW if level == "M1" else LAYER.M1_DRAW
    for layer in (LAYER.SIM_AREA, etch, other_draw):
        component.kdb_cell.shapes(layout.layer(*layer)).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(layout.layer(*LAYER.IND)).insert(kdb.DBox(40, 40, 60, 60))
    with pytest.raises(ValueError, match=f"{level} is fully etched away"):
        to_flip_chip_regions(component)
