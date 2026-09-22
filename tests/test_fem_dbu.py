"""Physical-unit checks for layouts with non-default database units."""

import gdsfactory as gf
import kfactory as kf
import klayout.db as kdb
import pytest
from kfactory.layer import LayerEnum

from qpdk.simulation import to_fem_regions
from qpdk.tech import LAYER


@pytest.mark.parametrize("dbu", [0.0005, 0.01])
def test_fem_conversion_preserves_physical_dimensions(dbu: float) -> None:
    kcl = kf.KCLayout(f"fem-dbu-{dbu}")
    kcl.layout.dbu = dbu
    component = gf.Component(kcl=kcl)
    layout = component.kdb_cell.layout()
    component.kdb_cell.shapes(layout.layer(*LAYER.SIM_AREA)).insert(
        kdb.DBox(0, 0, 100, 100)
    )
    component.add_port(
        name="probe",
        center=(50, 50),
        width=10,
        orientation=0,
        layer=layout.layer(*LAYER.M1_DRAW),
        port_type="electrical",
    )
    regions = to_fem_regions(component)
    assert regions.dbbox().width() == pytest.approx(100)
    assert regions.dbbox().height() == pytest.approx(100)
    assert regions.ports["probe"].center == pytest.approx((50, 50))
    assert regions.ports["probe"].width == pytest.approx(10)


@pytest.mark.parametrize(
    "layer", [LAYER.M2_DRAW, LAYER.M2_ETCH, LAYER.AB_DRAW, LAYER.AB_VIA]
)
def test_single_chip_fem_rejects_unsupported_metal(layer: LayerEnum) -> None:
    component = gf.Component()
    layout = component.kdb_cell.layout()
    for mask in (LAYER.SIM_AREA, layer):
        component.kdb_cell.shapes(layout.layer(*mask)).insert(kdb.DBox(0, 0, 100, 100))
    with pytest.raises(ValueError, match=layer.name):
        to_fem_regions(component)


def test_single_chip_fem_allows_junction_masks_for_lumped_ports() -> None:
    component = gf.Component()
    layout = component.kdb_cell.layout()
    for mask in (LAYER.SIM_AREA, LAYER.JJ_AREA, LAYER.JJ_PATCH):
        component.kdb_cell.shapes(layout.layer(*mask)).insert(kdb.DBox(0, 0, 100, 100))
    assert to_fem_regions(component).dbbox().width() == pytest.approx(100)
