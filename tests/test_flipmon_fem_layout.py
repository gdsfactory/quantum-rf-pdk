"""The boxed flipmon exposes one ring port and a movable top-chip mask."""

import gdsfactory as gf
import pytest

from qpdk.cells import flipmon_with_bbox
from qpdk.tech import LAYER


@pytest.mark.parametrize("alignment", [-15.0, 15.0])
def test_top_pad_and_moat_move_together(alignment: float) -> None:
    nominal = flipmon_with_bbox()
    shifted = flipmon_with_bbox(alignment=alignment, chip_gap=15.0, inductance_nh=10.0)
    for layer in (LAYER.M2_DRAW, LAYER.M2_ETCH):
        expected = nominal.get_region(layer).transformed(
            gf.kdb.Trans(round(alignment / nominal.kcl.dbu), 0)
        )
        assert (expected ^ shifted.get_region(layer)).is_empty()
    for layer in (LAYER.M1_DRAW, LAYER.M1_ETCH, LAYER.IND):
        assert (nominal.get_region(layer) ^ shifted.get_region(layer)).is_empty()
    assert {p.name for p in shifted.ports if p.port_type != "placement"} == {
        "outer_ring_outside"
    }
    assert shifted.settings.chip_gap == pytest.approx(15.0)
    assert shifted.settings.inductance_nh == pytest.approx(10.0)
    assert shifted.get_region(LAYER.SIM_AREA).bbox().to_dtype(
        nominal.kcl.dbu
    ) == gf.kdb.DBox(-400, -400, 400, 400)


@pytest.mark.parametrize(
    "settings",
    [
        {"chip_gap": 0.0},
        {"inductance_nh": 0.0},
        {"junction_capacitance_ff": -1.0},
        {"alignment": 102.5},
    ],
)
def test_invalid_chip_or_junction_settings_are_rejected(
    settings: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=r"positive|nonnegative|bump"):
        flipmon_with_bbox(**settings)
