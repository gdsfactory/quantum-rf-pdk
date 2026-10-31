"""Xmon circuit settings preserve the four physical arm ports."""

import pytest

from qpdk.cells import xmon_transmon


def test_junction_settings_are_available_on_the_layout() -> None:
    component = xmon_transmon(inductance_nh=10.0, junction_capacitance_ff=2.0)
    assert component.settings.inductance_nh == pytest.approx(10.0)
    assert component.settings.junction_capacitance_ff == pytest.approx(2.0)
    assert {p.name for p in component.ports if p.port_type != "placement"} == {
        "top_arm",
        "right_arm",
        "bottom_arm",
        "left_arm",
    }
    assert component.ports["junction"].port_type == "placement"


@pytest.mark.parametrize(
    "settings", [{"inductance_nh": 0.0}, {"junction_capacitance_ff": -1.0}]
)
def test_nonphysical_junction_settings_are_rejected(settings: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"positive inductance|nonnegative junction"):
        xmon_transmon(**settings)
