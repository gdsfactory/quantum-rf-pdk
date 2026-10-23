"""Tests for the nanobolometer and nanocalorimeter cells."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from qpdk.cells import nanobolometer, nanocalorimeter
from qpdk.tech import LAYER


@pytest.mark.parametrize(
    ("cell", "electrical_ports"),
    [(nanobolometer, {"e1", "e2"}), (nanocalorimeter, {"e1"})],
)
def test_ports(cell: gf.typings.ComponentFactory, electrical_ports: set[str]) -> None:
    """One CPW input facing west and one pad port per NIS probe."""
    c = cell()
    assert {p.name for p in c.ports.filter(port_type="optical")} == {"o1"}
    assert {p.name for p in c.ports.filter(port_type="electrical")} == electrical_ports
    assert c.ports["o1"].orientation == pytest.approx(180)


def test_probe_count_follows_positions() -> None:
    """Each probe position gets its own pad and port."""
    c = nanobolometer(probe_positions=(-0.3, 0.0, 0.3))
    assert len(c.ports.filter(port_type="electrical")) == 3


def test_info_volumes() -> None:
    """Volumes in ``info`` follow the drawn geometry in µm³."""
    c = nanobolometer(
        absorber_length=2.0,
        absorber_width=0.2,
        normal_metal_thickness=0.05,
        contact_overlap=0.4,
        extension_length=0.1,
    )
    area = 0.2 * 0.05
    assert c.info["absorber_volume"] == pytest.approx(2.0 * area)
    assert c.info["covered_volume"] == pytest.approx(2 * 0.4 * area)
    assert c.info["extension_volume"] == pytest.approx(2 * 0.1 * area)
    assert c.info["reservoir_volume"] == pytest.approx(
        c.info["covered_volume"] + c.info["extension_volume"]
    )
    assert c.info["absorber_squares"] == pytest.approx(10.0)


def test_normal_metal_extent() -> None:
    """The NM wire spans the absorber plus overlap and extension on both sides."""
    c = nanobolometer(absorber_length=1.0, contact_overlap=0.3, extension_length=0.2)
    box = c.dbbox(gf.get_layer(LAYER.NM_DRAW))
    assert box.height() == pytest.approx(1.0 + 2 * (0.3 + 0.2))
    assert box.width() == pytest.approx(0.135)


def test_calorimeter_is_a_parameter_set() -> None:
    """The calorimeter reuses the bolometer geometry with a long absorber."""
    c = nanocalorimeter()
    assert c.info["absorber_volume"] == pytest.approx(20.0 * 0.1 * 0.025)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"absorber_length": 0.0}, "must be positive"),
        ({"absorber_width": -0.1}, "must be positive"),
        ({"contact_overlap": 0.0}, "must be positive"),
        ({"extension_length": -0.1}, "must be non-negative"),
        ({"probe_positions": (0.6,)}, "does not fit"),
    ],
)
def test_invalid_geometry_raises(kwargs: dict, match: str) -> None:
    """Reject non-physical dimensions and probes off the absorber."""
    with pytest.raises(ValueError, match=match):
        nanobolometer(**kwargs)
