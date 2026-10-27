"""Tests for the gatemon qubit cell."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from klayout.db import DBox, DPoint, Region

from qpdk import PDK
from qpdk.cells.gatemon import gatemon, gatemon_coupler
from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER

PDK.activate()


def _metal_nets(**kwargs) -> tuple[Region, float]:
    """Return the merged final M1 metal of a gatemon and the database unit.

    Final metal is the ground plane outside the etch plus the drawn metal.
    The window ends at the gate port so the open end of the gate line is not
    connected to ground beyond the cell boundary.
    """
    c = gatemon(**kwargs)
    layout = c.kcl.layout
    draw = Region(c.begin_shapes_rec(layout.layer(*LAYER.M1_DRAW)))
    etch = Region(c.begin_shapes_rec(layout.layer(*LAYER.M1_ETCH)))
    bbox = c.dbbox()
    window = DBox(bbox.left - 20, bbox.bottom, bbox.right + 20, bbox.top + 20)
    metal = (Region(window.to_itype(layout.dbu)) - etch + draw).merged()
    return metal, layout.dbu


def _net_of(metal: Region, dbu: float, x: float, y: float) -> int:
    hits = [
        i
        for i, polygon in enumerate(metal.each())
        if polygon.inside(DPoint(x, y).to_itype(dbu))
    ]
    assert len(hits) == 1, f"point ({x}, {y}) is in {len(hits)} metal polygons"
    return hits[0]


def test_gatemon_ports() -> None:
    """Test that the gatemon exposes readout, gate and junction ports."""
    c = gatemon()
    assert {p.name for p in c.ports} == {"readout", "gate", "junction"}
    assert c.ports["readout"].orientation == 90
    assert c.ports["gate"].orientation == 270
    assert c.ports["junction"].port_type == "placement"
    assert c.info["qubit_type"] == "gatemon"


@pytest.mark.parametrize(("side", "sign"), [("bottom", -1), ("top", 1)])
def test_gatemon_coupler_preserves_the_physical_gap(side: str, sign: int) -> None:
    """A circuit reference plane must not add a conducting spur to the island."""
    cell = gatemon_coupler(qubit_side=side)
    line = straight(length=600)
    for layer in (LAYER.M1_DRAW, LAYER.M1_ETCH):
        assert (cell.get_region(layer) ^ line.get_region(layer)).is_empty()
    for name in ("o1", "o2"):
        assert cell.ports[name].center == line.ports[name].center
        assert cell.ports[name].width == line.ports[name].width
        assert cell.ports[name].orientation == line.ports[name].orientation
    assert cell.ports["qubit"].center == (200, sign * 45)
    assert cell.ports["qubit"].width == 150


def test_gatemon_no_shorts() -> None:
    """Test that island, gate and ground are three separate M1 nets.

    The M1 island, gate feed and ground remain separate before adding the
    transferred nanotube and insulated top gate.
    """
    metal, dbu = _metal_nets()
    assert metal.count() == 3

    island = _net_of(metal, dbu, 0, 0)
    upper_pad = _net_of(metal, dbu, -40, -50)
    gate_tip = _net_of(metal, dbu, 60, -134)
    gate_port = _net_of(metal, dbu, 60, -245)
    ground = _net_of(metal, dbu, -150, 0)
    lower_pad = _net_of(metal, dbu, -40, -220)

    assert island == upper_pad
    assert gate_tip == gate_port
    assert ground == lower_pad
    assert len({island, gate_tip, ground}) == 3


def test_gatemon_junction_bridges_lead_gap() -> None:
    """The transferred nanotube overlaps two contacts without a metal short."""
    c = gatemon()
    layout = c.kcl.layout
    contacts = c.get_region(LAYER.CNT_CONTACT)
    tube = c.get_region(LAYER.CNT)
    x, y = c.ports["junction"].center
    assert (x, y) == pytest.approx((-20, -134))

    assert contacts.merged().count() == 2
    assert tube.merged().flatten().count() == 1
    assert (contacts & tube).merged().count() == 2
    assert c.get_region(LAYER.JJ_AREA).is_empty()
    assert (c.get_region(LAYER.JJ_PATCH) & tube).is_empty()
    assert (c.get_region(LAYER.M1_DRAW) & tube).is_empty()
    gap = DBox(x - 0.05, y - 0.4, x + 0.05, y + 0.4).to_itype(layout.dbu)
    assert (contacts & Region(gap)).is_empty()
    assert tube.bbox().width() * layout.dbu == pytest.approx(0.002)
    shapes = list(c.shapes(layout.layer(*LAYER.CNT)).each())
    assert len(shapes) == 1
    assert shapes[0].is_path()


def test_gatemon_hbn_insulates_top_gate() -> None:
    """Every gate/contact overlap and the entire tube lie under hBN."""
    c = gatemon()
    hbn = c.get_region(LAYER.HBN)
    tube = c.get_region(LAYER.CNT)
    gate = c.get_region(LAYER.TOP_GATE)
    contacts = c.get_region(LAYER.CNT_CONTACT)
    patches = c.get_region(LAYER.JJ_PATCH)

    assert (tube - hbn).is_empty()
    assert hbn.bbox().left * c.kcl.dbu > -39
    assert not (gate & tube).is_empty()
    assert (gate & tube - hbn).is_empty()
    assert (gate & contacts).merged().count() == 2
    assert ((gate & contacts) - hbn).is_empty()
    assert (patches & hbn).is_empty()
    assert patches.merged().count() == 2


def test_gatemon_reconnection_patches() -> None:
    """Electrode patches reconnect to M1, and the gate contacts M1 directly."""
    c = gatemon()
    m1, dbu = _metal_nets()
    contacts = c.get_region(LAYER.CNT_CONTACT)
    gate = c.get_region(LAYER.TOP_GATE)
    patches = c.get_region(LAYER.JJ_PATCH)
    lower_metal = (m1 | contacts | patches).merged()
    assert lower_metal.count() == 3
    assert _net_of(lower_metal, dbu, 0, 0) == _net_of(lower_metal, dbu, -40, -100)
    assert _net_of(lower_metal, dbu, -150, 0) == _net_of(lower_metal, dbu, -40, -160)
    assert (patches & contacts).merged().count() == 2
    assert (patches & gate).is_empty()
    assert (gate & m1).merged().count() == 1
    assert ((gate & m1) & c.get_region(LAYER.HBN)).is_empty()


@given(
    junction_length=st.floats(min_value=0.2, max_value=4, allow_nan=False),
    overlap=st.floats(min_value=0.2, max_value=1, allow_nan=False),
)
@settings(deadline=None, max_examples=20)
def test_nanotube_enclosure(junction_length: float, overlap: float) -> None:
    """Coverage survives changes to the weak-link span and contact overlap."""
    c = gatemon(junction_length=junction_length, nanotube_contact_overlap=overlap)
    tube = c.get_region(LAYER.CNT)
    assert (tube - c.get_region(LAYER.HBN)).is_empty()
    assert (tube & c.get_region(LAYER.CNT_CONTACT)).merged().count() == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hbn_size": (10.0, 1.0)}, "enclose the entire nanotube"),
        ({"hbn_size": (20.0, 3.0)}, "insulate every"),
        ({"hbn_size": (80.0, 160.0)}, "between island and ground"),
        ({"hbn_size": (200.0, 10.0)}, "gate landing"),
        ({"nanotube_contact_overlap": 30.0}, "contact tips"),
        ({"junction_width": 0.001}, "database units"),
        ({"gate_tip_gap": 0.0}, "positive"),
        ({"lead_width": 61.0}, "contact branches"),
        ({"size_patch": (60.0, 55.0)}, "fit the electrodes"),
    ],
)
def test_gatemon_rejects_invalid_stack_geometry(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gatemon(**kwargs)


def test_gatemon_electrodes_under_left_half_of_island() -> None:
    """Test that the default electrodes sit under the island's left half.

    The gate line enters to the right of the electrodes, as in Fig. 1b of
    Riechert et al. (2025).
    """
    c = gatemon()
    junction_x = c.ports["junction"].x
    assert -75 < junction_x < 0
    assert c.ports["gate"].x > junction_x


@pytest.mark.parametrize(
    "electrode_offset",
    [-40.0, -20.0, 0.0],
)
def test_gatemon_no_shorts_for_electrode_offsets(electrode_offset: float) -> None:
    """Test that moving the electrodes keeps island, gate and ground separate."""
    metal, _ = _metal_nets(electrode_offset=electrode_offset)
    assert metal.count() == 3


def test_gatemon_gate_overlapping_electrode_raises() -> None:
    """Test that a gate line on top of the junction electrodes is rejected."""
    with pytest.raises(ValueError, match="junction electrodes"):
        gatemon(gate_offset=20)


def test_gatemon_gate_outside_pocket_raises() -> None:
    """Test that a gate line running outside the pocket is rejected."""
    with pytest.raises(ValueError, match="outside the pocket"):
        gatemon(gate_offset=170)


def test_gatemon_electrodes_outside_island_raises() -> None:
    """Test that electrodes beyond the island edge are rejected."""
    with pytest.raises(ValueError, match="under the island"):
        gatemon(electrode_offset=-60)
