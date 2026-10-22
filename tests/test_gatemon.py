"""Tests for the gatemon qubit cell."""

import pytest
from klayout.db import DBox, DPoint, Region

from qpdk import PDK
from qpdk.cells.gatemon import gatemon
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


def test_gatemon_no_shorts() -> None:
    """Test that island, gate and ground are three separate M1 nets.

    The island and its junction electrode, the gate line, and the ground
    plane with the lower junction electrode must not touch in M1; the
    junction itself is the only link between island and ground.
    """
    metal, dbu = _metal_nets()
    assert metal.count() == 3

    island = _net_of(metal, dbu, 0, 0)
    upper_lead = _net_of(metal, dbu, 0, -100)
    gate_tip = _net_of(metal, dbu, 10, -115)
    gate_port = _net_of(metal, dbu, 40, -204)
    ground = _net_of(metal, dbu, -150, 0)
    lower_electrode = _net_of(metal, dbu, 0, -150)

    assert island == upper_lead
    assert gate_tip == gate_port
    assert ground == lower_electrode
    assert len({island, gate_tip, ground}) == 3


def test_gatemon_junction_bridges_lead_gap() -> None:
    """Test that the lead is open in M1 and the junction marker bridges it."""
    c = gatemon()
    layout = c.kcl.layout
    draw = Region(c.begin_shapes_rec(layout.layer(*LAYER.M1_DRAW)))
    jj = Region(c.begin_shapes_rec(layout.layer(*LAYER.JJ_AREA)))
    x, y = c.ports["junction"].center
    assert (x, y) == pytest.approx((0, -115))

    assert len(list(jj.each_merged())) == 1
    # Junction marker overlaps the lead on both sides of the gap
    assert len(list((draw & jj).each_merged())) == 2
    # No M1 in the gap itself
    gap = DBox(-0.05, y - 0.4, 0.05, y + 0.4).to_itype(layout.dbu)
    assert (draw & Region(gap)).is_empty()


def test_gatemon_gate_overlapping_electrode_raises() -> None:
    """Test that a gate line on top of the junction electrodes is rejected."""
    with pytest.raises(ValueError, match="gate_offset"):
        gatemon(gate_offset=20)
