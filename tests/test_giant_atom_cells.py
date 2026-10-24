"""Tests for the giant-atom layout cells and sample chip."""

import math

import gdsfactory as gf
import pytest
from klayout.db import DBox, Region

from qpdk import PDK
from qpdk.cells.giant_atom import braided_giant_atoms, giant_atom_transmon
from qpdk.samples.giant_atom_chip import giant_atom_chip
from qpdk.tech import LAYER

PDK.activate()


def _region(component: gf.Component, layer) -> Region:
    """Return the merged shapes of ``layer`` in ``component``."""
    layout = component.kcl.layout
    index = layout.layer(*gf.get_layer_tuple(layer))
    return Region(component._base.kdb_cell.begin_shapes_rec(index)).merged()


def test_giant_atom_transmon_ports() -> None:
    """The cell exposes both waveguide ends and the junction port."""
    c = giant_atom_transmon()
    assert {p.name for p in c.ports} == {"o1", "o2", "junction"}
    assert c.ports["o1"].orientation == 180
    assert c.ports["o2"].orientation == 0
    assert c.ports["junction"].port_type == "placement"
    # The waveguide runs along y = 0 between the coupling points.
    assert c.ports["o1"].center[1] == 0
    assert c.ports["o2"].center[1] == 0


@pytest.mark.parametrize(
    ("coupler_lengths", "segment_lengths"),
    [
        ((160.0,), ()),
        ((160.0, 160.0), (13100.0,)),
        ((120.0, 160.0, 200.0), (6000.0, 9000.0)),
    ],
)
def test_waveguide_length(
    coupler_lengths: tuple[float, ...], segment_lengths: tuple[float, ...]
) -> None:
    """The drawn waveguide has the requested length between coupling points."""
    end_length = 100.0
    c = giant_atom_transmon(
        coupler_lengths=coupler_lengths,
        segment_lengths=segment_lengths,
        end_length=end_length,
    )
    expected = (
        sum(segment_lengths)
        + (coupler_lengths[0] + coupler_lengths[-1]) / 2
        + 2 * end_length
    )
    assert math.isclose(c.info["line_length"], expected, abs_tol=1e-2)
    assert len(c.info["meander_depths"]) == len(segment_lengths)


def test_braided_ports_and_symmetry() -> None:
    """Braided cell has one junction per atom on opposite sides of the line."""
    c = braided_giant_atoms()
    assert {p.name for p in c.ports} == {"o1", "o2", "junction_a", "junction_b"}
    assert c.ports["junction_a"].center[1] > 0
    assert c.ports["junction_b"].center[1] < 0
    depths = c.info["meander_depths"]
    assert len(depths) == 3
    assert depths[0] == pytest.approx(depths[1]) == pytest.approx(depths[2])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupler_lengths": (160.0, 160.0), "segment_lengths": ()}, "one segment"),
        ({"segment_lengths": (500.0,)}, "shorter than their spacing"),
        ({"meander_pitch": 60.0}, "meander_pitch"),
        ({"segment_lengths": (1050.0,)}, "Cannot meander"),
        ({"coupler_lengths": (1200.0, 1200.0)}, "overlap"),
        ({"arm_length": 20.0}, "too short for the airbridges"),
    ],
)
def test_invalid_inputs(kwargs: dict, match: str) -> None:
    """Inconsistent or unrealisable geometries raise a clear error."""
    with pytest.raises(ValueError, match=match):
        giant_atom_transmon(**kwargs)


def test_giant_atom_chip_builds() -> None:
    """The sample chip builds with launchers on both ends."""
    c = giant_atom_chip()
    names = {inst.cell.name for inst in c.insts}
    assert any("braided_giant_atoms" in name for name in names)
    assert sum("launcher" in name for name in (i.cell.name for i in c.insts)) == 2


@pytest.mark.parametrize("cell", [giant_atom_transmon, braided_giant_atoms])
def test_waveguide_keeps_cross_section_gap(cell) -> None:
    """The qubit etch does not widen the CPW gap set by the cross-section."""
    c = cell()
    xs = gf.get_cross_section("cpw")
    etch_sections = [s for s in xs.sections if s.layer == LAYER.M1_ETCH]
    gap = etch_sections[0].width
    dbu = c.kcl.layout.dbu
    etch = _region(c, LAYER.M1_ETCH)
    # Cuts across the straight feed sections, beside the first pad and at
    # both ends.
    for x in (c.ports["o1"].center[0] + 10, -100.0, c.ports["o2"].center[0] - 10):
        cut = etch & Region(DBox(x - 0.5, -40, x + 0.5, 40).to_itype(dbu))
        heights = sorted(round(p.bbox().height() * dbu, 3) for p in cut.each())
        assert heights == [gap, gap], (x, heights)
    # No etch on drawn metal anywhere.
    assert (etch & _region(c, LAYER.M1_DRAW)).is_empty()


@pytest.mark.parametrize(
    ("cell", "n_bridges"), [(giant_atom_transmon, 2), (braided_giant_atoms, 4)]
)
def test_enclosed_ground_is_bridged(cell, n_bridges: int) -> None:
    """Every arm of a multi-point atom carries an airbridge to tie the ground."""
    c = cell()
    bridge_spans = _region(c, LAYER.AB_DRAW)
    assert bridge_spans.count() == n_bridges
    assert _region(c, LAYER.AB_VIA).count() == 2 * n_bridges
    # Landing pads sit on ground metal, never on an etched gap.
    assert (_region(c, LAYER.AB_VIA) & _region(c, LAYER.M1_ETCH)).is_empty()
    assert _region(cell(airbridges=False), LAYER.AB_DRAW).is_empty()


def test_single_point_has_no_airbridge() -> None:
    """With one coupling point there is no enclosed ground and no bridge."""
    c = giant_atom_transmon(coupler_lengths=(160.0,), segment_lengths=())
    assert _region(c, LAYER.AB_DRAW).is_empty()


def test_braided_island_over_meander_raises() -> None:
    """An island that would cover another atom's meander is rejected."""
    with pytest.raises(ValueError, match="lies under the island"):
        braided_giant_atoms(island_extension=110.0)
