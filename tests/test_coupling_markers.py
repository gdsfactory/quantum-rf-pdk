"""Tests for the COUPLING intent layer and the helper that draws it."""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb
from hypothesis import given, settings, strategies as st
from kfactory.instance import DInstance

from qpdk import PDK
from qpdk.cells._coupling import mark_coupling_regions, reference_region
from qpdk.cells.capacitor import plate_capacitor
from qpdk.cells.waveguides import coupler_straight, straight
from qpdk.tech import LAYER
from qpdk.utils import remove_metadata_layers

PDK.activate()

SETTINGS = settings(max_examples=20, deadline=None)


def _region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    return kdb.Region(component.begin_shapes_rec(component.kcl.layer(*layer)))


def _parallel_pair(
    separation: float, length: float = 20.0
) -> tuple[gf.Component, DInstance, DInstance]:
    component = gf.Component()
    lower = component << straight(length=length, cross_section="cpw")
    upper = component << straight(length=length, cross_section="cpw")
    upper.dmovey(10 + separation)
    return component, lower, upper


def test_coupler_straight_marks_its_gap():
    component = coupler_straight(gap=16)
    markers = component.begin_shapes_rec(component.kcl.layer(*LAYER.COUPLING))

    assert not kdb.Region(markers).is_empty()
    # The marker covers the gap, not the conductors.
    assert (
        _region(component, LAYER.COUPLING) & _region(component, LAYER.M1_DRAW)
    ).is_empty()


def test_plate_capacitor_marks_its_gap():
    component = plate_capacitor(gap=7)
    assert not _region(component, LAYER.COUPLING).is_empty()


def test_marker_is_geometry_that_tapeout_strips():
    component = coupler_straight()
    cleaned = remove_metadata_layers(component)
    assert _region(cleaned, LAYER.COUPLING).is_empty()


def test_reference_region_returns_parent_frame_polygons():
    component, lower, upper = _parallel_pair(separation=26)

    lower_region = reference_region(lower)
    upper_region = reference_region(upper)
    assert lower_region.bbox().top < upper_region.bbox().bottom

    mark_coupling_regions(component, lower, upper, gap=16)
    marker = _region(component, LAYER.COUPLING)
    assert lower_region.bbox().top <= marker.bbox().bottom
    assert marker.bbox().top <= upper_region.bbox().bottom


@SETTINGS
@given(
    separation=st.integers(min_value=2, max_value=40),
    marker_gap=st.integers(min_value=2, max_value=40),
)
def test_marker_covers_the_corridor_between_the_lines(
    separation: int, marker_gap: int
) -> None:
    """The marker spans the gap, or narrows symmetrically when the parts are far apart."""
    component, lower, upper = _parallel_pair(separation)
    mark_coupling_regions(component, lower, upper, gap=marker_gap)
    marker = _region(component, LAYER.COUPLING)

    expected = max(0, min(separation, 2 * marker_gap - separation))
    if expected == 0:
        assert marker.is_empty()
        return
    measured = marker.bbox().height() * component.kcl.dbu
    assert abs(measured - expected) < 1e-9


@SETTINGS
@given(
    separation=st.integers(min_value=-20, max_value=20),
    marker_gap=st.integers(min_value=2, max_value=24),
)
def test_marker_never_touches_the_conductors(separation: int, marker_gap: int) -> None:
    component, lower, upper = _parallel_pair(separation, length=30.0)
    mark_coupling_regions(component, lower, upper, gap=marker_gap)

    marker = _region(component, LAYER.COUPLING)
    metal = _region(component, LAYER.M1_DRAW)
    assert (marker & metal).is_empty()


@SETTINGS
@given(
    separation=st.integers(min_value=2, max_value=40),
    narrow=st.integers(min_value=2, max_value=20),
    delta=st.integers(min_value=1, max_value=20),
)
def test_a_wider_marker_covers_a_narrower_one(
    separation: int, narrow: int, delta: int
) -> None:
    narrow_component, narrow_lower, narrow_upper = _parallel_pair(separation)
    wide_component, wide_lower, wide_upper = _parallel_pair(separation)
    mark_coupling_regions(narrow_component, narrow_lower, narrow_upper, gap=narrow)
    mark_coupling_regions(wide_component, wide_lower, wide_upper, gap=narrow + delta)

    narrow_marker = _region(narrow_component, LAYER.COUPLING)
    wide_marker = _region(wide_component, LAYER.COUPLING)
    assert (narrow_marker - wide_marker).is_empty()


@SETTINGS
@given(
    dx=st.integers(min_value=-20, max_value=20),
    dy=st.integers(min_value=-20, max_value=20),
)
def test_reference_region_follows_the_reference(dx: int, dy: int) -> None:
    component = gf.Component()
    ref = component << straight(length=12, cross_section="cpw")
    ref.dmove((dx, dy))

    region = reference_region(ref)
    assert region.bbox().left == round(dx / component.kcl.dbu)
    assert region.bbox().bottom == round(dy / component.kcl.dbu) - 5000
    assert region.area() == 10 * 12 * 10**6
