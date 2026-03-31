"""Tests for the QPDK Design Rule Checks (DRC) module."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component

from qpdk.cells.helpers import apply_additive_metals
from qpdk.drc import (
    DRCResults,
    _check_exclusion,
    _check_max_width,
    _check_min_space,
    _check_min_width,
    _check_no_overlap,
    run_drc,
)
from qpdk.tech import LAYER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rect(width: float, height: float, layer: tuple[int, int]) -> Component:
    """Create a simple rectangle component on *layer*."""
    return gf.components.rectangle(size=(width, height), layer=layer)


# ---------------------------------------------------------------------------
# Minimum width checks
# ---------------------------------------------------------------------------


class TestMinWidth:
    """Minimum-width check tests."""

    def test_passing_m1_etch(self):
        c = _rect(5.0, 5.0, LAYER.M1_ETCH)
        assert _check_min_width(c, LAYER.M1_ETCH, min_width=4.0) == 0

    def test_failing_m1_etch(self):
        c = _rect(0.5, 10.0, LAYER.M1_ETCH)
        assert _check_min_width(c, LAYER.M1_ETCH, min_width=1.0) > 0

    def test_empty_layer(self):
        c = _rect(5.0, 5.0, LAYER.M1_DRAW)
        assert _check_min_width(c, LAYER.M1_ETCH, min_width=1.0) == 0


# ---------------------------------------------------------------------------
# Minimum spacing checks
# ---------------------------------------------------------------------------


class TestMinSpace:
    """Minimum-spacing check tests."""

    def test_passing_space(self):
        c = Component()
        r1 = c << _rect(5.0, 5.0, LAYER.M1_ETCH)
        r2 = c << _rect(5.0, 5.0, LAYER.M1_ETCH)
        r1.dxmax = 0.0
        r2.dxmin = 3.0  # 3 µm gap – well above 1 µm rule
        assert _check_min_space(c, LAYER.M1_ETCH, min_space=1.0) == 0

    def test_failing_space(self):
        c = Component()
        r1 = c << _rect(5.0, 5.0, LAYER.M1_ETCH)
        r2 = c << _rect(5.0, 5.0, LAYER.M1_ETCH)
        r1.dxmax = 0.0
        r2.dxmin = 0.5  # 0.5 µm gap – below 1.0 µm rule
        assert _check_min_space(c, LAYER.M1_ETCH, min_space=1.0) > 0


# ---------------------------------------------------------------------------
# Maximum width checks
# ---------------------------------------------------------------------------


class TestMaxWidth:
    """Maximum-width check tests."""

    def test_passing_max_width(self):
        c = _rect(50.0, 50.0, LAYER.M1_ETCH)
        assert _check_max_width(c, LAYER.M1_ETCH, max_width=100.0) == 0

    def test_failing_max_width(self):
        c = _rect(200.0, 200.0, LAYER.M1_ETCH)
        assert _check_max_width(c, LAYER.M1_ETCH, max_width=100.0) > 0

    def test_empty_layer(self):
        c = _rect(200.0, 200.0, LAYER.M1_DRAW)
        assert _check_max_width(c, LAYER.M1_ETCH, max_width=100.0) == 0


# ---------------------------------------------------------------------------
# TSV / Indium-bump overlap checks
# ---------------------------------------------------------------------------


class TestTsvIndOverlap:
    """TSV / Indium-bump overlap detection tests."""

    def test_no_overlap(self):
        c = Component()
        r1 = c << _rect(15.0, 15.0, LAYER.TSV)
        r2 = c << _rect(15.0, 15.0, LAYER.IND)
        r1.dxmax = 0.0
        r2.dxmin = 20.0  # well separated
        assert _check_no_overlap(c, LAYER.TSV, LAYER.IND) == 0

    def test_overlap_detected(self):
        c = Component()
        c << _rect(15.0, 15.0, LAYER.TSV)
        c << _rect(15.0, 15.0, LAYER.IND)  # same location → full overlap
        assert _check_no_overlap(c, LAYER.TSV, LAYER.IND) > 0

    def test_partial_overlap(self):
        c = Component()
        r1 = c << _rect(15.0, 15.0, LAYER.TSV)
        r2 = c << _rect(15.0, 15.0, LAYER.IND)
        r1.dxmax = 0.0
        r2.dxmin = -5.0  # 5 µm overlap
        assert _check_no_overlap(c, LAYER.TSV, LAYER.IND) > 0

    def test_exclusion_check(self):
        c = Component()
        r1 = c << _rect(15.0, 15.0, LAYER.TSV)
        r2 = c << _rect(15.0, 15.0, LAYER.IND)
        r1.dxmax = 0.0
        r2.dxmin = 0.0005  # extremely close but not overlapping
        assert _check_exclusion(c, LAYER.TSV, LAYER.IND, min_space=1.0) > 0


# ---------------------------------------------------------------------------
# Full DRC run
# ---------------------------------------------------------------------------


class TestRunDrc:
    """Full DRC run integration tests."""

    def test_clean_component_passes(self):
        """A simple rectangle should pass default DRC."""
        c = _rect(10.0, 10.0, LAYER.M1_ETCH)
        results = run_drc(c)
        assert isinstance(results, DRCResults)
        assert results.passed

    def test_narrow_feature_fails(self):
        c = _rect(0.5, 10.0, LAYER.M1_ETCH)
        results = run_drc(c, m1_etch_min_width=1.0)
        assert not results.passed
        rule_names = [v.rule_name for v in results.violations]
        assert "M1_ETCH.min_width" in rule_names

    def test_overlapping_tsv_ind_fails(self):
        c = Component()
        c << _rect(15.0, 15.0, LAYER.TSV)
        c << _rect(15.0, 15.0, LAYER.IND)
        results = run_drc(c)
        assert not results.passed
        rule_names = [v.rule_name for v in results.violations]
        assert "TSV_IND.no_overlap" in rule_names

    def test_run_drc_by_name(self):
        """DRC can also accept a component name."""
        results = run_drc("indium_bump")
        assert isinstance(results, DRCResults)

    def test_print_summary_does_not_raise(self):
        c = _rect(10.0, 10.0, LAYER.M1_ETCH)
        results = run_drc(c)
        results.print_summary()  # should not raise


# ---------------------------------------------------------------------------
# Integration with apply_additive_metals
# ---------------------------------------------------------------------------


class TestDrcAfterAdditiveMetals:
    """DRC integration with ``apply_additive_metals``."""

    def test_additive_metals_then_drc(self):
        """DRC should work on a component after ``apply_additive_metals``."""
        from qpdk.cells.transmon import flipmon_with_bbox

        c = Component()
        c << flipmon_with_bbox()
        c = apply_additive_metals(c)
        results = run_drc(c)
        assert isinstance(results, DRCResults)
