"""Tests for qpdk.samples.all_cells."""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf

import qpdk
from qpdk.samples.all_cells import all_cells


def _mark_as_qpdk_cell(func):
    func.__module__ = "qpdk.cells.test"
    return func


def test_all_cells_does_not_treat_var_keyword_as_required(recwarn, monkeypatch) -> None:
    """Ensure variadic keyword parameters are not treated as required arguments."""

    @_mark_as_qpdk_cell
    def accepts_var_keyword(**options):
        _ = options
        return gf.Component()

    @_mark_as_qpdk_cell
    def requires_argument(length):
        _ = length
        return gf.Component()

    monkeypatch.setattr(
        qpdk,
        "PDK",
        SimpleNamespace(
            cells={
                "accepts_var_keyword": accepts_var_keyword,
                "requires_argument": requires_argument,
            }
        ),
    )

    _ = all_cells(spacing=201)

    warning_messages = [str(w.message) for w in recwarn]
    assert any("requires arguments ['length']" in msg for msg in warning_messages)
    assert not any("requires arguments ['options']" in msg for msg in warning_messages)


def test_all_cells_skips_unsupported_return_type(recwarn, monkeypatch) -> None:
    """Ensure unsupported return types are skipped with a warning."""

    @_mark_as_qpdk_cell
    def unsupported_cell():
        return "not a component"

    @_mark_as_qpdk_cell
    def valid_cell():
        return gf.Component()

    monkeypatch.setattr(
        qpdk,
        "PDK",
        SimpleNamespace(
            cells={"unsupported_cell": unsupported_cell, "valid_cell": valid_cell}
        ),
    )

    _ = all_cells(spacing=202)

    warning_messages = [str(w.message) for w in recwarn]
    assert any("unsupported return type str" in msg for msg in warning_messages)


def test_all_cells_sets_unconstrained_max_size_by_default(monkeypatch) -> None:
    """Ensure all_cells uses unconstrained max_size when not provided by callers."""
    captured_pack_kwargs = {}

    @_mark_as_qpdk_cell
    def valid_cell():
        return gf.Component()

    def fake_pack(components, spacing, **kwargs):
        _ = components
        _ = spacing
        captured_pack_kwargs.clear()
        captured_pack_kwargs.update(kwargs)
        return [gf.Component("packed_component")]

    monkeypatch.setattr(
        qpdk,
        "PDK",
        SimpleNamespace(cells={"valid_cell_for_pack": valid_cell}),
    )
    monkeypatch.setattr("qpdk.samples.all_cells.gf.pack", fake_pack)

    _ = all_cells(spacing=203)
    assert captured_pack_kwargs["max_size"] == (None, None)

    _ = all_cells(spacing=204, max_size=(100.0, 200.0))
    assert captured_pack_kwargs["max_size"] == (100.0, 200.0)
