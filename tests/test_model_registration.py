"""Discover model bindings from schematics without evaluating layouts or models."""

from __future__ import annotations

import inspect
import subprocess
import sys
from functools import partial
from types import SimpleNamespace

import pytest

import qpdk


def _declared_layout(length: float = 10.0, *, start_with_bend: bool = False) -> None:
    """Fail if registration tries to construct layout geometry."""
    raise AssertionError(
        f"Registration built a layout with {length=}, {start_with_bend=}"
    )


def _declared_model(f: float = 5e9, length: float = 10.0) -> dict:
    """Fail if registration tries to evaluate a SAX model."""
    raise AssertionError(f"Registration evaluated a model with {f=}, {length=}")


def test_schematic_declares_a_model_without_a_catalog_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new declaration discovers its model and forwards partial model settings."""
    descriptor = {
        "language": "sax",
        "name": "declared_model",
        "module": __name__,
        "qualname": "_declared_model",
    }
    factory = SimpleNamespace(
        schematic_driven=lambda: True,
        get_schematic=lambda **_kwargs: SimpleNamespace(info={"models": [descriptor]}),
    )
    monkeypatch.setattr(
        qpdk,
        "gf",
        SimpleNamespace(
            kcl=SimpleNamespace(
                factories=SimpleNamespace(
                    get_by_qualified_name=lambda _name: factory,
                )
            )
        ),
    )
    cells = {
        "declared": _declared_layout,
        "preset": partial(_declared_layout, length=25.0, start_with_bend=True),
    }
    bindings = qpdk._schematic_model_bindings(cells)
    registered = qpdk._register_sax_models({}, cells, bindings)

    assert registered["declared"] is registered["declared_model"] is _declared_model
    preset = registered["preset"]
    assert isinstance(preset, partial)
    assert preset.func is _declared_model
    assert preset.keywords == {"length": 25.0}
    assert inspect.signature(preset).parameters["length"].default == pytest.approx(25.0)


def test_undeclared_cell_models_stay_in_the_analytical_catalog() -> None:
    """Same-named analytical functions do not imply a supported layout boundary."""
    catalog = {"unmodeled": _declared_model, "primitive": _declared_model}
    registered = qpdk._register_sax_models(catalog, {"unmodeled": _declared_layout}, {})

    assert registered == {"primitive": _declared_model}
    assert catalog["unmodeled"] is _declared_model


def test_import_does_not_construct_layout_geometry() -> None:
    """PDK discovery must work before activation and leave the layout empty."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """from unittest.mock import patch
import gdsfactory as gf
with patch.object(gf.Component, "__init__", side_effect=AssertionError("layout built")):
    import qpdk
assert qpdk.PDK.models
assert not list(gf.kcl.each_cell())
""",
        ],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("is_alias", [False, True])
def test_named_schematic_discovers_virtual_cells_and_aliases(
    monkeypatch: pytest.MonkeyPatch,
    is_alias: bool,
) -> None:
    """Named declarations cover factories without native schematic support."""
    descriptor = {
        "language": "sax",
        "name": "declared_model",
        "module": __name__,
        "qualname": "_declared_model",
    }
    monkeypatch.setattr(
        qpdk._schematic,
        "declared_schematic",
        lambda **_kwargs: SimpleNamespace(info={"models": [descriptor]}),
        raising=False,
    )
    factory = SimpleNamespace(
        schematic_driven=lambda: True,
        get_schematic=lambda **_kwargs: SimpleNamespace(info={"models": []}),
    )
    monkeypatch.setattr(
        qpdk,
        "gf",
        SimpleNamespace(
            kcl=SimpleNamespace(
                factories=SimpleNamespace(
                    get_by_qualified_name=lambda _name: factory if is_alias else None,
                )
            )
        ),
    )
    cell = partial(_declared_layout, length=10.0) if is_alias else _declared_layout
    cells = {"declared": cell}
    bindings = qpdk._schematic_model_bindings(cells)
    registered = qpdk._register_sax_models({}, cells, bindings)

    assert registered["declared"] is registered["declared_model"] is _declared_model
