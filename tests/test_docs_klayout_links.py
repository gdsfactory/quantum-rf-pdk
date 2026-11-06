"""Tests for the ``klayout_links`` documentation extension.

KLayout's API docs have no Sphinx inventory, so the extension maps the
``klayout.*`` type names autodoc emits onto the per-class documentation pages
using the ``.pyi`` stubs of the installed wheel.  These tests pin that mapping
and the ``missing-reference`` handler.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import klayout
import pytest

import qpdk.cells
import qpdk.models

sys.path.insert(0, str(Path(__file__).parents[1] / "docs" / "_ext"))

addnodes = pytest.importorskip(
    "sphinx.addnodes", reason="docs dependency group not installed"
)
nodes = pytest.importorskip("docutils.nodes")
signature = pytest.importorskip("sphinx.util.inspect").signature
stringify_annotation = pytest.importorskip("sphinx.util.typing").stringify_annotation
klayout_links = pytest.importorskip("klayout_links")

BASE_URL = klayout_links.DEFAULT_DOCS_URL

#: The fully qualified names ``stringify_annotation`` emits for KLayout classes.
_ANNOTATED = re.compile(r"~klayout\.\w+\.\w+")


def _recorded_annotations(obj: Any) -> dict[str, str]:
    """Annotations of ``obj`` as autodoc records them, evaluated and shortened."""
    sig = signature(obj, type_aliases={})
    annotations = {
        name: parameter.annotation
        for name, parameter in sig.parameters.items()
        if parameter.annotation is not parameter.empty
    }
    if sig.return_annotation is not sig.empty:
        annotations["return"] = sig.return_annotation
    return {
        name: stringify_annotation(hint, "smart") for name, hint in annotations.items()
    }


@pytest.fixture(scope="module")
def links() -> dict[str, str]:
    """The link table built from the stubs the docs build would read."""
    return klayout_links._build_links(Path(klayout.__path__[0]), BASE_URL)


def test_both_module_spellings_resolve(links: dict[str, str]) -> None:
    """The core module from the annotation and the public one both link."""
    page = f"{BASE_URL}class_DCplxTrans.html"
    assert links["klayout.dbcore.DCplxTrans"] == page
    assert links["klayout.db.DCplxTrans"] == page


def test_every_annotated_klayout_type_has_a_page(links: dict[str, str]) -> None:
    """Every KLayout type in the public annotations maps to a doctree URL."""
    missing: set[str] = set()
    for module in (qpdk.cells, qpdk.models):
        for attr in dir(module):
            obj = getattr(module, attr)
            if attr.startswith("_") or not callable(obj):
                continue
            try:
                annotations = _recorded_annotations(obj)
            except (TypeError, ValueError):  # not introspectable, autodoc skips it too
                continue
            for hint in annotations.values():
                for name in _ANNOTATED.findall(hint):
                    if name[1:] not in links:
                        missing.add(f"{module.__name__}.{attr}: {name[1:]}")
    assert not missing


def test_handler_links_klayout_types(links: dict[str, str]) -> None:
    """An unresolved ``klayout`` class reference becomes an external link."""
    app = cast("Any", SimpleNamespace(_klayout_links=links))
    node = addnodes.pending_xref(
        "", refdomain="py", reftype="class", reftarget="klayout.dbcore.Region"
    )
    reference = klayout_links._resolve_reference(app, None, node, nodes.Text("Region"))
    assert isinstance(reference, nodes.reference)
    assert reference["refuri"] == f"{BASE_URL}class_Region.html"
    assert reference.astext() == "Region"


def test_handler_ignores_other_types(links: dict[str, str]) -> None:
    """A reference to something else is left to the other resolvers."""
    app = cast("Any", SimpleNamespace(_klayout_links=links))
    node = addnodes.pending_xref(
        "", refdomain="py", reftype="class", reftarget="gdsfactory.Component"
    )
    assert (
        klayout_links._resolve_reference(app, None, node, nodes.Text("Component"))
        is None
    )
