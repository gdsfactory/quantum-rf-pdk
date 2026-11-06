"""Tests for the ``alias_typehints`` documentation extension.

The extension rewrites the fully expanded ``Annotated[...]`` annotations that
autodoc records back into the alias names they came from.  These tests pin the
behaviour for the annotation shapes actually used across the PDK, so a change in
``sax``, ``gdsfactory`` or Sphinx that breaks the rewrite fails here rather than
silently shipping unreadable API docs.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import qpdk.cells
import qpdk.models

sys.path.insert(0, str(Path(__file__).parents[1] / "docs" / "_ext"))

stringify_annotation = pytest.importorskip(
    "sphinx.util.typing", reason="docs dependency group not installed"
).stringify_annotation
alias_typehints = pytest.importorskip("alias_typehints")

#: Longer than this and a parameter type is back to being hard to read.
MAX_RENDERED_LENGTH = 80


@pytest.fixture(scope="module")
def aliases() -> Any:
    """The alias table built from the modules the docs build scans."""
    return alias_typehints._build_alias_map(alias_typehints.DEFAULT_ALIAS_MODULES)


def _simplified(func: Callable[..., Any], aliases: Any) -> dict[str, str]:
    """Annotations of ``func`` as the docs build would render them."""
    return {
        name: alias_typehints.simplify(stringify_annotation(hint, "smart"), aliases)
        for name, hint in inspect.get_annotations(func).items()
    }


def test_sax_model_annotations_collapse_to_aliases(aliases: Any) -> None:
    """A SAX model's parameters render as the sax aliases they were written as."""
    assert _simplified(qpdk.models.airbridge, aliases) == {
        "f": "~sax.FloatArrayLike",
        "cpw_width": "~sax.Float",
        "bridge_width": "~sax.Float",
        "airgap_height": "~sax.Float",
        "loss_tangent": "~sax.Float",
        "return": "~sax.SDict",
    }


def test_reordered_union_members_still_fold(aliases: Any) -> None:
    """``float | ArrayLike`` is flattened by Python but still folds back."""
    simplified = _simplified(qpdk.models.cpw_cpw_coupling_capacitance, aliases)
    assert simplified["length"] == "~jax.typing.ArrayLike"
    assert simplified["gap"] == "~jax.typing.ArrayLike"


def test_no_annotation_renders_as_a_wall_of_text(aliases: Any) -> None:
    """Nothing in the public models or cells API renders as a wall of text."""
    too_long = {
        f"{name}.{param}": simplified
        for module in (qpdk.models, qpdk.cells)
        for name in dir(module)
        if not name.startswith("_") and callable(obj := getattr(module, name))
        for param, hint in inspect.get_annotations(obj).items()
        if len(
            simplified := alias_typehints.simplify(
                stringify_annotation(hint, "smart"), aliases
            )
        )
        > MAX_RENDERED_LENGTH
    }
    assert not too_long


def test_leftover_annotated_metadata_is_dropped() -> None:
    """Validator metadata from aliases we do not know about is still stripped."""
    text = "~typing.Annotated[str, ~pydantic.PlainValidator(func=~sax.val_port)]"
    assert alias_typehints._collapse_annotated(text) == "str"


def test_qualify_only_rewrites_whole_names() -> None:
    """Names are qualified once, and dotted names are left alone."""
    names = {
        "Callable": "collections.abc.Callable",
        "ArrayLike": "numpy.typing.ArrayLike",
        "optuna.Study": "optuna.study.Study",
    }
    assert alias_typehints._qualify("Callable[[ArrayLike], None]", names) == (
        "collections.abc.Callable[[numpy.typing.ArrayLike], None]"
    )
    assert (
        alias_typehints._qualify(
            "~collections.abc.Callable | ~jax.typing.ArrayLike", names
        )
        == "~collections.abc.Callable | ~jax.typing.ArrayLike"
    )
    assert alias_typehints._qualify("optuna.Study", names) == "optuna.study.Study"


def test_data_alias_reference_is_retyped() -> None:
    """A class reference to a data alias is retyped for intersphinx."""
    env = SimpleNamespace(
        intersphinx_cache={},
        intersphinx_inventory={"py:data": {"sax.SDictModel": object()}},
        intersphinx_named_inventory={},
    )
    node = {"refdomain": "py", "reftype": "class", "reftarget": "sax.SDictModel"}
    assert alias_typehints._resolve_data_alias(None, env, node, None) is None
    assert node["reftype"] == "data"


def test_other_references_keep_their_type() -> None:
    """References to real classes are left for the other resolvers."""
    node = {
        "refdomain": "py",
        "reftype": "class",
        "reftarget": "gdsfactory.Component",
    }
    alias_typehints._resolve_data_alias(None, SimpleNamespace(), node, None)
    assert node["reftype"] == "class"
