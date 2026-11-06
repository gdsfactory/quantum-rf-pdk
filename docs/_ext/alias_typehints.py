"""Render library type aliases instead of their fully-expanded definitions.

Most of the public API is annotated with aliases such as :obj:`sax.Float`,
:obj:`sax.FloatArrayLike` or :obj:`gdsfactory.typings.CrossSectionSpec`.  Modules
without ``from __future__ import annotations`` have already evaluated the aliases
into the :py:obj:`~typing.Annotated` / union objects they stand for by the time
autodoc sees them, so a single ``sax.Float`` parameter renders in the docs as::

    Annotated[
        float | floating,
        PlainValidator(
            func=~sax.saxtypes.core.val_float, json_schema_input_type=~typing.Any
        ),
    ]

This extension maps the expansions back onto the alias names.  Rather than
hand-maintaining regexes, it imports the modules listed in
``alias_typehints_modules`` and stringifies every public alias they export with
the same function autodoc uses, which yields an exact expansion → alias lookup
table that stays correct as the upstream aliases change.

The rewrite happens on ``autodoc-process-signature`` at a priority above
``sphinx.ext.autodoc.typehints.record_typehints`` (500), so the short names are
what ends up in ``current_document.autodoc_annotations`` and therefore what
``autodoc_typehints = "description"`` renders into the parameter list.  The alias
names are emitted fully qualified so intersphinx turns them into links;
``python_use_unqualified_type_names`` shortens the displayed text again.

Two related repairs also live here, because both are about turning rendered type
text into working links:

- ``alias_typehints_qualify`` rewrites names that Sphinx cannot evaluate and that
  upstream inventories store under a different qualified name (for example
  ``optuna.Study``, which optuna documents as ``optuna.study.Study``).
- ``py:class`` references to aliases that upstream inventories register as
  ``py:data`` (sax and numpy aliases are data, not classes) are retyped before
  intersphinx sees them, which otherwise ignores the entry.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import TYPE_CHECKING, Any

from sphinx.ext.intersphinx import InventoryAdapter
from sphinx.util import logging
from sphinx.util.typing import stringify_annotation

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment
    from sphinx.ext.autodoc import Options

logger = logging.getLogger(__name__)

#: Modules whose public names are scanned for type aliases.
DEFAULT_ALIAS_MODULES = ("sax", "gdsfactory.typings", "jax.typing", "numpy.typing")

#: An expansion is only worth replacing if it is actually longer and composite.
_MIN_GAIN = 4


def _public_names(module: Any) -> list[str]:
    """Names a module advertises, preferring ``__all__``."""
    names = getattr(module, "__all__", None) or dir(module)
    return [name for name in names if not name.startswith("_")]


def _is_alias(obj: Any) -> bool:
    """Whether ``obj`` looks like a type alias rather than a value or a class."""
    if obj is None or inspect.ismodule(obj) or inspect.isroutine(obj):
        return False
    # Real classes document themselves fine; it is the composite aliases
    # (unions, Annotated, parameterised generics) that need shortening.
    return not inspect.isclass(obj)


class AliasTable:
    """Stringified alias expansions, keyed for whole-string and union matching."""

    def __init__(self) -> None:
        """Start empty; ``add`` fills both lookups."""
        #: ``"<full expansion>" -> "~module.Alias"``
        self.exact: dict[str, str] = {}
        #: ``({"member", ...}, "~module.Alias")`` for aliases that are unions.
        #: Python flattens and reorders nested unions, so ``float | ArrayLike``
        #: does not contain ``ArrayLike``'s expansion as a substring; matching
        #: on the member set catches those.
        self.unions: list[tuple[frozenset[str], str]] = []

    def add(self, expansion: str, alias: str) -> None:
        """Register that ``expansion`` is what ``alias`` stands for."""
        if expansion in self.exact:
            # Two aliases can share an expansion (e.g. re-exports); keep the
            # first, which follows ``__all__`` order and so is the canonical one.
            return
        self.exact[expansion] = alias
        members = _split_top_level(expansion, "|")
        if len(members) > 1:
            self.unions.append((frozenset(members), alias))

    def __len__(self) -> int:
        """Number of registered aliases."""
        return len(self.exact)


def _build_alias_map(modules: tuple[str, ...]) -> AliasTable:
    """Collect the type aliases exported by ``modules``."""
    aliases = AliasTable()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # optional dependency, nothing to shorten
            logger.info("[alias_typehints] skipping %s (not importable)", module_name)
            continue

        for name in _public_names(module):
            obj = getattr(module, name, None)
            if not _is_alias(obj):
                continue
            expansion = stringify_annotation(obj, "smart")
            if ("[" not in expansion and "|" not in expansion) or len(expansion) < len(
                name
            ) + _MIN_GAIN:
                continue
            # ``~`` is Sphinx's "show only the last component" marker, matching
            # how it renders aliases it already knows how to name.
            aliases.add(expansion, f"~{module_name}.{name}")
    return aliases


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split ``text`` on ``sep`` occurrences that are not nested inside brackets."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _collapse_annotated(text: str) -> str:
    """Reduce any leftover ``Annotated[T, ...]`` to just ``T``.

    Aliases that are not exported by a scanned module (``sax``'s internal port
    types, for instance) still carry pydantic validator metadata that means
    nothing to a reader.  Only the first argument is the actual type.

    Returns:
        ``text`` with every ``Annotated`` wrapper replaced by its first argument,
        or unchanged if its brackets do not balance.
    """
    marker = "Annotated["
    while (index := text.find(marker)) != -1:
        open_bracket = index + len(marker) - 1
        depth = 0
        for i in range(open_bracket, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    inner = text[open_bracket + 1 : i]
                    # ``~typing.Annotated`` / ``typing.Annotated`` prefix too.
                    prefix_start = index
                    match = re.search(r"[~\w.]*$", text[:index])
                    if match:
                        prefix_start = index - len(match.group())
                    text = (
                        text[:prefix_start]
                        + _split_top_level(inner)[0].strip()
                        + text[i + 1 :]
                    )
                    break
        else:  # unbalanced, leave it alone rather than mangle it
            return text
    return text


def _fold_union(text: str, aliases: AliasTable) -> str:
    """Replace any subset of ``text``'s union members that forms a known alias."""
    members = _split_top_level(text, "|")
    if len(members) < 2:
        return text
    # Widest alias first: a union that matches several aliases should collapse
    # to the one covering the most members.
    for candidates, alias in sorted(
        aliases.unions, key=lambda item: len(item[0]), reverse=True
    ):
        if not candidates <= set(members):
            continue
        members = [alias, *(m for m in members if m not in candidates)]
    return " | ".join(members)


def simplify(annotation: str, aliases: AliasTable) -> str:
    """Shorten a stringified annotation using ``aliases``."""
    # Union folding goes first: substring replacement rewrites individual members
    # and would stop the member sets from lining up.
    annotation = _fold_union(annotation, aliases)
    # Longest first so an outer alias wins over an alias nested inside it.
    for expansion in sorted(aliases.exact, key=len, reverse=True):
        if expansion in annotation:
            annotation = annotation.replace(expansion, aliases.exact[expansion])
    return _collapse_annotated(annotation)


def _qualify(annotation: str, names: dict[str, str]) -> str:
    """Replace bare type names with the names the reference docs are keyed by.

    Only whole names are replaced, so ``Vector3`` is untouched by a ``Vector``
    entry and an already qualified name is not rewritten again.

    Returns:
        The annotation with each known whole name replaced.
    """
    for name, qualified in sorted(
        names.items(), key=lambda item: len(item[0]), reverse=True
    ):
        annotation = re.sub(
            rf"(?<![\w.]){re.escape(name)}(?![.\w])", qualified, annotation
        )
    return annotation


def _rewrite_recorded_typehints(
    app: Sphinx,
    _objtype: str,
    name: str,
    _obj: Any,
    _options: Options,
    _args: str | None,
    _retann: str | None,
) -> None:
    """Shorten the annotations ``record_typehints`` just stored for ``name``."""
    aliases = getattr(app, "_alias_typehints_map", None)
    if not aliases:
        return
    recorded = app.env.current_document.autodoc_annotations.get(name)
    if not recorded:
        return
    names = dict(app.config.alias_typehints_qualify)
    for param, annotation in recorded.items():
        recorded[param] = _qualify(simplify(annotation, aliases), names)


def _resolve_data_alias(
    _app: Sphinx,
    env: BuildEnvironment,
    node: Any,
    _contnode: Any,
) -> None:
    """Retarget a class reference to an alias documented as a data item."""
    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return
    inventory = InventoryAdapter(env).main_inventory
    target = node.get("reftarget")
    is_class = target in inventory.get("py:class", {}) or target in inventory.get(
        "py:exc", {}
    )
    if target in inventory.get("py:data", {}) and not is_class:
        # Intersphinx only looks in the inventory objtypes that match the
        # reference, and a data alias is invisible to a class reference.
        node["reftype"] = "data"


def _init_alias_map(app: Sphinx) -> None:
    """Build the alias table once, before any object is documented."""
    app._alias_typehints_map = _build_alias_map(
        tuple(app.config.alias_typehints_modules)
    )
    logger.info(
        "[alias_typehints] %d alias expansions registered",
        len(app._alias_typehints_map),
    )


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension."""
    app.add_config_value(
        "alias_typehints_modules", list(DEFAULT_ALIAS_MODULES), "env", types=(list,)
    )
    app.add_config_value("alias_typehints_qualify", {}, "env", types=(dict,))
    app.connect("builder-inited", _init_alias_map)
    # Priority above record_typehints (500) so we see, and replace, what it recorded.
    app.connect("autodoc-process-signature", _rewrite_recorded_typehints, priority=900)
    # Before intersphinx (500) so it resolves the retyped reference.
    app.connect("missing-reference", _resolve_data_alias, priority=400)
    return {"version": "1.0", "parallel_read_safe": True}
