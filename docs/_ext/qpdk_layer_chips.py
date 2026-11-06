"""Inline layer swatches for the docs tables.

``{layer}`M1``` typesets a layer name as code with the swatch KLayout draws it
in. The colour comes from ``qpdk/layers.yaml``, the same file
``docs/figures/_layers.typ`` reads, so a colour change reaches the table without
a second copy. Only HTML can carry an inline colour box, so the other builders
get the layer name alone.
"""

from pathlib import Path

import yaml
from docutils import nodes
from sphinx.util import logging

import qpdk

logger = logging.getLogger(__name__)

ROLE_NAME = "layer"

#: ``fill_color`` when a view sets only that, else ``color``, as in `_layers.typ`.
LAYERS_YAML = Path(qpdk.__file__).parent / "layers.yaml"


def _flatten(views: dict[str, dict]) -> dict[str, dict]:
    """Map every named layer view, descending into ``group_members``."""
    flat = {}
    for name, view in views.items():
        if not isinstance(view, dict):
            continue
        flat[name] = view
        flat.update(_flatten(view.get("group_members", {})))
    return flat


_LAYER_VIEWS = _flatten(
    yaml.safe_load(LAYERS_YAML.read_text(encoding="utf-8"))["LayerViews"]
)


def _swatch(label: str) -> nodes.raw | None:
    """Return the HTML swatch for ``label``, or ``None`` when it has no view."""
    # The metal faces are named without their mask suffix in the rule table.
    view = _LAYER_VIEWS.get(label) or _LAYER_VIEWS.get(f"{label}_DRAW")
    if view is None:
        logger.warning("no layer view for %r in %s", label, LAYERS_YAML.name)
        return None
    if color := view.get("fill_color") or view.get("color"):
        return nodes.raw(
            "",
            f'<span class="qpdk-layer-swatch" style="background:{color}"></span>',
            format="html",
        )
    logger.warning("layer view %r in %s sets no colour", label, LAYERS_YAML.name)
    return None


def _layer_role(
    _name: str,
    _rawtext: str,
    text: str,
    _lineno: int,
    inliner,
    _options: dict | None = None,
    _content: list[str] | None = None,
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """Render ``{layer}`name``` as the layer name plus an HTML swatch."""
    label = text.strip()
    rendered: list[nodes.Node] = [nodes.literal(label, label)]
    builder = inliner.document.settings.env.app.builder
    if builder.format == "html" and (swatch := _swatch(label)):
        rendered.insert(0, swatch)
    return rendered, []


def setup(app) -> dict[str, object]:
    """Register the ``layer`` role."""
    app.add_role(ROLE_NAME, _layer_role)
    return {"version": "0.1", "parallel_read_safe": True}
