"""Sphinx extension to convert GitHub-flavored Markdown admonitions to Sphinx admonitions.

GitHub renders blockquotes starting with ``[!NOTE]``, ``[!WARNING]``, etc. as
styled callouts.  MyST-parser does not recognise this syntax and passes them
through as plain blockquotes.  This extension post-processes the doctree and
promotes matching blockquotes to proper Sphinx admonition nodes so that the
same Markdown source renders correctly on both GitHub **and** Sphinx.
"""

from __future__ import annotations

import re
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform

_GFM_ALERT_RE = re.compile(
    r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$",
    re.IGNORECASE,
)

_GFM_TO_ADMONITION = {
    "NOTE": "note",
    "TIP": "tip",
    "IMPORTANT": "important",
    "WARNING": "warning",
    "CAUTION": "caution",
}


class GFMAdmonitionTransform(SphinxPostTransform):
    """Replace GFM-style alert blockquotes with Sphinx admonitions."""

    default_priority = 200

    def run(self, **kwargs: Any) -> None:
        for bq in self.document.findall(nodes.block_quote):
            admonition = _try_convert(bq)
            if admonition is not None:
                bq.replace_self(admonition)


def _try_convert(bq: nodes.block_quote) -> nodes.Admonition | None:
    """Return an admonition node if *bq* matches a GFM alert, else ``None``."""
    if not bq.children:
        return None

    first = bq.children[0]
    if not isinstance(first, nodes.paragraph):
        return None

    # The first paragraph should start with text like "[!NOTE]"
    first_text = first.astext().split("\n", 1)[0]
    m = _GFM_ALERT_RE.match(first_text)
    if m is None:
        return None

    kind = _GFM_TO_ADMONITION[m.group(1).upper()]
    admonition_cls = {
        "note": nodes.note,
        "tip": nodes.tip,
        "important": nodes.important,
        "warning": nodes.warning,
        "caution": nodes.caution,
    }[kind]

    admonition = admonition_cls()

    # Remove the "[!NOTE]" marker from the first paragraph.
    # The marker may be the only content, or there may be more text/nodes after it.
    remaining_children = _strip_marker(first, m.group(0))
    if remaining_children:
        new_para = nodes.paragraph("", *remaining_children)
        admonition += new_para

    # Append the rest of the blockquote children as-is.
    for child in bq.children[1:]:
        admonition += child.deepcopy()

    return admonition


def _strip_marker(para: nodes.paragraph, marker: str) -> list[nodes.Node]:
    """Remove the GFM marker text from the beginning of a paragraph.

    Returns the remaining child nodes (may be empty).
    """
    result: list[nodes.Node] = []
    found = False
    for child in para.children:
        if not found:
            if isinstance(child, nodes.Text):
                text = child.astext()
                if text.startswith(marker):
                    rest = text[len(marker) :].lstrip("\n").lstrip()
                    if rest:
                        result.append(nodes.Text(rest))
                    found = True
                    continue
            # If the first node isn't the marker text, bail out
            return list(para.children)
        result.append(child.deepcopy())
    return result


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_post_transform(GFMAdmonitionTransform)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
