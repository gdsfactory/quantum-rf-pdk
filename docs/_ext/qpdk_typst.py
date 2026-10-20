"""Adapt doctrees and the typsphinx translator for the Typst PDF build.

The doctree transforms drop or rewrite content that the HTML build hides or
renders differently (hidden cell parts, widget outputs, ANSI colours, ...), and
the translator overrides add Typst output for nodes typsphinx does not handle,
or handles badly.
"""

import json
import re
from pathlib import Path

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from sphinx_design.shared import PassthroughTextElement
from sphinxcontrib.mermaid import mermaid
from typsphinx.translator import TypstTranslator, escape_typst_string

_TYPST_VISIT_MATH_BLOCK = TypstTranslator.visit_math_block
_TYPST_VISIT_BULLET_LIST = TypstTranslator.visit_bullet_list
_TYPST_DEPART_BULLET_LIST = TypstTranslator.depart_bullet_list
_TYPST_VISIT_ADMONITION = TypstTranslator._visit_admonition
logger = logging.getLogger(__name__)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _typst_strip_ansi(app, doctree, _docname):
    """Strip ANSI colour escapes from notebook stream output.

    Cells that log through ``qpdk.logger`` emit coloured stderr.  myst-nb turns
    those escapes into styled spans for HTML, but the Typst builder has no such
    handling and the raw ``ESC[36m`` bytes end up printed in the PDF.
    """
    if app.builder.name not in {"typst", "typstpdf"}:
        return
    for node in list(doctree.findall(nodes.Text)):
        stripped = _ANSI_RE.sub("", node.astext())
        if stripped != node.astext():
            node.parent.replace(node, nodes.Text(stripped))


def _typst_drop_hidden_cell_parts(app, doctree, _docname):
    """Drop the notebook cell parts that HTML collapses behind a toggle.

    ``hide-input`` / ``hide-output`` cells (Colab install blocks, imports,
    helper code) render collapsed in HTML via myst-nb's ``HideInputCells``
    transform, which only runs for HTML.  A PDF cannot expand a toggle, so
    without this the Typst builders print every hidden cell in full.  Leaving
    the hidden parts out is the PDF equivalent of the collapsed default.
    """
    if app.builder.name not in {"typst", "typstpdf"}:
        return
    for cell in list(doctree.findall(nodes.container)):
        hide_mode = cell.get("hide_mode")
        if cell.get("nb_element") != "cell_code" or not hide_mode:
            continue
        hidden = {
            "all": {"cell_code_source", "cell_code_output"},
            "input+output": {"cell_code_source", "cell_code_output"},
            "input": {"cell_code_source"},
            "output": {"cell_code_output"},
        }.get(hide_mode, set())
        for child in list(cell.children):
            if child.get("nb_element") not in hidden:
                continue
            # Keep parts that carry a reference target (e.g. a named figure):
            # references to it are already resolved, and Typst fails to
            # compile a link to a missing label.
            if any(n.get("ids") for n in child.findall(nodes.Element)):
                continue
            cell.remove(child)
        if not cell.children:
            cell.parent.remove(cell)


def _typst_unlabel_cell_outputs(app, doctree, _docname):
    """Drop the ``myst-ansi`` language from notebook output blocks.

    myst-nb marks stream and ``text/plain`` outputs with its ``myst-ansi``
    lexer.  codly prints the language as a badge on every block, so each cell
    output in the PDF was labelled "myst-ansi"; HTML shows them unlabelled.
    """
    if app.builder.name not in {"typst", "typstpdf"}:
        return
    for block in doctree.findall(nodes.literal_block):
        if block.get("language") == "myst-ansi":
            block["language"] = ""


_WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


class _TypstDropWidgetOutputs(SphinxPostTransform):
    """Drop Jupyter widget outputs from the PDF.

    tqdm progress bars (``tqdm.auto``) are widgets: HTML renders them from the
    saved widget state (``100%``), but the Typst builders cannot, so myst-nb
    falls back to the bundle's ``text/plain`` -- the bar's *initial* repr,
    ``0%| | 0/200 [00:00<?, ?it/s]``.  Runs just before myst-nb's
    ``SelectMimeType`` (priority 4), while the bundle still lists every mime
    type.
    """

    default_priority = 3
    builders = ("typst", "typstpdf")

    def run(self, **_kwargs):
        """Remove widget bundles that would fall back to their initial repr."""
        for bundle in list(self.document.findall(nodes.container)):
            if bundle.get("nb_element") != "mime_bundle":
                continue
            mime_types = {child.get("mime_type") for child in bundle.children}
            # Only drop bundles whose sole fallback is the stale text repr; a
            # widget with an HTML or image fallback still has real output.
            if mime_types == {_WIDGET_MIME, "text/plain"}:
                output = bundle.parent
                output.remove(bundle)
                if not output.children and output.get("nb_element") == (
                    "cell_code_output"
                ):
                    output.parent.remove(output)


def _typst_drop_unresolved_myst_xrefs(app, doctree, _docname):
    """Render MyST cross-references that did not resolve as plain text.

    A few relative links in ``README.md`` (``docs/contributing.md``,
    ``LICENSE``) point at repository paths that are not documents, so MyST
    leaves them unresolved and marks them with the ``xref myst`` class (the
    ``myst.xref_missing`` warning suppressed above).  HTML already emits these
    as dead anchors (``href="#LICENSE"``), but typsphinx turns them into Typst
    labels, and an unresolvable ``link(<...>)`` aborts the whole PDF compile
    rather than degrading.  Unwrapping them to their own text keeps the wording
    and matches what the HTML link already does -- nothing.
    """
    if app.builder.name not in {"typst", "typstpdf"}:
        return
    for node in list(doctree.findall(nodes.reference)):
        if node.get("refid") and not node.get("refuri"):
            node.replace_self(node.children)


def _typst_lift_block_images(app, doctree, _docname):
    """Lift images and their links out of paragraphs.

    MyST renders standalone ``![]()`` images as a paragraph around the image,
    and typsphinx wraps the paragraph in ``par({...})``, so the ``image()``
    lands inside a paragraph and Typst silently drops it with "block may not
    occur inside of a paragraph".  Lifting the image to the paragraph's place
    emits it bare, the way the figure and plot directives already do.
    """
    if app.builder.name not in {"typst", "typstpdf"}:
        return
    for par in list(doctree.findall(nodes.paragraph)):
        replacement = []
        inline = []
        for child in par.children:
            is_image = isinstance(child, nodes.image) or (
                isinstance(child, nodes.reference)
                and len(child.children) == 1
                and isinstance(child.children[0], nodes.image)
            )
            if is_image:
                if inline:
                    replacement.append(nodes.paragraph("", *inline))
                    inline = []
                replacement.append(child)
            else:
                inline.append(child)
        if replacement:
            if inline:
                replacement.append(nodes.paragraph("", *inline))
            par.replace_self(replacement)


# Typst package that lays out and draws Mermaid diagrams inside the compiler.
_MERMAN = "@preview/merman:0.5.0"
# Colours ``themeCSS`` in conf.py reads, as ``--qpdk-mermaid-<name>``.
_MERMAID_COLOURS = (
    "node-fill",
    "node-stroke",
    "text",
    "line",
    "bg",
    "cluster-fill",
    "cluster-stroke",
)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_BLOCK_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_CSS_PROPERTY_RE = re.compile(r"--([\w-]+)\s*:\s*([^;]+);")
_CSS_VAR_RE = re.compile(r"var\(--([\w-]+)\)")


def _css_light_properties(path, names):
    """Return the values custom properties ``names`` take in the light theme.

    Only ``:root`` and ``html[data-theme="light"]`` rules of ``path`` are read,
    in file order, and ``var()`` references between them are resolved.  An
    undefined property, or one it references, raises ``ExtensionError``.

    Returns:
        A dict from each of ``names``, without the leading ``--``, to its value.
    """
    # pydata-sphinx-theme defines this one, not custom.css; the PDF page is
    # white (docs/typst/qpdk.typ).
    found = {"pst-color-background": "#ffffff"}
    css = _CSS_COMMENT_RE.sub("", path.read_text(encoding="utf-8"))
    for prelude, body in _CSS_BLOCK_RE.findall(css):
        # Drop any preceding @import statements caught up in the selector.
        selector = prelude.rsplit(";", 1)[-1].strip()
        if selector in {":root", 'html[data-theme="light"]'}:
            found.update(_CSS_PROPERTY_RE.findall(body))

    def resolve(name, seen=()):
        if name in seen or name not in found:
            raise ExtensionError(f"cannot resolve --{name} from {path}")
        return _CSS_VAR_RE.sub(
            lambda match: resolve(match.group(1), (*seen, name)),
            found[name].strip(),
        )

    return {name: resolve(name) for name in names}


def _mermaid_print_config(config, srcdir):
    """Return the mermaid config for print, derived from ``mermaid_init_config``.

    ``themeCSS`` reads the light-theme ``--qpdk-mermaid-*`` colours from
    custom.css, which a diagram drawn into the PDF has no page to inherit, so
    their values are substituted in.

    Raises:
        ExtensionError: If ``themeCSS`` still reads a custom property after
            the substitution.
    """
    init = json.loads(json.dumps(config.mermaid_init_config))
    init.pop("startOnLoad", None)
    init["theme"] = config.mermaid_light_theme
    # Plain SVG <text> labels; merman would otherwise have to strip the HTML
    # labels mermaid puts in <foreignObject>, which Typst cannot draw.
    init["htmlLabels"] = False
    colours = _css_light_properties(
        Path(srcdir) / "_static" / "css" / "custom.css",
        [f"qpdk-mermaid-{name}" for name in _MERMAID_COLOURS],
    )
    css = init.get("themeCSS", "")
    for name, colour in colours.items():
        css = css.replace(f"var(--{name})", colour)
    if "var(" in css:
        raise ExtensionError(
            "mermaid themeCSS uses a custom property with no print colour; "
            "add it to _MERMAID_COLOURS in docs/_ext/qpdk_typst.py"
        )
    init["themeCSS"] = css
    return init


def _typst_string(text):
    """Escape ``text`` for use inside a Typst double-quoted string literal.

    Returns:
        The escaped string, without the surrounding quotes.
    """
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )


_TABULAR_RE = re.compile(
    r"\\begin\{(?P<environment>tabular|longtable)\}\{(?P<columns>[^}]*)\}"
    r"(?P<body>.*?)\\end\{(?P=environment)\}",
    re.DOTALL,
)
_TABLE_WRAPPER_RE = re.compile(
    r"\\begin\{table\}(?:\[[^]]*])?(?P<content>.*?)\\end\{table\}", re.DOTALL
)
_TABULAR_START_RE = re.compile(r"\\begin\{(?:table|tabular|longtable)\}")
_TABLE_CAPTION_RE = re.compile(
    r"\\caption(?:\[[^]]*])?\{(?P<caption>(?:[^{}]|\{[^{}]*})*)\}", re.DOTALL
)
_LATEX_TEXT_ESCAPE_RE = re.compile(r"\\([&%_#])")


def _typst_cell(cell):
    """Convert one LaTeX table cell to a Typst content expression.

    ``$...$`` spans are handed to mitex; the rest becomes a plain string.

    Returns:
        A Typst expression for the cell content.
    """
    parts = []
    cell = _LATEX_TEXT_ESCAPE_RE.sub(r"\1", cell)
    for index, chunk in enumerate(re.split(r"\$([^$]*)\$", cell.strip())):
        if index % 2:
            parts.append(f"mi(`{chunk}`)")
        elif chunk:
            parts.append(f'text("{_typst_string(chunk)}")')
    return " + ".join(parts) if parts else '""'


def _tabular_to_typst(latex):
    """Convert a LaTeX ``tabular`` environment into a Typst ``table()`` call.

    ``qpdk.helper.display_dataframe`` gives its tables a ``_repr_latex_``
    (``DataFrame.to_latex()``, i.e. booktabs ``tabular``) so the LaTeX PDF
    rendered a real table.  mitex is a *math* translator and aborts the whole
    compile on ``tabular``, so the table is rebuilt as a native Typst one here
    rather than being dropped -- the alternative mime types these outputs
    carry are ``text/html`` and an unusable object ``repr``.

    Returns:
        The Typst ``table(...)`` source, or ``None`` if ``latex`` is not a
        single ``tabular`` or ``longtable`` environment, optionally wrapped
        in a LaTex ``table`` environment.
    """
    latex = latex.strip()
    caption = None
    if wrapper := _TABLE_WRAPPER_RE.fullmatch(latex):
        latex = wrapper["content"].strip()
        if caption_match := _TABLE_CAPTION_RE.search(latex):
            caption = caption_match["caption"]
            latex = _TABLE_CAPTION_RE.sub("", latex).strip()
        latex = re.sub(r"\\label\{[^}]*}", "", latex).strip()

    match = _TABULAR_RE.fullmatch(latex)
    if match is None:
        return None
    # Every spec `DataFrame.to_latex()` emits here is plain `[lr]+` (`llll`,
    # `lrrrr`), so counting alignment letters is enough; a spec with widths
    # (`p{3cm}`) or `@{}` padding would need real parsing.
    columns = sum(match["columns"].count(spec) for spec in "lcr")
    if any(token in match["body"] for token in (r"\multicolumn", r"\multirow")):
        return None
    rows = []
    for raw_row in match["body"].split(r"\\"):
        row = re.sub(r"\\(top|mid|bottom)rule", "", raw_row).strip()
        if row:
            cells = re.split(r"(?<!\\)&", row)
            if len(cells) != columns:
                return None
            rows.append([_typst_cell(cell) for cell in cells])
    if not rows or columns == 0:
        return None
    header, *body = rows
    lines = [
        f"table(\n  columns: {columns},",
        "  table.header(" + ", ".join(header) + "),",
    ]
    lines.extend("  " + ", ".join(row) + "," for row in body)
    lines.append(")")
    table = "\n".join(lines)
    if caption is None:
        return table
    return f"figure(\n  {table},\n  caption: [{_typst_cell(caption)}],\n)"


def _typst_add_block_separator(self):
    """Prepare a block expression following prose or another list-item block."""
    self._add_paragraph_separator()
    if self.in_list_item and self.list_item_needs_separator:
        self.add_text("\n")


def _typst_visit_math_block(self, node):
    """Render a ``tabular`` math block as a Typst table, else defer to typsphinx.

    Returns:
        Whatever typsphinx's own visitor returns, for ordinary math blocks.

    Raises:
        nodes.SkipNode: When the block was a table and is fully emitted here.
    """
    table = _tabular_to_typst(node.astext())
    if table is None:
        if _TABULAR_START_RE.search(node.astext()):
            logger.warning(
                "Rendering unsupported LaTeX table as a code block in the Typst output"
            )
            _typst_add_block_separator(self)
            self.add_text(
                f'raw("{_typst_string(node.astext())}", lang: "latex", block: true)\n'
            )
            if self.in_list_item:
                self.list_item_needs_separator = True
            raise nodes.SkipNode
        return _TYPST_VISIT_MATH_BLOCK(self, node)
    _typst_add_block_separator(self)
    self.add_text(table + "\n")
    if self.in_list_item:
        self.list_item_needs_separator = True
    raise nodes.SkipNode


def _typst_visit_bullet_list(self, node):
    """Open a list with the surrounding inline concat context suppressed.

    A list inside a field body (a Napoleon ``Returns:`` block whose text is
    followed by bullets) is emitted as ``list({...}, {...})`` while typsphinx
    still considers the field body an active ``+``-concatenation context.
    Treat the list itself as one concat element so both preceding and following
    prose are separated from the block expression.
    """
    self._enter_inline_concat_element()
    _TYPST_VISIT_BULLET_LIST(self, node)


def _typst_depart_bullet_list(self, node):
    """Restore the concat context around a completed list."""
    _TYPST_DEPART_BULLET_LIST(self, node)
    self._exit_inline_concat_element()


# HTML light-mode admonition colors, per gentle-clues function name.
# pydata-sphinx-theme maps note to its info token and custom.css overrides both
# `--pst-color-primary` (border) and `--pst-color-primary-bg` (title
# background) for `.admonition.note`; the rest are the theme's stock tokens.
# gentle-clues' own accent palette is catppuccin, which matches none of these.
_TYPST_CLUE_COLORS = {
    "info": ("#2a6fb5", "#e8eff8"),  # note: site accent, --pst-color-primary-bg
    "tip": ("#00843f", "#d6ece1"),  # hint/tip/seealso: --pst-color-success
    "warning": ("#f66a0a", "#f8e3d0"),  # warning/caution/important/attention
    "error": ("#d72d47", "#f9e1e4"),  # --pst-color-danger
    "danger": ("#d72d47", "#f9e1e4"),
    "task": ("#1f5994", "#e0c7ff"),  # todo: --pst-color-secondary
    "memo": ("#f66a0a", "#f8e3d0"),  # attention alias of warning
    "notify": ("#276be9", "#dce7fc"),  # generic admonition: base .admonition rule
    "abstract": ("#276be9", "#dce7fc"),  # topic: base .admonition rule
}


def _typst_visit_admonition(self, node, clue_type, custom_title=None):
    """Delegate to typsphinx's helper, remembering the clue type for depart."""
    self.__dict__.setdefault("_qpdk_clue_types", []).append(clue_type)
    _TYPST_VISIT_ADMONITION(self, node, clue_type, custom_title)


def _typst_depart_admonition(self):
    """Close the clue call with the site's accent and title background.

    Reimplements typsphinx's ``_depart_admonition`` to append
    ``accent-color``/``header-color`` per the mapping above.  gentle-clues has
    no global accent configuration and the generated per-document files bind
    its names through a wildcard import, so per-call keyword arguments are the
    only styleable hook.  ``header-color`` is passed explicitly because
    gentle-clues' default (the accent lightened 85%) does not land exactly on
    the theme's ``-bg`` tokens.

    The geometry matches pydata-sphinx-theme's ``.admonition`` rule: a
    ``border-left: .2rem`` accent bar (~2.4pt) with no outline on the other
    sides, and a ``border-radius: .25rem`` (~3pt).
    """
    self.add_text("}")

    title_expr = None
    if self._pending_admonition_title:
        title_expr = "{" + self._pending_admonition_title + "}"
    elif self._custom_admonition_title:
        title_expr = f'"{escape_typst_string(str(self._custom_admonition_title))}"'
    if title_expr:
        self.add_text(f", title: {title_expr}")

    colors = _TYPST_CLUE_COLORS.get(self._qpdk_clue_types.pop())
    if colors:
        accent, header = colors
        self.add_text(
            f', accent-color: rgb("{accent}"), header-color: rgb("{header}")'
            ", stroke-width: 2.4pt, border-width: 0pt, radius: 3pt"
        )

    self.add_text(")\n\n")

    if self.in_list_item:
        self.list_item_needs_separator = True


def _typst_visit_passthrough(self, node):
    """Render a sphinx-design ``PassthroughTextElement`` in the Typst output.

    sphinx-design registers a no-op visitor for this node on every builder it
    knows about (``visit_depart_null`` in ``sphinx_design/extension.py``), so
    the wrapper is invisible and its inline children render inline.  typsphinx
    emits into Typst *code* mode, where two adjacent expressions on one line
    are a syntax error (``expected semicolon or line break``), so a no-op here
    would emit ``text("title")par({...})`` and abort the whole compile.
    Wrapping the children in ``par({...})`` both supplies that separator and
    gives the card title its own block, which is how it reads in HTML.

    A card's ``:link:`` also arrives as one of these, holding a
    ``sd-hide-link-text`` reference: an invisible overlay that makes the whole
    card clickable in HTML, whose target is a bare URL fragment.  It has no
    Typst label to point at, and an unresolvable ``link(<...>)`` aborts the
    entire PDF compile, so it is dropped -- print has no stretched-link
    affordance for it to be, and HTML renders no visible text for it either.

    Raises:
        nodes.SkipNode: For a card's hidden stretched link, which is dropped.
    """
    if any(
        "sd-hide-link-text" in child.get("classes", [])
        for child in node.findall(nodes.reference)
    ):
        raise nodes.SkipNode
    _typst_add_block_separator(self)
    self.add_text("par({")
    state = self.__dict__.setdefault("_qpdk_paragraph_state", [])
    state.append((self.in_paragraph, self.paragraph_has_content))
    self.in_paragraph = True
    self.paragraph_has_content = False


def _typst_depart_passthrough(self, _node):
    """Close the ``par({`` opened by :func:`_typst_visit_passthrough`."""
    self.add_text("})\n\n")
    self.in_paragraph, self.paragraph_has_content = self._qpdk_paragraph_state.pop()


def _typst_visit_doctest_block(self, node):
    """Render a docutils ``doctest_block`` as a Typst code block.

    ``>>>`` examples in docstrings parse into ``doctest_block``, which
    typsphinx does not handle; without this the interpreter prompts would be
    dropped from the PDF with only a warning.

    Raises:
        nodes.SkipNode: Always, the block is emitted here in full.
    """
    _typst_add_block_separator(self)
    self.add_text(
        f'raw("{_typst_string(node.astext())}", lang: "pycon", block: true)\n'
    )
    if self.in_list_item:
        self.list_item_needs_separator = True
    raise nodes.SkipNode


def _typst_visit_mermaid(self, node):
    """Draw a sphinxcontrib-mermaid diagram with the merman Typst package.

    typsphinx has no visitor for this node and drops it with only an
    ``unknown node type`` warning.  HTML renders diagrams in the browser; for
    print, merman lays them out and draws them inside the Typst compiler, in
    the light palette, with real (selectable) text.  A diagram wider than the
    text column is shrunk to fit; narrower ones keep their natural size.  A
    diagram merman cannot parse fails the build rather than vanishing.

    Raises:
        nodes.SkipNode: Always, the diagram is emitted here in full.
    """
    config = json.dumps(
        _mermaid_print_config(self.config, self.builder.srcdir), sort_keys=True
    )
    args = [
        f'"{_typst_string(node["code"])}"',
        f'site-config: json(bytes("{_typst_string(config)}"))',
    ]
    if node.get("alt"):
        args.append(f'alt: "{_typst_string(node["alt"])}"')
    align = node.get("align") if node.get("align") in {"left", "right"} else "center"
    _typst_add_block_separator(self)
    self.add_text(
        f'{{\n  import "{_MERMAN}": mermaid\n'
        f"  let args = arguments({', '.join(args)})\n"
        f"  align({align}, layout(size => {{\n"
        "    let diagram = mermaid(..args)\n"
        "    if measure(diagram).width > size.width {\n"
        "      mermaid(..args, width: size.width)\n"
        "    } else { diagram }\n"
        "  }))\n}\n"
    )
    if self.in_list_item:
        self.list_item_needs_separator = True
    raise nodes.SkipNode


def _typst_visit_meta(_self, _node):
    """Drop docutils ``meta`` nodes, which carry HTML ``<meta>`` tags only.

    Raises:
        nodes.SkipNode: Always, the node has no print representation.
    """
    raise nodes.SkipNode


def setup(app: Sphinx) -> None:
    """Register the extension."""
    app.add_post_transform(_TypstDropWidgetOutputs)
    app.connect("doctree-resolved", _typst_drop_hidden_cell_parts)
    app.connect("doctree-resolved", _typst_unlabel_cell_outputs)
    app.connect("doctree-resolved", _typst_drop_unresolved_myst_xrefs)
    app.connect("doctree-resolved", _typst_strip_ansi)
    app.connect("doctree-resolved", _typst_lift_block_images)

    # NOTE: everything below reaches into typsphinx internals (the translator's
    # visitor methods, `_inline_concat_context`, `in_paragraph`), which is why
    # `pyproject.toml` pins `typsphinx<0.10`.  Re-check these when unpinning.
    #
    # Typst handlers for nodes typsphinx does not know about.  Without them
    # the nodes are dropped with a warning -- and, for PassthroughTextElement,
    # emit invalid Typst that aborts the whole compile (see its docstring).
    #
    # These are attached to the translator class directly rather than through
    # `app.add_node(..., typst=...)`: typsphinx constructs its translator with
    # a bare `TypstTranslator(document, builder)` (typsphinx/writer.py) instead
    # of `Sphinx.registry.create_translator()`, so nothing registered through
    # `add_node` ever reaches it.  docutils dispatches on the node class name,
    # so a `visit_<classname>` attribute on the class is picked up as-is.
    for node_class, visit, depart in (
        (PassthroughTextElement, _typst_visit_passthrough, _typst_depart_passthrough),
        (nodes.doctest_block, _typst_visit_doctest_block, None),
        (nodes.meta, _typst_visit_meta, None),
        (mermaid, _typst_visit_mermaid, None),
    ):
        name = node_class.__name__
        if hasattr(TypstTranslator, f"visit_{name}"):
            raise RuntimeError(
                f"typsphinx now implements visit_{name}; remove the QPDK override"
            )
        setattr(TypstTranslator, f"visit_{name}", visit)
        if depart is not None:
            if hasattr(TypstTranslator, f"depart_{name}"):
                raise RuntimeError(
                    f"typsphinx now implements depart_{name}; remove the QPDK override"
                )
            setattr(TypstTranslator, f"depart_{name}", depart)
    TypstTranslator.visit_math_block = _typst_visit_math_block
    TypstTranslator.visit_bullet_list = _typst_visit_bullet_list
    TypstTranslator.depart_bullet_list = _typst_depart_bullet_list
    TypstTranslator._visit_admonition = _typst_visit_admonition
    TypstTranslator._depart_admonition = _typst_depart_admonition
