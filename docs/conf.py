"""Sphinx configuration for Qpdk documentation."""

import re
from pathlib import Path

from docutils import nodes
from sphinx_design.shared import PassthroughTextElement
from typsphinx.translator import TypstTranslator

_TYPST_VISIT_MATH_BLOCK = TypstTranslator.visit_math_block
_TYPST_VISIT_LIST_ITEM = TypstTranslator.visit_list_item
_TYPST_DEPART_LIST_ITEM = TypstTranslator.depart_list_item

project = "qpdk"
author = "gdsfactory"
copyright = "gdsfactory"  # ruff: ignore[builtin-variable-shadowing]

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.svgbob",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_github_alerts",
    "sphinxcontrib.bibtex",
    "sphinxcontrib_bibtex_urn",
    "typsphinx",
]

# -- Plot directive configuration ---------------------------------------------
plot_pre_code = """
from matplotlib import pyplot as plt
from qpdk import PDK
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)

plt.style.use("qpdk")
PDK.activate()

# Monkey-patch Axes.set_title to use Outfit (bold) for figure titles,
# matching the Sphinx heading font (see docs/_static/css/custom.css).
import matplotlib.axes as _ma
_orig_title = _ma.Axes.set_title
def _qpdk_title(self, *args, **kwargs):
    kwargs.setdefault('fontfamily', 'Outfit')
    kwargs.setdefault('fontweight', 'bold')
    return _orig_title(self, *args, **kwargs)
_ma.Axes.set_title = _qpdk_title
del _qpdk_title, _orig_title
"""
plot_rcparams = {
    "svg.fonttype": "path",
    "pdf.compression": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plot_formats = ["svg", "pdf", "png"]
plot_apply_rcparams = True  # Ensure rcParams are applied even with :context:

exclude_patterns = [
    "_build",
    "_extra",
    "conf.py",
    "ipython_config.py",
    "*.mplstyle",
    "justfile_help.txt",
    "changelog.md",
    "Thumbs.db",
    ".DS_Store",
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "gdsfactory": ("https://gdsfactory.github.io/gdsfactory/", None),
    "sax": ("https://gdsfactory.github.io/sax/", None),
    "pyaedt": ("https://aedt.docs.pyansys.com/version/stable/", None),
}

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "substitution",
    "tasklist",
    "linkify",
]

# -- MathJax configuration ---------------------------------------------------
# NOTE: sphinx.ext.mathjax injects `window.MathJax = {options: {processHtmlClass: ...}}`
# as a separate <script> tag BEFORE our mathjax4_config assignment.  The second
# assignment overwrites the first, so we must include the Sphinx-required
# `processHtmlClass` regex here too; otherwise MathJax ignores all math that
# lives inside `<section class="mathjax_ignore">` elements (which is every
# section in the Sphinx HTML output when MyST is used).
mathjax4_config = {
    "options": {
        # Allow MathJax to process math nodes even inside mathjax_ignore regions.
        "processHtmlClass": "tex2jax_process|mathjax_process|math|output_area",
    },
    "output": {
        "font": "mathjax-fira",
    },
}

# -- Notebook execution (myst-nb) --------------------------------------------
nb_execution_mode = "cache"
# Exclude HFSS notebooks from execution as they depend on Ansys HFSS
# (proprietary/licensed software) and can be slow or impossible to run
# in typical documentation build environments.
nb_execution_excludepatterns = [
    "notebooks/hfss*",
    "notebooks/matlab_integration*",
]
nb_execution_timeout = -1
nb_execution_allow_errors = False
nb_execution_show_tb = True
nb_execution_raise_on_error = True
nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py"}],
}

# myst-nb ships a mime-priority table per builder and has no entry for the
# typsphinx builders, so every notebook output cell would be dropped from the
# PDF with a "No mime type available in priority list" warning.  This mirrors
# myst-nb's own ``latex`` priorities, with two changes for Typst: SVG is
# preferred over PNG (Typst embeds it natively, so plots stay vector), and
# ``application/pdf`` is left out entirely (Typst's ``image()`` cannot embed
# a PDF -- see TypstBuilder.supported_image_types).
nb_mime_priority_overrides = [
    (builder, mime, priority)
    for builder in ("typst", "typstpdf")
    for mime, priority in (
        ("image/svg+xml", 10),
        ("image/png", 20),
        ("image/jpeg", 30),
        ("text/latex", 40),
        ("text/markdown", 50),
        ("text/plain", 60),
    )
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_type_aliases = {
    "ComponentSpec": "ComponentSpec",
    "CrossSectionSpec": "CrossSectionSpec",
    "LayerSpec": "LayerSpec",
    "SDict": "sax.SDict",
    "sax.SDict": "sax.SDict",
    "sax.FloatArrayLike": "FloatArrayLike",
    "jax.typing.ArrayLike": "ArrayLike",
    "ArrayLike": "ArrayLike",
    "gt.CrossSectionSpec": "CrossSectionSpec",
    "gt.LayerSpec": "LayerSpec",
    "gt.ComponentSpec": "ComponentSpec",
    "gt.ComponentAllAngleSpec": "ComponentAllAngleSpec",
    "gt.Port": "Port",
    "gt.Ports": "Ports",
    "gt.Size": "Size",
    "gt.Ints": "Ints",
    "gt.Coordinate": "Coordinate",
    "gt.Coordinates": "Coordinates",
    "gt.Layer": "Layer",
    "FloatArrayLike": "FloatArrayLike",
    "sax.Float": "float",
    "Float": "float",
    "sax.SType": "SType",
    "SType": "SType",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

napoleon_preprocess_types = True
napoleon_type_aliases = autodoc_type_aliases

# -- Bibliography (sphinxcontrib-bibtex) --------------------------------------
bibtex_bibfiles = ["bibliography.bib"]

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_favicon = "_static/favicon.svg"
html_show_copyright = False
templates_path = ["templates"]
html_theme_options = {
    "logo": {
        "image_light": "_static/qpdk_mark.svg",
        "image_dark": "_static/qpdk_mark_dark.svg",
        "text": "QPDK",
    },
    "use_edit_page_button": True,
    "header_links_before_dropdown": 4,
    "secondary_sidebar_items": [
        "page-toc",
        "edit-this-page",
        "sourcelink",
        "colab-button.html.j2",
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gdsfactory/quantum-rf-pdk",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PDF",
            "url": "https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf",
            "icon": "fa-solid fa-file-pdf",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/qpdk/",
            "icon": "fa-solid fa-box",
        },
    ],
    "pygments_light_style": "tango",
    "pygments_dark_style": "dracula",
}
html_context = {
    "github_user": "gdsfactory",
    "github_repo": "quantum-rf-pdk",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path = ["_static"]
# Contents are copied verbatim to the output root, so ``docs/_extra/kwasm/``
# becomes ``kwasm/`` on the built site.  This is what serves the interactive
# viewer and the GDS files behind the "Dynamic" tab of each PCell; Sphinx does
# not copy them on its own since they are only referenced from ``.. raw:: html``.
html_extra_path = ["_extra"]
html_css_files = [
    "css/custom.css",
]

# -- Typst / PDF output (typsphinx) -------------------------------------------
# The fifth tuple element is a typsphinx *template registry key*, not a LaTeX
# documentclass.  ``"typst"`` is the built-in key.
typst_documents = [
    ("index", "qpdk.typ", "Qpdk", "gdsfactory", "typst"),
]
# Only ``papersize``, ``fontsize`` and ``lang`` are accepted here; everything
# else about the look of the PDF lives in the template.
typst_elements = {
    "papersize": "a4",
    "fontsize": "10pt",
}
typst_use_mitex = True
# Brand-matched template (Outfit/Inter/Code New Roman, #2a6fb5 accent), chosen
# over the stock typsphinx look and a classic serif report variant.  The whole
# containing directory is copied to the build as the template bundle, so keep
# `docs/typst/` to template files only.
typst_template = "typst/qpdk.typ"

# -- Warning suppression ------------------------------------------------------
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "bibtex.duplicate_citation",
]


def _repair_external_math(content):
    r"""Repair malformed LaTeX found in external docstring math (e.g. sax).

    - A doubled backslash before a command name (``\ln\\frac``): in math,
      ``\\`` is a row break and is always followed by whitespace or end of
      line, so ``\\<command>`` is always a typo.  It breaks the LaTeX build
      with ``Extra }, or forgotten \right.``.
    - A bare ``_`` inside ``\text{...}`` (``I_\text{n_ports}``): ``\text``
      typesets its argument in text mode, where an unescaped ``_`` fails the
      LaTeX build with ``Missing $ inserted``.
    - Consecutive bare subscripts on one token (``C_M_S``): invalid LaTeX
      ("Double subscript") unless grouped, so group them mechanically.

    Returns:
        The repaired math content.
    """
    content = re.sub(
        r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)", r"\1_{\2_\3}", content
    )
    content = re.sub(r"\\\\([a-zA-Z]+)", r"\\\1", content)
    return re.sub(
        r"\\text\{([^{}]*)\}",
        lambda match: "\\text{" + match.group(1).replace("_", r"\_") + "}",
        content,
    )


def _dollar_math_to_rst(lines):
    r"""Convert ``$…$`` and ``$$…$$`` math to RST ``:math:`` and ``.. math::`` directives.

    This is needed for docstrings from external libraries (e.g. sax) that use
    LaTeX dollar-sign conventions instead of RST math markup.
    """
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Detect display-math opening ``$$`` on its own line
        if stripped == "$$":
            indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
            result.extend((f"{indent}.. math::", ""))
            i += 1
            # Collect lines until closing ``$$``
            while i < len(lines) and lines[i].strip() != "$$":
                math_line = _repair_external_math(lines[i])
                # Ensure math content is indented under the directive
                if math_line.strip():
                    result.append(f"{indent}   {math_line.strip()}")
                else:
                    result.append("")
                i += 1
            result.append("")
            i += 1  # skip closing $$
            continue

        # Convert inline $…$ to :math:`…` (but not $$)
        converted = re.sub(
            r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
            lambda match: f":math:`{_repair_external_math(match.group(1))}`",
            lines[i],
        )
        result.append(converted)
        i += 1

    lines[:] = result


def replace_image_paths(app, docname, source):
    """Fix image paths and manually include README.md into index."""
    if docname == "index":
        readme = Path(app.srcdir).parent / "README.md"
        if readme.exists():
            content = readme.read_text(encoding="utf-8").split("_" * 70, 1)[-1]
            source[0] = re.sub(
                r"```\{include\}\s+\.\./README\.md.*?```",
                lambda _: content,
                source[0],
                flags=re.DOTALL,
            )

    source[0] = source[0].replace("docs/_static/images/", "/_static/images/")


def fix_notebook_edit_url(app, pagename, _templatename, context, _doctree):
    """Fix *Edit on GitHub* URLs for notebook pages.

    Notebooks are copied from ``notebooks/src/`` into ``docs/notebooks/`` during
    the documentation build, so Sphinx computes an edit URL pointing to
    ``docs/notebooks/<name>.py``.  The correct source lives at
    ``notebooks/src/<name>.py`` (or ``.m`` for the MATLAB notebook).

    This handler runs after the pydata-sphinx-theme ``setup_edit_url`` hook
    (priority > 500) and replaces ``get_edit_provider_and_url`` in the Jinja2
    context with a function that returns the right URL for notebook pages.
    """
    if not pagename.startswith("notebooks/"):
        return

    nb_name = pagename[len("notebooks/") :]
    src_root = Path(app.srcdir).parent / "notebooks" / "src"

    source_rel: str | None = None
    for ext in (".py", ".m"):
        if (src_root / f"{nb_name}{ext}").exists():
            source_rel = f"notebooks/src/{nb_name}{ext}"
            break

    if source_rel is None:
        return

    github_url = context.get("github_url", "https://github.com")
    github_user = context.get("github_user", "gdsfactory")
    github_repo = context.get("github_repo", "quantum-rf-pdk")
    github_version = context.get("github_version", "main")
    edit_url = (
        f"{github_url}/{github_user}/{github_repo}/edit/{github_version}/{source_rel}"
    )

    def _get_edit_provider_and_url():
        return "GitHub", edit_url

    context["get_edit_provider_and_url"] = _get_edit_provider_and_url


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
        if node.get("refuri"):
            continue
        if any(
            isinstance(child, nodes.Element)
            and {"xref", "myst"} <= set(child.get("classes", []))
            for child in node.children
        ):
            node.replace_self(node.children)


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
    r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", re.DOTALL
)


def _typst_cell(cell):
    """Convert one LaTeX table cell to a Typst content expression.

    ``$...$`` spans are handed to mitex; the rest becomes a plain string.

    Returns:
        A Typst expression for the cell content.
    """
    parts = []
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
        single ``tabular`` environment.
    """
    # fullmatch, not search: a block that merely *contains* a tabular also has
    # surrounding math, and the caller emits only what is returned here, so a
    # partial match would silently drop the rest.  Anything else falls through
    # to typsphinx's normal mitex path.
    match = _TABULAR_RE.fullmatch(latex.strip())
    if match is None:
        return None
    # Every spec `DataFrame.to_latex()` emits here is plain `[lr]+` (`llll`,
    # `lrrrr`), so counting alignment letters is enough; a spec with widths
    # (`p{3cm}`) or `@{}` padding would need real parsing.
    columns = sum(match.group(1).count(spec) for spec in "lcr")
    rows = []
    for raw_row in match.group(2).split(r"\\"):
        row = re.sub(r"\\(top|mid|bottom)rule", "", raw_row).strip()
        if row:
            rows.append([_typst_cell(cell) for cell in row.split("&")])
    if not rows or columns == 0:
        return None
    header, *body = rows
    lines = [
        f"table(\n  columns: {columns},",
        "  table.header(" + ", ".join(header) + "),",
    ]
    lines.extend("  " + ", ".join(row) + "," for row in body)
    lines.append(")")
    return "\n".join(lines)


def _typst_visit_math_block(self, node):
    """Render a ``tabular`` math block as a Typst table, else defer to typsphinx.

    Returns:
        Whatever typsphinx's own visitor returns, for ordinary math blocks.

    Raises:
        nodes.SkipNode: When the block was a table and is fully emitted here.
    """
    table = _tabular_to_typst(node.astext())
    if table is None:
        return _TYPST_VISIT_MATH_BLOCK(self, node)
    self.add_text(table + "\n")
    raise nodes.SkipNode


def _typst_visit_list_item(self, node):
    """Open a list item with the surrounding code-mode concat context suppressed.

    A list inside a field body (a Napoleon ``Returns:`` block whose text is
    followed by bullets) is emitted as ``list({...}, {...})`` while typsphinx
    still considers the field body an active ``+``-concatenation context, so
    every item after the first opens with a stray unary ``+`` and Typst fails
    with "cannot apply unary '+' to content".  Each item's ``{ }`` block is a
    fresh context, so suppress the outer one exactly as typsphinx's own
    ``_enter_inline_concat_element`` does for emphasis, strong and links.
    """
    _TYPST_VISIT_LIST_ITEM(self, node)
    context = self._inline_concat_context()
    self.__dict__.setdefault("_qpdk_list_item_concat", []).append(context)
    if context is not None:
        setattr(self, context[0], False)


def _typst_depart_list_item(self, node):
    """Restore the concat context suppressed by :func:`_typst_visit_list_item`."""
    context = self._qpdk_list_item_concat.pop()
    if context is not None:
        setattr(self, context[0], True)
    _TYPST_DEPART_LIST_ITEM(self, node)


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
    self.add_text(
        f'raw("{_typst_string(node.astext())}", lang: "pycon", block: true)\n'
    )
    raise nodes.SkipNode


def _typst_visit_meta(_self, _node):
    """Drop docutils ``meta`` nodes, which carry HTML ``<meta>`` tags only.

    Raises:
        nodes.SkipNode: Always, the node has no print representation.
    """
    raise nodes.SkipNode


def setup(app):
    """Sphinx setup."""
    # Regex for types to shorten in the final rendered docstring fields
    # Note: this is a bit hacky as it operates on the processed lines
    patterns = {
        r"Annotated\[Array \| ndarray \| .*?val_float_array.*?\]": "FloatArrayLike",
        r"Annotated\[float \| floating, PlainValidator\(func=~sax\.saxtypes\.core\.val_float, .*?\)\]": "float",
        r"CrossSection \| str \| dict\[str, Any\] \| Callable\[\[\.\.\.\], CrossSection\] \| SymmetricalCrossSection \| DCrossSection": "CrossSectionSpec",
    }

    def simplify_handler(_app, _what, _name, _obj, _options, lines):
        for i, line in enumerate(lines):
            for pattern, replacement in patterns.items():
                lines[i] = re.sub(pattern, replacement, line)

    def dollar_math_handler(_app, _what, _name, _obj, _options, lines):
        _dollar_math_to_rst(lines)

    app.connect("source-read", replace_image_paths)
    app.connect("doctree-resolved", _typst_drop_unresolved_myst_xrefs)
    app.connect("doctree-resolved", _typst_strip_ansi)
    # Convert $-delimited math before any other processing
    app.connect("autodoc-process-docstring", dollar_math_handler, priority=100)
    # We use a late priority to ensure we see the types added by autodoc
    app.connect("autodoc-process-docstring", simplify_handler, priority=999)
    # Fix Edit on GitHub URLs for notebook pages (runs after pydata-sphinx-theme's
    # setup_edit_url which is registered at the default priority of 500)
    app.connect("html-page-context", fix_notebook_edit_url, priority=600)

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
    ):
        name = node_class.__name__
        setattr(TypstTranslator, f"visit_{name}", visit)
        if depart is not None:
            setattr(TypstTranslator, f"depart_{name}", depart)
    TypstTranslator.visit_math_block = _typst_visit_math_block
    TypstTranslator.visit_list_item = _typst_visit_list_item
    TypstTranslator.depart_list_item = _typst_depart_list_item
