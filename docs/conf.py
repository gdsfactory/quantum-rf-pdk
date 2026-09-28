"""Sphinx configuration for Qpdk documentation."""

import json
import re
import shutil
import sys
import tempfile
from html import escape
from pathlib import Path

import typst
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx_design.shared import PassthroughTextElement
from typsphinx.translator import TypstTranslator, escape_typst_string

# Local Sphinx extensions live in ``docs/_ext`` and are imported by name below.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

_TYPST_VISIT_MATH_BLOCK = TypstTranslator.visit_math_block
_TYPST_VISIT_BULLET_LIST = TypstTranslator.visit_bullet_list
_TYPST_DEPART_BULLET_LIST = TypstTranslator.depart_bullet_list
_TYPST_VISIT_ADMONITION = TypstTranslator._visit_admonition
logger = logging.getLogger(__name__)

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
    "alias_typehints",  # local, see docs/_ext/
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
# Exclude the notebooks that drive external solvers from execution: Ansys HFSS
# (proprietary/licensed) and Elmer FEM (``ElmerGrid``/``ElmerSolver`` on PATH), plus
# MATLAB. These can be slow or impossible to run in typical documentation build
# environments.
nb_execution_excludepatterns = [
    "notebooks/hfss*",
    "notebooks/elmer*",
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
        ("text/latex", 10),
        ("image/svg+xml", 20),
        ("image/png", 30),
        ("image/jpeg", 40),
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

# typsphinx calls typst.compile() without font_paths or pdf_standards, and
# typst-py ignores TYPST_FONT_PATHS. PDF/A-2b is an export-time setting with
# no in-document equivalent, so it has to be injected here: typsphinx imports
# the module inside its compile function, which picks up this wrapper.
# font_paths is only needed when the families are not installed system-wide;
# docs.just fetches them into build/docs-fonts, CI installs them via fc-cache.
_local_fonts = Path(__file__).parent.parent / "build" / "docs-fonts"

_typst_compile = typst.compile


def _repair_svgbob_svgs(root):
    """Fix svgbob SVG font stacks Typst cannot resolve.

    sphinxcontrib-svgbob writes ``font-family: Iosevka Fixed, monospace``;
    Typst's SVG engine parses the comma stack as one unknown family, so the
    text falls back to LastResort (tofu, and a hard PDF/A-2b error for
    box-drawing glyphs like U+2577).  DejaVu Sans Mono ships with Typst and
    covers them.
    """
    # typsphinx rewrites image URIs source-root-relative, so the SVGs end up
    # nested one _build/typstpdf deeper than the compile root.
    for svg_path in root.rglob("*.svg"):
        src = svg_path.read_text()
        if "Iosevka Fixed" not in src:
            continue
        svg_path.write_text(
            src.replace(
                "font-family: Iosevka Fixed, monospace;",
                "font-family: DejaVu Sans Mono;",
            )
        )


def _rename_ipython_fences(root):
    """Rename ipython3 code fences to python so codly highlights them."""
    # Notebook code cells carry the kernel language ipython3, which
    # codly-languages has no lexer for, so they would render unhighlighted.
    for typ_path in root.rglob("*.typ"):
        src = typ_path.read_text()
        if "```ipython3" not in src:
            continue
        typ_path.write_text(src.replace("```ipython3", "```python"))


def _stage_template_assets(root):
    """Copy template-referenced docs/_static assets into the template bundle."""
    # These files live in docs/_static/ but are referenced only by the
    # template, and typsphinx copies just the document-referenced assets
    # into the build, so they have to be staged next to the template copy.
    bundle = root / "_template" / "typst"
    if bundle.is_dir():
        # Staged flat: the template reads read("custom.css") relative to
        # itself, so the css/ subpath from _static is dropped.
        for name in ("qpdk_logo.svg", "css/custom.css"):
            shutil.copy(
                Path(__file__).parent / "_static" / name, bundle / Path(name).name
            )


# Text column is 164mm wide (A4 minus the template margins); keep a hair of
# slack so an equation at exactly the column width does not get broken up.
_MATH_COLUMN_MM = 163

# latex bodies already carrying manual linebreaks or alignment markup
_MATH_SKIP = ("\\begin{", "\\\\")

_MATH_RELATIONS = (
    "\\approx",
    "\\equiv",
    "\\ge",
    "\\geq",
    "\\le",
    "\\leq",
    "\\propto",
    "\\sim",
)

_mitex_re = re.compile(r"mitex\(`(.*?)`\)", re.DOTALL)


def _math_widths(bodies, font_paths):
    """Measure rendered widths (mm) of block equations for LaTeX bodies."""
    if not bodies:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "math-widths.typ"
        probe.write_text(
            '#import "@preview/mitex:0.2.7": mitex\n'
            '#set text(font: ("Fira Math", "New Computer Modern Math"), size: 10pt)\n'
            # trailing comma: ("x") is a parenthesized string, not an array
            f"#let eqs = ({', '.join(json.dumps(b, ensure_ascii=False) for b in bodies)},)\n"
            "#context for eq in eqs {\n"
            "  metadata(measure(block(math.equation(block: true, mitex(eq)))).width / 1mm)\n"
            "}\n"
        )
        result = typst.query(str(probe), "metadata", font_paths=font_paths or [])
    return [row["value"] for row in json.loads(result)]


def _split_wide_math(body):
    """Split a LaTeX body at top-level binary +/- into (head, terms).

    head runs up to and including the first top-level relation.

    Returns:
        (head, terms) tuple, or None when there is no relation or no
        top-level term boundary.
    """
    cuts = []
    depth = 0
    relation_end = None
    prev = ""
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\":
            word = re.match(r"\\[a-zA-Z]+", body[i:])
            if word:
                word = word.group(0)
                i += len(word)
                # \left( / \right) bump depth twice, once for the word and
                # once for the delimiter char; consistent, so nesting works
                if word == "\\left":
                    depth += 1
                elif word == "\\right":
                    depth -= 1
                elif depth == 0 and relation_end is None and word in _MATH_RELATIONS:
                    relation_end = i
                prev = word
                continue
            # escaped delimiters like \{ or \% count as plain text
            i += 2
            prev = body[i - 1]
            continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif c == "=" and depth == 0:
            if relation_end is None:
                relation_end = i + 1
        elif (
            c in "+-"
            and depth == 0
            and relation_end is not None
            and prev
            not in {
                "",
                "+",
                "-",
                "=",
                "(",
                "[",
                "{",
                "^",
                "_",
                ",",
                "\\left",
            }
            and prev not in _MATH_RELATIONS
        ):
            cuts.append(i)
        if not c.isspace():
            prev = c
        i += 1
    if not cuts or relation_end is None:
        return None
    head = body[:relation_end].rstrip()
    starts = [relation_end, *cuts]
    terms = [body[a:b].strip() for a, b in zip(starts, [*cuts, n])]
    return head, terms


def _wrap_wide_math(root, font_paths):
    """Rewrite block mitex equations wider than the text column as multiline.

    Typst block equations never wrap, so an over-wide one just runs into the
    margin; break it after its relation and at top-level +/- term boundaries,
    aligned at the relation.  Terms are packed greedily, as many per line as
    fit beside the head.  A body whose multiline form still overflows is left
    untouched.
    """
    entries = []  # (path, span start, span end, latex body)
    for typ_path in root.rglob("*.typ"):
        entries.extend(
            (typ_path, m.start(1), m.end(1), m.group(1))
            for m in _mitex_re.finditer(typ_path.read_text())
            if not any(s in m.group(1) for s in _MATH_SKIP)
        )
    if not entries:
        return
    wide = [
        (entry, *split)
        for entry, width in zip(
            entries, _math_widths([e[3] for e in entries], font_paths)
        )
        if width > _MATH_COLUMN_MM and (split := _split_wide_math(entry[3]))
    ]
    if not wide:
        return

    # Greedy packing, one probe per round: every unfinished candidate
    # contributes its trial line, so packing all equations together costs
    # a handful of compiles.
    lines = [[] for _ in wide]  # accepted term groups per candidate
    taken = [0] * len(wide)
    failed = [False] * len(wide)
    while any(not fail and taken[k] < len(wide[k][2]) for k, fail in enumerate(failed)):
        trials = []
        order = []
        for k, (_, head, terms) in enumerate(wide):
            if failed[k] or taken[k] >= len(terms):
                continue
            group = lines[k][-1] if lines[k] else []
            order.append(k)
            trials.append(f"{head} {' '.join([*group, terms[taken[k]]])}")
        for k, width in zip(order, _math_widths(trials, font_paths)):
            terms = wide[k][2]
            if width <= _MATH_COLUMN_MM:
                if lines[k]:
                    lines[k][-1].append(terms[taken[k]])
                else:
                    lines[k].append([terms[taken[k]]])
            elif lines[k]:  # line full: start a new one at this term
                lines[k].append([terms[taken[k]]])
            else:  # a lone term beside the head overflows: give up
                failed[k] = True
                continue
            taken[k] += 1

    finals = []
    for (entry, head, _), groups, fail in zip(wide, lines, failed):
        if fail or not groups:
            continue
        finals.append((
            entry,
            f"{head} & {' '.join(groups[0])}"
            + "".join(f" \\\\ & {' '.join(g)}" for g in groups[1:]),
        ))
    rewrites = {}
    for (entry, split), width in zip(
        finals, _math_widths([split for _, split in finals], font_paths)
    ):
        if width <= _MATH_COLUMN_MM:
            rewrites.setdefault(entry[0], []).append((entry[1], entry[2], split))
    for typ_path, spans in rewrites.items():
        src = typ_path.read_text()
        for start, end, split in sorted(spans, reverse=True):
            src = src[:start] + split + src[end:]
        typ_path.write_text(src)


def _check_docs_fonts(font_paths):
    """Fail before compiling when required docs font variants do not resolve."""
    # typst.compile() silently discards font warnings, so an unfound family
    # just falls back to tofu; only compile_with_warnings reports it.
    fonts = (
        ("Outfit", 600),
        ("Outfit", 700),
        ("Inter", 400),
        ("Code New Roman", 400),
        ("Fira Math", 400),
    )
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "font-probe.typ"
        probe.write_text(
            "".join(
                f'#text(font: ("{family}",), weight: {weight})[a]\n'
                for family, weight in fonts
            )
        )
        _, warnings = typst.compile_with_warnings(
            str(probe), font_paths=font_paths or []
        )
    unresolved = [
        warning.message
        for warning in warnings
        if "unknown font family" in warning.message.lower()
        or "font variant" in warning.message.lower()
    ]
    if unresolved:
        raise RuntimeError(
            f"Typst cannot resolve the docs fonts: {'; '.join(unresolved)}. "
            "Run `just fetch-docs-fonts` or install them system-wide, then rebuild."
        )


def _typst_compile_with_fonts(*args, **kwargs):
    """Compile Typst as PDF/A-2b, with the local docs font cache when present.

    Returns:
        The compiled PDF bytes.
    """
    kwargs.setdefault("pdf_standards", ["a-2b"])
    if _local_fonts.is_dir():
        kwargs.setdefault("font_paths", [str(_local_fonts)])
    _check_docs_fonts(kwargs.get("font_paths"))
    root = Path(kwargs.get("root") or Path(args[0]).parent)
    if root.is_dir():
        _repair_svgbob_svgs(root)
        _rename_ipython_fences(root)
        _stage_template_assets(root)
        _wrap_wide_math(root, kwargs.get("font_paths"))
    return _typst_compile(*args, **kwargs)


typst.compile = _typst_compile_with_fonts

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


# Markdown links that are not images, external URLs, or in-page anchors.
_RELATIVE_LINK_RE = re.compile(r"(?<!!)(\[[^\]]*\])\((?!\w+:|#)([^)\s]+)\)")


def _rewrite_readme_links(content: str, srcdir: Path) -> str:
    """Rewrite repository-relative README links for the docs build.

    README links are written relative to the repository root so they work on
    GitHub.  The README is inlined into ``docs/index.md``, where the same paths
    resolve relative to ``docs/`` instead, so ``docs/contributing.md`` becomes a
    dangling ``docs/docs/contributing.md`` reference.  Point links at the page
    in the docs tree when there is one, and at GitHub otherwise (e.g.
    ``LICENSE``, which is not part of the documentation).

    Args:
        content: README Markdown about to be inlined into ``index``.
        srcdir: Documentation source directory, used to tell a link to a
            documentation page from a link to some other repository file.

    Returns:
        The same Markdown with repository-relative links resolvable from
        ``docs/index.md``.
    """
    # Assembled from parts so the link checker does not read the format string
    # itself as a (broken) URL.
    github_blob = "/".join((
        "https://github.com",
        html_context["github_user"],
        html_context["github_repo"],
        "blob",
        html_context["github_version"],
    ))

    def _rewrite(match: re.Match[str]) -> str:
        text, target = match.groups()
        in_docs = target.removeprefix("docs/")
        if (srcdir / in_docs).exists():
            return f"{text}({in_docs})"
        return f"{text}({github_blob}/{target})"

    return _RELATIVE_LINK_RE.sub(_rewrite, content)


def replace_image_paths(app, docname, source):
    """Fix image paths and manually include README.md into index."""
    if docname == "index":
        readme = Path(app.srcdir).parent / "README.md"
        if readme.exists():
            content = readme.read_text(encoding="utf-8").split("_" * 70, 1)[-1]
            content = _rewrite_readme_links(content, Path(app.srcdir))
            source[0] = re.sub(
                r"```\{include\}\s+\.\./README\.md.*?```",
                lambda _: content,
                source[0],
                flags=re.DOTALL,
            )

    source[0] = source[0].replace("docs/_static/images/", "/_static/images/")


def inline_figures(app: Sphinx, doctree: nodes.document, _docname: str) -> None:
    """Inline generated Typst SVGs in HTML so their text can be selected."""
    if app.builder.format != "html":
        return

    figure_sources = Path(app.srcdir) / "figures"
    for image in doctree.findall(nodes.image):
        uri = Path(image["uri"])
        if (
            uri.parent == Path("notebooks/figures")
            and (figure_sources / f"{uri.stem}.typ").is_file()
        ):
            svg = (Path(app.srcdir) / uri).read_text(encoding="utf-8")
            label = escape(image.get("alt", ""), quote=True)
            svg = svg.replace(
                "<svg ",
                f'<svg role="group" aria-label="{label}" ',
                1,
            )
            image.replace_self(nodes.raw("", svg, format="html"))


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


def _typst_visit_meta(_self, _node):
    """Drop docutils ``meta`` nodes, which carry HTML ``<meta>`` tags only.

    Raises:
        nodes.SkipNode: Always, the node has no print representation.
    """
    raise nodes.SkipNode


def setup(app):
    """Sphinx setup."""

    # Type shortening used to live here as regexes over the processed docstring
    # lines, but `autodoc_typehints = "description"` injects parameter types into
    # the doctree and never through `autodoc-process-docstring`, so they never
    # matched.  It is now done in the `alias_typehints` extension instead.
    def dollar_math_handler(_app, _what, _name, _obj, _options, lines):
        _dollar_math_to_rst(lines)

    # Earlier than the default 500 so the README splice lands before
    # sphinx_github_alerts' source-read hook, which then also converts the
    # README's `> [!NOTE]` alert instead of leaving it as a literal quote.
    app.connect("source-read", replace_image_paths, priority=400)
    app.connect("doctree-resolved", inline_figures)
    app.connect("doctree-resolved", _typst_drop_unresolved_myst_xrefs)
    app.connect("doctree-resolved", _typst_strip_ansi)
    app.connect("doctree-resolved", _typst_lift_block_images)
    # Convert $-delimited math before any other processing
    app.connect("autodoc-process-docstring", dollar_math_handler, priority=100)
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
