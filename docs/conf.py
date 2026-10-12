"""Sphinx configuration for Qpdk documentation."""

import sys
from pathlib import Path

# Local Sphinx extensions live in ``docs/_ext`` and are imported by name below.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))


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
    "sphinx_llm.txt",
    # local, see docs/_ext/
    "alias_typehints",
    "qpdk_source_fixes",
    "qpdk_theme_figures",
    "qpdk_typst_compile",
    "qpdk_typst",
]

# -- Plot directive configuration ---------------------------------------------

plot_pre_code = """
import os
from matplotlib import pyplot as plt
from qpdk import PDK
import qpdk.colormaps  # registers the ``image.cmap`` of both styles
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)

plt.style.use(os.environ.get("QPDK_MPL_STYLE", "qpdk"))
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
    # Local Sphinx extensions, not pages; myst-nb would otherwise run them as
    # jupytext notebooks.
    "_ext",
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

# -- Mermaid configuration ---------------------------------------------------
# QPDK diagram style, matching docs/qpdk.mplstyle and docs/_static/css/custom.css:
# Inter labels, Outfit bold subgraph titles, the QPDK accent for outlines and
# edges, and thin strokes (axes.linewidth / lines.linewidth).
#
# sphinxcontrib-mermaid only swaps the theme *name* when the page toggles
# between light and dark, so the colours can't live in ``themeVariables``.
# Instead ``themeCSS`` reads ``--qpdk-mermaid-*`` custom properties, which
# custom.css defines per ``data-theme`` and the inline SVG inherits.  The
# ``themeVariables`` below only seed mermaid's own layout and derived colours.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
_mermaid_theme_css = """
.node rect, .node circle, .node ellipse, .node polygon, .node path {
  fill: var(--qpdk-mermaid-node-fill);
  stroke: var(--qpdk-mermaid-node-stroke);
  stroke-width: 1.2px;
}
.node rect { rx: 6px; ry: 6px; }
.nodeLabel, .node .label, .edgeLabel, .cluster-label, .label text {
  color: var(--qpdk-mermaid-text);
  fill: var(--qpdk-mermaid-text);
}
.flowchart-link, .edgePath .path {
  stroke: var(--qpdk-mermaid-line);
  stroke-width: 1.5px;
}
.marker, .arrowheadPath {
  fill: var(--qpdk-mermaid-line);
  stroke: var(--qpdk-mermaid-line);
}
.edgeLabel, .edgeLabel p, .labelBkg {
  background-color: var(--qpdk-mermaid-bg);
  font-size: 12px;
}
.edgeLabel rect { fill: var(--qpdk-mermaid-bg); }
.cluster rect {
  fill: var(--qpdk-mermaid-cluster-fill);
  stroke: var(--qpdk-mermaid-cluster-stroke);
  stroke-dasharray: 4 3;
  rx: 8px;
  ry: 8px;
}
.cluster-label, .cluster-label span, .cluster-label p {
  color: var(--qpdk-mermaid-text);
  font-family: "Outfit", "Inter", sans-serif;
  font-weight: 700;
}
"""
mermaid_init_config = {
    "startOnLoad": False,
    "themeVariables": {
        "fontFamily": '"Inter", "DejaVu Sans", Helvetica, Arial, sans-serif',
        "fontSize": "14px",
        "primaryColor": "#e8eff8",
        "primaryBorderColor": "#2a6fb5",
        "primaryTextColor": "#0e1116",
        "lineColor": "#2a6fb5",
        "clusterBkg": "#f6f4ef",
    },
    "themeCSS": _mermaid_theme_css,
    "flowchart": {"curve": "basis", "padding": 12, "useMaxWidth": False},
}

# -- Notebook execution (myst-nb) --------------------------------------------
nb_execution_mode = "cache"
# myst-nb's default cache is ``<outdir>/../.jupyter_cache``.  Pin it to that same
# path for every build so the sphinx-llm Markdown pass (whose outdir is nested
# inside the HTML one) hits the cache filled by the HTML pass.
nb_execution_cache_path = str(Path(__file__).parent / "_build" / ".jupyter_cache")
# External-solver notebooks keep saved outputs; the docs build does not run them.
nb_execution_excludepatterns = [
    "notebooks/comsol*",
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
# The sphinx-llm Markdown builder has no myst-nb entry either; keep the text
# representation of each output so llms-full.txt carries printed results.
nb_mime_priority_overrides += [
    ("llms-markdown", "text/markdown", 10),
    ("llms-markdown", "text/plain", 20),
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

# -- llms.txt (sphinx-llm) -----------------------------------------------------
# sphinx_llm.txt runs a second, Markdown-builder pass after the HTML build and
# writes ``llms.txt`` (https://llmstxt.org/), ``llms-full.txt`` and a ``.md``
# copy of every page into the HTML output.  It is a no-op for other builders.
# The pass runs sequentially on the HTML doctree directory, so it reuses the
# pickled environment and reads no notebook again (``nb_execution_cache_path``
# above covers the case where it does).  In parallel mode it would get a fresh
# doctree directory and execute every notebook a second time.
llms_txt_build_parallel = False
# sphinx-markdown-builder has no writer for these nodes and omits them from the
# Markdown output; don't warn about it on every build.
llms_txt_suppress_unknown_node_warnings = [
    "PassthroughTextElement",
    "abbreviation",
    "admonition",
    "caption",
    "citation",
    "mermaid",
    "meta",
]
llms_txt_full_build = True
llms_txt_description = (
    "QPDK is an open-source process design kit (PDK) for superconducting quantum RF "
    "applications built on gdsfactory. It provides parametric quantum circuit "
    "components (transmon qubits, CPW resonators, Josephson junctions, etc.), "
    "analytical S-parameter models, routing utilities, and test-chip examples. "
    "Source: https://github.com/gdsfactory/quantum-rf-pdk; published as `qpdk` on PyPI."
)
markdown_http_base = "https://gdsfactory.github.io/quantum-rf-pdk"

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

# The Typst compile wrapper and the doctree/translator adaptations for the PDF
# live in the `qpdk_typst_compile` and `qpdk_typst` extensions in docs/_ext/.

# -- Warning suppression ------------------------------------------------------
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "bibtex.duplicate_citation",
]
