"""Sphinx configuration for Qpdk documentation."""

import re
from pathlib import Path

project = "qpdk"
author = "gdsfactory"
copyright = "gdsfactory"  # noqa: A001

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.katex",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.svgbob",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_github_alerts",
    "sphinxcontrib.bibtex",
    "sphinxcontrib_bibtex_urn",
]

# -- Plot directive configuration ---------------------------------------------
plot_pre_code = """
from matplotlib import pyplot as plt
from qpdk import PDK

plt.style.use("qpdk")
PDK.activate()
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
    "sax": ("https://flaport.github.io/sax/", None),
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
html_logo = "logo.png"
html_show_copyright = False
templates_path = ["templates"]
html_theme_options = {
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
html_css_files = [
    "css/custom.css",
]

# -- LaTeX / PDF output -------------------------------------------------------
latex_engine = "xelatex"
latex_documents = [
    ("index", "qpdk.tex", "Qpdk", "gdsfactory", "manual"),
]
latex_show_pagerefs = True
latex_show_urls = "footnote"
latex_use_xindy = True
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "fncychap": (
        r"\usepackage[Bjornstrup]{fncychap}"
        "\n"
        r"\ChNumVar{\fontsize{50}{54}\usefont{OT1}{pzc}{m}{n}\selectfont}"
        "\n"
        r"\ChTitleVar{\raggedleft\Large\sffamily\bfseries}"
    ),
}

# -- Warning suppression ------------------------------------------------------
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "bibtex.duplicate_citation",
]


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
                math_line = lines[i]
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
            r":math:`\1`",
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
    # Convert $-delimited math before any other processing
    app.connect("autodoc-process-docstring", dollar_math_handler, priority=100)
    # We use a late priority to ensure we see the types added by autodoc
    app.connect("autodoc-process-docstring", simplify_handler, priority=999)
