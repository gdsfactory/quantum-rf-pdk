"""Sphinx configuration for Qpdk documentation."""

import re

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
nb_execution_excludepatterns = [
    "notebooks/hfss*",
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
html_theme_options = {
    "use_edit_page_button": True,
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

    # We use a late priority to ensure we see the types added by autodoc
    app.connect("autodoc-process-docstring", simplify_handler, priority=999)
