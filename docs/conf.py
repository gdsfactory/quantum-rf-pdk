"""Sphinx configuration for Qpdk documentation."""

project = "Qpdk"
author = "gdsfactory"
copyright = "gdsfactory"  # noqa: A001

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgconverter",
    "sphinxcontrib.katex",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.svgbob",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

exclude_patterns = [
    "_build",
    "conf.py",
    "ipython_config.py",
    "*.mplstyle",
    "justfile_help.txt",
    "Thumbs.db",
    ".DS_Store",
]

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
    "gdsfactory.typings.CrossSectionSpec": "CrossSectionSpec",
    "gdsfactory.typings.LayerSpec": "LayerSpec",
    "gdsfactory.typings.ComponentSpec": "ComponentSpec",
    (
        "CrossSection | str | dict[str, Any] | "
        "Callable[[...], CrossSection] | SymmetricalCrossSection | DCrossSection"
    ): "CrossSectionSpec",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

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
            "name": "PyPI",
            "url": "https://pypi.org/project/qpdk/",
            "icon": "fa-solid fa-box",
        },
    ],
}
html_context = {
    "github_user": "gdsfactory",
    "github_repo": "quantum-rf-pdk",
    "github_version": "main",
    "doc_path": "docs",
}

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
