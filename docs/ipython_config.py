# pyrefly: ignore-errors
"""IPython configuration for documentation generation.

This configuration file is used when building Jupyter Book documentation
to ensure consistent, high-quality figures in the generated docs.
"""

# Configure inline backend for matplotlib
# Export figures in multiple formats for flexibility
c.InlineBackend.figure_formats = ["pdf", "svg", "png"]  # ruff: ignore[undefined-name]

# Use tight bounding box to remove excess whitespace
c.InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}  # ruff: ignore[undefined-name]

# Set higher DPI for PNG figures (default is 72, we use 300 for publication quality)
c.InlineBackend.rc = {"figure.dpi": 300}  # ruff: ignore[undefined-name]

# Execute these lines at IPython startup
c.InteractiveShellApp.exec_lines = [  # ruff: ignore[undefined-name]
    # Import matplotlib
    "from matplotlib import pyplot as plt",
    # Rebuild the matplotlib font cache so newly-installed fonts are discovered
    "import matplotlib.font_manager as _fm; _fm._load_fontmanager(try_read_cache=False)",
    # Configure matplotlib font embedding for better PDF/SVG compatibility
    # 'path' converts text to paths in SVG (more compatible, but larger files)
    "plt.rcParams['svg.fonttype'] = 'path'",
    # Maximum PDF compression
    "plt.rcParams['pdf.compression'] = 9",
    # TrueType font embedding (Type 42) for PDF/PS - more compatible than Type 3
    "plt.rcParams['pdf.fonttype'] = 42",
    "plt.rcParams['ps.fonttype'] = 42",
    # Load custom matplotlib style for QPDK documentation
    "plt.style.use('qpdk')",
    # Monkey-patch Axes.set_title so figure titles use Outfit (bold) to match
    # the Sphinx documentation heading font (see docs/_static/css/custom.css).
    "import matplotlib.axes as _ma; _orig_title = _ma.Axes.set_title",
    (
        "def _qpdk_title(self, *args, **kwargs):\n"
        "    kwargs.setdefault('fontfamily', 'Outfit')\n"
        "    kwargs.setdefault('fontweight', 'bold')\n"
        "    return _orig_title(self, *args, **kwargs)"
    ),
    "_ma.Axes.set_title = _qpdk_title; del _qpdk_title",
    # Suppress harmless logging warnings that clutter notebook output.
    # fontTools and matplotlib emit warnings via Python's logging module (not
    # the warnings module), so they must be suppressed by raising the log level.
    "import logging",
    # fontTools: timestamp and font-subsetting warnings during PDF/SVG export
    "logging.getLogger('fontTools').setLevel(logging.ERROR)",
    # matplotlib: missing-glyph warnings for special Unicode/TeX symbols
    "logging.getLogger('matplotlib.mathtext').setLevel(logging.ERROR)",
    # Raising a logger's level is too blunt when the same logger also emits
    # warnings worth keeping, so skip individual messages instead.  Filters
    # match on ``record.msg`` (the unformatted format string), so a pattern
    # stays stable across the message's ``%s`` arguments.
    (
        "def _qpdk_ignore_log_messages(logger_name, *patterns):\n"
        "    logging.getLogger(logger_name).addFilter(\n"
        "        lambda record: not any(p in record.msg for p in patterns)\n"
        "    )"
    ),
    # matplotlib.font_manager: Fira Math (``mathtext.bf`` in qpdk.mplstyle)
    # ships a Regular weight only, so every bold math title logs
    # "findfont: Failed to find font weight bold, now using 400." even though
    # the rendered output is exactly what we asked for.  The logger's other
    # warnings stay visible on purpose — "Font family ... not found" means the
    # documentation fonts failed to install, which we do want to hear about.
    (
        "_qpdk_ignore_log_messages("
        "'matplotlib.font_manager', 'Failed to find font weight')"
    ),
    # Suppress harmless Python warnings that clutter notebook output
    "import warnings",
    # polars row orientation inference during DataFrame construction
    "warnings.filterwarnings('ignore', message=r'.*Row orientation inferred.*')",
]
