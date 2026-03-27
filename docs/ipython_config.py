# pyrefly: ignore-errors
"""IPython configuration for documentation generation.

This configuration file is used when building Jupyter Book documentation
to ensure consistent, high-quality figures in the generated docs.
"""

# Configure inline backend for matplotlib
# Export figures in multiple formats for flexibility
c.InlineBackend.figure_formats = ["pdf", "svg", "png"]  # noqa: F821

# Use tight bounding box to remove excess whitespace
c.InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}  # noqa: F821

# Set higher DPI for PNG figures (default is 72, we use 300 for publication quality)
c.InlineBackend.rc = {"figure.dpi": 300}  # noqa: F821

# Execute these lines at IPython startup
c.InteractiveShellApp.exec_lines = [  # noqa: F821
    # Import matplotlib
    "from matplotlib import pyplot as plt",
    # Configure matplotlib font embedding for better PDF/SVG compatibility
    # 'path' converts text to paths in SVG (more compatible, but larger files)
    "plt.rcParams['svg.fonttype'] = 'path'",
    # Maximum PDF compression
    "plt.rcParams['pdf.compression'] = 9",
    # TrueType font embedding (Type 42) for PDF/PS - more compatible than Type 3
    "plt.rcParams['pdf.fonttype'] = 42",
    "plt.rcParams['ps.fonttype'] = 42",
    # Load custom matplotlib style for quantum-rf-pdk documentation
    "plt.style.use('qpdk')",
    # Suppress harmless logging warnings that clutter notebook output.
    # fontTools and matplotlib emit warnings via Python's logging module (not
    # the warnings module), so they must be suppressed by raising the log level.
    "import logging",
    # fontTools: timestamp and font-subsetting warnings during PDF/SVG export
    "logging.getLogger('fontTools').setLevel(logging.ERROR)",
    # matplotlib: missing-glyph warnings for special Unicode/TeX symbols
    "logging.getLogger('matplotlib.mathtext').setLevel(logging.ERROR)",
    # Suppress harmless Python warnings that clutter notebook output
    "import warnings",
    # polars row orientation inference during DataFrame construction
    "warnings.filterwarnings('ignore', message=r'.*Row orientation inferred.*')",
]
