"""IPython configuration for documentation generation.

This configuration file is used when building Jupyter Book documentation
to ensure consistent, high-quality figures in the generated docs.
"""

# Get the IPython configuration object
c = get_config()  # noqa: F821

# Configure inline backend for matplotlib
# Export figures in multiple formats for flexibility
c.InlineBackend.figure_formats = ["pdf", "svg", "png"]

# Use tight bounding box to remove excess whitespace
c.InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}

# Set higher DPI for PNG figures (default is 72, we use 300 for publication quality)
c.InlineBackend.rc = {"figure.dpi": 300}

# Execute these lines at IPython startup
c.InteractiveShellApp.exec_lines = [
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
    # Suppress fontTools warnings that can clutter notebook output
    "import logging",
    "logging.getLogger('fontTools').setLevel(logging.WARNING)",
    # Load custom matplotlib style for quantum-rf-pdk documentation
    "from pathlib import Path",
    # Try multiple locations for the style file
    "_style_candidates = [Path('docs/qpdk.mplstyle'), Path('/home/runner/.ipython/profile_default/qpdk.mplstyle'), Path.home() / '.ipython' / 'profile_default' / 'qpdk.mplstyle']",
    "_style_path = next((p for p in _style_candidates if p.exists()), None)",
    "plt.style.use(str(_style_path)) if _style_path else None",
]
