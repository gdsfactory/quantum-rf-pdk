"""Render docs figures for both the light and the dark theme.

Every ``.. plot::`` figure is rendered a second time with the dark style, and
notebook figures and svgbob diagrams get a recoloured dark copy; the light and
dark images are then paired so the theme shows only the matching one.  Generated
Typst SVGs are also inlined into the HTML so their text can be selected.
"""

import os
import re
from html import escape
from pathlib import Path

from docutils import nodes
from matplotlib.colors import to_hex
from matplotlib.sphinxext import plot_directive
from sphinx.application import Sphinx
from sphinxcontrib.svgbob._svgbob import to_svg

from qpdk.colormaps import QPDK, QPDK_DARK

#: Style used for the second, dark render of every ``.. plot::`` figure, and the
#: suffix its output files get.  See `_render_figures_for_both_themes`.
PLOT_DARK_STYLE = "qpdk-dark"
PLOT_DARK_SUFFIX = "_dark"
#: Selects the style for the render in progress.  ``plot_pre_code`` is executed
#: as source text in a namespace of the directive's own, so the style has to
#: reach it through the environment rather than as a variable.
PLOT_STYLE_ENV = "QPDK_MPL_STYLE"


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


_plot_render_figures = plot_directive.render_figures


def _render_figures_for_both_themes(*args, **kwargs):
    """Render every ``.. plot::`` figure a second time in the dark style.

    Matplotlib bakes colours into each artist as it is created, so a figure
    cannot be recoloured after the fact -- the snippet has to run again under
    a different style.  The second run writes to ``<basename>_dark.<fmt>``
    beside the first, and `pair_plot_images_by_theme` puts the two files on the
    page as a light/dark pair.

    Args:
        *args: Positional arguments for ``plot_directive.render_figures``.
        **kwargs: Keyword arguments for the same; ``output_base`` is the one
            this overrides for the dark render.

    Returns:
        The light render's results, unchanged, so the directive keeps emitting
        exactly the markup it did before.
    """
    light = _plot_render_figures(*args, **kwargs)
    # The PDF has a single, printed theme, and building it is already the slow
    # half of `just docs`.
    if plot_directive.setup.app.builder.format != "html":
        return light

    previous = os.environ.get(PLOT_STYLE_ENV)
    os.environ[PLOT_STYLE_ENV] = PLOT_DARK_STYLE
    try:
        _plot_render_figures(
            *args,
            **kwargs | {"output_base": kwargs["output_base"] + PLOT_DARK_SUFFIX},
        )
    finally:
        if previous is None:
            del os.environ[PLOT_STYLE_ENV]
        else:
            os.environ[PLOT_STYLE_ENV] = previous

    return light


#: How the colours ``qpdk.mplstyle`` leaves at matplotlib's defaults map onto
#: ``qpdk-dark.mplstyle``: background, ink, grid, the tab10 cycle, and the colormap.  Used to
#: derive the dark notebook figures, see `_write_dark_notebook_figure`.
_NOTEBOOK_FIGURE_DARK_COLOURS = {
    "#ffffff": "#14181e",
    "#000000": "#f0f2f6",
    "#333333": "#c5cbd5",
    "#b0b0b0": "#586272",
    "#1f77b4": "#76a9ee",
    "#ff7f0e": "#ffa95e",
    "#2ca02c": "#5fd38d",
    "#d62728": "#ff7b7b",
    "#9467bd": "#c79bf0",
    "#8c564b": "#c08f74",
    "#e377c2": "#f79fd0",
    "#7f7f7f": "#a3adb8",
    "#bcbd22": "#d7dd5c",
    "#17becf": "#5fd3d3",
}
#: Vector heatmaps (``pcolormesh``, ``contourf``) drawn with the ``qpdk`` colormap
#: take the matching ``qpdk-dark`` colour.  Rasters (``imshow``) are not recoloured.
_NOTEBOOK_FIGURE_DARK_COLOURS |= {
    light: to_hex(dark)
    for light, dark in zip(map(to_hex, QPDK(range(QPDK.N))), QPDK_DARK(range(QPDK.N)))
    if light not in _NOTEBOOK_FIGURE_DARK_COLOURS
}
_HEX_COLOUR_RE = re.compile(r"#[0-9a-fA-F]{6}\b")


def _write_dark_notebook_figure(light: Path, dark: Path) -> None:
    """Derive the dark render of a notebook figure from its light SVG.

    Notebook figures come out of a single kernel run, and running every
    notebook a second time under ``qpdk-dark`` would double the slowest part of
    the docs build.  Instead the light SVG is recoloured through
    `_NOTEBOOK_FIGURE_DARK_COLOURS`.  That covers everything the style sets
    (backgrounds, text, spines, grid, the default colour cycle), which is
    nearly all of what the notebooks draw.  Colours a notebook picks for itself
    and embedded rasters such as ``imshow`` output are left as they are.
    """
    svg = _HEX_COLOUR_RE.sub(
        lambda match: _NOTEBOOK_FIGURE_DARK_COLOURS.get(
            match.group().lower(), match.group()
        ),
        light.read_text(encoding="utf-8"),
    )
    # Matplotlib leaves out ``fill`` where it would be black, text included, so
    # SVG's own default has to change too.  Children that set a fill still win.
    svg = svg.replace(
        "<svg ", f'<svg fill="{_NOTEBOOK_FIGURE_DARK_COLOURS["#000000"]}" ', 1
    )
    dark.write_text(svg, encoding="utf-8")


def _is_notebook_output(node: nodes.Node) -> bool:
    """Whether ``node`` sits in a myst-nb cell output."""
    parent = node.parent
    while parent is not None:
        if parent.get("nb_element") == "mime_bundle":
            return True
        parent = parent.parent
    return False


#: pydata-sphinx-theme's dark ``--pst-color-text-base``, the colour of body
#: text.  Hard-coded because a file behind ``<img>`` cannot read the page's CSS
#: custom properties.
_DARK_INK = "#ced6dd"
#: pydata-sphinx-theme's dark ``--pst-color-background``, same caveat.
_DARK_PAPER = "#14181e"
#: Arrow and diamond markers are ``<polygon>``\ s without a class, so svgbob's
#: colour options never reach their fill and it stays the SVG default, black.
_SVGBOB_DARK_MARKER_STYLE = f"\n.svgbob marker polygon {{ fill: {_DARK_INK}; }}\n"


def _write_dark_svgbob(svgbob: nodes.Node, dark: Path) -> None:
    """Render an svgbob diagram again in the dark theme's text and background.

    Unlike a plot, a diagram should blend into the page, so it takes the
    theme's own colours rather than the dark matplotlib style's.
    """
    options = svgbob["options"]
    svg = to_svg(
        svgbob["code"],
        font_size=options.get("font-size"),
        font_family=options.get("font-family"),
        stroke_width=options.get("stroke-width"),
        scale=options.get("scale"),
        fill_color=_DARK_INK,
        stroke_color=_DARK_INK,
        background_color=_DARK_PAPER,
    )
    dark.write_text(
        svg.replace("</style>", f"{_SVGBOB_DARK_MARKER_STYLE}</style>", 1),
        encoding="utf-8",
    )


def pair_plot_images_by_theme(app: Sphinx, doctree: nodes.document) -> None:
    """Show the light or the dark render of a plot depending on the theme.

    Each plot image gains pydata-sphinx-theme's ``only-light`` class and is
    followed by a copy of itself pointing at the dark render, marked
    ``only-dark``.  The theme shows whichever matches ``html[data-theme]``,
    which its default ``auto`` mode resolves from the OS colour-scheme
    preference -- so this follows both the preference and the toggle button.

    ``.. plot::`` figures already have a dark render beside them (see
    `_render_figures_for_both_themes`); SVG notebook outputs and svgbob
    diagrams get one made here (see `_write_dark_notebook_figure` and
    `_write_dark_svgbob`).  Without the pair, pydata-sphinx-theme would also
    paint a white background and a dimming filter onto the image in dark mode.

    This runs before Sphinx's asset collector (priority < 500) so the dark
    files it introduces are copied into the output alongside the light ones.
    Images the plot directive emits for non-HTML builders are globbed
    (``<basename>.*``) rather than named, so the suffix test skips them.
    """
    if app.builder.format != "html":
        return

    document_dir = Path(doctree["source"]).parent
    for image in list(doctree.findall(nodes.image)):
        uri = Path(image["uri"])
        dark_uri = uri.with_name(f"{uri.stem}{PLOT_DARK_SUFFIX}{uri.suffix}")
        # Sphinx resolves a leading "/" against the source directory, which is
        # how myst-nb points at the outputs it writes beside the build.
        base_dir = Path(app.srcdir) if uri.is_absolute() else document_dir
        dark_path = base_dir / dark_uri.as_posix().lstrip("/")
        if not dark_path.is_file():
            if "svgbob" in image:
                _write_dark_svgbob(image["svgbob"], dark_path)
            elif uri.suffix == ".svg" and _is_notebook_output(image):
                _write_dark_notebook_figure(dark_path.with_name(uri.name), dark_path)
        if not dark_path.is_file():
            continue

        dark_image = image.deepcopy()
        dark_image["uri"] = dark_uri.as_posix()
        dark_image["classes"] = [*image["classes"], "only-dark"]
        image["classes"].append("only-light")
        image.parent.insert(image.parent.index(image) + 1, dark_image)


def setup(app: Sphinx) -> None:
    """Register the extension."""
    # Earlier than the default 500 so Sphinx's asset collector, which runs on
    # the same event, sees the dark images this adds and copies them out.
    app.connect("doctree-read", pair_plot_images_by_theme, priority=400)
    plot_directive.render_figures = _render_figures_for_both_themes
    app.connect("doctree-resolved", inline_figures)
