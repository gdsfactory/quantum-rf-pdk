"""Compile the Typst PDF with the docs fonts, as PDF/A-2b, after repairing the sources.

typsphinx hands its generated ``.typ`` sources straight to ``typst.compile``.
This extension wraps that call to inject the font paths and PDF standard
typsphinx cannot pass, and to patch up the sources it emits just before they are
compiled.  It also registers the local docs font cache with matplotlib.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import typst
from matplotlib import font_manager
from sphinx.application import Sphinx

#: The ``docs/`` directory.
_DOCS_DIR = Path(__file__).parent.parent

# typsphinx calls typst.compile() without font_paths or pdf_standards, and
# typst-py ignores TYPST_FONT_PATHS. PDF/A-2b is an export-time setting with
# no in-document equivalent, so it has to be injected here: typsphinx imports
# the module inside its compile function, which picks up this wrapper.
# font_paths is only needed when the families are not installed system-wide;
# docs.just fetches them into build/docs-fonts, CI installs them via fc-cache.
_local_fonts = _DOCS_DIR.parent / "build" / "docs-fonts"

#: Tells notebook kernels where the local docs font cache is, so
#: ``docs/ipython_config.py`` can register it with matplotlib.  Kernels inherit
#: this process's environment.
DOCS_FONTS_ENV = "QPDK_DOCS_FONTS_DIR"


def _register_docs_fonts_with_matplotlib():
    """Make the local docs font cache visible to matplotlib.

    Only the HTML CI job installs the docs fonts system-wide; the PDF job and
    local builds have nothing but ``build/docs-fonts``.  Without this, every
    figure falls back to DejaVu Sans and each plotting cell prints
    ``findfont: Font family 'Outfit' not found`` into its output, which the PDF
    then reproduces.  Registering here covers the ``.. plot::`` directive (it
    runs in this process); the environment variable covers notebook kernels.
    """
    if not _local_fonts.is_dir():
        return
    os.environ[DOCS_FONTS_ENV] = str(_local_fonts.resolve())
    for font in sorted(_local_fonts.iterdir()):
        if font.suffix in {".otf", ".ttf"}:
            font_manager.fontManager.addfont(str(font))


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


def _source_version():
    """Return the git tag at HEAD, else the short commit hash, else ``""``.

    Shown on the PDF cover so a downloaded copy can be traced back to the
    source it was built from.
    """

    def git(*args):
        try:
            out = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                ["git", *args],
                cwd=_DOCS_DIR,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return ""
        return out.strip()

    tags = git("tag", "--points-at", "HEAD", "--sort=-v:refname").splitlines()
    return tags[0] if tags else git("rev-parse", "--short=12", "HEAD")


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
            shutil.copy(_DOCS_DIR / "_static" / name, bundle / Path(name).name)
        (bundle / "version.txt").write_text(_source_version())


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


def setup(_app: Sphinx) -> None:
    """Register the extension."""
    _register_docs_fonts_with_matplotlib()
    typst.compile = _typst_compile_with_fonts
