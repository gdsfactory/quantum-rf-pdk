"""Rewrite docs sources and docstrings that would not otherwise render correctly.

Covers ``$``-delimited math in docstrings, the README splice into the index page
(image paths and relative links), and the *Edit on GitHub* link of notebook pages.
"""

import re
from pathlib import Path

from sphinx.application import Sphinx


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


def _rewrite_readme_links(content: str, srcdir: Path, html_context: dict) -> str:
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
        html_context: The ``html_context`` config value, which names the
            GitHub repository and branch to link other files to.

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
            content = _rewrite_readme_links(
                content, Path(app.srcdir), app.config.html_context
            )
            source[0] = re.sub(
                r"```\{include\}\s+\.\./README\.md.*?```",
                lambda _: content,
                source[0],
                flags=re.DOTALL,
            )

    source[0] = source[0].replace("docs/_static/images/", "/_static/images/")


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


def _dollar_math_handler(_app, _what, _name, _obj, _options, lines):
    """Convert ``$``-delimited math in an autodoc docstring to RST roles."""
    _dollar_math_to_rst(lines)


def setup(app: Sphinx) -> None:
    """Register the extension."""
    # Earlier than the default 500 so the README splice lands before
    # sphinx_github_alerts' source-read hook, which then also converts the
    # README's `> [!NOTE]` alert instead of leaving it as a literal quote.
    app.connect("source-read", replace_image_paths, priority=400)
    # Convert $-delimited math before any other processing
    app.connect("autodoc-process-docstring", _dollar_math_handler, priority=100)
    # Fix Edit on GitHub URLs for notebook pages (runs after pydata-sphinx-theme's
    # setup_edit_url which is registered at the default priority of 500)
    app.connect("html-page-context", fix_notebook_edit_url, priority=600)
