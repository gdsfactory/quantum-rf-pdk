---
applyTo: "docs/**,**/*.md,**/*.rst"
---

# Documentation review instructions

Documentation is built with Sphinx (`just docs` for HTML, `just docs-pdf` for the typst PDF). **Both must build for a PR
to be mergeable**, so treat a change that can break the build as blocking.

## Markdown and RST

- Markdown is formatted by `mdformat --wrap=120` and linted by `markdownlint-cli2` (`.github/.markdownlint-cli2.yaml`,
  line length 120). Do not suggest formatting that these would undo.
- RST is formatted by `docstrfmt` and linted by `sphinx-lint`.
- Links are checked by `lychee` (`.github/lychee.toml`). A new external link must be reachable and stable; if it is
  inherently flaky or rejects automated requests, it belongs in the `exclude` list with a comment explaining why.
- Use the RST `:math:` role for math, not `$...$`.

## Bibliography (`docs/bibliography.bib`)

- Entries are normalized by `bibtex-tidy` (sorted, curly-braced, months abbreviated, duplicates flagged).
- If an entry has a valid `doi` field, it **must not** also carry `url` or `urldate` — that duplicate citation
  information is stripped deliberately.
- Prefer a DOI over a URL for anything published. Flag a new entry with neither.
- `codespell` skips this file, so watch for typos in author names and titles yourself.

## Content

- New public API should be reachable from the docs. Check that a new cell or model shows up through the templates in
  `docs/templates/` and the generators in `.github/write_cells.py` / `.github/write_models.py`.
- Adding or removing a notebook requires an update to `docs/notebooks.rst`.
- `docs/index.md`, `docs/changelog.md` and `docs/make_commands.md` use include/literalinclude directives and are
  excluded from mdformat and markdownlint — do not propose reformatting them.
- Keep `AGENTS.md` consistent with `pyproject.toml`: the `check-agents-version-ranges` hook fails when the Python or
  gdsfactory version ranges quoted in `AGENTS.md` drift from the real ones.
