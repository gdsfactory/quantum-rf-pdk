# Copilot Custom Review Instructions for qpdk

## Repository Overview

This is **qpdk**, a Python-based superconducting microwave Process Design Kit (PDK) built on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/) for designing quantum devices and circuits (transmons,
resonators, couplers, airbridges, SNSPDs, etc.). It targets Python 3.12–3.13, uses `uv` as the package manager, and
`just` as the task runner.

## Build, Test, and Lint Commands

All commands use the `justfile`. Always prefer `just` commands over direct tool invocation.

| Task                                 | Command               |
| ------------------------------------ | --------------------- |
| Install dependencies                 | `just install`        |
| Run full test suite                  | `just test`           |
| Run GDS regression tests only        | `just test-gds`       |
| Regenerate GDS reference files       | `just test-gds-force` |
| Run pre-commit hooks (lint + format) | `just run-pre`        |
| Build documentation (HTML + PDF)     | `just docs`           |
| Build package                        | `just build`          |

Pre-commit hooks **must** pass before every commit. They include `ruff` (format + lint), `pyrefly` (type checking),
`yamlfmt`, `codespell`, `interrogate` (docstring coverage), `markdownlint`, `actionlint`, and more. Run with
`just run-pre` or `uvx prek run --all-files`.

## Project Layout

```text
qpdk/                   Core Python package
  cells/                Component definitions (transmons, resonators, …)
    __init__.py          Aggregates all cells via `from ... import *`
  models/               S-parameter and circuit models
    constants.py         Centralized physical constants (e, h, Φ_0, ε_0, …)
  klayout/              KLayout technology files
  tech.py               Layer stack, cross sections, LAYER enum
  layers.yaml           Layer definitions (must stay in sync with tech.py)
  logger.py             Centralized loguru logger
  samples/              Example layout and simulation scripts
tests/                  pytest test suite
  gds_ref/              GDS regression reference files
  models/               Model unit tests
  test_pdk.py           Component regression + netlist round-trip tests
docs/                   Sphinx documentation source
notebooks/src/          Jupytext notebook sources (.py percent format)
pyproject.toml          Project metadata, dependency specs, ruff/pyrefly/pytest config
justfile                Task runner recipes
.pre-commit-config.yaml Pre-commit hook definitions
.github/workflows/      CI pipelines (test, docs, build, release)
```

## Review Checklist — What to Verify on Every PR

### Code Style and Quality

- Python code is formatted and linted by **ruff** (config in `pyproject.toml`). Verify no lint violations are
  introduced.
- **Docstrings** follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and
  are required on all public modules, classes, and functions (`interrogate` enforced at 100% coverage). Verify new
  public APIs have docstrings.
- Use the RST `:math:` role for inline LaTeX in docstrings — never raw `$` or `$$`.
- Use `from qpdk import logger` (loguru) instead of `print()` for runtime output. The `T20` ruff rule forbids print in
  library code.
- Unicode math identifiers (Φ, ε, μ, π) are acceptable — the relevant ruff rules are suppressed.
- Type hints are expected. `pyrefly` runs in pre-commit. Verify new code is typed.
- Optional heavy dependencies (sax, scqubits, jaxellip, polars, etc.) must be **lazily imported** — ruff's
  `require-lazy` is configured for them in `pyproject.toml`.

### Component (Cell) Changes

- Every new cell **must** use the `@gf.cell` decorator with a `tags` parameter for categorization (e.g.,
  `@gf.cell(tags=("qubits", "transmons"))`).
- New cell modules **must** be re-exported via a `from qpdk.cells.<module> import *` line in `qpdk/cells/__init__.py`.
- Layer assignments **must** use the `LAYER` enum from `qpdk/tech.py`. Verify no ad-hoc layer tuples are introduced.
- Layer definitions in `qpdk/layers.yaml` and `qpdk/tech.py` **must** stay in sync — any change to one requires a
  matching change in the other.
- Components must produce valid netlists that can be round-tripped (component → netlist → component). This is
  automatically tested by `test_netlists` in `tests/test_pdk.py`.

### Model and Simulation Changes

- Prefer **JAX-compatible** functions (`jnp` over `np`, `jaxellip` for elliptic integrals) and use
  `@partial(jax.jit, inline=True)` for helper functions.
- Physical constants **must** come from `qpdk/models/constants.py` — verify no local redefinitions of `e`, `h`, `Φ_0`,
  `ε_0`, etc.
- New models need unit tests in `tests/models/` verifying behavior, passivity, and reciprocity.

### Testing

- New components must have **GDS regression tests** (settings + netlists). Reference files are generated with
  `just test-gds-force` and committed in `tests/gds_ref/`.
- Prefer the **hypothesis** library for property-based tests. When using it:
  - Do **not** combine `@given` with `@staticmethod` (causes `AttributeError` during collection).
  - Add `@settings(deadline=None)` when testing JAX JIT-compiled code.
- The full test suite runs across Python 3.12–3.13 on Ubuntu, macOS, and Windows. Verify no platform-specific
  assumptions.
- If GDS reference files changed, verify the diff is intentional and corresponds to the code change.

### Documentation

- Docstrings should explain the underlying quantum physics and design parameters, with citations where appropriate.
- Bibliography entries in `docs/bibliography.bib`: if a `doi` field is present, do **not** include `url` or `urldate`.
- Notebooks live in `notebooks/src/` as jupytext `.py` (percent format) files. When adding or removing notebooks, update
  `docs/notebooks.rst`.
- Documentation must build successfully (`just docs`) for a PR to be mergeable.

### Git and CI

- **Git LFS**: CSV files in `tests/models/data/` are LFS-tracked. GDS/OAS files are binary (`.gitattributes`). Verify
  new binary or large data files are handled correctly.
- Commit messages use **imperative mood** (e.g., "Add component", not "Added component").
- All CI checks must pass: pre-commit hooks, the full pytest suite, and the documentation build.

### Security and Supply Chain

- No private keys or sensitive fabrication parameters committed (the `detect-private-key` pre-commit hook helps, but
  reviewers should also verify).
- New dependencies should be justified. Check for known vulnerabilities and verify they are pinned appropriately in
  `pyproject.toml`.
