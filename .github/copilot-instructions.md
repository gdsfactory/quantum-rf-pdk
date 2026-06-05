# Copilot Custom Review Instructions for qpdk

## Repository Overview

This is **qpdk**, a Python-based superconducting microwave Process Design Kit (PDK) built on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/) for designing quantum devices and circuits (transmons,
resonators, couplers, airbridges, SNSPDs, fluxoniums, unimons, etc.). It targets Python 3.12–3.13, uses `uv` as the
package manager, `just` as the task runner, and `prek` (a parallel pre-commit runner) for linting.

## Build, Test, and Lint Commands

All commands use the `justfile` (with imports from `tests/test.just` and `docs/docs.just`). Always prefer `just`
commands over direct tool invocation.

| Task                                     | Command                   |
| ---------------------------------------- | ------------------------- |
| Install dependencies                     | `just install`            |
| Run full test suite                      | `just test`               |
| Run GDS regression tests only            | `just test-gds`           |
| Run GDS tests, stop at first failure     | `just test-gds-fail-fast` |
| Regenerate GDS reference files           | `just test-gds-force`     |
| Run model regression tests               | `just test-models`        |
| Regenerate model reference files         | `just test-models-force`  |
| Run HFSS simulation tests                | `just test-hfss`          |
| Run GDSFactory+ tests                    | `just test-gfp`           |
| Run pre-commit hooks (lint + format)     | `just run-pre`            |
| Build HTML documentation                 | `just docs`               |
| Build PDF documentation                  | `just docs-pdf`           |
| Build package                            | `just build`              |
| Show/preview a component interactively   | `just show`               |
| Run everything (test, lint, build, docs) | `just all`                |

Pre-commit hooks **must** pass before every commit. They include `ruff` (format + lint), `pyrefly` (type checking),
`yamlfmt`, `yamllint`, `codespell`, `interrogate` (docstring coverage), `markdownlint`, `mdformat`, `actionlint`,
`zizmor` (GitHub Actions security), `hadolint` (Dockerfile), `checkmake` (Makefile), `bibtex-tidy`, `sphinx-lint`,
`lychee` (link checking), `uv-lock`, `pdk-ci-workflow` structural checks, and more. Run with `just run-pre` or
`uvx prek run --all-files`.

## Project Layout

```text
qpdk/                   Core Python package
  __init__.py            PDK object, version, public API
  cells/                Component definitions (transmons, resonators, …)
    __init__.py          Aggregates all cells via `from ... import *`
    derived/             Composite/derived cells (e.g., transmon_with_resonator_and_probeline)
  models/               S-parameter and circuit models
    constants.py         Centralized physical constants (e, h, Φ_0, ε_0, …)
    math.py              Mathematical utilities for models
    perturbation.py      Perturbation theory models
    qubit.py             Qubit Hamiltonian and frequency models
  simulation/           HFSS/Q3D simulation automation (aedt_base, hfss, q3d)
  klayout/              KLayout technology files
  config.py             Path configuration (PATH dataclass)
  helper.py             Helper utilities
  utils.py              General utility functions
  tech.py               Layer stack, cross sections, LAYER enum
  layers.yaml           Layer definitions (must stay in sync with tech.py)
  logger.py             Centralized loguru logger
  samples/              Example layout and simulation scripts (.py, .pic.yml, .scm.yml)
tests/                  pytest test suite
  gds_ref/              GDS regression reference files
  models/               Model unit tests
  test_models_regression/ Model regression reference data
  helper/               Test helper utilities
  test_pdk.py           Component regression + netlist round-trip tests
  test.just             Test-related just recipes
docs/                   Sphinx documentation source
  docs.just             Documentation-related just recipes
notebooks/src/          Jupytext notebook sources (.py percent format, .m for MATLAB)
pyproject.toml          Project metadata, dependency specs, ruff/pyrefly/pytest config
justfile                Task runner recipes (imports test.just and docs.just)
.pre-commit-config.yaml Pre-commit hook definitions
.github/workflows/      CI pipelines (test, docs, build, release)
.changelog.d/           Towncrier changelog fragments
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
- Optional heavy dependencies (sax, scqubits, jaxellip, polars, netket, flax, pymablock, qutip-jax, qutip-qip, optax,
  optuna, sympy, pandas, trimesh, gplugins, pyaedt) must be **lazily imported** — ruff's `require-lazy` is configured
  for them in `pyproject.toml`.

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
- HFSS/Q3D simulation automation lives in `qpdk/simulation/` — changes there should be tested with `just test-hfss` when
  HFSS is available.

### Testing

- New components must have **GDS regression tests** (settings + netlists). Reference files are generated with
  `just test-gds-force` and committed in `tests/gds_ref/`.
- New models must have **model regression tests**. Reference files are generated with `just test-models-force` and
  committed in `tests/test_models_regression/`.
- Prefer the **hypothesis** library for property-based tests. When using it:
  - Do **not** combine `@given` with `@staticmethod` (causes `AttributeError` during collection).
  - Add `@settings(deadline=None)` when testing JAX JIT-compiled code.
- The full test suite runs across Python 3.12–3.13 on Ubuntu, macOS, and Windows. Verify no platform-specific
  assumptions.
- If GDS reference files changed, verify the diff is intentional and corresponds to the code change.
- Tests run in parallel using `pytest-xdist` (`-n auto`). Verify no test interdependencies.

### Documentation

- Docstrings should explain the underlying quantum physics and design parameters, with citations where appropriate.
- Bibliography entries in `docs/bibliography.bib`: if a `doi` field is present, do **not** include `url` or `urldate`.
- Notebooks live in `notebooks/src/` as jupytext `.py` (percent format) or `.m` (MATLAB) files. When adding or removing
  notebooks, update `docs/notebooks.rst`.
- Notebooks requiring external tools (HFSS, MATLAB) should be listed in `nb_execution_excludepatterns` in `docs/conf.py`
  and committed as pre-executed `.ipynb` files in `notebooks/`.
- Documentation must build successfully (`just docs`) for a PR to be mergeable.

### Git and CI

- **Git LFS**: CSV files in `tests/models/data/` are LFS-tracked. GDS/OAS files are binary (`.gitattributes`). Verify
  new binary or large data files are handled correctly.
- Commit messages use **imperative mood** (e.g., "Add component", not "Added component").
- All CI checks must pass: pre-commit hooks, the full pytest suite, and the documentation build.
- **Changelog**: Use [towncrier](https://towncrier.readthedocs.io/) fragments in `.changelog.d/` for user-facing
  changes. Fragment filenames follow the pattern `<issue_or_pr>.<type>.md`.

### Security and Supply Chain

- No private keys or sensitive fabrication parameters committed (the `detect-private-key` pre-commit hook helps, but
  reviewers should also verify).
- New dependencies should be justified. Check for known vulnerabilities and verify they are pinned appropriately in
  `pyproject.toml`.
- GitHub Actions workflows are checked by `actionlint` and `zizmor` for security best practices.
