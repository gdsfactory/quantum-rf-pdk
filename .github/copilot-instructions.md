# Copilot instructions for qpdk

## Repository overview

This is **qpdk**, a Python superconducting microwave Process Design Kit (PDK) built on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/) for designing quantum devices and circuits — transmons,
resonators, couplers, airbridges, SNSPDs, fluxoniums, unimons. It targets Python 3.12–3.14, uses `uv` as the package
manager, `just` as the task runner, and `prek` (a parallel pre-commit runner) for linting.

`AGENTS.md` at the repository root is the fuller contributor guide, including the coding-agent behavioural rules. This
file is the review-oriented summary; the detail for individual areas lives in the path-specific files listed below.

## Path-specific instructions

Detailed rules in `.github/instructions/` apply automatically to the areas below (via each file's `applyTo` frontmatter)
— consult them before commenting on a file they cover:

- `python.instructions.md` — all Python
- `cells.instructions.md` — `qpdk/cells/**`
- `tech-layers.instructions.md` — `qpdk/tech.py`, `qpdk/layers.yaml`, `qpdk/klayout/**`
- `models.instructions.md` — `qpdk/models/**`
- `simulation.instructions.md` — `qpdk/simulation/**`
- `samples.instructions.md` — `qpdk/samples/**`
- `tests.instructions.md` — `tests/**`
- `notebooks.instructions.md` — `notebooks/**`
- `docs.instructions.md` — `docs/**`, Markdown, RST
- `github-actions.instructions.md` — `.github/workflows/**`
- `dependencies.instructions.md` — `pyproject.toml`, `uv.lock`, `Dockerfile`, `.pre-commit-config.yaml`

## Build, test, and lint commands

All commands go through the `justfile` (which imports `tests/test.just` and `docs/docs.just`). Prefer `just` recipes
over direct tool invocation.

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

Pre-commit hooks **must** pass before every commit (`just run-pre`, or `uvx prek run --all-files`). They include `ruff`
(format + lint), `pyrefly` (type checking), `yamlfmt`, `yamllint`, `codespell`, `interrogate` (100% docstring coverage),
`markdownlint`, `mdformat`, `docstrfmt`, `sphinx-lint`, `actionlint`, `zizmor`, `hadolint`, `checkmake`, `bibtex-tidy`,
`typstyle`, `lychee`, `uv-lock`, and the shared `pdk-ci-workflow` structural checks.

## Project layout

```text
qpdk/                   Core Python package
  __init__.py            PDK object, version, public API
  cells/                Component definitions (transmons, resonators, …)
    __init__.py          Aggregates all cells via `from ... import *`
    derived/             Composite cells (e.g. transmon_with_resonator_and_probeline)
  models/               S-parameter and circuit models
    constants.py         Centralized physical constants (e, h, Φ_0, ε_0, …)
  simulation/           HFSS/Q3D automation (aedt_base, hfss, q3d)
  klayout/              KLayout technology files
  samples/              Example designs (.py, .pic.yml, .gsch)
  tech.py               Layer map, layer stack, cross sections
  layers.yaml           Layer views (must stay in sync with tech.py)
  logger.py             Centralized loguru logger
  config.py             Path configuration (PATH dataclass)
tests/                  pytest suite
  gds_ref/              GDS regression reference files
  test_pdk/             Settings and netlist regression YAML
  test_models_regression/ Model regression reference data
  models/               Model unit tests
docs/                   Sphinx documentation source
notebooks/src/          Jupytext notebook sources (.py percent format, .m for MATLAB)
pyproject.toml          Metadata, dependencies, ruff/pyrefly/pytest/interrogate config
justfile                Task runner recipes
.pre-commit-config.yaml Pre-commit hook definitions
.github/workflows/      CI pipelines (test, docs, build, release, drc)
```

## Review priorities

Comment on these in roughly this order, and stay quiet about the rest:

1. **Correctness of the physics.** Wrong units, a wrong formula, a non-passive passive model, or a layer on the wrong
   mask are the failures that reach silicon. They outrank every style concern here.
1. **Breaking changes to the public surface.** Renamed cells, renamed ports, changed port types, changed cross-section
   defaults, or changed layer numbers break user designs and stored netlists. Always call them out explicitly.
1. **Regression references.** A component or model change should come with regenerated references; a reference diff
   should come with a code change that explains it. Mismatches in either direction are worth a comment.
1. **Missing tests** for new public behaviour.
1. **Missing or wrong docstrings** on new public API — `interrogate` fails CI at anything under 100% coverage.
1. **Security**, per the rules in the path-specific files.

## Cross-cutting rules

- **Logging:** use `from qpdk import logger` (loguru), never `print()`, in library code.
- **Math in docstrings:** use the RST `:math:` role, never `$...$`.
- **Upright vs italic in math (ISO 80000-2):** italic is for variables only. Units and descriptive
  subscripts/superscripts are upright with `\text{}` — `\,\text{GHz}`, `E_\text{J}`, `C_\text{q}`, `Q_\text{ext}`.
  Numeric and mathematical indices stay italic — `S_{21}`, `T_1`, `f_{01}`, `\sum_k a_k`. Flag `\mathrm{}`/`\textrm{}`
  in new math (use `\text{}`), and an italic unit or descriptive script. `\mathtt{}` for Python identifiers is fine.
- **Physical constants:** import from `qpdk/models/constants.py`; never redefine locally.
- **Layers:** use the `LAYER` map from `qpdk/tech.py`; raw layer tuples are rejected by pre-commit.
- **Git LFS:** CSV data under `tests/models/data/` is LFS-tracked; GDS/OAS are binary per `.gitattributes`. Check new
  large or binary files are handled the same way.
- **Commit messages:** imperative mood ("Add component", not "Added component").
- **Release notes:** drafted automatically by release-drafter from PR titles and labels, so a PR title should read as a
  changelog entry on its own. `CHANGELOG.md` follows Keep a Changelog and is updated at release time, not per PR.
- **Secrets:** never approve a committed key, token, or licence file.

## What not to flag

Avoid noise on things the toolchain already owns or deliberately allows:

- Formatting that `ruff format`, `mdformat`, `yamlfmt`, `docstrfmt` or `typstyle` controls.
- Line length in Python (`E501` is disabled) — 120 characters is the Markdown/RST limit only.
- Unicode math identifiers such as `Φ_0`, `ε_0`, `μ_0`, `π` — intentional and rule-suppressed.
- Star imports in `qpdk/**/__init__.py` — they are how the cell registry is assembled.
- `assert`, missing docstrings, or `print()` in `tests/`, `qpdk/samples/`, and `notebooks/`.
- Any rule listed under `[tool.ruff.lint] ignore` in `pyproject.toml`.
- Purely stylistic rewrites of existing code that the PR did not touch.
