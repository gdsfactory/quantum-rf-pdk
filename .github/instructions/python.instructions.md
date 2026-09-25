---
applyTo: "**/*.py"
---

# Python code review instructions

Applies to all Python in the repository. See also the path-specific files for `qpdk/cells/`, `qpdk/models/`,
`qpdk/simulation/`, `qpdk/samples/`, `tests/`, and `notebooks/`.

## Style and linting

- Lint and format are enforced by `ruff` (config in `pyproject.toml`, `preview = true`). Do not suggest changes that
  `ruff format` would undo, and do not suggest re-enabling rules listed under `[tool.ruff.lint] ignore`.

- Line length is not enforced (`E501` ignored), but keep new lines under 120 characters for readability.

- Unicode math identifiers (`Φ_0`, `ε_0`, `μ`, `π`) are intentional — `RUF001`/`RUF002`/`RUF003`/`PLC2401` are disabled.
  Do not flag them.

- Flag `print()` in library code. Use the centralized loguru logger instead:

  ```python
  from qpdk import logger

  logger.info("Writing GDS to {}", path)  # good
  print("Writing GDS")  # bad — T20 ruff violation outside samples/notebooks/tests
  ```

- Imports belong at module top level. The only exception is a dependency listed in `require-lazy` (see below) or one
  that is genuinely absent from `pyproject.toml`.

## Typing

- New public functions and methods need type hints. `pyrefly` runs in pre-commit; `mypy` is configured `strict`.
- Prefer precise types over `Any`. Use `cast()` only when there is no alternative, and say why in a comment.
- Flag `# type: ignore` / `# pyrefly: ignore` added without a justification comment.

## Docstrings

- Google-style docstrings are required on all public modules, classes, and functions. `interrogate` runs at
  `fail-under = 100`, so a missing docstring is a CI failure, not a nit. (`qpdk/**/__init__.py`, `qpdk/samples/`,
  `tests/`, and `notebooks/` are excluded.)

- Use the RST `:math:` role for inline math — never `$...$` or `$$...$$`, which Sphinx will not render:

  ```python
  """Return the resonator frequency :math:`f_0 = 1 / (2 \\pi \\sqrt{LC})`."""
  ```

- For physics-bearing code, the docstring should explain the device behaviour and its parameters, with a citation into
  `docs/bibliography.bib` where one exists.

- Set non-variables upright, per ISO 80000-2. Italic is for variables; units and descriptive subscripts/superscripts use
  `\text{}`. Numeric and mathematical indices (`S_{21}`, `T_1`, `f_{01}`, `\sum_k a_k`) stay italic. Prefer `\text{}`
  over `\mathrm{}`/`\textrm{}`; `\mathtt{}` is for literal Python identifiers.

  ```python
  """Resonator with :math:`f_\\text{r} = 6.5\\,\\text{GHz}` and :math:`Q_\\text{ext}` set by the coupler."""
  ```

- The same rule applies to math in matplotlib labels and in notebook sources under `notebooks/src/`. Those need a raw
  string — `r"..."` or `rf"..."` — because `"\text{}"` in a plain string silently becomes a tab character. In an
  f-string, literal braces must be doubled: `rf"$f_\text{{r}} = {f / 1e9:.2f}\,\text{{GHz}}$"`.

## Lazy imports of heavy/optional dependencies

These packages must only be imported inside the function that uses them — ruff's `flake8-tidy-imports.require-lazy`
enforces it: `trimesh`, `polars`, `pyaedt`, `gplugins`, `jaxellip`, `optax`, `optuna`, `sax`, `sympy`, `pandas`,
`netket`, `flax`, `pymablock`, `qutip-jax`, `qutip-qip`, `scqubits`.

```python
def simulate() -> None:
    import sax  # good — lazy, keeps `import qpdk` fast and optional deps optional
```

Flag any of them appearing in a module-level import block. Guard the import behind a clear error message if the extra
may not be installed (`uv sync --extra models`).

## Security

- `flake8-bandit` (`S`) is enabled. Flag `subprocess` calls built from untrusted input, `eval`/`exec`, and hard-coded
  secrets. `S404`/`S607`/`PLW1510` are deliberately ignored — do not flag those.
- Never approve committed private keys, API tokens, or customer fabrication parameters.
