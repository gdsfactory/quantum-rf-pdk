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

## JAX over numpy

- Prefer `jax.numpy` (`jnp`) or `jax.*` over `numpy` in new or changed code, notably tests and notebooks. Flag a new
  `np.` call where a `jnp` equivalent would do.
- Do not flag `numpy` where it is required: `qpdk/` code that must import without jax (jax is not a core dependency),
  `np.testing`, file I/O (`np.loadtxt`, `np.savetxt`, `np.load`), `np.random.default_rng`, in-place array mutation, and
  arrays passed to tools that need real numpy (MATLAB, HFSS, COMSOL, scqubits, netket).

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

## Modernization

The supported range is `requires-python` in `pyproject.toml` (currently 3.12–3.14). Review new or changed code for
idioms that range already allows, and for ones it does not yet allow:

- **Available now — suggest it.** Ruff's `UP` (pyupgrade) rules already rewrite the mechanical cases — `Optional`/
  `Union`, `typing.List`/`Dict`, `typing` vs `collections.abc`, PEP 695 type parameters, dead `sys.version_info`
  branches — so do not repeat them. Flag what a linter cannot see: hand-rolled equivalents of `itertools.batched`,
  `itertools.pairwise`, `functools.cache`, `contextlib.chdir`, `typing.override`, `typing.Self`, or `match`; `os.path`
  string juggling where `pathlib` is clearer; and `typing_extensions` imports of names `typing` already has at the
  floor.

- **Blocked by the floor — ask for a marker.** When a newer Python would delete a workaround but the minimum version
  rules it out, the code should carry a one-line marker next to the workaround (two lines at most), as `AGENTS.md`
  requires:

  ```python
  # TODO(Python 3.13): @warnings.deprecated (PEP 702) - replaces this, and type checkers flag call sites statically.
  def deprecated(msg: str | Callable | None = None) -> Any: ...
  ```

  If a PR adds or touches such a workaround without the marker, suggest the exact comment, with the version, the feature
  (and PEP where there is one), and what to change. `rg "TODO\(Python"` is how the cleanups are found when the floor
  moves, so an unmarked workaround is one nobody removes.

- **Check existing markers.** Flag a `TODO(Python 3.X)` whose version is at or below the current floor — the upgrade is
  no longer blocked and should be done, not left as a comment. Flag a malformed marker (missing version, feature, or
  action) and one that cites a feature from the wrong release.

- **Do not suggest syntax above the floor.** Code must run on every version in the range; suggesting e.g. PEP 810
  `lazy import` or PEP 758 unparenthesized `except` in live code is a bug while 3.12 is supported. Do not ask for a
  marker on a swap that would change behaviour — for example a deferred optional import wrapped in
  `try/except ModuleNotFoundError`, where `lazy import` would move the failure past the handler.

Keep these comments low priority: one comment per pattern, never above correctness, breaking-change, or test findings,
and only on code the PR adds or touches.

## Security

- `flake8-bandit` (`S`) is enabled. Flag `subprocess` calls built from untrusted input, `eval`/`exec`, and hard-coded
  secrets. `S404`/`S607`/`PLW1510` are deliberately ignored — do not flag those.
- Never approve committed private keys, API tokens, or customer fabrication parameters.
