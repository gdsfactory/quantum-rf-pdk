---
applyTo: "pyproject.toml,uv.lock,Dockerfile,.pre-commit-config.yaml"
---

# Dependency and configuration review instructions

## `pyproject.toml` and `uv.lock`

- The two must stay in sync. `uv-lock` runs in pre-commit; a `pyproject.toml` dependency change with no `uv.lock` diff
  (or vice versa) will fail CI.
- `uv.lock` is generated. Flag hand edits, and do not review its contents line by line — check only that the diff is
  plausibly the result of the declared change and that nothing unrelated was regenerated wholesale.
- Runtime dependencies are deliberately minimal (`gdsfactory`, `typing-extensions`). **A new runtime dependency needs
  justification**; anything used only by models, simulation, or docs belongs in an optional extra (`models`, `hfss`,
  `scqubits`, `netket`, `qutip`, `graphics`, `circulax`, `pymablock`, `ray`, `gdsfactoryplus`) or a dependency group
  (`dev`, `lint`, `test`, `docs`, `docs-models`, `stubs`, `github`).
- A package added to an extra that is heavy or optional should usually also be added to ruff's
  `[tool.ruff.lint.flake8-tidy-imports] require-lazy` list, so it stays lazily imported.
- `gdsfactory` is pinned with a compatible-release specifier (`~=`). Bumping it commonly shifts GDS regression
  references — check the PR regenerates them and says so.
- Version numbers are checked by the `check-version-sync` hook: `pyproject.toml`, `qpdk/__init__.py` and `README.md`
  must agree. Use `uv version --bump patch` rather than editing by hand.
- `check-pyproject-sections` from `pdk-ci-workflow-public` enforces the shared PDK section layout — do not reorder or
  drop sections such as `[tool.gdsfactoryplus]`.
- Prefer widening a constraint over pinning exactly; prefer an upper bound only where a known incompatibility exists,
  and note the reason in a comment.

## `.pre-commit-config.yaml`

- Hook repos are pinned by `rev`. Bumps should come from `just update-pre` (or Dependabot), not hand-edited.
- Every hook carries a `priority` comment convention documented at the top of the file (10 → 50). A new hook needs the
  right priority band, or it will run before the formatter that fixes its input.
- Removing or excluding a hook weakens CI for the whole repo. Ask for the reason.

## `Dockerfile`

- Linted by `hadolint`. Pin base image tags, combine `RUN` layers where sensible, and clean package manager caches in
  the same layer.
- No secrets, tokens, or licence files baked into an image layer.
