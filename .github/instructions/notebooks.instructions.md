---
applyTo: "notebooks/**"
---

# Notebook review instructions

Notebooks are generated artifacts. The **source of truth is `notebooks/src/`**: jupytext `.py` files in percent format
(`# %%` cell markers), plus `notebooks/src/matlab_integration.m` for the MATLAB kernel (`% %%` markers). The `.ipynb`
files at `notebooks/` are produced by `just convert-notebooks` (via the `convert-notebooks` pre-commit hook).

## What to flag

- **A hand-edited `.ipynb` with no matching change in `notebooks/src/`.** The next hook run will overwrite it. The
  `check-notebook-sources` hook also fails on an `.ipynb` with no source file.
- A new notebook source that is not added to `docs/notebooks.rst`. That page describes each notebook and groups it by
  simulation approach; adding or removing a notebook requires updating it.
- A notebook needing external tooling (HFSS, MATLAB) that is not listed in `nb_execution_excludepatterns` in
  `docs/conf.py` — the docs build will try to execute it and fail.
- A missing first cell with the Google Colab install snippet tagged `tags=["hide-input", "hide-output"]`, so the
  notebook runs online without cluttering the rendered docs.
- Committed cell outputs. `nbstripout` strips them, except for the pre-executed HFSS notebooks (`hfss_driven_capacitor`,
  `hfss_eigenmode_resonator`, `hfss_q2d_cpw_impedance`), which intentionally keep theirs.

## Style

- `notebooks/**/*.py` is exempt from `D100`, `T201` and `E402` — do not flag a missing module docstring, `print()`, or
  imports after `PDK.activate()`.
- Comments in `notebooks/src/` are reflowed by the `matlab-reflow-comments` hook at 100 characters. Structural comments
  must keep an inner indent of either zero or two-or-more spaces, or the hook will join adjacent lines.
- Notebooks are documentation: prefer prose markdown cells explaining the physics over bare code, and keep runtimes
  short since the docs build executes them.
