---
applyTo: "qpdk/simulation/**/*.py,notebooks/src/hfss*.py,tests/hfss/**/*.py"
---

# Simulation automation review instructions

QPDK does not wrap Ansys AEDT. The HFSS notebooks and `tests/hfss/` use `gplugins.ansys` (which drives `pyaedt`)
directly, with `qpdk.simulation.layout.prepare_metal_layout` for geometry and `qpdk.tech.material_properties` passed as
`material_properties`. Fix generic AEDT behaviour in gplugins, not here, and do not add QPDK subclasses, re-exports or
default-wrappers around gplugins classes. `qpdk/simulation/` holds the shared layout preparation and the COMSOL and
MATLAB helpers. None of the solver code runs in normal CI — the `hfss` pytest marker gates the AEDT tests and
`just test-hfss` needs a licensed local install — so review has to carry more weight than usual here.

## What to check

- `import qpdk` and `import qpdk.simulation` must work without the `hfss` extra, so nothing under `qpdk/` imports
  `gplugins.ansys` or `pyaedt`. Ruff does not check this: its `require-lazy` setting only covers PEP 810 `lazy import`
  statements, so review has to. Tests and notebooks import `gplugins.ansys` directly; the test environment installs the
  `hfss` extra, so do not guard them with `pytest.importorskip`.
- AEDT sessions and projects are external resources. Verify they are released on every path — prefer a context manager
  or `try`/`finally` over a bare `close()` at the end of a happy path.
- Non-graphical mode must stay the default for anything that could run unattended. Flag a hard-coded
  `non_graphical=False`.
- No hard-coded local paths, machine names, licence servers, or Windows-only path separators. Use `pathlib` and
  configurable arguments.
- Solver settings that materially change results (mesh refinement, adaptive passes, convergence deltas, frequency sweep
  type and range) should be explicit named parameters with documented defaults, not magic numbers buried in the body.
- Units are a recurring source of error across the pyaedt boundary — check that geometry passed in microns is not
  silently interpreted as millimetres, and that frequencies carry their unit strings.

## COMSOL

The generic COMSOL code (polygon extraction, sheet and metal builders, RF and electrostatic studies, mesh, plotting, and
result helpers, and the `COMSOL` model class) lives in `gplugins.comsol`, and the notebooks import it from there
directly. `qpdk/simulation/comsol.py` is a single module that keeps only the QPDK-specific inputs:

- `prepare_comsol_layout` inverts the M1_ETCH mask into M1_DRAW metal with a ground margin, rejects unsupported layers,
  defaults the feeds to `coupling_o1`/`coupling_o2`, and then calls the gplugins extraction.
- `SILICON_RELATIVE_PERMITTIVITY` and `AIR_RELATIVE_PERMITTIVITY` come from `qpdk.tech`; callers pass them to
  `COMSOL.create_sheet` explicitly.

Flag re-exports, subclasses, or wrappers of gplugins COMSOL APIs added here, and flag generic COMSOL logic: it belongs
in `gplugins.comsol`. `qpdk/simulation/__init__.py` must not import the COMSOL module, so `qpdk.simulation` stays
importable without the `comsol` extra.

## Tests

- COMSOL tests live in `tests/comsol/` and cover only the QPDK side; the generic behaviour is tested in gplugins.
- New AEDT workflow code needs a test marked `@pytest.mark.hfss` in `tests/hfss/`, even though it will be skipped in CI.
- Pure logic (setup construction, result parsing, unit conversion) should be factored out and tested **without** AEDT so
  that something is actually exercised in CI. Flag a change where all new logic is unreachable without a licence.
