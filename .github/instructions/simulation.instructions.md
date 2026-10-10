---
applyTo: "qpdk/simulation/**/*.py"
---

# Simulation automation review instructions

`qpdk/simulation/` drives Ansys HFSS and Q3D through `pyaedt` (`aedt_base.py`, `hfss.py`, `q3d.py`). None of this runs
in normal CI — the `hfss` pytest marker gates it and `just test-hfss` needs a licensed local install — so review has to
carry more weight than usual here.

## What to check

- `pyaedt` and `polars` are in ruff's `require-lazy` list. Import them inside functions, never at module level, so
  `import qpdk` works without the `hfss` extra installed.
- Guard the optional import with an actionable message pointing at `uv sync --extra hfss`, rather than letting a bare
  `ModuleNotFoundError` escape.
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
result helpers, and the `COMSOL` model class) lives in `gplugins.comsol`. `qpdk/simulation/comsol/` keeps only thin
wrappers:

- `layout.py` inverts the M1_ETCH mask into M1_DRAW metal with a ground margin, rejects unsupported layers, defaults the
  feeds to `coupling_o1`/`coupling_o2`, and then calls the gplugins extraction.
- `sheet.py`, `metal.py`, and `model.py` only change defaults (the technology permittivities, the model name).
- `rf.py`, `capacitance.py`, `mesh.py`, `plotting.py`, and `results.py` are plain re-exports.

Flag generic COMSOL logic added here; it belongs in `gplugins.comsol`. `gplugins` is in ruff's `require-lazy` list, so
`qpdk/simulation/__init__.py` must keep reaching the COMSOL modules through `_LAZY_IMPORTS`, never a module-level
import.

## Tests

- COMSOL wrapper tests live in `tests/comsol/` and cover only the QPDK side; the generic behaviour is tested in
  gplugins.
- New simulation code needs a test marked `@pytest.mark.hfss` in `tests/hfss/test_hfss.py`, even though it will be
  skipped in CI.
- Pure logic (setup construction, result parsing, unit conversion) should be factored out and tested **without** AEDT so
  that something is actually exercised in CI. Flag a change where all new logic is unreachable without a licence.
