---
applyTo: "qpdk/simulation/**/*.py"
---

# Simulation automation review instructions

`qpdk/simulation/` drives Ansys HFSS and Q3D through `gplugins.ansys`, which wraps `pyaedt`. The generic AEDT code lives
in gplugins; `aedt_base.py`, `hfss.py` and `q3d.py` are thin wrappers that add QPDK defaults (layer stack, materials,
singleton classes) and QPDK-only helpers. Fix generic AEDT behaviour in gplugins, not here. None of this runs in normal
CI — the `hfss` pytest marker gates it and `just test-hfss` needs a licensed local install — so review has to carry more
weight than usual here.

## What to check

- `import qpdk` and `import qpdk.simulation` must work without the `hfss` extra. `qpdk.simulation` and `aedt_base.py`
  expose the AEDT names lazily through `__getattr__`; keep it that way. `hfss.py`, `q3d.py` and `_aedt.py` subclass
  gplugins classes, so they import gplugins at module level, which is fine only because nothing imports them eagerly.
  Anywhere else, import `gplugins`, `pyaedt` and `polars` inside functions. Ruff does not check this: its `require-lazy`
  setting only covers PEP 810 `lazy import` statements, so review has to.
- Guard every gplugins import with an `ImportError` carrying `aedt_base.MISSING_HFSS_EXTRA`, which points at
  `uv sync --extra hfss`, rather than letting a bare `ModuleNotFoundError` escape.
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

## Tests

- New simulation code needs a test marked `@pytest.mark.hfss` in `tests/hfss/`, even though it will be skipped in CI.
- Pure logic (setup construction, result parsing, unit conversion) should be factored out and tested **without** AEDT so
  that something is actually exercised in CI. Flag a change where all new logic is unreachable without a licence.
