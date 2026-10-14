---
applyTo: "qpdk/samples/**"
---

# Sample and schematic review instructions

`qpdk/samples/` holds runnable example designs: full test chips, routing demos, simulation walkthroughs, and the
numbered `sample0.py`–`sample6.py` primers. Samples are user-facing documentation — readability matters more here than
anywhere else in the package.

## Python samples

- `qpdk/samples/*.py` is exempt from `D100`, `T201` and `PLW0717`, and is excluded from `pyrefly` — do not flag a
  missing module docstring, `print()`, or missing type hints.
- A sample is still expected to run end to end. Flag code that cannot execute as written (undefined names, a component
  that was renamed, a missing `PDK.activate()`).
- Comment the *design intent* and the physics, not the gdsfactory mechanics. A reader should learn why the chip is laid
  out this way.
- Samples are tested to produce valid GDS (`tests/test_pdk.py`). A new sample should be reachable from that coverage.
- Prefer composing PDK cells over drawing raw geometry — a sample that reaches for polygons is usually a sign a cell is
  missing.

## Schematic sources (`.gsch`)

- `.gsch` files are the source of truth for schematic samples. Do not check in generated YAML fallbacks.
- Python samples use `qpdk.read.from_gsch`; simulation tests use `qpdk.read.read_gsch` for an in-memory netlist.
- Run `just fetch-gfp` or set `GFP_BIN` to the converter bundled with GDSFactory+ before building these samples.
- Instance and port names must match the cell definitions in `qpdk/cells/`.
