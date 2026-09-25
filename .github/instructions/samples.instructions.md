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

## Schematic and netlist sources (`.gsch`, `.pic.yml`)

- A sample may exist in three linked forms: a Python script, a `.pic.yml` layout netlist, and a `.gsch` schematic. The
  `check-schematic-sources` pre-commit hook converts the `.gsch` and asserts it matches its `.pic.yml` fallback.
- **When one of the pair changes, the other must change with it.** A `.pic.yml` edited alone (or vice versa) fails the
  hook. Flag it.
- `.pic.yml` files are excluded from the `check-yaml` hook, so a syntax error there will not be caught until the test
  suite runs. Read new YAML carefully: instance names, port references, and routing settings.
- Instance and port names in the netlist must match the cell definitions in `qpdk/cells/`. A renamed port in a cell is a
  breaking change for every sample netlist that references it.
