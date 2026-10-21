# VACASK raw-file fixtures

Small SPICE raw files written by VACASK 0.3.4 for the reader tests in `tests/test_vacask.py`. They are generated from
`fixtures.sim` with `josephson_junction.va` from `qpdk/simulation/vacask_models/` next to it.

| File                      | Content                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| `tran1.raw`, `tran1a.raw` | Real transient analysis, binary and ASCII                             |
| `ac1.raw`, `ac1a.raw`     | Complex AC analysis, binary and ASCII                                 |
| `acsw.raw`, `acswa.raw`   | AC analysis swept over the bias current, binary and ASCII             |
| `multi.raw`               | `ac1.raw` followed by `acsw.raw`, to test files holding several plots |

The ASCII files come from the same netlist with `rawfile="ascii"` added to the `options` line and the analyses renamed
with an `a` suffix.
