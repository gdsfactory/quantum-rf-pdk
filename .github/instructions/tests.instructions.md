---
applyTo: "tests/**"
---

# Test review instructions

## General

- Tests run in parallel with `pytest-xdist` (`-n auto`). Flag anything order-dependent: shared module-level mutable
  state, reliance on another test having run, or writes to a fixed path outside `tmp_path`.
- The suite runs on Python 3.12–3.14 across Ubuntu, macOS and Windows. Flag hard-coded POSIX paths, `/tmp`, shell
  invocations, or line-ending assumptions. Use `pathlib` (ruff `PTH` is enabled) and the `skip_windows` marker when a
  test genuinely cannot run there.
- Use the registered markers rather than inventing new ones: `gfp` (needs GDSFactory+), `hfss` (needs HFSS),
  `skip_windows`.
- `tests/**` is exempt from `S101`, `D102`, `D103`, and `T20` — do not flag bare `assert`, missing docstrings, or
  `print` in tests.

## Hypothesis

Property-based tests are preferred where inputs are generic. Two gotchas that have bitten this repo before:

- Do **not** combine `@given` with `@staticmethod`. Hypothesis inspects `__code__` during collection and raises
  `AttributeError` on the staticmethod descriptor.
- Add `@settings(deadline=None)` to any `@given` test that calls JAX JIT-compiled code — first-call compilation overhead
  otherwise produces flaky `DeadlineExceeded` failures.

```python
@settings(deadline=None)
@given(width=st.floats(min_value=1.0, max_value=50.0))
def test_cpw_impedance(width: float) -> None: ...
```

## Regression tests

- GDS/settings/netlist references live in `tests/gds_ref/` and `tests/test_pdk/`; regenerate with `just test-gds-force`.
- Model references live in `tests/test_models_regression/`; regenerate with `just test-models-force`.
- Regenerated references must be committed in the same PR as the code change that caused them.
- **Never accept a regenerated reference as a fix for a failing test without an explanation.** Ask what physically or
  geometrically changed. A reference diff that is larger than the code change suggests an unintended side effect (these
  files also legitimately shift on a gdsfactory version bump — that reason should be stated explicitly).
- Conversely, a component or model change with no reference diff at all usually means the new code path is untested.

## Coverage

- New public behaviour needs a test. Check that a new cell reaches `tests/test_pdk.py`'s parametrized coverage and a new
  model reaches `tests/models/`.
- Prefer asserting on physics-meaningful quantities (a resonance frequency, an impedance, S-parameter magnitude) over
  golden strings.
