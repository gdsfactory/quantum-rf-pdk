---
applyTo: "qpdk/models/**/*.py"
---

# Model review instructions

`qpdk/models/` holds the S-parameter and circuit models (resonators, couplers, waveguides, capacitors, inductors,
junctions, qubit Hamiltonians, perturbation theory). Media definitions for coplanar waveguides live in
`qpdk/models/cpw.py`.

## JAX and jittability

- SAX models **must** stay JIT-compilable. Flag anything that breaks tracing:
  - Python control flow on traced array values (`if x > 0:`, `for` over array entries, `while`).
  - In-place mutation (`arr[i] = ...`); use `arr.at[i].set(...)`.
  - Data-dependent shapes, or `.item()` / `float()` on a traced value.
  - Use `jax.lax.cond`, `lax.scan`, `lax.fori_loop` instead.
- Prefer `jnp` over `np`, and `jaxellip` over `scipy` for elliptic integrals.
- Helper functions should be JIT-compiled: `@partial(jax.jit, inline=True)`.
- If a change adds a model, ask whether it actually jits — a model that only works eagerly will fail downstream in
  circuit simulation, not in the unit test that calls it directly.

## Physical constants

Constants (`e`, `h`, `π`, `c_0`, `Φ_0`, `ε_0`, `μ_0`, `Z_0_FREE`) **must** be imported from `qpdk/models/constants.py`,
which sources them from `scipy.constants`. Flag any locally redefined numeric constant:

```python
from qpdk.models.constants import Φ_0  # good

Φ_0 = 2.067833848e-15  # bad — duplicate of the centralized constant
```

## SAX port terminations

Omitting a model port from a SAX circuit applies a **matched load** (`Z_L = Z_0`, `Γ = 0`), not an open circuit.

| Termination | Impedance   | Reflection |
| ----------- | ----------- | ---------- |
| Matched     | `Z_L = Z_0` | `Γ = 0`    |
| Open        | `Z_L → ∞`   | `Γ = +1`   |
| Short       | `Z_L = 0`   | `Γ = -1`   |

Flag a circuit that leaves a port dangling where the physics requires an open or a short. For a quarter-wave coupled
resonator, `resonator_o1` is open and `resonator_o2` is shorted — prefer `quarter_wave_resonator_coupled`, which applies
the short internally, or wire an explicit short model.

## Physics correctness

- Check units and their consistency (Hz vs rad/s, µm vs m, F vs fF). Unit errors are the most common real bug here and
  they pass the type checker.
- Passive components must produce **passive** and **reciprocal** S-matrices. If a new model can violate either, say so.
- Docstrings should state the model's assumptions and validity range (frequency, geometry, lossless vs lossy) and cite
  `docs/bibliography.bib` where the formula comes from a paper.

## Imports

`sax`, `jaxellip`, `sympy`, `polars`, `pandas`, `scqubits`, `gplugins`, `optax`, `optuna`, `netket`, `flax`,
`pymablock`, `qutip-jax`, and `qutip-qip` are all in ruff's `require-lazy` list — import them inside the function that
uses them, not at module level.

## Tests that must accompany a model change

- Unit tests in `tests/models/` covering behaviour, passivity, and reciprocity.
- Model regression references are regenerated with `just test-models-force` into `tests/test_models_regression/` and
  committed alongside the change. A numeric reference diff needs an explanation in the PR.
