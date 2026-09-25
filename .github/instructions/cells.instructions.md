---
applyTo: "qpdk/cells/**/*.py"
---

# Cell (component) review instructions

`qpdk/cells/` holds the gdsfactory component definitions — transmons, resonators, couplers, airbridges, SNSPDs,
fluxoniums, unimons, launchers, junctions, TSVs, bumps. `qpdk/cells/derived/` holds composites built from them.

## Registration

- Every cell **must** be decorated with `@gf.cell` and carry a `tags` tuple for categorization:

  ```python
  @gf.cell(tags=("qubits", "transmons"))
  def transmon(...) -> gf.Component: ...
  ```

- A new module **must** be re-exported from `qpdk/cells/__init__.py` with `from qpdk.cells.<module> import *`, otherwise
  the component never reaches the PDK cell registry. Flag a new cell file that is not added there.

- Top-level cell modules must not contain an `if __name__ == "__main__":` block — the `check-no-main-in-cells`
  pre-commit hook rejects it. Put runnable demos in `qpdk/samples/` instead.

## Layers

- Layer assignments **must** come from the `LAYER` map (`LayerMapQPDK`) in `qpdk/tech.py`. Raw layer tuples/ints are
  rejected by the `check-no-raw-layers` pre-commit hook:

  ```python
  c.add_polygon(points, layer=LAYER.M1_DRAW)  # good
  c.add_polygon(points, layer=(1, 0))  # bad — raw layer
  ```

- Mind the additive/subtractive pairs: `*_DRAW` is positive metal, `*_ETCH` is a negative/etch region, and additive wins
  where they overlap. Flag a change that mixes them up.

- Adding a new layer is rare. If a change does add one, it must land in **both** `qpdk/tech.py` (`LayerMapQPDK`) and
  `qpdk/layers.yaml` (`LayerViews`) with identical layer number and datatype, and be reflected in the KLayout files
  under `qpdk/klayout/`.

## Ports and netlists

- Components must round-trip: component → netlist → component. This is checked by `test_netlists` in
  `tests/test_pdk.py`. Flag changes to port names, port ordering, or port types that would silently break a stored
  netlist reference.
- Keep `port_type` consistent with what the port physically is (`electrical` vs `optical`/RF), and keep naming
  consistent with neighbouring cells in the same module. A renamed port is a breaking change for user YAML netlists —
  call it out.
- Cross sections come from `qpdk/tech.py`. Prefer an existing cross section over constructing one inline.

## Parameters and geometry

- Dimensions are in microns (gdsfactory convention). Physically meaningful defaults belong in the signature, not in the
  body.
- Prefer composing existing cells over duplicating geometry code. If a change copies an existing cell with two numbers
  changed, suggest parameterizing the original instead.
- Docstrings should state the quantum-device purpose and what each geometric parameter controls, with a citation where
  the geometry follows a published design.

## Tests that must accompany a cell change

- New or changed cells need regenerated GDS regression references: `just test-gds-force`, with the resulting files in
  `tests/gds_ref/` and the settings/netlist YAML under `tests/test_pdk/` committed in the same PR.
- If reference YAML or GDS files changed, verify the diff is a plausible consequence of the code change. A settings diff
  with no corresponding code change — or a code change with no reference diff — is a red flag worth a comment.
