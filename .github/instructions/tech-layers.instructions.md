---
applyTo: "qpdk/tech.py,qpdk/layers.yaml,qpdk/klayout/**"
---

# Technology and layer stack review instructions

These files define the fabrication reality the whole PDK is built on. A mistake here is silently wrong in every
component, so review them more strictly than ordinary code.

## Layer definitions must stay in sync

A layer exists in several places and **all of them must agree on layer number and datatype**:

- `qpdk/tech.py` — the `LayerMapQPDK` class (aliased `LAYER` / `L`), the `LayerStack`, and the connectivity specs.
- `qpdk/layers.yaml` — the `LayerViews` used for display and for the KLayout layer properties.
- `qpdk/klayout/` — the technology (`.lyt`) and layer property (`.lyp`) files.

Flag any change that touches one of these without the others. The `check-tech-structure` pre-commit hook checks the
structure but cannot catch a mismatched datatype.

## Layer semantics

- `*_DRAW` is additive (positive metal), `*_ETCH` is subtractive; additive wins where they overlap. Confirm a new layer
  follows that naming and that its role is commented.
- Distinguish fabrication layers from non-fabrication ones (`TEXT`, `LABEL_*`, `SIM_AREA`, `SIM_ONLY`, `WG`,
  `ERROR_PATH`). A simulation- or annotation-only layer must never reach a fab deck, and should be reflected in
  `NON_METADATA_LAYERS` only if it genuinely is one.
- Changing an existing layer number or datatype is a breaking change for every stored GDS, every reference file, and any
  customer design. It needs an explicit justification in the PR, not a silent edit.

## Layer stack and materials

- Thicknesses, `zmin`, and material properties must be physically consistent (no negative thickness, no overlapping
  levels that should be disjoint, superconductor properties matching the named material).
- `[tool.elvis]` in `pyproject.toml` lists `short-layers` and `connected-layers` for LVS. Adding a conductive layer
  usually means updating those too — flag when it is missed.

## Cross sections

- Cross sections are the shared vocabulary for routing. Prefer extending an existing one over adding a near-duplicate.
- A CPW cross section's `width` / `gap` determine the characteristic impedance. If a default changes, every existing
  design silently changes impedance — call that out and check the GDS regression diff reflects it.
- Renaming a cross section or a cell breaks user netlists. Prefer adding an alias over renaming.

## Blast radius

Almost any change here regenerates GDS regression references. Verify `just test-gds-force` output is committed and that
the size of the reference diff matches the scope of the change.
