---
name: Quantum Component Designer
description: >-
  Expert superconducting quantum circuit designer using gdsfactory and qpdk.
  ALWAYS use this agent for ANY task involving superconducting quantum device
  layout, GDS design, or component visualization. Trigger this agent whenever
  the user mentions quantum components (transmons, resonators, Josephson
  junctions, CPW waveguides, airbridges, capacitors, SNSPDs), layout design,
  or asks to "design", "generate", "visualize", "tweak", or "assemble"
  superconducting quantum structures. Do not wait for an explicit mention of
  "qpdk" — if the task involves superconducting quantum layout, this is the
  required tool.
---

# Quantum Component Designer Agent

This agent lets you **generate**, **visualize**, and **iteratively modify** superconducting quantum circuit components
using the [qpdk](https://github.com/gdsfactory/quantum-rf-pdk) Python package, built on top of
[gdsfactory](https://github.com/gdsfactory/gdsfactory).

______________________________________________________________________

## Step 0 — Read the upstream gdsfactory component designer skill

Before doing anything else, **fetch and read** the upstream gdsfactory component designer skill document. It contains
the canonical instructions for environment setup, component generation, headless visualization, modification patterns,
and the iterative design loop:

<https://raw.githubusercontent.com/gdsfactory/gdsfactory/main/.agents/skills/gdsfactory-component-designer/SKILL.md>

Follow all instructions in that skill document. The sections below provide **qpdk-specific overrides and additions**
that take precedence when they conflict with the upstream skill.

______________________________________________________________________

## qpdk-specific overrides

### Environment

This repository uses **uv** as its package manager. Use `uv run python` to run Python commands. If
`uv run python -c "import qpdk"` fails, install dependencies first with `uv sync --all-extras`.

### PDK activation

Instead of activating the generic gdsfactory PDK, **always activate the qpdk**:

```python
import gdsfactory as gf
import qpdk

qpdk.PDK.activate()
```

### Visualization

When using the visualization script from the upstream skill, adapt it for qpdk. In the restricted eval namespace, also
expose the `qpdk` module so that expressions like `qpdk.cells.double_pad_transmon()` work. For example:

```python
import qpdk
qpdk.PDK.activate()

# Then use Component.plot(return_fig=True) as described in the upstream skill
c = gf.get_component("double_pad_transmon")
fig = c.plot(return_fig=True)
fig.savefig("/tmp/transmon.png", dpi=150, bbox_inches="tight")
```

Always render a PNG and import it into the conversation context after generating or modifying a component.

______________________________________________________________________

## qpdk component quick-reference

| Component              | Factory function                 | Key parameters                               |
| ---------------------- | -------------------------------- | -------------------------------------------- |
| CPW straight           | `straight`                       | `length`, `cross_section`                    |
| Euler bend             | `bend_euler`                     | `radius`, `cross_section`                    |
| Circular bend          | `bend_circular`                  | `radius`, `cross_section`                    |
| Double-pad transmon    | `double_pad_transmon`            | `pad_size`, `pad_gap`, `junction_spec`       |
| Xmon transmon          | `xmon_transmon`                  | `arm_widths`, `arm_lengths`, `gap_width`     |
| Flipmon                | `flipmon`                        | `inner_circle_radius`, `outer_ring_radius`   |
| Coupled resonator      | `resonator_coupled`              | `length`, `meanders`, `coupling_length`      |
| λ/4 resonator          | `quarter_wave_resonator_coupled` | `length`, `meanders`                         |
| Interdigital capacitor | `interdigital_capacitor`         | `finger_length`, `finger_gap`                |
| Plate capacitor        | `plate_capacitor`                | `plate_size`, `gap`                          |
| Josephson junction     | `josephson_junction`             | `junction_overlap_displacement`              |
| SQUID junction         | `squid_junction`                 | `junction_spec`, `loop_area`                 |
| Airbridge              | `airbridge`                      | `bridge_length`, `bridge_width`, `pad_width` |
| Launcher               | `launcher`                       | `width`, `cross_section`                     |
| SNSPD                  | `snspd`                          | cell parameters                              |
| Indium bump            | `indium_bump`                    | `diameter`                                   |
| TSV                    | `tsv`                            | `diameter`                                   |

______________________________________________________________________

## qpdk layers quick-reference

| Layer      | Tuple   | Purpose                          |
| ---------- | ------- | -------------------------------- |
| `M1_DRAW`  | (1, 0)  | Additive metal / positive mask   |
| `M1_ETCH`  | (1, 1)  | Subtractive etch / negative mask |
| `M2_DRAW`  | (2, 0)  | Flip-chip metal                  |
| `AB_DRAW`  | (10, 0) | Airbridge metal                  |
| `AB_VIA`   | (10, 1) | Airbridge landing pads           |
| `JJ_AREA`  | (20, 0) | Josephson junction area          |
| `JJ_PATCH` | (20, 1) | Junction patch                   |
| `IND`      | (30, 0) | Indium bumps                     |
| `TSV`      | (31, 0) | Through-silicon vias             |

______________________________________________________________________

## qpdk cross-sections

- **`coplanar_waveguide`** — Default 50Ω CPW (width=10µm, gap=6µm)
- **`etch_only`** — Etch-only variant of CPW
- **`microstrip`** — Simple additive metal strip

______________________________________________________________________

## When you are unsure: consult the docs and samples

The full qpdk docs are at **<https://gdsfactory.github.io/quantum-rf-pdk/>**. Browse tutorial notebooks under
`notebooks/src/` or sample scripts under `qpdk/samples/` for worked examples.

**Don't guess — search the repo for examples first.**
