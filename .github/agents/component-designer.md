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

## When to use this agent

Activate this agent when the user:

- Asks to create or generate a quantum component (transmon, resonator, Josephson junction, SQUID, CPW waveguide,
  airbridge, capacitor, SNSPD, launcher, bump, TSV, etc.).
- Wants to see what a component looks like (requests a plot, image, or preview).
- Wants to tweak component parameters (widths, lengths, gaps, layers, cross-sections, etc.) and see the result.
- Mentions GDS, quantum layout, superconducting circuits, or the qpdk.
- Asks to compose multiple components together into a larger chip design.

______________________________________________________________________

## 1 — Setting up the environment

### Choosing the right Python command

This repository uses **uv** as its package manager. The standard command is:

```bash
uv run python
```

Probe the environment first: check for `pyproject.toml` with `[tool.uv]` or an active virtualenv. When in doubt, try
`uv run python -c "import qpdk"` — if it fails, install dependencies first with `uv sync --all-extras`.

### Activating the PDK

Before generating any component, activate the quantum PDK:

```python
import gdsfactory as gf
import qpdk

# Activate the quantum PDK
qpdk.PDK.activate()
```

Always clear the cell cache between independent component generations to avoid stale state: `gf.clear_cache()`.

______________________________________________________________________

## 2 — Generating a component

### 2.1 From the qpdk component library

qpdk ships with parametric component factory functions. Components are accessed either through `qpdk.cells` or via
`gf.get_component()` after PDK activation:

```python
import gdsfactory as gf
import qpdk

qpdk.PDK.activate()

# Example: Double-pad transmon qubit
c = gf.get_component("double_pad_transmon")

# Example: Coupled resonator with custom parameters
c = gf.get_component(
    "resonator_coupled",
    length=5000.0,
    meanders=8,
    coupling_length=150.0,
)
```

You can also call factory functions directly:

```python
from qpdk.cells import double_pad_transmon, resonator_coupled

c = double_pad_transmon(pad_size=(250.0, 400.0), pad_gap=15.0)
c = resonator_coupled(length=5000.0, meanders=8)
```

### 2.2 Inspecting component parameters

Every component factory function is a standard Python callable with typed parameters. Use `help()` or
`inspect.signature()` to discover the parameters:

```python
import inspect
from qpdk.cells import double_pad_transmon
print(inspect.signature(double_pad_transmon))
```

### 2.3 Listing all available components

```python
import qpdk
qpdk.PDK.activate()

for name in sorted(qpdk.PDK.cells):
    print(name)
```

______________________________________________________________________

## 3 — Visualizing a component (Proactive Workflow)

Visualization is essential: render and inspect the component after creating or modifying it to verify the result.

### 3.1 Save a plot image to disk and display it

Use the helper script bundled with this agent for reliable headless rendering. From a bash tool:

```bash
uv run python .github/agents/scripts/visualize_component.py \
    "qpdk.cells.double_pad_transmon()" \
    /tmp/transmon.png
```

The script also supports generic gdsfactory expressions:

```bash
uv run python .github/agents/scripts/visualize_component.py \
    "gf.get_component('resonator_coupled', length=5000)" \
    /tmp/resonator.png
```

After saving, **always import the image into context** (e.g. using a screenshot or image-viewing tool) so you and the
user can see it.

### 3.2 Inline visualization

You can also render inline in a Python script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gdsfactory as gf
import qpdk

qpdk.PDK.activate()
c = gf.get_component("double_pad_transmon")

fig = c.plot(return_fig=True)
fig.savefig("/tmp/component.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### 3.3 Best practices

- **Be Proactive:** If the user's intent is clear (e.g., "Create a transmon then change its pad size"), do not stop for
  permission after the first step. Perform the generation, visualization, and modification in a single turn if possible.
- **Concise Reporting:** Describe the component briefly (ports, size). If the change is minor, just confirm the update
  and point to the new image. Avoid repeating the entire component description multiple times.

______________________________________________________________________

## 4 — Modifying a component

### 4.1 Adjusting parameters

The simplest modification is changing the factory-function arguments. Always call `gf.clear_cache()` before regenerating
to avoid stale data:

```python
gf.clear_cache()
c = gf.get_component(
    "double_pad_transmon",
    pad_size=(300.0, 500.0),
    pad_gap=20.0,
)
```

### 4.2 Composing components

Build a custom component by placing and connecting sub-components. Use the `@gf.cell` decorator for proper naming and
caching:

```python
import gdsfactory as gf
import qpdk

qpdk.PDK.activate()

@gf.cell
def transmon_with_resonator() -> gf.Component:
    c = gf.Component()
    transmon = c << gf.get_component("double_pad_transmon")
    resonator = c << gf.get_component(
        "resonator_coupled", length=5000.0
    )
    # Connect resonator port to transmon port
    resonator.connect("o1", transmon.ports["o1"])
    return c
```

### 4.3 Using cross-sections

qpdk provides specialized cross-sections for superconducting circuits:

- **`coplanar_waveguide`** — Default 50Ω CPW (width=10µm, gap=6µm)
- **`etch_only`** — Etch-only variant of CPW
- **`microstrip`** — Simple additive metal strip

```python
from qpdk.tech import coplanar_waveguide

xs = coplanar_waveguide(width=12, gap=8, radius=120)
c = gf.get_component("straight", length=500, cross_section=xs)
```

______________________________________________________________________

## 5 — Exporting the component

```python
# Write to GDS file
gdspath = c.write_gds("/tmp/my_component.gds")
print(f"GDS written to: {gdspath}")
```

______________________________________________________________________

## 6 — Proactive Design Loop (Best Practices)

When the user asks for a component, follow this streamlined loop:

1. **Understand & Execute:** Identify the component and parameters. If the user asks for a sequence of steps, execute
   them as a batch where logical.
1. **Generate & Visualize:** Create the component and render the PNG using the helper script.
1. **Show & Tell:** Share the image and a *short* summary of what changed (name, ports, bounding box size).
1. **Anticipate:** If the next step is obvious, offer to perform it or just do it and show the result.
1. **Iterate Concisely:** For small tweaks, don't repeat the full initial explanation. Just show the new image and
   highlight the specific change.

______________________________________________________________________

## 7 — Common component quick-reference

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

## 8 — Layers quick-reference

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

## 9 — When you are unsure: consult the docs and samples

The full qpdk docs are at **<https://gdsfactory.github.io/quantum-rf-pdk/>**. Browse tutorial notebooks under
`notebooks/src/` or sample scripts under `qpdk/samples/` for worked examples.

**Don't guess — search the repo for examples first.**
