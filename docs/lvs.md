# Layout vs. Schematic (LVS) Verification

```{contents}
:local:
:depth: 2
```

## What is LVS?

**Layout vs. Schematic (LVS)** is a verification step that compares the physical layout of a circuit (the GDS file)
against its intended schematic (a netlist). The goal is to ensure that the fabricated chip will actually implement the
designed circuit — that every device is present, every connection is correct, and no unintended shorts or opens exist.

In a superconducting quantum PDK the "devices" are not transistors but quantum components: transmon qubits,
coplanar-waveguide resonators, Josephson junctions, meander inductors, and interdigital capacitors. LVS for quantum
circuits therefore needs custom device-extraction rules rather than the standard MOSFET models found in semiconductor
PDKs.

```{mermaid}
flowchart LR
    GDS["GDS Layout"] --> LVS["KLayout LVS"]
    CIR["SPICE Netlist\n(Schematic)"] --> LVS
    LVS --> LVSDB[".lvsdb Report"]
    LVSDB --> Browser["KLayout Netlist\nDatabase Browser"]
```

## How the QPDK LVS Deck Works

The LVS deck lives in `qpdk/klayout/lvs/qpdk.lvs` and is generated from the Jinja2 template `qpdk.lvs.j2` by the
rendering script `render_lvs.py`.

### Layer Connectivity

The deck encodes the physical layer stack connectivity of the QPDK technology:

| Connection Path              | Via / Bump Layer |
| ---------------------------- | ---------------- |
| M1_DRAW ↔ M2_DRAW            | TSV (31/0)       |
| M1_DRAW ↔ M2_DRAW            | IND (30/0)       |
| M1_DRAW ↔ M1_DRAW (over gap) | AB_DRAW (10/0)   |
| M2_DRAW ↔ M2_DRAW (over gap) | AB_DRAW (10/0)   |

These mirror the `LAYER_CONNECTIVITY` definition in `qpdk/tech.py`.

### Port Layers

Each physical metal has a dedicated **port layer** used by the LVS engine to identify net terminals (pins):

| Port Layer | GDS Layer | Connected Metal |
| ---------- | --------- | --------------- |
| PORT_M1    | 1/10      | M1_DRAW (1/0)   |
| PORT_M2    | 2/10      | M2_DRAW (2/0)   |

Components should place small rectangular shapes on the appropriate port layer at every electrical port location. The
LVS deck connects each port layer to its metal, so KLayout can trace which nets reach which pins.

### Device Marker Layers

Device detection uses a **marker-layer** approach. Every instance of a quantum component includes a bounding polygon on
its own marker layer:

| Device Type            | Marker Layer | GDS Layer | Physical Layer |
| ---------------------- | ------------ | --------- | -------------- |
| Transmon qubit         | MK_TRANSMON  | 200/0     | M1_DRAW        |
| CPW resonator          | MK_RESONATOR | 201/0     | M1_DRAW        |
| Meander inductor       | MK_INDUCTOR  | 202/0     | M1_DRAW        |
| Interdigital capacitor | MK_CAPACITOR | 203/0     | M1_DRAW        |
| Josephson junction     | MK_JJ        | 204/0     | JJ_AREA        |

The LVS deck performs a boolean **AND** of the marker layer with the physical layer to isolate the device region:

```ruby
# Example: Transmon detection
transmon_region = m1_draw & mk_transmon
```

### Jinja2 Templating

The LVS deck is generated from a Jinja2 template (`qpdk.lvs.j2`) so that adding new device types only requires updating
the data tables in `render_lvs.py`. For example, to add a new device type you would append an entry to the `DEVICES`
list:

```python
_device(
    var="snspd",
    description="Superconducting nanowire single-photon detector",
    physical_layer="m1_draw",
    marker_var="mk_snspd",
    class_name="SubCircuit",
    model_name="SNSPD",
    terminals=[
        {"name": "input", "layer": "port_m1"},
        {"name": "output", "layer": "port_m1"},
    ],
)
```

Then re-render with:

```bash
uv run python qpdk/klayout/lvs/render_lvs.py
```

## Running LVS

### Prerequisites

- [KLayout](https://www.klayout.de/) ≥ 0.28 installed (the `klayout` command must be on your `PATH`).
- A GDS layout file exported from gdsfactory.
- A SPICE netlist (`*.cir`) representing the intended schematic.

### Invocation

Run KLayout in batch mode (`-b`) with the LVS deck:

```bash
klayout -b -r qpdk/klayout/lvs/qpdk.lvs \
    -rd input=my_layout.gds \
    -rd schematic=my_netlist.cir \
    -rd report=lvs_report.lvsdb
```

Optional switches:

- `-rd topcell=CELL_NAME` — specify the top-level cell (auto-detected if omitted).

The command produces a `.lvsdb` report file that can be inspected in KLayout.

### Adding Port and Marker Shapes to Components

For the LVS deck to detect devices and pins, your components must include shapes on the appropriate marker and port
layers. The recommended approach is to add them inside the `@gf.cell` function body.

#### Port shapes

Add a small rectangle at each port location on the appropriate port layer:

```python
import gdsfactory as gf
from qpdk.tech import LAYER

@gf.cell
def my_component():
    c = gf.Component()
    # ... draw the component geometry ...

    # Add port shapes for LVS
    port_size = 1.0  # µm, small marker rectangle
    for port in c.ports:
        if port.layer == LAYER.M1_DRAW:
            port_layer = LAYER.PORT_M1
        elif port.layer == LAYER.M2_DRAW:
            port_layer = LAYER.PORT_M2
        else:
            continue
        c.add_polygon(
            [
                (port.x - port_size / 2, port.y - port_size / 2),
                (port.x + port_size / 2, port.y - port_size / 2),
                (port.x + port_size / 2, port.y + port_size / 2),
                (port.x - port_size / 2, port.y + port_size / 2),
            ],
            layer=port_layer,
        )
    return c
```

#### Device marker shapes

Add a bounding polygon on the device marker layer that covers the entire device area:

```python
from qpdk.tech import LAYER

@gf.cell
def my_transmon(**kwargs):
    c = gf.Component()
    # ... draw the transmon ...

    # Add device marker for LVS
    bbox = c.bbox()
    c.add_polygon(
        [
            (bbox[0][0], bbox[0][1]),
            (bbox[1][0], bbox[0][1]),
            (bbox[1][0], bbox[1][1]),
            (bbox[0][0], bbox[1][1]),
        ],
        layer=LAYER.MK_TRANSMON,
    )
    return c
```

## Inspecting LVS Results

After running LVS you will have a `.lvsdb` file. Open it in KLayout:

1. Launch KLayout with your layout: `klayout my_layout.gds`
1. Go to **Tools → Netlist Database Browser** (or press `Ctrl+Shift+N`)
1. Load the `.lvsdb` file

The Netlist Database Browser shows:

- **Circuits (left panel):** A hierarchical list of extracted circuits and their schematic counterparts.
- **Comparison status:** Green check marks (✓) for matching circuits, red crosses (✗) for mismatches.
- **Nets:** Click any circuit to see its nets. Mismatched nets are highlighted.
- **Cross-probing:** Click a net or device in the browser to highlight it in the layout view — invaluable for tracking
  down the root cause of mismatches.

### Common LVS Errors

| Error                | Likely Cause                                 |
| -------------------- | -------------------------------------------- |
| Missing device       | Marker layer polygon missing in component    |
| Extra device         | Spurious marker polygons in layout           |
| Net mismatch (short) | Unintended metal overlap connecting two nets |
| Net mismatch (open)  | Missing via/bump between metal layers        |
| Pin count mismatch   | Port shapes missing or on wrong port layer   |

## References

- [KLayout LVS Manual](https://www.klayout.de/doc/manual/lvs.html)
- [KLayout LVS API Reference](https://www.klayout.de/doc-qt5/about/lvs_ref.html)
- [KLayout Ruby API](https://www.klayout.de/doc-qt5/code/index.html)
