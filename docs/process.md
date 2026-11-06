---
myst:
  html_meta:
    description: The QPDK fabrication process, the layer stack of films and materials it is made of and the design rules the DRC deck enforces on layouts.
---

# Process

QPDK describes a generic superconducting process in two files: the layers and their GDS numbers in `qpdk/layers.yaml`,
and their physical stack, materials and cross-sections in `qpdk/tech.py`. Cells draw on those layers, the field solvers
extrude the stack, and the DRC deck enforces its geometry.

![Two panels. The left one, layer stack, draws a section through the films: a silicon substrate carrying a niobium M1 film with an aluminium junction pad on top. The right one, design rules, shows the same films from above as masks, a coplanar waveguide of ground, signal and ground strips separated by etched slots, with the signal width w and the slot width s marked.](notebooks/figures/process-two-views.svg)

Two pages cover the process:

- [Layer stack](layer_stack.md): every level of the base, flip-chip, backside and gatemon stacks, with its material,
  thickness, height and GDS layer.
- [Design rules](design_rules.md): the subtractive mask model, the rules the deck runs, the rule profiles, and the
  intent layers that coupled cells emit.

To change the stack, the materials or the rules for your own fab, see
[Customizing QPDK for Your Process](customizing.md).

```{toctree}
:hidden:

layer_stack
design_rules
```
