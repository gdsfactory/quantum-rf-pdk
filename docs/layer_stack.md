---
myst:
  html_meta:
    description: Cross-section of the QPDK layer stack, showing every physical level with its material, thickness, height and GDS layer.
---

# Layer Stack

`qpdk.tech.LAYER_STACK` gives each GDS layer a material, a thickness and a height $z\_\\text{min}$. Field solvers and 3D
views extrude the layout from it. The cross-section below shows every level, with heights measured from the top of the
silicon substrate.

![Not-to-scale cross-section of the QPDK layer stack: a 500 µm silicon substrate with a TiN through-silicon via; 200 nm niobium M1 ground and signal films with M1_ETCH slots; a niobium airbridge on 100 nm vias over the signal line; a 70 nm AlOx/Al Josephson junction; a 15 nm NbTiN film; a 10 µm indium bump on an M1 pad; and vacuum above the metal](notebooks/figures/layer-stack.svg)

Notes:

- `M1_ground` has no drawn layer of its own. It is derived as `SIM_AREA` minus `M1_ETCH` and `M1_DRAW`, so the solver
  sees the implicit ground plane around etched slots. `M1` is the drawn metal (`M1_DRAW`).
- `Substrate` and `Vacuum` both extrude `SIM_AREA`, the simulation region.
- Other stacks build on this one. `LAYER_STACK_FLIP_CHIP` adds the top chip: `M2` and `M2_ground` (Nb, 200 nm) at $z =
  10,\\text{µm}$, the indium bump height, and a second 500 µm substrate above them. `LAYER_STACK_GATEMON` and
  `LAYER_STACK_3D` add the gatemon contact, hBN and gate films.
- The material properties, such as $\\varepsilon\_\\text{r}$ and $\\tan\\delta$, are in `qpdk.tech.material_properties`.
