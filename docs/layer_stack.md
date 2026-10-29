---
myst:
  html_meta:
    description: Cross-sections of the QPDK layer stacks, showing every physical level of the base, flip-chip, backside and gatemon stacks with its material, thickness, height and GDS layer.
---

# Layer Stack

`qpdk.tech.LAYER_STACK` gives each GDS layer a material, a thickness and a height `zmin`. Field solvers and 3D views
extrude the layout from it. The figure below shows every level of every stack. The main cross-section is
`LAYER_STACK_FLIP_CHIP_BACKSIDE`, which holds the base stack, the flip-chip top chip and the backside metal of both
chips, with heights measured from the top of the bottom silicon substrate. The panel on the right shows the gatemon
films.

![Not-to-scale cross-sections of the QPDK layer stacks. Main panel: a 500 µm silicon substrate with a TiN through-silicon via; 200 nm niobium M1 ground and signal films with M1_ETCH slots; a niobium airbridge on 100 nm vias over the signal line; a 70 nm AlOx/Al Josephson junction; a 15 nm NbTiN film; a 10 µm indium bump from an M1 pad up to a flip-chip top chip carrying 200 nm niobium M2 and M2_ground films at 10 µm under a second 500 µm silicon substrate, crossed by a TSV2 via up to a 200 nm niobium MB2 backside metal with MB2_ETCH slots on its outer face, with vacuum above it; vacuum between the chips; and a 200 nm niobium backside metal with MB_ETCH slots on the bottom face of the bottom substrate, where the TSV lands, with vacuum below it. Gatemon panel: a 150 nm oxide on the substrate, Nb/Au contacts to a nanotube, aluminium reconnection films to 155 nm M1, and an aluminium top gate on hBN](notebooks/figures/layer-stack.svg)

Notes:

- `M1_ground` has no drawn layer of its own. It is derived as `SIM_AREA` minus `M1_ETCH` and `M1_DRAW`, so the solver
  sees the implicit ground plane around etched slots. `M1` is the drawn metal (`M1_DRAW`).
- `Substrate` and `Vacuum` both extrude `SIM_AREA`, the simulation region.
- `LAYER_STACK_FLIP_CHIP` adds the top chip: `M2` (`M2_DRAW`) and `M2_ground` (Nb, 200 nm) at z = 10 µm, the indium bump
  height, and `Substrate_top`, a second 500 µm silicon substrate above them. `M2_ground` is derived like `M1_ground`,
  from `SIM_AREA` minus `M2_ETCH` and `M2_DRAW`.
- `LAYER_STACK_BACKSIDE` adds `MB`, a 200 nm Nb backside metal on the bottom face of the substrate at z = −500.2 µm, and
  `Vacuum_bottom`, 500 µm of vacuum below it. `MB` is derived like `M1_ground`, from `SIM_AREA` minus `MB_ETCH` and
  `MB_DRAW`, and the TSVs connect it to the front-side `M1`.
- `LAYER_STACK_FLIP_CHIP_BACKSIDE` puts backside metal on both chips. On top of the flip-chip and backside levels it
  adds `TSV2` (TiN, `TSV2` layer), running up through `Substrate_top` from z = 10.2 µm, `MB2`, a 200 nm Nb film on the
  top chip's outer face at z = 510.2 µm, derived like `MB` from `SIM_AREA` minus `MB2_ETCH` and `MB2_DRAW`, and
  `Vacuum_top`, 500 µm of vacuum above it. TSV cells take `chip="bottom"` or `chip="top"` to pick `M1`/`MB`/`TSV` or
  `M2`/`MB2`/`TSV2`, as listed in `qpdk.tech.CHIP_LAYERS`.
- `LAYER_STACK_GATEMON` (`get_gatemon_layer_stack()`) puts z = 0 at the top of a 150 nm oxide, makes `M1` 155 nm thick,
  and adds the contact, hBN, top gate and reconnection films. The hBN thickness is a parameter (30 nm by default) and
  the top gate sits on it. The nanotube itself is not extruded.
- `LAYER_STACK_3D` is the base stack plus the gatemon films, with `Reconnection` moved to the junction height. It also
  previews the drawn `M2`, `MB`, `MB2` and `TSV2` shapes, but not their ground planes, which would otherwise appear
  under every layout. View a flip-chip or backside design with its physical stack for those.
- The material properties, such as relative permittivity and loss tangent, are in `qpdk.tech.material_properties`.
