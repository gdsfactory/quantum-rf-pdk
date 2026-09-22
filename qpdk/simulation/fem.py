"""Volumetric-FEM helpers for qpdk layouts.

Converts the PDK's subtractive etch masks into the explicit conductor and
dielectric regions that full-wave 3-D solvers (Palace through gsim, Meep, ...)
expect, and provides the matching layer stacks. gsim is part of the ``models``
extra, so it is still imported lazily by the functions that need it.
"""

from __future__ import annotations

from typing import Any

import gdsfactory as gf
import klayout.db as kdb

from qpdk.tech import LAYER, material_properties

__all__ = [
    "FEM_LAYERS",
    "FLIP_CHIP_FEM_LAYERS",
    "flip_chip_stack",
    "single_chip_stack",
    "to_fem_regions",
    "to_flip_chip_regions",
]

# Layer numbering understood by gsim's Palace and Meep workflows: dielectric
# substrate, zero-thickness conductor sheet, and the vacuum above the chip.
# The GDS numbers come from the unused simulation-only block, next to
# SIM_AREA (98,0) / SIM_ONLY (99,0): they must not collide with real mask
# layers (M1_DRAW is (1,0), M2_DRAW (2,0), ...), or the PDK's own
# additive-metal pass would destroy a written-back FEM component and any
# GDS round-trip would show dielectric as metal.
FEM_LAYERS = {"SUBSTRATE": (91, 0), "SUPERCONDUCTOR": (92, 0), "VACUUM": (93, 0)}


def to_fem_regions(component: gf.Component) -> gf.Component:
    """Convert an etch-based qpdk layout into explicit FEM regions.

    The PDK draws metal subtractively: ``M1_ETCH`` marks where metal is
    removed and additive shapes on ``M1_DRAW`` punch through the etch mask
    (the conductor is ``SIM_AREA - (M1_ETCH - M1_DRAW)``), matching
    ``get_layer_stack()``'s M1 definition. The conductor, substrate and
    vacuum regions are copied onto the layer numbering of
    :data:`FEM_LAYERS`, with the zero-thickness conductor later becoming a
    perfect-electric-conductor boundary in the solver.

    Args:
        component: Layout with a ``SIM_AREA`` layer and simulation ports.

    Returns:
        Component carrying the ``SUBSTRATE``, ``SUPERCONDUCTOR`` and
        ``VACUUM`` regions plus the input ports.

    Raises:
        ValueError: If the component carries no ``SIM_AREA`` shapes or the
            simulation area is fully etched away, either of which would
            silently produce an empty FEM model.
    """
    layout = component.kdb_cell.layout()
    sim_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.SIM_AREA))
    )
    etch_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.M1_ETCH))
    )
    # Conductor = SIM_AREA - ETCH + DRAW, the same rule the PDK's
    # additive-metal pass applies.
    conductor_region = sim_region - etch_region

    if sim_region.is_empty():
        msg = "no SIM_AREA layer in the component; the FEM model would be empty"
        raise ValueError(msg)
    if conductor_region.is_empty():
        msg = "SIM_AREA is fully etched away; the FEM conductor would be empty"
        raise ValueError(msg)

    etched = gf.Component()
    el = etched.kdb_cell.layout()
    for name, region in [
        ("SUPERCONDUCTOR", conductor_region),
        ("SUBSTRATE", sim_region),
        ("VACUUM", sim_region),
    ]:
        etched.kdb_cell.shapes(el.layer(*FEM_LAYERS[name])).insert(region)
    for port in component.ports:
        etched.add_port(name=port.name, port=port)
    return etched


def single_chip_stack(
    substrate_thickness: float = 500.0, vacuum_thickness: float = 500.0
) -> Any:
    r"""Return a gsim layer stack for a single qpdk silicon chip.

    The substrate uses the microwave silicon from the qpdk material
    properties (:math:`\varepsilon_r = 11.45`,
    :math:`\tan\delta = 2.7 \times 10^{-6}`), so FEM and analytical CPW
    models describe the same chip. The zero-thickness niobium conductor
    becomes a perfect electric conductor in the solver; its material entry
    only names the layer.

    Args:
        substrate_thickness: Silicon thickness in μm.
        vacuum_thickness: Air height above the chip in μm.

    Returns:
        A ``gsim.common.stack.LayerStack`` (requires gsim from the ``models``
        extra).
    """
    from gsim.common.stack import Layer, LayerStack
    from gsim.common.stack.materials import MATERIALS_DB

    silicon = material_properties["Si"]
    stack = LayerStack(pdk_name="qpdk")
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=FEM_LAYERS["SUBSTRATE"],
        zmin=0.0,
        zmax=substrate_thickness,
        thickness=substrate_thickness,
        material="qpdk-silicon",
        layer_type="dielectric",
    )
    stack.layers["SUPERCONDUCTOR"] = Layer(
        name="SUPERCONDUCTOR",
        gds_layer=FEM_LAYERS["SUPERCONDUCTOR"],
        zmin=substrate_thickness,
        zmax=substrate_thickness,
        thickness=0,
        material="qpdk-niobium",
        layer_type="conductor",
    )
    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=FEM_LAYERS["VACUUM"],
        zmin=substrate_thickness,
        zmax=substrate_thickness + vacuum_thickness,
        thickness=vacuum_thickness,
        material="vacuum",
        layer_type="dielectric",
    )
    stack.materials = {
        "qpdk-silicon": {
            "permittivity": silicon["relative_permittivity"],
            "loss_tangent": silicon["loss_tangent"],
        },
        "qpdk-niobium": {"permittivity": 1.0},
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }
    return stack


# Layer numbering for two-chip (flip-chip) models: the bottom chip sits below
# the inter-chip gap, the top chip is flipped so its metal faces down. The
# GDS numbers continue the unused simulation-only block above FEM_LAYERS,
# above the label layers (100/101), so they cannot collide with real mask
# layers either (see the FEM_LAYERS note).
FLIP_CHIP_FEM_LAYERS = {
    "SUBSTRATE": (102, 0),  # bottom chip silicon
    "M1": (103, 0),  # bottom chip metal (zero-thickness PEC)
    "VACUUM": (104, 0),  # inter-chip gap
    "BUMP": (105, 0),  # indium bumps (conductive vias)
    "M2": (106, 0),  # top chip metal (zero-thickness PEC)
    "SUBSTRATE_TOP": (107, 0),  # top chip silicon
}


def to_flip_chip_regions(component: gf.Component) -> gf.Component:
    """Convert a two-metal-layer qpdk layout into explicit flip-chip regions.

    Both metal levels follow the same subtractive convention as
    :func:`to_fem_regions` (conductor is ``SIM_AREA - (ETCH - DRAW)``, per
    ``LAYER_STACK_FLIP_CHIP``), and the indium bump layer is copied to its
    own region. The regions land on the layer numbering of
    :data:`FLIP_CHIP_FEM_LAYERS`: two substrate slabs, two zero-thickness
    metal sheets, the vacuum gap, and the bumps.

    Args:
        component: Flip-chip layout with a ``SIM_AREA`` layer, M1/M2 metal
            and ``IND`` bump shapes, plus simulation ports.

    Returns:
        Component carrying the flip-chip regions plus the input ports.

    Raises:
        ValueError: If the component carries no ``SIM_AREA`` shapes, a metal
            level has neither etch nor draw shapes (which the subtractive
            convention would turn into a solid metal plane shorting the
            model), or there are no indium bumps to connect the two chips.
    """
    layout = component.kdb_cell.layout()
    sim_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.SIM_AREA))
    )
    if sim_region.is_empty():
        msg = "no SIM_AREA layer in the component; the FEM model would be empty"
        raise ValueError(msg)

    def _level_conductor(
        etch: tuple[int, int], draw: tuple[int, int], name: str
    ) -> kdb.Region:
        """Conductor = SIM_AREA - ETCH + DRAW for one metal level."""
        etch_region = kdb.Region(
            component.kdb_cell.begin_shapes_rec(layout.layer(*etch))
        )
        draw_region = kdb.Region(
            component.kdb_cell.begin_shapes_rec(layout.layer(*draw))
        )
        if etch_region.is_empty() and draw_region.is_empty():
            # A level with no etch and no draw shapes would degenerate to a
            # solid metal plane, silently shorting the model.
            msg = (
                f"metal level {name} has neither etch nor draw shapes;"
                " the subtractive convention would make it a solid plane"
            )
            raise ValueError(msg)
        return ((sim_region - etch_region) | (draw_region & sim_region)).merged()

    m1_region = _level_conductor(LAYER.M1_ETCH, LAYER.M1_DRAW, "M1")
    m2_region = _level_conductor(LAYER.M2_ETCH, LAYER.M2_DRAW, "M2")
    # Bumps outside the simulation area would poke into empty space and
    # grow the model bounding box, so clip them to it.
    bump_region = (
        kdb.Region(component.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.IND)))
        & sim_region
    ).merged()
    if bump_region.is_empty():
        msg = "no indium bumps in the component; the two chips would not connect"
        raise ValueError(msg)

    regions = gf.Component()
    rl = regions.kdb_cell.layout()
    for name, region in [
        ("SUBSTRATE", sim_region),
        ("M1", m1_region),
        ("VACUUM", sim_region),
        ("BUMP", bump_region),
        ("M2", m2_region),
        ("SUBSTRATE_TOP", sim_region),
    ]:
        regions.kdb_cell.shapes(rl.layer(*FLIP_CHIP_FEM_LAYERS[name])).insert(region)
    for port in component.ports:
        regions.add_port(name=port.name, port=port)
    return regions


def flip_chip_stack(
    substrate_thickness: float = 500.0,
    bump_thickness: float = 10.0,
) -> Any:
    """Return a gsim layer stack for a face-to-face qpdk flip-chip pair.

    The bottom chip carries M1 with the substrate below it; the top chip is
    flipped so its M2 faces down across the indium-bump gap, with its
    substrate above. Both metals become perfect electric conductors and the
    bumps conductive vias; the substrate uses the qpdk microwave silicon, so
    FEM and analytical models describe the same chips.

    Args:
        substrate_thickness: Silicon thickness of each chip in μm.
        bump_thickness: Gap height between the chips (bump height) in μm.

    Returns:
        A ``gsim.common.stack.LayerStack`` (requires gsim from the ``models``
        extra).
    """
    from gsim.common.stack import Layer, LayerStack
    from gsim.common.stack.materials import MATERIALS_DB

    silicon = material_properties["Si"]
    stack = LayerStack(pdk_name="qpdk")
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=FLIP_CHIP_FEM_LAYERS["SUBSTRATE"],
        zmin=-substrate_thickness,
        zmax=0.0,
        thickness=substrate_thickness,
        material="qpdk-silicon",
        layer_type="dielectric",
    )
    stack.layers["M1"] = Layer(
        name="M1",
        gds_layer=FLIP_CHIP_FEM_LAYERS["M1"],
        zmin=0.0,
        zmax=0.0,
        thickness=0,
        material="qpdk-niobium",
        layer_type="conductor",
    )
    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=FLIP_CHIP_FEM_LAYERS["VACUUM"],
        zmin=0.0,
        zmax=bump_thickness,
        thickness=bump_thickness,
        material="vacuum",
        layer_type="dielectric",
    )
    stack.layers["BUMP"] = Layer(
        name="BUMP",
        gds_layer=FLIP_CHIP_FEM_LAYERS["BUMP"],
        zmin=0.0,
        zmax=bump_thickness,
        thickness=bump_thickness,
        material="qpdk-indium",
        layer_type="via",
    )
    stack.layers["M2"] = Layer(
        name="M2",
        gds_layer=FLIP_CHIP_FEM_LAYERS["M2"],
        zmin=bump_thickness,
        zmax=bump_thickness,
        thickness=0,
        material="qpdk-niobium",
        layer_type="conductor",
    )
    stack.layers["SUBSTRATE_TOP"] = Layer(
        name="SUBSTRATE_TOP",
        gds_layer=FLIP_CHIP_FEM_LAYERS["SUBSTRATE_TOP"],
        zmin=bump_thickness,
        zmax=bump_thickness + substrate_thickness,
        thickness=substrate_thickness,
        material="qpdk-silicon",
        layer_type="dielectric",
    )
    stack.materials = {
        "qpdk-silicon": {
            "permittivity": silicon["relative_permittivity"],
            "loss_tangent": silicon["loss_tangent"],
        },
        "qpdk-niobium": {"permittivity": 1.0},
        # Indium σ from literature; without a conductivity gsim degrades the
        # via to a 2-D PEC sheet at its base, which no longer bridges the
        # two conductor planes.
        "qpdk-indium": {"conductivity": 1.16e7},
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }
    return stack
