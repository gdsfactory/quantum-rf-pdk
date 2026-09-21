"""Volumetric-FEM helpers for qpdk layouts.

Converts the PDK's subtractive etch masks into the explicit conductor and
dielectric regions that full-wave 3-D solvers (Palace through gsim, Meep, ...)
expect, and provides the matching single-chip layer stack. gsim is not a
qpdk dependency, so it is imported lazily by the functions that need it.
"""

from __future__ import annotations

from typing import Any

import gdsfactory as gf
import klayout.db as kdb

from qpdk.tech import LAYER, material_properties
from qpdk.utils import apply_additive_metals

__all__ = ["FEM_LAYERS", "single_chip_stack", "to_fem_regions"]

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
    processed = apply_additive_metals(component.copy())

    layout = processed.kdb_cell.layout()
    sim_region = kdb.Region(
        processed.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.SIM_AREA))
    )
    etch_region = kdb.Region(
        processed.kdb_cell.begin_shapes_rec(layout.layer(*LAYER.M1_ETCH))
    )
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
    for port in processed.ports:
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
        A ``gsim.common.stack.LayerStack`` (requires gsim, which is not a
        qpdk dependency).
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
        material="silicon",
        layer_type="dielectric",
    )
    stack.layers["SUPERCONDUCTOR"] = Layer(
        name="SUPERCONDUCTOR",
        gds_layer=FEM_LAYERS["SUPERCONDUCTOR"],
        zmin=substrate_thickness,
        zmax=substrate_thickness,
        thickness=0,
        material="niobium",
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
        "silicon": {
            "permittivity": silicon["relative_permittivity"],
            "loss_tangent": silicon["loss_tangent"],
        },
        "niobium": {"permittivity": 1.0},
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }
    return stack
