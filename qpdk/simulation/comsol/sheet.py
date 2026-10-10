r"""Build air and silicon domains with QPDK metal sheets at their interface.

The builder lives in :mod:`gplugins.comsol.sheet`. This wrapper only swaps in
the QPDK technology defaults for the dielectric permittivities, taken from the
materials of the layer stack's substrate and vacuum levels. Physics and studies
are added separately by :func:`~qpdk.simulation.comsol.rf.add_cpw_rf_study` or
:func:`~qpdk.simulation.comsol.capacitance.add_capacitance_study`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.simulation.comsol._gplugins import import_gplugins_comsol, reexport
from qpdk.tech import get_layer_material_properties

if TYPE_CHECKING:
    import mph
    from gplugins.comsol.sheet import AIR_SELECTION, SILICON_SELECTION

    from qpdk.simulation.comsol.layout import ComsolLayout

__all__ = [
    "AIR_RELATIVE_PERMITTIVITY",
    "AIR_SELECTION",
    "SILICON_RELATIVE_PERMITTIVITY",
    "SILICON_SELECTION",
    "build_comsol_sheet_model",
]

#: Canonical relative permittivity of the silicon substrate and of the air above
#: it, from the materials of the QPDK layer stack's substrate and vacuum levels.
SILICON_RELATIVE_PERMITTIVITY = get_layer_material_properties("Substrate")[
    "relative_permittivity"
]
AIR_RELATIVE_PERMITTIVITY = get_layer_material_properties("Vacuum")[
    "relative_permittivity"
]

__getattr__ = reexport(__name__, "sheet", ["AIR_SELECTION", "SILICON_SELECTION"])


def build_comsol_sheet_model(
    client: mph.Client,
    layout: ComsolLayout,
    name: str,
    *,
    substrate_thickness_um: float = 200.0,
    air_height_um: float = 200.0,
    lateral_margin_um: float = 0.0,
    silicon_relative_permittivity: float = SILICON_RELATIVE_PERMITTIVITY,
    air_relative_permittivity: float = AIR_RELATIVE_PERMITTIVITY,
) -> mph.Model:
    """Create a COMSOL 3D model holding the layout's metal as interface sheets.

    Calls :func:`gplugins.comsol.sheet.build_comsol_sheet_model` with the QPDK
    technology permittivities as defaults; see there for the geometry built and
    the licences it needs.

    Args:
        client: A connected :class:`mph.Client`.
        layout: Extracted metal polygons and bounding box in µm.
        name: Name of the COMSOL model.
        substrate_thickness_um: Silicon thickness below the interface, in µm.
        air_height_um: Air height above the interface, in µm.
        lateral_margin_um: Margin around the layout bounding box for both
            blocks, in µm.
        silicon_relative_permittivity: Relative permittivity of the silicon
            block, positive and finite. Defaults to the QPDK technology value
            for Si.
        air_relative_permittivity: Relative permittivity of the air block,
            positive and finite. Defaults to the QPDK technology value for
            vacuum.

    Returns:
        The MPh model, with geometry, selections, and materials only.
    """
    return import_gplugins_comsol("sheet").build_comsol_sheet_model(
        client,
        layout,
        name,
        substrate_thickness_um=substrate_thickness_um,
        air_height_um=air_height_um,
        lateral_margin_um=lateral_margin_um,
        silicon_relative_permittivity=silicon_relative_permittivity,
        air_relative_permittivity=air_relative_permittivity,
    )
