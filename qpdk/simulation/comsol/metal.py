"""Build unsolved extruded metal geometry from an extracted QPDK layout.

The builder lives in :mod:`gplugins.comsol.metal`; this wrapper only keeps the
QPDK default model name. :class:`~qpdk.simulation.comsol.model.COMSOL` builds a
model through it; the sheet builder for RF and electrostatic studies lives in
:mod:`qpdk.simulation.comsol.sheet`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplugins.comsol.metal import build_comsol_metal_model as _build_metal_model

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol.layout import ComsolLayout

__all__ = ["build_comsol_metal_model"]


def build_comsol_metal_model(
    client: mph.Client,
    layout: ComsolLayout,
    *,
    metal_thickness_um: float = 0.2,
    name: str = "QPDK metal",
) -> mph.Model:
    """Create a COMSOL 3D model holding the layout's metal as extruded polygons.

    Calls :func:`gplugins.comsol.metal.build_comsol_metal_model` with the QPDK
    default model name.

    Args:
        client: A connected :class:`mph.Client`.
        layout: Extracted metal polygons and feed ports in µm.
        metal_thickness_um: Extrusion height in µm, strictly positive.
        name: Name of the COMSOL model.

    Returns:
        The MPh model. Geometry only: no physics, materials, ports, or studies
        have been added, and the model has not been saved.
    """
    return _build_metal_model(
        client, layout, metal_thickness_um=metal_thickness_um, name=name
    )
