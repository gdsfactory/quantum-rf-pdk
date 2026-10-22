"""Facing coupling pads in the finite ground environment used by the FEM model."""

import gdsfactory as gf
from gdsfactory.component import Component

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.tech import LAYER

__all__ = ["coupling_pad_pair"]


@gf.cell(tags=["capacitors", "couplers"])
def coupling_pad_pair(
    length: float = 80.0,
    width: float = 10.0,
    gap: float = 7.0,
    offset: float = 0.0,
) -> Component:
    """Two facing plate capacitor pads with the dataset's coplanar ground frame.

    Reuses two :func:`plate_capacitor_single` cells. The first pad is centered
    at the origin and the second is shifted vertically by ``offset``. Ports
    ``o1`` and ``o2`` lie at the outer ends of the pads' short CPW necks.
    A grounded frame, 45 um wide and 10 um from the electrode bounding box,
    matches :func:`qpdk.models.coupling_pad_pair` on silicon. The frame is
    part of the modeled environment and surrounds both port reference planes.

    The model includes the coupling and both capacitances to ground. Its
    validated dimensions are length 40 to 120 um, width 5 to 20 um, gap 4 to
    10 um and offset -12 to 12 um. The lookup returns NaN outside that range.

    Args:
        length: Vertical pad length in um.
        width: Horizontal pad width in um.
        gap: Horizontal separation between facing pad edges in um.
        offset: Vertical displacement of the second pad in um.

    Returns:
        Component with two signal ports and an implicit common ground.

    Raises:
        ValueError: If length, width or gap is not positive.
    """
    if min(length, width, gap) <= 0:
        raise ValueError("Pad dimensions and gap must be positive")
    component = Component()
    pad = plate_capacitor_single(length=length, width=width)
    first = component.add_ref(pad)
    second = component.add_ref(pad)
    second.drotate(180)
    second.dmove((width + gap, offset))
    component.add_port("o1", port=first.ports["o1"])
    component.add_port("o2", port=second.ports["o1"])

    dbu = component.kcl.dbu
    electrodes = component.get_region(LAYER.M1_DRAW).merged()
    inner = gf.kdb.Region(electrodes.bbox().enlarged(round(10.0 / dbu)))
    outer = gf.kdb.Region(inner.bbox().enlarged(round(45.0 / dbu)))
    for polygon in (outer - inner).each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for polygon in (inner - electrodes).each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_ETCH)
    return component
