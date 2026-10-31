"""Planar LC resonator for inductive-detection electron spin resonance."""

from __future__ import annotations

import math

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

from qpdk.cells._schematic import esr_lc_resonator_schematic
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.waveguides import straight
from qpdk.tech import get_etch_section
from qpdk.utils import add_rect

__all__ = ["esr_lc_resonator"]

ESR_WIRE_WIDTH_RANGE = (0.1, 5.0)
"""Allowed wire width range in µm."""

ESR_FILM_THICKNESS = 0.05
"""Film thickness in µm used for the ``mode_volume`` estimate in ``info``.

Both design references pattern the resonator from a 50 nm aluminium film.
"""

_MU_0 = 4e-7 * math.pi
"""Vacuum permeability in H/m, to the precision of the strip formula."""


def _wire_mode_volume(wire_width: float, wire_length: float) -> float:
    r"""Mode volume in µm³ from the geometric inductance of the wire alone.

    Pure-Python twin of :func:`qpdk.models.esr.esr_mode_volume` evaluated with
    :func:`qpdk.models.inductor.self_inductance_strip`, so the cell does not
    import the JAX models. ``tests/models/test_esr.py`` checks the two agree.

    Returns:
        Mode volume in µm³.
    """
    length, width, thickness = (
        wire_length * 1e-6,
        wire_width * 1e-6,
        ESR_FILM_THICKNESS * 1e-6,
    )
    inductance = (_MU_0 * length / (2 * math.pi)) * (
        math.log(2 * length / (width + thickness))
        + 0.5
        + (width + thickness) / (3 * length)
    )
    return 4 * wire_width**2 * inductance / _MU_0 * 1e6


@gf.cell(
    tags=["resonators", "inductors", "capacitors"],
    schematic_function=esr_lc_resonator_schematic,
)
def esr_lc_resonator(
    wire_width: float = 5.0,
    wire_length: float = 730.0,
    fingers: int = 12,
    finger_length: float = 800.0,
    finger_gap: float = 50.0,
    finger_thickness: float = 50.0,
    coupling_gap: float = 50.0,
    ground_gap: float = 50.0,
    feedline_length: float = 2000.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Planar LC resonator for inductive-detection electron spin resonance.

    A floating :func:`~qpdk.cells.capacitor.interdigital_capacitor` whose two
    electrodes are bridged by a short, narrow wire that forms the inductance.
    The microwave magnetic field is concentrated around the wire, where the
    spins sit :cite:`bienfaitReachingQuantumLimit2016,ranjanElectronSpinResonance2020`.

    The capacitor is rotated so that its bus bars run parallel to a CPW
    feedline. The upper bus bar sits ``coupling_gap`` below the feedline centre
    conductor and couples to it capacitively. Both bus bars extend to the
    right past the last finger to a rail, one ``finger_gap`` away, that joins
    them: a lead as wide as a finger runs from each bus bar towards the middle
    of the rail, and the wire spans the remaining ``wire_length`` between the
    two leads. Neither electrode touches the ground plane, so the resonator
    couples to ground only through its stray capacitances:

    .. svgbob::

        o1 ─────────────────────────────────── o2   feedline
             ┌─────────────────────────┐
             │ ┌──┐ ┌──┐ ┌──┐ ┌──┐  ┌──┤
             │ │  │ │  │ │  │ │  │  │  │ lead
             │ │  │ │  │ │  │ │  │  │  │
             │ │  │ │  │ │  │ │  │  └┐┌┘
             │ │  │ │  │ │  │ │  │   ││  wire
             │ │  │ │  │ │  │ │  │  ┌┘└┐
             │ │  └─┘  └─┘  └─┘  │  │  │ lead
             └─┴─────────────────┴──┴──┘
        ─────────────────────────────────────── ground

    As in :cite:`ranjanElectronSpinResonance2020`, the wire sits midway between
    the two electrodes. Both references float the resonator in a 3D cavity and
    drive it with antennas; here it is coupled to an on-chip CPW instead.

    ``info["wire_squares"]`` is ``wire_length / wire_width``.
    ``info["mode_volume"]`` is :func:`qpdk.models.esr.esr_mode_volume` in µm³
    (1 µm³ = 1 fL), evaluated with the geometric inductance of the wire alone
    for a :data:`ESR_FILM_THICKNESS` film. That counts only the field energy
    of the wire, so the stray inductance of the leads and the capacitor
    electrodes is excluded. The matching SAX model is
    :func:`qpdk.models.esr_lc_resonator`.

    Args:
        wire_width: Wire width in µm, between 0.1 and 5.
        wire_length: Wire length in µm, at most
            ``finger_length + finger_gap``, the span between the bus bars.
        fingers: Number of capacitor fingers (at least 2).
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers, and between the
            last finger and the rail, in µm.
        finger_thickness: Width of each capacitor finger, of the bus bars and
            of the leads in µm.
        coupling_gap: Gap between the feedline centre conductor and the upper
            bus bar in µm.
        ground_gap: Gap between the resonator and the surrounding ground plane
            in µm.
        feedline_length: Length of the feedline in µm.
        cross_section: Cross-section of the feedline.

    Returns:
        Component with feedline ports ``o1`` and ``o2``.

    Raises:
        ValueError: If the wire width is out of range, ``fingers`` is below 2,
            a length is not positive, the wire is wider than a lead or longer
            than the span between the bus bars, the bus-bar length is off the
            2 nm grid, or the feedline is shorter than the etch window.
    """
    w_min, w_max = ESR_WIRE_WIDTH_RANGE
    if not w_min <= wire_width <= w_max:
        raise ValueError(
            f"wire_width must be between {w_min} and {w_max} µm, got {wire_width}"
        )
    if fingers < 2:
        raise ValueError(f"fingers must be at least 2, got {fingers}")
    for name, value in {
        "wire_length": wire_length,
        "finger_length": finger_length,
        "finger_gap": finger_gap,
        "finger_thickness": finger_thickness,
        "coupling_gap": coupling_gap,
        "ground_gap": ground_gap,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if wire_width > finger_thickness:
        raise ValueError(
            f"wire_width={wire_width} must not exceed "
            f"finger_thickness={finger_thickness}"
        )
    span = finger_length + finger_gap
    if wire_length > span:
        raise ValueError(
            f"wire_length={wire_length} must not exceed the span between the bus "
            f"bars, finger_length + finger_gap = {span}"
        )

    xs = gf.get_cross_section(cross_section)
    metal_layer = xs.layer
    etch_layer = get_etch_section(xs).layer
    centre_width = xs.width

    # Extent of the capacitor along the feedline (bus bar length) and across it.
    # The capacitor's port stubs are as wide as the bus bars, so they only
    # thicken each bus bar by one finger thickness.
    bar_length = fingers * finger_thickness + (fingers - 1) * finger_gap
    bar_height = 2 * finger_thickness
    idc_height = 2 * bar_height + span

    # The bus-bar length becomes a port width, which must be a 2 nm multiple.
    bar_nm = bar_length * 1e3
    if abs(bar_nm - 2 * round(bar_nm / 2)) > 1e-6:
        raise ValueError(
            f"fingers*finger_thickness + (fingers-1)*finger_gap = {bar_length} µm "
            "must be a multiple of 0.002 µm"
        )

    resonator_width = bar_length + finger_gap + finger_thickness
    window_width = 2 * ground_gap + resonator_width
    if feedline_length < window_width:
        raise ValueError(
            f"feedline_length={feedline_length} must be at least the etch window "
            f"width {window_width}"
        )

    idc = interdigital_capacitor(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
        layer_metal=metal_layer,
        etch_layer=None,
        etch_bbox_margin=finger_thickness,
        cross_section=gf.cross_section.cross_section(
            width=bar_length, layer=metal_layer
        ),
    )

    c = Component()

    feed = c << straight(length=feedline_length, cross_section=cross_section)
    feed.dcenter = (0, 0)

    idc_top = -centre_width / 2 - coupling_gap
    idc_bottom = idc_top - idc_height
    idc_left = -window_width / 2 + ground_gap
    idc_right = idc_left + bar_length

    idc_ref = c << idc
    # Clockwise turn: the electrode holding the first finger becomes the upper
    # bus bar, and that finger becomes the right-most one.
    idc_ref.rotate(-90)
    idc_ref.dcenter = ((idc_left + idc_right) / 2, (idc_top + idc_bottom) / 2)

    add_rect(
        c,
        etch_layer,
        x0=-window_width / 2,
        x1=window_width / 2,
        y0=idc_bottom - ground_gap,
        y1=-centre_width / 2,
    )

    # Bus-bar extensions out to the rail.
    rail_left = idc_right + finger_gap
    rail_right = rail_left + finger_thickness
    for y0 in (idc_top - bar_height, idc_bottom):
        add_rect(c, metal_layer, x0=idc_right, x1=rail_right, y0=y0, y1=y0 + bar_height)

    # Leads from each bus bar to the wire, which sits midway between the bars.
    span_top = idc_top - bar_height
    span_bottom = idc_bottom + bar_height
    span_mid = (span_top + span_bottom) / 2
    lead_length = (span - wire_length) / 2
    if lead_length > 0:
        for y0 in (span_top - lead_length, span_bottom):
            add_rect(
                c,
                metal_layer,
                x0=rail_left,
                x1=rail_right,
                y0=y0,
                y1=y0 + lead_length,
            )
    add_rect(
        c,
        metal_layer,
        x_center=(rail_left + rail_right) / 2,
        width=wire_width,
        y_center=span_mid,
        height=wire_length,
    )

    c.add_ports(feed.ports)
    c.info["wire_squares"] = wire_length / wire_width
    c.info["mode_volume"] = _wire_mode_volume(wire_width, wire_length)
    return c
