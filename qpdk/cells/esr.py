"""Planar LC resonator for inductive-detection electron spin resonance."""

from __future__ import annotations

import math

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

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


@gf.cell(tags=["resonators", "inductors", "capacitors"])
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

    An :func:`~qpdk.cells.capacitor.interdigital_capacitor` is shunted by a
    short, narrow wire that forms the inductance. The microwave magnetic field
    is concentrated around the wire, where the spins sit
    :cite:`bienfaitReachingQuantumLimit2016,ranjanElectronSpinResonance2020`.

    The capacitor is rotated so that its bus bars run parallel to a CPW
    feedline. The upper bus bar sits ``coupling_gap`` below the feedline centre
    conductor and couples to it capacitively. The lower bus bar is tied to the
    ground plane along its bottom edge. The wire leaves the outermost finger of
    the upper electrode at the finger's mid-length and runs ``wire_length`` to
    the ground plane, so the resonator is a parallel LC to ground,
    shunt-coupled to the feedline:

    .. svgbob::

        o1 ─────────────────────────────────── o2   feedline
             ┌─────────────────────┐
             │ ┌──┐ ┌──┐ ┌──┐ ┌──┐ │
             │ │  │ │  │ │  │ │  ├─────wire─────┤ ground
             │ │  └─┘  └─┘  └─┘  │ │
             └─┴─────────────────┴─┘
        ─────────────────────────────────────── ground

    If ``wire_length`` is shorter than ``ground_gap``, a ground tongue as wide
    as a finger reaches in from the right-hand ground plane to the end of the
    wire. In both references the resonator floats in a 3D cavity and is driven
    by antennas; here one electrode is grounded and the coupling is to an
    on-chip CPW.

    ``info["wire_squares"]`` is ``wire_length / wire_width``.
    ``info["mode_volume"]`` is :func:`qpdk.models.esr.esr_mode_volume` in µm³
    (1 µm³ = 1 fL), evaluated with the geometric inductance of the wire alone
    for a :data:`ESR_FILM_THICKNESS` film. That counts only the field energy
    of the wire, so the stray inductance of the capacitor electrodes is
    excluded. The matching SAX model is :func:`qpdk.models.esr_lc_resonator`.

    Args:
        wire_width: Wire width in µm, between 0.1 and 5.
        wire_length: Wire length in µm between the capacitor finger and the
            ground plane.
        fingers: Number of capacitor fingers (at least 2).
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger and of the bus bars
            in µm.
        coupling_gap: Gap between the feedline centre conductor and the upper
            bus bar in µm.
        ground_gap: Gap between the capacitor and the ground plane on the left,
            and the minimum gap on the right, in µm.
        feedline_length: Length of the feedline in µm.
        cross_section: Cross-section of the feedline.

    Returns:
        Component with feedline ports ``o1`` and ``o2``.

    Raises:
        ValueError: If the wire width is out of range, ``fingers`` is below 2,
            a length is not positive, the wire is wider than a finger is long,
            the bus-bar length is off the 2 nm grid, or the feedline is
            shorter than the etch window.
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
    if wire_width > finger_length:
        raise ValueError(
            f"wire_width={wire_width} must not exceed finger_length={finger_length}"
        )

    xs = gf.get_cross_section(cross_section)
    metal_layer = xs.layer
    etch_layer = get_etch_section(xs).layer
    centre_width = xs.width

    # Extent of the capacitor along the feedline (bus bar length) and across it.
    # The capacitor's port stubs are as wide as the bus bars, so they only
    # thicken each bus bar by one finger thickness.
    bar_length = fingers * finger_thickness + (fingers - 1) * finger_gap
    idc_height = 4 * finger_thickness + finger_length + finger_gap

    # The bus-bar length becomes a port width, which must be a 2 nm multiple.
    bar_nm = bar_length * 1e3
    if abs(bar_nm - 2 * round(bar_nm / 2)) > 1e-6:
        raise ValueError(
            f"fingers*finger_thickness + (fingers-1)*finger_gap = {bar_length} µm "
            "must be a multiple of 0.002 µm"
        )

    right_gap = max(wire_length, ground_gap)
    window_width = ground_gap + bar_length + right_gap
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
        y0=idc_bottom,
        y1=-centre_width / 2,
    )

    # Mid-length of the right-most finger, which hangs from the upper bus bar.
    wire_y = idc_top - 2 * finger_thickness - finger_length / 2
    wire_end = idc_right + wire_length
    add_rect(
        c,
        metal_layer,
        x0=idc_right,
        x1=wire_end,
        y_center=wire_y,
        height=wire_width,
    )
    if wire_end < window_width / 2:
        add_rect(
            c,
            metal_layer,
            x0=wire_end,
            x1=window_width / 2,
            y_center=wire_y,
            height=finger_thickness,
        )

    c.add_ports(feed.ports)
    c.info["wire_squares"] = wire_length / wire_width
    c.info["mode_volume"] = _wire_mode_volume(wire_width, wire_length)
    return c
