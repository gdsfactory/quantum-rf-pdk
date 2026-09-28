r"""Through-silicon via (TSV) component library for gdsfactory.

The geometry follows the superconducting TSVs of
:cite:`mallekFabricationSuperconductingThroughsilicon2021`: slot-shaped vias
(:math:`10 \times 20\,\text{µm}`) lined with superconducting TiN, connecting
the front-side metal (side A, ``M1``) to the backside metal (side B, ``MB``).
Backside geometry is drawn in the same (front-side, top-down) coordinate frame
as ``M1``, so the backside mask is the mirror image of the drawn ``MB`` layers.
"""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import add_etch_gap, straight, taper_cross_section
from qpdk.tech import LAYER, coplanar_waveguide, get_etch_section


def _stadium_points(width: float, length: float, n: int = 32) -> np.ndarray:
    """Points of a stadium (slot) of total ``length`` along x, centred at the origin."""
    r = width / 2
    dx = max(length - width, 0.0) / 2
    t = np.linspace(-np.pi / 2, np.pi / 2, n)
    right = np.column_stack([dx + r * np.cos(t), r * np.sin(t)])
    left = np.column_stack([-dx - r * np.cos(t), -r * np.sin(t)])
    return np.vstack([right, left])


@gf.cell(tags=("interconnects", "3d-integration"))
def tsv(
    width: float = 10.0,
    length: float = 20.0,
    pad_margin: float = 2.0,
    layer: LayerSpec = LAYER.TSV,
    layer_pad_top: LayerSpec | None = LAYER.M1_DRAW,
    layer_pad_bottom: LayerSpec | None = LAYER.MB_DRAW,
) -> Component:
    r"""Creates a slot-shaped superconducting through-silicon via (TSV).

    The default :math:`10 \times 20\,\text{µm}` slot is the TSV of
    :cite:`mallekFabricationSuperconductingThroughsilicon2021`, see also
    :cite:`yostSolidstateQubitsIntegrated2020`. Use ``width == length`` for a
    round via.

    Landing pads on the front- and backside metal enclose the via by
    ``pad_margin``, so the via stays covered by metal even when placed next to
    an etched gap.

    .. svgbob::

         .-----------.
        (   TSV slot  )  ── center
         '-----------'

    Args:
        width: Via width (short axis, along y) in µm.
        length: Via length (long axis, along x) in µm.
        pad_margin: Landing-pad enclosure of the via on each metal layer in µm.
        layer: Layer of the via.
        layer_pad_top: Front-side landing pad layer. ``None`` omits the pad.
        layer_pad_bottom: Backside landing pad layer. ``None`` omits the pad.

    Returns:
        A gdsfactory Component representing the TSV.

    Raises:
        ValueError: If ``length`` is smaller than ``width``.
    """
    if length < width:
        msg = f"TSV length ({length}) must be at least its width ({width})."
        raise ValueError(msg)
    c = Component()
    c.add_polygon(_stadium_points(width, length), layer=layer)
    for pad_layer in (layer_pad_top, layer_pad_bottom):
        if pad_layer is not None and pad_margin > 0:
            c.add_polygon(
                _stadium_points(width + 2 * pad_margin, length + 2 * pad_margin),
                layer=pad_layer,
            )
    c.add_port(
        name="center",
        center=(0, 0),
        orientation=0,
        layer=layer,
        width=width,
        port_type="placement",
    )
    return c


def _cpw_pad_cross_section(
    cross_section: CrossSectionSpec, width: float, gap: float
) -> gf.CrossSection:
    """CPW cross-section with the layers of ``cross_section`` and a new width and gap."""
    xs = gf.get_cross_section(cross_section)
    return coplanar_waveguide(
        width=width,
        gap=gap,
        waveguide_layer=xs.layer,
        etch_layer=get_etch_section(xs).layer,
        radius=None,
    )


def _add_tapered_pad(
    c: Component,
    cross_section: CrossSectionSpec,
    pad_width: float,
    pad_gap: float,
    pad_length: float,
    taper_length: float,
    mirror: bool,
) -> gf.Port:
    """Add a CPW taper into a closed pad centred at the origin.

    The taper enters from ``-x`` (``+x`` when ``mirror``) and the pad is closed by
    an etch gap on the far side.

    Returns:
        The CPW port at the narrow end of the taper.
    """
    pad_xs = _cpw_pad_cross_section(cross_section, pad_width, pad_gap)
    pad = c << straight(length=pad_length, cross_section=pad_xs)
    pad.move((-pad_length / 2, 0))
    if mirror:
        pad.mirror_x(0)
    taper = c << taper_cross_section(
        length=taper_length,
        cross_section1=cross_section,
        cross_section2=pad_xs,
        linear=True,
    )
    taper.connect("o2", pad.ports["o1"])
    # The pad ends in a closing gap, so it has no external port. Flatten it (and
    # the gap) so their instance ports cannot coincide with those of the pad on
    # the other side of the substrate.
    add_etch_gap(c, pad.ports["o2"], pad_xs).flatten()
    pad.flatten()
    return taper.ports["o1"]


def _add_transition_vias(
    c: Component,
    pad_width: float,
    pad_gap: float,
    pad_length: float,
    via_width: float,
    via_length: float,
    via_pad_margin: float,
    n_signal_vias: int,
    signal_via_pitch: float,
    n_ground_vias_per_side: int,
    ground_via_pitch: float,
    ground_via_distance: float,
) -> None:
    """Add the signal TSVs inside the pad and the ground-stitching TSVs around it."""
    signal_span = (n_signal_vias - 1) * signal_via_pitch + via_width
    if (
        signal_span + 2 * via_pad_margin > pad_width
        or via_length + 2 * via_pad_margin > pad_length
    ):
        msg = (
            f"{n_signal_vias} signal TSV(s) of {via_width}x{via_length} µm at pitch "
            f"{signal_via_pitch} µm do not fit in a {pad_length}x{pad_width} µm pad."
        )
        raise ValueError(msg)
    if ground_via_distance < via_width / 2 + via_pad_margin:
        msg = (
            f"ground_via_distance={ground_via_distance} µm puts the ground TSV "
            "landing pads into the CPW gap."
        )
        raise ValueError(msg)

    via = tsv(width=via_width, length=via_length, pad_margin=via_pad_margin)
    for y in (np.arange(n_signal_vias) - (n_signal_vias - 1) / 2) * signal_via_pitch:
        (c << via).move((0, float(y)))

    y_ground = pad_width / 2 + pad_gap + ground_via_distance
    xs = (
        np.arange(n_ground_vias_per_side) - (n_ground_vias_per_side - 1) / 2
    ) * ground_via_pitch
    for x in xs:
        for y in (y_ground, -y_ground):
            (c << via).move((float(x), y))


@gf.cell(tags=("interconnects", "3d-integration", "waveguides"))
def tsv_transition(
    cross_section: CrossSectionSpec = "cpw",
    taper_length: float = 100.0,
    pad_width: float = 40.0,
    pad_gap: float = 24.0,
    pad_length: float = 30.0,
    via_width: float = 10.0,
    via_length: float = 20.0,
    via_pad_margin: float = 2.0,
    n_signal_vias: int = 2,
    signal_via_pitch: float = 20.0,
    n_ground_vias_per_side: int = 4,
    ground_via_pitch: float = 30.0,
    ground_via_distance: float = 15.0,
) -> Component:
    r"""Front-side CPW tapered into a TSV landing pad.

    The CPW widens linearly over ``taper_length`` into a pad carrying
    ``n_signal_vias`` parallel signal TSVs, surrounded by
    ``2 * n_ground_vias_per_side`` TSVs stitching the front and back ground
    planes. On the backside, the signal TSVs land on an isolated ``MB`` pad of
    the same size, clear of the backside ground by ``pad_gap``. Connect other
    backside structures at the ``backside`` placement port.

    The defaults follow the transition of
    :cite:`mallekFabricationSuperconductingThroughsilicon2021`, Fig. 6: two
    signal TSVs in parallel (a single TSV gives a large impedance mismatch) and
    eight ground TSVs. The pad gap keeps the pad close to the aspect ratio
    :math:`w / (w + 2s)` of the default :math:`50\,\Omega` CPW; the taper and
    pad dimensions are the knobs to optimise the transition impedance, see the
    ``palace_tsv_transition`` notebook.

    .. svgbob::

                  o   o   o   o      <- ground TSVs
               ___________________
        o1 ── /      [==]        |
              \      [==]        |   <- signal TSVs in pad
               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  o   o   o   o

    Args:
        cross_section: Front-side CPW cross-section at the ``o1`` port.
        taper_length: Length of the linear taper from the CPW to the pad in µm.
        pad_width: Centre-conductor width of the pad in µm.
        pad_gap: Gap between the pad and ground in µm.
        pad_length: Length of the pad along the propagation direction in µm.
        via_width: TSV width in µm.
        via_length: TSV length in µm.
        via_pad_margin: TSV landing-pad enclosure in µm.
        n_signal_vias: Number of parallel signal TSVs in the pad.
        signal_via_pitch: Centre-to-centre pitch of the signal TSVs (along y) in µm.
        n_ground_vias_per_side: Number of ground TSVs on each side of the pad.
        ground_via_pitch: Centre-to-centre pitch of the ground TSVs (along x) in µm.
        ground_via_distance: Distance from the ground-plane edge to the ground
            TSV centres in µm.

    Returns:
        Component with the CPW port ``o1`` and the ``backside`` placement port
        at the signal TSVs.
    """
    c = Component()
    port = _add_tapered_pad(
        c, cross_section, pad_width, pad_gap, pad_length, taper_length, mirror=False
    )
    # Isolated backside landing pad for the signal TSVs
    backside_pad_xs = _cpw_pad_cross_section("cpw_backside", pad_width, pad_gap)
    backside_pad = c << straight(length=pad_length, cross_section=backside_pad_xs)
    backside_pad.move((-pad_length / 2, 0))
    for p in backside_pad.ports:
        add_etch_gap(c, p, backside_pad_xs).flatten()
    # The pad is only reached through the TSVs, so it contributes polygons but
    # no ports (which would otherwise overlap the frontside taper port).
    backside_pad.flatten()
    _add_transition_vias(
        c,
        pad_width,
        pad_gap,
        pad_length,
        via_width,
        via_length,
        via_pad_margin,
        n_signal_vias,
        signal_via_pitch,
        n_ground_vias_per_side,
        ground_via_pitch,
        ground_via_distance,
    )
    c.add_port(name="o1", port=port)
    c.add_port(
        name="backside",
        center=(0, 0),
        orientation=0,
        layer=LAYER.MB_DRAW,
        width=pad_width,
        port_type="placement",
    )
    return c


@gf.cell(tags=("interconnects", "3d-integration", "waveguides"))
def tsv_transition_double_sided(
    cross_section: CrossSectionSpec = "cpw",
    cross_section_backside: CrossSectionSpec = "cpw_backside",
    taper_length: float = 100.0,
    taper_length_backside: float = 100.0,
    pad_width: float = 40.0,
    pad_gap: float = 24.0,
    pad_length: float = 30.0,
    via_width: float = 10.0,
    via_length: float = 20.0,
    via_pad_margin: float = 2.0,
    n_signal_vias: int = 2,
    signal_via_pitch: float = 20.0,
    n_ground_vias_per_side: int = 4,
    ground_via_pitch: float = 30.0,
    ground_via_distance: float = 15.0,
) -> Component:
    r"""CPW transition from the front side to the backside through TSVs.

    A front-side CPW (``o1``, on ``M1``) tapers into a pad, the signal passes
    through ``n_signal_vias`` parallel TSVs, and a backside CPW (``o2``, on
    ``MB``) tapers out of an identical pad in the opposite direction.
    ``2 * n_ground_vias_per_side`` TSVs stitch the front and back ground planes
    around the pad. This is the side A to side B transition of
    :cite:`mallekFabricationSuperconductingThroughsilicon2021`, Fig. 6.

    Simulate it with :data:`~qpdk.tech.LAYER_STACK_BACKSIDE`; the backside
    metal also turns every CPW on the chip into a conductor-backed CPW, see
    the ``conductor_backed`` option of :func:`~qpdk.models.cpw_parameters`.

    .. svgbob::

         front (M1)              back (MB)
               ______________________
        o1 ── /     [==]            \ ── o2
              \     [==]            /
               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    Args:
        cross_section: Front-side CPW cross-section at ``o1``.
        cross_section_backside: Backside CPW cross-section at ``o2``.
        taper_length: Length of the front-side taper in µm.
        taper_length_backside: Length of the backside taper in µm.
        pad_width: Centre-conductor width of both pads in µm.
        pad_gap: Gap between the pads and ground in µm.
        pad_length: Length of the pads along the propagation direction in µm.
        via_width: TSV width in µm.
        via_length: TSV length in µm.
        via_pad_margin: TSV landing-pad enclosure in µm.
        n_signal_vias: Number of parallel signal TSVs.
        signal_via_pitch: Centre-to-centre pitch of the signal TSVs (along y) in µm.
        n_ground_vias_per_side: Number of ground TSVs on each side of the pads.
        ground_via_pitch: Centre-to-centre pitch of the ground TSVs (along x) in µm.
        ground_via_distance: Distance from the ground-plane edge to the ground
            TSV centres in µm.

    Returns:
        Component with the front-side port ``o1`` and the backside port ``o2``.
    """
    c = Component()
    port_front = _add_tapered_pad(
        c, cross_section, pad_width, pad_gap, pad_length, taper_length, mirror=False
    )
    port_back = _add_tapered_pad(
        c,
        cross_section_backside,
        pad_width,
        pad_gap,
        pad_length,
        taper_length_backside,
        mirror=True,
    )
    _add_transition_vias(
        c,
        pad_width,
        pad_gap,
        pad_length,
        via_width,
        via_length,
        via_pad_margin,
        n_signal_vias,
        signal_via_pitch,
        n_ground_vias_per_side,
        ground_via_pitch,
        ground_via_distance,
    )
    c.add_port(name="o1", port=port_front)
    c.add_port(name="o2", port=port_back)
    return c
