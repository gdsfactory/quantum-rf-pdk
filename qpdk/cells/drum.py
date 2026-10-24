r"""Suspended vacuum-gap (drum) capacitor and drum LC resonator cells.

The drum is a parallel-plate capacitor whose top electrode is an aluminium
membrane suspended a vacuum gap above a bottom electrode in ``M1``
:cite:`cicakLowlossSuperconductingResonant2010,teufelCircuitCavityElectromechanics2011`.
The membrane is drawn on ``VG_TOP`` and the sacrificial layer that holds it up
during fabrication on ``VG_SAC``. Wherever ``VG_TOP`` lies outside ``VG_SAC``
the membrane lands on ``M1`` and is clamped there.

The gap is not a cell parameter. It is set by the layer stack, see
:func:`qpdk.tech.get_vacuum_gap`.
"""

from __future__ import annotations

import math
from typing import Literal

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from kfactory import kdb
from klayout.db import Region

from qpdk.cells.inductor import meander_inductor
from qpdk.cells.waveguides import straight
from qpdk.tech import (
    LAYER,
    get_etch_section,
    get_vacuum_gap,
    meander_inductor_cross_section,
)

_CIRCLE_POINTS = 128


def _disc(radius: float, center: tuple[float, float] = (0.0, 0.0)) -> Region:
    """Return a polygonal disc of ``radius`` centred on ``center``."""
    x, y = center
    box = kdb.DBox(x - radius, y - radius, x + radius, y + radius)
    return Region(kdb.DPolygon.ellipse(box, _CIRCLE_POINTS).to_itype(0.001))


def _box(x0: float, y0: float, x1: float, y1: float) -> Region:
    """Return the axis-aligned box spanned by two corners."""
    return Region(kdb.DBox(x0, y0, x1, y1).to_itype(0.001))


def _layer_region(component: Component, layer: LayerSpec) -> Region:
    """Return the merged, flattened shapes of ``component`` on ``layer``."""
    layer_tuple = gf.get_layer_tuple(layer)
    return Region(
        kdb.RecursiveShapeIterator(
            component.kcl.layout,
            component._base.kdb_cell,  # pyright: ignore[reportPrivateUsage]
            component.kcl.layout.layer(layer_tuple[0], layer_tuple[1]),
        )
    ).merged()


@gf.cell(tags=["capacitors"])
def vacuum_gap_capacitor(
    radius: float = 7.5,
    support: Literal["rim", "legs"] = "rim",
    etch_gap: float = 2.0,
    rim_width: float = 3.0,
    n_legs: int = 4,
    leg_width: float = 2.0,
    cross_section: CrossSectionSpec = meander_inductor_cross_section,
) -> Component:
    r"""Creates a suspended vacuum-gap parallel-plate (drum) capacitor.

    A circular bottom electrode of radius :math:`a` in ``M1`` sits under an
    aluminium membrane on ``VG_TOP``, separated by the vacuum gap :math:`d` of
    the layer stack, after
    :cite:`cicakLowlossSuperconductingResonant2010,teufelCircuitCavityElectromechanics2011`.
    The membrane is clamped onto an ``M1`` ring around the electrode, either
    along its whole edge (``support="rim"``) or through ``n_legs`` radial legs
    (``support="legs"``). ``VG_SAC`` marks the sacrificial layer under the
    suspended part.

    .. svgbob::

              .-----------.  ring + membrane (o2)
             /  .-------.  \
        o1 ---=|  bottom |   |--- o2
             \  '-------'  /
              '-----------'

    Port ``o1`` connects to the bottom electrode through a lead that passes
    under the membrane in a slot of the ring. Port ``o2`` connects to the ring
    and through it to the membrane. The plate overlap is set by the bottom
    electrode, so the parallel-plate capacitance is
    :math:`C = \varepsilon_0 \pi a^2 / d`, see
    :func:`qpdk.models.drum.vacuum_gap_capacitance_analytical`.

    Args:
        radius: Bottom-electrode radius :math:`a` in µm.
        support: How the membrane is held, ``"rim"`` or ``"legs"``.
        etch_gap: Etched gap in µm between electrode and ring, and between ring
            and ground.
        rim_width: Width of the ``M1`` ring the membrane lands on, in µm.
        n_legs: Number of radial legs for ``support="legs"``.
        leg_width: Width of each leg in µm for ``support="legs"``.
        cross_section: Cross-section of the two leads and ports.

    Returns:
        Component: The drum capacitor with ports ``o1`` (bottom electrode) and
            ``o2`` (membrane).

    Raises:
        ValueError: If a dimension is not positive, ``support`` is unknown,
            ``n_legs`` < 1, or a leg would land on the lead slot.
    """
    for name, value in (
        ("radius", radius),
        ("etch_gap", etch_gap),
        ("rim_width", rim_width),
        ("leg_width", leg_width),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if support not in {"rim", "legs"}:
        raise ValueError(f"support must be 'rim' or 'legs', got {support!r}")
    if support == "legs" and n_legs < 1:
        raise ValueError(f"n_legs must be at least 1, got {n_legs}")

    xs = gf.get_cross_section(cross_section)
    wire_width = xs.width
    etch_section = get_etch_section(xs)
    slot_half_width = wire_width / 2 + etch_section.width

    r_ring_in = radius + etch_gap
    r_ring_out = r_ring_in + rim_width
    r_land = (r_ring_in + r_ring_out) / 2
    r_etch = r_ring_out + etch_gap

    # The o1 lead runs west under the membrane through a slot in the ring
    slot = _box(-r_etch, -slot_half_width, 0, slot_half_width)
    ring = _disc(r_ring_out) - _disc(r_ring_in) - slot
    metal = (
        _disc(radius)
        + _box(-r_etch, -wire_width / 2, 0, wire_width / 2)
        + ring
        + _box(r_land, -wire_width / 2, r_etch, wire_width / 2)
    ).merged()
    etch = (
        _disc(r_etch) + _box(-r_etch, -slot_half_width, r_etch, slot_half_width)
    ) - metal

    sacrificial = (
        _disc(r_ring_in) + (_box(-r_land, -slot_half_width, 0, slot_half_width))
    ).merged()

    if support == "rim":
        membrane = _disc(r_land)
    else:
        membrane = _disc(radius + etch_gap / 2)
        offset = 0.5 if n_legs % 2 == 0 else 0.0
        for k in range(n_legs):
            angle = 360 * (k + offset) / n_legs
            leg = _box(0, -leg_width / 2, r_land, leg_width / 2).transformed(
                kdb.DCplxTrans(1, angle, False, 0, 0).to_itrans(0.001)
            )
            if not (leg - _disc(r_ring_in) & slot).is_empty():
                raise ValueError(
                    f"Leg at {angle:.1f}° lands on the o1 lead slot; "
                    "change n_legs or leg_width"
                )
            membrane += leg
        membrane.merge()

    c = Component()
    c.add_polygon(metal, layer=xs.layer)
    c.add_polygon(etch, layer=etch_section.layer)
    c.add_polygon(membrane, layer=LAYER.VG_TOP)
    c.add_polygon(sacrificial, layer=LAYER.VG_SAC)

    for name, x, orientation in (("o1", -r_etch, 180), ("o2", r_etch, 0)):
        c.add_port(
            name=name,
            center=(x, 0),
            width=wire_width,
            orientation=orientation,
            layer=xs.layer,
            port_type="optical",
            cross_section=xs,
        )

    c.info["vacuum_gap"] = get_vacuum_gap()
    c.info["overlap_area"] = math.pi * radius**2
    return c


@gf.cell(tags=["resonators", "capacitors", "inductors", "couplers"])
def drum_lc_resonator(
    radius: float = 7.5,
    support: Literal["rim", "legs"] = "rim",
    n_turns: int = 71,
    turn_length: float = 300.0,
    inductor_cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    coupling_gap: float = 6.0,
    coupling_pad_size: float = 20.0,
    margin: float = 10.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Creates a drum LC resonator side-coupled to a CPW feedline.

    A meander inductor is shunted to ground by a
    :func:`vacuum_gap_capacitor`, forming the mechanically compliant LC
    circuit of :cite:`teufelCircuitCavityElectromechanics2011`. Unlike that
    device, which couples inductively, the resonator here couples capacitively
    to the feedline through a pad at ``coupling_gap`` from the centre
    conductor. The bottom electrode is the hot node and the membrane is
    grounded through the drum ring.

    .. svgbob::

        o1 =================================== o2  feedline
                           [pad]
                 +------------+--+
                 |  meander   L  +--(drum)--GND
                 +------------+
                 GND

    The resonance is :math:`\omega_\text{r} = 1/\sqrt{L (C_\text{d} + C_\text{c})}`,
    see :func:`qpdk.models.drum.drum_lc_resonator`.

    Args:
        radius: Drum bottom-electrode radius in µm.
        support: How the drum membrane is held, ``"rim"`` or ``"legs"``.
        n_turns: Number of meander inductor runs. Must be odd.
        turn_length: Length of each meander run in µm.
        inductor_cross_section: Cross-section of the meander wire, the drum
            leads and the internal wiring.
        coupling_gap: Gap between the coupling pad and the feedline centre
            conductor in µm.
        coupling_pad_size: Side length of the square coupling pad in µm.
        margin: Spacing in µm between the pad, inductor, drum and the edge of
            the etched window.
        cross_section: Cross-section of the feedline.

    Returns:
        Component: The resonator with feedline ports ``o1`` and ``o2``.

    Raises:
        ValueError: If ``n_turns`` is even or a spacing is not positive.
    """
    if n_turns % 2 == 0:
        raise ValueError(
            "n_turns must be odd so that the inductor ends on the drum side"
        )
    for name, value in (
        ("coupling_gap", coupling_gap),
        ("coupling_pad_size", coupling_pad_size),
        ("margin", margin),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    feed_xs = gf.get_cross_section(cross_section)
    feed_etch = get_etch_section(feed_xs)
    wire_xs = gf.get_cross_section(inductor_cross_section)
    wire_width = wire_xs.width

    tmp = Component()
    ind = tmp << meander_inductor(
        n_turns=n_turns,
        turn_length=turn_length,
        cross_section=inductor_cross_section,
        add_etch=False,
    )
    drum = tmp << vacuum_gap_capacitor(
        radius=radius, support=support, cross_section=inductor_cross_section
    )

    # Coupling pad below the feedline, its right edge flush with the inductor end
    pad_top = -feed_xs.width / 2 - coupling_gap
    pad_bottom = pad_top - coupling_pad_size
    ind.dmove((
        0,
        pad_bottom - margin - wire_width / 2 - ind.ports["o2"].dcenter[1],
    ))
    node_x, node_y = ind.ports["o2"].dcenter
    drum.dmove((node_x + margin - drum.ports["o1"].dcenter[0], node_y))

    x_left = ind.ports["o1"].dcenter[0] - margin
    x_right = drum.ports["o2"].dcenter[0]
    y_bottom = ind.dbbox().bottom - margin
    window_top = -feed_xs.width / 2 - feed_etch.width

    feed = tmp << straight(
        length=x_right - x_left + 2 * margin, cross_section=cross_section
    )
    feed.dmove((x_left - margin, 0))

    metal = (
        _layer_region(tmp, wire_xs.layer)
        + _box(node_x - coupling_pad_size, pad_bottom, node_x, pad_top)
        # Pad to inductor end, inductor end to drum, inductor start to ground
        + _box(node_x - wire_width, node_y, node_x, pad_bottom)
        + _box(
            node_x, node_y - wire_width / 2, node_x + margin, node_y + wire_width / 2
        )
        + _box(
            x_left,
            ind.ports["o1"].dcenter[1] - wire_width / 2,
            ind.ports["o1"].dcenter[0],
            ind.ports["o1"].dcenter[1] + wire_width / 2,
        )
    ).merged()
    etch = (
        _layer_region(tmp, LAYER.M1_ETCH) + _box(x_left, y_bottom, x_right, window_top)
    ) - metal

    c = Component()
    c.add_polygon(metal, layer=wire_xs.layer)
    c.add_polygon(etch, layer=LAYER.M1_ETCH)
    c.add_polygon(_layer_region(tmp, LAYER.VG_TOP), layer=LAYER.VG_TOP)
    c.add_polygon(_layer_region(tmp, LAYER.VG_SAC), layer=LAYER.VG_SAC)
    c.add_ports(feed.ports)

    c.info["vacuum_gap"] = get_vacuum_gap()
    c.info["total_wire_length"] = ind.cell.info["total_wire_length"]
    c.info["n_squares"] = ind.cell.info["n_squares"]
    return c
