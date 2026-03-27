"""Fluxonium qubit components."""

from __future__ import annotations

import math
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.helpers import transform_component
from qpdk.cells.inductor import meander_inductor
from qpdk.cells.junction import josephson_junction
from qpdk.tech import (
    LAYER,
    get_etch_section,
    meander_inductor_cross_section,
)


@gf.cell(check_instances=False)
def fluxonium(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 120.0,
    junction_spec: ComponentSpec = josephson_junction,
    junction_displacement: DCplxTrans | None = None,
    inductor_n_turns: int = 59,
    inductor_cross_section: CrossSectionSpec = meander_inductor_cross_section,
    connection_wire_width: float = 4.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    r"""Creates a fluxonium qubit with capacitor pads, Josephson junction, and superinductor.

    A fluxonium qubit consists of a single small Josephson junction shunted
    by a large inductance (superinductor) and a capacitor.  The two capacitor
    pads are connected in parallel by the junction and a tightly meandering
    inductor that provides the required large inductance.

    .. svgbob::

        +---------+           +---------+
        |         |  ───────  |         |
        |         |  ┌──────  |         |
        |  pad1   |  ───────  |  pad2   |
        |         |  inductor |         |
        |         |  ───────  |         |
        |         |====XX=====|         |
        +---------+    JJ     +---------+

    See :cite:`manucharyan_fluxonium_2009` for the original proposal and
    :cite:`nguyen_blueprint_2019` for a modern high-coherence implementation.

    Args:
        pad_size: (width, height) of each capacitor pad in µm.
        pad_gap: Gap between the two capacitor pads in µm.  Must be large
            enough to accommodate the meander inductor.
        junction_spec: Component specification for the single Josephson
            junction.
        junction_displacement: Optional transformation applied to the
            junction after default placement.
        inductor_n_turns: Number of horizontal meander runs in the
            superinductor.  Must be odd so both ports face outward.
        inductor_cross_section: Cross-section specification for the meander
            inductor.
        connection_wire_width: Width of the wires connecting the pads to
            the junction and inductor in µm.
        layer_metal: Layer for the metal pads and connection wires.

    Returns:
        Component: A gdsfactory component with the fluxonium geometry and
            ports for both pads and the junction.
    """
    if inductor_n_turns % 2 == 0:
        raise ValueError(
            "inductor_n_turns must be odd so that both inductor ports "
            "face outward (left and right) for connection to the pads"
        )

    xs = gf.get_cross_section(inductor_cross_section)
    inductor_wire_width = xs.width
    etch_section = get_etch_section(xs)
    inductor_wire_gap = 2 * etch_section.width

    c = Component()
    pad_width, pad_height = pad_size

    # -- Geometry budget -----------------------------------------------
    # The junction and inductor share the vertical space between the pads.
    junction_comp = gf.get_component(junction_spec)
    junction_rotated_height = junction_comp.size_info.width  # after 45° rotation
    junction_margin = 10.0  # clearance above/below junction

    inductor_turn_length = pad_gap - 2 * connection_wire_width
    inductor_total_height = (
        inductor_n_turns * inductor_wire_width
        + max(0, inductor_n_turns - 1) * inductor_wire_gap
    )

    # -- Capacitor pads ------------------------------------------------
    def create_capacitor_pad(x_offset: float) -> gf.ComponentReference:
        pad = gf.components.rectangle(size=pad_size, layer=layer_metal)
        pad_ref = c.add_ref(pad)
        pad_ref.move((x_offset, -pad_height / 2))
        return pad_ref

    create_capacitor_pad(-pad_width - pad_gap / 2)
    create_capacitor_pad(pad_gap / 2)

    # -- Josephson junction (at the bottom of the gap) -----------------
    junction_ref = c.add_ref(junction_comp)
    junction_ref.rotate(45)
    junction_y = -pad_height / 2 + junction_rotated_height / 2 + junction_margin
    junction_ref.dcenter = (0, junction_y)
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # -- Superinductor (meander inductor, fills most of the gap) -------
    # We disable the internal etch bounding box for the sub-component
    # because it is handled by the fluxonium bounding box.
    inductor = meander_inductor(
        n_turns=inductor_n_turns,
        turn_length=inductor_turn_length,
        cross_section=inductor_cross_section,
        wire_gap=inductor_wire_gap,
        etch_bbox_margin=0,
        add_etch=False,
    )
    inductor_ref = c.add_ref(inductor)
    # Place inductor above the junction, filling the remaining gap
    inductor_y = pad_height / 2 - inductor_total_height / 2
    inductor_ref.dcenter = (0, inductor_y)

    # -- Connection wires from pads to inductor and junction -----------
    # The inductor (odd n_turns) has o1 at its bottom-left and o2 at its
    # top-right.  Vertical bus bars along each gap edge carry current
    # from the pad inner edge down to the junction and up to both
    # inductor ports.
    ind_o1 = inductor_ref.ports["o1"]
    ind_o2 = inductor_ref.ports["o2"]

    bus_left_x = -pad_gap / 2 + connection_wire_width / 2
    bus_right_x = pad_gap / 2 - connection_wire_width / 2
    bus_bottom_y = junction_y - connection_wire_width / 2
    bus_top_y = pad_height / 2  # top of the pad area

    # Left bus bar: full height of gap
    _add_rect(
        c,
        layer=layer_metal,
        x_center=bus_left_x,
        width=connection_wire_width,
        y0=bus_bottom_y,
        y1=bus_top_y,
    )
    # Right bus bar: full height of gap (symmetric)
    _add_rect(
        c,
        layer=layer_metal,
        x_center=bus_right_x,
        width=connection_wire_width,
        y0=bus_bottom_y,
        y1=bus_top_y,
    )

    # Horizontal stubs: bus bars → inductor ports
    _add_rect(
        c,
        layer=layer_metal,
        x0=-pad_gap / 2,
        x1=ind_o1.dcenter[0],
        y_center=ind_o1.dcenter[1],
        height=inductor_wire_width,
    )
    _add_rect(
        c,
        layer=layer_metal,
        x0=ind_o2.dcenter[0],
        x1=pad_gap / 2,
        y_center=ind_o2.dcenter[1],
        height=inductor_wire_width,
    )

    # Horizontal wires: bus bars → junction
    jj_half_w = junction_ref.size_info.width / 2
    _add_rect(
        c,
        layer=layer_metal,
        x0=-pad_gap / 2,
        x1=-jj_half_w,
        y_center=junction_y,
        height=connection_wire_width,
    )
    _add_rect(
        c,
        layer=layer_metal,
        x0=jj_half_w,
        x1=pad_gap / 2,
        y_center=junction_y,
        height=connection_wire_width,
    )

    # -- Ports ---------------------------------------------------------
    ports_config = [
        {
            "name": "left_pad",
            "center": (-pad_width - pad_gap / 2, 0),
            "width": pad_height,
            "orientation": 180,
            "layer": layer_metal,
        },
        {
            "name": "left_pad_inner",
            "center": (-pad_gap / 2, 0),
            "width": pad_height,
            "orientation": 0,
            "layer": layer_metal,
            "port_type": "placement",
        },
        {
            "name": "right_pad",
            "center": (pad_width + pad_gap / 2, 0),
            "width": pad_height,
            "orientation": 0,
            "layer": layer_metal,
        },
        {
            "name": "right_pad_inner",
            "center": (pad_gap / 2, 0),
            "width": pad_height,
            "orientation": 180,
            "layer": layer_metal,
            "port_type": "placement",
        },
        {
            "name": "junction",
            "center": junction_ref.dcenter,
            "width": _snap_to_grid(junction_ref.size_info.height),
            "orientation": 90,
            "layer": LAYER.JJ_AREA,
            "port_type": "placement",
        },
    ]
    for port_config in ports_config:
        c.add_port(**port_config)

    # -- Metadata ------------------------------------------------------
    c.info["qubit_type"] = "fluxonium"
    c.info["inductor_n_turns"] = inductor_n_turns
    c.info["inductor_total_wire_length"] = inductor.info["total_wire_length"]

    return c


def _snap_to_grid(value: float, grid: float = 0.002) -> float:
    """Snap a value to the nearest even grid multiple (for port widths)."""
    return math.ceil(value / grid) * grid


def _add_rect(
    c: Component,
    layer: LayerSpec,
    *,
    x0: float | None = None,
    x1: float | None = None,
    y0: float | None = None,
    y1: float | None = None,
    x_center: float | None = None,
    y_center: float | None = None,
    width: float | None = None,
    height: float | None = None,
) -> None:
    """Add a rectangle to *c* using flexible coordinates.

    Provide either ``(x0, x1)`` **or** ``(x_center, width)`` for the
    horizontal extent (likewise for the vertical extent).
    """
    if x0 is not None and x1 is not None:
        x_lo, x_hi = min(x0, x1), max(x0, x1)
    elif x_center is not None and width is not None:
        x_lo, x_hi = x_center - width / 2, x_center + width / 2
    else:
        raise ValueError("Provide (x0, x1) or (x_center, width)")

    if y0 is not None and y1 is not None:
        y_lo, y_hi = min(y0, y1), max(y0, y1)
    elif y_center is not None and height is not None:
        y_lo, y_hi = y_center - height / 2, y_center + height / 2
    else:
        raise ValueError("Provide (y0, y1) or (y_center, height)")

    c.add_polygon([(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)], layer=layer)


@gf.cell
def fluxonium_with_bbox(
    bbox_extension: float = 200.0,
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 120.0,
    junction_spec: ComponentSpec = josephson_junction,
    junction_displacement: DCplxTrans | None = None,
    inductor_n_turns: int = 59,
    inductor_cross_section: CrossSectionSpec = meander_inductor_cross_section,
    connection_wire_width: float = 4.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Creates a fluxonium qubit with an etched bounding box.

    See :func:`~fluxonium` for more details.

    Args:
        bbox_extension: Extension size for the bounding box in µm.
        pad_size: (width, height) of each capacitor pad in µm.
        pad_gap: Gap between the two capacitor pads in µm.
        junction_spec: Component specification for the single Josephson
            junction.
        junction_displacement: Optional transformation applied to the
            junction.
        inductor_n_turns: Number of horizontal meander runs in the
            superinductor.
        inductor_cross_section: Cross-section specification for the meander
            inductor.
        connection_wire_width: Width of the wires connecting the pads to
            the junction and inductor in µm.
        layer_metal: Layer for the metal pads and connection wires.

    Returns:
        Component: A gdsfactory component with the fluxonium geometry
            and etched box.
    """
    c = gf.Component()
    flux_ref = c << fluxonium(
        pad_size=pad_size,
        pad_gap=pad_gap,
        junction_spec=junction_spec,
        junction_displacement=junction_displacement,
        inductor_n_turns=inductor_n_turns,
        inductor_cross_section=inductor_cross_section,
        connection_wire_width=connection_wire_width,
        layer_metal=layer_metal,
    )
    flux_size = (flux_ref.size_info.width, flux_ref.size_info.height)
    bbox_size = (
        flux_size[0] + 2 * bbox_extension,
        flux_size[1] + 2 * bbox_extension,
    )

    bbox = gf.container(
        partial(
            gf.components.rectangle,
            size=bbox_size,
            layer=LAYER.M1_ETCH,
        ),
        partial(
            transform_component, transform=DCplxTrans(*(-e / 2 for e in bbox_size))
        ),
    )
    # Remove additive metal from etch
    bbox = gf.boolean(
        A=bbox,
        B=c,
        operation="-",
        layer=LAYER.M1_ETCH,
        layer1=LAYER.M1_ETCH,
        layer2=LAYER.M1_DRAW,
    )
    bbox_ref = c.add_ref(bbox)
    c.absorb(bbox_ref)

    c.add_ports(flux_ref.ports)
    return c


if __name__ == "__main__":
    from qpdk.helper import show_components

    show_components(
        fluxonium,
        fluxonium_with_bbox,
    )
