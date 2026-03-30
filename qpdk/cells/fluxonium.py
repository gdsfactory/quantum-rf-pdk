"""Fluxonium qubit components."""

from __future__ import annotations

import math
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.helpers import add_rect, transform_component
from qpdk.cells.inductor import meander_inductor
from qpdk.cells.junction import josephson_junction
from qpdk.tech import (
    LAYER,
    get_etch_section,
    superinductor_cross_section,
)

__all__ = ["fluxonium", "fluxonium_with_bbox"]


@gf.cell(check_instances=False)
def fluxonium(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 25.0,
    junction_spec: ComponentSpec = josephson_junction,
    junction_displacement: DCplxTrans | None = None,
    junction_margin: float = 1.0,
    inductor_n_turns: int = 155,
    inductor_margin_x: float = 1.0,
    inductor_cross_section: CrossSectionSpec = superinductor_cross_section,
    connection_wire_width: float = 0.5,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    r"""Creates a fluxonium qubit with capacitor pads, Josephson junction, and superinductor.

    .. svgbob::

        +---------+           +---------+
        |         |           |         |
        |         |           |         |
        |  pad1   |           |  pad2   |
        |         |           |         |
        |         |           |         |
        +----+----+           +----+----+
             |                     |
             |      JJ             |
             +======XX=============+
             |                     |
             |      inductor       |
             +---------------------+

    See :cite:`manucharyan_fluxonium_2009` and :cite:`nguyen_blueprint_2019`.

    Args:
        pad_size: (width, height) of each capacitor pad in µm.
        pad_gap: Gap between the two capacitor pads in µm.
        junction_spec: Component specification for the Josephson junction.
        junction_displacement: Optional transformation applied to the junction.
        junction_margin: Vertical margin between the junction and capacitor pads in µm.
        inductor_n_turns: Number of horizontal meander runs. Must be odd.
        inductor_margin_x: Horizontal margin for the inductor in µm.
        inductor_cross_section: Cross-section for the meander inductor.
        connection_wire_width: Width of the connecting wires in µm.
        layer_metal: Layer for the metal pads and connection wires.

    Returns:
        The fluxonium component.

    Raises:
        ValueError: If inductor_n_turns is even or if pad_gap is too small.
    """
    if inductor_n_turns % 2 == 0:
        raise ValueError("inductor_n_turns must be odd")

    xs = gf.get_cross_section(inductor_cross_section)
    inductor_wire_width = xs.width
    etch_section = get_etch_section(xs)
    inductor_wire_gap = 2 * etch_section.width

    c = Component()
    pad_width, pad_height = pad_size

    junction_comp = gf.get_component(junction_spec)
    junction_rotated_height = junction_comp.size_info.width

    inductor_turn_length = pad_gap - 2 * inductor_margin_x
    inductor_total_height = (
        inductor_n_turns * inductor_wire_width
        + max(0, inductor_n_turns - 1) * inductor_wire_gap
    )

    if inductor_turn_length <= 0:
        raise ValueError(f"pad_gap={pad_gap} is too small")

    # Capacitor pads
    def create_capacitor_pad(x_offset: float) -> gf.ComponentReference:
        pad = gf.components.rectangle(size=pad_size, layer=layer_metal)
        pad_ref = c.add_ref(pad)
        pad_ref.move((x_offset, -pad_height / 2))
        return pad_ref

    create_capacitor_pad(-pad_width - pad_gap / 2)
    create_capacitor_pad(pad_gap / 2)

    # Josephson junction
    junction_ref = c.add_ref(junction_comp)
    junction_ref.rotate(45)
    junction_y = -pad_height / 2 - junction_margin - junction_rotated_height / 2
    junction_ref.dcenter = (0, junction_y)
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Superinductor
    inductor = meander_inductor(
        n_turns=inductor_n_turns,
        turn_length=inductor_turn_length,
        cross_section=inductor_cross_section,
        wire_gap=inductor_wire_gap,
        etch_bbox_margin=0,
        add_etch=False,
    )
    inductor_ref = c.add_ref(inductor)
    inductor_y = (
        junction_ref.dcenter[1]
        - junction_rotated_height / 2
        - junction_margin
        - inductor_total_height / 2
    )
    inductor_ref.dcenter = (0, inductor_y)

    # Connection wires
    ind_o1 = inductor_ref.ports["o1"]
    ind_o2 = inductor_ref.ports["o2"]

    bus_left_x = -pad_gap / 2 - connection_wire_width / 2
    bus_right_x = pad_gap / 2 + connection_wire_width / 2
    bus_top_y = -pad_height / 2 + 0.1
    jj_conn_y0 = junction_ref.dcenter[1] - connection_wire_width / 2

    # Left bus bar
    add_rect(
        c,
        layer=layer_metal,
        x_center=bus_left_x,
        width=connection_wire_width,
        y0=jj_conn_y0,
        y1=bus_top_y,
    )
    add_rect(
        c,
        layer=LAYER.NbTiN,
        x0=-pad_gap / 2 - inductor_wire_width,
        x1=-pad_gap / 2,
        y0=ind_o1.dcenter[1],
        y1=jj_conn_y0,
    )

    # Right bus bar
    add_rect(
        c,
        layer=layer_metal,
        x_center=bus_right_x,
        width=connection_wire_width,
        y0=jj_conn_y0,
        y1=bus_top_y,
    )
    add_rect(
        c,
        layer=LAYER.NbTiN,
        x0=pad_gap / 2,
        x1=pad_gap / 2 + inductor_wire_width,
        y0=ind_o2.dcenter[1],
        y1=jj_conn_y0,
    )

    # Transitions
    add_rect(
        c,
        layer=layer_metal,
        x0=-pad_gap / 2 - connection_wire_width,
        x1=-pad_gap / 2,
        y0=-pad_height / 2,
        y1=-pad_height / 2 + connection_wire_width,
    )
    add_rect(
        c,
        layer=layer_metal,
        x0=pad_gap / 2,
        x1=pad_gap / 2 + connection_wire_width,
        y0=-pad_height / 2,
        y1=-pad_height / 2 + connection_wire_width,
    )

    # Inductor stubs
    add_rect(
        c,
        layer=LAYER.NbTiN,
        x0=-pad_gap / 2,
        x1=ind_o1.dcenter[0],
        y_center=ind_o1.dcenter[1],
        height=inductor_wire_width,
    )
    add_rect(
        c,
        layer=LAYER.NbTiN,
        x0=ind_o2.dcenter[0],
        x1=pad_gap / 2,
        y_center=ind_o2.dcenter[1],
        height=inductor_wire_width,
    )

    # Junction leads
    jj_p1 = junction_ref.ports["left_wide"].dcenter
    jj_p2 = junction_ref.ports["right_wide"].dcenter

    # Junction wires
    add_rect(
        c,
        layer=layer_metal,
        x0=-pad_gap / 2,
        x1=jj_p1[0],
        y_center=jj_p1[1],
        height=connection_wire_width,
    )
    add_rect(
        c,
        layer=layer_metal,
        x0=jj_p2[0],
        x1=pad_gap / 2,
        y_center=jj_p2[1],
        height=connection_wire_width,
    )

    # Ports
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

    c.info["qubit_type"] = "fluxonium"
    c.info["inductor_n_turns"] = inductor_n_turns
    c.info["inductor_total_wire_length"] = inductor.info["total_wire_length"]

    return c


def _snap_to_grid(value: float, grid: float = 0.002) -> float:
    """Snap a value up to the next grid multiple."""
    return math.ceil(value / grid) * grid


@gf.cell
def fluxonium_with_bbox(
    bbox_extension: float = 200.0,
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 25.0,
    junction_spec: ComponentSpec = josephson_junction,
    junction_displacement: DCplxTrans | None = None,
    junction_margin: float = 1.0,
    inductor_n_turns: int = 155,
    inductor_margin_x: float = 1.0,
    inductor_cross_section: CrossSectionSpec = superinductor_cross_section,
    connection_wire_width: float = 0.5,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Fluxonium with an etched bounding box.

    Args:
        bbox_extension: Extension of the bounding box from the fluxonium edge in µm.
        pad_size: (width, height) of each capacitor pad in µm.
        pad_gap: Gap between the two capacitor pads in µm.
        junction_spec: Component specification for the Josephson junction.
        junction_displacement: Optional transformation applied to the junction.
        junction_margin: Vertical margin between the junction and capacitor pads in µm.
        inductor_n_turns: Number of horizontal meander runs. Must be odd.
        inductor_margin_x: Horizontal margin for the inductor in µm.
        inductor_cross_section: Cross-section for the meander inductor.
        connection_wire_width: Width of the connecting wires in µm.
        layer_metal: Layer for the metal pads and connection wires.

    Returns:
        The fluxonium component with a bounding box.
    """
    c = gf.Component()
    flux_ref = c << fluxonium(
        pad_size=pad_size,
        pad_gap=pad_gap,
        junction_spec=junction_spec,
        junction_displacement=junction_displacement,
        junction_margin=junction_margin,
        inductor_n_turns=inductor_n_turns,
        inductor_margin_x=inductor_margin_x,
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
