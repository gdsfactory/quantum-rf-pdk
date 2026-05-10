"""Cos(2φ) transmon qubit components.

Implements the Fourier-engineered cos(2φ) transmon design based on an
interference-based architecture with an asymmetric SQUID loop.

See :cite:`zhurbinaCoherenceLimitationsFourierEngineered2026` for details.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.junction import josephson_junction
from qpdk.tech import LAYER

__all__ = ["cos2phi_squid_junction", "cos2phi_transmon"]


@gf.cell(check_instances=False, tags=("junctions",))
def cos2phi_squid_junction(
    junction_spec: ComponentSpec = josephson_junction,
    main_loop_area: float = 100.0,
    small_loop_area: float = 25.0,
    connection_wire_width: float = 2.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_jj_area: LayerSpec = LAYER.JJ_AREA,
) -> Component:
    r"""Creates a cos(2φ) asymmetric SQUID junction.

    This component implements the interference-based junction structure for
    a cos(2φ) transmon. The left arm contains two Josephson junctions in series
    (fixed), while the right arm contains a single junction in series with a
    smaller SQUID loop (tunable via flux).

    .. svgbob::

                      o_top
                        |
              .---------+---------.
              |                   |
             [JJ1]               [JJ3]
              |                   |
             [JJ2]         .-----+-----.
              |            |            |
              |           [JJ4]       [JJ5]
              |            |            |
              |            '-----+-----'
              |                  |  small SQUID
              '---------+--------'
                        |
                      o_bot

    The main SQUID loop is formed by the left arm (JJ1 + JJ2) and the right
    arm (JJ3 + small SQUID). The small SQUID allows tuning the effective
    transparency of the right arm.

    See :cite:`zhurbinaCoherenceLimitationsFourierEngineered2026` for details.

    Args:
        junction_spec: Component specification for individual Josephson junctions.
        main_loop_area: Area of the main SQUID loop in µm².
        small_loop_area: Area of the small tunable SQUID loop in µm².
        connection_wire_width: Width of connection wires between junctions in µm.
        layer_metal: Layer for the metal connection wires.
        layer_jj_area: Layer for marking the junction area boundary.

    Returns:
        The cos(2φ) asymmetric SQUID junction component.
    """
    c = Component()

    junction_comp = gf.get_component(junction_spec)
    jj_width = junction_comp.size_info.width
    jj_height = junction_comp.size_info.height

    # Compute loop dimensions from areas
    main_loop_width = main_loop_area**0.5
    main_loop_height = main_loop_area / main_loop_width
    small_loop_width = small_loop_area**0.5
    small_loop_height = small_loop_area / small_loop_width

    # Total vertical extent for each arm (from top bus to bottom bus)
    # Left arm: wire + JJ1 + wire + JJ2 + wire

    # ===== LEFT ARM: Two JJs in series =====
    left_jj1 = c << gf.get_component(junction_spec)
    left_jj1.rotate(45)
    left_jj1.dcenter = (-main_loop_width / 2, main_loop_height / 4)

    left_jj2 = c << gf.get_component(junction_spec)
    left_jj2.rotate(45)
    left_jj2.dcenter = (-main_loop_width / 2, -main_loop_height / 4)

    # Connection wire between left JJ1 and JJ2
    left_conn_height = abs(left_jj1.dcenter[1] - left_jj2.dcenter[1]) - jj_height * 0.7
    if left_conn_height > 0:
        left_conn = c << gf.components.rectangle(
            size=(connection_wire_width, left_conn_height),
            layer=layer_metal,
            centered=True,
        )
        left_conn.dcenter = (-main_loop_width / 2, 0)

    # ===== RIGHT ARM: One JJ in series with a small SQUID =====
    # Right arm single junction (upper part)
    right_jj = c << gf.get_component(junction_spec)
    right_jj.rotate(45)
    right_jj.dcenter = (main_loop_width / 2, main_loop_height / 4)

    # Small SQUID loop (lower part of right arm)
    small_squid_jj_left = c << gf.get_component(junction_spec)
    small_squid_jj_left.rotate(45)
    small_squid_jj_left.dcenter = (
        main_loop_width / 2 - small_loop_width / 2,
        -main_loop_height / 4,
    )

    small_squid_jj_right = c << gf.get_component(junction_spec)
    small_squid_jj_right.rotate(45)
    small_squid_jj_right.dcenter = (
        main_loop_width / 2 + small_loop_width / 2,
        -main_loop_height / 4,
    )

    # Connection wire between right JJ and small SQUID top
    right_conn_height = (
        abs(right_jj.dcenter[1] - small_squid_jj_left.dcenter[1]) - jj_height * 0.7
    )
    if right_conn_height > 0:
        # Wire from right JJ down to small SQUID top junction node
        right_conn = c << gf.components.rectangle(
            size=(connection_wire_width, right_conn_height),
            layer=layer_metal,
            centered=True,
        )
        right_conn.dcenter = (main_loop_width / 2, 0)

    # Small SQUID connection wires (horizontal bus bars top and bottom)
    small_squid_top_bus = c << gf.components.rectangle(
        size=(small_loop_width, connection_wire_width),
        layer=layer_metal,
        centered=True,
    )
    small_squid_top_bus.dcenter = (
        main_loop_width / 2,
        -main_loop_height / 4 + small_loop_height / 2,
    )

    small_squid_bot_bus = c << gf.components.rectangle(
        size=(small_loop_width, connection_wire_width),
        layer=layer_metal,
        centered=True,
    )
    small_squid_bot_bus.dcenter = (
        main_loop_width / 2,
        -main_loop_height / 4 - small_loop_height / 2,
    )

    # ===== BUS BARS (top and bottom of main loop) =====
    top_bus = c << gf.components.rectangle(
        size=(main_loop_width, connection_wire_width),
        layer=layer_metal,
        centered=True,
    )
    top_bus.dcenter = (0, main_loop_height / 2)

    bot_bus = c << gf.components.rectangle(
        size=(main_loop_width, connection_wire_width),
        layer=layer_metal,
        centered=True,
    )
    bot_bus.dcenter = (0, -main_loop_height / 2)

    # ===== VERTICAL CONNECTION WIRES (arms to bus bars) =====
    # Left arm top connection (top bus to left JJ1)
    left_top_wire_y0 = left_jj1.dcenter[1] + jj_height * 0.35
    left_top_wire_y1 = main_loop_height / 2
    if left_top_wire_y1 > left_top_wire_y0:
        wire = c << gf.components.rectangle(
            size=(connection_wire_width, left_top_wire_y1 - left_top_wire_y0),
            layer=layer_metal,
        )
        wire.dmove((-main_loop_width / 2 - connection_wire_width / 2, left_top_wire_y0))

    # Left arm bottom connection (left JJ2 to bottom bus)
    left_bot_wire_y0 = -main_loop_height / 2
    left_bot_wire_y1 = left_jj2.dcenter[1] - jj_height * 0.35
    if left_bot_wire_y1 > left_bot_wire_y0:
        wire = c << gf.components.rectangle(
            size=(connection_wire_width, left_bot_wire_y1 - left_bot_wire_y0),
            layer=layer_metal,
        )
        wire.dmove((-main_loop_width / 2 - connection_wire_width / 2, left_bot_wire_y0))

    # Right arm top connection (top bus to right JJ)
    right_top_wire_y0 = right_jj.dcenter[1] + jj_height * 0.35
    right_top_wire_y1 = main_loop_height / 2
    if right_top_wire_y1 > right_top_wire_y0:
        wire = c << gf.components.rectangle(
            size=(connection_wire_width, right_top_wire_y1 - right_top_wire_y0),
            layer=layer_metal,
        )
        wire.dmove((main_loop_width / 2 - connection_wire_width / 2, right_top_wire_y0))

    # Right arm bottom connection (small SQUID bottom bus to main bottom bus)
    right_bot_wire_y0 = -main_loop_height / 2
    right_bot_wire_y1 = small_squid_bot_bus.dcenter[1] - connection_wire_width / 2
    if right_bot_wire_y1 > right_bot_wire_y0:
        wire = c << gf.components.rectangle(
            size=(connection_wire_width, right_bot_wire_y1 - right_bot_wire_y0),
            layer=layer_metal,
        )
        wire.dmove((main_loop_width / 2 - connection_wire_width / 2, right_bot_wire_y0))

    # ===== JJ AREA MARKER =====
    jj_area_marker = c << gf.components.rectangle(
        size=(
            main_loop_width + jj_width,
            main_loop_height + jj_height,
        ),
        layer=layer_jj_area,
        centered=True,
    )
    jj_area_marker.dcenter = (0, 0)

    # ===== PORTS =====
    c.add_port(
        name="o_top",
        center=(0, main_loop_height / 2 + connection_wire_width / 2),
        width=connection_wire_width,
        orientation=90,
        layer=layer_metal,
        port_type="electrical",
    )
    c.add_port(
        name="o_bot",
        center=(0, -main_loop_height / 2 - connection_wire_width / 2),
        width=connection_wire_width,
        orientation=270,
        layer=layer_metal,
        port_type="electrical",
    )
    c.add_port(
        name="loop_center",
        center=(0, 0),
        width=main_loop_width,
        orientation=0,
        layer=layer_jj_area,
        port_type="placement",
    )

    return c


@gf.cell(check_instances=False, tags=("qubits", "transmons"))
def cos2phi_transmon(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 30.0,
    junction_spec: ComponentSpec = josephson_junction,
    junction_displacement: DCplxTrans | None = None,
    main_loop_area: float = 100.0,
    small_loop_area: float = 25.0,
    connection_wire_width: float = 2.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    r"""Creates a Fourier-engineered cos(2φ) transmon qubit.

    This qubit uses an interference-based architecture to suppress odd harmonics
    of the effective qubit potential, enabling coherent tunnelling of Cooper-pair
    pairs. The junction structure consists of an asymmetric SQUID with:

    - **Left arm**: two Josephson junctions in series (fixed).
    - **Right arm**: one Josephson junction in series with a small SQUID loop
      (tunable via external flux).

    The small SQUID allows continuous tuning of the effective transparency of
    the right arm, enabling a transition from a conventional transmon regime to
    a :math:`\cos(2\varphi)`-dominated potential.

    .. svgbob::

        +---------+                         +---------+
        |         |                         |         |
        |  pad1   |     asymmetric SQUID    |  pad2   |
        |         |  [JJ1+JJ2 || JJ3+SQUID] |         |
        +---------+                         +---------+
         left_pad                             right_pad

    See :cite:`zhurbinaCoherenceLimitationsFourierEngineered2026` for details.

    Args:
        pad_size: (width, height) of each capacitor pad in µm.
        pad_gap: Gap between the two capacitor pads in µm.
        junction_spec: Component specification for individual Josephson junctions.
        junction_displacement: Optional transformation applied to the junction structure.
        main_loop_area: Area of the main SQUID loop in µm².
        small_loop_area: Area of the small tunable SQUID loop in µm².
        connection_wire_width: Width of connection wires in µm.
        layer_metal: Layer for the metal pads.

    Returns:
        Component: A gdsfactory component with the cos(2φ) transmon geometry.
    """
    c = Component()

    pad_width, pad_height = pad_size

    # Create capacitor pads
    def create_capacitor_pad(x_offset: float) -> gf.ComponentReference:
        pad = gf.components.rectangle(
            size=pad_size,
            layer=layer_metal,
        )
        pad_ref = c.add_ref(pad)
        pad_ref.move((x_offset, -pad_height / 2))
        return pad_ref

    create_capacitor_pad(-pad_width - pad_gap / 2)
    create_capacitor_pad(pad_gap / 2)

    # Create and place the asymmetric SQUID junction
    squid_ref = c.add_ref(
        cos2phi_squid_junction(
            junction_spec=junction_spec,
            main_loop_area=main_loop_area,
            small_loop_area=small_loop_area,
            connection_wire_width=connection_wire_width,
            layer_metal=layer_metal,
        )
    )
    squid_ref.dcenter = (0, 0)
    if junction_displacement:
        squid_ref.transform(junction_displacement)

    # Add ports
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
            "center": squid_ref.dcenter,
            "width": round(squid_ref.size_info.height * 500) / 500,
            "orientation": 90,
            "layer": LAYER.JJ_AREA,
            "port_type": "placement",
        },
    ]
    for port_config in ports_config:
        c.add_port(**port_config)

    # Add metadata
    c.info["qubit_type"] = "cos2phi_transmon"

    return c
