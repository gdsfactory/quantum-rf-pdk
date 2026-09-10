"""Superconducting nanowire single-photon detector (SNSPD)."""

from __future__ import annotations

from typing import Literal

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Port, Size

from qpdk.tech import LAYER


@gf.cell(tags=("detectors",))
def snspd(
    wire_width: float = 0.2,
    wire_pitch: float = 0.6,
    size: Size = (10, 8),
    num_squares: int | None = None,
    turn_ratio: float = 4,
    terminals_same_side: bool = False,
    layer: LayerSpec = LAYER.NbTiN,
    port_type: Literal["electrical", "optical"] = "electrical",
) -> Component:
    """Creates an optimally-rounded SNSPD.

    .. svgbob::

        e1 ─────────────────────╮
        ╭───────────────────────╯
        ╰───────────────────────╮
        ╭───────────────────────╯
        ╰───────────────────────╮
        ╭───────────────────────╯
        ╰───────────────────────╮
           e2 ──────────────────╯

    Args:
        wire_width: Width of the wire.
        wire_pitch: Distance between two adjacent wires. Must be greater than `width`.
        size: Float2
            (width, height) of the rectangle formed by the outer boundary of the
            SNSPD.
        num_squares: int | None = None
            Total number of squares inside the SNSPD length. If given, overrides
            `size` with an approximately square SNSPD. The meander count is
            quantized to whole wire pitches, so the achieved square count
            (reported in `info["num_squares"]`) is close to but not exactly the
            requested one.
        turn_ratio: float
            Specifies how much of the SNSPD width is dedicated to the 180 degree
            turn. A `turn_ratio` of 10 will result in 20% of the width being
            comprised of the turn.
        terminals_same_side: If True, both ports will be located on the same side of the SNSPD.
        layer: layer spec to put polygon geometry on.
        port_type: type of port to add to the component (`"electrical"` or `"optical"`).

    Returns:
        A Component containing the SNSPD geometry.

    Raises:
        ValueError: If parameters are invalid or the SNSPD is too small for
            at least 3 meanders.
    """
    if num_squares is not None:
        if num_squares <= 0:
            raise ValueError(f"num_squares={num_squares} must be a positive integer.")
        # num_squares overrides size: build a square SNSPD with the requested
        # total number of squares. The meander count below is quantized to
        # whole wire pitches, so the achieved square count (reported in
        # `info["num_squares"]`) is close to but not exactly the request.
        xy = np.sqrt(num_squares * wire_pitch * wire_width)
        size = (xy, xy)

    xsize, ysize = size

    if xsize <= 0 or ysize <= 0:
        raise ValueError(f"size={size} dimensions must be positive.")
    if wire_pitch <= wire_width:
        raise ValueError(
            f"wire_pitch={wire_pitch} must be greater than wire_width={wire_width}."
        )

    num_meanders = int(np.ceil(ysize / wire_pitch))

    if (not terminals_same_side and (num_meanders % 2) == 0) or (
        terminals_same_side and (num_meanders % 2) == 1
    ):
        num_meanders += 1

    if num_meanders < 3:
        raise ValueError(
            f"num_meanders={num_meanders} is too small; the SNSPD needs at least "
            "3 meanders. Increase `size` or `num_squares`, or decrease `wire_pitch`."
        )

    D = Component()
    hairpin = gf.c.optimal_hairpin(
        width=wire_width,
        pitch=wire_pitch,
        turn_ratio=turn_ratio,
        length=xsize / 2,
        num_pts=20,
        layer=layer,
    )

    start_nw = D.add_ref(gf.c.compass(size=(xsize / 2, wire_width), layer=layer))

    hp_prev = D.add_ref(hairpin)
    hp_prev.connect("e1", start_nw.ports["e3"])
    alternate = True
    last_port: Port | None = None
    for _n in range(2, num_meanders):
        hp = D.add_ref(hairpin)
        if alternate:
            hp.connect("e2", hp_prev.ports["e2"])
        else:
            hp.connect("e1", hp_prev.ports["e1"])
        last_port = hp.ports["e2"] if terminals_same_side else hp.ports["e1"]
        hp_prev = hp
        alternate = not alternate

    finish_se = D.add_ref(gf.c.compass(size=(xsize / 2, wire_width), layer=layer))
    if last_port is not None:
        finish_se.connect("e3", last_port)

    # The nanowire geometry itself only makes electrical connections
    # (`optimal_hairpin` has electrical ports), so honor `port_type` on the
    # exposed terminal ports.
    port_prefix = "e" if port_type == "electrical" else "o"
    D.add_port(port=start_nw.ports["e1"], name=f"{port_prefix}1", port_type=port_type)
    D.add_port(port=finish_se.ports["e1"], name=f"{port_prefix}2", port_type=port_type)

    D.info["num_squares"] = num_meanders * (xsize / wire_width)
    D.info["area"] = xsize * ysize
    D.info["xsize"] = xsize
    D.info["ysize"] = ysize
    D.flatten()
    return D
