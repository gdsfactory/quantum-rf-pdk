"""Xmon and its local readout paddle share one circuit model boundary."""

import gdsfactory as gf

from qpdk.cells._schematic import sax_model, schematic
from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.transmon import xmon_transmon
from qpdk.tech import LAYER

__all__ = ["xmon_readout"]

_xmon_schematic = schematic(
    symbol="xmon_readout",
    tags=["qubits", "transmons"],
    ports=[{"name": "readout", "side": "left", "type": "photonic"}],
    models=[sax_model("xmon_readout", "qpdk.models.qubit_readout", ["readout"])],
)


@gf.cell(tags=["qubits", "transmons"], schematic_function=_xmon_schematic)
def xmon_readout(
    arm_scale: float = 1.0,
    gap_width: float = 10.0,
    coupling_gap: float = 5.0,
    paddle_length: float = 60.0,
    inductance_nh: float = 10.0,
    junction_capacitance_ff: float = 0.0,
) -> gf.Component:
    """Grounded Xmon and paddle with one ``readout`` circuit pin.

    ``arm_scale`` scales the default lengths (160,120,160,120) um. Arm widths
    stay (30,20,30,20) um. Other dimensions are in um. The island, grounded
    junction and local paddle capacitance are inside the model boundary.
    Inductance in nH and additional junction capacitance in fF are explicit model
    inputs. The external distributed CPW begins at the paddle pin.

    Args:
        arm_scale: Common multiplier of the four arm lengths.
        gap_width: Etched gap around the arms in um.
        coupling_gap: Edge-to-edge arm-to-paddle gap in um.
        paddle_length: Length of the 20 um wide paddle in um.
        inductance_nh: Linearized Josephson inductance in nH.
        junction_capacitance_ff: Additional junction capacitance in fF.

    Returns:
        Local Xmon and readout paddle, with the four arms as one circuit node.

    Raises:
        ValueError: If dimensions or model parameters are nonphysical.
    """
    if (
        min(arm_scale, gap_width, coupling_gap, paddle_length, inductance_nh) <= 0
        or junction_capacitance_ff < 0
    ):
        raise ValueError(
            "Require positive dimensions and inductance, and nonnegative junction capacitance"
        )
    c = gf.Component()
    qubit = c << xmon_transmon(
        arm_lengths=(
            160 * arm_scale,
            120 * arm_scale,
            160 * arm_scale,
            120 * arm_scale,
        ),
        gap_width=gap_width,
    )
    paddle = c << plate_capacitor_single(length=paddle_length, width=20)
    paddle.dmove((
        qubit.ports["left_arm"].dx - coupling_gap - 10,
        qubit.ports["left_arm"].dy,
    ))
    pin = paddle.ports["o1"]
    c.add_polygon(
        [
            (pin.dx - 8, pin.dy - 20),
            (pin.dx, pin.dy - 20),
            (pin.dx, pin.dy + 20),
            (pin.dx - 8, pin.dy + 20),
        ],
        layer=LAYER.M1_ETCH,
    )
    c.add_port("readout", port=pin)
    c.info["qubit_type"] = "transmon"
    return c
