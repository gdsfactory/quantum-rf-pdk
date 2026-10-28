"""Local FEM qubit and paddle blocks for distributed readout circuits.

Only the readout pin is external. Qubit electrodes obey Kirchhoff's law with
zero external injected current. The Maxwell matrices include the complete local
paddle capacitance, so no extra coupling capacitor belongs beside this block.
Circuit inputs use nH and fF to survive layout-settings serialization.
"""

from functools import cache

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.datasets import junction_model
from qpdk.models.jit import jit


@cache
def _xmon_readout_model() -> sax.SDictModel:
    """Load the local matrix once, outside traced numerical evaluation."""
    return junction_model(
        "xmon_readout_palace",
        junction=("island", None),
        inductance=10e-9,
        exposed_ports=("readout",),
    )


@jit(inline=True)
def xmon_readout(
    f: ArrayLike = DEFAULT_FREQUENCY,
    arm_scale: ArrayLike = 1.0,
    gap_width: ArrayLike = 10.0,
    coupling_gap: ArrayLike = 5.0,
    paddle_length: ArrayLike = 60.0,
    inductance_nh: ArrayLike = 10.0,
    junction_capacitance_ff: ArrayLike = 0.0,
    z_ref: ArrayLike = 50.0,
) -> sax.SDict:
    """Linearized grounded Xmon and paddle, exposing only ``readout``.

    All four arms share one island potential. ``arm_scale`` multiplies the
    default lengths (160,120,160,120) um while widths stay (30,20,30,20) um.
    Other dimensions are in um, frequency in Hz, inductance in nH and junction
    capacitance in fF. The junction connects the island to reference ground.
    The lossless model includes the local paddle, but no distributed CPW,
    package or nonlinear spectrum. Out-of-grid dimensions return NaN.
    Geometry derivatives change slope at interpolation grid knots.

    Returns:
        One-port JAX SAX response with the island internal to the block.
    """
    with jax.ensure_compile_time_eval():
        model = _xmon_readout_model()
    return model(
        f=f,
        arm_scale=arm_scale,
        gap_width=gap_width,
        coupling_gap=coupling_gap,
        paddle_length=paddle_length,
        inductance=jnp.asarray(inductance_nh) * 1e-9,
        junction_capacitance=jnp.asarray(junction_capacitance_ff) * 1e-15,
        z_ref=z_ref,
    )
