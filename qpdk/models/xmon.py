"""Bare Xmon capacitance and a grounded junction share one four-arm node."""

from functools import cache

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from sax.models.rf import electrical_open, tee

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.datasets import Dataset, capacitance_model
from qpdk.models.generic import shunt_admittance
from qpdk.models.jit import jit


@jit(inline=True)
def _junction(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance_nh: ArrayLike = 7.0,
    junction_capacitance_ff: ArrayLike = 0.0,
    z_ref: float = 50.0,
) -> sax.SDict:
    """Keep junction admittance separate from the FEM island capacitance."""
    omega = 2 * jnp.pi * jnp.asarray(f)
    admittance = 1 / (1j * omega * jnp.asarray(inductance_nh) * 1e-9)
    admittance += 1j * omega * jnp.asarray(junction_capacitance_ff) * 1e-15
    return shunt_admittance(f=f, y=admittance, z0=z_ref)


def _open(f: ArrayLike = DEFAULT_FREQUENCY) -> sax.SDict:
    """Hide the static port count from SAX's traced circuit settings."""
    return electrical_open(f=f, n_ports=1)


@cache
def _xmon_fem_circuit() -> tuple[sax.SDictModel, tuple[float, float]]:
    """Join the four arms, FEM capacitor and junction at one ideal node."""
    data = Dataset("xmon_palace")
    scale_domain = data.grid("maxwell_capacitance").domain["arm_scale"]
    circuit, _ = sax.circuit(
        {
            "instances": {
                "top": "tee",
                "middle": "tee",
                "bottom": "tee",
                "shunt": "tee",
                "capacitance": "capacitance",
                "junction": "junction",
                "open": "open",
            },
            "connections": {
                "top,o3": "middle,o1",
                "middle,o3": "bottom,o1",
                "bottom,o3": "shunt,o1",
                "shunt,o2": "capacitance,island",
                "shunt,o3": "junction,o1",
                "junction,o2": "open,o1",
            },
            "ports": {
                "top_arm": "top,o1",
                "right_arm": "top,o2",
                "bottom_arm": "middle,o2",
                "left_arm": "bottom,o2",
            },
        },
        models={
            "tee": tee,
            "capacitance": capacitance_model(data),
            "junction": _junction,
            "open": _open,
        },
        ignore_impossible_connections=False,
    )
    return circuit, scale_domain


@jit(inline=True)
def xmon_fem(
    f: ArrayLike = DEFAULT_FREQUENCY,
    arm_width: tuple[float, float, float, float] | jax.Array = (30.0, 20.0, 30.0, 20.0),
    arm_lengths: tuple[float, float, float, float] | jax.Array = (
        160.0,
        120.0,
        160.0,
        120.0,
    ),
    gap_width: ArrayLike = 10.0,
    inductance_nh: ArrayLike = 7.0,
    junction_capacitance_ff: ArrayLike = 0.0,
    z_ref: ArrayLike = 50.0,
) -> sax.SDict:
    """Four galvanic arm ports on one bare equipotential Xmon island.

    Arm lengths scale uniformly from (160,120,160,120) um and widths stay
    (30,20,30,20) um. Unsupported arm shapes or out-of-grid dimensions return
    NaN. Frequency is positive in Hz, inductance in nH and additional junction capacitance
    in fF. All arm ports use the same real reference impedance in ohms.

    The FEM block includes a 300 um ground frame outside the etched moat. It
    omits readout paddles, neighbours, distributed arms, package and nonlinear
    spectra. The four arm ports have equal potential, so the island capacitance
    appears once. Explicitly open unused arm ports in a SAX circuit. Omitting
    them would attach matched loads.

    Returns:
        Jittable scattering matrix with the four layout arm names.
    """
    lengths = jnp.asarray(arm_lengths)
    widths = jnp.asarray(arm_width)
    scale = lengths / jnp.asarray([160.0, 120.0, 160.0, 120.0])
    supported = jnp.all(jnp.isclose(scale, scale[0], rtol=0, atol=1e-12))
    supported &= jnp.all(
        jnp.isclose(widths, jnp.asarray([30.0, 20.0, 30.0, 20.0]), rtol=0, atol=1e-12)
    )
    supported &= (jnp.asarray(inductance_nh) > 0) & (
        jnp.asarray(junction_capacitance_ff) >= 0
    )
    supported &= (jnp.asarray(z_ref) > 0) & (jnp.asarray(f) > 0)
    with jax.ensure_compile_time_eval():
        model, (scale_min, scale_max) = _xmon_fem_circuit()
    supported &= (lengths[0] >= 160 * scale_min) & (lengths[0] <= 160 * scale_max)
    # XLA's reciprocal multiplication can put a valid endpoint one ULP outside.
    arm_scale = jnp.clip(scale[0], scale_min, scale_max)
    result = model(
        f=f,
        capacitance={"arm_scale": arm_scale, "gap_width": gap_width, "z_ref": z_ref},
        junction={
            "inductance_nh": inductance_nh,
            "junction_capacitance_ff": junction_capacitance_ff,
            "z_ref": z_ref,
        },
    )
    return {
        ports: jnp.where(supported, values, jnp.nan) for ports, values in result.items()
    }
