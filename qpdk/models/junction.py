"""Josephson junction Models."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
import scipy.constants
from sax.models.rf import impedance

from qpdk.models.constants import DEFAULT_FREQUENCY


@partial(jax.jit, inline=True)
def josephson_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic: sax.Float = 1e-6,
    capacitance: sax.Float = 5e-15,
    resistance: sax.Float = 10e3,
    ib: sax.Float = 0.0,
    z0: sax.Complex = 50,
) -> sax.SType:
    r"""Josephson junction (RCSJ) small-signal Sax model.

    Linearized RCSJ model consisting of a bias-dependent Josephson inductance
    in parallel with capacitance and resistance.

    Valid in the superconducting (zero-voltage) state and for small AC signals.

    Default capacitance taken from :cite:`shcherbakovaFabricationMeasurementsHybrid2015`.

    See :cite:`McCumber1968` for details.

    Args:
        f: Array of frequency points in Hz
        ic: Critical current I_c in Amperes
        capacitance: Junction capacitance C in Farads
        resistance: Shunt resistance R in Ohms
        ib: DC bias current I_b in Amperes (\|ib\| < ic)
        z0: Reference impedance in Ω

    Returns:
        sax.SType: S-parameters dictionary
    """
    ω = 2 * jnp.pi * jnp.asarray(f)

    # Bias-dependent phase factor
    cos_phi0 = jnp.sqrt(1.0 - (ib / ic) ** 2)

    # Josephson inductance
    LJ = scipy.constants.physical_constants["mag. flux quantum"][0] / (
        2 * jnp.pi * ic * cos_phi0
    )

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    Y_L = 1 / (1j * ω * LJ)

    # Total impedance
    Z_JJ = 1 / (Y_R + Y_C + Y_L)

    return impedance(f=f, z=Z_JJ, z0=z0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sax.set_port_naming_strategy("optical")

    f = jnp.linspace(1e9, 10e9, 500)
    s = josephson_junction(f=f, ic=1e-6, capacitance=50e-15, resistance=10e3, ib=0.5e-6)
    plt.figure()
    plt.plot(f / 1e9, jnp.abs(s[("o1", "o1")]), label="|S11|")
    plt.plot(f / 1e9, jnp.abs(s[("o1", "o2")]), label="|S12|")
    plt.plot(f / 1e9, jnp.abs(s[("o2", "o2")]), label="|S22|")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Josephson Junction S-parameters")
    plt.show()
