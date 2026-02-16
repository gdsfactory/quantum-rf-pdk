"""Generic Models."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from matplotlib import pyplot as plt
from sax.models.rf import capacitor, gamma_0_load, impedance, inductor, tee

from qpdk.models.constants import DEFAULT_FREQUENCY

__all__ = ["josephson_junction", "open", "short", "short_2_port", "tee"]


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def short(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_ports: int = 1,
) -> sax.SType:
    r"""Electrical short connections Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as shorted

    Returns:
        sax.SType: S-parameters dictionary where :math:`S = -I_\text{n\_ports}`
    """
    return gamma_0_load(f=f, gamma_0=-1, n_ports=n_ports)


@jax.jit
def short_2_port(f: sax.FloatArrayLike = DEFAULT_FREQUENCY) -> sax.SType:
    """Electrical short 2-port connection Sax model.

    Args:
        f: Array of frequency points in Hz

    Returns:
        sax.SType: S-parameters dictionary
    """
    return short(f=f, n_ports=2)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def open(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_ports: int = 1,
) -> sax.SType:
    r"""Electrical open connection Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as opened

    Returns:
        sax.SType: S-parameters dictionary where :math:`S = I_\text{n\_ports}`
    """
    return gamma_0_load(f=f, gamma_0=1, n_ports=n_ports)


@partial(jax.jit, inline=True)
def josephson_junction(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic: sax.Float = 1e-6,
    capacitance: sax.Float = 50e-15,
    resistance: sax.Float = 10e3,
    ib: sax.Float = 0.0,
    z0: sax.Complex = 50,
) -> sax.SType:
    r"""Josephson junction (RCSJ) small-signal Sax model.

    Linearized RCSJ model consisting of a bias-dependent Josephson inductance
    in parallel with a capacitance and resistance.

    Valid in the superconducting (zero-voltage) state and for small AC signals.

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
    # Flux quantum [Wb]
    PHI0 = 2.067833848e-15

    ω = 2 * jnp.pi * jnp.asarray(f)

    # Bias-dependent phase factor
    cos_phi0 = jnp.sqrt(1.0 - (ib / ic) ** 2)

    # Josephson inductance
    LJ = PHI0 / (2 * jnp.pi * ic * cos_phi0)

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    Y_L = 1 / (1j * ω * LJ)

    # Total impedance
    Z_JJ = 1 / (Y_R + Y_C + Y_L)

    return impedance(f=f, z=Z_JJ, z0=z0)


if __name__ == "__main__":
    f = jnp.linspace(1e9, 25e9, 201)
    S = gamma_0_load(f=f, gamma_0=0.5 + 0.5j, n_ports=2)
    for key in S:
        plt.plot(f / 1e9, abs(S[key]) ** 2, label=key)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("S")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    S_cap = capacitor(f=f, capacitance=(capacitance := 100e-15))
    # print(S_cap)
    plt.figure()
    # Polar plot of S21 and S11
    plt.subplot(121, projection="polar")
    plt.plot(jnp.angle(S_cap[("o1", "o1")]), abs(S_cap[("o1", "o1")]), label="$S_{11}$")
    plt.plot(jnp.angle(S_cap[("o1", "o2")]), abs(S_cap[("o2", "o1")]), label="$S_{21}$")
    plt.title("S-parameters capacitor")
    plt.legend()
    # Magnitude and phase vs frequency
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_cap[("o1", "o1")]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_cap[("o1", "o2")]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap[("o1", "o1")]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap[("o1", "o2")]),
        label="∠S21",
        color="C1",
        linestyle="--",
    )
    ax2.set_ylabel("Phase [rad]")
    ax2.legend(loc="upper right")

    plt.title(f"Capacitor $S$-parameters ($C={capacitance * 1e15}\\,$fF)")
    plt.show(block=False)

    S_ind = inductor(f=f, inductance=(inductance := 1e-9))
    # print(S_ind)
    plt.figure()
    plt.subplot(121, projection="polar")
    plt.plot(jnp.angle(S_ind[("o1", "o1")]), abs(S_ind[("o1", "o1")]), label="$S_{11}$")
    plt.plot(jnp.angle(S_ind[("o1", "o2")]), abs(S_ind[("o2", "o1")]), label="$S_{21}$")
    plt.title("S-parameters inductor")
    plt.legend()
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_ind[("o1", "o1")]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_ind[("o1", "o2")]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind[("o1", "o1")]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind[("o1", "o2")]),
        label="∠S21",
        color="C1",
        linestyle="--",
    )
    ax2.set_ylabel("Phase [rad]")
    ax2.legend(loc="upper right")

    plt.title(f"Inductor $S$-parameters ($L={inductance * 1e9}\\,$nH)")
    plt.show()
