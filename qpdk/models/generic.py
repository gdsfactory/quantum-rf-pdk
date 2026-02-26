"""Generic Models."""

import jax
import jax.numpy as jnp
import sax
from matplotlib import pyplot as plt
from sax.models.rf import (
    admittance,
    capacitor,
    electrical_open,
    electrical_short,
    gamma_0_load,
    impedance,
    inductor,
    tee,
)

from qpdk.models.constants import DEFAULT_FREQUENCY

__all__ = [
    "admittance",
    "capacitor",
    "electrical_open",
    "electrical_short",
    "electrical_short_2_port",
    "gamma_0_load",
    "impedance",
    "inductor",
    "lc_resonator",
    "lc_resonator_coupled",
    "open",
    "short",
    "short_2_port",
    "tee",
]


@jax.jit
def electrical_short_2_port(f: sax.FloatArrayLike = DEFAULT_FREQUENCY) -> sax.SType:
    """Electrical short 2-port connection Sax model.

    Args:
        f: Array of frequency points in Hz

    Returns:
        sax.SType: S-parameters dictionary
    """
    return electrical_short(f=f, n_ports=2)


short = electrical_short
open = electrical_open
short_2_port = electrical_short_2_port


def lc_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
) -> sax.SType:
    r"""LC resonator Sax model with capacitor and inductor in parallel.

    The resonance frequency is given by:

    .. math::

        f_r = \frac{1}{2 \pi \sqrt{LC}}

    .. svgbob::

        o1 ──┬──L──┬── o2
             │     │
             └──C──┘

    If grounded=True, a 2-port short is connected to port o2:

    .. svgbob::

        o1 ──┬──L──┬──|
             │     │  | (2-port ground)
             └──C──┘  |

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the resonator in Farads (default: 100 fF).
        inductance: Inductance of the resonator in Henries (default: 1 nH).
        grounded: If True, add a 2-port ground to the second port (default: False).

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.
    """
    f = jnp.asarray(f)

    # Create component instances
    instances = {
        "capacitor": capacitor(f=f, capacitance=capacitance),
        "inductor": inductor(f=f, inductance=inductance),
        "tee_1": tee(f=f),
        "tee_2": tee(f=f),
    }

    # Connect capacitor and inductor in parallel using two tees
    connections = {
        "tee_1,o2": "capacitor,o1",
        "tee_1,o3": "inductor,o1",
        "capacitor,o2": "tee_2,o2",
        "inductor,o2": "tee_2,o3",
    }

    if grounded:
        # Add a 2-port short to the second port
        instances["ground"] = electrical_short(f=f, n_ports=2)
        connections["tee_2,o1"] = "ground,o1"
        ports = {
            "o1": "tee_1,o1",
            "o2": "ground,o2",
        }
    else:
        ports = {
            "o1": "tee_1,o1",
            "o2": "tee_2,o1",
        }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def lc_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    coupling_capacitance: float = 0.0,
    coupling_inductance: float = 0.0,
) -> sax.SType:
    r"""Coupled LC resonator Sax model.

    This model extends the basic LC resonator by adding a coupling network
    consisting of a parallel capacitor and inductor connected to one port
    of the LC resonator via a tee junction.

    The resonance frequency of the main LC resonator is given by:

    .. math::

        f_r = \frac{1}{2 \pi \sqrt{LC}}

    The coupling network modifies the effective coupling to the resonator.

    .. svgbob::

                     ┌──Lc──┬
                     │      │
        o1 ───┬──────┼──Cc──┼────┬──L──┬── (grounded or o2)
              │      │      │    │     │
              │      └──────┘    └──C──┘
             tee (coupling)     (LC resonator)

    Where Lc and Cc are the coupling inductance and capacitance.

    If either coupling_capacitance or coupling_inductance is zero, that
    element is omitted from the coupling network.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the main resonator in Farads (default: 100 fF).
        inductance: Inductance of the main resonator in Henries (default: 1 nH).
        grounded: If True, the resonator is grounded (default: False).
        coupling_capacitance: Coupling capacitance in Farads (default: 0).
            If zero, no coupling capacitor is added.
        coupling_inductance: Coupling inductance in Henries (default: 0).
            If zero, no coupling inductor is added.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.
    """
    f = jnp.asarray(f)

    # Get the base LC resonator
    resonator = lc_resonator(
        f=f, capacitance=capacitance, inductance=inductance, grounded=grounded
    )

    # If no coupling elements, just return the resonator
    if coupling_capacitance == 0.0 and coupling_inductance == 0.0:
        return resonator

    # Build the coupling network with a tee and parallel coupling elements
    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "tee_coupling": tee(f=f),
    }

    # Start with connections from tee to resonator
    connections: dict[str, str] = {
        "tee_coupling,o1": "resonator,o1",
    }

    # Add coupling elements based on what's provided
    if coupling_capacitance > 0.0 and coupling_inductance > 0.0:
        # Both coupling elements: connect in parallel via another tee
        instances["coupling_tee_2"] = tee(f=f)
        instances["coupling_cap"] = capacitor(f=f, capacitance=coupling_capacitance)
        instances["coupling_ind"] = inductor(f=f, inductance=coupling_inductance)

        connections["tee_coupling,o3"] = "coupling_tee_2,o1"
        connections["coupling_tee_2,o2"] = "coupling_cap,o1"
        connections["coupling_tee_2,o3"] = "coupling_ind,o1"
        connections["coupling_cap,o2"] = "coupling_ind,o2"
    elif coupling_capacitance > 0.0:
        # Only coupling capacitor
        instances["coupling_cap"] = capacitor(f=f, capacitance=coupling_capacitance)
        connections["tee_coupling,o3"] = "coupling_cap,o1"
        connections["coupling_cap,o2"] = "coupling_cap,o2"
    elif coupling_inductance > 0.0:
        # Only coupling inductor
        instances["coupling_ind"] = inductor(f=f, inductance=coupling_inductance)
        connections["tee_coupling,o3"] = "coupling_ind,o1"
        connections["coupling_ind,o2"] = "coupling_ind,o2"

    ports = {
        "o1": "tee_coupling,o2",
        "o2": "resonator,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


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
