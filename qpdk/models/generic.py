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
    "series_impedance",
    "short",
    "short_2_port",
    "shunt_admittance",
    "tee",
]


@jax.jit
def series_impedance(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    z: sax.Float = 0.0,
    z0: float = 50.0,
) -> sax.SDict:
    r"""Two-port series impedance Sax model.

    .. svgbob::

        o1 ─── Z ─── o2

    See :cite:`m.pozarMicrowaveEngineering2012` (Ch. 4, Table 4.1, Table 4.2, Problem 4.11)
    for the S-parameter derivation.

    Args:
        f: Array of frequency points in Hz.
        z: Complex impedance in Ohms.
        z0: Reference characteristic impedance in Ohms.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    zn = jnp.asarray(z) / z0
    s11 = zn / (zn + 2.0)
    s21 = 2.0 / (zn + 2.0)
    sdict: sax.SDict = {
        ("o1", "o1"): s11,
        ("o2", "o2"): s11,
        ("o1", "o2"): s21,
        ("o2", "o1"): s21,
    }
    return sdict


@jax.jit
def shunt_admittance(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    y: sax.Float = 0.0,
    z0: float = 50.0,
) -> sax.SDict:
    r"""Two-port shunt admittance Sax model.

    .. svgbob::

             o1 ──┬── o2
                  │
                  Y
                  │
                 GND

    See :cite:`m.pozarMicrowaveEngineering2012` (Ch. 4, Table 4.1, Table 4.2, Problem 4.11)
    for the S-parameter derivation.

    Args:
        f: Array of frequency points in Hz.
        y: Complex admittance in Siemens.
        z0: Reference characteristic impedance in Ohms.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    yn = jnp.asarray(y) * z0
    s11 = -yn / (yn + 2.0)
    s21 = 2.0 / (yn + 2.0)
    sdict: sax.SDict = {
        ("o1", "o1"): s11,
        ("o2", "o2"): s11,
        ("o1", "o2"): s21,
        ("o2", "o1"): s21,
    }
    return sdict


@jax.jit
def electrical_short_2_port(f: sax.FloatArrayLike = DEFAULT_FREQUENCY) -> sax.SDict:
    """Electrical short 2-port connection Sax model.

    Args:
        f: Array of frequency points in Hz

    Returns:
        sax.SDict: S-parameters dictionary
    """
    return electrical_short(f=f, n_ports=2)


short = electrical_short
open = electrical_open  # noqa: A001
short_2_port = electrical_short_2_port


@jax.jit(static_argnames=["grounded"])
def lc_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    ground_capacitance: float = 0.0,
) -> sax.SDict:
    r"""LC resonator Sax model with capacitor and inductor in parallel.

    The resonance frequency is given by:

    .. svgbob::

        o1 ──┬──L──┬── o2
             │     │
             └──C──┘

    If grounded=True, a 2-port short is connected to port o2:

    .. svgbob::

        o1 ──┬──L──┬──.
             │     │  | "2-port ground"
             └──C──┘  |
                     "o2"

    Optional ground capacitances Cg can be added to both ports:

    .. svgbob::

             ┌────── C ──────┐
        o1 ──┼────── L ──────┼── o2
             │               │
            Cg              Cg
             │               │
            GND             GND

    .. math::

        f_r = \frac{1}{2 \pi \sqrt{LC}}

    For theory and relation to superconductors, see :cite:`gaoPhysicsSuperconductingMicrowave2008`.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the resonator in Farads.
        inductance: Inductance of the resonator in Henries.
        grounded: If True, add a 2-port ground to the second port.
        ground_capacitance: Parasitic capacitance to ground Cg at each port in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    f = jnp.asarray(f)
    omega = 2 * jnp.pi * f
    z0 = 50.0

    # Calculate physical values
    y_g = 1j * omega * ground_capacitance
    y_lc = 1j * omega * capacitance + 1.0 / (1j * omega * inductance + 1e-25)
    z_lc = 1.0 / (y_lc + 1e-25)

    instances = {
        "cg1": shunt_admittance(f=f, y=y_g, z0=z0),
        "lc": series_impedance(f=f, z=z_lc, z0=z0),
        "cg2": shunt_admittance(f=f, y=y_g, z0=z0),
    }

    connections = {
        "cg1,o2": "lc,o1",
        "lc,o2": "cg2,o1",
    }

    port_o1 = "cg1,o1"
    port_o2 = "cg2,o2"

    if grounded:
        instances["ground"] = electrical_short(f=f, n_ports=2)
        connections[port_o2] = "ground,o1"
        ports = {
            "o1": port_o1,
            "o2": "ground,o2",
        }
    else:
        ports = {
            "o1": port_o1,
            "o2": port_o2,
        }

    return sax.evaluate_circuit_fg((connections, ports), instances)


@jax.jit(static_argnames=["grounded"])
def lc_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    ground_capacitance: float = 0.0,
    coupling_capacitance: float = 10e-15,
    coupling_inductance: float = 0.0,
) -> sax.SDict:
    r"""Coupled LC resonator Sax model.

    This model extends the basic LC resonator by adding a coupling network
    consisting of a parallel capacitor and inductor connected in series
    to one port of the LC resonator.

    The resonance frequency of the main LC resonator is given by:

    .. math::

        f_r = \frac{1}{2 \pi \sqrt{LC}}

    The coupling network modifies the effective coupling to the resonator.

    .. svgbob::


                 +──Lc──+    +──L──+
        o1 ──────│      │────|     │─── o2 or grounded o2
                 +──Cc──+    +──C──+
                           "LC resonator"

    Where :math:`L_\text{c}` and :math:`C_\text{c}` are the coupling inductance and capacitance, respectively.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the main resonator in Farads.
        inductance: Inductance of the main resonator in Henries.
        grounded: If True, the resonator is grounded.
        ground_capacitance: Parasitic capacitance to ground Cg at each port in Farads.
        coupling_capacitance: Coupling capacitance in Farads.
        coupling_inductance: Coupling inductance in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    f = jnp.asarray(f)
    omega = 2 * jnp.pi * f
    z0 = 50.0

    resonator = lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        ground_capacitance=ground_capacitance,
    )

    # Combined coupling admittance (parallel Cc and Lc)
    y_coupling = 1j * omega * coupling_capacitance + 1.0 / (
        1j * omega * coupling_inductance + 1e-25
    )
    z_coupling = 1.0 / (y_coupling + 1e-25)

    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "coupling": series_impedance(f=f, z=z_coupling, z0=z0),
    }

    connections = {
        "coupling,o2": "resonator,o1",
    }

    ports = {
        "o1": "coupling,o1",
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
    plt.plot(jnp.angle(S_cap["o1", "o1"]), abs(S_cap["o1", "o1"]), label="$S_{11}$")
    plt.plot(jnp.angle(S_cap["o1", "o2"]), abs(S_cap["o2", "o1"]), label="$S_{21}$")
    plt.title("S-parameters capacitor")
    plt.legend()
    # Magnitude and phase vs frequency
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_cap["o1", "o1"]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_cap["o1", "o2"]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap["o1", "o1"]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap["o1", "o2"]),
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
    plt.plot(jnp.angle(S_ind["o1", "o1"]), abs(S_ind["o1", "o1"]), label="$S_{11}$")
    plt.plot(jnp.angle(S_ind["o1", "o2"]), abs(S_ind["o2", "o1"]), label="$S_{21}$")
    plt.title("S-parameters inductor")
    plt.legend()
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_ind["o1", "o1"]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_ind["o1", "o2"]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind["o1", "o1"]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind["o1", "o2"]),
        label="∠S21",
        color="C1",
        linestyle="--",
    )
    ax2.set_ylabel("Phase [rad]")
    ax2.legend(loc="upper right")

    plt.title(f"Inductor $S$-parameters ($L={inductance * 1e9}\\,$nH)")
    plt.show()
