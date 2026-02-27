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


@jax.jit(static_argnames=["grounded"])
def lc_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
) -> sax.SType:
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

    .. math::

        f_r = \frac{1}{2 \pi \sqrt{LC}}

    For theory and relation to superconductors, see :cite:`gaoPhysicsSuperconductingMicrowave2008`.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the resonator in Farads.
        inductance: Inductance of the resonator in Henries.
        grounded: If True, add a 2-port ground to the second port.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.
    """
    f = jnp.asarray(f)

    instances = {
        "capacitor": capacitor(f=f, capacitance=capacitance),
        "inductor": inductor(f=f, inductance=inductance),
        "tee_1": tee(f=f),
        "tee_2": tee(f=f),
    }

    connections = {
        "tee_1,o2": "capacitor,o1",
        "tee_1,o3": "inductor,o1",
        "capacitor,o2": "tee_2,o2",
        "inductor,o2": "tee_2,o3",
    }

    if grounded:
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


@jax.jit(static_argnames=["grounded"])
def _lc_resonator_coupled_impl(
    f: sax.FloatArrayLike,
    capacitance: float,
    inductance: float,
    grounded: bool,
    coupling_capacitance: float,
    coupling_inductance: float,
) -> sax.SType:
    """Internal JIT-compiled implementation of lc_resonator_coupled.

    This function contains the actual circuit construction and should not
    be called directly. Use lc_resonator_coupled instead.
    """
    f = jnp.asarray(f)
    resonator = lc_resonator(
        f=f, capacitance=capacitance, inductance=inductance, grounded=grounded
    )

    # Always use the full tee network topology for consistent behavior
    # When an element has zero value, it naturally produces the correct S-parameters
    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "tee_between": tee(f=f),
        "tee_outer": tee(f=f),
        "inductive_coupling": inductor(f=f, inductance=coupling_inductance),
        "capacitive_coupling": capacitor(f=f, capacitance=coupling_capacitance),
    }

    connections = {
        "tee_outer,o2": "inductive_coupling,o1",
        "tee_outer,o3": "capacitive_coupling,o1",
        "inductive_coupling,o2": "tee_between,o2",
        "capacitive_coupling,o2": "tee_between,o3",
        "tee_between,o1": "resonator,o1",
    }

    ports = {
        "o1": "tee_outer,o1",
        "o2": "resonator,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def lc_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    coupling_capacitance: float = 10e-15,
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


                 +──Lc──+    +──L──+
        o1 ──────│      │────|     │─── o2 or grounded o2
                 +──Cc──+    +──C──+
                           "LC resonator"

    Where :math:`L_\text{c}` and :math:`C_\text{c}` are the coupling inductance and capacitance, respectively.

    .. note::

        The coupling parameters (coupling_capacitance and coupling_inductance) are
        not static arguments, allowing them to be differentiated with JAX transforms
        such as :func:`jax.grad`. This is important for circuit optimization workflows.
        At least one of the coupling parameters must be non-zero.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Capacitance of the main resonator in Farads.
        inductance: Inductance of the main resonator in Henries.
        grounded: If True, the resonator is grounded.
        coupling_capacitance: Coupling capacitance in Farads.
        coupling_inductance: Coupling inductance in Henries.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.

    Raises:
        ValueError: If both coupling_capacitance and coupling_inductance are zero.
    """
    # Validate that at least one coupling element is non-zero
    # This validation is done outside the JIT boundary to avoid tracer issues
    if coupling_inductance == 0.0 and coupling_capacitance == 0.0:
        raise ValueError(
            "At least one of coupling_capacitance or coupling_inductance must be non-zero. "
            "Both cannot be zero simultaneously."
        )

    return _lc_resonator_coupled_impl(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=coupling_inductance,
    )


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
