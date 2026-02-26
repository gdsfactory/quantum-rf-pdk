"""Qubit Models."""

import jax.numpy as jnp
import sax

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.generic import lc_resonator_coupled


def lc_resonator_capacitive(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance: sax.Float = 1e-9,
    capacitance: sax.Float = 1e-15,
    coupling_capacitance: sax.Float = 10e-15,
    grounded: bool = False,
) -> sax.SType:
    r"""Transmon qubit LC resonator model with capacitive coupling.

    Models a parallel LC resonator representing a transmon qubit, coupled via
    a series capacitor to an external port. The resonator consists of a parallel
    combination of an inductor (representing the Josephson inductance) and a
    capacitor (representing the shunt capacitance).

    .. svgbob::

                    ┌──────────┐
        o1 ─────C_c─┤ L  ||  C ├─── o2 (or GND)
                    └──────────┘

    The resonance frequency is given by:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Args:
        f: Array of frequency points in Hz
        inductance: Resonator inductance :math:`L` in Henries (default: 1 nH)
        capacitance: Resonator capacitance :math:`C` in Farads (default: 1 fF)
        coupling_capacitance: Coupling capacitance :math:`C_c` in Farads (default: 10 fF)
        grounded: If True, the resonator o2 port is grounded (default: False)

    Returns:
        sax.SType: S-parameters dictionary with ports ("o1", "o2")

    Note:
        For a transmon qubit, typical values are:
        - Inductance: 0.5-2 nH (set by junction critical current)
        - Capacitance: 50-100 fF (shunt capacitance to ground)
        - Coupling capacitance: 1-50 fF (controls coupling strength)

    References:
        See :cite:`kochChargeinsensitiveQubitDesign2007a` for transmon design details.
    """
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=0.0,
    )


def lc_resonator_inductive(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance: sax.Float = 1e-9,
    capacitance: sax.Float = 1e-15,
    coupling_inductance: sax.Float = 1e-9,
    grounded: bool = False,
) -> sax.SType:
    r"""Transmon qubit LC resonator model with inductive coupling.

    Models a parallel LC resonator representing a transmon qubit, coupled via
    an inductor to an external port. The resonator consists of a parallel
    combination of an inductor (representing the Josephson inductance) and a
    capacitor (representing the shunt capacitance).

    .. svgbob::

                    ┌──────────┐
        o1 ─────L_c─┤ L  ||  C ├─── o2 (or GND)
                    └──────────┘

    The resonance frequency is given by:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Args:
        f: Array of frequency points in Hz
        inductance: Resonator inductance :math:`L` in Henries (default: 1 nH)
        capacitance: Resonator capacitance :math:`C` in Farads (default: 1 fF)
        coupling_inductance: Coupling inductor :math:`L_c` in Henries (default: 1 nH)
        grounded: If True, the resonator o2 port is grounded (default: False)

    Returns:
        sax.SType: S-parameters dictionary with ports ("o1", "o2")

    Note:
        For a transmon qubit with inductive coupling, typical values are:
        - Inductance: 0.5-2 nH (set by junction critical current)
        - Capacitance: 50-100 fF (shunt capacitance to ground)
        - Coupling inductance: 0.5-2 nH

    References:
        See :cite:`kochChargeinsensitiveQubitDesign2007a` for transmon design details.
    """
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        coupling_capacitance=0.0,
        coupling_inductance=coupling_inductance,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sax.set_port_naming_strategy("optical")

    # Test capacitive coupling
    f = jnp.linspace(1e9, 10e9, 500)
    s_cap = lc_resonator_capacitive(
        f=f,
        inductance=1e-9,
        capacitance=50e-15,
        coupling_capacitance=10e-15,
    )
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(f / 1e9, jnp.abs(s_cap[("o1", "o1")]), label="|S11|")
    plt.plot(f / 1e9, jnp.abs(s_cap[("o1", "o2")]), label="|S12|")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("LC Resonator - Capacitive Coupling")
    plt.grid(True)

    # Test inductive coupling
    s_ind = lc_resonator_inductive(
        f=f,
        inductance=1e-9,
        capacitance=50e-15,
        coupling_inductance=1e-9,
    )
    plt.subplot(122)
    plt.plot(f / 1e9, jnp.abs(s_ind[("o1", "o1")]), label="|S11|")
    plt.plot(f / 1e9, jnp.abs(s_ind[("o1", "o2")]), label="|S12|")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("LC Resonator - Inductive Coupling")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
