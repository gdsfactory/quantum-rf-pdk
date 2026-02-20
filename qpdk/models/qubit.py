"""Qubit Models."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from sax.models.rf import capacitor, inductor, tee

from qpdk.models.constants import DEFAULT_FREQUENCY


@partial(jax.jit, inline=True)
def lc_resonator_capacitive(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance: sax.Float = 1e-9,
    capacitance: sax.Float = 1e-15,
    coupling_capacitance: sax.Float = 10e-15,
    z0: sax.Complex = 50,
) -> sax.SType:
    r"""Transmon qubit LC resonator model with capacitive coupling.

    Models a parallel LC resonator representing a transmon qubit, coupled via
    a series capacitor to an external port. The resonator consists of a parallel
    combination of an inductor (representing the Josephson inductance) and a
    capacitor (representing the shunt capacitance).

    .. svgbob::

                    ┌──────────┐
        o1 ─────C_c─┤ L   ||  C ├─── GND
                    └──────────┘

    The resonance frequency is given by:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Args:
        f: Array of frequency points in Hz
        inductance: Resonator inductance :math:`L` in Henries (default: 1 nH)
        capacitance: Resonator capacitance :math:`C` in Farads (default: 1 fF)
        coupling_capacitance: Coupling capacitance :math:`C_c` in Farads (default: 10 fF)
        z0: Reference impedance in Ω (default: 50)

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
    # Create coupling capacitor
    c_coupling = capacitor(f=f, capacitance=coupling_capacitance, z0=z0)

    # Create parallel LC resonator components
    l_res = inductor(f=f, inductance=inductance, z0=z0)
    c_res = capacitor(f=f, capacitance=capacitance, z0=z0)

    # Build circuit using SAX
    # Port naming: o1 is input, o2 is the LC resonator (grounded)
    instances = {
        "c_coupling": c_coupling,
        "tee": tee(f=f),
        "l_res": l_res,
        "c_res": c_res,
    }

    # Connect: input -> coupling cap -> tee (port 0)
    #          tee port 1 -> inductor -> ground
    #          tee port 2 -> capacitor -> ground
    connections = {
        "c_coupling,o2": "tee,o1",
        "tee,o2": "l_res,o1",
        "tee,o3": "c_res,o1",
    }

    # External ports
    ports = {
        "o1": "c_coupling,o1",
        "o2": "l_res,o2",  # Grounded port (for reference)
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


@partial(jax.jit, inline=True)
def lc_resonator_inductive(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    inductance: sax.Float = 1e-9,
    capacitance: sax.Float = 1e-15,
    mutual_inductance: sax.Float = 0.1e-9,
    coupling_inductance: sax.Float = 1e-9,
    z0: sax.Complex = 50,
) -> sax.SType:
    r"""Transmon qubit LC resonator model with inductive coupling.

    Models a parallel LC resonator representing a transmon qubit, coupled via
    mutual inductance to an external inductor. The resonator consists of a parallel
    combination of an inductor (representing the Josephson inductance) and a
    capacitor (representing the shunt capacitance).

    .. svgbob::

                    ┌──────────┐
        o1 ───L_c~~~┤ L   ||  C ├─── GND
                    └──────────┘

    The mutual inductance :math:`M` provides inductive coupling between the
    coupling inductor :math:`L_c` and the resonator inductor :math:`L`.

    The coupled inductance is approximated as:

    .. math::

        L_{eff} = L_c + L - 2M

    Args:
        f: Array of frequency points in Hz
        inductance: Resonator inductance :math:`L` in Henries (default: 1 nH)
        capacitance: Resonator capacitance :math:`C` in Farads (default: 1 fF)
        mutual_inductance: Mutual inductance :math:`M` in Henries (default: 0.1 nH)
        coupling_inductance: Coupling inductor :math:`L_c` in Henries (default: 1 nH)
        z0: Reference impedance in Ω (default: 50)

    Returns:
        sax.SType: S-parameters dictionary with ports ("o1", "o2")

    Note:
        For a transmon qubit with inductive coupling, typical values are:
        - Inductance: 0.5-2 nH (set by junction critical current)
        - Capacitance: 50-100 fF (shunt capacitance to ground)
        - Mutual inductance: 0.01-0.5 nH (controls coupling strength)
        - Coupling inductance: 0.5-2 nH

    References:
        See :cite:`kochChargeinsensitiveQubitDesign2007a` for transmon design details.
    """
    # Create coupling inductor
    l_coupling = inductor(f=f, inductance=coupling_inductance, z0=z0)

    # Create effective inductance accounting for mutual coupling
    # In a transformer model, the effective inductance seen is modified by mutual inductance
    # For a simplified model, we reduce the total inductance by the mutual term
    # This is an approximation; a full transformer model would be more complex
    l_eff = jnp.maximum(inductance - mutual_inductance, 1e-12)  # Ensure positive
    l_res = inductor(f=f, inductance=l_eff, z0=z0)

    # Create resonator capacitor
    c_res = capacitor(f=f, capacitance=capacitance, z0=z0)

    # Build circuit using SAX
    instances = {
        "l_coupling": l_coupling,
        "tee": tee(f=f),
        "l_res": l_res,
        "c_res": c_res,
    }

    # Connect: input -> coupling inductor -> tee (port 0)
    #          tee port 1 -> resonator inductor -> ground
    #          tee port 2 -> capacitor -> ground
    connections = {
        "l_coupling,o2": "tee,o1",
        "tee,o2": "l_res,o1",
        "tee,o3": "c_res,o1",
    }

    # External ports
    ports = {
        "o1": "l_coupling,o1",
        "o2": "l_res,o2",  # Grounded port (for reference)
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


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
        mutual_inductance=0.1e-9,
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
