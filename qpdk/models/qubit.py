"""Qubit LC resonator models.

This module provides LC resonator models for superconducting transmon qubits
and coupled qubit systems. The models are based on the standard LC resonator
formulation with appropriate grounding configurations.

For double-island transmon qubits, we use an ungrounded LC resonator since
the two islands are floating. For shunted transmon qubits, one island is
grounded, so we use a grounded LC resonator.

The helper functions convert between qubit Hamiltonian parameters (charging
energy $E_C$, Josephson energy $E_J$, coupling strength $g$) and the
corresponding circuit parameters (capacitance, inductance).

References:
    - Koch et al., "Charge-insensitive qubit design derived from the
      Cooper pair box", Phys. Rev. A 76, 042319 (2007)
    - Gao, "The physics of superconducting microwave resonators",
      PhD thesis, Caltech (2008)
"""

from functools import partial

import jax
import jax.numpy as jnp
import sax
import scipy.constants

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.generic import lc_resonator, lc_resonator_coupled

__all__ = [
    "coupling_strength_to_capacitance",
    "double_island_transmon",
    "ec_to_capacitance",
    "ej_to_inductance",
    "shunted_transmon",
    "transmon_coupled",
]

# Physical constants
_e = scipy.constants.e  # electron charge (C)
_h = scipy.constants.h  # Planck constant (J·s)
_Φ_0 = scipy.constants.physical_constants["mag. flux quantum"][0]  # flux quantum (Wb)


def ec_to_capacitance(ec_ghz: float) -> float:
    r"""Convert charging energy $E_C$ to total capacitance $C_\Sigma$.

    The charging energy is related to capacitance by:

    .. math::

        E_C = \frac{e^2}{2 C_\Sigma}

    where $e$ is the electron charge.

    Args:
        ec_ghz: Charging energy in GHz.

    Returns:
        Total capacitance in Farads.

    Example:
        >>> C = ec_to_capacitance(0.2)  # 0.2 GHz (200 MHz) charging energy
        >>> print(f"{C * 1e15:.1f} fF")  # ~96 fF
    """
    ec_joules = ec_ghz * 1e9 * _h  # Convert GHz to Joules
    return _e**2 / (2 * ec_joules)


def ej_to_inductance(ej_ghz: float) -> float:
    r"""Convert Josephson energy $E_J$ to Josephson inductance $L_J$.

    The Josephson energy is related to inductance by:

    .. math::

        E_J = \frac{\Phi_0^2}{4 \pi^2 L_J} = \frac{(\hbar / 2e)^2}{L_J}

    This is equivalent to:

    .. math::

        L_J = \frac{\Phi_0}{2 \pi I_c}

    where $I_c$ is the critical current and $\Phi_0$ is the magnetic flux quantum.

    Args:
        ej_ghz: Josephson energy in GHz.

    Returns:
        Josephson inductance in Henries.

    Example:
        >>> L = ej_to_inductance(20.0)  # 20 GHz Josephson energy
        >>> print(f"{L * 1e9:.2f} nH")  # ~1.0 nH
    """
    ej_joules = ej_ghz * 1e9 * _h  # Convert GHz to Joules
    # L_J = Φ_0² / (4π² E_J)
    return _Φ_0**2 / (4 * jnp.pi**2 * ej_joules)


def coupling_strength_to_capacitance(
    g_ghz: float,
    c_sigma: float,
    c_r: float,
    omega_q_ghz: float,
    omega_r_ghz: float,
) -> float:
    r"""Convert coupling strength $g$ to coupling capacitance $C_c$.

    In the dispersive limit ($g \ll \omega_q, \omega_r$), the coupling strength
    can be related to a coupling capacitance via:

    .. math::

        g \approx \frac{1}{2} \frac{C_c}{\sqrt{C_\Sigma C_r}} \sqrt{\omega_q \omega_r}

    Solving for $C_c$:

    .. math::

        C_c = \frac{2g}{\sqrt{\omega_q \omega_r}} \sqrt{C_\Sigma C_r}

    See Savola et al. (2023) for details.

    Args:
        g_ghz: Coupling strength in GHz.
        c_sigma: Total qubit capacitance in Farads.
        c_r: Total resonator capacitance in Farads.
        omega_q_ghz: Qubit frequency in GHz (angular frequency / 2π).
        omega_r_ghz: Resonator frequency in GHz (angular frequency / 2π).

    Returns:
        Coupling capacitance in Farads.

    Example:
        >>> C_c = coupling_strength_to_capacitance(
        ...     g_ghz=0.1,
        ...     c_sigma=100e-15,  # 100 fF
        ...     c_r=50e-15,  # 50 fF
        ...     omega_q_ghz=5.0,
        ...     omega_r_ghz=7.0,
        ... )
        >>> print(f"{C_c * 1e15:.2f} fF")
    """
    # Use frequencies directly (already in GHz, ratio is dimensionless)
    sqrt_omega = jnp.sqrt(omega_q_ghz * omega_r_ghz)
    sqrt_c = jnp.sqrt(c_sigma * c_r)
    return 2 * g_ghz / sqrt_omega * sqrt_c


@partial(jax.jit, inline=True)
def double_island_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
) -> sax.SType:
    r"""LC resonator model for a double-island transmon qubit.

    A double-island transmon has two superconducting islands connected by
    Josephson junctions, with both islands floating (not grounded). This is
    modeled as an ungrounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_q \approx \frac{1}{2\pi} \sqrt{8 E_J E_C} - E_C

    For the LC model, the resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters.

    .. svgbob::

        o1 ──┬──L──┬── o2
             │     │
             └──C──┘

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance $C_\Sigma$ of the qubit in Farads.
        inductance: Josephson inductance $L_J$ in Henries.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.

    Example:
        >>> import jax.numpy as jnp
        >>> f = jnp.linspace(4e9, 8e9, 100)
        >>> S = double_island_transmon(f=f, capacitance=80e-15, inductance=1.2e-9)
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=False,
    )


@partial(jax.jit, inline=True)
def shunted_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
) -> sax.SType:
    r"""LC resonator model for a shunted transmon qubit.

    A shunted transmon has one island grounded and the other island connected
    to the junction. This is modeled as a grounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_q \approx \frac{1}{2\pi} \sqrt{8 E_J E_C} - E_C

    For the LC model, the resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters.

    .. svgbob::

        o1 ──┬──L──┬──.
             │     │  | "2-port ground"
             └──C──┘  |
                     "o2"

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance $C_\Sigma$ of the qubit in Farads.
        inductance: Josephson inductance $L_J$ in Henries.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.

    Example:
        >>> import jax.numpy as jnp
        >>> f = jnp.linspace(4e9, 8e9, 100)
        >>> S = shunted_transmon(f=f, capacitance=80e-15, inductance=1.2e-9)
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=True,
    )


@partial(jax.jit, static_argnames=["grounded"])
def transmon_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    coupling_capacitance: float = 10e-15,
    coupling_inductance: float = 0.0,
) -> sax.SType:
    r"""Coupled transmon qubit model.

    This model extends the basic transmon qubit by adding a coupling network
    consisting of a parallel capacitor and/or inductor. This can represent
    capacitive or inductive coupling between qubits or between a qubit and
    a readout resonator.

    The coupling network is connected in series with the LC resonator:

    .. svgbob::

             +──Lc──+    +──L──+
        o1 ──│      │────|     │─── o2 or grounded o2
             +──Cc──+    +──C──+
                       "LC resonator"

    For capacitive coupling (common for qubit-resonator coupling):
        - Set ``coupling_capacitance`` to the coupling capacitor value
        - Set ``coupling_inductance=0.0``

    For inductive coupling (common for flux-tunable coupling):
        - Set ``coupling_inductance`` to the coupling inductor value
        - Set ``coupling_capacitance=0.0`` (or small value)

    Use :func:`coupling_strength_to_capacitance` to convert from the
    qubit-resonator coupling strength $g$ to the coupling capacitance.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance $C_\Sigma$ of the qubit in Farads.
        inductance: Josephson inductance $L_J$ in Henries.
        grounded: If True, the qubit is a shunted transmon (grounded).
            If False, it is a double-island transmon (ungrounded).
        coupling_capacitance: Coupling capacitance $C_c$ in Farads.
        coupling_inductance: Coupling inductance $L_c$ in Henries.

    Returns:
        sax.SType: S-parameters dictionary with ports o1 and o2.

    Example:
        >>> import jax.numpy as jnp
        >>> f = jnp.linspace(4e9, 8e9, 100)
        >>> # Capacitively coupled transmon
        >>> S = transmon_coupled(
        ...     f=f,
        ...     capacitance=80e-15,
        ...     inductance=1.2e-9,
        ...     coupling_capacitance=5e-15,
        ... )
    """
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=coupling_inductance,
    )
