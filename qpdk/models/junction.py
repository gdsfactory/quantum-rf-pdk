"""Josephson junction Models."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec
from sax.models.rf import electrical_short

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0
from qpdk.models.generic import series_impedance
from qpdk.models.jit import jit
from qpdk.models.waveguides import straight


def _warn_if_overbiased(ib: float, ic: float) -> None:
    """Host callback to warn if bias current exceeds critical current.

    Uses NumPy, not JAX: dispatching JAX operations from inside a host callback
    can deadlock against the computation that invoked it.
    """
    if np.any(np.abs(np.asarray(ib)) >= np.asarray(ic)):
        warnings.warn(
            "DC bias |I_b| >= I_c detected. Linearized RCSJ model is invalid in the voltage state. "
            "Please check your 'ib' and 'ic' ('ic_tot', 'flux', 'asymmetry') inputs.",
            RuntimeWarning,
            stacklevel=0,
        )


@jit(inline=True)
def josephson_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic: sax.Float = 1e-6,
    capacitance: sax.Float = 5e-15,
    resistance: sax.Float = 10e3,
    ib: sax.Float = 0.0,
) -> sax.SDict:
    r"""Josephson junction (RCSJ) small-signal Sax model.

    Linearized RCSJ model consisting of a bias-dependent Josephson inductance
    in parallel with capacitance and resistance.

    Valid in the superconducting (zero-voltage) state and for small AC signals.

    Default capacitance taken from :cite:`shcherbakovaFabricationMeasurementsHybrid2015`.

    See :cite:`McCumber1968` for details.

    Args:
        f: Array of frequency points in Hz
        ic: Critical current :math:`I_\text{c}` in Amperes
        capacitance: Junction capacitance :math:`C` in Farads
        resistance: Shunt resistance :math:`R` in Ohms
        ib: DC bias current :math:`I_\text{b}` in Amperes (:math:`\|I_\text{b}\| < I_\text{c}`)

    Returns:
        sax.SDict: S-parameters dictionary
    """
    jax.debug.callback(_warn_if_overbiased, ib, ic)

    ω = 2 * jnp.pi * jnp.asarray(f)

    # Bias-dependent phase factor
    # jnp.clip ensures we don't get NaNs during compilation tracing or slightly overbiased states
    cos_Φ_0 = jnp.sqrt(jnp.clip(1.0 - (ib / ic) ** 2, min=1e-10))

    # Josephson inductance
    Lⱼ = Φ_0 / (2 * jnp.pi * ic * cos_Φ_0)

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    Y_L = 1 / (1j * ω * Lⱼ)

    # Total admittance
    Y_JJ = Y_R + Y_C + Y_L

    return series_impedance(f=f, z=1 / Y_JJ)


@jit(inline=True)
def squid_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic_tot: sax.Float = 2e-6,
    asymmetry: sax.Float = 0.0,
    capacitance: sax.Float = 10e-15,
    resistance: sax.Float = 5e3,
    ib: sax.Float = 0.0,
    flux: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""DC SQUID small-signal Sax model in the zero-screening limit.

    Treats the DC SQUID as a single effective RCSJ whose critical current is
    tunable by an external magnetic flux. Assumes negligible loop geometric inductance.

    See :cite:`kochChargeinsensitiveQubitDesign2007a` and :cite:`tinkhamIntroductionSuperconductivity2015`
    for details on asymmetric SQUIDs and effective Josephson inductance.

    Args:
        f: Array of frequency points in Hz
        ic_tot: Total critical current sum :math:`I_{\text{c}1} + I_{\text{c}2}` in Amperes
        asymmetry: Junction asymmetry :math:`(I_{\text{c}1} - I_{\text{c}2}) / I_{\text{c,tot}}`
        capacitance: Total SQUID capacitance :math:`C_1 + C_2` in Farads
        resistance: Total SQUID shunt resistance :math:`R_1 || R_2` in Ohms
        ib: DC bias current :math:`I_\text{b}` in Amperes
        flux: External magnetic flux :math:`\Phi_{\text{ext}}` in Webers

    Returns:
        sax.SDict: S-parameters dictionary
    """
    ω = 2 * jnp.pi * jnp.asarray(f)

    # Normalized flux
    phi_reduced = jnp.pi * flux / Φ_0

    # Flux-tunable critical current for an asymmetric SQUID
    ic_squid = ic_tot * jnp.sqrt(
        jnp.cos(phi_reduced) ** 2 + (asymmetry**2) * jnp.sin(phi_reduced) ** 2
    )

    jax.debug.callback(_warn_if_overbiased, ib, ic_squid)

    # Bias-dependent phase factor
    # Guard against ic_squid == 0 (e.g. symmetric SQUID at half-flux quantum)
    ratio_sq = jnp.where(
        ic_squid > 0, (ib / jnp.where(ic_squid > 0, ic_squid, 1.0)) ** 2, 0.0
    )
    cos_Φ_0_eff = jnp.sqrt(jnp.clip(1.0 - ratio_sq, min=1e-10))

    # Effective Josephson inductance (will go to inf when ic_squid == 0)
    Lⱼ_eff = Φ_0 / (2 * jnp.pi * jnp.where(ic_squid > 0, ic_squid, 1e-20) * cos_Φ_0_eff)

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    # Guard against division by zero in Y_L if L_j_eff somehow goes to zero or Y_L -> 0 if L_j_eff is inf
    Y_L = jnp.where(ic_squid > 0, 1 / (1j * ω * Lⱼ_eff), 0j)

    # Total SQUID admittance
    Y_SQUID = Y_R + Y_C + Y_L

    return series_impedance(f=f, z=1 / Y_SQUID)


def squid_termination(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 10.0,
    cross_section: CrossSectionSpec = "cpw",
    ic_tot: sax.Float = 4.8e-6,
    asymmetry: sax.Float = 0.0,
    capacitance: sax.Float = 10e-15,
    resistance: sax.Float = 5e3,
    flux: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""One-port Sax model of a CPW lead shorted to ground through a DC SQUID.

    This is the termination of a flux-driven :math:`\lambda/4` Josephson
    parametric amplifier (JPA): a CPW lead of ``length``, then the
    :func:`~qpdk.models.junction.squid_junction` admittance in series with an
    ideal short. It models the :func:`~qpdk.cells.squid_termination` cell. The flux-tunable SQUID inductance
    :math:`L_\text{s}(\Phi) = \Phi_0 / (2 \pi I_\text{c,tot} |\cos(\pi \Phi / \Phi_0)|)`
    pulls the resonance below :math:`\omega_{\lambda/4}`
    :cite:`yamamotoFluxdrivenJosephsonParametric2008`.

    The loop geometric inductance is neglected, which holds for
    :math:`\beta_\text{L} = 2 L_\text{loop} I_\text{c} / \Phi_0 \lesssim 0.1`.
    The default :math:`I_\text{c,tot} = 2 \times 2.4\,\text{µA}` follows the
    low-:math:`\beta_\text{L}` WMI devices :cite:`pogorzalekHystereticFluxResponse2017`.
    It is an assumption, not a value extracted from the layout.

    Args:
        f: Array of frequency points in Hz
        length: Length of the CPW lead in µm
        cross_section: Cross-section of the CPW lead
        ic_tot: Total critical current :math:`I_{\text{c}1} + I_{\text{c}2}` in Amperes
        asymmetry: Junction asymmetry :math:`(I_{\text{c}1} - I_{\text{c}2}) / I_{\text{c,tot}}`
        capacitance: Total SQUID capacitance :math:`C_1 + C_2` in Farads
        resistance: Total SQUID shunt resistance :math:`R_1 || R_2` in Ohms
        flux: External magnetic flux :math:`\Phi_{\text{ext}}` in Webers

    Returns:
        sax.SDict: S-parameters dictionary with the single port ``o1``
    """
    instances: dict[str, sax.SType] = {
        "lead": straight(f=f, length=length, cross_section=cross_section),
        "squid": squid_junction(
            f=f,
            ic_tot=ic_tot,
            asymmetry=asymmetry,
            capacitance=capacitance,
            resistance=resistance,
            flux=flux,
        ),
        "short": electrical_short(f=f),
    }
    connections = {"lead,o2": "squid,o1", "squid,o2": "short,o1"}
    ports = {"o1": "lead,o1"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sax.set_port_naming_strategy("optical")

    f = jnp.linspace(1e9, 10e9, 500)
    s = josephson_junction(f=f, ic=1e-6, capacitance=50e-15, resistance=10e3, ib=0.5e-6)
    plt.figure()
    plt.plot(f / 1e9, jnp.abs(s["o1", "o1"]), label="|S11|")
    plt.plot(f / 1e9, jnp.abs(s["o1", "o2"]), label="|S12|")
    plt.plot(f / 1e9, jnp.abs(s["o2", "o2"]), label="|S22|")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Josephson Junction S-parameters")
    plt.show()
