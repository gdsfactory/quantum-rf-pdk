"""Josephson junction Models."""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import sax
from sax.models.rf import admittance

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0


def _warn_if_overbiased(ib: float, ic: float) -> None:
    """Host callback to warn if bias current exceeds critical current."""
    if jnp.any(jnp.abs(ib) >= ic):
        warnings.warn(
            "DC bias |I_b| >= I_c detected. Linearized RCSJ model is invalid in the voltage state.",
            RuntimeWarning,
            stacklevel=2,
        )


@partial(jax.jit, inline=True)
def josephson_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic: sax.Float = 1e-6,
    capacitance: sax.Float = 5e-15,
    resistance: sax.Float = 10e3,
    ib: sax.Float = 0.0,
    z0: sax.Complex = 50,
) -> sax.SDict:
    r"""Josephson junction (RCSJ) small-signal Sax model.

    Linearized RCSJ model consisting of a bias-dependent Josephson inductance
    in parallel with capacitance and resistance.

    Valid in the superconducting (zero-voltage) state and for small AC signals.

    Default capacitance taken from :cite:`shcherbakovaFabricationMeasurementsHybrid2015`.

    See :cite:`McCumber1968` for details.

    Args:
        f: Array of frequency points in Hz
        ic: Critical current :math:`I_c` in Amperes
        capacitance: Junction capacitance :math:`C` in Farads
        resistance: Shunt resistance :math:`R` in Ohms
        ib: DC bias current :math:`I_b` in Amperes (:math:`\|I_b\| < I_c`)
        z0: Reference impedance in Ω

    Returns:
        sax.SDict: S-parameters dictionary
    """
    jax.debug.callback(_warn_if_overbiased, ib, ic)

    ω = 2 * jnp.pi * jnp.asarray(f)

    # Bias-dependent phase factor
    # jnp.clip ensures we don't get NaNs during compilation tracing or slightly overbiased states
    cos_Φ_0 = jnp.sqrt(jnp.clip(1.0 - (ib / ic) ** 2, a_min=1e-10))

    # Josephson inductance
    Lⱼ = Φ_0 / (2 * jnp.pi * ic * cos_Φ_0)

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    Y_L = 1 / (1j * ω * Lⱼ)

    # Total admittance
    Y_JJ = Y_R + Y_C + Y_L

    return admittance(f=f, y=Y_JJ, z0=z0)


@partial(jax.jit, inline=True)
def squid_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic_tot: sax.Float = 2e-6,
    asymmetry: sax.Float = 0.0,
    capacitance: sax.Float = 10e-15,
    resistance: sax.Float = 5e3,
    ib: sax.Float = 0.0,
    flux: sax.Float = 0.0,
    z0: sax.Complex = 50,
) -> sax.SDict:
    r"""DC SQUID small-signal Sax model in the zero-screening limit.

    Treats the DC SQUID as a single effective RCSJ whose critical current is
    tunable by an external magnetic flux. Assumes negligible loop geometric inductance.

    See :cite:`kochChargeinsensitiveQubitDesign2007a` and :cite:`tinkhamIntroductionSuperconductivity2015`
    for details on asymmetric SQUIDs and effective Josephson inductance.

    Args:
        f: Array of frequency points in Hz
        ic_tot: Total critical current sum :math:`I_{c1} + I_{c2}` in Amperes
        asymmetry: Junction asymmetry :math:`(I_{c1} - I_{c2}) / I_{c,tot}`
        capacitance: Total SQUID capacitance :math:`C_1 + C_2` in Farads
        resistance: Total SQUID shunt resistance :math:`R_1 || R_2` in Ohms
        ib: DC bias current :math:`I_b` in Amperes
        flux: External magnetic flux :math:`\Phi_{ext}` in Webers
        z0: Reference impedance in Ω

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
    # Note: Model is only valid when |I_b| < ic_squid. Clip to prevent NaNs if violated.
    cos_Φ_0_eff = jnp.sqrt(jnp.clip(1.0 - (ib / ic_squid) ** 2, a_min=1e-10))

    # Effective Josephson inductance
    Lⱼ_eff = Φ_0 / (2 * jnp.pi * ic_squid * cos_Φ_0_eff)

    # Admittances (parallel RCSJ)
    Y_R = 1 / resistance
    Y_C = 1j * ω * capacitance
    Y_L = 1 / (1j * ω * Lⱼ_eff)

    # Total SQUID admittance
    Y_SQUID = Y_R + Y_C + Y_L

    return admittance(f=f, y=Y_SQUID, z0=z0)


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
