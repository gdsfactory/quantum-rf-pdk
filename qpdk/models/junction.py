"""Josephson junction Models."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import sax
from sax.models.rf import admittance

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0
from qpdk.models.jit import jit


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

    return admittance(f=f, y=Y_JJ)


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

    return admittance(f=f, y=Y_SQUID)


def _abcd_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Multiply two batches of 2×2 ABCD matrices with shape ``(..., 2, 2)``."""
    return jnp.einsum("...ij,...jk->...ik", a, b)


def _abcd_power(t: jax.Array, n: int) -> jax.Array:
    r"""Raise a batch of ABCD matrices to a static integer power by squaring.

    Uses :math:`\mathcal{O}(\log n)` matrix products, unrolled at trace time.

    Returns:
        The batch of matrices :math:`T^n`.
    """
    result = jnp.broadcast_to(jnp.eye(2, dtype=t.dtype), t.shape)
    base = t
    while n > 0:
        if n & 1:
            result = _abcd_matmul(result, base)
        n >>= 1
        if n:
            base = _abcd_matmul(base, base)
    return result


def _abcd_to_sdict(abcd: jax.Array, z0: sax.Float) -> sax.SDict:
    """Convert a batch of reciprocal ABCD matrices to a two-port S-parameter dictionary.

    See :cite:`m.pozarMicrowaveEngineering2012` (Ch. 4, Table 4.2). The network is
    assumed reciprocal, :math:`AD - BC = 1`, so :math:`S_{12} = S_{21}` exactly
    instead of carrying the round-off of a long cascade into the determinant.

    Returns:
        The S-parameter dictionary with ports ``o1`` and ``o2``.
    """
    a, b, c, d = abcd[..., 0, 0], abcd[..., 0, 1], abcd[..., 1, 0], abcd[..., 1, 1]
    denominator = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / denominator
    s22 = (-a + b / z0 - c * z0 + d) / denominator
    s21 = 2 / denominator
    s12 = s21
    return {
        ("o1", "o1"): s11,
        ("o1", "o2"): s12,
        ("o2", "o1"): s21,
        ("o2", "o2"): s22,
    }


@jit(inline=True, static_argnames=("n_junctions",))
def josephson_junction_array(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_junctions: int = 43,
    ic: sax.Float = Φ_0 / (2 * np.pi * 1.9e-9),
    capacitance: sax.Float = 40e-15,
    ground_capacitance: sax.Float = 0.04e-15,
    resistance: sax.Float = np.inf,
    ib: sax.Float = 0.0,
    z0: sax.Float = 50.0,
) -> sax.SDict:
    r"""Josephson junction array (junction chain) small-signal Sax model.

    A chain of :math:`N` identical linearised junctions in series between ports
    ``o1`` and ``o2``. Each junction is the parallel RCSJ element of
    :func:`josephson_junction`, with Josephson inductance
    :math:`L_\text{J} = \Phi_0 / (2 \pi I_\text{c} \cos\varphi_0)`, junction
    capacitance :math:`C_\text{J}` and resistance :math:`R`. Each of the
    :math:`N - 1` interior islands has a capacitance :math:`C_\text{g}` to ground:

    .. svgbob::

        o1 ──[J]──┬──[J]──┬── ... ──┬──[J]── o2
                  │       │         │
                 C_g     C_g       C_g
                  │       │         │
                 GND     GND       GND

    The two-port is evaluated as the ABCD cascade

    .. math::

        \mathbf{T} = \left(\mathbf{T}_\text{J} \mathbf{T}_\text{g}\right)^{N-1} \mathbf{T}_\text{J},
        \qquad
        \mathbf{T}_\text{J} = \begin{pmatrix} 1 & Z_\text{J} \\ 0 & 1 \end{pmatrix},
        \qquad
        \mathbf{T}_\text{g} = \begin{pmatrix} 1 & 0 \\ \mathrm{j} \omega C_\text{g} & 1 \end{pmatrix},

    with :math:`Z_\text{J} = (1/R + \mathrm{j}\omega C_\text{J} + 1/(\mathrm{j}\omega L_\text{J}))^{-1}`,
    and converted to S-parameters referenced to :math:`Z_0`
    :cite:`m.pozarMicrowaveEngineering2012`. The cost grows as :math:`\log N`.

    For a lossless chain with both ends grounded, the cascade resonates at the
    normal modes of the array :cite:`maslukMicrowaveCharacterizationJosephson2012`:

    .. math::

        \omega_k = \omega_\text{p} \sqrt{\frac{1 - \cos(\pi k / N)}
        {C_\text{g} / (2 C_\text{J}) + 1 - \cos(\pi k / N)}},
        \qquad k = 1, \dots, N - 1,

    where :math:`\omega_\text{p} = 1 / \sqrt{L_\text{J} C_\text{J}}` is the
    plasma frequency that bounds the spectrum from above.

    Assumptions:
        - Small-signal, zero-voltage state: each junction is a linear inductor.
        - Identical junctions and uniform :math:`C_\text{g}` on the interior
          islands. The ground capacitance of the two end islands belongs to
          whatever the ports connect to, so :math:`N = 1` is a single junction.
        - Nearest-neighbour coupling only: no long-range island-to-island
          capacitance.

    Default per-junction values are the fit to the 80-junction array of
    :cite:`maslukMicrowaveCharacterizationJosephson2012`
    (:math:`L_\text{J} = 1.9\,\text{nH}`, :math:`C_\text{J} = 40\,\text{fF}`,
    :math:`C_0 = 0.04\,\text{fF}`), and the default :math:`N = 43` is the
    fluxonium array of :cite:`manucharyan_fluxonium_2009`.

    Args:
        f: Array of frequency points in Hz.
        n_junctions: Number of junctions :math:`N` in the chain. Static under ``jit``.
        ic: Critical current :math:`I_\text{c}` of each junction in Amperes.
        capacitance: Capacitance :math:`C_\text{J}` of each junction in Farads.
        ground_capacitance: Capacitance :math:`C_\text{g}` of each interior
            island to ground in Farads.
        resistance: Shunt resistance :math:`R` of each junction in Ohms.
        ib: DC bias current :math:`I_\text{b}` through the chain in Amperes
            (:math:`|I_\text{b}| < I_\text{c}`).
        z0: Reference impedance :math:`Z_0` of both ports in Ohms.

    Returns:
        sax.SDict: S-parameters dictionary with ports ``o1`` and ``o2``.

    Raises:
        ValueError: If ``n_junctions`` is smaller than 1.
    """
    if n_junctions < 1:
        raise ValueError(f"n_junctions must be at least 1, got {n_junctions}")

    jax.debug.callback(_warn_if_overbiased, ib, ic)

    ω = 2 * jnp.pi * jnp.asarray(f)
    cos_Φ_0 = jnp.sqrt(jnp.clip(1.0 - (ib / ic) ** 2, min=1e-10))
    Lⱼ = Φ_0 / (2 * jnp.pi * ic * cos_Φ_0)
    Y_JJ = 1 / resistance + 1j * ω * capacitance + 1 / (1j * ω * Lⱼ)
    Z_JJ = 1 / Y_JJ
    Y_g = 1j * ω * ground_capacitance * jnp.ones_like(Z_JJ)

    one = jnp.ones_like(Z_JJ)
    zero = jnp.zeros_like(Z_JJ)
    t_junction = jnp.stack(
        [jnp.stack([one, Z_JJ], axis=-1), jnp.stack([zero, one], axis=-1)], axis=-2
    )
    t_cell = jnp.stack(
        [
            jnp.stack([one + Z_JJ * Y_g, Z_JJ], axis=-1),
            jnp.stack([Y_g, one], axis=-1),
        ],
        axis=-2,
    )
    abcd = _abcd_matmul(_abcd_power(t_cell, n_junctions - 1), t_junction)
    return _abcd_to_sdict(abcd, z0)


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
