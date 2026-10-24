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


_ABCD_POWER_BITS = 24
"""Bits of the chain length handled by :func:`_abcd_power` (:math:`N < 2^{24}`)."""


def _abcd_power(t: jax.Array, n: jax.Array) -> jax.Array:
    r"""Raise a batch of ABCD matrices to a traced integer power by squaring.

    Uses a fixed :math:`2 \times 24` matrix products, so ``n`` may be a traced
    array (as in a SAX circuit) without recompiling for every value.

    Returns:
        The batch of matrices :math:`T^n`.
    """
    result = jnp.broadcast_to(jnp.eye(2, dtype=t.dtype), t.shape)
    base = t
    for bit in range(_ABCD_POWER_BITS):
        take = ((n >> bit) & 1).astype(bool)
        result = jnp.where(take, _abcd_matmul(result, base), result)
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


@jit(inline=True)
def _josephson_junction_array(
    *,
    f: sax.FloatArrayLike,
    n: jax.Array,
    ic: sax.Float,
    capacitance: sax.Float,
    ground_capacitance: sax.Float,
    resistance: sax.Float,
    ib: sax.Float,
    z0: sax.Float,
) -> sax.SDict:
    """Jitted body of :func:`josephson_junction_array` with an integer ``n``.

    Returns:
        The S-parameter dictionary, NaN where ``n`` is out of range.
    """
    in_range = (n >= 1) & (n < 2**_ABCD_POWER_BITS)
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
    abcd = _abcd_matmul(_abcd_power(t_cell, n - 1), t_junction)
    abcd = jnp.where(in_range, abcd, jnp.nan)
    return _abcd_to_sdict(abcd, z0)


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
    :cite:`m.pozarMicrowaveEngineering2012`. The power is taken by repeated
    squaring, so the cost does not grow with :math:`N` and ``n_junctions`` may be
    traced (for example set from a SAX netlist).

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
    :math:`C_\text{g} = 0.04\,\text{fF}`). The default :math:`N = 43` matches the
    :func:`~qpdk.cells.josephson_junction_array` cell, whose default junction
    count and target :math:`E_\text{L}` follow :cite:`manucharyan_fluxonium_2009`.
    The default junctions of the two differ (:math:`43 \times 1.9\,\text{nH} = 82\,\text{nH}`
    here against the cell's estimate of :math:`314\,\text{nH}`): to simulate a laid
    out array, pass ``ic`` from the cell's ``info["critical_current_nA"]``
    (converted to A) together with a junction capacitance for that process.

    Args:
        f: Array of frequency points in Hz.
        n_junctions: Number of junctions :math:`N` in the chain,
            :math:`1 \le N < 2^{24}`. Non-integer values are truncated.
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
        ValueError: If a concrete ``n_junctions`` is outside
            :math:`1 \le N < 2^{24}`. A traced out-of-range value returns NaN.
    """
    n = jnp.asarray(n_junctions).astype(jnp.int32)
    if not isinstance(n, jax.core.Tracer) and not 1 <= int(n) < 2**_ABCD_POWER_BITS:
        raise ValueError(
            f"n_junctions must satisfy 1 <= n_junctions < 2**{_ABCD_POWER_BITS}, "
            f"got {n_junctions}"
        )
    return _josephson_junction_array(
        f=f,
        n=n,
        ic=ic,
        capacitance=capacitance,
        ground_capacitance=ground_capacitance,
        resistance=resistance,
        ib=ib,
        z0=z0,
    )


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
