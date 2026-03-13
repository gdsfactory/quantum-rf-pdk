r"""Perturbation theory helpers for superconducting quantum circuits.

This module provides helper functions for computing effective Hamiltonians
and energy corrections using quasi-degenerate perturbation theory via
`pymablock <https://pymablock.readthedocs.io/>`_
:cite:`arayaDayPymablockAlgorithmPackage2025`.

The primary application is computing the **dispersive shift** :math:`\chi`
of a resonator coupled to a transmon qubit, which is the key quantity for
dispersive readout of superconducting qubits
:cite:`blaisCircuitQuantumElectrodynamics2021,kochChargeinsensitiveQubitDesign2007a`.

The transmon-resonator Hamiltonian (without the rotating wave approximation)
is:

.. math::

    \mathcal{H} = -\omega_t\, a_t^\dagger a_t
    + \frac{\alpha}{2}\, a_t^{\dagger 2} a_t^2
    + \omega_r\, a_r^\dagger a_r
    - g\,(a_t^\dagger - a_t)(a_r^\dagger - a_r)

where :math:`\omega_t` is the transmon frequency, :math:`\omega_r` the resonator
frequency, :math:`\alpha` the anharmonicity, and :math:`g` the coupling strength.

References:
    - :cite:`arayaDayPymablockAlgorithmPackage2025`
    - :cite:`blaisCircuitQuantumElectrodynamics2021`
    - :cite:`kochChargeinsensitiveQubitDesign2007a`
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import sympy
from jax.typing import ArrayLike
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp


def transmon_resonator_hamiltonian() -> tuple[
    sympy.Expr, sympy.Expr, tuple[sympy.Symbol, ...]
]:
    r"""Build the symbolic transmon-resonator Hamiltonian.

    Constructs the unperturbed (:math:`H_0`) and perturbation (:math:`H_p`)
    parts of the transmon-resonator Hamiltonian using bosonic operators.

    The Hamiltonian is split as :math:`H = H_0 + H_p` where:

    .. math::

        \begin{aligned}
        H_0 &= -\omega_t\, a_t^\dagger a_t
              + \frac{\alpha}{2}\, a_t^{\dagger 2} a_t^2
              + \omega_r\, a_r^\dagger a_r \\
        H_p &= -g\,(a_t^\dagger - a_t)(a_r^\dagger - a_r)
        \end{aligned}

    Returns:
        A tuple ``(H_0, H_p, symbols)`` where ``symbols`` is
        ``(omega_t, omega_r, alpha, g)``.

    Example:
        >>> H_0, H_p, (omega_t, omega_r, alpha, g) = transmon_resonator_hamiltonian()
    """
    omega_t, omega_r, alpha, g = sympy.symbols(
        r"\omega_{t} \omega_{r} \alpha g", real=True, positive=True
    )

    a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

    H_0 = (
        -omega_t * Dagger(a_t) * a_t
        + omega_r * Dagger(a_r) * a_r
        + alpha * Dagger(a_t) ** 2 * a_t**2 / 2
    )

    H_p = -g * (Dagger(a_t) - a_t) * (Dagger(a_r) - a_r)

    return H_0, H_p, (omega_t, omega_r, alpha, g)


@partial(jax.jit, inline=True)
def dispersive_shift(
    ω_t_ghz: float | ArrayLike,
    ω_r_ghz: float | ArrayLike,
    α_ghz: float | ArrayLike,
    g_ghz: float | ArrayLike,
) -> float | jax.Array:
    r"""Compute the dispersive shift numerically.

    Evaluates the second-order dispersive shift for a transmon coupled
    to a resonator. Uses the analytical formula derived from perturbation
    theory (without the rotating wave approximation):

    .. math::

        \chi = \frac{2g^2}{\Delta - \alpha}
             - \frac{2g^2}{\Delta}
             - \frac{2g^2}{\omega_t + \omega_r + \alpha}
             + \frac{2g^2}{\omega_t + \omega_r}

    where :math:`\Delta = \omega_t - \omega_r`.  The first two terms give
    the rotating-wave-approximation (RWA) contribution

    .. math::

        \chi_\text{RWA} = \frac{2 \alpha g^2}{\Delta(\Delta - \alpha)}

    and the last two are corrections from the counter-rotating terms.

    All parameters are in GHz, and the returned value is also in GHz.

    Args:
        ω_t_ghz: Transmon frequency in GHz.
        ω_r_ghz: Resonator frequency in GHz.
        α_ghz: Transmon anharmonicity in GHz (positive value).
        g_ghz: Coupling strength in GHz.

    Returns:
        Dispersive shift :math:`\chi` in GHz.

    Example:
        >>> χ = dispersive_shift(5.0, 7.0, 0.2, 0.1)
        >>> print(f"χ = {χ * 1e3:.2f} MHz")
    """
    Δ = ω_t_ghz - ω_r_ghz

    # Full expression including counter-rotating terms
    return (
        2 * g_ghz**2 / (Δ - α_ghz)
        - 2 * g_ghz**2 / Δ
        - 2 * g_ghz**2 / (ω_t_ghz + ω_r_ghz + α_ghz)
        + 2 * g_ghz**2 / (ω_t_ghz + ω_r_ghz)
    )


@partial(jax.jit, inline=True)
def dispersive_shift_to_coupling(
    χ_ghz: float | ArrayLike,
    ω_t_ghz: float | ArrayLike,
    ω_r_ghz: float | ArrayLike,
    α_ghz: float | ArrayLike,
) -> float | jax.Array:
    r"""Compute the coupling strength from a target dispersive shift.

    Inverts the dispersive shift relation to find the coupling strength
    :math:`g` required to achieve a desired :math:`\chi`.  Uses only the
    dominant rotating-wave term:

    .. math::

        g \approx \sqrt{\frac{-\chi\,\Delta\,(\Delta - \alpha)}{2\alpha}}

    where :math:`\Delta = \omega_t - \omega_r`.

    Note:
        The expression under the square root may be negative when the
        sign of the target :math:`\chi` is inconsistent with the detuning
        and anharmonicity (e.g., positive :math:`\chi` with
        :math:`\Delta < 0`).  In that case the absolute value is taken
        so that the returned coupling strength is always real and
        non-negative, but the caller should verify self-consistency
        via :func:`dispersive_shift`.

    Args:
        χ_ghz: Target dispersive shift in GHz (typically negative).
        ω_t_ghz: Transmon frequency in GHz.
        ω_r_ghz: Resonator frequency in GHz.
        α_ghz: Transmon anharmonicity in GHz (positive value;
            the physical anharmonicity of a transmon is negative, but
            following the Hamiltonian convention used throughout this
            module, :math:`\alpha` is taken as positive).

    Returns:
        Coupling strength :math:`g` in GHz.

    Example:
        >>> g = dispersive_shift_to_coupling(-0.001, 5.0, 7.0, 0.2)
        >>> print(f"g = {g * 1e3:.1f} MHz")
    """
    Δ = ω_t_ghz - ω_r_ghz
    g_squared = -χ_ghz * Δ * (Δ - α_ghz) / (2 * α_ghz)
    return jnp.sqrt(jnp.abs(g_squared))


@partial(jax.jit, inline=True)
def ej_ec_to_frequency_and_anharmonicity(
    ej_ghz: float | ArrayLike, ec_ghz: float | ArrayLike
) -> tuple[float | jax.Array, float | jax.Array]:
    r"""Convert :math:`E_J` and :math:`E_C` to qubit frequency and anharmonicity.

    Uses the standard transmon approximations:

    .. math::

        \begin{aligned}
        \omega_q &\approx \sqrt{8 E_J E_C} - E_C \\
        \alpha &\approx E_C
        \end{aligned}

    Note:
        The physical anharmonicity of a transmon is *negative*
        (:math:`\alpha = -E_C`), but the Hamiltonian convention used
        in this module and in pymablock takes :math:`\alpha` as positive.

    Args:
        ej_ghz: Josephson energy in GHz.
        ec_ghz: Charging energy in GHz.

    Returns:
        Tuple of ``(ω_q_ghz, α_ghz)``.

    Example:
        >>> ω_q, α = ej_ec_to_frequency_and_anharmonicity(20.0, 0.2)
        >>> print(f"ω_q = {ω_q:.2f} GHz, α = {α:.1f} GHz")
    """
    return jnp.sqrt(8 * ej_ghz * ec_ghz) - ec_ghz, ec_ghz


@partial(jax.jit, inline=True)
def purcell_decay_rate(
    g_ghz: float | ArrayLike,
    ω_t_ghz: float | ArrayLike,
    ω_r_ghz: float | ArrayLike,
    κ_ghz: float | ArrayLike,
) -> float | jax.Array:
    r"""Estimate the Purcell decay rate of a transmon through a resonator.

    The Purcell effect limits qubit lifetime when coupled to a lossy
    resonator.  In the dispersive regime:

    .. math::

        \gamma_\text{Purcell} = \kappa \left(\frac{g}{\Delta}\right)^2

    where :math:`\kappa` is the resonator decay rate and
    :math:`\Delta = \omega_t - \omega_r`.

    Args:
        g_ghz: Coupling strength in GHz.
        ω_t_ghz: Transmon frequency in GHz.
        ω_r_ghz: Resonator frequency in GHz.
        κ_ghz: Resonator linewidth (decay rate) in GHz.

    Returns:
        Purcell decay rate in GHz.

    Example:
        >>> γ = purcell_decay_rate(0.1, 5.0, 7.0, 0.001)
        >>> T_purcell = 1 / (γ * 1e9)
        >>> print(f"T_Purcell = {T_purcell * 1e6:.0f} µs")
    """
    Δ = ω_t_ghz - ω_r_ghz
    return κ_ghz * (g_ghz / Δ) ** 2


@partial(jax.jit, inline=True)
def resonator_linewidth_from_q(
    ω_r_ghz: float | ArrayLike, q_ext: float | ArrayLike
) -> float | jax.Array:
    r"""Compute resonator linewidth from external quality factor.

    .. math::

        \kappa = \frac{\omega_r}{Q_\text{ext}}

    Args:
        ω_r_ghz: Resonator frequency in GHz.
        q_ext: External quality factor.

    Returns:
        Resonator linewidth :math:`\kappa` in GHz.

    Example:
        >>> κ = resonator_linewidth_from_q(7.0, 10_000)
        >>> print(f"κ = {κ * 1e6:.1f} kHz")
    """
    return ω_r_ghz / q_ext


@partial(jax.jit, inline=True)
def measurement_induced_dephasing(
    χ_ghz: float | ArrayLike, κ_ghz: float | ArrayLike, n_bar: float | ArrayLike
) -> float | jax.Array:
    r"""Estimate measurement-induced dephasing rate.

    During dispersive readout, photons in the resonator cause
    additional dephasing of the qubit:

    .. math::

        \Gamma_\phi = \frac{8 \chi^2 \bar{n}}{\kappa}

    where :math:`\bar{n}` is the mean photon number in the resonator.

    Args:
        χ_ghz: Dispersive shift in GHz.
        κ_ghz: Resonator linewidth in GHz.
        n_bar: Mean photon number in the resonator during measurement.

    Returns:
        Measurement-induced dephasing rate in GHz.

    Example:
        >>> Γ_φ = measurement_induced_dephasing(-0.001, 0.001, 5.0)
        >>> print(f"Γ_φ = {Γ_φ * 1e6:.1f} kHz")
    """
    return 8 * χ_ghz**2 * n_bar / κ_ghz
