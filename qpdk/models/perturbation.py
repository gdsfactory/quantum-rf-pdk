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

import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

pytestmark = pytest.mark.hamiltonian


def transmon_resonator_hamiltonian() -> tuple[
    sympy.Expr, sympy.Expr, tuple[sympy.Symbol, ...]
]:
    r"""Build the symbolic transmon-resonator Hamiltonian.

    Constructs the unperturbed (:math:`H_0`) and perturbation (:math:`H_p`)
    parts of the transmon-resonator Hamiltonian using bosonic operators.

    The Hamiltonian is split as :math:`H = H_0 + H_p` where:

    .. math::

        H_0 &= -\omega_t\, a_t^\dagger a_t
              + \frac{\alpha}{2}\, a_t^{\dagger 2} a_t^2
              + \omega_r\, a_r^\dagger a_r \\
        H_p &= -g\,(a_t^\dagger - a_t)(a_r^\dagger - a_r)

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


def dispersive_shift_symbolic() -> tuple[sympy.Expr, tuple[sympy.Symbol, ...]]:
    r"""Compute the symbolic dispersive shift using pymablock.

    Uses quasi-degenerate perturbation theory (via pymablock) to compute
    the second-order dispersive shift :math:`\chi` of a resonator coupled
    to a transmon qubit:

    .. math::

        \chi = \frac{E^{(2)}_{11} - E^{(2)}_{10}}{2}
             - \frac{E^{(2)}_{01} - E^{(2)}_{00}}{2}

    where :math:`E^{(2)}_{ij}` is the second-order energy correction for
    the state with :math:`i` transmon excitations and :math:`j` resonator
    excitations.

    This function uses the second-quantized approach, which yields
    the result in terms of number operators that are then evaluated
    for specific occupation numbers.

    Returns:
        A tuple ``(chi, symbols)`` where ``chi`` is the symbolic expression
        for the dispersive shift and ``symbols`` is
        ``(omega_t, omega_r, alpha, g)``.

    Example:
        >>> chi_sym, (omega_t, omega_r, alpha, g) = dispersive_shift_symbolic()
        >>> # Evaluate numerically
        >>> chi_val = float(chi_sym.subs({omega_t: 5.0, omega_r: 7.0, alpha: 0.2, g: 0.1}))
    """
    from pymablock import block_diagonalize
    from pymablock.number_ordered_form import NumberOperator

    H_0, H_p, syms = transmon_resonator_hamiltonian()
    *_, g = syms

    a_t = BosonOp("a_t")
    a_r = BosonOp("a_r")

    H_tilde, _U, _U_adj = block_diagonalize(H_0 + H_p, symbols=[g])

    # Second-order effective energy
    E_eff = H_tilde[0, 0, 2]

    # Substitute occupation numbers to compute dispersive shift
    N_a_t = NumberOperator(a_t)
    N_a_r = NumberOperator(a_r)

    E_00 = E_eff.subs({N_a_t: 0, N_a_r: 0})
    E_01 = E_eff.subs({N_a_t: 0, N_a_r: 1})
    E_10 = E_eff.subs({N_a_t: 1, N_a_r: 0})
    E_11 = E_eff.subs({N_a_t: 1, N_a_r: 1})

    chi = E_11 - E_10 - E_01 + E_00

    # Convert from NumberOrderedForm to a standard sympy expression
    if hasattr(chi, "as_expr"):
        chi = chi.as_expr()

    return chi, syms


def dispersive_shift(
    omega_t_ghz: float,
    omega_r_ghz: float,
    alpha_ghz: float,
    g_ghz: float,
) -> float:
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
        omega_t_ghz: Transmon frequency in GHz.
        omega_r_ghz: Resonator frequency in GHz.
        alpha_ghz: Transmon anharmonicity in GHz (positive value).
        g_ghz: Coupling strength in GHz.

    Returns:
        Dispersive shift :math:`\chi` in GHz.

    Example:
        >>> chi = dispersive_shift(5.0, 7.0, 0.2, 0.1)
        >>> print(f"χ = {chi * 1e3:.2f} MHz")
    """
    delta = omega_t_ghz - omega_r_ghz

    # Full expression including counter-rotating terms
    return (
        2 * g_ghz**2 / (delta - alpha_ghz)
        - 2 * g_ghz**2 / delta
        - 2 * g_ghz**2 / (omega_t_ghz + omega_r_ghz + alpha_ghz)
        + 2 * g_ghz**2 / (omega_t_ghz + omega_r_ghz)
    )


def dispersive_shift_to_coupling(
    chi_ghz: float,
    omega_t_ghz: float,
    omega_r_ghz: float,
    alpha_ghz: float,
) -> float:
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
        chi_ghz: Target dispersive shift in GHz (typically negative).
        omega_t_ghz: Transmon frequency in GHz.
        omega_r_ghz: Resonator frequency in GHz.
        alpha_ghz: Transmon anharmonicity in GHz (positive value;
            the physical anharmonicity of a transmon is negative, but
            following the Hamiltonian convention used throughout this
            module, :math:`\alpha` is taken as positive).

    Returns:
        Coupling strength :math:`g` in GHz.

    Example:
        >>> g = dispersive_shift_to_coupling(-0.001, 5.0, 7.0, 0.2)
        >>> print(f"g = {g * 1e3:.1f} MHz")
    """
    import math

    delta = omega_t_ghz - omega_r_ghz
    g_squared = -chi_ghz * delta * (delta - alpha_ghz) / (2 * alpha_ghz)
    return math.sqrt(abs(g_squared))


def chi_to_readout_frequency_shift(chi_ghz: float) -> float:
    r"""Convert dispersive shift to readout frequency shift in Hz.

    The readout resonator frequency shifts by :math:`\pm\chi` depending on
    the qubit state.  The total frequency difference between qubit states
    :math:`|0\rangle` and :math:`|1\rangle` is :math:`2\chi`.

    Args:
        chi_ghz: Dispersive shift in GHz.

    Returns:
        Readout frequency shift :math:`2\chi` in Hz.

    Example:
        >>> shift_hz = chi_to_readout_frequency_shift(-0.001)
        >>> print(f"Readout shift = {shift_hz / 1e6:.2f} MHz")
    """
    return 2 * chi_ghz * 1e9


def qubit_frequency_from_ej_ec(ej_ghz: float, ec_ghz: float) -> float:
    r"""Estimate the transmon qubit frequency from :math:`E_J` and :math:`E_C`.

    Uses the standard transmon approximation:

    .. math::

        \omega_q \approx \sqrt{8 E_J E_C} - E_C

    Args:
        ej_ghz: Josephson energy in GHz.
        ec_ghz: Charging energy in GHz.

    Returns:
        Qubit frequency in GHz.

    Example:
        >>> f_q = qubit_frequency_from_ej_ec(20.0, 0.2)
        >>> print(f"f_q = {f_q:.2f} GHz")
    """
    import math

    return math.sqrt(8 * ej_ghz * ec_ghz) - ec_ghz


def anharmonicity_from_ec(ec_ghz: float) -> float:
    r"""Estimate the transmon anharmonicity from :math:`E_C`.

    In the transmon regime (:math:`E_J \gg E_C`), the anharmonicity is
    approximately equal to the charging energy:

    .. math::

        \alpha \approx E_C

    Note:
        The physical anharmonicity of a transmon is *negative*
        (:math:`\alpha = -E_C`), but the Hamiltonian convention used
        in this module and in pymablock takes :math:`\alpha` as positive.

    Args:
        ec_ghz: Charging energy in GHz.

    Returns:
        Anharmonicity in GHz (positive by convention).

    Example:
        >>> alpha = anharmonicity_from_ec(0.2)
        >>> print(f"alpha = {alpha:.1f} GHz")  # 0.2 GHz = 200 MHz
    """
    return ec_ghz


def ej_ec_to_frequency_and_anharmonicity(
    ej_ghz: float, ec_ghz: float
) -> tuple[float, float]:
    r"""Convert :math:`E_J` and :math:`E_C` to qubit frequency and anharmonicity.

    Convenience wrapper combining :func:`qubit_frequency_from_ej_ec` and
    :func:`anharmonicity_from_ec`.

    Args:
        ej_ghz: Josephson energy in GHz.
        ec_ghz: Charging energy in GHz.

    Returns:
        Tuple of ``(omega_q_ghz, alpha_ghz)``.

    Example:
        >>> omega_q, alpha = ej_ec_to_frequency_and_anharmonicity(20.0, 0.2)
        >>> print(f"omega_q = {omega_q:.2f} GHz, alpha = {alpha:.1f} GHz")
    """
    return qubit_frequency_from_ej_ec(ej_ghz, ec_ghz), anharmonicity_from_ec(ec_ghz)


def purcell_decay_rate(
    g_ghz: float, omega_t_ghz: float, omega_r_ghz: float, kappa_ghz: float
) -> float:
    r"""Estimate the Purcell decay rate of a transmon through a resonator.

    The Purcell effect limits qubit lifetime when coupled to a lossy
    resonator.  In the dispersive regime:

    .. math::

        \gamma_\text{Purcell} = \kappa \left(\frac{g}{\Delta}\right)^2

    where :math:`\kappa` is the resonator decay rate and
    :math:`\Delta = \omega_t - \omega_r`.

    Args:
        g_ghz: Coupling strength in GHz.
        omega_t_ghz: Transmon frequency in GHz.
        omega_r_ghz: Resonator frequency in GHz.
        kappa_ghz: Resonator linewidth (decay rate) in GHz.

    Returns:
        Purcell decay rate in GHz.

    Example:
        >>> gamma = purcell_decay_rate(0.1, 5.0, 7.0, 0.001)
        >>> T_purcell = 1 / (gamma * 1e9)
        >>> print(f"T_Purcell = {T_purcell * 1e6:.0f} µs")
    """
    delta = omega_t_ghz - omega_r_ghz
    return kappa_ghz * (g_ghz / delta) ** 2


def resonator_linewidth_from_q(omega_r_ghz: float, q_ext: float) -> float:
    r"""Compute resonator linewidth from external quality factor.

    .. math::

        \kappa = \frac{\omega_r}{Q_\text{ext}}

    Args:
        omega_r_ghz: Resonator frequency in GHz.
        q_ext: External quality factor.

    Returns:
        Resonator linewidth :math:`\kappa` in GHz.

    Example:
        >>> kappa = resonator_linewidth_from_q(7.0, 10_000)
        >>> print(f"κ = {kappa * 1e6:.1f} kHz")
    """
    return omega_r_ghz / q_ext


def measurement_induced_dephasing(
    chi_ghz: float, kappa_ghz: float, n_bar: float
) -> float:
    r"""Estimate measurement-induced dephasing rate.

    During dispersive readout, photons in the resonator cause
    additional dephasing of the qubit:

    .. math::

        \Gamma_\phi = \frac{8 \chi^2 \bar{n}}{\kappa}

    where :math:`\bar{n}` is the mean photon number in the resonator.

    Args:
        chi_ghz: Dispersive shift in GHz.
        kappa_ghz: Resonator linewidth in GHz.
        n_bar: Mean photon number in the resonator during measurement.

    Returns:
        Measurement-induced dephasing rate in GHz.

    Example:
        >>> gamma_phi = measurement_induced_dephasing(-0.001, 0.001, 5.0)
        >>> print(f"Γ_φ = {gamma_phi * 1e6:.1f} kHz")
    """
    return 8 * chi_ghz**2 * n_bar / kappa_ghz
