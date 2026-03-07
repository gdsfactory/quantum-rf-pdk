r"""Coplanar waveguide (CPW) and microstrip electromagnetic analysis.

This module provides JAX-jittable functions for computing the characteristic
impedance, effective permittivity, and propagation constant of coplanar
waveguides and microstrip lines.  All results are obtained analytically so
the functions compose freely with JAX transformations (``jit``, ``grad``,
``vmap``, …).

CPW Theory
----------
The quasi-static CPW analysis follows the conformal-mapping approach
described by Simons :cite:`simonsCoplanarWaveguideCircuits2001` (ch. 2) and
Ghione & Naldi :cite:`ghioneAnalyticalFormulasCoplanar1984`.
Conductor thickness corrections use the first-order formulae of
Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
(§7.5, Eqs. 7.98-7.100).

Microstrip Theory
-----------------
The microstrip analysis uses the Hammerstad-Jensen
:cite:`hammerstadAccurateModelsComputer1980` closed-form expressions for
effective permittivity and characteristic impedance, as presented in
Pozar :cite:`m.pozarMicrowaveEngineering2012` (ch. 3, §3.8).

General
-------
The ABCD-to-S-parameter conversion is the standard microwave-network
relation from Pozar :cite:`m.pozarMicrowaveEngineering2012` (ch. 4).

The implementation was cross-checked against the Qucs-S model
(see `Qucs technical documentation`_, §12.4 for CPW, §12.1 for microstrip)
and the ``scikit-rf`` :class:`~skrf.media.cpw.CPW` class.

.. _Qucs technical documentation:
   https://qucs.sourceforge.net/docs/technical/technical.pdf

Functions
---------
All geometry parameters are in **SI base units** (metres, etc.) unless
noted otherwise.  Frequency is in **Hz**.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jaxellip
import scipy.constants
from jax.typing import ArrayLike

c_0: float = scipy.constants.speed_of_light
"""Speed of light in vacuum (m/s)."""

μ_0: float = scipy.constants.mu_0
"""Permeability of free space (H/m)."""

ε_0: float = scipy.constants.epsilon_0
"""Permittivity of free space (F/m)."""

Z_0_FREE: float = float(jnp.sqrt(μ_0 / ε_0))
r"""Impedance of free space :math:`\sqrt{\mu_0/\varepsilon_0}` (Ω)."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


@partial(jax.jit, inline=True)
def _ellipk_ratio(m: ArrayLike) -> jax.Array:
    r"""Ratio of complete elliptic integrals :math:`K(m)/K(1-m)`.

    Args:
        m: Elliptic-integral parameter (squared modulus :math:`k^2`).
    """
    return jaxellip.ellipk(m) / jaxellip.ellipk(1.0 - m)


# ===================================================================
# Coplanar Waveguide (CPW)
# ===================================================================


@partial(jax.jit, inline=True)
def cpw_epsilon_eff(
    w: ArrayLike,
    s: ArrayLike,
    h: ArrayLike,
    ep_r: ArrayLike,
) -> jax.Array:
    r"""Effective permittivity of a CPW on a finite-height substrate.

    Uses conformal mapping
    (Simons :cite:`simonsCoplanarWaveguideCircuits2001`, Eq. 2.2.20-2.2.22;
    Ghione & Naldi :cite:`ghioneAnalyticalFormulasCoplanar1984`):

    .. math::

        k_0     &= \frac{w}{w + 2s} \\
        k_1     &= \frac{\sinh(\pi w / 4h)}
                         {\sinh\bigl(\pi(w + 2s) / 4h\bigr)} \\
        q_1     &= \frac{K(k_1^2)\,/\,K(1 - k_1^2)}
                         {K(k_0^2)\,/\,K(1 - k_0^2)}  \\
        \varepsilon_{\mathrm{eff}}
                &= 1 + \frac{q_1\,(\varepsilon_r - 1)}{2}

    where :math:`K` is the complete elliptic integral of the first kind in
    the *parameter* convention (:math:`m = k^2`).

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        h: Substrate height (m).
        ep_r: Relative permittivity of the substrate.

    Returns:
        Effective permittivity (dimensionless).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    # Free-space modulus
    k0 = w / (w + 2.0 * s)

    # Substrate-corrected modulus
    k1 = jnp.sinh(jnp.pi * w / (4.0 * h)) / jnp.sinh(jnp.pi * (w + 2.0 * s) / (4.0 * h))

    # Filling factor q₁ = [K(k₁)/K(k₁')] / [K(k₀)/K(k₀')]
    q1 = _ellipk_ratio(k1**2) / _ellipk_ratio(k0**2)

    return 1.0 + q1 * (ep_r - 1.0) / 2.0


@partial(jax.jit, inline=True)
def cpw_z0(
    w: ArrayLike,
    s: ArrayLike,
    ep_eff: ArrayLike,
) -> jax.Array:
    r"""Characteristic impedance of a CPW.

    .. math::

        Z_0 = \frac{30\,\pi}
                    {\sqrt{\varepsilon_{\mathrm{eff}}}\;
                     K(k_0^2)\,/\,K(1 - k_0^2)}

    (Simons :cite:`simonsCoplanarWaveguideCircuits2001`, Eq. 2.2.14.)

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        ep_eff: Effective permittivity (see :func:`cpw_epsilon_eff`).

    Returns:
        Characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    k0 = w / (w + 2.0 * s)
    return 30.0 * jnp.pi / (jnp.sqrt(ep_eff) * _ellipk_ratio(k0**2))


@partial(jax.jit, inline=True)
def cpw_thickness_correction(
    w: ArrayLike,
    s: ArrayLike,
    t: ArrayLike,
    ep_eff: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Apply conductor thickness correction to CPW ε_eff and Z₀.

    First-order correction from
    Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
    (§7.5, Eqs. 7.98-7.100):

    .. math::

        \Delta &= \frac{1.25\,t}{\pi}
                  \left(1 + \ln\frac{4\pi w}{t}\right) \\
        k_e    &= k_0 + (1 - k_0^2)\,\frac{\Delta}{2s} \\
        \varepsilon_{\mathrm{eff},t}
               &= \varepsilon_{\mathrm{eff}}
                  - \frac{0.7\,(\varepsilon_{\mathrm{eff}} - 1)\,t/s}
                         {K(k_0^2)/K(1-k_0^2) + 0.7\,t/s} \\
        Z_{0,t} &= \frac{30\pi}
                        {\sqrt{\varepsilon_{\mathrm{eff},t}}\;
                         K(k_e^2)/K(1-k_e^2)}

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        t: Conductor thickness (m).
        ep_eff: Uncorrected effective permittivity.

    Returns:
        ``(ep_eff_t, z0_t)`` — thickness-corrected effective permittivity
        and characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    t = jnp.asarray(t, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    k0 = w / (w + 2.0 * s)
    q0 = _ellipk_ratio(k0**2)

    # Effective width increase (GGBB96 Eq. 7.98)
    delta = (1.25 * t / jnp.pi) * (1.0 + jnp.log(4.0 * jnp.pi * w / t))

    # Modified modulus for Z₀ (GGBB96, between 7.99 and 7.100)
    ke = k0 + (1.0 - k0**2) * delta / (2.0 * s)
    ke = jnp.clip(ke, 1e-12, 1.0 - 1e-12)

    # Modified ε_eff (GGBB96 Eq. 7.100)
    ep_eff_t = ep_eff - (0.7 * (ep_eff - 1.0) * t / s) / (q0 + 0.7 * t / s)

    # Modified Z₀
    z0_t = 30.0 * jnp.pi / (jnp.sqrt(ep_eff_t) * _ellipk_ratio(ke**2))

    return ep_eff_t, z0_t


# ===================================================================
# Microstrip
# ===================================================================


@partial(jax.jit, inline=True)
def microstrip_epsilon_eff(
    w: ArrayLike,
    h: ArrayLike,
    ep_r: ArrayLike,
) -> jax.Array:
    r"""Effective permittivity of a microstrip line.

    Uses the Hammerstad-Jensen
    :cite:`hammerstadAccurateModelsComputer1980` formula as given in
    Pozar :cite:`m.pozarMicrowaveEngineering2012` (Eq. 3.195-3.196):

    .. math::

        \varepsilon_{\mathrm{eff}} = \frac{\varepsilon_r + 1}{2}
            + \frac{\varepsilon_r - 1}{2}
              \left(\frac{1}{\sqrt{1 + 12\,h/w}}
                    + 0.04\,(1 - w/h)^2\;\Theta(1 - w/h)\right)

    where the last term contributes only for narrow strips (:math:`w/h < 1`).

    Args:
        w: Strip width (m).
        h: Substrate height (m).
        ep_r: Relative permittivity of the substrate.

    Returns:
        Effective permittivity (dimensionless).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    u = w / h
    f_u = 1.0 / jnp.sqrt(1.0 + 12.0 / u)

    # Extra correction for narrow strips (w/h < 1)
    narrow_correction = 0.04 * (1.0 - u) ** 2
    f_u = jnp.where(u < 1.0, f_u + narrow_correction, f_u)

    return (ep_r + 1.0) / 2.0 + (ep_r - 1.0) / 2.0 * f_u


@partial(jax.jit, inline=True)
def microstrip_z0(
    w: ArrayLike,
    h: ArrayLike,
    ep_eff: ArrayLike,
) -> jax.Array:
    r"""Characteristic impedance of a microstrip line.

    Uses the Hammerstad-Jensen
    :cite:`hammerstadAccurateModelsComputer1980` approximation as given in
    Pozar :cite:`m.pozarMicrowaveEngineering2012` (Eq. 3.197-3.198):

    .. math::

        Z_0 = \begin{cases}
            \displaystyle\frac{60}{\sqrt{\varepsilon_{\mathrm{eff}}}}
            \ln\!\left(\frac{8h}{w} + \frac{w}{4h}\right)
            & w/h \le 1 \\[6pt]
            \displaystyle\frac{120\pi}
            {\sqrt{\varepsilon_{\mathrm{eff}}}\,
             \bigl[w/h + 1.393 + 0.667\ln(w/h + 1.444)\bigr]}
            & w/h \ge 1
        \end{cases}

    Args:
        w: Strip width (m).
        h: Substrate height (m).
        ep_eff: Effective permittivity (see :func:`microstrip_epsilon_eff`).

    Returns:
        Characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    u = w / h

    # Narrow strip (w/h <= 1)
    z_narrow = (60.0 / jnp.sqrt(ep_eff)) * jnp.log(8.0 / u + u / 4.0)

    # Wide strip (w/h >= 1)
    z_wide = (
        120.0 * jnp.pi / (jnp.sqrt(ep_eff) * (u + 1.393 + 0.667 * jnp.log(u + 1.444)))
    )

    return jnp.where(u <= 1.0, z_narrow, z_wide)


@partial(jax.jit, inline=True)
def microstrip_thickness_correction(
    w: ArrayLike,
    h: ArrayLike,
    t: ArrayLike,
    ep_r: ArrayLike,
    ep_eff: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Conductor thickness correction for a microstrip line.

    Uses the widely-adopted Schneider correction as presented in
    Pozar :cite:`m.pozarMicrowaveEngineering2012` (§3.8) and
    Gupta et al. :cite:`guptaMicrostripLinesSlotlines1996` (§2.2.4):

    .. math::

        w_e &= w + \frac{t}{\pi}
               \ln\frac{4e}{\sqrt{(t/h)^2 + (t/(w\pi + 1.1t\pi))^2}} \\
        \varepsilon_{\mathrm{eff},t}
            &= \varepsilon_{\mathrm{eff}}
               - \frac{(\varepsilon_r - 1)\,t/h}
                      {4.6\,\sqrt{w/h}}

    Then the corrected :math:`Z_0` is computed with the effective width
    :math:`w_e` and corrected :math:`\varepsilon_{\mathrm{eff},t}`.

    Args:
        w: Strip width (m).
        h: Substrate height (m).
        t: Conductor thickness (m).
        ep_r: Relative permittivity of the substrate.
        ep_eff: Uncorrected effective permittivity.

    Returns:
        ``(w_eff, ep_eff_t, z0_t)`` — effective width (m),
        thickness-corrected effective permittivity,
        and characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    h = jnp.asarray(h, dtype=float)
    t = jnp.asarray(t, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)

    e = jnp.e  # Euler's number

    # Effective width (Schneider)
    term = jnp.sqrt((t / h) ** 2 + (t / (w * jnp.pi + 1.1 * t * jnp.pi)) ** 2)
    w_eff = w + (t / jnp.pi) * jnp.log(4.0 * e / term)

    # Corrected epsilon_eff
    ep_eff_t = ep_eff - (ep_r - 1.0) * t / h / (4.6 * jnp.sqrt(w / h))

    # Corrected Z0
    z0_t = microstrip_z0(w_eff, h, ep_eff_t)

    return w_eff, ep_eff_t, z0_t


# ===================================================================
# Common: propagation & S-parameters
# ===================================================================


@partial(jax.jit, inline=True)
def propagation_constant(
    f: ArrayLike,
    ep_eff: ArrayLike,
    tand: ArrayLike = 0.0,
    ep_r: ArrayLike = 1.0,
) -> jax.Array:
    r"""Complex propagation constant of a quasi-TEM transmission line.

    For the general lossy case
    (Simons :cite:`simonsCoplanarWaveguideCircuits2001`, §2.2.4):

    .. math::

        \gamma = \alpha_d + j\,\beta

    where the **dielectric attenuation** is

    .. math::

        \alpha_d = \frac{\pi f}{c_0}
                   \frac{\varepsilon_r}{\sqrt{\varepsilon_{\mathrm{eff}}}}
                   \frac{\varepsilon_{\mathrm{eff}} - 1}
                        {\varepsilon_r - 1}
                   \tan\delta

    and the **phase constant** is

    .. math::

        \beta = \frac{2\pi f}{c_0}\,\sqrt{\varepsilon_{\mathrm{eff}}}

    For a superconducting line (:math:`\tan\delta = 0`) the propagation
    is purely imaginary: :math:`\gamma = j\beta`.

    Args:
        f: Frequency (Hz).
        ep_eff: Effective permittivity.
        tand: Dielectric loss tangent (default 0 — lossless).
        ep_r: Substrate relative permittivity (only needed when ``tand > 0``).

    Returns:
        Complex propagation constant :math:`\gamma` (1/m).
    """
    f = jnp.asarray(f, dtype=float)
    ep_eff = jnp.asarray(ep_eff, dtype=float)
    tand = jnp.asarray(tand, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    beta = 2.0 * jnp.pi * f * jnp.sqrt(ep_eff) / c_0

    # Dielectric attenuation constant (Simons Eq. 2.2.41)
    alpha_d = (
        jnp.pi
        * f
        / c_0
        * (ep_r / jnp.sqrt(ep_eff))
        * ((ep_eff - 1.0) / (ep_r - 1.0))
        * tand
    )
    # Guard against ep_r == 1 (vacuum) where division would be 0/0.
    # When ep_r == 1 there is no substrate, so alpha_d = 0 by definition.
    alpha_d = jnp.where(jnp.abs(ep_r - 1.0) < 1e-15, 0.0, alpha_d)

    return alpha_d + 1j * beta


# Keep backward-compatible alias
cpw_gamma = propagation_constant


@partial(jax.jit, inline=True)
def transmission_line_s_params(
    gamma: ArrayLike,
    z0: ArrayLike,
    length: ArrayLike,
    z_ref: ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""S-parameters of a uniform transmission line (ABCD→S conversion).

    The ABCD matrix of a line with characteristic impedance :math:`Z_0`,
    propagation constant :math:`\gamma`, and length :math:`\ell` is

    .. math::

        \begin{pmatrix} A & B \\ C & D \end{pmatrix}
        = \begin{pmatrix}
            \cosh\theta & Z_0\sinh\theta \\
            \sinh\theta / Z_0 & \cosh\theta
          \end{pmatrix}, \quad \theta = \gamma\,\ell.

    Converting to S-parameters referenced to :math:`Z_{\mathrm{ref}}`
    (Pozar :cite:`m.pozarMicrowaveEngineering2012`, Table 4.2):

    .. math::

        S_{11} &= \frac{A + B/Z_{\mathrm{ref}} - C\,Z_{\mathrm{ref}} - D}
                       {A + B/Z_{\mathrm{ref}} + C\,Z_{\mathrm{ref}} + D} \\
        S_{21} &= \frac{2}
                       {A + B/Z_{\mathrm{ref}} + C\,Z_{\mathrm{ref}} + D}

    When ``z_ref`` is ``None`` the reference impedance defaults to ``z0``
    (matched case), giving :math:`S_{11} = 0` and
    :math:`S_{21} = e^{-\gamma\ell}`.

    Args:
        gamma: Complex propagation constant (1/m).
        z0: Characteristic impedance (Ω).
        length: Physical length (m).
        z_ref: Reference (port) impedance (Ω).  Defaults to ``z0``.

    Returns:
        ``(S11, S21)`` — complex S-parameter arrays.
    """
    gamma = jnp.asarray(gamma, dtype=complex)
    z0 = jnp.asarray(z0, dtype=complex)
    length = jnp.asarray(length, dtype=float)

    if z_ref is None:
        z_ref = z0
    z_ref = jnp.asarray(z_ref, dtype=complex)

    theta = gamma * length

    cosh_t = jnp.cosh(theta)
    sinh_t = jnp.sinh(theta)

    # ABCD elements (symmetric line: A = D)
    a = cosh_t
    b = z0 * sinh_t
    c = sinh_t / z0

    denom = a + b / z_ref + c * z_ref + a  # A + B/Zr + C·Zr + D with D = A
    s11 = (b / z_ref - c * z_ref) / denom  # (A-D) = 0 for symmetric line
    s21 = 2.0 / denom

    return s11, s21
