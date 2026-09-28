r"""Coplanar waveguide (CPW) and microstrip electromagnetic analysis.

This module provides JAX-jittable functions for computing the characteristic
impedance, effective permittivity, and propagation constant of coplanar
waveguides and microstrip lines.  All results are obtained analytically so
the functions compose freely with JAX transformations (``jit``, ``grad``,
``vmap``, …).

The electromagnetic core functions are provided by :mod:`sax.models.rf` and
re-exported here for convenience.  This module adds layout-to-model helpers
that extract physical dimensions from the qpdk layer stack and cross-section
specifications.

CPW Theory
----------
The quasi-static CPW analysis follows the conformal-mapping approach
described by Simons :cite:`simonsCoplanarWaveguideCircuits2001` (ch. 2) and
Ghione & Naldi :cite:`ghioneAnalyticalFormulasCoplanar1984`.
Conductor thickness corrections use the first-order formulae of
Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
(§7.3, Eqs. 7.98-7.100).

Conductor-backed CPW
--------------------
With metal on the backside of the substrate (see
:data:`~qpdk.tech.LAYER_STACK_BACKSIDE`), the CPW becomes a conductor-backed
CPW. :func:`cbcpw_parameters` follows Simons
:cite:`simonsCoplanarWaveguideCircuits2001` (ch. 3); pass
``conductor_backed=True`` to :func:`cpw_parameters` and the functions built on
it to use it.

Microstrip Theory
-----------------
The microstrip analysis uses the Hammerstad-Jensen
:cite:`hammerstadAccurateModelsMicrostrip1980` closed-form expressions for
effective permittivity and characteristic impedance, as presented in
Pozar :cite:`m.pozarMicrowaveEngineering2012` (ch. 3, §3.8).

General
-------
The ABCD-to-S-parameter conversion is the standard microwave-network
relation from Pozar :cite:`m.pozarMicrowaveEngineering2012` (ch. 4).

The implementation was cross-checked against the Qucs-S model
(see `Qucs technical documentation`_, §12 for CPW, §11 for microstrip).

.. _Qucs technical documentation:
   https://qucs.sourceforge.net/docs/technical/technical.pdf

Functions
---------
All geometry parameters are in **SI base units** (metres, etc.) unless
noted otherwise.  Frequency is in **Hz**.
"""

from functools import cache, partial
from typing import cast

import gdsfactory as gf
import jax
import jax.numpy as jnp
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import (
    cpw_epsilon_eff,
    cpw_thickness_correction,
    cpw_z0,
    ellipk_ratio,
    microstrip_epsilon_eff,
    microstrip_thickness_correction,
    microstrip_z0,
    propagation_constant,
    transmission_line_s_params,
)

from qpdk.tech import LAYER_STACK, get_etch_section, material_properties

__all__ = [
    "cbcpw_parameters",
    "cpw_ep_r_from_cross_section",
    "cpw_epsilon_eff",
    "cpw_parameters",
    "cpw_thickness_correction",
    "cpw_z0",
    "cpw_z0_from_cross_section",
    "get_cpw_dimensions",
    "get_cpw_substrate_params",
    "microstrip_epsilon_eff",
    "microstrip_thickness_correction",
    "microstrip_z0",
    "propagation_constant",
    "transmission_line_s_params",
]


@partial(jax.jit, inline=True)
def cbcpw_parameters(
    w: ArrayLike,
    s: ArrayLike,
    h: ArrayLike,
    t: ArrayLike,
    ep_r: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Effective permittivity and impedance of a conductor-backed CPW.

    Quasi-static conformal mapping for a CPW on a substrate of height :math:`h`
    whose backside is metallised, Simons
    :cite:`simonsCoplanarWaveguideCircuits2001` (§3.2):

    .. math::

        \begin{aligned}
            k_0 &= \frac{w}{w + 2s}, \qquad
            k_3 = \frac{\tanh(\pi w / 4h)}{\tanh\bigl(\pi (w + 2s) / 4h\bigr)} \\
            q_i &= K(k_i^2) / K(1 - k_i^2) \\
            \varepsilon_\text{eff} &= \frac{q_0 + \varepsilon_\text{r} q_3 + 1.4\,t/s}{q_0 + q_3 + 1.4\,t/s} \\
            Z_0 &= \frac{60\pi}{\sqrt{\varepsilon_\text{eff}}\,(q_\text{e} + q_3 q_\text{e} / q_0)}
        \end{aligned}

    The conductor thickness enters through the Gupta et al.
    :cite:`guptaMicrostripLinesSlotlines1996` terms used by
    :func:`~sax.models.rf.cpw_thickness_correction`: the extra
    :math:`0.7\,t/s` slot capacitance per side and :math:`q_\text{e}`, the
    capacitance ratio for the thickness-widened :math:`k_\text{e}`. Applying them
    to both half-spaces is a heuristic, chosen so that the result reduces
    exactly to the thickness-corrected CPW of
    :func:`~sax.models.rf.cpw_thickness_correction` for
    :math:`h \gg w + 2s`, where the backside metal has no effect.

    Args:
        w: Centre-conductor width (m).
        s: Gap to ground plane (m).
        h: Substrate height (m).
        t: Conductor thickness (m).
        ep_r: Relative permittivity of the substrate.

    Returns:
        ``(ep_eff, z0)`` — effective permittivity and characteristic impedance (Ω).
    """
    w = jnp.asarray(w, dtype=float)
    s = jnp.asarray(s, dtype=float)
    h = jnp.asarray(h, dtype=float)
    t = jnp.asarray(t, dtype=float)
    ep_r = jnp.asarray(ep_r, dtype=float)

    k0 = w / (w + 2.0 * s)
    k3 = jnp.tanh(jnp.pi * w / (4.0 * h)) / jnp.tanh(jnp.pi * (w + 2.0 * s) / (4.0 * h))
    q0 = ellipk_ratio(k0**2)
    q3 = ellipk_ratio(k3**2)

    t_safe = jnp.where(t < 1e-15, 1e-15, t)
    delta = (1.25 * t / jnp.pi) * (1.0 + jnp.log(4.0 * jnp.pi * w / t_safe))
    ke = jnp.clip(k0 + (1.0 - k0**2) * delta / (2.0 * s), 1e-12, 1.0 - 1e-12)
    ke = jnp.where(t <= 0, k0, ke)
    qe = ellipk_ratio(ke**2)

    slot = 1.4 * t / s
    ep_eff = (q0 + ep_r * q3 + slot) / (q0 + q3 + slot)
    z0 = 60.0 * jnp.pi / (jnp.sqrt(ep_eff) * (qe + q3 * qe / q0))
    return ep_eff, z0


# ===================================================================
# Layout-to-Model Helpers
# ===================================================================


@cache
def get_cpw_substrate_params() -> tuple[float, float, float, float]:
    r"""Extract substrate parameters from the PDK layer stack.

    Returns bare floats instead of something like a dataclass for
    easy integration with JAX-jittable functions.

    Returns:
        ``(h, t, ep_r, tand)`` — substrate height (µm), conductor thickness (µm),
        relative permittivity, and loss tangent :math:`\tan\,\delta`.
    """
    h = LAYER_STACK.layers["Substrate"].thickness  # µm
    t = LAYER_STACK.layers["M1"].thickness  # µm
    substrate_mat = material_properties[
        cast(str, LAYER_STACK.layers["Substrate"].material)
    ]
    ep_r = substrate_mat["relative_permittivity"]
    tand = substrate_mat.get("loss_tangent", 0.0)
    return float(h), float(t), float(ep_r), float(tand)


def get_cpw_dimensions(
    cross_section: CrossSectionSpec, **kwargs
) -> tuple[float, float]:
    """Extracts CPW width and gap from a cross-section specification.

    Args:
        cross_section: A gdsfactory cross-section specification.
        **kwargs: Additional keyword arguments passed to `gf.get_cross_section`.

    Returns:
        tuple[float, float]: Width and gap of the CPW.

    Raises:
        ValueError: If the conductor width or etch gap is not positive, or if
            no etch section is found.
    """
    # Make sure a PDK is activated
    from qpdk import PDK  # ruff: ignore[import-outside-top-level]

    PDK.activate()
    xs = gf.get_cross_section(cross_section, **kwargs)

    width = xs.width
    if not width > 0:
        msg = (
            f"Cross-section '{xs.name}' has non-positive conductor width {width}. "
            "CPW conductor width must be positive."
        )
        raise ValueError(msg)
    etch_section = get_etch_section(xs)
    etch_gap = etch_section.width
    if not etch_gap > 0:
        msg = (
            f"Cross-section '{xs.name}' has non-positive etch gap {etch_gap}. "
            "CPW etch gap must be positive."
        )
        raise ValueError(msg)
    return width, etch_gap


# Note: do not decorate this function with `functools.cache` (or any cache that
# retains return values). The sax helpers called below are jitted with
# `inline=True`, so when this function runs inside a `jax.jit` trace its return
# values contain tracers; caching them would leak tracers into later traces.
def cpw_parameters(
    width: float,
    gap: float,
    *,
    tand: float | None = None,
    conductor_backed: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute complex effective permittivity and characteristic impedance for a CPW.

    Uses the JAX-jittable functions from :mod:`sax.models.rf` with the
    PDK layer stack (substrate height, conductor thickness, material
    permittivity).

    Dielectric loss is included via the filling factor :math:`q` and the
    provided *tand*:

    .. math::

        \varepsilon_{\text{eff, complex}} = \varepsilon_{\text{eff}} \left( 1 - j q \frac{\varepsilon_\text{r}}{\varepsilon_{\text{eff}}} \tan \delta \right)

    where :math:`q = (\varepsilon_{\text{eff}} - 1) / (\varepsilon_\text{r} - 1)`.

    Conductor thickness corrections follow
    Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
    (§7.3, Eqs. 7.98-7.100).

    Args:
        width: Centre-conductor width in µm.
        gap: Gap between centre conductor and ground plane in µm.
        tand: Loss tangent :math:`\tan\,\delta` of the substrate. If None, uses the PDK default.
        conductor_backed: If True, the substrate backside is metallised (see
            :data:`~qpdk.tech.LAYER_STACK_BACKSIDE`) and :func:`cbcpw_parameters`
            is used instead of the CPW model.

    Returns:
        ``(ep_eff, z0)`` — complex effective permittivity (dimensionless) and
        characteristic impedance (Ω).
    """
    width = float(width)
    gap = float(gap)

    h_um, t_um, ep_r, tand_default = get_cpw_substrate_params()
    if tand is None:
        tand = tand_default

    # Convert to SI (metres)
    w_m = width * 1e-6
    s_m = gap * 1e-6
    h_m = h_um * 1e-6
    t_m = t_um * 1e-6

    if conductor_backed:
        ep_eff, z0_val = cbcpw_parameters(w_m, s_m, h_m, t_m, ep_r)
    else:
        # Base (zero-thickness) quantities
        ep_eff = cpw_epsilon_eff(w_m, s_m, h_m, ep_r)

        if t_um > 0:
            ep_eff, z0_val = cpw_thickness_correction(w_m, s_m, t_m, ep_eff)
        else:
            z0_val = cpw_z0(w_m, s_m, ep_eff)

    if tand > 0 and ep_r > 1.0001:
        # Calculate the dielectric filling factor (q)
        q = (ep_eff - 1.0) / (ep_r - 1.0)
        # Substitute the real effective permittivity with a complex model.
        ep_eff = ep_eff * (1 - 1j * q * (ep_r / ep_eff) * tand)

    return jnp.asarray(ep_eff), jnp.asarray(z0_val)


def cpw_z0_from_cross_section(
    cross_section: CrossSectionSpec,
    f: ArrayLike | None = None,
    conductor_backed: bool = False,
) -> jnp.ndarray:
    """Characteristic impedance of a CPW defined by a layout cross-section.

    Args:
        cross_section: A gdsfactory cross-section specification.
        f: Frequency array (Hz). Used only to determine the output shape;
           the impedance is frequency-independent in the quasi-static model.
        conductor_backed: If True, use the conductor-backed CPW model, see
            :func:`cpw_parameters`.

    Returns:
        Characteristic impedance broadcast to the shape of *f* (Ω).
    """
    width, gap = get_cpw_dimensions(cross_section)
    _ep_eff, z0_val = cpw_parameters(width, gap, conductor_backed=conductor_backed)
    z0 = jnp.asarray(z0_val)
    if f is not None:
        f = jnp.asarray(f)
        z0 = jnp.broadcast_to(z0, f.shape)
    return z0


def cpw_ep_r_from_cross_section(
    cross_section: CrossSectionSpec,  # ruff: ignore[unused-function-argument]
) -> float:
    r"""Substrate relative permittivity for a given cross-section.

    .. note::
        The substrate permittivity is determined by the PDK layer stack
        (``LAYER_STACK["Substrate"]``), not by the cross-section geometry.
        All CPW cross-sections on the same substrate share the same
        :math:`\varepsilon_\text{r}`.  The *cross_section* parameter is accepted
        for API symmetry with :func:`cpw_z0_from_cross_section`.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        Relative permittivity of the substrate.
    """
    _h, _t, ep_r, _tand = get_cpw_substrate_params()
    return ep_r
