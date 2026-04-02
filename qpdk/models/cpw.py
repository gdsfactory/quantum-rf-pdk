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

from functools import cache
from typing import cast

import gdsfactory as gf
import jax.numpy as jnp
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import (
    cpw_epsilon_eff,
    cpw_thickness_correction,
    cpw_z0,
    microstrip_epsilon_eff,
    microstrip_thickness_correction,
    microstrip_z0,
    propagation_constant,
    transmission_line_s_params,
)

from qpdk.tech import LAYER_STACK, get_etch_section, material_properties

__all__ = [
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


# ===================================================================
# Layout-to-Model Helpers
# ===================================================================


@cache
def get_cpw_substrate_params() -> tuple[float, float, float]:
    """Extract substrate parameters from the PDK layer stack.

    Returns:
        ``(h, t, ep_r)`` — substrate height (µm), conductor thickness (µm),
        and relative permittivity.
    """
    h = LAYER_STACK.layers["Substrate"].thickness  # µm
    t = LAYER_STACK.layers["M1"].thickness  # µm
    ep_r = material_properties[cast(str, LAYER_STACK.layers["Substrate"].material)][
        "relative_permittivity"
    ]
    return float(h), float(t), float(ep_r)


def get_cpw_dimensions(
    cross_section: CrossSectionSpec, **kwargs
) -> tuple[float, float]:
    """Extracts CPW width and gap from a cross-section specification.

    Args:
        cross_section: A gdsfactory cross-section specification.
        **kwargs: Additional keyword arguments passed to `gf.get_cross_section`.

    Returns:
        tuple[float, float]: Width and gap of the CPW.
    """
    # Make sure a PDK is activated
    from qpdk import PDK  # noqa: PLC0415

    PDK.activate()
    xs = gf.get_cross_section(cross_section, **kwargs)

    width = xs.width
    etch_section = get_etch_section(xs)
    return width, etch_section.width


@cache
def cpw_parameters(
    width: float,
    gap: float,
) -> tuple[float, float]:
    r"""Compute effective permittivity and characteristic impedance for a CPW.

    Uses the JAX-jittable functions from :mod:`sax.models.rf` with the
    PDK layer stack (substrate height, conductor thickness, material
    permittivity).

    Conductor thickness corrections follow
    Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
    (§7.3, Eqs. 7.98-7.100).

    Args:
        width: Centre-conductor width in µm.
        gap: Gap between centre conductor and ground plane in µm.

    Returns:
        ``(ep_eff, z0)`` — effective permittivity (dimensionless) and
        characteristic impedance (Ω).
    """
    width = float(width)
    gap = float(gap)

    h_um, t_um, ep_r = get_cpw_substrate_params()

    # Convert to SI (metres)
    w_m = width * 1e-6
    s_m = gap * 1e-6
    h_m = h_um * 1e-6
    t_m = t_um * 1e-6

    # Base (zero-thickness) quantities
    ep_eff = cpw_epsilon_eff(w_m, s_m, h_m, ep_r)

    if t_um > 0:
        ep_eff_t, z0_val = cpw_thickness_correction(w_m, s_m, t_m, ep_eff)
        return float(ep_eff_t), float(z0_val)

    z0_val = cpw_z0(w_m, s_m, ep_eff)
    return float(ep_eff), float(z0_val)


def cpw_z0_from_cross_section(
    cross_section: CrossSectionSpec,
    f: ArrayLike | None = None,
) -> jnp.ndarray:
    """Characteristic impedance of a CPW defined by a layout cross-section.

    Args:
        cross_section: A gdsfactory cross-section specification.
        f: Frequency array (Hz). Used only to determine the output shape;
           the impedance is frequency-independent in the quasi-static model.

    Returns:
        Characteristic impedance broadcast to the shape of *f* (Ω).
    """
    width, gap = get_cpw_dimensions(cross_section)
    _ep_eff, z0_val = cpw_parameters(width, gap)
    z0 = jnp.asarray(z0_val)
    if f is not None:
        f = jnp.asarray(f)
        z0 = jnp.broadcast_to(z0, f.shape)
    return z0


def cpw_ep_r_from_cross_section(
    cross_section: CrossSectionSpec,  # noqa: ARG001
) -> float:
    r"""Substrate relative permittivity for a given cross-section.

    .. note::
        The substrate permittivity is determined by the PDK layer stack
        (``LAYER_STACK["Substrate"]``), not by the cross-section geometry.
        All CPW cross-sections on the same substrate share the same
        :math:`\varepsilon_r`.  The *cross_section* parameter is accepted
        for API symmetry with :func:`cpw_z0_from_cross_section`.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        Relative permittivity of the substrate.
    """
    _h, _t, ep_r = get_cpw_substrate_params()
    return ep_r
