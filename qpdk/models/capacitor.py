"""Capacitor Models."""

from functools import partial

import jax
import jax.numpy as jnp
import jaxellip
import numpy as np
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0
from qpdk.models.generic import capacitor
from qpdk.models.media import cross_section_to_media


@partial(jax.jit, inline=True)
def _plate_capacitor_capacitance_analytical(
    length: float,
    width: float,
    gap: float,
    ep_r: float,
) -> float:
    """Analytical formula for plate capacitor capacitance."""
    # Simple parallel plate capacitor model with fringing fields correction
    A = length * width * 1e-12  # Area in m² (convert from μm²)
    d = gap * 1e-6  # Separation in m (convert from μm)
    c0 = ε_0 * ep_r * A / d  # Capacitance without fringing fields in F
    fringing_factor = (
        1 + (2 * gap) / width
    )  # Empirical fringing fields correction factor
    return c0 * fringing_factor


@partial(jax.jit, inline=True)
def _interdigital_capacitor_capacitance_analytical(
    fingers: int,
    finger_length: float,
    finger_gap: float,
    thickness: float,
    ep_r: float,
) -> float:
    """Analytical formula for interdigital capacitor capacitance.

    See :cite:`igrejaAnalyticalEvaluationInterdigital2004,gonzalezDesignFabricationInterdigital2015`.
    """
    # Geometric parameters
    n = fingers
    l_overlap = finger_length * 1e-6  # Overlap length in m
    w = thickness  # Finger width
    g = finger_gap  # Finger gap
    eta = w / (w + g)  # Metallization ratio

    # Elliptic integral moduli
    k_i = jnp.sin(jnp.pi * eta / 2)
    k_i_prime = jnp.cos(jnp.pi * eta / 2)
    k_e = 2 * jnp.sqrt(eta) / (1 + eta)
    k_e_prime = (1 - eta) / (1 + eta)

    # Complete elliptic integrals of the first kind K(k)
    # jaxellip.ellipk takes m = k**2 as argument, similar to scipy.special.ellipk
    ki_over_kip = jaxellip.ellipk(k_i**2) / jaxellip.ellipk(k_i_prime**2)
    ke_over_kep = jaxellip.ellipk(k_e**2) / jaxellip.ellipk(k_e_prime**2)

    # Capacitances per unit length (interior and exterior)
    c_i = ε_0 * l_overlap * (ep_r + 1) * ki_over_kip
    c_e = ε_0 * l_overlap * (ep_r + 1) * ke_over_kep

    # Total mutual capacitance
    return jnp.where(
        n == 2,
        c_e / 2,
        (n - 3) * c_i / 2 + 2 * (c_i * c_e) / (c_i + c_e),
    )


def plate_capacitor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 26.0,
    width: float = 5.0,
    gap: float = 7.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    r"""Plate capacitor Sax model.

    Args:
        f: Array of frequency points in Hz
        length: Length of the capacitor pad in μm
        width: Width of the capacitor pad in μm
        gap: Gap between plates in μm
        cross_section: Cross-section specification

    Returns:
        sax.SType: S-parameters dictionary
    """
    f_arr = jnp.asarray(f)
    media = cross_section_to_media(cross_section)
    media_instance = media(frequency=skrf.Frequency.from_f(np.asarray(f_arr)))
    z0 = media_instance.z0
    ep_r = float(media_instance.ep_r)
    capacitance = _plate_capacitor_capacitance_analytical(
        length=length, width=width, gap=gap, ep_r=ep_r
    )
    return capacitor(f=f_arr, capacitance=capacitance, z0=z0)


def interdigital_capacitor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    fingers: int = 4,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    r"""Interdigital capacitor Sax model.

    Args:
        f: Array of frequency points in Hz
        fingers: Total number of fingers (must be >= 2)
        finger_length: Length of each finger in μm
        finger_gap: Gap between adjacent fingers in μm
        thickness: Thickness of fingers in μm
        cross_section: Cross-section specification

    Returns:
        sax.SType: S-parameters dictionary
    """
    f_arr = jnp.asarray(f)
    media = cross_section_to_media(cross_section)
    media_instance = media(frequency=skrf.Frequency.from_f(np.asarray(f_arr)))
    z0 = media_instance.z0
    ep_r = float(media_instance.ep_r)
    capacitance = _interdigital_capacitor_capacitance_analytical(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        ep_r=ep_r,
    )
    return capacitor(f=f_arr, capacitance=capacitance, z0=z0)
