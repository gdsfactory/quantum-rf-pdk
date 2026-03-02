"""Capacitor Models."""

import jax.numpy as jnp
import numpy as np
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec
from scipy.special import ellipk

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0
from qpdk.models.generic import capacitor
from qpdk.models.media import cross_section_to_media


def _get_plate_capacitor_extraction_results(
    *,
    length: float = 26.0,
    width: float = 5.0,
    gap: float = 7.0,
    cross_section: CrossSectionSpec = "cpw",
) -> float:
    """Extract plate capacitor capacitance analytically."""
    media = cross_section_to_media(cross_section)
    frequency = skrf.Frequency.from_f(DEFAULT_FREQUENCY)
    # Simple parallel plate capacitor model with fringing fields correction
    ep_r = media(frequency=frequency).ep_r  # Relative permittivity of the substrate
    A = length * width * 1e-12  # Area in m² (convert from μm²)
    d = gap * 1e-6  # Separation in m (convert from μm)
    c0 = ε_0 * ep_r * A / d  # Capacitance without fringing fields in F
    fringing_factor = (
        1 + (2 * gap) / width
    )  # Empirical fringing fields correction factor
    return c0 * fringing_factor


def _get_interdigital_capacitor_extraction_results(
    *,
    fingers: int = 4,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
    cross_section: CrossSectionSpec = "cpw",
) -> float:
    """Extract interdigital capacitor capacitance analytically."""
    media = cross_section_to_media(cross_section)
    frequency = skrf.Frequency.from_f(DEFAULT_FREQUENCY)
    ep_r = media(frequency=frequency).ep_r

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
    # SciPy's ellipk takes m = k**2 as argument
    ki_over_kip = ellipk(float(k_i**2)) / ellipk(float(k_i_prime**2))
    ke_over_kep = ellipk(float(k_e**2)) / ellipk(float(k_e_prime**2))

    # Capacitances per unit length (interior and exterior)
    c_i = ε_0 * l_overlap * (ep_r + 1) * ki_over_kip
    c_e = ε_0 * l_overlap * (ep_r + 1) * ke_over_kep

    # Total mutual capacitance
    c_total = c_e / 2 if n == 2 else (n - 3) * c_i / 2 + 2 * (c_i * c_e) / (c_i + c_e)

    return float(c_total)


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
    z0 = media(frequency=skrf.Frequency.from_f(np.asarray(f_arr))).z0
    capacitance = _get_plate_capacitor_extraction_results(
        length=length, width=width, gap=gap, cross_section=cross_section
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
    z0 = media(frequency=skrf.Frequency.from_f(np.asarray(f_arr))).z0
    capacitance = _get_interdigital_capacitor_extraction_results(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        cross_section=cross_section,
    )
    return capacitor(f=f_arr, capacitance=capacitance, z0=z0)
