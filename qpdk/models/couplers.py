"""Coupler models."""

from functools import partial
from typing import cast

import gdsfactory as gf
import jax
import jax.numpy as jnp
import jaxellip
import numpy as np
import sax
import skrf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import capacitor, tee

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0
from qpdk.models.media import cross_section_to_media
from qpdk.models.waveguides import straight


@partial(jax.jit, inline=True)
def cpw_cpw_coupling_capacitance_analytical(
    length: float,
    gap: float,
    width: float,
    cpw_gap: float,
    ep_r: float,
) -> float:
    r"""Analytical formula for ECCPW mutual capacitance.

    The model follows the edge-coupled coplanar waveguide (ECCPW) formula
    using conformal mapping for even and odd modes:

    .. math::

        x_1 &= s_c / 2 \\
        x_2 &= x_1 + W \\
        x_3 &= x_2 + G \\
        k_e &= \frac{x_1}{x_2} \sqrt{\frac{x_3^2 - x_2^2}{x_3^2 - x_1^2}} \\
        k_o &= \frac{x_1}{x_3} \sqrt{\frac{x_3^2 - x_2^2}{x_2^2 - x_1^2}} \\
        C_{\text{even}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_e)}{K(k_e')} \\
        C_{\text{odd}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_o)}{K(k_o')} \\
        C_m &= L \frac{C_{\text{odd}} - C_{\text{even}}}{2}

    where :math:`s_c` is the separation (gap) between inner edges, :math:`W` is the
    center conductor width, and :math:`G` is the gap to the ground plane.

    See :cite:`simonsCoplanarWaveguideCircuits2001`.

    Args:
        length: The coupling length in µm.
        gap: The gap (separation) between the two center conductors in µm.
        width: Center conductor width in µm.
        cpw_gap: Gap between center conductor and ground plane in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        The total coupling capacitance in Farads.
    """
    ep_eff = (ep_r + 1) / 2

    # Geometric parameters in m (convert from μm)
    s_c = gap * 1e-6
    w_m = width * 1e-6
    g_m = cpw_gap * 1e-6

    x1 = s_c / 2
    x2 = x1 + w_m
    x3 = x2 + g_m

    # Even-mode modulus
    ke_sq = (x1**2 / x2**2) * ((x3**2 - x2**2) / (x3**2 - x1**2))
    ke_prime_sq = 1 - ke_sq

    # Odd-mode modulus
    ko_sq = (x1**2 / x3**2) * ((x3**2 - x2**2) / (x2**2 - x1**2))
    ko_prime_sq = 1 - ko_sq

    # Capacitances per unit length
    c_even_pul = (
        2 * ε_0 * ep_eff * jaxellip.ellipk(ke_sq) / jaxellip.ellipk(ke_prime_sq)
    )
    c_odd_pul = 2 * ε_0 * ep_eff * jaxellip.ellipk(ko_sq) / jaxellip.ellipk(ko_prime_sq)

    # Total mutual capacitance
    return (length * 1e-6) * (c_odd_pul - c_even_pul) / 2


def cpw_cpw_coupling_capacitance(
    f: sax.FloatArrayLike,
    length: float,
    gap: float,
    cross_section: CrossSectionSpec,
) -> float:
    r"""Calculate the coupling capacitance between two parallel CPWs.

    Args:
        f: Frequency array in Hz.
        length: The coupling length in µm.
        gap: The gap between the two center conductors in µm.
        cross_section: The cross-section of the CPW.

    Returns:
        The total coupling capacitance in Farads.
    """
    f_arr = jnp.asarray(f)
    media = cross_section_to_media(cross_section)
    media_instance = media(
        frequency=skrf.Frequency.from_f(np.atleast_1d(np.asarray(f_arr)))
    )
    ep_r = float(media_instance.ep_r)

    # Extract CPW dimensions from cross-section
    xs: CrossSection
    if isinstance(cross_section, CrossSection):
        xs = cross_section
    elif callable(cross_section):
        xs = cast(CrossSection, cross_section())
    else:
        xs = gf.get_cross_section(cross_section)

    width = xs.width
    try:
        cpw_gap = next(
            section.width
            for section in xs.sections
            if section.name and "etch_offset" in section.name
        )
    except StopIteration:
        # Fallback to default CPW gap if not found in sections
        gf.logger.warning(
            "CPW gap not found in cross-section sections. Using default value of 6.0 µm."
        )
        cpw_gap = 6.0

    return cpw_cpw_coupling_capacitance_analytical(
        length=length,
        gap=gap,
        width=width,
        cpw_gap=cpw_gap,
        ep_r=ep_r,
    )


def coupler_straight(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 20.0,
    gap: int | float = 0.27,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for two coupled coplanar waveguides, :func:`~qpdk.cells.waveguides.coupler_straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SType: S-parameters dictionary

    .. code::

        o2──────▲───────o3
                │gap
        o1──────▼───────o4
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    straight_settings = {"length": length / 2, "cross_section": cross_section}
    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(
            f, length, gap, cross_section
        ),
        "z0": cross_section_to_media(cross_section)(
            frequency=skrf.Frequency.from_f(
                np.atleast_1d(np.asarray(f_flat)), unit="Hz"
            )
        ).z0.reshape(f.shape),
    }

    # Create straight instances with shared settings
    straight_instances = {
        f"straight_{i}_{j}": straight(f=f, **straight_settings)
        for i in [1, 2]
        for j in [1, 2]
    }
    tee_instances = {f"tee_{i}": tee(f=f) for i in [1, 2]}

    instances = {
        **straight_instances,
        **tee_instances,
        "capacitor": capacitor(f=f, **capacitor_settings),
    }
    connections = {
        "straight_1_1,o1": "tee_1,o1",
        "straight_1_2,o1": "tee_1,o2",
        "straight_2_1,o1": "tee_2,o1",
        "straight_2_2,o1": "tee_2,o2",
        "tee_1,o3": "capacitor,o1",
        "tee_2,o3": "capacitor,o2",
    }
    ports = {
        "o2": "straight_1_1,o2",
        "o3": "straight_1_2,o2",
        "o1": "straight_2_1,o2",
        "o4": "straight_2_2,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def coupler_ring(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 20.0,
    gap: int | float = 0.27,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for two coupled coplanar waveguides in a ring configuration.

    The implementation is the same as straight coupler for now.

    TODO: Fetch coupling capacitance from a curved simulation library.

    Args:
        f: Array of frequency points in Hz
        length: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return coupler_straight(f=f, length=length, gap=gap, cross_section=cross_section)
