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
def cpw_cpw_coupling_capacitance_per_length_analytical(
    gap: float | ArrayLike,
    width: float | ArrayLike,
    cpw_gap: float | ArrayLike,
    ep_r: float | ArrayLike,
) -> float | jax.Array:
    r"""Analytical formula for ECCPW mutual capacitance per unit length.

    The model follows the edge-coupled coplanar waveguide (ECCPW) formula
    using conformal mapping for even and odd modes:

    .. math::

        x_1 &= s_c / 2 \\
        x_2 &= x_1 + W \\
        x_3 &= x_2 + G \\
        k_e &= \sqrt{\frac{x_2^2 - x_1^2}{x_3^2 - x_1^2}} \\
        k_o &= \frac{x_1}{x_2} \sqrt{\frac{x_3^2 - x_2^2}{x_3^2 - x_1^2}} \\
        C_{\text{even}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_e)}{K(k_e')} \\
        C_{\text{odd}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_o')}{K(k_o)} \\
        C_m &= \frac{C_{\text{odd}} - C_{\text{even}}}{2}

    where :math:`s_c` is the separation (gap) between inner edges, :math:`W` is the
    center conductor width, and :math:`G` is the gap to the ground plane.

    See :cite:`simonsCoplanarWaveguideCircuits2001`.

    Args:
        gap: The gap (separation) between the two center conductors in µm.
        width: Center conductor width in µm.
        cpw_gap: Gap between center conductor and ground plane in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        The mutual coupling capacitance per unit length in Farads/meter.
    """
    ε_eff = (ep_r + 1) / 2

    # Geometric parameters in m (convert from μm)
    s_c = gap * 1e-6
    w_m = width * 1e-6
    g_m = cpw_gap * 1e-6

    x1 = s_c / 2
    x2 = x1 + w_m
    x3 = x2 + g_m

    # Even-mode modulus
    ke_sq = (x2**2 - x1**2) / (x3**2 - x1**2)
    ke_prime_sq = 1 - ke_sq

    # Odd-mode modulus
    ko_sq = (x1**2 / x2**2) * ((x3**2 - x2**2) / (x3**2 - x1**2))
    ko_prime_sq = 1 - ko_sq

    # Capacitances per unit length
    c_even_pul = 2 * ε_0 * ε_eff * jaxellip.ellipk(ke_sq) / jaxellip.ellipk(ke_prime_sq)
    c_odd_pul = 2 * ε_0 * ε_eff * jaxellip.ellipk(ko_prime_sq) / jaxellip.ellipk(ko_sq)

    # Mutual capacitance per unit length
    return (c_odd_pul - c_even_pul) / 2


def cpw_cpw_coupling_capacitance(
    f: sax.FloatArrayLike,
    length: float | ArrayLike,
    gap: float | ArrayLike,
    cross_section: CrossSectionSpec,
) -> float | jax.Array:
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

    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap,
        width=width,
        cpw_gap=cpw_gap,
        ep_r=ep_r,
    )
    return c_pul * length * 1e-6


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
        "capacitance": cpw_cpw_coupling_capacitance(f, length, gap, cross_section),
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lengths = jnp.linspace(10, 1000, 10)
    gaps = jnp.geomspace(0.1, 5.0, 6)
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    plt.figure(figsize=(10, 6))

    # Calculate capacitance per unit length for all gaps simultaneously (shape: (6,))
    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gaps, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Broadcast to compute total capacitance for all lengths and gaps (shape: (6, 1000))
    capacitances = c_pul[:, None] * lengths[None, :] * 1e-6 * 1e15  # Convert to fF

    for i, gap in enumerate(gaps):
        plt.plot(lengths, capacitances[i], label=f"gap = {gap:.1f} µm")

    plt.xlabel("Coupling Length (µm)")
    plt.ylabel("Mutual Capacitance (fF)")
    plt.title(
        rf"CPW-CPW Coupling Capacitance ($\mathtt{{width}}=${width} µm, $\mathtt{{cpw\_gap}}=${cpw_gap} µm, $\epsilon_r={ep_r}$)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
