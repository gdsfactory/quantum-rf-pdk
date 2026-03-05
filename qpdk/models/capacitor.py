"""Capacitor Models."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0
from qpdk.models.generic import capacitor
from qpdk.models.math import (
    capacitance_per_length_conformal,
    ellipk_ratio,
    epsilon_eff,
)
from qpdk.models.media import cross_section_to_media


@partial(jax.jit, inline=True)
def plate_capacitor_capacitance_analytical(
    length: float,
    width: float,
    gap: float,
    ep_r: float,
) -> float:
    r"""Analytical formula for plate capacitor capacitance.

    The model assumes two coplanar rectangular pads on a substrate.
    The capacitance is calculated using conformal mapping:

    .. math::

        k &= \frac{s}{s + 2W} \\
        k' &= \sqrt{1 - k^2} \\
        \epsilon_{\text{eff}} &= \frac{\epsilon_r + 1}{2} \\
        C &= \epsilon_0 \epsilon_{\text{eff}} L \frac{K(k')}{K(k)}

    where :math:`s` is the gap, :math:`W` is the pad width, and :math:`L` is the pad length.

    See :cite:`chenCompactInductorcapacitorResonators2023`.
    """
    # Conformal mapping for coplanar pads
    k_sq = (gap / (gap + 2 * width)) ** 2

    # C = ε_0 * ε_eff * L * K(k') / K(k)
    # Uses K(1-m)/K(m) which is the inverse of ellipk_ratio(m)
    c_pul = ε_0 * epsilon_eff(ep_r) / ellipk_ratio(k_sq)

    return (length * 1e-6) * c_pul


@partial(jax.jit, inline=True)
def interdigital_capacitor_capacitance_analytical(
    fingers: int,
    finger_length: float,
    finger_gap: float,
    thickness: float,
    ep_r: float,
) -> jax.Array:
    r"""Analytical formula for interdigital capacitor capacitance.

    The formula uses conformal mapping for the interior and exterior regions of
    the interdigital structure:

    .. math::

        \eta &= \frac{w}{w + g} \\
        k_i &= \sin\left(\frac{\pi \eta}{2}\right) \\
        k_e &= \frac{2\sqrt{\eta}}{1 + \eta} \\
        C_i &= \epsilon_0 L (\epsilon_r + 1) \frac{K(k_i)}{K(k_i')} \\
        C_e &= \epsilon_0 L (\epsilon_r + 1) \frac{K(k_e)}{K(k_e')}

    The total mutual capacitance for :math:`n` fingers is:

    .. math::

        C = \begin{cases}
            C_e / 2 & \text{if } n=2 \\
            (n - 3) \frac{C_i}{2} + 2 \frac{C_i C_e}{C_i + C_e} & \text{if } n > 2
        \end{cases}

    where :math:`w` is the finger thickness (width), :math:`g` is the finger gap, and
    :math:`L` is the overlap length.

    See :cite:`igrejaAnalyticalEvaluationInterdigital2004,gonzalezDesignFabricationInterdigital2015`.
    """
    # Geometric parameters
    n = fingers
    l_overlap = finger_length * 1e-6  # Overlap length in m
    w = thickness  # Finger width
    g = finger_gap  # Finger gap
    η = w / (w + g)  # Metallization ratio

    # Elliptic integral moduli squared
    ki_sq = jnp.sin(jnp.pi * η / 2) ** 2
    ke_sq = (2 * jnp.sqrt(η) / (1 + η)) ** 2

    # Capacitances per unit length (interior and exterior)
    # Factor is 2.0 since interdigital formula uses (ep_r + 1) = 2 * ep_eff
    c_i = l_overlap * 2.0 * capacitance_per_length_conformal(m=ki_sq, ep_r=ep_r)
    c_e = l_overlap * 2.0 * capacitance_per_length_conformal(m=ke_sq, ep_r=ep_r)

    # Total mutual capacitance
    # Simplifies to c_e/2 for n=2
    return jnp.where(  # pyrefly: ignore[bad-return]
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
) -> sax.SDict:
    r"""Plate capacitor Sax model.

    Args:
        f: Array of frequency points in Hz
        length: Length of the capacitor pad in μm
        width: Width of the capacitor pad in μm
        gap: Gap between plates in μm
        cross_section: Cross-section specification

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f_arr = jnp.asarray(f)
    media = cross_section_to_media(cross_section)
    media_instance = media(
        frequency=skrf.Frequency.from_f(np.atleast_1d(np.asarray(f_arr)))
    )
    z0 = media_instance.z0
    ep_r = float(media_instance.ep_r)
    capacitance = plate_capacitor_capacitance_analytical(
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
) -> sax.SDict:
    r"""Interdigital capacitor Sax model.

    Args:
        f: Array of frequency points in Hz
        fingers: Total number of fingers (must be >= 2)
        finger_length: Length of each finger in μm
        finger_gap: Gap between adjacent fingers in μm
        thickness: Thickness of fingers in μm
        cross_section: Cross-section specification

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f_arr = jnp.asarray(f)
    media = cross_section_to_media(cross_section)
    media_instance = media(
        frequency=skrf.Frequency.from_f(np.atleast_1d(np.asarray(f_arr)))
    )
    z0 = media_instance.z0
    ep_r = float(media_instance.ep_r)
    capacitance = interdigital_capacitor_capacitance_analytical(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        ep_r=ep_r,
    )
    return capacitor(f=f_arr, capacitance=capacitance, z0=z0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Plot Plate Capacitor Capacitance vs. Length for different Gaps
    lengths = jnp.linspace(10, 500, 100)
    gaps_plate = jnp.geomspace(1.0, 20.0, 5)
    width_plate = 10.0
    ep_r = 11.7

    plt.figure(figsize=(10, 6))

    # Broadcast to compute total capacitance for all lengths and gaps (shape: (5, 100))
    # length * 1e-6 happens in the analytical formula, here we replicate it
    capacitances_plate = (
        plate_capacitor_capacitance_analytical(
            length=lengths[None, :],
            width=width_plate,
            gap=gaps_plate[:, None],
            ep_r=ep_r,
        )
        * 1e15
    )  # Convert to fF

    for i, gap in enumerate(gaps_plate):
        plt.plot(lengths, capacitances_plate[i], label=f"gap = {gap:.1f} µm")

    plt.xlabel("Pad Length (µm)")
    plt.ylabel("Capacitance (fF)")
    plt.title(
        rf"Plate Capacitor Capacitance ($\mathtt{{width}}=${width_plate} µm, $\epsilon_r={ep_r}$)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Plot Interdigital Capacitor Capacitance vs. Finger Length for different Finger Counts
    finger_lengths = jnp.linspace(10, 100, 100)
    finger_counts = jnp.arange(2, 11, 2)  # [2, 4, 6, 8, 10]
    finger_gap = 2.0
    thickness = 5.0

    plt.figure(figsize=(10, 6))

    # Broadcast to compute total capacitance for all lengths and counts (shape: (5, 100))
    capacitances_idc = (
        interdigital_capacitor_capacitance_analytical(
            fingers=finger_counts[:, None],
            finger_length=finger_lengths[None, :],
            finger_gap=finger_gap,
            thickness=thickness,
            ep_r=ep_r,
        )
        * 1e15
    )  # Convert to fF

    for i, n in enumerate(finger_counts):
        plt.plot(finger_lengths, capacitances_idc[i], label=f"n = {n} fingers")

    plt.xlabel("Overlap Length (µm)")
    plt.ylabel("Mutual Capacitance (fF)")
    plt.title(
        rf"Interdigital Capacitor Capacitance ($\mathtt{{finger\_gap}}=${finger_gap} µm, $\mathtt{{thickness}}=${thickness} µm, $\epsilon_r={ep_r}$)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
