"""Coupler models."""

from functools import partial
from typing import cast

import gdsfactory as gf
import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import (
    capacitor,
    cpw_epsilon_eff,
    cpw_thickness_correction,
    propagation_constant,
    tee,
    transmission_line_s_params,
)

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0, π
from qpdk.models.cpw import (
    cpw_ep_r_from_cross_section,
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
    get_cpw_substrate_params,
)
from qpdk.models.math import (
    capacitance_per_length_conformal,
    ellipk_ratio,
    epsilon_eff,
)
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

        \begin{aligned}
        x_1 &= s_c / 2 \\
        x_2 &= x_1 + W \\
        x_3 &= x_2 + G \\
        k_e &= \sqrt{\frac{x_2^2 - x_1^2}{x_3^2 - x_1^2}} \\
        k_o &= \frac{x_1}{x_2} \sqrt{\frac{x_3^2 - x_2^2}{x_3^2 - x_1^2}} \\
        C_{\text{even}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_e)}{K(k_e')} \\
        C_{\text{odd}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_o')}{K(k_o)} \\
        C_m &= \frac{C_{\text{odd}} - C_{\text{even}}}{2}
        \end{aligned}

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
    # Geometric parameters in m (convert from μm)
    s_c = gap * 1e-6
    w_m = width * 1e-6
    g_m = cpw_gap * 1e-6

    x1 = s_c / 2
    x2 = x1 + w_m
    x3 = x2 + g_m

    # Even-mode modulus squared
    ke_sq = (x2**2 - x1**2) / (x3**2 - x1**2)

    # Odd-mode modulus squared
    ko_sq = (x1**2 / x2**2) * ((x3**2 - x2**2) / (x3**2 - x1**2))

    # Capacitances per unit length
    # Factor is 2.0 since ECCPW formula uses 2 * ε_0 * ε_eff
    c_even_pul = 2.0 * capacitance_per_length_conformal(m=ke_sq, ep_r=ep_r)
    # c_odd uses K(1-m)/K(m) which is the inverse of ellipk_ratio(m)
    c_odd_pul = 2.0 * ε_0 * epsilon_eff(ep_r) / ellipk_ratio(ko_sq)

    # Mutual capacitance per unit length
    return (c_odd_pul - c_even_pul) / 2


def cpw_cpw_coupling_capacitance(
    f: sax.FloatArrayLike,  # ruff: ignore[unused-function-argument]
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

    Note:
        Raises ``ValueError`` via :func:`~qpdk.models.cpw.get_cpw_dimensions`
        if the cross-section has no section with 'etch' in its name, so the
        CPW gap cannot be determined, or if the conductor width or etch gap
        is not positive.
    """
    ep_r = cpw_ep_r_from_cross_section(cross_section)

    width, cpw_gap = get_cpw_dimensions(cross_section)

    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap,
        width=width,
        cpw_gap=cpw_gap,
        ep_r=ep_r,
    )
    return c_pul * length * 1e-6


def coupler_straight(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 10,
    gap: int | float = 16,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for two coupled coplanar waveguides, :func:`~qpdk.cells.waveguides.coupler_straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SDict: S-parameters dictionary

    .. code::

        o2──────▲───────o3
                │gap
        o1──────▼───────o4
    """
    f = jnp.asarray(f)
    straight_settings = {"length": length / 2, "cross_section": cross_section}
    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(f, length, gap, cross_section),
        "z0": cpw_z0_from_cross_section(cross_section, f),
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


def _get_cross_section(cross_section: CrossSectionSpec) -> gf.CrossSection:
    """Resolve a cross-section spec against the activated PDK.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        gf.CrossSection: The resolved cross-section.
    """
    # Local import: qpdk/__init__ imports this module.
    from qpdk import PDK  # ruff: ignore[import-outside-top-level]

    PDK.activate()
    return gf.get_cross_section(cross_section)


def _access_line_s_params(
    f: ArrayLike,
    length: ArrayLike,
    cross_section: CrossSectionSpec,
    z_ref: ArrayLike,
) -> sax.SDict:
    """Access line S-parameters referenced to *z_ref* rather than its own impedance.

    An access cross-section with the same conductor width but a different gap
    has a different characteristic impedance, so the junction to the coupled
    section reflects. Passing ``z_ref`` keeps that step in the model.

    Args:
        f: Array of frequency points in Hz.
        length: Propagation length in µm.
        cross_section: The cross-section of the access line.
        z_ref: Reference impedance of the coupled section in Ω.

    Returns:
        sax.SDict: Two-port S-parameters of the line.
    """
    width, gap = get_cpw_dimensions(cross_section)
    h, t, ep_r, tand = get_cpw_substrate_params()
    ep_eff = cpw_epsilon_eff(width * 1e-6, gap * 1e-6, h * 1e-6, ep_r)
    ep_eff, z0 = cpw_thickness_correction(width * 1e-6, gap * 1e-6, t * 1e-6, ep_eff)

    f = jnp.asarray(f)
    gamma = propagation_constant(f.ravel(), ep_eff, tand=tand, ep_r=ep_r)
    s11, s21 = transmission_line_s_params(gamma, z0, jnp.asarray(length) * 1e-6, z_ref)
    return sax.reciprocal({
        ("o1", "o1"): s11.reshape(f.shape),
        ("o1", "o2"): s21.reshape(f.shape),
        ("o2", "o2"): s11.reshape(f.shape),
    })


def coupler_ring(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length_x: int | float = 20.0,
    gap: int | float = 16,
    cross_section: CrossSectionSpec = "cpw",
    radius: float | None = None,
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float | None = None,
) -> sax.SDict:
    r"""S-parameter model for two coupled coplanar waveguides in a ring configuration.

    The coupled section is :func:`~qpdk.models.couplers.coupler_straight` of
    length *length_x*. Uncoupled propagation is cascaded onto all four ports so
    that the model reference planes land on the ports of
    :func:`~qpdk.cells.waveguides.coupler_ring`: the lower access lines are
    straights of *length_extension*, the upper ones quarter-circle arcs of
    radius *radius*.

    TODO: Fetch coupling capacitance from a curved simulation library.

    Args:
        f: Array of frequency points in Hz
        length_x: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.
        radius: Bend radius of the upper access lines in µm, clamped to the
            bend cross-section's ``radius_min`` like the layout bend. ``None``
            uses the radius of *cross_section*.
        cross_section_bend: Cross-section of the upper access lines. ``None``
            uses *cross_section*.
        length_extension: Length of the lower access lines in µm, always
            derived from the requested *radius*. ``None`` uses ``3.0 + radius``,
            matching gdsfactory's ``coupler_ring``.

    Returns:
        sax.SDict: S-parameters dictionary

    Raises:
        ValueError: If *radius* is ``None`` and *cross_section* has no radius.

    .. code::

        o2──────▲───────o3
                │gap
        o1──────▼───────o4
    """
    xs_main = _get_cross_section(cross_section)
    if radius is None:
        if xs_main.radius is None:
            msg = (
                f"Cross-section '{xs_main.name}' has no radius. "
                "Pass radius explicitly to coupler_ring."
            )
            raise ValueError(msg)
        radius = xs_main.radius
    if length_extension is None:
        length_extension = 3.0 + radius

    xs_bend = cross_section_bend or cross_section
    bend_radius_min = _get_cross_section(xs_bend).radius_min
    bend_radius = (
        radius if bend_radius_min is None else jnp.maximum(radius, bend_radius_min)
    )
    # Upper arms are quarter circles, so each reference plane sits one arc out.
    arc_length = π * bend_radius / 2
    z_ref = cpw_z0_from_cross_section(cross_section)

    instances = {
        "coupler": coupler_straight(
            f=f, length=length_x, gap=gap, cross_section=cross_section
        ),
        "lower_left": straight(
            f=f, length=length_extension, cross_section=cross_section
        ),
        "lower_right": straight(
            f=f, length=length_extension, cross_section=cross_section
        ),
        "upper_left": _access_line_s_params(
            f=f, length=arc_length, cross_section=xs_bend, z_ref=z_ref
        ),
        "upper_right": _access_line_s_params(
            f=f, length=arc_length, cross_section=xs_bend, z_ref=z_ref
        ),
    }
    connections = {
        "lower_left,o2": "coupler,o1",
        "upper_left,o2": "coupler,o2",
        "upper_right,o1": "coupler,o3",
        "lower_right,o1": "coupler,o4",
    }
    ports = {
        "o1": "lower_left,o1",
        "o2": "upper_left,o1",
        "o3": "upper_right,o2",
        "o4": "lower_right,o2",
    }
    return sax.evaluate_circuit_fg((connections, ports), instances)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lengths = jnp.linspace(10, 1000, 10)
    gaps = jnp.geomspace(0.1, 5.0, 6)
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    plt.figure(figsize=(10, 6))

    # Calculate capacitance per unit length for all gaps simultaneously (shape: (6,))
    c_pul = cast(
        jax.Array,
        cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=gaps, width=width, cpw_gap=cpw_gap, ep_r=ep_r
        ),
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
