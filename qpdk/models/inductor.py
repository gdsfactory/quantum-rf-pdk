"""Inductor and lumped-element resonator models."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY, μ_0, π
from qpdk.models.cpw import (
    cpw_ep_r_from_cross_section,
    get_cpw_dimensions,
    get_cpw_substrate_params,
)
from qpdk.models.generic import inductor, lc_resonator


@partial(jax.jit, inline=True)
def self_inductance_strip(l: float, w: float, t: float) -> jax.Array:
    r"""Analytical formula for the self-inductance of a rectangular metal strip.

    Uses the formula from :cite:`chenCompactInductorcapacitorResonators2023`:

    .. math::

        L_s = \frac{\mu_0 l}{2\pi} \left[ \ln\left(\frac{2l}{w+t}\right) + 0.5 + \frac{w+t}{3l} \right]

    Args:
        l: Length of the strip in m.
        w: Width of the strip in m.
        t: Thickness of the strip in m.

    Returns:
        Self-inductance in Henries.
    """
    return (μ_0 * l / (2 * π)) * (jnp.log(2 * l / (w + t)) + 0.5 + (w + t) / (3 * l))


@partial(jax.jit, inline=True)
def mutual_inductance_parallel_strips(l: float, d: float) -> jax.Array:
    r"""Analytical formula for the mutual inductance between two parallel metal strips.

    Uses the formula from :cite:`chenCompactInductorcapacitorResonators2023`:

    .. math::

        L_m(d) = \frac{\mu_0 l}{2\pi} \left[ \ln \left( \frac{l}{d} + \sqrt{1 + \frac{l^2}{d^2}} \right) - \sqrt{1 + \frac{d^2}{l^2}} + \frac{d}{l} \right]

    Args:
        l: Length of the strips in m.
        d: Center-to-center distance between the strips in m.

    Returns:
        Mutual inductance in Henries.
    """
    return (μ_0 * l / (2 * π)) * (
        jnp.log(l / d + jnp.sqrt(1 + (l / d) ** 2)) - jnp.sqrt(1 + (d / l) ** 2) + d / l
    )


@partial(jax.jit, inline=True)
def meander_inductor_inductance_analytical(
    n_turns: int,
    turn_length: float,
    wire_width: float,
    wire_gap: float,
    sheet_inductance: float,
    thickness: float | None = None,
) -> jax.Array:
    r"""Analytical formula for meander inductor inductance.

    The total inductance is the sum of geometric and kinetic contributions:

    .. math::

        L_{\text{total}} = L_g + L_k

    The geometric inductance :math:`L_g` is calculated by summing the
    self-inductances of all horizontal segments and the mutual inductances
    between all pairs of parallel segments, following
    :cite:`chenCompactInductorcapacitorResonators2023`:

    .. math::

        L_g = N L_s + 2 \sum_{k=1}^{N-1} (N-k) (-1)^k L_m(k p)

    where :math:`N` is the number of turns and :math:`p` is the pitch.

    The kinetic inductance :math:`L_k` is calculated from the sheet
    inductance :math:`L_\square`:

    .. math::

        L_k = L_\square \cdot \frac{\ell_{\text{total}}}{w}

    Args:
        n_turns: Number of horizontal meander runs.
        turn_length: Length of each horizontal run in µm.
        wire_width: Width of the meander wire in µm.
        wire_gap: Gap between adjacent meander runs in µm.
        sheet_inductance: Sheet inductance per square in H/□.
        thickness: Thickness of the metal film in µm. If None, it is
            fetched from the PDK technology parameters.

    Returns:
        Total inductance in Henries.
    """
    if thickness is None:
        _h, thickness, _ep_r = get_cpw_substrate_params()

    # Convert to SI (meters)
    l_m = turn_length * 1e-6
    w_m = wire_width * 1e-6
    g_m = wire_gap * 1e-6
    t_m = thickness * 1e-6
    p_m = w_m + g_m  # Pitch (center-to-center)

    # 1. Geometric Inductance
    # Self-inductance of horizontal segments (turns)
    L_s_horiz = self_inductance_strip(l_m, w_m, t_m)

    # Self-inductance of vertical connection segments
    # There are (n_turns - 1) such segments, each of length p_m (the pitch)
    # They are also wire_width wide.
    L_s_vert = self_inductance_strip(p_m, w_m, t_m)

    # Mutual inductance sum between horizontal segments
    ks = jnp.arange(1, 501)
    mask = ks < n_turns
    L_m_sum = jnp.sum(
        jnp.where(
            mask,
            (n_turns - ks)
            * ((-1.0) ** ks)
            * mutual_inductance_parallel_strips(l_m, ks * p_m),
            0.0,
        )
    )

    L_g = n_turns * L_s_horiz + (n_turns - 1) * L_s_vert + 2 * L_m_sum

    # 2. Kinetic Inductance
    # Total wire length in µm (horizontal runs + vertical connections)
    total_length_um = n_turns * turn_length + jnp.maximum(0, n_turns - 1) * wire_gap
    n_squares = total_length_um / wire_width
    L_k = sheet_inductance * n_squares

    return L_g + L_k


def meander_inductor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_turns: int = 5,
    turn_length: float = 200.0,
    cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    sheet_inductance: float = 0.4e-12,
) -> sax.SDict:
    r"""Meander inductor SAX model.

    Computes the inductance from the meander geometry and returns
    S-parameters of an equivalent lumped inductor.

    The model extracts the center conductor width and gap from the provided
    cross-section. To ensure the etched regions of adjacent meander runs
    do not overlap and interfere with the characteristic impedance of each other,
    the vertical pitch is calculated as:

    .. math::

        p = w + 2 \cdot g

    where :math:`w` is the center conductor width and :math:`g` is the gap
    width. This corresponds to a metal-to-metal spacing of :math:`2g`.

    Args:
        f: Array of frequency points in Hz.
        n_turns: Number of horizontal meander runs.
        turn_length: Length of each horizontal run in µm.
        cross_section: Cross-section specification for the meander wire.
            Used to determine the wire width and the gap between runs.
        sheet_inductance: Sheet inductance per square in H/□.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    f_arr = jnp.asarray(f)
    wire_width, wire_gap_half = get_cpw_dimensions(cross_section)
    wire_gap = 2 * wire_gap_half

    inductance = meander_inductor_inductance_analytical(
        n_turns=n_turns,
        turn_length=turn_length,
        wire_width=wire_width,
        wire_gap=wire_gap,
        sheet_inductance=sheet_inductance,
    )
    return inductor(f=f_arr, inductance=inductance)


def lumped_element_resonator(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    fingers: int = 20,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    finger_thickness: float = 5.0,
    n_turns: int = 5,
    sheet_inductance: float = 0.4e-12,
    cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    grounded: bool = False,
) -> sax.SDict:
    r"""Lumped-element LC resonator SAX model.

    Combines an interdigital capacitor and a meander inductor in parallel
    to form an LC resonator.  The resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    where :math:`C` is computed from the interdigital capacitor geometry
    using :func:`~qpdk.models.capacitor.interdigital_capacitor_capacitance_analytical`
    and :math:`L` is computed from the meander inductor geometry using
    :func:`meander_inductor_inductance_analytical`.

    The inductor section uses the width and gap derived from the
    `cross_section` to ensure consistent RF behavior across the meander.
    The vertical spacing between meander runs is set to twice the etch gap
    to prevent overlap of the etched regions.

    See :cite:`kimThinfilmSuperconductingResonator2011,chenCompactInductorcapacitorResonators2023`.

    Args:
        f: Array of frequency points in Hz.
        fingers: Number of interdigital capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger in µm.
        n_turns: Number of horizontal meander inductor runs (must be odd to
            match the cell geometry where the path spans left-to-right bus bars).
        sheet_inductance: Sheet inductance per square in H/□.
        cross_section: Cross-section specification. Used for substrate
            permittivity and to determine inductor wire width and gap.
        grounded: If True, one port of the resonator is grounded.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    f_arr = jnp.asarray(f)

    ep_r = cpw_ep_r_from_cross_section(cross_section)

    capacitance = interdigital_capacitor_capacitance_analytical(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
        ep_r=ep_r,
    )

    wire_width, wire_gap_half = get_cpw_dimensions(cross_section)
    wire_gap = 2 * wire_gap_half

    cap_width = 2 * finger_thickness + finger_length + finger_gap
    meander_turn_length = cap_width - 4 * wire_width

    inductance = meander_inductor_inductance_analytical(
        n_turns=n_turns,
        turn_length=meander_turn_length,
        wire_width=wire_width,
        wire_gap=wire_gap,
        sheet_inductance=sheet_inductance,
    )

    return lc_resonator(
        f=f_arr,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
    )
