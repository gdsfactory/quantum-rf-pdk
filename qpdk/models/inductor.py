"""Inductor and lumped-element resonator models."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_ep_r_from_cross_section
from qpdk.models.generic import inductor, lc_resonator


@partial(jax.jit, inline=True)
def meander_inductor_inductance_analytical(
    n_turns: int,
    turn_length: float,
    wire_width: float,
    wire_gap: float,
    sheet_inductance: float,
) -> jax.Array:
    r"""Analytical formula for meander inductor inductance.

    The total inductance is dominated by kinetic inductance for
    superconducting thin films:

    .. math::

        L = L_\square \cdot \frac{\ell_{\text{total}}}{w}

    where :math:`L_\square` is the sheet inductance per square,
    :math:`\ell_{\text{total}}` is the total wire length, and
    :math:`w` is the wire width.

    The sheet inductance combines kinetic and geometric contributions.
    For a superconducting film of thickness :math:`t` and London
    penetration depth :math:`\lambda_L`:

    .. math::

        L_\square^{\text{kin}} = \frac{\mu_0 \lambda_L^2}{t}

    See :cite:`chenCompactInductorcapacitorResonators2023` for details
    on analytical models of meander inductors in LC resonators.

    Args:
        n_turns: Number of horizontal meander runs.
        turn_length: Length of each horizontal run in µm.
        wire_width: Width of the meander wire in µm.
        wire_gap: Gap between adjacent meander runs in µm.
        sheet_inductance: Sheet inductance per square in H/□.
            Includes both kinetic and geometric contributions.
            Typical values for 200 nm Nb: 0.4–2.0 pH/□.

    Returns:
        Total inductance in Henries.
    """
    # Total wire length in µm (horizontal runs + vertical connections)
    total_length_um = n_turns * turn_length + jnp.maximum(0, n_turns - 1) * wire_gap
    # Number of squares
    n_squares = total_length_um / wire_width
    # Total inductance
    return sheet_inductance * n_squares  # pyrefly: ignore[bad-return]


def meander_inductor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_turns: int = 5,
    turn_length: float = 200.0,
    wire_width: float = 2.0,
    wire_gap: float = 4.0,
    sheet_inductance: float = 0.4e-12,
) -> sax.SDict:
    r"""Meander inductor SAX model.

    Computes the inductance from the meander geometry and returns
    S-parameters of an equivalent lumped inductor.

    Args:
        f: Array of frequency points in Hz.
        n_turns: Number of horizontal meander runs.
        turn_length: Length of each horizontal run in µm.
        wire_width: Width of the meander wire in µm.
        wire_gap: Gap between adjacent meander runs in µm.
        sheet_inductance: Sheet inductance per square in H/□.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    f_arr = jnp.asarray(f)
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
    wire_width: float = 2.0,
    wire_gap: float = 4.0,
    sheet_inductance: float = 0.4e-12,
    cross_section: CrossSectionSpec = "cpw",
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

    See :cite:`kimThinfilmSuperconductingResonator2011,chenCompactInductorcapacitorResonators2023`.

    Args:
        f: Array of frequency points in Hz.
        fingers: Number of interdigital capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger in µm.
        n_turns: Number of horizontal meander inductor runs.
        wire_width: Width of the inductor wire in µm.
        wire_gap: Gap between adjacent inductor runs in µm.
        sheet_inductance: Sheet inductance per square in H/□.
        cross_section: Cross-section specification (used for substrate permittivity).
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

    inductance = meander_inductor_inductance_analytical(
        n_turns=n_turns,
        turn_length=finger_length + finger_gap,
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
