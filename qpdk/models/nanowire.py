"""Model of a lumped resonator made of a high-kinetic-inductance nanowire."""

import math

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY, π
from qpdk.models.cpw import cpw_ep_r_from_cross_section, cpw_z0_from_cross_section
from qpdk.models.generic import shunt_admittance
from qpdk.models.inductor import meander_inductor_inductance_analytical
from qpdk.tech import MeanderCornerStyle, meander_n_squares_drawn

__all__ = [
    "grounded_pad_capacitance_analytical",
    "nanowire_resonator",
    "nanowire_resonator_parameters",
]


def grounded_pad_capacitance_analytical(
    pad_width: float,
    pad_height: float,
    gap: float,
    ep_r: jax.typing.ArrayLike,
) -> jax.Array:
    r"""Capacitance between a coplanar pad and a ground plane edge.

    The pad is a strip of width :math:`W` (``pad_height``) and length
    :math:`\ell` (``pad_width``) at distance :math:`s` from the straight edge
    of a semi-infinite ground plane in the same plane.

    The 2D capacitance of two coplanar electrodes on a line depends only on the
    cross ratio of their four edge positions, because a Möbius map of the half
    plane preserves both the cross ratio and the capacitance. For edges at
    :math:`-\infty, 0, s, s + W` the cross ratio is :math:`\lambda = W/(W + s)`,
    and two symmetric pads with edges at :math:`\pm 1, \pm 1/k` have
    :math:`\lambda = \bigl((1 - k)/(1 + k)\bigr)^2`. So the grounded pad has the
    capacitance of symmetric coplanar pads with modulus

    .. math::

        k = \frac{1 - \sqrt{\lambda}}{1 + \sqrt{\lambda}},

    which :func:`~qpdk.models.capacitor.plate_capacitor_capacitance_analytical`
    evaluates :cite:`chenCompactInductorcapacitorResonators2023`. Fringe fields
    at the two ends of the pad are neglected.

    Args:
        pad_width: Pad length :math:`\ell` along the ground edge in µm.
        pad_height: Pad width :math:`W` across the gap in µm.
        gap: Gap :math:`s` between the pad and the ground edge in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        Capacitance in F.
    """
    sqrt_lambda = math.sqrt(pad_height / (pad_height + gap))
    k = (1 - sqrt_lambda) / (1 + sqrt_lambda)
    return plate_capacitor_capacitance_analytical(
        length=pad_width, width=(1 - k) / 2, gap=k, ep_r=ep_r
    )


def nanowire_resonator_parameters(
    wire_width: float = 0.1,
    wire_length: float = 40.0,
    n_turns: int = 1,
    wire_gap: float | None = None,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
    pad_width: float = 300.0,
    pad_height: float = 150.0,
    sheet_inductance: float = 40e-12,
    thickness: float = 0.015,
    coupling_capacitance: float = 2e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> dict[str, jax.Array | float]:
    r"""Circuit parameters of :func:`qpdk.cells.nanowire_resonator`.

    The resonator is a linear LC to ground, coupled to the feedline through
    :math:`C_\text{c}`:

    .. math::

        L &= L_\square N_\square + L_\text{geo} \\
        \omega_\text{r} &= \frac{1}{\sqrt{L (C + C_\text{c})}} \\
        Q_\text{c} &= \frac{2 (C + C_\text{c})}{\omega_\text{r} C_\text{c}^2 Z_0}

    :math:`N_\square` is the drawn square count from
    :func:`qpdk.tech.meander_n_squares_drawn`, :math:`L_\text{geo}` the geometric
    inductance of the sharp meander from
    :func:`~qpdk.models.inductor.meander_inductor_inductance_analytical`, and
    :math:`C` the pad-to-ground capacitance from
    :func:`grounded_pad_capacitance_analytical` with the wire length as gap.
    :math:`Q_\text{c}` follows from the feedline loading the resonator with
    :math:`Z_0/2` through :math:`C_\text{c}`, valid for
    :math:`\omega_\text{r} C_\text{c} Z_0 \ll 1`.

    Assumptions:

    * The kinetic inductance is linear: no current or power dependence.
    * The far pad is grounded and its ground is a semi-infinite plane.
    * Capacitance from the upper pad to the side ground and to the feedline
      ground is neglected, and so are the fringe fields at the pad ends.
    * :math:`C_\text{c}` is a parameter, not derived from ``coupling_gap``.
    * Losses are not modelled: the internal quality factor is infinite.

    Args:
        wire_width: Nanowire width in µm.
        wire_length: Length of each nanowire run in µm.
        n_turns: Number of nanowire runs.
        wire_gap: Gap between runs in µm. Defaults to ``wire_width``.
        corner_style: Corner style of a meandered wire.
        radius: Centreline radius for ``"round"`` corners in µm.
        pad_width: Pad size along the feedline in µm.
        pad_height: Pad size across the feedline in µm.
        sheet_inductance: Kinetic sheet inductance :math:`L_\square` in H/□.
            The default is the TiN film of
            :cite:`joshiStrongKineticinductanceKerr2022`.
        thickness: Film thickness in µm, used for :math:`L_\text{geo}`.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` in F.
        cross_section: Feedline cross-section, used for the substrate
            permittivity and :math:`Z_0`.

    Returns:
        Dictionary with ``n_squares``, ``kinetic_inductance``,
        ``geometric_inductance``, ``inductance`` (H), ``capacitance`` (F),
        ``coupling_capacitance`` (F), ``z0`` (Ω), ``resonance_frequency`` (Hz)
        and ``coupling_quality_factor``.
    """
    wire_gap = wire_width if wire_gap is None else wire_gap
    n_squares = meander_n_squares_drawn(
        n_turns, wire_length, wire_width, wire_gap, corner_style, radius
    )
    kinetic = sheet_inductance * n_squares
    geometric = meander_inductor_inductance_analytical(
        n_turns=n_turns,
        turn_length=wire_length,
        wire_width=wire_width,
        wire_gap=wire_gap,
        sheet_inductance=0.0,
        thickness=thickness,
    )
    inductance = kinetic + geometric
    ep_r = cpw_ep_r_from_cross_section(cross_section)
    capacitance = grounded_pad_capacitance_analytical(
        pad_width, pad_height, wire_length, ep_r
    )
    z0 = jnp.real(cpw_z0_from_cross_section(cross_section))
    c_total = capacitance + coupling_capacitance
    omega_r = 1 / jnp.sqrt(inductance * c_total)
    q_c = 2 * c_total / (omega_r * coupling_capacitance**2 * z0)
    return {
        "n_squares": n_squares,
        "kinetic_inductance": kinetic,
        "geometric_inductance": geometric,
        "inductance": inductance,
        "capacitance": capacitance,
        "coupling_capacitance": coupling_capacitance,
        "z0": z0,
        "resonance_frequency": omega_r / (2 * π),
        "coupling_quality_factor": q_c,
    }


def nanowire_resonator(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    wire_width: float = 0.1,
    wire_length: float = 40.0,
    n_turns: int = 1,
    wire_gap: float | None = None,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
    pad_width: float = 300.0,
    pad_height: float = 150.0,
    sheet_inductance: float = 40e-12,
    thickness: float = 0.015,
    coupling_capacitance: float = 2e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""SAX model of :func:`qpdk.cells.nanowire_resonator`.

    The resonator branch, :math:`C_\text{c}` in series with :math:`L \parallel C`,
    shunts the feedline at a single reference plane:

    .. math::

        Y(\omega) = \left(\frac{1}{\text{j}\omega C_\text{c}}
            + \frac{1}{\text{j}\omega C + 1/(\text{j}\omega L)}\right)^{-1}

    and the two-port follows from
    :func:`~qpdk.models.generic.shunt_admittance`
    :cite:`m.pozarMicrowaveEngineering2012`. :math:`|S_{21}|` vanishes at
    :math:`\omega_\text{r} = 1/\sqrt{L (C + C_\text{c})}`. Circuit values and
    assumptions are those of
    :func:`~qpdk.models.nanowire.nanowire_resonator_parameters`; the feedline
    itself is treated as lossless and of zero length.

    Design references for the nanowire are
    :cite:`joshiStrongKineticinductanceKerr2022,maleevaCircuitQuantumElectrodynamics2018`.

    Args:
        f: Array of frequency points in Hz.
        wire_width: Nanowire width in µm.
        wire_length: Length of each nanowire run in µm.
        n_turns: Number of nanowire runs.
        wire_gap: Gap between runs in µm. Defaults to ``wire_width``.
        corner_style: Corner style of a meandered wire.
        radius: Centreline radius for ``"round"`` corners in µm.
        pad_width: Pad size along the feedline in µm.
        pad_height: Pad size across the feedline in µm.
        sheet_inductance: Kinetic sheet inductance in H/□.
        thickness: Film thickness in µm.
        coupling_capacitance: Coupling capacitance in F.
        cross_section: Feedline cross-section.

    Returns:
        sax.SDict: S-parameters with ports ``o1`` and ``o2``.
    """
    p = nanowire_resonator_parameters(
        wire_width=wire_width,
        wire_length=wire_length,
        n_turns=n_turns,
        wire_gap=wire_gap,
        corner_style=corner_style,
        radius=radius,
        pad_width=pad_width,
        pad_height=pad_height,
        sheet_inductance=sheet_inductance,
        thickness=thickness,
        coupling_capacitance=coupling_capacitance,
        cross_section=cross_section,
    )
    jw = 1j * 2 * π * jnp.asarray(f)
    z_lc = 1 / (jw * p["capacitance"] + 1 / (jw * p["inductance"]))
    y = 1 / (1 / (jw * p["coupling_capacitance"]) + z_lc)
    # Normalise to the feedline impedance, which is traced under jax.jit.
    return shunt_admittance(f=f, y=y * p["z0"], z0=1.0)
