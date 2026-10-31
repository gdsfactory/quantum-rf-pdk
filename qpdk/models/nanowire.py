"""Model of a lumped resonator made of a high-kinetic-inductance nanowire."""

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from sax.models.rf import capacitor, electrical_short, inductor, tee

from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY, π
from qpdk.models.cpw import (
    cpw_ep_r_from_cross_section,
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
)
from qpdk.models.inductor import meander_inductor_inductance_analytical
from qpdk.models.math import capacitance_per_length_conformal
from qpdk.models.waveguides import straight
from qpdk.tech import MeanderCornerStyle, meander_n_squares_drawn

__all__ = [
    "grounded_pad_capacitance_analytical",
    "nanowire_pad_capacitances_analytical",
    "nanowire_resonator",
    "nanowire_resonator_parameters",
]


def grounded_pad_capacitance_analytical(
    pad_width: jax.typing.ArrayLike,
    pad_height: jax.typing.ArrayLike,
    gap: jax.typing.ArrayLike,
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

    The cross-ratio mapping above is derived here from conformal invariance,
    not quoted from a reference; ``tests/models/test_nanowire.py`` checks it
    against the symmetric-pad formula.

    Args:
        pad_width: Pad length :math:`\ell` along the ground edge in µm.
        pad_height: Pad width :math:`W` across the gap in µm.
        gap: Gap :math:`s` between the pad and the ground edge in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        Capacitance in F.
    """
    sqrt_lambda = jnp.sqrt(pad_height / (pad_height + gap))
    k = (1 - sqrt_lambda) / (1 + sqrt_lambda)
    return plate_capacitor_capacitance_analytical(
        length=pad_width, width=(1 - k) / 2, gap=k, ep_r=ep_r
    )


def _coplanar_strips_capacitance_per_length(
    cross_ratio: jax.typing.ArrayLike, ep_r: jax.typing.ArrayLike
) -> jax.Array:
    r"""Capacitance per length in F/m of two coplanar electrodes with cross ratio :math:`\lambda`.

    See :func:`grounded_pad_capacitance_analytical` for the mapping to the modulus
    :math:`k = (1 - \sqrt{\lambda})/(1 + \sqrt{\lambda})`.

    Returns:
        Capacitance per unit length in F/m.
    """
    sqrt_lambda = jnp.sqrt(cross_ratio)
    k = (1 - sqrt_lambda) / (1 + sqrt_lambda)
    return capacitance_per_length_conformal(1 - k**2, ep_r)


def _cross_ratio(
    a: jax.typing.ArrayLike,
    b: jax.typing.ArrayLike,
    c: jax.typing.ArrayLike,
    d: jax.typing.ArrayLike,
) -> jax.Array:
    r"""Cross ratio of the electrodes ``[a, b]`` and ``[c, d]`` on the real line.

    ``d < a`` means the second electrode wraps through infinity, as for a ground
    plane on both sides.

    Returns:
        The cross ratio :math:`\lambda`.
    """
    a, b, c, d = map(jnp.asarray, (a, b, c, d))
    return (b - a) * (d - c) / ((c - a) * (d - b))


def nanowire_pad_capacitances_analytical(
    pad_width: jax.typing.ArrayLike,
    pad_height: jax.typing.ArrayLike,
    pad_separation: jax.typing.ArrayLike,
    pad_ground_gap: jax.typing.ArrayLike,
    coupling_gap: jax.typing.ArrayLike,
    centre_width: jax.typing.ArrayLike,
    centre_gap: jax.typing.ArrayLike,
    ep_r: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Pad-to-ground and pad-to-feedline capacitance of the nanowire resonator.

    All conductors are zero-thickness sheets on the substrate surface, so each
    2D cross-section is a set of electrodes on a line and its capacitances follow
    from conformal mapping with :math:`\varepsilon_\text{eff} = (\varepsilon_\text{r} + 1)/2`.
    Two such cross-sections are superposed:

    * Across the feedline, along the pad width :math:`\ell`, the line holds the
      ground pad (a half line, at ``pad_separation`` :math:`d` from the pad), the
      pad of height :math:`h`, the feedline centre conductor of width :math:`w`
      at ``coupling_gap`` :math:`g`, and the feedline ground at ``centre_gap``
      :math:`s`. With :math:`C_{X}` the capacitance between electrode :math:`X`
      and the other two shorted together, the mutual capacitance of pad and
      centre conductor is

      .. math::

          C_\text{c} = \frac{\ell}{2} \bigl(C_\text{pad} + C_\text{centre} - C_\text{ground}\bigr),

      exact for the three-electrode line. Each :math:`C_X` is a two-electrode
      capacitance with the modulus of
      :func:`grounded_pad_capacitance_analytical`.
    * The pad-to-ground capacitance is the pad against the ground pad alone,
      from :func:`grounded_pad_capacitance_analytical` with gap :math:`d`, plus
      the pad against the side grounds at ``pad_ground_gap`` over its height
      :math:`h`, as a CPW centre conductor of width :math:`\ell`.

    Superposing the two directions neglects the pad corners and ends. Checked
    against a method-of-moments extraction of the drawn 3D geometry (pad, ground
    pad, side grounds and a 10/6 µm feedline on silicon) at five geometries
    with :math:`\ell` = 200–300 µm, :math:`h` = 80–150 µm, :math:`d` = 40–60 µm,
    :math:`g` = 4–10 µm and ``pad_ground_gap`` = 20–50 µm,
    :math:`C_\text{c}` agrees within −10 % to +2 %, low for larger
    :math:`g` because end fringing to the centre conductor is neglected, and the
    pad-to-ground capacitance within +3 % to +7 %. The reference values are in
    ``tests/models/test_nanowire.py``.

    Args:
        pad_width: Pad size :math:`\ell` along the feedline in µm.
        pad_height: Pad size :math:`h` across the feedline in µm.
        pad_separation: Gap :math:`d` between the pad and the ground pad in µm.
        pad_ground_gap: Gap between the pad and the side grounds in µm.
        coupling_gap: Gap :math:`g` between the pad and the centre conductor in µm.
        centre_width: Feedline centre conductor width :math:`w` in µm.
        centre_gap: Feedline gap :math:`s` in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        Tuple of the pad-to-ground capacitance and the coupling capacitance
        :math:`C_\text{c}`, both in F.
    """
    pad_width, pad_height, pad_separation, pad_ground_gap = map(
        jnp.asarray, (pad_width, pad_height, pad_separation, pad_ground_gap)
    )
    # Electrode edges across the feedline, from the ground pad up.
    ground_bottom = -pad_separation
    pad_bottom, pad_top = 0.0, pad_height
    centre_bottom = pad_top + coupling_gap
    centre_top = centre_bottom + centre_width
    ground_top = centre_top + centre_gap
    c_pad = _coplanar_strips_capacitance_per_length(
        _cross_ratio(pad_bottom, pad_top, centre_bottom, ground_bottom), ep_r
    )
    c_centre = _coplanar_strips_capacitance_per_length(
        _cross_ratio(centre_bottom, centre_top, ground_top, pad_top), ep_r
    )
    c_ground = _coplanar_strips_capacitance_per_length(
        _cross_ratio(pad_bottom, centre_top, ground_top, ground_bottom), ep_r
    )
    coupling = (c_pad + c_centre - c_ground) / 2 * pad_width * 1e-6

    k_side = pad_width / (pad_width + 2 * pad_ground_gap)
    side = 4 * capacitance_per_length_conformal(k_side**2, ep_r) * pad_height * 1e-6
    bottom = grounded_pad_capacitance_analytical(
        pad_width, pad_height, pad_separation, ep_r
    )
    return bottom + side, coupling


def nanowire_resonator_parameters(
    wire_width: float = 0.1,
    wire_length: float = 40.0,
    n_turns: int = 1,
    wire_gap: float | None = None,
    lead_length: float | None = None,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
    pad_width: float = 300.0,
    pad_height: float = 150.0,
    pad_ground_gap: float = 50.0,
    coupling_gap: float = 4.0,
    sheet_inductance: float = 40e-12,
    thickness: float = 0.015,
    coupling_capacitance: float | None = None,
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
    :func:`qpdk.tech.meander_n_squares_drawn` plus ``lead_length / wire_width``
    for each of the two leads, :math:`L_\text{geo}` the geometric
    inductance of the sharp meander from
    :func:`~qpdk.models.inductor.meander_inductor_inductance_analytical`, and
    :math:`C` and :math:`C_\text{c}` the pad-to-ground and pad-to-feedline
    capacitances from :func:`nanowire_pad_capacitances_analytical` for the drawn
    pads, with the pad separation ``wire_length + 2 * lead_length``.
    :math:`Q_\text{c}` follows from the feedline loading the resonator with
    :math:`Z_0/2` through :math:`C_\text{c}`, valid for
    :math:`\omega_\text{r} C_\text{c} Z_0 \ll 1`.

    Assumptions:

    * The kinetic inductance is linear: no current or power dependence.
    * :math:`C` and :math:`C_\text{c}` are closed-form approximations that
      neglect the pad corners and ends. Against a method-of-moments
      extraction they agree within the bands stated in
      :func:`nanowire_pad_capacitances_analytical`, which shift
      :math:`f_\text{r}` by a few percent. Pass ``coupling_capacitance`` to
      use an extracted value instead.
    * The wire and its leads add no capacitance.
    * The feedline is a lossless CPW of ``cross_section``, and the resonator
      couples to it at a single point, its midpoint.
    * Losses are not modelled: the internal quality factor is infinite.

    Args:
        wire_width: Nanowire width in µm.
        wire_length: Length of each nanowire run in µm.
        n_turns: Number of nanowire runs.
        wire_gap: Gap between runs in µm. Defaults to ``wire_width``.
        lead_length: Length of each straight lead between the meander and a
            pad in µm. Defaults to ``wire_gap``.
        corner_style: Corner style of a meandered wire.
        radius: Centreline radius for ``"round"`` corners in µm.
        pad_width: Pad size along the feedline in µm.
        pad_height: Pad size across the feedline in µm.
        pad_ground_gap: Gap between the pads and the side grounds in µm.
        coupling_gap: Gap between the upper pad and the feedline centre
            conductor in µm.
        sheet_inductance: Kinetic sheet inductance :math:`L_\square` in H/□.
            The PDK has no calibrated NbTiN value. The 40 pH/□ default is the
            TiN film of :cite:`joshiStrongKineticinductanceKerr2022`, an example
            only; set it from a measurement of the film in use.
        thickness: Film thickness in µm, used for :math:`L_\text{geo}`.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` in F.
            ``None`` derives it from the layout.
        cross_section: Feedline cross-section, used for the substrate
            permittivity, the feedline dimensions and :math:`Z_0`.

    Returns:
        Dictionary with ``n_squares``, ``kinetic_inductance``,
        ``geometric_inductance``, ``inductance`` (H), ``capacitance`` (F),
        ``coupling_capacitance`` (F), ``z0`` (Ω), ``resonance_frequency`` (Hz)
        and ``coupling_quality_factor``.

    Raises:
        ValueError: If ``n_turns`` is not a positive odd number.
    """
    if n_turns < 1 or n_turns % 2 == 0:
        raise ValueError(
            f"n_turns must be a positive odd number, as in the cell, got {n_turns}"
        )
    wire_gap = wire_width if wire_gap is None else wire_gap
    lead_length = wire_gap if lead_length is None else lead_length
    n_squares = (
        meander_n_squares_drawn(
            n_turns, wire_length, wire_width, wire_gap, corner_style, radius
        )
        + 2 * lead_length / wire_width
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
    centre_width, centre_gap = get_cpw_dimensions(cross_section)
    capacitance, layout_coupling = nanowire_pad_capacitances_analytical(
        pad_width,
        pad_height,
        wire_length + 2 * lead_length,
        pad_ground_gap,
        coupling_gap,
        centre_width,
        centre_gap,
        ep_r,
    )
    c_c = layout_coupling if coupling_capacitance is None else coupling_capacitance
    z0 = jnp.real(cpw_z0_from_cross_section(cross_section))
    c_total = capacitance + c_c
    omega_r = 1 / jnp.sqrt(inductance * c_total)
    q_c = 2 * c_total / (omega_r * c_c**2 * z0)
    return {
        "n_squares": n_squares,
        "kinetic_inductance": kinetic,
        "geometric_inductance": geometric,
        "inductance": inductance,
        "capacitance": capacitance,
        "coupling_capacitance": c_c,
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
    lead_length: float | None = None,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
    pad_width: float = 300.0,
    pad_height: float = 150.0,
    pad_ground_gap: float = 50.0,
    coupling_gap: float = 4.0,
    sheet_inductance: float = 40e-12,
    thickness: float = 0.015,
    coupling_capacitance: float | None = None,
    feedline_length: float = 500.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""SAX model of :func:`qpdk.cells.nanowire_resonator`.

    The circuit is built from generic SAX elements
    :cite:`m.pozarMicrowaveEngineering2012`. The feedline is two
    :func:`~qpdk.models.straight` halves of ``feedline_length / 2``, joined by a
    :func:`~sax.models.rf.tee`. The third tee port leads through the coupling
    :func:`~sax.models.rf.capacitor` :math:`C_\text{c}` to the resonator node,
    where a :func:`~sax.models.rf.capacitor` :math:`C` and an
    :func:`~sax.models.rf.inductor` :math:`L` each end in an
    :func:`~sax.models.rf.electrical_short` to ground:

    .. svgbob::

        o1 ─── feed_1 ───┬─── feed_2 ─── o2
                         │
                         Cc
                         │
                    ┌────┴────┐
                    C         L
                    │         │
                   GND       GND

    :math:`|S_{21}|` vanishes at
    :math:`\omega_\text{r} = 1/\sqrt{L (C + C_\text{c})}`. Circuit values and
    assumptions are those of
    :func:`~qpdk.models.nanowire.nanowire_resonator_parameters`.

    Design references for the nanowire are
    :cite:`joshiStrongKineticinductanceKerr2022,maleevaCircuitQuantumElectrodynamics2018`.

    Args:
        f: Array of frequency points in Hz.
        wire_width: Nanowire width in µm.
        wire_length: Length of each nanowire run in µm.
        n_turns: Number of nanowire runs.
        wire_gap: Gap between runs in µm. Defaults to ``wire_width``.
        lead_length: Length of each straight lead between the meander and a
            pad in µm. Defaults to ``wire_gap``.
        corner_style: Corner style of a meandered wire.
        radius: Centreline radius for ``"round"`` corners in µm.
        pad_width: Pad size along the feedline in µm.
        pad_height: Pad size across the feedline in µm.
        pad_ground_gap: Gap between the pads and the side grounds in µm.
        coupling_gap: Gap between the upper pad and the feedline in µm.
        sheet_inductance: Kinetic sheet inductance in H/□. The default is a TiN
            example, not a calibrated NbTiN value.
        thickness: Film thickness in µm.
        coupling_capacitance: Coupling capacitance in F. ``None`` derives it
            from the layout.
        feedline_length: Length of the feedline in µm.
        cross_section: Feedline cross-section.

    Returns:
        sax.SDict: S-parameters with ports ``o1`` and ``o2``.
    """
    p = nanowire_resonator_parameters(
        wire_width=wire_width,
        wire_length=wire_length,
        n_turns=n_turns,
        wire_gap=wire_gap,
        lead_length=lead_length,
        corner_style=corner_style,
        radius=radius,
        pad_width=pad_width,
        pad_height=pad_height,
        pad_ground_gap=pad_ground_gap,
        coupling_gap=coupling_gap,
        sheet_inductance=sheet_inductance,
        thickness=thickness,
        coupling_capacitance=coupling_capacitance,
        cross_section=cross_section,
    )
    f_arr = jnp.asarray(f)
    z0 = p["z0"]
    instances: dict[str, sax.SType] = {
        "feed_1": straight(
            f=f_arr, length=feedline_length / 2, cross_section=cross_section
        ),
        "feed_2": straight(
            f=f_arr, length=feedline_length / 2, cross_section=cross_section
        ),
        "tee_feed": tee(f=f_arr),
        "coupling": capacitor(f=f_arr, capacitance=p["coupling_capacitance"], z0=z0),
        "tee_node": tee(f=f_arr),
        "pad": capacitor(f=f_arr, capacitance=p["capacitance"], z0=z0),
        "wire": inductor(f=f_arr, inductance=p["inductance"], z0=z0),
        "ground_pad": electrical_short(f=f_arr),
        "ground_wire": electrical_short(f=f_arr),
    }
    connections = {
        "feed_1,o2": "tee_feed,o1",
        "tee_feed,o2": "feed_2,o1",
        "tee_feed,o3": "coupling,o1",
        "coupling,o2": "tee_node,o1",
        "tee_node,o2": "pad,o1",
        "pad,o2": "ground_pad,o1",
        "tee_node,o3": "wire,o1",
        "wire,o2": "ground_wire,o1",
    }
    ports = {"o1": "feed_1,o1", "o2": "feed_2,o2"}
    return sax.evaluate_circuit_fg((connections, ports), instances)
