r"""Suspended vacuum-gap (drum) capacitor and drum LC resonator models.

A drum is a parallel-plate capacitor whose top electrode is a thin membrane
suspended a vacuum gap :math:`d` above a bottom electrode
:cite:`cicakLowlossSuperconductingResonant2010`. Shunting it with an inductor
gives an LC resonator whose frequency depends on the membrane position, which
is the circuit electromechanics setup of
:cite:`teufelCircuitCavityElectromechanics2011`.

The gap :math:`d` and the membrane thickness :math:`t` are layer-stack
parameters, read from the ``VacuumGap_Top`` level by default
(see :func:`qpdk.tech.get_vacuum_gap`).

Assumptions shared by every function in this module:

- The plates are parallel, rigid and coaxial for the capacitance.
- The overlap is set by the bottom-electrode radius :math:`a`, the membrane
  being larger than the electrode.
- Displacements are small, :math:`x \ll d`.
- Electrostatic softening, Casimir forces and every nonlinear term are
  neglected.
"""

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY, h, ε_0, π
from qpdk.models.generic import capacitor, shunt_admittance
from qpdk.models.inductor import (
    _get_wire_width_and_gap,
    meander_inductor_inductance_analytical,
)
from qpdk.models.jit import jit
from qpdk.tech import LAYER_STACK, get_layer_material_properties, get_vacuum_gap

VACUUM_GAP: float = get_vacuum_gap()
"""Default vacuum gap in µm, from the ``VacuumGap_Top`` layer-stack level."""

DRUM_THICKNESS: float = LAYER_STACK.layers["VacuumGap_Top"].thickness
"""Default membrane thickness in µm, from the ``VacuumGap_Top`` layer-stack level."""

DRUM_DENSITY: float = get_layer_material_properties("VacuumGap_Top")["density"]
"""Default membrane mass density in kg/m³, from the ``VacuumGap_Top`` material."""

DRUM_MECHANICAL_FREQUENCY: float = 10.69e6
"""Default fundamental drum-mode frequency in Hz.

This is the design value of :cite:`teufelCircuitCavityElectromechanics2011`.
It depends on the film stress, which no formula here fixes, so it is an input.
"""


@jit(inline=True)
def vacuum_gap_capacitance_analytical(
    radius: ArrayLike,
    gap: ArrayLike,
    fringing: ArrayLike = 1.0,
) -> jax.Array:
    r"""Capacitance of a circular vacuum-gap parallel-plate capacitor.

    The parallel-plate term follows
    :cite:`cicakLowlossSuperconductingResonant2010,teufelCircuitCavityElectromechanics2011`.
    The fringing correction is Kirchhoff's asymptotic result for two coaxial
    circular plates of radius :math:`a` at separation :math:`d \ll a`, as given in
    :cite:`norgrenCapacitanceCircularParallel2009` (eq. 38 of the preprint):

    .. math::

        C = \frac{\varepsilon_0 \pi a^2}{d}
            + f\,\varepsilon_0 a \left[\ln\!\left(\frac{16 \pi a}{d}\right) - 1\right]

    where :math:`f` switches the fringing term on (1) or off (0).

    Assumptions: zero-thickness plates of equal radius in vacuum, no ground
    plane nearby. For a 15 µm diameter drum with a 50 nm gap the fringing term
    adds 1.7 %.

    Args:
        radius: Radius :math:`a` of the overlapping plates in µm.
        gap: Vacuum gap :math:`d` in µm.
        fringing: Weight :math:`f` of the fringing term, 1 to include it and 0
            for the bare parallel-plate value.

    Returns:
        The capacitance in F.
    """
    a = jnp.asarray(radius) * 1e-6
    d = jnp.asarray(gap) * 1e-6
    parallel_plate = ε_0 * π * a**2 / d
    fringe = ε_0 * a * (jnp.log(16 * π * a / d) - 1)
    return parallel_plate + jnp.asarray(fringing) * fringe


@jit(inline=True)
def drum_effective_mass(
    radius: ArrayLike,
    thickness: ArrayLike = DRUM_THICKNESS,
    density: ArrayLike = DRUM_DENSITY,
    mass_factor: ArrayLike = 1.0,
) -> jax.Array:
    r"""Effective mass of the fundamental drum mode.

    .. math::

        m_\text{eff} = \mu\,\rho \pi a^2 t

    With the default :math:`\mu = 1` the effective mass is the full disc mass.
    This is the uniform-plate (piston) picture behind
    :math:`G = -\omega_\text{c}/(2d)` in
    :cite:`teufelCircuitCavityElectromechanics2011`. For its 15 µm diameter,
    100 nm thick Al drum the disc mass is 47.7 pg, close to the quoted
    :math:`m \approx 50\,\text{pg}`, and it gives :math:`x_\text{zp} = 4.06\,\text{fm}`
    at 10.69 MHz, matching the quoted 4.1 fm. The paper gives no separate
    mode-shape factor.

    A clamped-membrane mode shape would change both :math:`m_\text{eff}` and
    :math:`\partial C/\partial x`. That correction is not part of the cited
    model; ``mass_factor`` only scales the mass.

    Args:
        radius: Drum radius :math:`a` in µm.
        thickness: Membrane thickness :math:`t` in µm.
        density: Membrane mass density :math:`\rho` in kg/m³.
        mass_factor: Dimensionless effective-mass factor :math:`\mu`.

    Returns:
        The effective mass in kg.
    """
    a = jnp.asarray(radius) * 1e-6
    t = jnp.asarray(thickness) * 1e-6
    return jnp.asarray(mass_factor) * density * π * a**2 * t


@jit(inline=True)
def drum_zero_point_motion(
    effective_mass: ArrayLike,
    mechanical_frequency: ArrayLike = DRUM_MECHANICAL_FREQUENCY,
) -> jax.Array:
    r"""Zero-point motion of a mechanical mode.

    .. math::

        x_\text{zp} = \sqrt{\frac{\hbar}{2 m_\text{eff} \Omega_\text{m}}}

    as used in :cite:`teufelCircuitCavityElectromechanics2011`.

    Args:
        effective_mass: Effective mass :math:`m_\text{eff}` in kg.
        mechanical_frequency: Mechanical frequency :math:`\Omega_\text{m}/2\pi` in Hz.

    Returns:
        The zero-point motion in m.
    """
    omega_m = 2 * π * jnp.asarray(mechanical_frequency)
    ħ = h / (2 * π)
    return jnp.sqrt(ħ / (2 * jnp.asarray(effective_mass) * omega_m))


@jit(inline=True)
def drum_electromechanical_coupling(
    resonance_frequency: ArrayLike,
    gap: ArrayLike = VACUUM_GAP,
    participation: ArrayLike = 1.0,
) -> jax.Array:
    r"""Electromechanical coupling of a drum-capacitor LC resonator.

    For a parallel-plate capacitor :math:`C = \varepsilon_0 A/(d - x)`, where
    :math:`x` is the membrane displacement toward the bottom electrode, and
    :math:`\omega_\text{c} = 1/\sqrt{LC}`,
    :cite:`teufelCircuitCavityElectromechanics2011` gives for :math:`x \ll d`

    .. math::

        G = \frac{\mathrm{d}\omega_\text{c}}{\mathrm{d}x}
          = -\eta \frac{\omega_\text{c}}{2d}

    where :math:`\eta` is the fraction of the total resonator capacitance that
    the drum provides.

    Args:
        resonance_frequency: Resonance frequency :math:`\omega_\text{c}/2\pi` in Hz.
        gap: Vacuum gap :math:`d` in µm.
        participation: Capacitance participation ratio :math:`\eta`.

    Returns:
        :math:`G` in rad·s⁻¹·m⁻¹. Divide by :math:`2\pi` for Hz/m.
    """
    omega_c = 2 * π * jnp.asarray(resonance_frequency)
    d = jnp.asarray(gap) * 1e-6
    return -jnp.asarray(participation) * omega_c / (2 * d)


@jit(inline=True)
def drum_single_photon_coupling(
    resonance_frequency: ArrayLike,
    effective_mass: ArrayLike,
    gap: ArrayLike = VACUUM_GAP,
    participation: ArrayLike = 1.0,
    mechanical_frequency: ArrayLike = DRUM_MECHANICAL_FREQUENCY,
) -> jax.Array:
    r"""Single-photon electromechanical coupling rate of a drum resonator.

    .. math::

        g_0 = |G|\,x_\text{zp}

    with :math:`G` from :func:`drum_electromechanical_coupling` and
    :math:`x_\text{zp}` from :func:`drum_zero_point_motion`
    :cite:`teufelCircuitCavityElectromechanics2011`.

    Args:
        resonance_frequency: Resonance frequency :math:`\omega_\text{c}/2\pi` in Hz.
        effective_mass: Effective mass :math:`m_\text{eff}` in kg.
        gap: Vacuum gap :math:`d` in µm.
        participation: Capacitance participation ratio :math:`\eta`.
        mechanical_frequency: Mechanical frequency :math:`\Omega_\text{m}/2\pi` in Hz.

    Returns:
        :math:`g_0` in rad/s. Divide by :math:`2\pi` for Hz.
    """
    big_g = drum_electromechanical_coupling(
        resonance_frequency=resonance_frequency,
        gap=gap,
        participation=participation,
    )
    x_zp = drum_zero_point_motion(
        effective_mass=effective_mass, mechanical_frequency=mechanical_frequency
    )
    return jnp.abs(big_g) * x_zp


def vacuum_gap_capacitor(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    radius: float = 7.5,
    gap: float = VACUUM_GAP,
    fringing: float = 1.0,
) -> sax.SDict:
    r"""Vacuum-gap drum capacitor SAX model.

    A two-port series capacitor between the bottom electrode (``o1``) and the
    suspended membrane (``o2``), with the capacitance from
    :func:`vacuum_gap_capacitance_analytical`
    :cite:`cicakLowlossSuperconductingResonant2010`.

    .. svgbob::

        o1 ──┤├── o2

    Args:
        f: Array of frequency points in Hz.
        radius: Bottom-electrode radius :math:`a` in µm.
        gap: Vacuum gap :math:`d` in µm. Defaults to the layer-stack value.
        fringing: Weight of the fringing term, 1 to include it and 0 to drop it.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    f_arr = jnp.asarray(f)
    capacitance = vacuum_gap_capacitance_analytical(
        radius=radius, gap=gap, fringing=fringing
    )
    return capacitor(f=f_arr, capacitance=capacitance)


def drum_lc_resonator(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    radius: float = 7.5,
    gap: float = VACUUM_GAP,
    n_turns: int = 71,
    turn_length: float = 300.0,
    inductor_cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    sheet_inductance: float = 0.4e-12,
    coupling_capacitance: float = 1e-15,
    parasitic_capacitance: float = 0.0,
) -> sax.SDict:
    r"""Drum LC resonator side-coupled to a feedline, SAX model.

    A meander inductor :math:`L` is shunted by the drum capacitance
    :math:`C_\text{d}` (plus an optional parasitic :math:`C_\text{p}`) and
    coupled to the feedline through :math:`C_\text{c}`, as in
    :cite:`teufelCircuitCavityElectromechanics2011` but with capacitive instead
    of inductive coupling:

    .. svgbob::

        o1 ──────┬────── o2
                 │
                ─┴─ Cc
                ─┬─
              ┌──┴──┐
              L   Cd+Cp
              └──┬──┘
                GND

    The feedline sees the shunt admittance

    .. math::

        Y = \left[\frac{1}{j\omega C_\text{c}}
            + \frac{1}{j\omega (C_\text{d} + C_\text{p}) + 1/(j\omega L)}\right]^{-1}

    which has its transmission zero at
    :math:`\omega_\text{r} = 1/\sqrt{L (C_\text{d} + C_\text{p} + C_\text{c})}`.
    :math:`C_\text{d}` comes from :func:`vacuum_gap_capacitance_analytical`
    and :math:`L` from :func:`~qpdk.models.inductor.meander_inductor_inductance_analytical`.
    Losses are neglected.

    Args:
        f: Array of frequency points in Hz.
        radius: Drum bottom-electrode radius in µm.
        gap: Vacuum gap in µm. Defaults to the layer-stack value.
        n_turns: Number of meander inductor runs.
        turn_length: Length of each meander run in µm.
        inductor_cross_section: Cross-section of the meander wire.
        sheet_inductance: Sheet inductance per square in H/□.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` to the
            feedline in F.
        parasitic_capacitance: Extra capacitance :math:`C_\text{p}` in parallel
            with the drum in F.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    f_arr = jnp.asarray(f)
    omega = 2 * π * f_arr
    wire_width, wire_gap = _get_wire_width_and_gap(inductor_cross_section, None)
    inductance = meander_inductor_inductance_analytical(
        n_turns=n_turns,
        turn_length=turn_length,
        wire_width=wire_width,
        wire_gap=wire_gap,
        sheet_inductance=sheet_inductance,
    )
    capacitance = (
        vacuum_gap_capacitance_analytical(radius=radius, gap=gap)
        + parasitic_capacitance
    )
    y_lc = 1j * omega * capacitance + 1.0 / (1j * omega * inductance + 1e-25)
    z_branch = 1.0 / (1j * omega * coupling_capacitance + 1e-25) + 1.0 / (y_lc + 1e-25)
    y_branch = 1.0 / (z_branch + 1e-25)
    return shunt_admittance(f=f_arr, y=y_branch)
