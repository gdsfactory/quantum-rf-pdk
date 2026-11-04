r"""Suspended vacuum-gap (drum) capacitor and drum LC resonator models.

A drum is a parallel-plate capacitor whose top electrode is a thin aluminium
membrane suspended a vacuum gap :math:`d` above a bottom electrode
:cite:`cicakLowlossSuperconductingResonant2010`. Shunting it with a meander
inductor gives an LC resonator whose frequency depends on the membrane
position :cite:`teufelCircuitCavityElectromechanics2011`.

The defaults follow the single-drum device of
:cite:`pirkkalainenSqueezingQuantumNoise2015` (Aalto University): a 15 µm
diameter, 120 nm thick Al drum 70 nm above its bottom electrode, at the end of
a meander inductor of about 14 nH, with about 16 fF of other circuit
capacitance, a 6.90 GHz cavity and a 13.03 MHz drum mode.

That device is on quartz (relative permittivity about 4). Its authors note that
silicon or sapphire roughly doubles the stray capacitance, so the 16 fF default
underestimates :math:`C_\text{p}` on this PDK's silicon substrate. The
mechanical defaults are the paper's values for its 15 µm membrane. The
membrane drawn by :func:`qpdk.cells.drum.vacuum_gap_capacitor` spans
:math:`2(a + \text{etch\_gap})`, 19 µm for the default cell, so its mechanics
differ; pass the membrane radius to :func:`drum_effective_mass`.

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
- The Pirkkalainen et al. circuit values (16 fF, 14 nH) are taken from a quartz
  substrate and are not rescaled to silicon.
"""

import warnings
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY, h, ε_0, π
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell
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

DRUM_MECHANICAL_FREQUENCY: float = 13.032e6
"""Default fundamental drum-mode frequency in Hz.

This is the measured value of :cite:`pirkkalainenSqueezingQuantumNoise2015`.
It depends on the film stress, which no formula here fixes, so it is an input.
"""

DRUM_PARASITIC_CAPACITANCE: float = 16e-15
"""Default capacitance in F in parallel with the drum.

The EM-simulated value C ≈ 16 fF of
:cite:`pirkkalainenSqueezingQuantumNoise2015`, the resonator capacitance other
than the drum. It is a property of that layout, not of this PDK's meander.
"""

DRUM_INDUCTOR_TURNS: int = 75
"""Default number of meander runs.

With 300 µm runs of the default meander cross-section and 0.4 pH/□ this gives
an analytical inductance of about 14 nH, the EM-simulated value of
:cite:`pirkkalainenSqueezingQuantumNoise2015`.
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
    :cite:`norgrenCapacitanceCircularParallel2009`:

    .. math::

        C = \frac{\varepsilon_0 \pi a^2}{d}
            + f\,\varepsilon_0 a \left[\ln\!\left(\frac{16 \pi a}{d}\right) - 1\right]

    where :math:`f` switches the fringing term on (1) or off (0).

    Assumptions: zero-thickness plates of equal radius in vacuum, no ground
    plane nearby. For a 15 µm diameter drum with a 70 nm gap the fringing term
    adds 2.2 %. The :func:`~qpdk.cells.drum.vacuum_gap_capacitor` cell draws a
    membrane larger than the bottom electrode, with a grounded ring a few µm
    away, so this term is only an estimate of the fringing there.

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
    This choice is the uniform-plate (piston) picture behind
    :math:`G = -\omega_\text{c}/(2d)` in
    :cite:`teufelCircuitCavityElectromechanics2011`. For its 15 µm diameter,
    100 nm thick Al drum the disc mass is 47.7 pg, close to the quoted
    :math:`m \approx 50\,\text{pg}`, and it gives :math:`x_\text{zp} = 4.06\,\text{fm}`
    at 10.69 MHz, matching the quoted 4.1 fm.

    For the default 120 nm drum of :cite:`pirkkalainenSqueezingQuantumNoise2015`
    the disc mass is 57.3 pg and :math:`x_\text{zp} = 3.4\,\text{fm}` at
    13.03 MHz, against the quoted :math:`x_\text{zp} \sim 5\,\text{fm}`. That
    paper does not state its effective-mass convention, so it does not fix
    :math:`\mu`.

    A clamped-membrane mode shape changes both :math:`m_\text{eff}` and
    :math:`\partial C/\partial x`. That correction is not part of the cited
    model; ``mass_factor`` only scales the mass. Referred to the centre
    displacement, the fundamental :math:`J_0(j_{0,1} r/a)` mode of a clamped
    circular membrane has :math:`\mu = J_1(j_{0,1})^2 \approx 0.27`, which
    gives 6.5 fm for the default drum. Use it only together with a
    mode-weighted :math:`\partial C/\partial x`, see
    :func:`drum_electromechanical_coupling`.

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

    This expression treats the whole membrane as a rigid plate. With the clamped edge,
    only the centre moves fully, so the parallel-plate
    :math:`\partial C/\partial x` overestimates :math:`g_0` by about a factor
    of two :cite:`cattiauxLinearCouplingMicrowave2020`. For the default drum
    the parallel-plate :math:`\varepsilon_0 \pi a^2/d^2` is 319 nF/m, against
    the :math:`\partial C_\text{g}/\partial x \sim 120\,\text{nF/m}` quoted by
    :cite:`pirkkalainenSqueezingQuantumNoise2015`. Scale ``participation`` if
    a mode-weighted value is known.

    The default :math:`\eta = 1` is the limit in which the drum provides all
    of the resonator capacitance, as for Teufel et al. For the default
    :func:`drum_lc_resonator` the drum gives 22.9 fF of 39.9 fF, so
    :math:`\eta \approx 0.57`; pass it explicitly.

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
    radius: sax.FloatArrayLike = 7.5,
    gap: sax.FloatArrayLike = VACUUM_GAP,
    n_turns: int = DRUM_INDUCTOR_TURNS,
    turn_length: float = 300.0,
    inductor_cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    sheet_inductance: float = 0.4e-12,
    coupling_capacitance: sax.FloatArrayLike = 1e-15,
    parasitic_capacitance: sax.FloatArrayLike = DRUM_PARASITIC_CAPACITANCE,
) -> sax.SDict:
    r"""Drum LC resonator side-coupled to a feedline, SAX model.

    A meander inductor :math:`L` is shunted by the drum capacitance
    :math:`C_\text{d}` (plus a parasitic :math:`C_\text{p}`) and coupled to
    the feedline through :math:`C_\text{c}`, as in
    :cite:`pirkkalainenSqueezingQuantumNoise2015`. The drum sits at the open end
    of the meander, and the resonator couples capacitively to a transmission
    line as in :cite:`ockeloenkorppiLowNoiseAmplificationFrequency2016`. Here it
    hangs off a single feedline, which
    :cite:`mercierdelepinayRealizationDirectionalAmplification2019` also uses
    for readout. :cite:`teufelCircuitCavityElectromechanics2011` coupled
    inductively instead.

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

    With :math:`L = 14\,\text{nH}` and :math:`C = 16\,\text{fF}` from
    :cite:`pirkkalainenSqueezingQuantumNoise2015` and the parallel-plate drum
    capacitance of 22.3 fF, :math:`1/(2\pi\sqrt{L(C + C_\text{d})})` is
    6.87 GHz, against the measured 6.90 GHz.

    Limits of this lumped model:

    - The meander's own capacitance to ground is not computed. The default
      ``parasitic_capacitance`` is the 16 fF of
      :cite:`pirkkalainenSqueezingQuantumNoise2015`, which belongs to that
      layout. It lowers :math:`\omega_\text{r}` and the participation
      :math:`\eta` of :func:`drum_electromechanical_coupling`. Pass an
      extracted value for any other layout.
    - The default meander is about 23 mm of wire, which is not short compared
      with the wavelength at the resonance. Distributed effects are neglected.
    - ``coupling_capacitance`` is an input. It is not derived from the
      coupling pad of :func:`~qpdk.cells.drum.drum_lc_resonator`.

    Extract :math:`C_\text{p}` and :math:`C_\text{c}` with an EM solver
    before relying on the resonance frequency of a layout;
    :func:`drum_lc_resonator_fem_model` takes them from the Palace dataset of
    the default cell family.

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
            with the drum in F. Defaults to the 16 fF of
            :cite:`pirkkalainenSqueezingQuantumNoise2015`.

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


# ---------------------------------------------------------------------------
# Layout-backed capacitances from the Palace electrostatic dataset
# ---------------------------------------------------------------------------

DRUM_DATASET = "drum_lc_resonator_palace"
"""Dataset generated by ``qpdk/models/datasets/recipes/drum.py``."""

_DRUM_TERMINALS = ("node", "membrane", "feed")


def drum_capacitance_reduction(
    maxwell: ArrayLike, weights: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Reduce a drum resonator Maxwell matrix to :math:`(C_\text{c}, C_\text{p}, C_\text{d})`.

    The terminals are ``node``, ``membrane``, ``feed`` and the meander blocks,
    in that order. ``weights`` are their voltages relative to the node at the
    lumped resonance: 1 for the node, 0 for the grounded membrane and the
    feedline, and the mean position along the wire for each meander block. With
    :math:`v` the weights and :math:`C` the matrix, the charge on the feed and
    on the membrane gives

    .. math::

        C_\text{c} = -\sum_j C_{\text{feed},j} v_j, \qquad
        C_\text{d} = -\sum_j C_{\text{membrane},j} v_j,

    and the rest of the stored energy, :math:`v^\mathsf{T} C v - C_\text{c} - C_\text{d}`,
    is the parasitic :math:`C_\text{p}` to ground, including the meander's own
    capacitance. The three add up to the total energy, so a meander block at
    weight :math:`v` coupled to the feed adds :math:`v` to :math:`C_\text{c}`
    and :math:`v^2 - v` to :math:`C_\text{p}`.

    Args:
        maxwell: Maxwell capacitance matrices in F, shape ``(..., T, T)``.
        weights: Terminal voltages relative to the node, shape ``(T,)``.

    Returns:
        ``(C_c, C_p, C_d)`` in F with the leading shape of ``maxwell``.
    """
    c = jnp.asarray(maxwell)
    v = jnp.asarray(weights)
    charge = jnp.einsum("...ij,j->...i", c, v)
    c_c = -charge[..., 2]
    c_d = -charge[..., 1]
    return c_c, charge @ v - c_c - c_d, c_d


def _drum_lookup_from(data: Dataset) -> tuple[GridInterpolator, jax.Array]:
    """Check the terminals and weights, then build the interpolator."""
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    terminals = grid.terminals
    if terminals[:3] != _DRUM_TERMINALS or not all(
        name.startswith("meander_") for name in terminals[3:]
    ):
        msg = (
            f"Expected terminals {_DRUM_TERMINALS} then meander blocks, got {terminals}"
        )
        raise ValueError(msg)
    weights = data.metadata.provenance.get("terminal_weights")
    if weights is None or len(weights) != len(terminals):
        msg = "The dataset provenance needs terminal_weights, one per terminal"
        raise ValueError(msg)
    check_maxwell(grid.values)
    return GridInterpolator(grid), jnp.asarray(weights, dtype=float)


@cache
def _cached_drum_lookup(
    location: Path | str,
) -> tuple[GridInterpolator, jax.Array] | None:
    """Open a dataset once per process; the bundled one may not exist yet."""
    try:
        data = Dataset(location)
    except FileNotFoundError:
        if location != DRUM_DATASET:
            raise
        warnings.warn(
            f"Dataset {DRUM_DATASET!r} is not available yet; drum FEM capacitances are NaN.",
            stacklevel=4,
        )
        return None
    return _drum_lookup_from(data)


def _drum_lookup(
    dataset: Dataset | Path | str,
) -> tuple[GridInterpolator, jax.Array] | None:
    """Load the drum Maxwell matrix once, outside numerical evaluation."""
    if isinstance(dataset, Dataset):
        return _drum_lookup_from(dataset)
    return _cached_drum_lookup(dataset)


def _drum_capacitances(
    lookup: tuple[GridInterpolator, jax.Array] | None, **point: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Interpolate and reduce, or return NaN without data."""
    if lookup is None:
        shape = jnp.broadcast_shapes(*(jnp.shape(value) for value in point.values()))
        nan = jnp.full(shape, jnp.nan)
        return nan, nan, nan
    interpolator, weights = lookup
    return drum_capacitance_reduction(interpolator(**point), weights)


def drum_capacitances(
    *,
    coupling_gap: ArrayLike = 6.0,
    coupling_pad_size: ArrayLike = 20.0,
    radius: ArrayLike = 7.5,
    turn_length: ArrayLike = 300.0,
    dataset: Dataset | Path | str = DRUM_DATASET,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Return the extracted capacitances of a :func:`~qpdk.cells.drum_lc_resonator` layout, in F.

    Dimensions are in µm and match the cell arguments; the dataset fixes the
    other cell arguments at their defaults, including 75 meander runs. The
    defaults are the default cell. See :func:`drum_capacitance_reduction` for
    how the Palace Maxwell matrix becomes

    - :math:`C_\text{c}`, the coupling to the feedline,
    - :math:`C_\text{p}`, the parasitic capacitance to ground of the pad,
      wiring and meander, weighted by the meander's linear voltage profile,
    - :math:`C_\text{d}`, the drum capacitance at the stack's vacuum gap,
      including fringing and the electrode lead under the membrane.

    The bundled dataset ``drum_lc_resonator_palace`` holds the 54 solved
    geometries; without it every value is NaN and a warning is issued once.
    Dimensions outside the stored grid also return NaN. Supports JIT, vmap and
    grad.

    Args:
        coupling_gap: Gap between the coupling pad and the feedline in µm.
        coupling_pad_size: Side of the square coupling pad in µm.
        radius: Drum bottom-electrode radius in µm.
        turn_length: Length of each meander run in µm.
        dataset: Dataset or path with terminals ``node``, ``membrane``,
            ``feed`` and ``meander_*`` and ``terminal_weights`` provenance.

    Returns:
        ``(C_c, C_p, C_d)`` in F, each with the broadcast input shape.
    """
    with jax.ensure_compile_time_eval():
        lookup = _drum_lookup(dataset)
    return _drum_capacitances(
        lookup,
        coupling_gap=coupling_gap,
        coupling_pad_size=coupling_pad_size,
        radius=radius,
        turn_length=turn_length,
    )


def drum_lc_resonator_fem_model(
    *, dataset: Dataset | Path | str = DRUM_DATASET
) -> sax.SDictModel:
    r"""Build :func:`drum_lc_resonator` with capacitances from the Palace dataset.

    :math:`C_\text{c}` and :math:`C_\text{p}` come from
    :func:`drum_capacitances` at the layout dimensions. The extracted drum
    capacitance replaces the analytical one at the stack gap, so the model's
    ``parasitic_capacitance`` is
    :math:`C_\text{p} + C_\text{d}^\text{FEM} - C_\text{d}^\text{analytical}(d_0)`.
    A different ``gap`` still moves only the analytical parallel-plate term, so
    membrane motion is modelled as in :func:`drum_lc_resonator` around the
    extracted total.

    The returned model takes ``f`` in Hz, the four layout dimensions of
    :func:`drum_capacitances` in µm, ``gap`` in µm and ``sheet_inductance`` in
    H/□. The meander has the dataset's 75 runs. It returns NaN until the
    dataset exists, and outside the dataset's grid. The lumped-model limits of
    :func:`drum_lc_resonator` still apply: the default meander is electrically
    long at its resonance.

    Args:
        dataset: Dataset or path accepted by :func:`drum_capacitances`.

    Returns:
        Jittable two-port SAX model with ports ``o1`` and ``o2``.
    """
    lookup = _drum_lookup(dataset)

    def model(
        f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
        coupling_gap: sax.FloatArrayLike = 6.0,
        coupling_pad_size: sax.FloatArrayLike = 20.0,
        radius: sax.FloatArrayLike = 7.5,
        turn_length: float = 300.0,
        gap: sax.FloatArrayLike = VACUUM_GAP,
        sheet_inductance: float = 0.4e-12,
    ) -> sax.SDict:
        """Evaluate the lumped drum resonator with extracted capacitances."""
        c_c, c_p, c_d = _drum_capacitances(
            lookup,
            coupling_gap=coupling_gap,
            coupling_pad_size=coupling_pad_size,
            radius=radius,
            turn_length=turn_length,
        )
        parasitic = (
            c_p + c_d - vacuum_gap_capacitance_analytical(radius=radius, gap=VACUUM_GAP)
        )
        return drum_lc_resonator(
            f=f,
            radius=radius,
            gap=gap,
            turn_length=turn_length,
            sheet_inductance=sheet_inductance,
            coupling_capacitance=c_c,
            parasitic_capacitance=parasitic,
        )

    return jit(model)
