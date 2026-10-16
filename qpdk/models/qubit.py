r"""Qubit LC resonator models.

This module provides LC resonator models for superconducting transmon qubits
and coupled qubit systems. The models are based on the standard LC resonator
formulation with appropriate grounding configurations.

For double-pad transmon qubits, we use an ungrounded LC resonator since
the two islands are floating. For shunted transmon qubits, one island is
grounded, so we use a grounded LC resonator.

The helper functions convert between qubit Hamiltonian parameters (charging
energy :math:`E_\text{C}`, Josephson energy :math:`E_\text{J}`, coupling strength :math:`g`) and the
corresponding circuit parameters (capacitance, inductance).

References:
    - :cite:`kochChargeinsensitiveQubitDesign2007a`
    - :cite:`gaoPhysicsSuperconductingMicrowave2008`
"""

import math
from functools import cache

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from sax.models.rf import capacitor, electrical_short, tee

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0, e, h
from qpdk.models.datasets import (
    Dataset,
    GridInterpolator,
    capacitance_model,
    differential_capacitance,
    junction_model,
)
from qpdk.models.generic import lc_resonator, lc_resonator_coupled
from qpdk.models.jit import jit
from qpdk.models.waveguides import straight_shorted


def flipmon_unbonded_model() -> sax.Model:
    """Return the three-electrode electrostatic coupon as a JAX SAX model.

    ``inner_pad``, ``outer_ring`` and ``top_pad`` remain separate ports, all
    referenced to the two chip ground frames. The coupon has two 300 um
    silicon chips and 0.2 um metal. It omits the junction and bonding bump.
    The returned model accepts ``chip_gap``, ``alignment``, ``inner_radius``
    and ``top_radius`` in um, frequency ``f`` in Hz and ``z_ref`` in ohms.
    Alignment shifts the top pad horizontally while ground frames stay fixed.
    Dimensions outside the stored grid return NaN.

    Returns:
        Jittable three-port model retaining all Maxwell capacitances.
    """
    return capacitance_model("flipmon_unbonded_palace")


def flipmon_ideal_bond_model(
    *, inductance: float, junction_capacitance: float = 0.0
) -> sax.Model:
    """Join inner and top pads with an ideal wire, then add a linearized junction.

    Supply the Josephson inductance in H and additional junction capacitance
    in F. The returned JAX model accepts the same geometry and frequency
    arguments as :func:`flipmon_unbonded_model`. Its ports are ``inner_pad``
    and ``outer_ring``, both referenced to chip ground.

    The ideal wire preserves capacitance to ground but adds no bump field or
    inductance. This reduction has no finite-bump validation and describes
    small oscillations, not the nonlinear qubit spectrum. See
    :func:`~qpdk.models.datasets.junction_model` for the circuit convention.

    Returns:
        Jittable two-port model with explicit junction parameters.
    """
    return junction_model(
        "flipmon_unbonded_palace",
        junction=("inner_pad", "outer_ring"),
        inductance=inductance,
        junction_capacitance=junction_capacitance,
        terminal_groups=(("inner_pad", "top_pad"), ("outer_ring",)),
    )


@cache
def _double_pad_capacitance_lookup() -> GridInterpolator:
    """Load the finite-ground pad dataset outside numerical evaluation."""
    return GridInterpolator(
        Dataset("double_pad_transmon_palace").grid(
            "maxwell_capacitance", allow_missing=False
        )
    )


def double_pad_transmon_capacitance(
    *,
    pad_width: float = 250.0,
    pad_height: float = 400.0,
    pad_gap: float = 20.0,
    ground_clearance: float = 200.0,
) -> jax.Array:
    r"""Return the capacitance seen by charge moving between the two pads, in F.

    Both pads float inside a 300 um wide coplanar ground frame on silicon.
    Dimensions are in um. The stored Palace matrix includes pad-to-pad and
    pad-to-ground fields; :func:`~qpdk.models.datasets.differential_capacitance`
    combines them for equal and opposite pad charges.

    The junction and neighbouring devices are omitted. Add the junction's
    capacitance in parallel before computing :math:`E_\text{C}=e^2/(2C)`.
    Dimensions outside the stored grid return NaN. Supports JIT, vmap and grad.

    Returns:
        Differential pad capacitance in F, excluding the junction capacitance.
    """
    with jax.ensure_compile_time_eval():
        lookup = _double_pad_capacitance_lookup()
    return differential_capacitance(
        lookup(
            pad_width=pad_width,
            pad_height=pad_height,
            pad_gap=pad_gap,
            ground_clearance=ground_clearance,
        )
    )


def double_pad_transmon_fem_model(
    *, inductance: float, junction_capacitance: float = 0.0
) -> sax.Model:
    """Build a linearized transmon circuit from the stored two-pad Palace matrix.

    Supply the Josephson inductance in H and additional junction capacitance
    in F. The returned JAX model accepts frequency in Hz, ``z_ref`` in ohms,
    and ``pad_width``, ``pad_height``, ``pad_gap``, ``ground_clearance`` in um.
    Its ``left_pad`` and ``right_pad`` ports both reference the finite ground
    frame. All pad-to-ground and mutual capacitances are retained.

    This describes small oscillations, not the nonlinear qubit spectrum.
    No readout, feedline, package lid or backside ground is included.
    See :func:`~qpdk.models.datasets.junction_model` for the circuit convention.

    Returns:
        Jittable two-port SAX model with explicit junction parameters.
    """
    return junction_model(
        "double_pad_transmon_palace",
        junction=("left_pad", "right_pad"),
        inductance=inductance,
        junction_capacitance=junction_capacitance,
    )


@jit(inline=True)
def ec_to_capacitance(ec_ghz: ArrayLike) -> jax.Array:
    r"""Convert charging energy :math:`E_\text{C}` to total capacitance :math:`C_\Sigma`.

    The charging energy is related to capacitance by:

    .. math::

        E_\text{C} = \frac{e^2}{2 C_\Sigma}

    where :math:`e` is the electron charge.

    Args:
        ec_ghz: Charging energy in GHz.

    Returns:
        Total capacitance in Farads.

    Example:
        >>> C = ec_to_capacitance(0.2)  # 0.2 GHz (200 MHz) charging energy
        >>> print(f"{C * 1e15:.1f} fF")  # ~96 fF
    """
    ec_joules = jnp.asarray(ec_ghz) * 1e9 * h  # Convert GHz to Joules
    return e**2 / (2 * ec_joules)


@jit(inline=True)
def ej_to_inductance(ej_ghz: ArrayLike) -> jax.Array:
    r"""Convert Josephson energy :math:`E_\text{J}` to Josephson inductance :math:`L_\text{J}`.

    The Josephson energy is related to inductance by:

    .. math::

        E_\text{J} = \frac{\Phi_0^2}{4 \pi^2 L_\text{J}} = \frac{(\hbar / 2e)^2}{L_\text{J}}

    This is equivalent to:

    .. math::

        L_\text{J} = \frac{\Phi_0}{2 \pi I_\text{c}}

    where :math:`I_\text{c}` is the critical current and :math:`\Phi_0` is the magnetic flux quantum.

    Args:
        ej_ghz: Josephson energy in GHz.

    Returns:
        Josephson inductance in Henries.

    Example:
        >>> L = ej_to_inductance(20.0)  # 20 GHz Josephson energy
        >>> print(f"{L * 1e9:.2f} nH")  # ~1.0 nH
    """
    ej_joules = jnp.asarray(ej_ghz) * 1e9 * h  # Convert GHz to Joules
    return Φ_0**2 / (4 * math.pi**2 * ej_joules)


el_to_inductance = ej_to_inductance


@jit(inline=True)
def coupling_strength_to_capacitance(
    g_ghz: ArrayLike,
    c_sigma: ArrayLike,
    c_r: ArrayLike,
    f_q_ghz: ArrayLike,
    f_r_ghz: ArrayLike,
) -> jax.Array:
    r"""Convert coupling strength :math:`g` to coupling capacitance :math:`C_\text{c}`.

    In the dispersive limit (:math:`g \ll f_\text{q}, f_\text{r}`), the coupling strength
    can be related to a coupling capacitance via:

    .. math::

        g \approx \frac{1}{2} \frac{C_\text{c}}{\sqrt{C_\Sigma C_\text{r}}} \sqrt{f_\text{q} f_\text{r}}

    Solving for :math:`C_\text{c}`:

    .. math::

        C_\text{c} = \frac{2g}{\sqrt{f_\text{q} f_\text{r}}} \sqrt{C_\Sigma C_\text{r}}

    See :cite:`Savola2023,krantzQuantumEngineersGuide2019` for details.

    Args:
        g_ghz: Coupling strength in GHz.
        c_sigma: Total qubit capacitance in Farads.
        c_r: Total resonator capacitance in Farads.
        f_q_ghz: Qubit frequency in GHz.
        f_r_ghz: Resonator frequency in GHz.

    Returns:
        Coupling capacitance in Farads.

    Example:
        >>> C_c = coupling_strength_to_capacitance(
        ...     g_ghz=0.1,
        ...     c_sigma=100e-15,  # 100 fF
        ...     c_r=50e-15,  # 50 fF
        ...     f_q_ghz=5.0,
        ...     f_r_ghz=7.0,
        ... )
        >>> print(f"{C_c * 1e15:.2f} fF")
    """
    # Use frequencies directly (already in GHz, ratio is dimensionless)
    sqrt_freq = jnp.sqrt(f_q_ghz * f_r_ghz)
    sqrt_c = jnp.sqrt(c_sigma * c_r)
    return 2 * jnp.asarray(g_ghz) / sqrt_freq * sqrt_c


@jit(inline=True)
def fluxonium(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
) -> sax.SDict:
    r"""S-parameter model for a fluxonium qubit.

    A fluxonium qubit consists of a Josephson junction shunted by a large
    superinductance and a capacitor. In the linear regime, this is modeled
    as a parallel LCL circuit.

    For an accurate microwave model reflecting the real physical layout,
    the circuit is treated as ungrounded (floating) with optional symmetric
    parasitic capacitances connecting both ends to ground :cite:`nguyen_blueprint_2019`.

    .. svgbob::

             ┌────── C ──────┐
        o1 ──┼────── Lj ─────┼── o2
             ├────── Ls ─────┤
             │               │
            Cg              Cg
             │               │
            GND             GND

    Note:
        The total shunt capacitance `capacitance` should include the pads,
        junction capacitance, and parasitic meander capacitance. Spurious
        array self-resonance (SR) modes can be phenomenologically modeled
        by adding to the shunt capacitance or using a multimode model.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total shunt capacitance :math:`C_\Sigma` in Farads.
        josephson_inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        superinductance: Superinductance :math:`L_\text{s}` in Henries.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` at each port in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    # Combine the two inductors in parallel: L = (L1 * L2) / (L1 + L2)
    L_total = (josephson_inductance * superinductance) / (
        josephson_inductance + superinductance
    )
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=L_total,
        grounded=False,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def fluxonium_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
) -> sax.SDict:
    """S-parameter model for a fluxonium qubit with bounding box ports.

    This model is the same as :func:`fluxonium`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return fluxonium(
        f=f,
        capacitance=capacitance,
        josephson_inductance=josephson_inductance,
        superinductance=superinductance,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def double_island_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""LC resonator model for a double-island transmon qubit.

    A double-island transmon has two superconducting islands connected by
    Josephson junctions, with both islands floating (not grounded). This is
    modeled as an ungrounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_\text{q} \approx \frac{1}{2\pi} \sqrt{8 E_\text{J} E_\text{C}} - E_\text{C}

    For the LC model, the resonance frequency is:

    .. math::

        f_\text{r} = \frac{1}{2\pi\sqrt{LC}}

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters.

    .. svgbob::

        o1 ──┬──L──┬── o2
             │     │
             └──C──┘

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
        inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` at each port in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=False,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def double_island_transmon_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """LC resonator model for a double-island transmon qubit with bounding box ports.

    This model is the same as :func:`double_island_transmon`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return double_island_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def flipmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""LC resonator model for a flipmon qubit.

    This model is identical to :func:`double_island_transmon`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return double_island_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def flipmon_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """LC resonator model for a flipmon qubit with bounding box ports.

    This model is the same as :func:`flipmon`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return flipmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def shunted_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""LC resonator model for a shunted transmon qubit.

    A shunted transmon has one island grounded and the other island connected
    to the junction. This is modeled as a grounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_\text{q} \approx \frac{1}{2\pi} \sqrt{8 E_\text{J} E_\text{C}} - E_\text{C}

    For the LC model, the resonance frequency is:

    .. math::

        f_\text{r} = \frac{1}{2\pi\sqrt{LC}}

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters.

    .. svgbob::

        o1 ──┬──L──┬──.
             │     │  | "2-port ground"
             └──C──┘  |
                     "o2"

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
        inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` at the floating port in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=True,
        ground_capacitance=ground_capacitance,
    )


@jit(static_argnames=["grounded"])
def transmon_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 1e-9,
    grounded: bool = False,
    ground_capacitance: sax.FloatArrayLike = 0.0,
    coupling_capacitance: sax.FloatArrayLike = 10e-15,
    coupling_inductance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""Coupled transmon qubit model.

    This model extends the basic transmon qubit by adding a coupling network
    consisting of a parallel capacitor and/or inductor. This can represent
    capacitive or inductive coupling between qubits or between a qubit and
    a readout resonator.

    The coupling network is connected in series with the LC resonator:

    .. svgbob::

             +──Lc──+    +──L──+
        o1 ──│      │────|     │─── o2 or grounded o2
             +──Cc──+    +──C──+
                       "LC resonator"

    For capacitive coupling (common for qubit-resonator coupling):
        - Set ``coupling_capacitance`` to the coupling capacitor value
        - Set ``coupling_inductance=0.0``

    For inductive coupling (common for flux-tunable coupling):
        - Set ``coupling_inductance`` to the coupling inductor value
        - Set ``coupling_capacitance=0.0`` (or small value)

    Use :func:`coupling_strength_to_capacitance` to convert from the
    qubit-resonator coupling strength :math:`g` to the coupling capacitance.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
        inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        grounded: If True, the qubit is a shunted transmon (grounded).
            If False, it is a double-pad transmon (ungrounded).
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` in Farads.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` in Farads.
        coupling_inductance: Coupling inductance :math:`L_\text{c}` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        ground_capacitance=ground_capacitance,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=coupling_inductance,
    )


@jit
def fluxonium_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
    coupling_capacitance: float = 10e-15,
    coupling_inductance: float = 0.0,
) -> sax.SDict:
    r"""Coupled fluxonium qubit model.

    This model adds a coupling network to the fluxonium model.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total shunt capacitance :math:`C_\Sigma` in Farads.
        josephson_inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        superinductance: Superinductance :math:`L_\text{s}` in Henries.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` in Farads.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` in Farads.
        coupling_inductance: Coupling inductance :math:`L_\text{c}` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    L_total = (josephson_inductance * superinductance) / (
        josephson_inductance + superinductance
    )
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=L_total,
        grounded=False,
        ground_capacitance=ground_capacitance,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=coupling_inductance,
    )


def fluxonium_with_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
    resonator_length: float = 5000.0,
    resonator_cross_section: str = "cpw",
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    r"""Model for a fluxonium qubit coupled to a quarter-wave resonator.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total shunt capacitance :math:`C_\Sigma` in Farads.
        josephson_inductance: Josephson inductance :math:`L_\text{J}` in Henries.
        superinductance: Superinductance :math:`L_\text{s}` in Henries.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}` in Farads.
        resonator_length: Length of the quarter-wave resonator in µm.
        resonator_cross_section: Cross-section specification for the resonator.
        coupling_capacitance: Coupling capacitance between qubit and resonator in Farads.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    f = jnp.asarray(f)
    resonator = straight_shorted(
        f=f,
        length=resonator_length,
        cross_section=resonator_cross_section,
    )
    qubit = fluxonium(
        f=f,
        capacitance=capacitance,
        josephson_inductance=josephson_inductance,
        superinductance=superinductance,
        ground_capacitance=ground_capacitance,
    )
    coupling_cap = capacitor(f=f, capacitance=coupling_capacitance)
    tee_junction = tee(f=f)

    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "qubit": qubit,
        "coupling_capacitor": coupling_cap,
        "tee": tee_junction,
    }

    connections = {
        "resonator,o1": "tee,o1",
        "tee,o2": "coupling_capacitor,o1",
        "coupling_capacitor,o2": "qubit,o1",
    }

    ports = {
        "o1": "tee,o3",
        "o2": "qubit,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def qubit_with_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    qubit_capacitance: float = 100e-15,
    qubit_inductance: float = 1e-9,
    qubit_grounded: bool = False,
    resonator_length: float = 5000.0,
    resonator_cross_section: str = "cpw",
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    r"""Model for a transmon qubit coupled to a quarter-wave resonator.

    This model corresponds to the layout function
    :func:`~qpdk.cells.derived.transmon_with_resonator.transmon_with_resonator`.

    The model combines:
    - A transmon qubit (LC resonator)
    - A quarter-wave coplanar waveguide resonator
    - A coupling capacitor connecting the qubit to the resonator

    .. svgbob::

                  "quarter-wave resonator"
                  (straight_shorted)
                         │
              o1 ── tee ─┤
                         │
                         +── Cc ── qubit ── o2

    The qubit can be either:
    - A double-island transmon (``qubit_grounded=False``): both islands floating
    - A shunted transmon (``qubit_grounded=True``): one island grounded

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters (:math:`E_\text{C}`, :math:`E_\text{J}`) to circuit parameters.

    Note:
        This function is not JIT-compiled because it depends on :func:`~straight_shorted`,
        which internally uses cpw_parameters for transmission line modeling.

    Args:
        f: Array of frequency points in Hz.
        qubit_capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
            Convert from charging energy using :func:`ec_to_capacitance`.
        qubit_inductance: Josephson inductance :math:`L_\text{J}` in Henries.
            Convert from Josephson energy using :func:`ej_to_inductance`.
        qubit_grounded: If True, the qubit is a shunted transmon (grounded).
            If False, it is a double-island transmon (ungrounded).
        resonator_length: Length of the quarter-wave resonator in µm.
        resonator_cross_section: Cross-section specification for the resonator.
        coupling_capacitance: Coupling capacitance between qubit and resonator in
            Farads. Use :func:`coupling_strength_to_capacitance` to convert from
            qubit-resonator coupling strength :math:`g`.

    Returns:
        sax.SDict: S-parameters dictionary with ports ``o1`` (resonator input)
            and ``o2`` (qubit ground or floating).
    """
    f = jnp.asarray(f)

    # Create instances for circuit composition
    resonator = straight_shorted(
        f=f,
        length=resonator_length,
        cross_section=resonator_cross_section,
    )
    qubit_func = shunted_transmon if qubit_grounded else double_island_transmon
    qubit = qubit_func(
        f=f,
        capacitance=qubit_capacitance,
        inductance=qubit_inductance,
    )
    coupling_cap = capacitor(f=f, capacitance=coupling_capacitance)
    tee_junction = tee(f=f)
    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "qubit": qubit,
        "coupling_capacitor": coupling_cap,
        "tee": tee_junction,
    }

    # Connect: resonator -- tee -- capacitor -- qubit
    # The tee splits the resonator signal to the coupling capacitor
    connections = {
        "resonator,o1": "tee,o1",
        "tee,o2": "coupling_capacitor,o1",
        "coupling_capacitor,o2": "qubit,o1",
    }

    if qubit_grounded:
        instances["qubit_terminator"] = electrical_short(f=f, n_ports=1)
        connections["qubit,o2"] = "qubit_terminator,o1"

    ports = {
        "o1": "tee,o3",  # External port for resonator coupling
    }
    if not qubit_grounded:
        ports["o2"] = "qubit,o2"  # Qubit floating port

    return sax.evaluate_circuit_fg((connections, ports), instances)


def flipmon_with_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    qubit_capacitance: float = 100e-15,
    qubit_inductance: float = 1e-9,
    resonator_length: float = 5000.0,
    resonator_cross_section: str = "cpw",
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    """Model for a flipmon qubit coupled to a quarter-wave resonator.

    This model is identical to :func:`qubit_with_resonator` but the qubit is set to floating.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return qubit_with_resonator(
        f=f,
        qubit_capacitance=qubit_capacitance,
        qubit_inductance=qubit_inductance,
        qubit_grounded=False,  # Flipmon is ungrounded
        resonator_length=resonator_length,
        resonator_cross_section=resonator_cross_section,
        coupling_capacitance=coupling_capacitance,
    )


def double_island_transmon_with_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    qubit_capacitance: float = 100e-15,
    qubit_inductance: float = 1e-9,
    resonator_length: float = 5000.0,
    resonator_cross_section: str = "cpw",
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    """Model for a double-island transmon qubit coupled to a quarter-wave resonator.

    This model is identical to :func:`qubit_with_resonator` but the qubit is set to floating.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return qubit_with_resonator(
        f=f,
        qubit_capacitance=qubit_capacitance,
        qubit_inductance=qubit_inductance,
        qubit_grounded=False,
        resonator_length=resonator_length,
        resonator_cross_section=resonator_cross_section,
        coupling_capacitance=coupling_capacitance,
    )


def transmon_with_resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    qubit_capacitance: float = 100e-15,
    qubit_inductance: float = 1e-9,
    qubit_grounded: bool = False,
    resonator_length: float = 5000.0,
    resonator_cross_section: str = "cpw",
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    """Model for a transmon qubit coupled to a quarter-wave resonator.

    This model is identical to :func:`qubit_with_resonator`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return qubit_with_resonator(
        f=f,
        qubit_capacitance=qubit_capacitance,
        qubit_inductance=qubit_inductance,
        qubit_grounded=qubit_grounded,
        resonator_length=resonator_length,
        resonator_cross_section=resonator_cross_section,
        coupling_capacitance=coupling_capacitance,
    )


@jit(inline=True)
def xmon_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
) -> sax.SDict:
    """LC resonator model for an Xmon style transmon qubit.

    An Xmon transmon is typically shunted, so this model wraps :func:`shunted_transmon`.

    Returns:
        sax.SDict: S-parameters dictionary.
    """
    return shunted_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
    )


@jit(inline=True)
def double_pad_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    inductance: sax.FloatArrayLike = 7e-9,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """LC resonator model for a double-pad transmon qubit.

    Physically the :func:`double_island_transmon` model, with ports renamed to
    ``left_pad`` and ``right_pad`` to match the layout cell ports.

    Returns:
        sax.SDict: S-parameters dictionary with ports left_pad and right_pad.
    """
    return sax.rename_ports(
        double_island_transmon(
            f=f,
            capacitance=capacitance,
            inductance=inductance,
            ground_capacitance=ground_capacitance,
        ),
        {"o1": "left_pad", "o2": "right_pad"},
    )


# Aliases for backward compatibility or to match cell naming
double_pad_transmon_with_bbox = double_island_transmon_with_bbox
double_pad_transmon_with_resonator = double_island_transmon_with_resonator


def flipmon_readout_coupon_model() -> sax.Model:
    """Return the complete static readout coupon as a JAX SAX model.

    The ports include a separate unbonded top pad in addition to
    ``island_a``, ``island_b``, ``readout`` and ``feedline``. Each port
    references chip ground. The returned model accepts ``pad_length``,
    ``pad_offset``, ``feedline_gap`` and ``coupling_length`` in um,
    frequency ``f`` in Hz and ``z_ref`` in ohms.

    The feedline ends are cut open and the entire readout conductor shares
    one potential. This does not describe a shorted resonator mode. Adding
    it to a distributed circuit would duplicate route and ground capacitance.
    No driven/eigenmode or resonance validation is implied.

    Returns:
        Jittable five-port static coupon model; out-of-grid inputs return NaN.
    """
    return capacitance_model("flipmon_readout_coupon_palace")


def transmon_readout_coupon_model() -> sax.Model:
    """Return the complete static readout coupon as a JAX SAX model.

    The ports are
    ``island_a``, ``island_b``, ``readout`` and ``feedline``. Each port
    references chip ground. The returned model accepts ``pad_length``,
    ``pad_offset``, ``feedline_gap`` and ``coupling_length`` in um,
    frequency ``f`` in Hz and ``z_ref`` in ohms.
    Feedline gaps from 14 to 20 um retain a ground strip between the coupled lines.

    The feedline ends are cut open and the entire readout conductor shares
    one potential. This does not describe a shorted resonator mode. Adding
    it to a distributed circuit would duplicate route and ground capacitance.
    No driven/eigenmode or resonance validation is implied.

    Returns:
        Jittable four-port static coupon model; out-of-grid inputs return NaN.
    """
    return capacitance_model("transmon_readout_coupon_palace")


if __name__ == "__main__":
    from typing import TypedDict

    import matplotlib.pyplot as plt

    from qpdk import PDK

    class _QubitConfig(TypedDict):
        """Configuration for a qubit type in the demo plot."""

        label: str
        grounded: bool
        linestyle: str

    PDK.activate()

    f = jnp.linspace(1e9, 10e9, 2001)

    # Calculate bare qubit resonance frequency
    C_q = 100e-15
    L_q = 7e-9
    f_q_bare = 1 / (2 * jnp.pi * jnp.sqrt(L_q * C_q))

    configs: list[_QubitConfig] = [
        {"label": "Shunted Transmon", "grounded": True, "linestyle": "-"},
        {"label": "Double Island Transmon", "grounded": False, "linestyle": "--"},
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for config in configs:
        S_coupled = qubit_with_resonator(
            f=f,
            qubit_capacitance=C_q,
            qubit_inductance=L_q,
            qubit_grounded=config["grounded"],
            resonator_length=5000.0,
            resonator_cross_section="cpw",
            coupling_capacitance=20e-15,
        )

        s11 = S_coupled["o1", "o1"]
        s11_mag = 20 * jnp.log10(jnp.abs(s11))
        s11_phase = jnp.unwrap(jnp.angle(s11))

        # Plot S11 magnitude
        ax1.plot(
            f / 1e9,
            s11_mag,
            label=f"$|S_{{11}}|$ {config['label']}",
            linestyle=config["linestyle"],
        )

        # Plot S11 phase
        ax2.plot(
            f / 1e9,
            s11_phase,
            label=f"$\\angle S_{{11}}$ {config['label']}",
            linestyle=config["linestyle"],
        )

    for ax in (ax1, ax2):
        ax.axvline(
            f_q_bare / 1e9,
            color="r",
            linestyle=":",
            label=rf"Bare Qubit ($f_\text{{q}} = {f_q_bare / 1e9:.3f}\,\text{{GHz}}$)",
        )
        ax.grid(True)
        ax.legend()

    ax2.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax2.set_ylabel("Phase (rad)")
    fig.suptitle(
        rf"Qubit Coupled to Quarter-Wave Resonator ($C_\text{{q}} = {C_q * 1e15:.0f}\,\text{{fF}}$, $L_\text{{q}} = {L_q * 1e9:.0f}\,\text{{nH}}$)"
    )
    plt.tight_layout()
    plt.show()
