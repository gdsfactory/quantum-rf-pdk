r"""Giant-atom transmon models.

A giant atom couples to a waveguide at several points that are spaced by a
distance comparable to the wavelength :cite:`kockumDesigningFrequencydependentRelaxation2014`.
Interference between the coupling points makes the decay rate and the Lamb
shift depend on frequency, as demonstrated with transmons by
:cite:`kannanWaveguideQuantumElectrodynamics2020` and
:cite:`vadirajEngineeringLevelStructure2021`.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY, c_0, π
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.generic import capacitor, tee
from qpdk.models.qubit import shunted_transmon
from qpdk.models.waveguides import straight

__all__ = [
    "giant_atom_decay_rate_and_lamb_shift",
    "giant_atom_transmon",
]


def _coupling_arrays(
    coupling_capacitances: Sequence[float] | ArrayLike,
    segment_lengths: Sequence[float] | ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Return the coupling capacitances and segment lengths as 1D arrays.

    Raises:
        ValueError: If there is not exactly one segment fewer than coupling points.
    """
    capacitances = jnp.atleast_1d(jnp.asarray(coupling_capacitances, dtype=float))
    lengths = jnp.atleast_1d(jnp.asarray(segment_lengths, dtype=float))
    if capacitances.ndim != 1 or lengths.ndim != 1:
        msg = "coupling_capacitances and segment_lengths must be one-dimensional."
        raise ValueError(msg)
    if lengths.shape[0] != capacitances.shape[0] - 1:
        msg = (
            f"Got {capacitances.shape[0]} coupling points and "
            f"{lengths.shape[0]} segment lengths; there must be one segment "
            "fewer than coupling points."
        )
        raise ValueError(msg)
    return capacitances, lengths


def giant_atom_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    coupling_capacitances: tuple[float, ...] = (8e-15, 8e-15),
    segment_lengths: tuple[float, ...] = (13100.0,),
    qubit_capacitance: float = 100e-15,
    qubit_inductance: float = 7e-9,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""S-parameter model of a transmon coupled to a CPW at :math:`K` points.

    The circuit is a SAX netlist of existing models. The waveguide is a chain
    of :func:`~qpdk.models.waveguides.straight` sections joined by ideal
    tees. Each tee branches through a coupling
    :func:`~qpdk.models.generic.capacitor` :math:`C_k` to one qubit node,
    which is grounded through the linear transmon model
    :func:`~qpdk.models.qubit.shunted_transmon`.

    .. svgbob::

        o1 ──┬── ℓ_1 ──┬── ··· ──┬── o2
             │         │         │
            C_1       C_2       C_K
             │         │         │
             └─────────┴────┬────┘
                            │
                        "L_J ∥ C_q"
                            │
                           GND

    Weak coupling and a lossless line give the decay rate of
    :cite:`kockumDesigningFrequencydependentRelaxation2014` (Eq. 15):

    .. math::

        \Gamma(\omega) = \sum_{j,k} \sqrt{\gamma_j \gamma_k} \cos(\omega \tau_{jk}),

    where :math:`\tau_{jk}` is the travel time between points :math:`j` and
    :math:`k`, and :math:`\gamma_k` is the decay rate of point :math:`k` alone.
    The transmission therefore shows a dip of width :math:`\Gamma`, see
    :func:`giant_atom_decay_rate_and_lamb_shift`.

    With a single coupling point the model is a small atom side-coupled to
    the line.

    Note:
        ``segment_lengths`` is measured along the waveguide between the
        centres of neighbouring coupling points. Ports ``o1`` and ``o2`` sit at
        the first and last coupling points.

    Args:
        f: Array of frequency points in Hz.
        coupling_capacitances: Coupling capacitance :math:`C_k` at each of the
            :math:`K` coupling points in F.
        segment_lengths: Waveguide length between consecutive coupling points
            in µm, :math:`K - 1` values.
        qubit_capacitance: Capacitance :math:`C_\text{q}` of the island to
            ground in F, without the coupling capacitances.
        qubit_inductance: Josephson inductance :math:`L_\text{J}` in H.
        cross_section: Cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters with the waveguide ports ``o1`` and ``o2``.
    """
    f = jnp.asarray(f)
    capacitances, lengths = _coupling_arrays(coupling_capacitances, segment_lengths)
    n_points = capacitances.shape[0]

    # ``shunted_transmon`` already grounds the LC internally; its ``o2`` is an
    # isolated ideal short and is left out of the circuit. Terminating it with
    # another short would close a lossless |Γ| = 1 loop and return NaN.
    instances: dict[str, sax.SType] = {
        "qubit": shunted_transmon(
            f=f, capacitance=qubit_capacitance, inductance=qubit_inductance
        ),
    }
    connections: dict[str, str] = {}

    for k in range(n_points):
        instances[f"line_tee_{k}"] = tee(f=f)
        instances[f"coupler_{k}"] = capacitor(f=f, capacitance=capacitances[k])
        connections[f"line_tee_{k},o3"] = f"coupler_{k},o1"
    for k in range(n_points - 1):
        instances[f"segment_{k}"] = straight(
            f=f, length=lengths[k], cross_section=cross_section
        )
        connections[f"line_tee_{k},o2"] = f"segment_{k},o1"
        connections[f"segment_{k},o2"] = f"line_tee_{k + 1},o1"

    # Join the K coupler outputs and the qubit at one node with K - 1 tees.
    node = "coupler_0,o2"
    for k in range(1, n_points):
        instances[f"node_tee_{k}"] = tee(f=f)
        connections[node] = f"node_tee_{k},o1"
        connections[f"coupler_{k},o2"] = f"node_tee_{k},o2"
        node = f"node_tee_{k},o3"
    connections[node] = "qubit,o1"

    ports = {"o1": "line_tee_0,o1", "o2": f"line_tee_{n_points - 1},o2"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


def giant_atom_decay_rate_and_lamb_shift(
    f: ArrayLike,
    coupling_capacitances: tuple[float, ...] = (8e-15, 8e-15),
    segment_lengths: tuple[float, ...] = (13100.0,),
    qubit_capacitance: float = 100e-15,
    cross_section: CrossSectionSpec = "cpw",
    characteristic_impedance: float = 50.0,
) -> tuple[jax.Array, jax.Array]:
    r"""Analytic decay rate and Lamb shift of a giant transmon.

    Implements Eqs. (15) and (16) of
    :cite:`kockumDesigningFrequencydependentRelaxation2014` for :math:`K`
    coupling points at positions :math:`x_k` along the waveguide:

    .. math::

        \Gamma(\omega) &= \sum_{j,k} \sqrt{\gamma_j \gamma_k}
            \cos(\omega \tau_{jk}), \\
        \Delta(\omega) &= \sum_{j<k} \sqrt{\gamma_j \gamma_k}
            \sin(\omega \tau_{jk}),

    with :math:`\tau_{jk} = |x_j - x_k| / v` and
    :math:`v = c_0 / \sqrt{\varepsilon_\text{eff}}` for the CPW. The Lamb
    shift :math:`\Delta` adds to the transition frequency. For one coupling
    point, :math:`\Gamma = \gamma_1` and :math:`\Delta = 0`.

    The rate of a single capacitive coupling point follows from the real part
    of the admittance that the line loads onto the qubit,
    :math:`\Gamma = \operatorname{Re} Y / C_\Sigma`
    :cite:`houckControllingSpontaneousEmission2008`. The line looks like
    :math:`Z_0 / 2` from a side-coupling point, because the qubit radiates into
    both directions:

    .. math::

        \gamma_k = \frac{\omega^2 C_k^2 Z_0}{2 C_\Sigma}, \qquad
        C_\Sigma = C_\text{q} + \sum_k C_k.

    Assumptions:

    - Weak coupling, :math:`\gamma_k \ll \omega` and
      :math:`\omega C_k Z_0 \ll 1`.
    - Markovian: the travel time between the outermost points is much shorter
      than :math:`1 / \Gamma`, so :math:`\Gamma` and :math:`\Delta` are
      evaluated at the transition frequency.
    - Zero temperature, a lossless and matched waveguide, a two-level
      (linear) qubit near its transition, and no reflections at the coupling
      points.
    - The paper takes :math:`\gamma_k` as frequency independent. Here it
      scales as :math:`\omega^2`, as for a capacitive coupling and as used by
      :cite:`kannanWaveguideQuantumElectrodynamics2020`.

    Note:
        qpdk's :func:`~qpdk.models.waveguides.straight` is matched to its own
        impedance. Inside a netlist whose lumped elements use the 50 Ω port
        reference, the line therefore acts as a 50 Ω line. The default
        ``characteristic_impedance`` keeps this function consistent with
        :func:`giant_atom_transmon`.

    Args:
        f: Transition frequency in Hz at which :math:`\Gamma` and
            :math:`\Delta` are evaluated.
        coupling_capacitances: Coupling capacitance :math:`C_k` at each
            coupling point in F.
        segment_lengths: Waveguide length between consecutive coupling points
            in µm, :math:`K - 1` values.
        qubit_capacitance: Capacitance :math:`C_\text{q}` of the island to
            ground in F, without the coupling capacitances.
        cross_section: Cross-section of the waveguide, which sets
            :math:`\varepsilon_\text{eff}`.
        characteristic_impedance: Line impedance :math:`Z_0` in Ω.

    Returns:
        ``(Γ, Δ)``: the decay rate and the Lamb shift as angular frequencies in
        rad/s, with the shape of ``f``. Divide by :math:`2\pi` for Hz.
    """
    capacitances, lengths = _coupling_arrays(coupling_capacitances, segment_lengths)
    ω = 2 * π * jnp.asarray(f)[..., None, None]

    width, gap = get_cpw_dimensions(cross_section)
    ep_eff, _ = cpw_parameters(width, gap)
    velocity = c_0 / jnp.sqrt(jnp.real(ep_eff))

    positions = jnp.concatenate([jnp.zeros(1), jnp.cumsum(lengths)]) * 1e-6
    τ = jnp.abs(positions[:, None] - positions[None, :]) / velocity

    c_sigma = qubit_capacitance + jnp.sum(capacitances)
    sqrt_γγ = (
        ω**2
        * characteristic_impedance
        * capacitances[:, None]
        * capacitances[None, :]
        / (2 * c_sigma)
    )
    upper = jnp.triu(jnp.ones_like(τ), k=1)

    Γ = jnp.sum(sqrt_γγ * jnp.cos(ω * τ), axis=(-2, -1))
    Δ = jnp.sum(upper * sqrt_γγ * jnp.sin(ω * τ), axis=(-2, -1))
    return Γ, Δ
