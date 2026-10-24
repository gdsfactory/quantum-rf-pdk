r"""Gatemon qubit models.

This module provides a SAX microwave S-parameter model and a quantum
Hamiltonian model for the gatemon qubit.

A gatemon is a transmon whose Josephson junction is a gate-tunable weak link
made from a quantum conductor, such as a semiconducting nanowire, graphene or
a carbon nanotube :cite:`riechertCarbonNanotubeGatemon2025`. The weak link
carries supercurrent through a few conduction channels :math:`p` with
transmissions :math:`\tau_p \in [0, 1]`, which a gate voltage tunes.

**Junction potential** — In the short-junction limit, each channel hosts a
pair of Andreev bound states at energies :math:`\pm E_p(\varphi)` with

.. math::

    E_p(\varphi) = \Delta \sqrt{1 - \tau_p \sin^2(\varphi / 2)}

where :math:`\Delta` is the induced superconducting gap. In the even-parity
ground state the junction potential is

.. math::

    U(\varphi) = -\sum_p E_p(\varphi)
               = -\Delta \sum_p \sqrt{1 - \tau_p \sin^2(\varphi / 2)}

In the tunnel limit :math:`\tau_p \to 0` this reduces to
:math:`-E_\text{J} \cos\varphi` plus a constant, with
:math:`E_\text{J} = \Delta \sum_p \tau_p / 4`. At high transmission the
potential contains higher harmonics :math:`\cos(k \varphi)`, which reduces
the magnitude of the anharmonicity below the transmon value
:math:`E_\text{C}` :cite:`kringhojAnharmonicitySuperconductingQubit2018`.

**Hamiltonian model** — The gatemon Hamiltonian

.. math::

    \hat{H} = 4 E_\text{C} (\hat{n} - n_\text{g})^2 + U(\hat{\varphi})

is diagonalised in the charge basis, see :func:`gatemon_hamiltonian`.

**Microwave model** — :func:`gatemon` linearises the junction around
:math:`\varphi = 0`, see its docstring.

Parity switching, quasiparticle poisoning and the gate-voltage dependence of
:math:`\tau_p` are not modelled.

References:
    - :cite:`riechertCarbonNanotubeGatemon2025`
    - :cite:`kringhojAnharmonicitySuperconductingQubit2018`
"""

import math
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from scipy.constants import Boltzmann

from qpdk.models.constants import DEFAULT_FREQUENCY, e, h
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell
from qpdk.models.generic import tee
from qpdk.models.jit import jit
from qpdk.models.qubit import ej_to_inductance, shunted_transmon
from qpdk.models.waveguides import straight

__all__ = [
    "andreev_bound_state_energies",
    "gatemon",
    "gatemon_capacitances",
    "gatemon_coupler",
    "gatemon_energies",
    "gatemon_fem_model",
    "gatemon_frequency_and_anharmonicity",
    "gatemon_hamiltonian",
    "gatemon_island",
    "gatemon_josephson_energy",
    "gatemon_potential",
    "gatemon_supercurrent",
]

DEFAULT_DELTA_GHZ = 50.0
"""Default induced gap :math:`\\Delta` in GHz (about 207 µeV)."""

DEFAULT_TRANSMISSIONS: tuple[float, ...] = (0.8,)
"""Default channel transmissions: a single channel, as in a carbon nanotube."""

TransmissionsLike = Sequence[float] | ArrayLike
"""One transmission per conduction channel, each in :math:`[0, 1]`."""

# ---------------------------------------------------------------------------
# Junction: Andreev bound states, potential and supercurrent
# ---------------------------------------------------------------------------


@jit(inline=True)
def andreev_bound_state_energies(
    phi: ArrayLike,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
) -> jax.Array:
    r"""Compute the positive Andreev bound state energies of a short junction.

    .. math::

        E_p(\varphi) = \Delta \sqrt{1 - \tau_p \sin^2(\varphi / 2)}

    Args:
        phi: Superconducting phase difference :math:`\varphi` in radians.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.

    Returns:
        Array of shape ``(*jnp.shape(phi), n_channels)`` with the energies in GHz.
    """
    phi = jnp.asarray(phi)[..., None]
    tau = jnp.ravel(jnp.asarray(transmissions, dtype=float))
    return jnp.asarray(delta_ghz) * jnp.sqrt(1 - tau * jnp.sin(phi / 2) ** 2)


@jit(inline=True)
def gatemon_potential(
    phi: ArrayLike,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
) -> jax.Array:
    r"""Compute the even-parity junction potential of a gatemon.

    .. math::

        U(\varphi) = -\Delta \sum_p \sqrt{1 - \tau_p \sin^2(\varphi / 2)}

    Args:
        phi: Superconducting phase difference :math:`\varphi` in radians.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.

    Returns:
        Potential energy in GHz, with the shape of ``phi``.
    """
    return -jnp.sum(andreev_bound_state_energies(phi, delta_ghz, transmissions), -1)


@jit(inline=True)
def gatemon_josephson_energy(
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
) -> jax.Array:
    r"""Compute the effective Josephson energy of a gatemon junction.

    The effective Josephson energy is the curvature of the junction potential
    at :math:`\varphi = 0`,

    .. math::

        E_\text{J} = \left.\frac{\partial^2 U}{\partial \varphi^2}\right|_{\varphi=0}
                   = \frac{\Delta}{4} \sum_p \tau_p ,

    which equals the Josephson energy of the :math:`-E_\text{J}\cos\varphi`
    potential obtained in the tunnel limit :math:`\tau_p \to 0`.

    Args:
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.

    Returns:
        Effective Josephson energy in GHz.
    """
    tau = jnp.ravel(jnp.asarray(transmissions, dtype=float))
    return jnp.asarray(delta_ghz) * jnp.sum(tau) / 4


@jit(inline=True)
def gatemon_supercurrent(
    phi: ArrayLike,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    temperature: ArrayLike = 0.0,
) -> jax.Array:
    r"""Compute the current-phase relation of a gatemon junction.

    .. math::

        I(\varphi) = -\frac{2e}{\hbar} \sum_p
            \tanh\left(\frac{E_p}{2 k_\text{B} T}\right)
            \frac{\partial E_p}{\partial \varphi}

    At :math:`T = 0` this is :math:`(2e/\hbar)\,\partial U/\partial\varphi`.
    The critical current of a single ballistic channel
    (:math:`\tau = 1`, :math:`T = 0`) is :math:`e\Delta/\hbar`, reached as
    :math:`\varphi \to \pi`. Exactly at :math:`\tau = 1`, :math:`\varphi = \pi`
    the current is discontinuous and this function returns 0.

    Args:
        phi: Superconducting phase difference :math:`\varphi` in radians.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.
        temperature: Temperature :math:`T` in Kelvin.

    Returns:
        Supercurrent in Amperes, with the shape of ``phi``.
    """
    energies = andreev_bound_state_energies(phi, delta_ghz, transmissions)
    phi = jnp.asarray(phi)[..., None]
    tau = jnp.ravel(jnp.asarray(transmissions, dtype=float))
    # dE_p/dφ = -Δ τ_p sin φ / (4 sqrt(1 - τ_p sin²(φ/2))) = -Δ² τ_p sin φ / (4 E_p)
    safe_energies = jnp.where(energies > 0, energies, 1.0)
    d_energies = jnp.where(
        energies > 0,
        -(jnp.asarray(delta_ghz) ** 2) * tau * jnp.sin(phi) / (4 * safe_energies),
        0.0,
    )
    # tanh(E / 2 k_B T) with E in GHz, equal to 1 at T = 0
    temperature = jnp.asarray(temperature, dtype=float)
    safe_temperature = jnp.where(temperature > 0, temperature, 1.0)
    occupation = jnp.where(
        temperature > 0,
        jnp.tanh(energies * 1e9 * h / (2 * Boltzmann * safe_temperature)),
        1.0,
    )
    # (2e/ħ) × h × 1e9 converts GHz to Amperes: 4π e × 1e9
    return -4 * math.pi * e * 1e9 * jnp.sum(occupation * d_energies, -1)


# ---------------------------------------------------------------------------
# Quantum Hamiltonian model
# ---------------------------------------------------------------------------


def _n_phi(n_max: int) -> int:
    r"""Return the number of phase samples for the Fourier expansion of U(φ).

    The Hamiltonian needs harmonics up to ``2 * n_max``; oversampling by a
    factor of four keeps aliasing of the slowly decaying harmonics at
    :math:`\tau \to 1` negligible.
    """
    return max(512, 8 * (2 * n_max + 1))


@jit(static_argnames=("n_max",))
def gatemon_hamiltonian(
    ec_ghz: ArrayLike = 0.26,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    n_g: ArrayLike = 0.0,
    n_max: int = 30,
) -> jax.Array:
    r"""Build the gatemon Hamiltonian matrix in the charge basis.

    .. math::

        \hat{H} = 4 E_\text{C} (\hat{n} - n_\text{g})^2 + U(\hat{\varphi}),
        \qquad
        U(\varphi) = -\Delta \sum_p \sqrt{1 - \tau_p \sin^2(\varphi / 2)}

    The potential is :math:`2\pi`-periodic, so it is expanded numerically in
    Fourier harmonics :math:`U(\varphi) = \sum_k U_k e^{i k \varphi}` with an
    FFT. Since :math:`e^{i k \hat{\varphi}}` shifts the charge by :math:`k`
    Cooper pairs, the matrix elements are
    :math:`H_{nm} = 4 E_\text{C} (n - n_\text{g})^2 \delta_{nm} + U_{|n-m|}`.
    For a tunnel junction only :math:`U_{\pm 1} = -E_\text{J}/2` survives,
    recovering the transmon Hamiltonian of
    :cite:`kochChargeinsensitiveQubitDesign2007a`. Higher transmissions add
    longer-range couplings :cite:`kringhojAnharmonicitySuperconductingQubit2018`.

    Args:
        ec_ghz: Charging energy :math:`E_\text{C}` in GHz. The default is the
            value of device A in :cite:`riechertCarbonNanotubeGatemon2025`.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.
        n_g: Offset charge :math:`n_\text{g}` in units of :math:`2e`.
        n_max: Charge basis truncation; the basis is
            :math:`n \in \{-n_{\max}, \dots, n_{\max}\}`.

    Returns:
        A ``(2*n_max+1, 2*n_max+1)`` real symmetric matrix in GHz.
    """
    n_phi = _n_phi(n_max)
    phi = 2 * jnp.pi * jnp.arange(n_phi) / n_phi
    # U is real and even in φ, so its Fourier coefficients are real and even
    harmonics = (
        jnp.fft.rfft(gatemon_potential(phi, delta_ghz, transmissions)).real / n_phi
    )

    n = jnp.arange(-n_max, n_max + 1)
    distance = jnp.abs(n[:, None] - n[None, :])
    return jnp.diag(4 * ec_ghz * (n - n_g) ** 2) + harmonics[distance]


@jit(static_argnames=("n_max", "n_levels"))
def gatemon_energies(
    ec_ghz: ArrayLike = 0.26,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    n_g: ArrayLike = 0.0,
    n_max: int = 30,
    n_levels: int = 5,
) -> jax.Array:
    r"""Compute the lowest energy levels of the gatemon Hamiltonian.

    Diagonalises :func:`gatemon_hamiltonian` and returns the ``n_levels``
    lowest eigenvalues relative to the ground state.

    Args:
        ec_ghz: Charging energy :math:`E_\text{C}` in GHz.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.
        n_g: Offset charge :math:`n_\text{g}` in units of :math:`2e`.
        n_max: Charge basis truncation, see :func:`gatemon_hamiltonian`.
        n_levels: Number of levels to return.

    Returns:
        Array of shape ``(n_levels,)`` with energies in GHz, ground state at zero.
    """
    eigenvalues = jnp.linalg.eigvalsh(
        gatemon_hamiltonian(ec_ghz, delta_ghz, transmissions, n_g, n_max)
    )
    return (eigenvalues - eigenvalues[0])[:n_levels]


@jit(static_argnames=("n_max",))
def gatemon_frequency_and_anharmonicity(
    ec_ghz: ArrayLike = 0.26,
    delta_ghz: ArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    n_g: ArrayLike = 0.0,
    n_max: int = 30,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute the qubit frequency and anharmonicity of a gatemon.

    .. math::

        f_{01} = E_1 - E_0, \qquad \alpha = f_{12} - f_{01} = E_2 - 2 E_1

    with :math:`E_0 = 0`. The anharmonicity is negative. For a tunnel
    junction with :math:`E_\text{J} \gg E_\text{C}`,
    :math:`\alpha \approx -E_\text{C}`. For a single channel near
    :math:`\tau = 1` its magnitude drops towards :math:`E_\text{C}/4`
    :cite:`kringhojAnharmonicitySuperconductingQubit2018`.

    Args:
        ec_ghz: Charging energy :math:`E_\text{C}` in GHz.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Channel transmissions :math:`\tau_p`.
        n_g: Offset charge :math:`n_\text{g}` in units of :math:`2e`.
        n_max: Charge basis truncation, see :func:`gatemon_hamiltonian`.

    Returns:
        Tuple of ``(f01_ghz, alpha_ghz)``.

    Example:
        >>> f01, alpha = gatemon_frequency_and_anharmonicity(
        ...     ec_ghz=0.26, delta_ghz=50.0, transmissions=(0.8,)
        ... )
    """
    energies = gatemon_energies(ec_ghz, delta_ghz, transmissions, n_g, n_max, 3)
    return energies[1], energies[2] - 2 * energies[1]


# ---------------------------------------------------------------------------
# SAX microwave model
# ---------------------------------------------------------------------------


@jit(inline=True)
def gatemon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 100e-15,
    delta_ghz: sax.FloatArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    ground_capacitance: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    r"""LC resonator model for a grounded gatemon qubit.

    The island is shunted to ground through the junction, so this model wraps
    :func:`~qpdk.models.qubit.shunted_transmon`. The junction is linearised
    around :math:`\varphi = 0`: its inductance is

    .. math::

        L_\text{J} = \left(\frac{\Phi_0}{2\pi}\right)^2 \frac{1}{E_\text{J}},
        \qquad
        E_\text{J} = \left.\frac{\partial^2 U}{\partial \varphi^2}\right|_{\varphi=0}
                   = \frac{\Delta}{4} \sum_p \tau_p ,

    see :func:`gatemon_josephson_energy`. This small-signal model has no
    anharmonicity; use :func:`gatemon_frequency_and_anharmonicity` for the
    qubit spectrum. The defaults give :math:`E_\text{J} = 10` GHz.

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance :math:`C_\Sigma` of the island in Farads.
        delta_ghz: Induced superconducting gap :math:`\Delta` in GHz.
        transmissions: Gate-tuned channel transmissions :math:`\tau_p`.
        ground_capacitance: Parasitic capacitance to ground :math:`C_\text{g}`
            at the floating port in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return shunted_transmon(
        f=f,
        capacitance=capacitance,
        inductance=ej_to_inductance(gatemon_josephson_energy(delta_ghz, transmissions)),
        ground_capacitance=ground_capacitance,
    )


@jit(inline=True)
def gatemon_island(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: sax.FloatArrayLike = 49.1e-15,
    gate_capacitance: sax.FloatArrayLike = 20e-15,
    delta_ghz: sax.FloatArrayLike = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    gate_ground_capacitance: sax.FloatArrayLike = 35e-15,
) -> sax.SDict:
    r"""Model the island and gate terminals used by the gatemon layout.

    The ``readout`` port touches the island itself. Its coupling capacitor is
    in :func:`gatemon_coupler`, not in this model. The island has a grounded
    Josephson inductance and a shunt capacitance, with a mutual capacitance
    to ``gate``. The nominal 49.1 fF shunt, 20 fF gate and external 5.4 fF
    readout capacitance give :math:`E_\text{C}/h \approx 260\,\text{MHz}`.
    Each contact has 18 square micrometres of top-gate overlap. With the
    nominal 30 nm hBN and relative permittivity 3.4, the parallel-plate
    estimate is about 18 fF per contact. Including the M1 feed gives rounded
    estimates of 20 fF to the island and 35 fF to ground. The island shunt
    retains the charging-energy target. A full hybrid extraction and measured
    flake thickness are needed to replace these estimates.

    Args:
        f: Positive frequency in Hz.
        capacitance: Island capacitance to ground in F, excluding mutual terms.
        gate_capacitance: Island-to-gate capacitance in F.
        delta_ghz: Induced superconducting gap in GHz.
        transmissions: Gate-tuned channel transmissions. Zero opens the junction.
        gate_ground_capacitance: Gate capacitance to ground in F, excluding the mutual term.

    Returns:
        Jittable two-port S-parameters named readout and gate, referenced to 50 ohm.
    """
    omega = 2 * jnp.pi * jnp.asarray(f)
    mutual = 1j * omega * gate_capacitance
    inverse_l = 1 / ej_to_inductance(gatemon_josephson_energy(delta_ghz, transmissions))
    island = 1j * omega * capacitance + inverse_l / (1j * omega)
    y = jnp.stack(
        [
            jnp.stack([island + mutual, -mutual], -1),
            jnp.stack([-mutual, mutual + 1j * omega * gate_ground_capacitance], -1),
        ],
        -2,
    )
    eye = jnp.eye(2)
    values = jnp.linalg.solve(eye + 50 * y, eye - 50 * y)
    ports = ("readout", "gate")
    return {
        (a, b): values[..., i, j]
        for i, a in enumerate(ports)
        for j, b in enumerate(ports)
    }


def gatemon_coupler(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 600.0,
    coupling_position: sax.Float = 200.0,
    coupling_capacitance: sax.FloatArrayLike = 5.4e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""Model a complete CPW section with a capacitive branch to the island.

    Split the distributed line at ``coupling_position`` and connect the
    lumped coupling capacitor there. Both line lengths contribute to the
    resonator. The nominal capacitance gives roughly 50 MHz coupling near
    the paper's readout frequency; it is not a geometric extraction.

    Args:
        f: Frequency in Hz.
        length: Complete coupling CPW length in µm.
        coupling_position: Capacitive tap measured from o1 in µm.
        coupling_capacitance: CPW-to-island capacitance in F.
        cross_section: Cross-section of the CPW.

    Returns:
        Three-port S-parameters named o1, o2 and qubit.
    """
    normalized = 2j * jnp.pi * jnp.asarray(f) * coupling_capacitance * 50
    reflection = 1 / (1 + 2 * normalized)
    transmission = 2 * normalized * reflection
    branch = {
        ("o1", "o1"): reflection,
        ("o2", "o2"): reflection,
        ("o1", "o2"): transmission,
        ("o2", "o1"): transmission,
    }
    instances: dict[str, sax.SType] = {
        "left": straight(f=f, length=coupling_position, cross_section=cross_section),
        "right": straight(
            f=f, length=length - coupling_position, cross_section=cross_section
        ),
        "tap": tee(f=f),
        "cap": branch,
    }
    connections = {"left,o2": "tap,o1", "tap,o2": "right,o1", "tap,o3": "cap,o1"}
    ports = {"o1": "left,o1", "o2": "right,o2", "qubit": "cap,o2"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


# ---------------------------------------------------------------------------
# Layout-backed model from the Palace electrostatic dataset
# ---------------------------------------------------------------------------

GATEMON_DATASET = "gatemon_palace"
"""Dataset generated by ``qpdk/models/datasets/data/gatemon.py``."""

_GATEMON_TERMINALS = ("island", "gate", "readout")


def _lookup_from(data: Dataset) -> GridInterpolator:
    """Check the terminals and Maxwell matrices, then build the interpolator."""
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    if grid.terminals != _GATEMON_TERMINALS:
        msg = f"Expected terminals {_GATEMON_TERMINALS}, got {grid.terminals}"
        raise ValueError(msg)
    check_maxwell(grid.values)
    return GridInterpolator(grid)


@cache
def _cached_lookup(location: Path | str) -> GridInterpolator:
    """Open a dataset by name or path once per process."""
    return _lookup_from(Dataset(location))


def _gatemon_lookup(dataset: Dataset | Path | str) -> GridInterpolator:
    """Load the gatemon Maxwell matrix once, outside numerical evaluation."""
    if isinstance(dataset, Dataset):
        return _lookup_from(dataset)
    return _cached_lookup(dataset)


def gatemon_capacitances(
    *,
    island_size: ArrayLike = 150.0,
    pocket_gap: ArrayLike = 60.0,
    pocket_gap_top: ArrayLike = 40.0,
    gate_tip_gap: ArrayLike = 5.0,
    dataset: Dataset | Path | str = GATEMON_DATASET,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Return the island capacitances of a :func:`~qpdk.cells.gatemon` layout, in F.

    Dimensions are in µm and follow the dataset recipe's layout rule. The
    nominal point matches the default device. Below 150 µm, the contact and
    gate-tip dimensions scale with the island. The gate feed moves right as
    needed to retain its hBN landing clearance. The Palace Maxwell matrix gives

    - :math:`C_\Sigma = C_{ii}`, the total island capacitance, so
      :math:`E_\text{C} = e^2 / 2 C_\Sigma`,
    - :math:`C_\text{g} = -C_{ig}` to the gate line, which sets the island offset charge
      :math:`n_\text{g} = C_\text{g} V_\text{g} / 2e`,
    - :math:`C_\text{c} = -C_{ir}` to the readout CPW over the pocket.

    The bundled recipe extracts only M1 sheets. Contact leads, the local top
    gate, hBN and thermal oxide need a separate calculation. Its
    :math:`C_\text{g}` is coupling to the M1 feed, not the nanotube gate lever arm.

    :math:`C_\Sigma` already includes :math:`C_\text{g}` and
    :math:`C_\text{c}`. The junction is omitted; add its capacitance to
    :math:`C_\Sigma` before computing :math:`E_\text{C}`, then pass
    :math:`E_\text{C}` to :func:`gatemon_frequency_and_anharmonicity` for the
    spectrum. Dimensions
    outside the stored grid return NaN. Supports JIT, vmap and grad.

    Args:
        island_size: Side of the square island in µm.
        pocket_gap: Etched gap around the island sides and bottom in µm.
        pocket_gap_top: Etched gap between the island and the readout in µm.
        gate_tip_gap: Lateral gap from hBN to the M1 gate landing in µm.
        dataset: Dataset or path with terminals ``island``, ``gate``, ``readout``.

    Returns:
        ``(C_sigma, C_g, C_c)`` in F, each with the broadcast input shape.
    """
    with jax.ensure_compile_time_eval():
        lookup = _gatemon_lookup(dataset)
    c = lookup(
        island_size=island_size,
        pocket_gap=pocket_gap,
        pocket_gap_top=pocket_gap_top,
        gate_tip_gap=gate_tip_gap,
    )
    return c[..., 0, 0], -c[..., 0, 1], -c[..., 0, 2]


def gatemon_fem_model(
    *,
    delta_ghz: float = DEFAULT_DELTA_GHZ,
    transmissions: TransmissionsLike = DEFAULT_TRANSMISSIONS,
    junction_capacitance: float = 0.0,
    dataset: Dataset | Path | str = GATEMON_DATASET,
) -> sax.SDictModel:
    r"""Build a linearised gatemon circuit from the stored Palace matrix.

    The ports ``island``, ``gate`` and ``readout`` all reference chip ground.
    Their admittance is :math:`j\omega C` from the full Maxwell matrix, plus the
    junction between the island and ground,

    .. math::

        Y_\text{J} = \frac{1}{j\omega L_\text{J}} + j\omega C_\text{J},
        \qquad
        L_\text{J} = \left(\frac{\Phi_0}{2\pi}\right)^2 \frac{1}{E_\text{J}},

    with :math:`E_\text{J}` from the Andreev bound states
    (:func:`gatemon_josephson_energy`). The returned JAX model accepts
    frequency in Hz, ``z_ref`` in ohms, ``delta_ghz``, ``transmissions`` and
    ``junction_capacitance``, and the layout dimensions of
    :func:`gatemon_capacitances` in µm.

    This describes small oscillations around :math:`\varphi = 0`, not the
    nonlinear spectrum. The ``readout`` port is an open CPW segment: drive it
    for the coupling, but its self-capacitance belongs to the segment, so do not
    add it to a distributed resonator model.

    The bundled recipe includes only M1 sheets. Its matrix omits the contact
    leads, local top gate, hBN and thermal oxide, so it does not determine the
    nanotube's gate lever arm or the complete capacitance of the hybrid device.

    Args:
        delta_ghz: Default induced gap :math:`\Delta` in GHz.
        transmissions: Default channel transmissions :math:`\tau_p`.
        junction_capacitance: Default extra junction capacitance in F.
        dataset: Dataset or path with terminals ``island``, ``gate``, ``readout``.

    Returns:
        Jittable three-port SAX model.
    """
    lookup = _gatemon_lookup(dataset)
    defaults = dict(
        island_size=150.0, pocket_gap=60.0, pocket_gap_top=40.0, gate_tip_gap=5.0
    )
    island = jnp.zeros((3, 3)).at[0, 0].set(1.0)
    identity = jnp.eye(3)

    def model(
        f: ArrayLike = DEFAULT_FREQUENCY,
        z_ref: ArrayLike = 50.0,
        delta_ghz: ArrayLike = delta_ghz,
        transmissions: TransmissionsLike = transmissions,
        junction_capacitance: ArrayLike = junction_capacitance,
        **params: ArrayLike,
    ) -> sax.SDict:
        """Evaluate the lossless linearised gatemon at positive frequencies."""
        capacitance = lookup(**{**defaults, **params})
        inductance = ej_to_inductance(
            gatemon_josephson_energy(delta_ghz, transmissions)
        )
        omega = 2 * jnp.pi * jnp.asarray(f)
        y_junction = 1j * omega * jnp.asarray(junction_capacitance) + 1 / (
            1j * omega * inductance
        )
        admittance = (
            1j * omega[..., None, None] * capacitance
            + y_junction[..., None, None] * island
        )
        normalized = jnp.asarray(z_ref)[..., None, None] * admittance
        values = jnp.linalg.solve(identity + normalized, identity - normalized)
        valid = (
            (jnp.asarray(f) > 0)
            & (inductance > 0)
            & (jnp.asarray(junction_capacitance) >= 0)
        )
        values = jnp.where(valid[..., None, None], values, jnp.nan)
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(_GATEMON_TERMINALS)
            for j, b in enumerate(_GATEMON_TERMINALS)
        }

    sax.replace_kwargs(
        model,
        f=DEFAULT_FREQUENCY,
        z_ref=50.0,
        delta_ghz=delta_ghz,
        transmissions=transmissions,
        junction_capacitance=junction_capacitance,
        **defaults,
    )
    return jit(model)
