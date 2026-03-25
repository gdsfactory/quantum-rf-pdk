"""Unimon qubit models.

This module provides both a SAX microwave S-parameter model and a quantum
Hamiltonian model for the unimon qubit.

**Microwave model** — The unimon is modeled as two quarter-wave CPW
transmission-line sections connected by a Josephson junction (SQUID).
The S-parameters describe the linear microwave response of the structure.

**Hamiltonian model** — The effective single-mode Hamiltonian of the unimon is

.. math::

    \\hat{H} = 4 E_C \\hat{n}^2
             + \\tfrac{1}{2} E_L (\\hat{\\varphi} - \\varphi_{\\text{ext}})^2
             - E_J \\cos\\hat{\\varphi}

where :math:`E_C` is the charging energy, :math:`E_L` the inductive energy
from the geometric inductance of the resonator arms, :math:`E_J` the
Josephson energy, and :math:`\\varphi_{\\text{ext}}` the external flux bias
(in units of the reduced flux quantum).

The unimon operates in the regime :math:`E_L \\sim E_J`, which is distinct
from both the transmon (:math:`E_J \\gg E_C`, no geometric inductance) and
the fluxonium (:math:`E_L \\ll E_J`).

References:
    - :cite:`hyyppaUnimonQubit2022`
    - :cite:`tuokkolaMultimodePhysicsUnimon2024`
    - :cite:`sajadiParameterOptimizationUnimon2025`
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import sax
from sax.models.rf import capacitor, electrical_short, tee

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0, e, h
from qpdk.models.generic import lc_resonator
from qpdk.models.waveguides import straight_shorted


# ---------------------------------------------------------------------------
# Helper: energy-scale conversions
# ---------------------------------------------------------------------------


@partial(jax.jit, inline=True)
def el_to_inductance(el_ghz: float) -> float:
    r"""Convert inductive energy :math:`E_L` to geometric inductance :math:`L`.

    The inductive energy is related to the geometric inductance by:

    .. math::

        E_L = \frac{\Phi_0^2}{4 \pi^2 \cdot 2 L}
            = \frac{(\hbar / 2e)^2}{2 L}

    Solving for :math:`L`:

    .. math::

        L = \frac{\Phi_0^2}{8 \pi^2 E_L}

    Args:
        el_ghz: Inductive energy in GHz.

    Returns:
        Geometric inductance in Henries.

    Example:
        >>> L = el_to_inductance(5.0)  # 5 GHz inductive energy
        >>> print(f"{L * 1e9:.2f} nH")
    """
    el_joules = el_ghz * 1e9 * h
    return Φ_0**2 / (8 * math.pi**2 * el_joules)


# ---------------------------------------------------------------------------
# SAX microwave model
# ---------------------------------------------------------------------------


def unimon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    arm_length: float = 4000.0,
    cross_section: str = "cpw",
    junction_capacitance: float = 5e-15,
    junction_inductance: float = 7e-9,
) -> sax.SDict:
    r"""SAX S-parameter model for a unimon qubit.

    The unimon is modeled as two shorted quarter-wave CPW transmission lines
    connected by a parallel LC element representing the Josephson junction
    (linearised RCSJ model).

    .. svgbob::

        o1 ── λ/4 TL ──┬──L_J──┬── λ/4 TL ── o2
                        │       │
                        └──C_J──┘
                        (junction)

    The two ports ``o1`` and ``o2`` correspond to the shorted ends of the
    resonator arms, which are used for external coupling.

    Args:
        f: Array of frequency points in Hz.
        arm_length: Length of each quarter-wave resonator arm in µm.
        cross_section: Cross-section specification for the CPW arms.
        junction_capacitance: Junction capacitance :math:`C_J` in Farads.
        junction_inductance: Junction inductance :math:`L_J` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports ``o1`` and ``o2``.
    """
    f = jnp.asarray(f)

    # Two quarter-wave shorted transmission-line arms
    arm_left = straight_shorted(f=f, length=arm_length, cross_section=cross_section)
    arm_right = straight_shorted(f=f, length=arm_length, cross_section=cross_section)

    # Josephson junction as parallel LC
    junction = lc_resonator(
        f=f,
        capacitance=junction_capacitance,
        inductance=junction_inductance,
        grounded=False,
    )

    # Tee junctions to connect junction in between arms
    tee_l = tee(f=f)
    tee_r = tee(f=f)

    # Terminal shorts for the shorted ends of the arms
    short_l = electrical_short(f=f, n_ports=2)
    short_r = electrical_short(f=f, n_ports=2)

    instances: dict[str, sax.SType] = {
        "arm_left": arm_left,
        "arm_right": arm_right,
        "junction": junction,
        "tee_l": tee_l,
        "tee_r": tee_r,
        "short_l": short_l,
        "short_r": short_r,
    }

    connections = {
        # Left arm connects to left tee
        "arm_left,o1": "tee_l,o1",
        # Shorted end of left arm
        "arm_left,o2": "short_l,o1",
        # Right arm connects to right tee
        "arm_right,o1": "tee_r,o1",
        # Shorted end of right arm
        "arm_right,o2": "short_r,o1",
        # Junction between tees
        "tee_l,o2": "junction,o1",
        "junction,o2": "tee_r,o2",
    }

    ports = {
        "o1": "tee_l,o3",
        "o2": "tee_r,o3",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def unimon_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    arm_length: float = 4000.0,
    cross_section: str = "cpw",
    junction_capacitance: float = 5e-15,
    junction_inductance: float = 7e-9,
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    r"""SAX model for a unimon qubit with capacitive coupling.

    Extends :func:`unimon` by adding a coupling capacitor to one port,
    representing the capacitive coupling to a readout resonator or
    probe line.

    .. svgbob::

        o1 ── Cc ── unimon ── o2

    Args:
        f: Array of frequency points in Hz.
        arm_length: Length of each quarter-wave resonator arm in µm.
        cross_section: Cross-section specification for the CPW arms.
        junction_capacitance: Junction capacitance :math:`C_J` in Farads.
        junction_inductance: Junction inductance :math:`L_J` in Henries.
        coupling_capacitance: Coupling capacitance in Farads.

    Returns:
        sax.SDict: S-parameters dictionary with ports ``o1`` and ``o2``.
    """
    f = jnp.asarray(f)

    instances: dict[str, sax.SType] = {
        "unimon": unimon(
            f=f,
            arm_length=arm_length,
            cross_section=cross_section,
            junction_capacitance=junction_capacitance,
            junction_inductance=junction_inductance,
        ),
        "coupling_cap": capacitor(f=f, capacitance=coupling_capacitance),
    }

    connections = {
        "coupling_cap,o2": "unimon,o1",
    }

    ports = {
        "o1": "coupling_cap,o1",
        "o2": "unimon,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


# ---------------------------------------------------------------------------
# Quantum Hamiltonian model
# ---------------------------------------------------------------------------


def _build_phase_operator(n_states: int) -> jax.Array:
    r"""Build the phase operator :math:`\hat{\varphi}` in the charge basis.

    In the charge basis :math:`|n\rangle`, the phase operator acts as a
    shift operator:

    .. math::

        \hat{\varphi} |n\rangle = |n-1\rangle + |n+1\rangle

    This is represented as an off-diagonal matrix (tridiagonal with ones
    on the super- and sub-diagonals), which is the standard representation
    used in circuit-QED Hamiltonian diagonalisation.

    Note:
        This operator is **not** the exact phase operator (which is not
        well-defined on a finite Hilbert space), but the standard
        approximation used for superconducting-circuit Hamiltonians
        in the charge basis.  It becomes exact in the limit
        :math:`n_{\\text{states}} \\to \\infty`.

    Args:
        n_states: Dimension of the truncated Hilbert space.

    Returns:
        A ``(n_states, n_states)`` real matrix.
    """
    return jnp.diag(jnp.ones(n_states - 1), k=1) + jnp.diag(
        jnp.ones(n_states - 1), k=-1
    )


def _build_number_operator(n_states: int, n_max: int) -> jax.Array:
    r"""Build the Cooper-pair number operator :math:`\hat{n}` in the charge basis.

    In the charge basis :math:`|n\rangle` with
    :math:`n \in \{-n_{\max}, \ldots, +n_{\max}\}`, the number operator is
    diagonal:

    .. math::

        \hat{n} = \operatorname{diag}(-n_{\max}, \ldots, +n_{\max})

    Args:
        n_states: Dimension of the truncated Hilbert space
            (should be ``2 * n_max + 1``).
        n_max: Maximum charge number included in the truncation.

    Returns:
        A ``(n_states, n_states)`` real diagonal matrix.
    """
    return jnp.diag(jnp.arange(-n_max, n_max + 1, dtype=jnp.float64))


def _build_cosine_operator(n_states: int) -> jax.Array:
    r"""Build the cosine operator :math:`\cos\hat{\varphi}` in the charge basis.

    In the charge basis the cosine of the phase is a nearest-neighbour
    hopping operator:

    .. math::

        \cos\hat{\varphi} = \tfrac{1}{2}
        \bigl( e^{i\hat{\varphi}} + e^{-i\hat{\varphi}} \bigr)

    which acts as :math:`\cos\hat{\varphi}|n\rangle =
    \tfrac{1}{2}(|n+1\rangle + |n-1\rangle)`.

    Args:
        n_states: Dimension of the truncated Hilbert space.

    Returns:
        A ``(n_states, n_states)`` real matrix.
    """
    return 0.5 * (
        jnp.diag(jnp.ones(n_states - 1), k=1)
        + jnp.diag(jnp.ones(n_states - 1), k=-1)
    )


def unimon_hamiltonian(
    ec_ghz: float = 1.0,
    el_ghz: float = 5.0,
    ej_ghz: float = 10.0,
    phi_ext: float = jnp.pi,
    n_max: int = 30,
) -> jax.Array:
    r"""Build the unimon Hamiltonian matrix in the charge basis.

    The effective single-mode unimon Hamiltonian is

    .. math::

        \hat{H} = 4 E_C \hat{n}^2
                 + \tfrac{1}{2} E_L (\hat{\varphi} - \varphi_{\text{ext}})^2
                 - E_J \cos\hat{\varphi}

    This Hamiltonian is identical in form to the fluxonium Hamiltonian but
    the unimon operates in a qualitatively different parameter regime
    where :math:`E_L \sim E_J` (rather than :math:`E_L \ll E_J` for
    fluxonium).

    The Hamiltonian is constructed in the charge basis
    :math:`|n\rangle` with :math:`n \in \{-n_{\max}, \ldots, +n_{\max}\}`,
    giving a matrix of size :math:`(2 n_{\max} + 1) \times (2 n_{\max} + 1)`.

    See :cite:`hyyppaUnimonQubit2022,sajadiParameterOptimizationUnimon2025`
    for the derivation and parameter regimes.

    Args:
        ec_ghz: Charging energy :math:`E_C` in GHz.
        el_ghz: Inductive energy :math:`E_L` in GHz.
        ej_ghz: Josephson energy :math:`E_J` in GHz.
        phi_ext: External flux bias :math:`\varphi_{\text{ext}}`
            in radians (reduced flux-quantum units).
            The optimal operating point is :math:`\pi`.
        n_max: Charge-basis truncation; Hilbert-space dimension is
            :math:`2 n_{\max} + 1`.

    Returns:
        A ``(2*n_max+1, 2*n_max+1)`` Hermitian matrix (in GHz).
    """
    n_states = 2 * n_max + 1

    n_hat = _build_number_operator(n_states, n_max)
    cos_phi = _build_cosine_operator(n_states)
    phi_hat = _build_phase_operator(n_states)

    # H = 4 E_C n^2 + 0.5 E_L (phi - phi_ext)^2 - E_J cos(phi)
    # Expand the quadratic: (phi - phi_ext)^2 = phi^2 - 2*phi_ext*phi + phi_ext^2
    H = (
        4 * ec_ghz * n_hat @ n_hat
        + 0.5 * el_ghz * (phi_hat @ phi_hat - 2 * phi_ext * phi_hat + phi_ext**2 * jnp.eye(n_states))
        - ej_ghz * cos_phi
    )

    return H


def unimon_energies(
    ec_ghz: float = 1.0,
    el_ghz: float = 5.0,
    ej_ghz: float = 10.0,
    phi_ext: float = jnp.pi,
    n_max: int = 30,
    n_levels: int = 5,
) -> jax.Array:
    r"""Compute the lowest energy eigenvalues of the unimon Hamiltonian.

    Diagonalises the Hamiltonian returned by :func:`unimon_hamiltonian` and
    returns the ``n_levels`` lowest eigenvalues, shifted so that the ground
    state energy is zero.

    Args:
        ec_ghz: Charging energy :math:`E_C` in GHz.
        el_ghz: Inductive energy :math:`E_L` in GHz.
        ej_ghz: Josephson energy :math:`E_J` in GHz.
        phi_ext: External flux :math:`\varphi_{\text{ext}}` in radians.
        n_max: Charge-basis truncation.
        n_levels: Number of lowest energy levels to return.

    Returns:
        Array of shape ``(n_levels,)`` with energies in GHz,
        referenced to the ground state.

    Example:
        >>> energies = unimon_energies(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0)
        >>> f_01 = float(energies[1])  # qubit frequency in GHz
        >>> alpha = float(energies[2] - 2 * energies[1])  # anharmonicity
    """
    H = unimon_hamiltonian(
        ec_ghz=ec_ghz,
        el_ghz=el_ghz,
        ej_ghz=ej_ghz,
        phi_ext=phi_ext,
        n_max=n_max,
    )
    eigenvalues = jnp.linalg.eigvalsh(H)
    # Shift to ground state = 0
    eigenvalues = eigenvalues - eigenvalues[0]
    return eigenvalues[:n_levels]


def unimon_frequency_and_anharmonicity(
    ec_ghz: float = 1.0,
    el_ghz: float = 5.0,
    ej_ghz: float = 10.0,
    phi_ext: float = jnp.pi,
    n_max: int = 30,
) -> tuple[float, float]:
    r"""Compute the qubit transition frequency and anharmonicity of the unimon.

    The qubit frequency is

    .. math::

        f_{01} = E_1 - E_0

    and the anharmonicity is

    .. math::

        \alpha = (E_2 - E_1) - (E_1 - E_0) = E_2 - 2 E_1

    (since :math:`E_0 = 0` after shifting).

    Args:
        ec_ghz: Charging energy :math:`E_C` in GHz.
        el_ghz: Inductive energy :math:`E_L` in GHz.
        ej_ghz: Josephson energy :math:`E_J` in GHz.
        phi_ext: External flux :math:`\varphi_{\text{ext}}` in radians.
        n_max: Charge-basis truncation.

    Returns:
        Tuple of (frequency_ghz, anharmonicity_ghz).

    Example:
        >>> f01, alpha = unimon_frequency_and_anharmonicity(
        ...     ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0
        ... )
        >>> print(f"f_01 = {f01:.3f} GHz, alpha = {alpha:.3f} GHz")
    """
    energies = unimon_energies(
        ec_ghz=ec_ghz,
        el_ghz=el_ghz,
        ej_ghz=ej_ghz,
        phi_ext=phi_ext,
        n_max=n_max,
        n_levels=3,
    )
    f_01 = float(energies[1])
    alpha = float(energies[2] - 2 * energies[1])
    return f_01, alpha
