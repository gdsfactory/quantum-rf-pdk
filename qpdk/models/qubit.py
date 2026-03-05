"""Qubit LC resonator models.

This module provides LC resonator models for superconducting transmon qubits
and coupled qubit systems. The models are based on the standard LC resonator
formulation with appropriate grounding configurations.

For double-pad transmon qubits, we use an ungrounded LC resonator since
the two islands are floating. For shunted transmon qubits, one island is
grounded, so we use a grounded LC resonator.

The helper functions convert between qubit Hamiltonian parameters (charging
energy :math:`E_C`, Josephson energy :math:`E_J`, coupling strength :math:`g`) and the
corresponding circuit parameters (capacitance, inductance).

References:
    - :cite:`kochChargeinsensitiveQubitDesign2007a`
    - :cite:`gaoPhysicsSuperconductingMicrowave2008`
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import sax
from sax.models.rf import capacitor, electrical_short, tee

from qpdk.models.constants import DEFAULT_FREQUENCY, Φ_0, e, h
from qpdk.models.generic import lc_resonator, lc_resonator_coupled
from qpdk.models.waveguides import straight_shorted


@partial(jax.jit, inline=True)
def ec_to_capacitance(ec_ghz: float) -> float:
    r"""Convert charging energy :math:`E_C` to total capacitance :math:`C_\Sigma`.

    The charging energy is related to capacitance by:

    .. math::

        E_C = \frac{e^2}{2 C_\Sigma}

    where :math:`e` is the electron charge.

    Args:
        ec_ghz: Charging energy in GHz.

    Returns:
        Total capacitance in Farads.

    Example:
        >>> C = ec_to_capacitance(0.2)  # 0.2 GHz (200 MHz) charging energy
        >>> print(f"{C * 1e15:.1f} fF")  # ~96 fF
    """
    ec_joules = ec_ghz * 1e9 * h  # Convert GHz to Joules
    return e**2 / (2 * ec_joules)


@partial(jax.jit, inline=True)
def ej_to_inductance(ej_ghz: float) -> float:
    r"""Convert Josephson energy :math:`E_J` to Josephson inductance :math:`L_J`.

    The Josephson energy is related to inductance by:

    .. math::

        E_J = \frac{\Phi_0^2}{4 \pi^2 L_J} = \frac{(\hbar / 2e)^2}{L_J}

    This is equivalent to:

    .. math::

        L_J = \frac{\Phi_0}{2 \pi I_c}

    where :math:`I_c` is the critical current and :math:`\Phi_0` is the magnetic flux quantum.

    Args:
        ej_ghz: Josephson energy in GHz.

    Returns:
        Josephson inductance in Henries.

    Example:
        >>> L = ej_to_inductance(20.0)  # 20 GHz Josephson energy
        >>> print(f"{L * 1e9:.2f} nH")  # ~1.0 nH
    """
    ej_joules = ej_ghz * 1e9 * h  # Convert GHz to Joules
    return Φ_0**2 / (4 * math.pi**2 * ej_joules)


@partial(jax.jit, inline=True)
def coupling_strength_to_capacitance(
    g_ghz: float,
    c_sigma: float,
    c_r: float,
    f_q_ghz: float,
    f_r_ghz: float,
) -> jax.Array:
    r"""Convert coupling strength :math:`g` to coupling capacitance :math:`C_c`.

    In the dispersive limit (:math:`g \ll f_q, f_r`), the coupling strength
    can be related to a coupling capacitance via:

    .. math::

        g \approx \frac{1}{2} \frac{C_c}{\sqrt{C_\Sigma C_r}} \sqrt{f_q f_r}

    Solving for :math:`C_c`:

    .. math::

        C_c = \frac{2g}{\sqrt{f_q f_r}} \sqrt{C_\Sigma C_r}

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
    return 2 * g_ghz / sqrt_freq * sqrt_c


@partial(jax.jit, inline=True)
def double_island_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SDict:
    r"""LC resonator model for a double-island transmon qubit.

    A double-island transmon has two superconducting islands connected by
    Josephson junctions, with both islands floating (not grounded). This is
    modeled as an ungrounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_q \approx \frac{1}{2\pi} \sqrt{8 E_J E_C} - E_C

    For the LC model, the resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    Use :func:`ec_to_capacitance` and :func:`ej_to_inductance` to convert
    from qubit Hamiltonian parameters.

    .. svgbob::

        o1 ──┬──L──┬── o2
             │     │
             └──C──┘

    Args:
        f: Array of frequency points in Hz.
        capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
        inductance: Josephson inductance :math:`L_J` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=False,
    )


@partial(jax.jit, inline=True)
def double_island_transmon_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SType:
    """LC resonator model for a double-island transmon qubit with bounding box ports.

    This model is the same as :func:`double_island_transmon`.
    """
    return double_island_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
    )


@partial(jax.jit, inline=True)
def flipmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SType:
    r"""LC resonator model for a flipmon qubit.

    This model is identical to :func:`double_island_transmon`.
    """
    return double_island_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
    )


@partial(jax.jit, inline=True)
def flipmon_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SType:
    """LC resonator model for a flipmon qubit with bounding box ports.

    This model is the same as :func:`flipmon`.
    """
    return flipmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
    )


@partial(jax.jit, inline=True)
def shunted_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SDict:
    r"""LC resonator model for a shunted transmon qubit.

    A shunted transmon has one island grounded and the other island connected
    to the junction. This is modeled as a grounded parallel LC resonator.

    The qubit frequency is approximately:

    .. math::

        f_q \approx \frac{1}{2\pi} \sqrt{8 E_J E_C} - E_C

    For the LC model, the resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

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
        inductance: Josephson inductance :math:`L_J` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=True,
    )


@partial(jax.jit, static_argnames=["grounded"])
def transmon_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 1e-9,
    grounded: bool = False,
    coupling_capacitance: float = 10e-15,
    coupling_inductance: float = 0.0,
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
        inductance: Josephson inductance :math:`L_J` in Henries.
        grounded: If True, the qubit is a shunted transmon (grounded).
            If False, it is a double-pad transmon (ungrounded).
        coupling_capacitance: Coupling capacitance :math:`C_c` in Farads.
        coupling_inductance: Coupling inductance :math:`L_c` in Henries.

    Returns:
        sax.SDict: S-parameters dictionary with ports o1 and o2.
    """
    return lc_resonator_coupled(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
        grounded=grounded,
        coupling_capacitance=coupling_capacitance,
        coupling_inductance=coupling_inductance,
    )


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
    from qubit Hamiltonian parameters (:math:`E_C`, :math:`E_J`) to circuit parameters.

    Note:
        This function is not JIT-compiled because it depends on :func:`~straight_shorted`,
        which internally uses scikit-rf for transmission line modeling.

    Args:
        f: Array of frequency points in Hz.
        qubit_capacitance: Total capacitance :math:`C_\Sigma` of the qubit in Farads.
            Convert from charging energy using :func:`ec_to_capacitance`.
        qubit_inductance: Josephson inductance :math:`L_J` in Henries.
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
    # Use a 1-port short to terminate the internally shorted resonator end
    # to avoid dangling ports in the circuit evaluation.
    # Also short the grounded qubit's o2 port
    terminator = electrical_short(f=f, n_ports=2 if qubit_grounded else 1)

    instances: dict[str, sax.SType] = {
        "resonator": resonator,
        "qubit": qubit,
        "coupling_capacitor": coupling_cap,
        "tee": tee_junction,
        "terminator": terminator,
    }

    # Connect: resonator -- tee -- capacitor -- qubit
    # The tee splits the resonator signal to the coupling capacitor
    connections = {
        "resonator,o1": "tee,o1",
        "resonator,o2": "terminator,o1",  # Explicitly terminate
        "tee,o2": "coupling_capacitor,o1",
        "coupling_capacitor,o2": "qubit,o1",
    }

    if qubit_grounded:
        connections["qubit,o2"] = "terminator,o2"

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


@partial(jax.jit, inline=True)
def xmon_transmon(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
) -> sax.SType:
    """LC resonator model for an Xmon style transmon qubit.

    An Xmon transmon is typically shunted, so this model wraps :func:`shunted_transmon`.
    """
    return shunted_transmon(
        f=f,
        capacitance=capacitance,
        inductance=inductance,
    )


# Aliases for backward compatibility or to match cell naming
double_pad_transmon = double_island_transmon
double_pad_transmon_with_bbox = double_island_transmon_with_bbox
double_pad_transmon_with_resonator = double_island_transmon_with_resonator


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from qpdk import PDK

    PDK.activate()

    f = jnp.linspace(1e9, 10e9, 2001)

    # Calculate bare qubit resonance frequency
    C_q = 100e-15
    L_q = 7e-9
    f_q_bare = 1 / (2 * jnp.pi * jnp.sqrt(L_q * C_q))

    configs = [
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
            label=rf"Bare Qubit ($f_q = {f_q_bare / 1e9:.3f}$ GHz)",
        )
        ax.grid(True)
        ax.legend()

    ax2.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax2.set_ylabel("Phase (rad)")
    fig.suptitle(
        rf"Qubit Coupled to Quarter-Wave Resonator ($C_q=${C_q * 1e15:.0f} fF, $L_q=${L_q * 1e9:.0f} nH)"
    )
    plt.tight_layout()
    plt.show()
