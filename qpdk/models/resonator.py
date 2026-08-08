"""Resonators."""

from typing import Any

import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from sax.models.rf import capacitor, electrical_open, electrical_short, tee

from qpdk.helper import deprecated
from qpdk.models.constants import DEFAULT_FREQUENCY, c_0
from qpdk.models.couplers import cpw_cpw_coupling_capacitance
from qpdk.models.cpw import (
    cpw_parameters,
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
)
from qpdk.models.waveguides import launcher, straight, straight_shorted


def _resonator_test_chip_model(
    f: sax.FloatArrayLike,
    *,
    probeline_length: float,
    resonator_lengths: tuple[tuple[float, ...], tuple[float, ...]],
    coupling_gaps: tuple[tuple[float, ...], tuple[float, ...]],
    coupling_length: float = 200.0,
    cross_section: CrossSectionSpec = "coplanar_waveguide",
) -> sax.SDict:
    """Build SAX model for two probelines with coupled resonators."""
    f_arr = jnp.asarray(f)
    resonator_spacing = probeline_length / (len(resonator_lengths[0]) + 1)

    instances = {}
    connections = {}
    ports = {}

    for probeline_idx, (port_names, lengths, gaps) in enumerate(
        zip(
            (("o3", "o4"), ("o1", "o2")),
            resonator_lengths,
            coupling_gaps,
        )
    ):
        west_launcher = f"launcher_{probeline_idx}_west"
        east_launcher = f"launcher_{probeline_idx}_east"
        instances[west_launcher] = launcher(
            f=f_arr,
            cross_section_small=cross_section,
        )
        instances[east_launcher] = launcher(
            f=f_arr,
            cross_section_small=cross_section,
        )
        ports[port_names[0]] = f"{west_launcher},waveport"
        ports[port_names[1]] = f"{east_launcher},waveport"

        first_lead = f"lead_{probeline_idx}_west"
        instances[first_lead] = straight(
            f=f_arr,
            length=200.0,
            cross_section=cross_section,
        )
        connections[f"{west_launcher},o1"] = f"{first_lead},o1"

        for resonator_idx, (length, gap) in enumerate(zip(lengths, gaps)):
            resonator_name = f"resonator_{probeline_idx}_{resonator_idx}"
            instances[resonator_name] = quarter_wave_resonator_coupled(
                f=f_arr,
                length=length,
                coupling_gap=gap,
                coupling_straight_length=coupling_length,
                cross_section=cross_section,
                cross_section_non_resonator=cross_section,
            )

            if resonator_idx == 0:
                connections[f"{first_lead},o2"] = f"{resonator_name},coupling_o1"
            else:
                previous_resonator = f"resonator_{probeline_idx}_{resonator_idx - 1}"
                inter_resonator = f"lead_{probeline_idx}_{resonator_idx}"
                instances[inter_resonator] = straight(
                    f=f_arr,
                    length=resonator_spacing,
                    cross_section=cross_section,
                )
                connections[f"{previous_resonator},coupling_o2"] = (
                    f"{inter_resonator},o1"
                )
                connections[f"{inter_resonator},o2"] = f"{resonator_name},coupling_o1"

        last_resonator = f"resonator_{probeline_idx}_{len(lengths) - 1}"
        final_lead = f"lead_{probeline_idx}_east"
        instances[final_lead] = straight(
            f=f_arr,
            length=400.0,
            cross_section=cross_section,
        )
        connections[f"{last_resonator},coupling_o2"] = f"{final_lead},o1"
        connections[f"{final_lead},o2"] = f"{east_launcher},o1"

    return sax.evaluate_circuit_fg((connections, ports), instances)


def resonator_test_chip_python(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    probeline_length: float = 9000.0,
    probeline_separation: float = 1000.0,  # noqa: ARG001
    resonator_length: float = 4000.0,
    coupling_length: float = 200.0,
    coupling_gap: float = 16.0,
    cross_section: CrossSectionSpec = "coplanar_waveguide",
) -> sax.SDict:
    """SAX model for the four-port resonator test chip sample.

    The layout sample is a composite factory. Keeping its four-port model as
    a SAX leaf lets recursive netlist resolution treat the complete chip like
    an optical composite component while preserving each resonator's settings.

    Returns:
        SAX S-parameter dictionary for the four external ports.
    """
    total_resonators = 16
    lengths = [
        resonator_length * (0.9 + 0.375 * index / (total_resonators - 1))
        for index in range(total_resonators)
    ]
    per_line_lengths: tuple[tuple[float, ...], tuple[float, ...]] = (
        tuple(lengths[0::2]),
        tuple(lengths[1::2]),
    )
    per_line_count = len(per_line_lengths[0])

    return _resonator_test_chip_model(
        f,
        probeline_length=probeline_length,
        resonator_lengths=per_line_lengths,
        coupling_gaps=((coupling_gap,) * per_line_count,) * 2,
        coupling_length=coupling_length,
        cross_section=cross_section,
    )


def resonator_test_chip_yaml(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
) -> sax.SDict:
    """SAX model for ``resonator_test_chip_yaml.pic.yml``.

    This top-level model keeps gdsfactoryplus 1.x recursive netlists from
    expanding settings-specific resonator cell names that have no matching
    PDK model key.

    Returns:
        SAX S-parameter dictionary for the four external ports.
    """
    lengths = (3600.0, 3800.0, 4000.0, 4200.0, 4400.0, 4600.0, 4800.0, 5000.0)
    return _resonator_test_chip_model(
        f,
        probeline_length=9000.0,
        resonator_lengths=(lengths, lengths),
        coupling_gaps=(
            (16.0,) * 8,
            (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0),
        ),
    )


def quarter_wave_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 5000.0,
    coupling_gap: float = 0.27,
    coupling_straight_length: float = 20,
    cross_section: CrossSectionSpec = "cpw",
    cross_section_non_resonator: CrossSectionSpec | None = None,
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.

    Args:
        cross_section: The cross-section of the CPW.
        f: Frequency in Hz at which to evaluate the S-parameters.
        length: Total length of the resonator in μm.
        coupling_gap: Gap between the resonator and the probeline in μm.
        coupling_straight_length: Length of the coupling section in μm.
        cross_section_non_resonator: Cross-section of the coupling waveguide. If
            ``None``, uses ``cross_section``.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f_arr = jnp.asarray(f)

    instances = {
        "resonator": resonator_coupled(
            f=f_arr,
            length=length,
            coupling_gap=coupling_gap,
            coupling_straight_length=coupling_straight_length,
            cross_section=cross_section,
            cross_section_non_resonator=cross_section_non_resonator,
            open_start=True,
            open_end=False,
        ),
        "short": electrical_short(f=f_arr),
    }

    connections = {
        "resonator,resonator_o2": "short,o1",
    }

    ports = {
        "coupling_o1": "resonator,coupling_o1",
        "coupling_o2": "resonator,coupling_o2",
        "resonator_o1": "resonator,resonator_o1",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 5000.0,
    coupling_gap: float = 0.27,
    coupling_straight_length: float = 20,
    cross_section: CrossSectionSpec = "cpw",
    cross_section_non_resonator: CrossSectionSpec | None = None,
    open_start: bool = True,
    open_end: bool = False,
) -> sax.SDict:
    """Model for a coplanar waveguide resonator coupled to a probeline.

    Args:
        cross_section: The cross-section of the CPW.
        f: Frequency in Hz at which to evaluate the S-parameters.
        length: Total length of the resonator in μm.
        coupling_gap: Gap between the resonator and the probeline in μm.
        coupling_straight_length: Length of the coupling section in μm.
        cross_section_non_resonator: Cross-section of the coupling waveguide. If
            ``None``, uses ``cross_section``.
        open_start: If True, adds an electrical open at the start.
        open_end: If True, adds an electrical open at the end.

    Returns:
        sax.SDict: S-parameters dictionary with 4 ports.
    """
    f_arr = jnp.asarray(f)
    if cross_section_non_resonator is None:
        cross_section_non_resonator = cross_section

    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(
            f_arr, coupling_straight_length, coupling_gap, cross_section
        ),
        "z0": cpw_z0_from_cross_section(cross_section, f_arr),
    }

    instances = {
        "coupling_1": straight(
            f=f_arr,
            length=coupling_straight_length / 2,
            cross_section=cross_section_non_resonator,
        ),
        "coupling_2": straight(
            f=f_arr,
            length=coupling_straight_length / 2,
            cross_section=cross_section_non_resonator,
        ),
        "resonator_1": straight(
            f=f_arr, length=coupling_straight_length / 2, cross_section=cross_section
        ),
        "resonator_2": straight(
            f=f_arr,
            length=length - coupling_straight_length / 2,
            cross_section=cross_section,
        ),
        "tee_1": tee(f=f_arr),
        "tee_2": tee(f=f_arr),
        "capacitor": capacitor(f=f_arr, **capacitor_settings),
    }

    connections = {
        "coupling_1,o2": "tee_1,o1",
        "coupling_2,o1": "tee_1,o2",
        "resonator_1,o2": "tee_2,o1",
        "resonator_2,o1": "tee_2,o2",
        "tee_1,o3": "capacitor,o1",
        "tee_2,o3": "capacitor,o2",
    }

    ports = {
        "coupling_o1": "coupling_1,o1",
        "coupling_o2": "coupling_2,o2",
    }

    if open_start:
        instances["open_start_term"] = electrical_open(f=f_arr, n_ports=2)
        connections["resonator_1,o1"] = "open_start_term,o1"
        ports["resonator_o1"] = "open_start_term,o2"
    else:
        ports["resonator_o1"] = "resonator_1,o1"

    if open_end:
        instances["open_end_term"] = electrical_open(f=f_arr, n_ports=2)
        connections["resonator_2,o2"] = "open_end_term,o1"
        ports["resonator_o2"] = "open_end_term,o2"
    else:
        ports["resonator_o2"] = "resonator_2,o2"

    return sax.evaluate_circuit_fg((connections, ports), instances)


def resonator_frequency(
    *,
    length: float,
    epsilon_eff: float | None = None,
    media: Any = None,
    cross_section: CrossSectionSpec = "cpw",
    is_quarter_wave: bool = True,
) -> float:
    r"""Calculate the resonance frequency of a quarter- or half-wave CPW resonator.

    .. math::

        \begin{aligned}
        f &= \frac{v_p}{4L}  \mathtt{ (quarter-wave resonator)} \\
        f &= \frac{v_p}{2L}  \mathtt{ (half-wave resonator)}
        \end{aligned}

    The phase velocity is :math:`v_p = c_0 / \sqrt{\varepsilon_{\mathrm{eff}}}`.

    See :cite:`simonsCoplanarWaveguideCircuits2001,m.pozarMicrowaveEngineering2012` for details.

    Args:
        length: Length of the resonator in μm.
        epsilon_eff: Effective permittivity.  If ``None`` (default),
            computed from *cross_section* using :func:`~qpdk.models.cpw.cpw_parameters`.
        media: Deprecated. Use *epsilon_eff* or *cross_section* instead.
        cross_section: Cross-section specification (used only when
            *epsilon_eff* and *media* are not provided).
        is_quarter_wave: If True, calculates for a quarter-wave resonator; if False, for a half-wave resonator.
            default is True.

    Returns:
        float: Resonance frequency in Hz.
    """
    if epsilon_eff is None:
        if media is not None:
            deprecated(
                "The 'media' argument is deprecated. Use 'epsilon_eff' or 'cross_section' instead."
            )(lambda: None)()
            epsilon_eff = float(jnp.real(jnp.mean(media.ep_r)))
        else:
            width, gap = get_cpw_dimensions(cross_section)
            epsilon_eff, _z0 = cpw_parameters(width, gap)

    v_p = c_0 / jnp.sqrt(jnp.real(epsilon_eff))
    coefficient = 4 if is_quarter_wave else 2
    return float(jnp.squeeze(jnp.real(v_p / (coefficient * length * 1e-6))))


def resonator(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a simple transmission line resonator.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return straight(f=f, length=length, cross_section=cross_section)


def resonator_half_wave(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a half-wave resonator (open at both ends).

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return straight(f=f, length=length, cross_section=cross_section)


def resonator_quarter_wave(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a quarter-wave resonator (shorted at one end).

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return straight_shorted(f=f, length=length, cross_section=cross_section)
