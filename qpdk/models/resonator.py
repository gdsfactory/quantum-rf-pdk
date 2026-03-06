"""Resonators."""

import jax.numpy as jnp
import numpy as np
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec
from numpy.typing import NDArray
from sax.models.rf import capacitor, electrical_open, electrical_short, tee
from skrf.media import Media

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.couplers import cpw_cpw_coupling_capacitance
from qpdk.models.media import cross_section_to_media
from qpdk.models.waveguides import straight, straight_shorted


def quarter_wave_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 5000.0,
    coupling_gap: float = 0.27,
    coupling_straight_length: float = 20,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.

    Args:
        cross_section: The cross-section of the CPW.
        f: Frequency in Hz at which to evaluate the S-parameters.
        length: Total length of the resonator in μm.
        coupling_gap: Gap between the resonator and the probeline in μm.
        coupling_straight_length: Length of the coupling section in μm.

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
        open_start: If True, adds an electrical open at the start.
        open_end: If True, adds an electrical open at the end.

    Returns:
        sax.SDict: S-parameters dictionary with 4 ports.
    """
    f_arr = jnp.asarray(f)
    f_flat = f_arr.ravel()

    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(
            f_arr, coupling_straight_length, coupling_gap, cross_section
        ),
        "z0": cross_section_to_media(cross_section)(
            frequency=skrf.Frequency.from_f(
                np.atleast_1d(np.asarray(f_flat)), unit="Hz"
            )
        ).z0.reshape(f_arr.shape),
    }

    instances = {
        "coupling_1": straight(
            f=f_arr, length=coupling_straight_length / 2, cross_section=cross_section
        ),
        "coupling_2": straight(
            f=f_arr, length=coupling_straight_length / 2, cross_section=cross_section
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
    media: Media,
    is_quarter_wave: bool = True,
) -> NDArray:
    r"""Calculate the resonance frequency of a quarter-wave resonator.

    .. math::

        f &= \frac{v_p}{4L}  \mathtt{ (quarter-wave resonator)} \\
        f &= \frac{v_p}{2L}  \mathtt{ (half-wave resonator)}

    There is some variation according to the frequency range specified for ``media`` due to how
    :math:`v_p` is calculated in skrf. The phase velocity is given by :math:`v_p = i \cdot \omega / \gamma`,
    where :math:`\gamma` is the complex propagation constant and :math:`\omega` is the angular frequency.

    See :cite:`simonsCoplanarWaveguideCircuits2001,m.pozarMicrowaveEngineering2012` for details.

    Args:
        length: Length of the resonator in μm.
        media: skrf media object defining the CPW (or other) properties.
        is_quarter_wave: If True, calculates for a quarter-wave resonator; if False, for a half-wave resonator.
            default is True.

    Returns:
        float: Resonance frequency in Hz.
    """
    coefficient = 4 if is_quarter_wave else 2  # Quarter-wave resonator
    a = media.v_p / (coefficient * length * 1e-6)
    return a.mean().real


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
