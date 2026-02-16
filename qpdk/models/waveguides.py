"""Waveguides."""

import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from skrf import Frequency
from sax.models.rf import capacitor, inductor, gamma_0_load, admittance, impedance, tee

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.generic import short_2_port
from qpdk.models.media import cross_section_to_media
from qpdk.tech import coplanar_waveguide


def straight(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a straight waveguide.

    See `scikit-rf <skrf>`_ for details on analytical formulæ.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary

    .. _skrf: https://scikit-rf.org/
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    # Keep f as tuple for scikit-rf, convert to array only for final JAX operations
    media = cross_section_to_media(cross_section)
    skrf_media = media(frequency=Frequency.from_f(np.asarray(f_flat), unit="Hz"))
    transmission_line = skrf_media.line(d=np.asarray(length), unit="um")
    sdict = {
        ("o1", "o1"): jnp.array(transmission_line.s[:, 0, 0]).reshape(*f.shape),
        ("o1", "o2"): jnp.array(transmission_line.s[:, 0, 1]).reshape(*f.shape),
        ("o2", "o2"): jnp.array(transmission_line.s[:, 1, 1]).reshape(*f.shape),
    }
    return sax.reciprocal(sdict)


def straight_shorted(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a straight waveguide with one shorted end.

    This may be used to model a quarter-wave coplanar waveguide resonator.

    Note:
        The port ``o2`` is internally shorted and should not be used.
        It seems to be a Sax limitation that we need to define at least two ports.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    instances = {"straight": straight(**kwargs), "short": short_2_port(f=f)}
    connections = {
        "straight,o2": "short,o1",
    }
    ports = {
        "o1": "straight,o1",
        "o2": "short_2_port,o2",  # don't use: shorted!
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def bend_circular(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a circular bend, wrapped to to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)


def bend_euler(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for an Euler bend, wrapped to to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)


def bend_s(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for an S-bend, wrapped to to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)  # pyrefly: ignore[bad-keyword-argument]


def rectangle(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a rectangular section, wrapped to to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)  # pyrefly: ignore[bad-keyword-argument]


def taper_cross_section(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section_1: CrossSectionSpec = "cpw",  # noqa: ARG001
    cross_section_2: CrossSectionSpec = "cpw",  # noqa: ARG001
    n_points: int = 50,
) -> sax.SType:
    """S-parameter model for a cross-section taper using linear interpolation.

    Uses jax.scipy.interpolate.RegularGridInterpolator to efficiently interpolate
    media parameters (width and gap) along the taper length.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section_1: Cross-section for the start of the taper.
        cross_section_2: Cross-section for the end of the taper.
        n_points: Number of segments to divide the taper into for simulation.

    Returns:
        sax.SType: S-parameters dictionary
    """
    # Ensure n_points is a concrete Python int

    # Get media parameters at the start and end of the taper
    f = jnp.asarray(f)
    # dummy_freq = Frequency.from_f(np.asarray(f), unit="Hz")
    # media_1 = cross_section_to_media(cross_section_1)
    # media_2 = cross_section_to_media(cross_section_2)
    # media_1_obj = media_1(frequency=dummy_freq)
    # media_2_obj = media_2(frequency=dummy_freq)

    # width_1 = getattr(media_1_obj, "w", 0.0)
    # width_2 = getattr(media_2_obj, "w", 0.0)
    # gap_1 = getattr(media_1_obj, "s", 0.0)
    # gap_2 = getattr(media_2_obj, "s", 0.0)

    # Create interpolation grid points using physical positions
    # position_grid = jnp.array([0.0, length])
    # width_values = jnp.array([width_1, width_2])
    # gap_values = jnp.array([gap_1, gap_2])

    # Create interpolators for width and gap
    # width_interpolator = jax.scipy.interpolate.RegularGridInterpolator(
    #     (position_grid,), width_values, method="linear"
    # )
    # gap_interpolator = jax.scipy.interpolate.RegularGridInterpolator(
    #     (position_grid,), gap_values, method="linear"
    # )

    segment_length = length / n_points
    # Compute physical positions for each segment
    # positions = jnp.linspace(0, length, num=n_points)

    instances = {
        **{
            f"straight_{i}": straight(
                f=f,
                length=segment_length,
            )
            for i in range(n_points)
        }
    }
    connections = {
        **{f"straight_{i},o2": f"straight_{i + 1},o1" for i in range(n_points - 1)}
    }
    ports = {
        "o1": "straight_0,o1",
        "o2": f"straight_{n_points - 1},o2",
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def launcher(
    f: ArrayLike = DEFAULT_FREQUENCY,
    straight_length: sax.Float = 200.0,
    taper_length: sax.Float = 100.0,
    cross_section_big: CrossSectionSpec | None = None,
    cross_section_small: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a launcher, effectively a straight section followed by a taper.

    Args:
        f: Array of frequency points in Hz
        straight_length: Length of the straight section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_big: Cross-section for the wide section.
        cross_section_small: Cross-section for the narrow section.

    Returns:
        sax.SType: S-parameters dictionary
    """
    f = jnp.asarray(f)
    if cross_section_big is None:
        cross_section_big = coplanar_waveguide(width=200, gap=100)

    instances = {
        "straight": straight(
            f=f,
            length=straight_length,
            cross_section=cross_section_big,
        ),
        "taper": taper_cross_section(
            f=f,
            length=taper_length,
            cross_section_1=cross_section_big,
            cross_section_2=cross_section_small,
        ),
    }
    connections = {
        "straight,o2": "taper,o1",
    }
    ports = {
        "waveport": "straight,o1",
        "o1": "taper,o2",
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)
