"""Coupler models."""

import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from skrf import Frequency

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.generic import capacitor, tee
from qpdk.models.media import cross_section_to_media
from qpdk.models.waveguides import straight


def cpw_cpw_coupling_capacitance(
    f: sax.FloatArrayLike,  # noqa: ARG001
    length: float,  # noqa: ARG001
    gap: float,  # noqa: ARG001
    cross_section: CrossSectionSpec,  # noqa: ARG001
) -> float:
    """Calculate the coupling capacitance between two parallel CPWs.

    TODO: this is a placeholder function and needs to be implemented properly.

    Args:
        length: The coupling length in µm.
        gap: The gap between the two CPWs in µm.
        cross_section: The cross-section of the CPW.
        f: Frequency array in Hz.

    Returns:
        The total coupling capacitance in Farads.
    """
    return 60e-15


def coupler_straight(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 20.0,
    gap: int | float = 0.27,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for two coupled coplanar waveguides, :func:`~qpdk.cells.waveguides.coupler_straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SType: S-parameters dictionary

    .. code::

        o2──────▲───────o3
                │gap
        o1──────▼───────o4
    """
    f = jnp.asarray(f)
    straight_settings = {"length": length / 2, "cross_section": cross_section}
    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(
            f, length, gap, cross_section
        ),  # gap * 1e-18 * f,  # TODO implement FEM simulation retrieval or use some paper
        "z0": cross_section_to_media(cross_section)(
            frequency=Frequency.from_f(np.array(f), unit="Hz")
        ).z0,
    }

    # Create straight instances with shared settings
    straight_instances = {
        f"straight_{i}_{j}": straight(**straight_settings)
        for i in [1, 2]
        for j in [1, 2]
    }
    tee_instances = {f"tee_{i}": tee() for i in [1, 2]}

    instances = {
        **straight_instances,
        **tee_instances,
        "capacitor": capacitor(**capacitor_settings),
    }
    connections = {
        "straight_1_1,o1": "tee_1,o1",
        "straight_1_2,o1": "tee_1,o2",
        "straight_2_1,o1": "tee_2,o1",
        "straight_2_2,o1": "tee_2,o2",
        "tee_1,o3": "capacitor,o1",
        "tee_2,o3": "capacitor,o2",
    }
    ports = {
        "o2": "straight_1_1,o2",
        "o3": "straight_1_2,o2",
        "o1": "straight_2_1,o2",
        "o4": "straight_2_2,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)
