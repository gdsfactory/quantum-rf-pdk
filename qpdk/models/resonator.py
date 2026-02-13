"""Resonators."""

from functools import partial

import jax.numpy as jnp
import numpy as np
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec
from numpy.typing import NDArray
from skrf.media import Media

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.couplers import cpw_cpw_coupling_capacitance
from qpdk.models.media import cross_section_to_media


def quarter_wave_resonator_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 5000.0,
    coupling_gap: float = 0.27,
    coupling_straight_length: float = 20,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.

    TODO: implement with purely sax circuits instead of skrf components.
    Sax circuit version is commented out above but gives differing results.

    ```{svgbob}

                        o1────────────────────o2  ┬
                                                  | coupling_gap
        short--resonator--────────────────────o3  ┴

    ```

    Args:
        cross_section: The cross-section of the CPW.
        f: Frequency in Hz at which to evaluate the S-parameters.
        length: Total length of the resonator in μm.
        coupling_gap: Gap between the resonator and the probeline in μm.
        coupling_straight_length: Length of the coupling section in μm.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    coupling_capacitance = cpw_cpw_coupling_capacitance(
        f_flat, length, coupling_gap, cross_section
    )
    media: Media = cross_section_to_media(cross_section)(
        frequency=skrf.Frequency.from_f(np.array(f_flat), unit="Hz")
    )  # type: ignore

    kwargs = {"d": length, "unit": "um"}
    transmission_line = media.line(**kwargs)
    quarter_wave_resonator = transmission_line ** media.short()
    kwargs = {"C": coupling_capacitance, "name": "C_coupling"}
    coupling_capacitor = media.capacitor(**kwargs)

    # Create tee junction for parallel capacitor connection
    resonator_tee = media.tee()
    # Connect capacitor to port 1 and resonator to port 2, leaving port 0 open
    resonator_with_cap = skrf.connect(resonator_tee, 1, coupling_capacitor, 0)
    resonator_coupled = skrf.connect(resonator_with_cap, 1, quarter_wave_resonator, 0)

    kwargs = {"d": coupling_straight_length / 2, "unit": "um"}
    probeline_factory = partial(media.line, **kwargs)
    probeline = skrf.connect(
        skrf.connect(probeline_factory(), 1, media.tee(), 0), 2, probeline_factory(), 0
    )
    all_network = skrf.connect(probeline, 1, resonator_coupled, 0)

    ports = ["coupling_o1", "coupling_o2", "resonator_o1"]
    sdict = {
        (ports[i], ports[j]): jnp.array(all_network.s[:, i, j])
        for i in range(len(ports))
        for j in range(i, len(ports))
    }
    return sax.reciprocal({k: v.reshape(*f.shape) for k, v in sdict.items()})


def resonator_frequency(
    *,
    length: float,
    media: Media,
    is_quarter_wave: bool = True,
) -> NDArray:
    r"""Calculate the resonance frequency of a quarter-wave resonator.

    .. math::

        f &= \frac{v_p}{4L}  \text{ (quarter-wave resonator)} \\
        f &= \frac{v_p}{2L}  \text{ (half-wave resonator)}

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
