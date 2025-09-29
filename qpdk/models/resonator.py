"""Resonator models."""

import inspect
from collections.abc import Callable
from functools import partial
from typing import cast

import jax.numpy as jnp
import sax
import skrf
from jax._src.util import Array
from numpy.typing import NDArray
from skrf.media import CPW, Media

from qpdk import LAYER_STACK
from qpdk.tech import coplanar_waveguide, material_properties

_coplanar_waveguide_xsection_signature = inspect.signature(coplanar_waveguide)


def cpw_media_skrf(
    width: float = _coplanar_waveguide_xsection_signature.parameters["width"].default,
    gap: float = _coplanar_waveguide_xsection_signature.parameters["gap"].default,
) -> partial[CPW]:
    """Create a partial coplanar waveguide (CPW) media object using scikit-rf.

    Args:
        width: Width of the center conductor in μm.
        gap: Width of the gap between the center conductor and ground planes in μm.

    Returns:
        partial[skrf.media.CPW]: A CPW media object with specified dimensions.
    """
    # Convert μm to m for skrf
    return partial(
        CPW,
        w=width * 1e-6,
        s=gap * 1e-6,
        h=LAYER_STACK.layers["Substrate"].thickness * 1e-6,
        t=LAYER_STACK.layers["M1"].thickness * 1e-6,
        ep_r=material_properties[cast(str, LAYER_STACK.layers["Substrate"].material)][
            "relative_permittivity"
        ],
        rho=1e-100,  # set to a very low value to avoid warnings
        tand=0,
    )


def resonator_frequency(
    length: float, media: Media, is_quarter_wave: bool = True
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


def quarter_wave_resonator_coupled_to_probeline(
    media: Callable[[skrf.Frequency], Media],
    f: skrf.NumberLike | Array | None = None,
    coupling_capacitance: float = 15e-15,
    length: float = 4000,
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.

    Args:
        media: skrf media object defining the CPW (or other) properties.
        f: Frequency in Hz at which to evaluate the S-parameters.
        coupling_capacitance: Coupling capacitance in Farads.
        length: Length of the resonator in μm.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f = f if f is not None else jnp.array([1e9, 5e9])
    media: Media = media(frequency=skrf.Frequency.from_f(f, unit="Hz"))  # type: ignore

    transmission_line = media.line(d=length, unit="um")
    quarter_wave_resonator = transmission_line ** media.short()
    coupling_capacitor = media.capacitor(coupling_capacitance, name="C_coupling")
    resonator_coupled = coupling_capacitor**quarter_wave_resonator
    probeline_factory = partial(media.line, d=5000, unit="um")
    probeline = skrf.connect(
        skrf.connect(probeline_factory(), 1, media.tee(), 0), 2, probeline_factory(), 0
    )
    all_network = skrf.connect(probeline, 1, resonator_coupled, 0)

    sdict = {
        ("o1", "o1"): jnp.array(all_network.s[:, 0, 0]),
        ("o1", "o2"): jnp.array(all_network.s[:, 0, 1]),
    }
    return sax.reciprocal(sdict)


if __name__ == "__main__":
    cpw = cpw_media_skrf(width=10, gap=6)(
        frequency=skrf.Frequency(2, 9, 101, unit="GHz")
    )
    print(f"{cpw=!r}")
    print(f"{cpw.z0.mean().real=!r}")  # Characteristic impedance

    res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
    print("Resonance frequency (quarter-wave):", res_freq / 1e9, "GHz")
