from functools import partial
from typing import Callable

import jax.numpy as jnp
import sax
import skrf
from matplotlib import pyplot as plt
from skrf.media import CPW, Media
from skrf.network import connect

from qpdk import PDK
from qpdk.tech import material_properties


def cpw_media_skrf(width: float, gap: float) -> partial[CPW]:
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
        h=PDK.layer_stack.layers["Substrate"].thickness * 1e-6,
        t=PDK.layer_stack.layers["M1"].thickness * 1e-6,
        ep_r=material_properties[PDK.layer_stack.layers["Substrate"].material][
            "relative_permittivity"
        ],
    )


def resonator_frequency(
    length: float, media: Media, is_quarter_wave: bool = True
) -> float:
    """Calculate the resonance frequency of a quarter-wave resonator.

    .. math::

        f = \\frac{v_p}{4L}  \\text{ (quarter-wave resonator)}
        f = \\frac{v_p}{2L}  \\text{ (half-wave resonator)}

    See :cite:`m.pozarMicrowaveEngineering2012` for details.

    Args:
        length: Length of the resonator in μm.
        media: skrf media object defining the CPW (or other) properties.
        is_quarter_wave: If True, calculates for a quarter-wave resonator; if False, for a half-wave resonator.
            default is True.

    Returns:
        float: Resonance frequency in Hz.
    """
    coefficient = 4 if is_quarter_wave else 2  # Quarter-wave resonator
    return media.v_p / (coefficient * length * 1e-6)


def quarter_wave_resonator(
    media: Callable[[jnp.ndarray], Media],
    f: jnp.ndarray = [1e9, 5e9],
    length: float = 4000,
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator.

    Args:
        media: skrf media object defining the CPW (or other) properties.
        f: Frequency in Hz at which to evaluate the S-parameters.
        length: Length of the resonator in μm.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    media = media(frequency=skrf.Frequency.from_f(f, unit="Hz"))
    transmission_line = media.line(d=length * 1e-6, unit="um")
    quarter_wave_resonator = transmission_line ** media.short()
    quarter_wave_resonator.name = "quarter_wave_resonator"
    probeline_in = media.line(d=5000/2, unit="um")
    probeline_out = media.line(d=5000/2, unit="um")
    coupling_capacitor = media.capacitor(60e-15, name="C_coupling")
    all_network = skrf.parallelconnect(
        [quarter_wave_resonator, coupling_capacitor], [0, 0], name="resonator_coupled"
    )
    all_network = skrf.parallelconnect([probeline_in, coupling_capacitor], [0, 0], name="probeline_with_coupled_resonator")
    all_network = all_network ** probeline_out

    sdict = {
        ("o1", "o1"): jnp.array(all_network.s[:, 0, 0]),
        ("o1", "o2"): jnp.array(all_network.s[:, 0, 1]),
    }
    return sax.reciprocal(sdict)


if __name__ == "__main__":
    cpw = cpw_media_skrf(width=10, gap=6)()
    print(cpw.z0)
    print(cpw.v_p)

    res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
    print(res_freq)

    circuit, info = sax.circuit(
        netlist={
            "instances": {
                "R1": "quarter_wave_resonator",
            },
            "connections": {},
            "ports": {
                "in": "R1,o1",
                "out": "R1,o2",
            },
        },
        models={
            "quarter_wave_resonator": partial(
                quarter_wave_resonator,
                media=cpw_media_skrf(width=10, gap=6),
                length=4000,
            )
        },
    )

    frequencies = jnp.linspace(1e9, 10e9, 501)
    S = circuit(f=frequencies)
    print(S)
    print(info)
    plt.plot(frequencies / 1e9, abs(S["in", "out"]) ** 2)
    # plt.ylim(-0.05, 1.05)
    plt.xlabel("f [GHz]")
    plt.ylabel("T")
    # plt.ylim(-0.05, 1.05)
    plt.show()
