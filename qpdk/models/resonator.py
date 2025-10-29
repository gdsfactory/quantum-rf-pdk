"""Resonator models."""

import jax.numpy as jnp
import sax
import skrf
from jax.typing import ArrayLike
from numpy.typing import NDArray
from skrf.media import Media

from qpdk.models.couplers import coupler_straight
from qpdk.models.media import MediaCallable, cpw_media_skrf
from qpdk.models.waveguides import straight_shorted


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


def resonator_coupled(
    f: ArrayLike = jnp.array([5e9]),
    media: MediaCallable = cpw_media_skrf(),
    coupling_gap: int | float = 0.27,
    coupling_length: float = 20,
    length: float = 5000,
) -> sax.SDict:
    """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.

    ```{svgbob}

    Todo:
    ```

    Args:
        media: skrf media object defining the CPW (or other) properties.
        f: Frequency in Hz at which to evaluate the S-parameters.
        coupling_gap: Gap between the resonator and the probeline in μm.
        coupling_length: Length of the coupling section in μm.
        length: Total length of the resonator in μm.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler": {
                    "component": "coupler_straight",
                    "settings": {
                        "f": f,
                        "length": coupling_length,
                        "gap": coupling_gap,
                        "media": media,
                    },
                },
                "resonator": {
                    "component": "straight_shorted",
                    "settings": {
                        "f": f,
                        "length": length - coupling_length,
                        "media": media,
                    },
                },
            },
            "connections": {
                "coupler,o4": "resonator,o1",
            },
            "ports": {
                "o1": "coupler,o1",
                "o2": "coupler,o2",
                "o3": "coupler,o3",
                # "o4": "resonator,o2",
            },
        },
        models={
            "straight_shorted": straight_shorted,
            "coupler_straight": coupler_straight,
        },
    )

    return circuit(f=f)


if __name__ == "__main__":
    cpw = cpw_media_skrf(width=10, gap=6)(
        frequency=skrf.Frequency(2, 9, 101, unit="GHz")
    )
    print(f"{cpw=!r}")
    print(f"{cpw.z0.mean().real=!r}")  # Characteristic impedance

    res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
    print("Resonance frequency (quarter-wave):", res_freq / 1e9, "GHz")

    # Plot resonator_coupled example
    f = jnp.linspace(0.5e9, 9e9, 1001)
    resonator = resonator_coupled(
        f=f,
        media=cpw_media_skrf(width=10, gap=6),
        coupling_gap=0.27,
        length=4000,
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key in resonator:
        ax.plot(f / 1e9, 20 * jnp.log10(jnp.abs(resonator[key])), label=f"$S${key}")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(r"$S$-parameters: $\mathtt{resonator\_coupled}$")
    ax.grid(True, which="both")
    ax.legend()

    plt.show()
