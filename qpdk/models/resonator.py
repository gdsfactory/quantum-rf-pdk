"""Resonator models."""

from functools import partial

import jax.numpy as jnp
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from numpy.typing import NDArray
from skrf.media import Media

from qpdk.models.couplers import cpw_cpw_coupling_capacitance
from qpdk.models.media import cross_section_to_media


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


# TODO for some reason this given different results to QucS and skrf.
#
# def resonator_coupled(
#     f: ArrayLike = jnp.array([5e9]),
#     cross_section: CrossSectionSpec = "cpw",
#     coupling_gap: int | float = 0.27,
#     coupling_length: float = 20,
#     length: float = 5000,
# ) -> sax.SDict:
#     """Model for a quarter-wave coplanar waveguide resonator coupled to a probeline.
#
#     ```{svgbob}
#
#     Todo:
#     ```
#
#     Args:
#         cross_section: The cross-section of the CPW.
#         f: Frequency in Hz at which to evaluate the S-parameters.
#         coupling_gap: Gap between the resonator and the probeline in μm.
#         coupling_length: Length of the coupling section in μm.
#         length: Total length of the resonator in μm.
#
#     Returns:
#         sax.SDict: S-parameters dictionary
#     """
#     circuit, _ = sax.circuit(
#         netlist={
#             "instances": {
#                 "coupler": {
#                     "component": "coupler_straight",
#                     "settings": {
#                         "f": f,
#                         "length": coupling_length,
#                         "gap": coupling_gap,
#                         "cross_section": cross_section,
#                     },
#                 },
#                 "resonator": {
#                     "component": "straight_shorted",
#                     "settings": {
#                         "f": f,
#                         "length": length - coupling_length,
#                         "cross_section": cross_section,
#                     },
#                 },
#             },
#             "connections": {
#                 "coupler,o4": "resonator,o1",
#             },
#             "ports": {
#                 "o1": "coupler,o1",
#                 "o2": "coupler,o2",
#                 "o3": "coupler,o3",
#                 # "o4": "resonator,o2",
#             },
#         },
#         models={
#             "straight_shorted": straight_shorted,
#             "coupler_straight": coupler_straight,
#         },
#     )
#
#     return circuit(f=f)


def quarter_wave_resonator_coupled(
    f: ArrayLike = jnp.array([5e9]),
    cross_section: CrossSectionSpec = "cpw",
    length: float = 5000,
    coupling_gap: int | float = 0.27,
    coupling_straight_length: float = 20,
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
    f = f if f is not None else jnp.array([1e9, 5e9])
    coupling_capacitance = cpw_cpw_coupling_capacitance(
        length, coupling_gap, cross_section, f
    )
    media: Media = cross_section_to_media(cross_section)(
        frequency=skrf.Frequency.from_f(f, unit="Hz")
    )  # type: ignore

    transmission_line = media.line(d=length, unit="um")
    quarter_wave_resonator = transmission_line ** media.short()
    coupling_capacitor = media.capacitor(coupling_capacitance, name="C_coupling")

    # Create tee junction for parallel capacitor connection
    resonator_tee = media.tee()
    # Connect capacitor to port 1 and resonator to port 2, leaving port 0 open
    resonator_with_cap = skrf.connect(resonator_tee, 1, coupling_capacitor, 0)
    resonator_coupled = skrf.connect(resonator_with_cap, 1, quarter_wave_resonator, 0)

    probeline_factory = partial(media.line, d=coupling_straight_length / 2, unit="um")
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
    return sax.reciprocal(sdict)


if __name__ == "__main__":
    from qpdk.tech import coplanar_waveguide

    cs = coplanar_waveguide(width=10, gap=6)
    cpw = cross_section_to_media(cs)(frequency=skrf.Frequency(2, 9, 101, unit="GHz"))
    print(f"{cpw=!r}")
    print(f"{cpw.z0.mean().real=!r}")  # Characteristic impedance

    res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
    print("Resonance frequency (quarter-wave):", res_freq / 1e9, "GHz")

    # Plot resonator_coupled example
    f = jnp.linspace(0.1e9, 9e9, 1001)
    resonator = quarter_wave_resonator_coupled(
        f=f,
        cross_section=cs,
        coupling_gap=0.27,
        length=4000,
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key in [("o2", "o3"), ("o1", "o2"), ("o1", "o3")]:
        ax.plot(f / 1e9, 20 * jnp.log10(jnp.abs(resonator[key])), label=f"$S${key}")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(r"$S$-parameters: $\mathtt{resonator\_coupled}$ (3-port)")
    ax.grid(True, which="both")
    ax.legend()

    plt.show()
