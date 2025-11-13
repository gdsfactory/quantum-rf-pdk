"""S-parameter models for couplers."""

import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from skrf import Frequency

from qpdk.models.generic import capacitor, tee
from qpdk.models.media import cross_section_to_media
from qpdk.models.waveguides import straight


def cpw_cpw_coupling_capacitance(
    length: float,
    gap: float,
    cross_section: CrossSectionSpec,
    f: ArrayLike = jnp.array([5e9]),
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
    # Create a media instance to extract parameters. Frequency doesn't matter for geometry.
    media = cross_section_to_media(cross_section)
    media_instance = media(frequency=Frequency.from_f(f, unit="Hz"))
    ep_r = media_instance.ep_r  # noqa: F841
    w_m = getattr(media_instance, "w", 10e-6)  # noqa: F841
    s_m = getattr(media_instance, "s", 6e-6)  # noqa: F841
    length_m = length * 1e-6  # noqa: F841
    gap_m = gap * 1e-6  # noqa: F841

    # TODO: Find a paper with some values

    # TODO hardcoded placeholder value
    return 60e-15


def coupler_straight(
    f: ArrayLike = jnp.array([5e9]),
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
    straight_settings = {"length": length / 2, "cross_section": cross_section}
    capacitor_settings = {
        "capacitance": cpw_cpw_coupling_capacitance(
            length, gap, cross_section, f
        ),  # gap * 1e-18 * f,  # TODO implement FEM simulation retrieval or use some paper
        "z0": cross_section_to_media(cross_section)(
            frequency=Frequency.from_f(f, unit="Hz")
        ).z0,
    }

    # Create straight instances with shared settings
    straight_instances = {
        f"straight_{i}_{j}": {
            "component": "straight",
            "settings": straight_settings,
        }
        for i in [1, 2]
        for j in [1, 2]
    }
    tee_instances = {f"tee_{i}": {"component": "tee"} for i in [1, 2]}

    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                **straight_instances,
                **tee_instances,
                "capacitor": {
                    "component": "capacitor",
                    "settings": capacitor_settings,
                },
            },
            "connections": {
                "straight_1_1,o1": "tee_1,o1",
                "straight_1_2,o1": "tee_1,o2",
                "straight_2_1,o1": "tee_2,o1",
                "straight_2_2,o1": "tee_2,o2",
                "tee_1,o3": "capacitor,o1",
                "tee_2,o3": "capacitor,o2",
            },
            "ports": {
                "o2": "straight_1_1,o2",
                "o3": "straight_1_2,o2",
                "o1": "straight_2_1,o2",
                "o4": "straight_2_2,o2",
            },
        },
        models={
            "straight": straight,
            "capacitor": capacitor,
            "tee": tee,
        },
    )

    return circuit(f=f)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Define frequency range from 1 GHz to 10 GHz with 201 points
    f = jnp.linspace(1e9, 10e9, 201)

    # Calculate coupler S-parameters for a 20 um straight coupler with 0.27 um gap
    coupler = coupler_straight(f=f, length=20, gap=0.27)

    # Create figure with single plot for comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Define S-parameters to plot
    s_params = [
        (("o1", "o1"), "$S_{11}$ Reflection"),
        (("o1", "o2"), "$S_{12}$ Coupled branch 1"),
        (("o1", "o3"), "$S_{13}$ Coupled branch 2"),
        (("o1", "o4"), "$S_{14}$ Insertion loss (direct through)"),
    ]

    # Plot each S-parameter for both coupler implementations
    default_color_cycler = plt.cm.tab10.colors
    for idx, (ports, label) in enumerate(s_params):
        color = default_color_cycler[idx % len(default_color_cycler)]
        # Plot both implementations with same color but different linestyles
        ax.plot(
            f / 1e9,
            20 * jnp.log10(jnp.abs(coupler[ports])),
            linestyle="-",
            color=color,
            label=f"{label} coupler_straight",
        )

    # Configure plot
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("$S$-parameter [dB]")
    ax.set_title(r"$S$-parameters: $\mathtt{coupler\_straight}$")
    ax.grid(True, which="both")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Example calculation of coupling capacitance
    from qpdk.tech import coplanar_waveguide

    cs = coplanar_waveguide(width=10, gap=6)
    coupling_capacitance = cpw_cpw_coupling_capacitance(
        length=20.0, gap=0.27, cross_section=cs, f=f
    )
    print(
        "Coupling capacitance for 20 um length and 0.27 um gap:",
        coupling_capacitance,
        "F",
    )
