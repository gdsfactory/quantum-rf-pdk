# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Resonator Test Chip
#
# This example demonstrates creating a resonator test chip for characterizing superconducting microwave resonators.
#
# The design is inspired by {cite:p}`norrisImprovedParameterTargeting2024`.

# %%
import gdsfactory as gf
import numpy as np

from qpdk import tech
from qpdk.cells._schematic import sax_model, schematic
from qpdk.cells.chip import chip_edge
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import quarter_wave_resonator_coupled
from qpdk.cells.waveguides import straight
from qpdk.logger import logger
from qpdk.tech import (
    route_bundle_cpw,
    route_bundle_sbend,
)
from qpdk.utils import fill_magnetic_vortices


def _name_new_route_instance(
    component: gf.Component,
    existing_names: set[str],
    name: str,
) -> None:
    """Name the single instance added by a one-connection route bundle."""
    new_instances = [
        instance for instance in component.insts if instance.name not in existing_names
    ]
    if len(new_instances) != 1:
        raise RuntimeError(
            f"Expected route {name!r} to add one instance, got {len(new_instances)}"
        )
    new_instances[0].name = name


# %% [markdown]
# ## Resonator Test Chip Function
#
# Creates a test chip with two probelines and multiple resonators for characterization.


# %%
resonator_test_chip_python_schematic = schematic(
    symbol="resonator_test_chip_python",
    tags=["samples", "resonators"],
    ports=[
        {"name": "o1", "side": "left", "type": "photonic"},
        {"name": "o3", "side": "left", "type": "photonic"},
        {"name": "o2", "side": "right", "type": "photonic"},
        {"name": "o4", "side": "right", "type": "photonic"},
    ],
    models=[
        sax_model(
            name="resonator_test_chip_python",
            module="qpdk.models.resonator",
            qualname="resonator_test_chip_python",
            port_order=["o1", "o2", "o3", "o4"],
        )
    ],
)


@gf.cell(schematic_function=resonator_test_chip_python_schematic)
def resonator_test_chip_python(
    probeline_length: float = 9000.0,
    probeline_separation: float = 1000.0,
    resonator_length: float = 4000.0,
    coupling_length: float = 200.0,
    coupling_gap: float = 16.0,
) -> gf.Component:
    """Creates a resonator test chip with two probelines and 16 resonators.

    The chip features two horizontal probelines running west to east, each with
    launchers on both ends. Eight quarter-wave resonators are coupled to each
    probeline, with systematically varied lengths for characterization studies.

    Args:
        probeline_length: Length of each probeline in µm.
        probeline_separation: Vertical separation between probelines in µm.
        resonator_length: Nominal resonator length in µm. Resonator lengths are
            varied around this value to separate their resonance frequencies.
        coupling_length: Length of coupling region between resonator and probeline in µm.
        coupling_gap: Gap between resonator and probeline for coupling in µm.

    Returns:
        Component: A gdsfactory component containing the complete test chip layout.
    """
    c = gf.Component()

    # Use a registered cross-section name so serialized netlists can resolve it.
    cross_section = "coplanar_waveguide"

    # CPW width and gap barely shift effective permittivity. Vary length instead
    # to create distinct resonances while keeping all model cross-sections named.
    n_resonators_total = 16
    n_resonators_per_probeline = n_resonators_total // 2
    resonator_lengths = np.linspace(
        0.9 * resonator_length,
        1.275 * resonator_length,
        n_resonators_total,
    )

    probeline_y_positions = [0, probeline_separation]

    for probeline_idx, y_pos in enumerate(probeline_y_positions):
        probeline_name = "bot" if probeline_idx == 0 else "top"

        # Add launchers at both ends
        launcher_west = c.add_ref(launcher(), name=f"probe_west_{probeline_name}")
        launcher_west.move((0, y_pos))
        launcher_east = c.add_ref(launcher(), name=f"probe_east_{probeline_name}")
        launcher_east.mirror_x()
        launcher_east.move((probeline_length, y_pos))

        # Expose launcher waveports for circuit simulation.
        port_names = ("o3", "o4") if probeline_idx == 0 else ("o1", "o2")
        c.add_port(port_names[0], port=launcher_west.ports["waveport"])
        c.add_port(port_names[1], port=launcher_east.ports["waveport"])

        # Add resonators along the probeline
        lengths = resonator_lengths[probeline_idx::2]
        resonator_spacing = probeline_length / (n_resonators_per_probeline + 1)

        previous_port = launcher_west.ports["o1"]
        for res_idx in range(n_resonators_per_probeline):
            # Calculate resonator position along probeline
            x_position = (res_idx + 1) * resonator_spacing

            # Create quarter-wave resonator with unique length.
            coupled_resonator = quarter_wave_resonator_coupled(
                length=float(lengths[res_idx]),
                meanders=6,
                cross_section=cross_section,
                open_start=True,
                cross_section_non_resonator=cross_section,
                coupling_straight_length=coupling_length,
                coupling_gap=coupling_gap,
            )
            resonator_ref = c.add_ref(
                coupled_resonator,
                name=f"resonator_{probeline_name}_{res_idx + 1}",
            )
            # Position resonator above probeline
            if probeline_idx != 0:
                resonator_ref.mirror_y()

            # The coupled resonator origin is its ``coupling_o1`` port, so place
            # that port directly at the intended probeline position.
            resonator_ref.move((x_position, y_pos))
            logger.debug(f"Added resonator {res_idx} at x={x_position} µm")

            if res_idx == 0:
                # Add some straight before connecting the first resonator
                first_straight_ref = c.add_ref(
                    straight(length=200.0, cross_section=cross_section),
                    name=f"probeline_straight_west_{probeline_name}",
                )
                first_straight_ref.connect("o1", resonator_ref.ports["coupling_o1"])
                existing_names = {instance.name for instance in c.insts}
                route_bundle_sbend(
                    c,
                    ports1=[previous_port],
                    ports2=[first_straight_ref.ports["o2"]],
                    cross_section=cross_section,
                )
                _name_new_route_instance(
                    c,
                    existing_names,
                    f"probeline_sbend_west_{probeline_name}",
                )
            else:
                existing_names = {instance.name for instance in c.insts}
                route_bundle_cpw(
                    c,
                    ports1=[previous_port],
                    ports2=[resonator_ref.ports["coupling_o1"]],
                    cross_section=cross_section,
                )
                _name_new_route_instance(
                    c,
                    existing_names,
                    f"probeline_straight_{probeline_name}_{res_idx}",
                )

            previous_port = resonator_ref.ports["coupling_o2"]

        # Add some straight before connecting to the final launcher
        final_straight_ref = c.add_ref(
            straight(length=400.0, cross_section=cross_section),
            name=f"probeline_straight_east_{probeline_name}",
        )
        final_straight_ref.connect("o1", previous_port)

        # Connect final launcher to probeline
        existing_names = {instance.name for instance in c.insts}
        route_bundle_sbend(
            c,
            ports1=[final_straight_ref.ports["o2"]],
            ports2=[launcher_east.ports["o1"]],
            cross_section=cross_section,
        )
        _name_new_route_instance(
            c,
            existing_names,
            f"probeline_sbend_east_{probeline_name}",
        )

    return c


resonator_test_chip_python.schematic_function = resonator_test_chip_python_schematic


# %% [markdown]
# ## Filled Resonator Test Chip
#
# Version of the test chip with magnetic vortex trapping holes in the ground plane.


# %%
@gf.cell
def filled_resonator_test_chip() -> gf.Component:
    """Creates a resonator test chip filled with magnetic vortex trapping holes.

    This version includes the complete resonator test chip layout with additional
    ground plane holes to trap magnetic vortices, improving the performance of
    superconducting quantum circuits. Includes chip edge components with extra
    y-padding to keep resonators away from the chip edges.

    Returns:
        Component: Test chip with ground plane fill patterns and chip edges.
    """
    c = gf.Component()
    test_chip = resonator_test_chip_python()
    c << test_chip
    chip_edge_ref = c << chip_edge(
        size=(test_chip.xsize + 200, test_chip.ysize + 800),
        width=100.0,
        layer=tech.LAYER.M1_ETCH,
    )
    chip_edge_ref.move((test_chip.xmin - 100, test_chip.ymin - 400))
    return fill_magnetic_vortices(
        component=c,
        rectangle_size=(15.0, 15.0),
        gap=70.0,
        stagger=2,
    )


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # Create and display the filled version
    filled_chip = filled_resonator_test_chip()
    filled_chip.show()
