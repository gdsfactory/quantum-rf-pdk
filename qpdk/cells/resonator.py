"""Resonator components."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.waveguides import bend_circular, straight
from qpdk.helper import show_components
from qpdk.tech import get_etch_section


@gf.cell
def resonator(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    open_start: bool = True,
    open_end: bool = False,
) -> Component:
    """Creates a meandering coplanar waveguide resonator.

    Changing `open_start` and `open_end` appropriately allows creating
    a shorted quarter-wave resonator or an open half-wave resonator.

    .. svgbob::

        o1 ─────┐
                │
        ┌───────┘
        │
        └───────┐
                │
        ┌───────┘
        │
        └────── o2

    See :cite:`m.pozarMicrowaveEngineering2012` for details

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        start_with_bend: If True, starts the resonator with a bend.
        end_with_bend: If True, ends the resonator with a bend.
        open_start: If True, adds an etch section at the start of the resonator.
        open_end: If True, adds an etch section at the end of the resonator.

    Returns:
        Component: A gdsfactory component with meandering resonator geometry.
    """
    c = Component()
    cross_section = gf.get_cross_section(cross_section)
    bend = gf.get_component(
        bend_spec, cross_section=cross_section, angle=180, angular_step=4
    )

    num_straights = meanders + 1
    if start_with_bend:
        num_straights -= 1
    if end_with_bend:
        num_straights -= 1

    if num_straights < 0:
        raise ValueError(
            "Cannot have fewer than 0 straight sections. Reduce meanders or adjust bend start/end settings."
        )

    straight_comp = None
    if num_straights > 0:
        length_per_one_straight = (
            length - meanders * bend.info["length"]
        ) / num_straights

        if length_per_one_straight <= 0:
            raise ValueError(
                f"Resonator length {length} is too short for {meanders} meanders with current bend spec {bend}. "
                f"Increase length, reduce meanders, or change the bend spec."
            )

        straight_comp = straight(
            length=length_per_one_straight,
            cross_section=cross_section,
        )

    # Route meandering quarter-wave resonator
    previous_port = None
    first_ref = None
    last_ref = None

    for i in range(meanders):
        # Determine if we should add a straight before this bend
        if i == 0 and start_with_bend:
            # First element is a bend
            bend_ref = c.add_ref(bend)
            if i % 2 == 0:
                bend_ref.mirror()
                bend_ref.rotate(90)
            first_ref = bend_ref
            previous_port = bend_ref.ports["o2"]
        else:
            if straight_comp is None:
                raise ValueError("straight_comp is required but not initialized.")
            straight_ref = c.add_ref(straight_comp)
            if i == 0:
                first_ref = straight_ref
            else:
                straight_ref.connect("o1", previous_port)

            bend_ref = c.add_ref(bend)
            if i % 2 == 0:
                bend_ref.mirror()
                bend_ref.rotate(90)

            bend_ref.connect("o1", straight_ref.ports["o2"])
            previous_port = bend_ref.ports["o2"]

        last_ref = bend_ref

    # Final section
    if not end_with_bend:
        if straight_comp is None:
            raise ValueError("straight_comp is required but not initialized.")
        final_straight_ref = c.add_ref(straight_comp)
        if previous_port:
            final_straight_ref.connect("o1", previous_port)
        last_ref = final_straight_ref
        if first_ref is None:
            first_ref = final_straight_ref

    if first_ref is None or last_ref is None:
        raise ValueError("Resonator could not be generated correctly.")

    actual_length = meanders * bend.info["length"]
    if num_straights > 0:
        if straight_comp is None:
            raise ValueError("straight_comp is required but not initialized.")
        actual_length += num_straights * straight_comp.info["length"]

    # Etch at the open end
    if open_end or open_start:
        cross_section_etch_section = get_etch_section(cross_section)

        open_etch_comp = gf.c.rectangle(
            size=(
                cross_section_etch_section.width,
                2 * cross_section_etch_section.width + cross_section.width,
            ),
            layer=cross_section_etch_section.layer,
            centered=True,
            port_type="optical",
            port_orientations=(0, 180),
        )

        def _add_etch_at_port(port_name, ref_port, output_port):
            """Helper function to add etch at a specific port."""
            open_etch = c.add_ref(open_etch_comp)
            open_etch.connect(
                port_name,
                ref_port,
                allow_width_mismatch=True,
                allow_layer_mismatch=True,
            )
            c.add_port(
                output_port, port=open_etch.ports[output_port], port_type="placement"
            )

        if open_end:
            _add_etch_at_port("o1", last_ref.ports["o2"], "o2")
        if open_start:
            _add_etch_at_port("o2", first_ref.ports["o1"], "o1")

    if not open_end:
        c.add_port("o2", port=last_ref.ports["o2"])

    if not open_start:
        c.add_port("o1", port=first_ref.ports["o1"])

    # Add metadata
    c.info["length"] = actual_length
    c.info["resonator_type"] = "quarter_wave"
    c.info["cross_section"] = cross_section.name
    # c.info["frequency_estimate"] = (
    #     3e8 / (4 * length * 1e-6) / 1e9
    # )  # GHz, rough estimate

    return c


# A quarter-wave resonator is shorted at one end and has maximum electric field
# at the open end, making it suitable for capacitive coupling.
resonator_quarter_wave = partial(resonator, open_start=False, open_end=True)
# A half-wave resonator is open at both ends
resonator_half_wave = partial(resonator, open_start=True, open_end=True)

# Quarter-wave resonator starting with a bend
resonator_quarter_wave_bend_start = partial(
    resonator_quarter_wave, start_with_bend=True
)
# Half-wave resonator starting with a bend
resonator_half_wave_bend_start = partial(resonator_half_wave, start_with_bend=True)

# Resonator ending with a bend
resonator_quarter_wave_bend_end = partial(resonator_quarter_wave, end_with_bend=True)
resonator_half_wave_bend_end = partial(resonator_half_wave, end_with_bend=True)

# Both
resonator_quarter_wave_bend_both = partial(
    resonator_quarter_wave, start_with_bend=True, end_with_bend=True
)
resonator_half_wave_bend_both = partial(
    resonator_half_wave, start_with_bend=True, end_with_bend=True
)


@gf.cell
def resonator_coupled(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    open_start: bool = True,
    open_end: bool = False,
    cross_section_non_resonator: CrossSectionSpec = "cpw",
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 20.0,
) -> Component:
    """Creates a meandering coplanar waveguide resonator with a coupling waveguide.

    This component combines a resonator with a parallel coupling waveguide placed
    at a specified gap for proximity coupling. Similar to the design described in
    :cite:`besedinQualityFactorTransmission2018a`.

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        start_with_bend: If True, starts the resonator with a bend.
        end_with_bend: If True, ends the resonator with a bend.
        open_start: If True, adds an etch section at the start of the resonator.
        open_end: If True, adds an etch section at the end of the resonator.
        cross_section_non_resonator: Cross-section specification for the coupling waveguide.
        coupling_straight_length: Length of the coupling waveguide section in μm.
        coupling_gap: Gap between the resonator and coupling waveguide in μm.
            Measured from edges of the center conductors.

    Returns:
        Component: A gdsfactory component with meandering resonator and coupling waveguide.
    """
    c = Component()

    resonator_ref = c.add_ref(
        resonator(
            length=length,
            meanders=meanders,
            bend_spec=bend_spec,
            cross_section=cross_section,
            start_with_bend=start_with_bend,
            end_with_bend=end_with_bend,
            open_start=open_start,
            open_end=open_end,
        )
    )

    cross_section_obj = gf.get_cross_section(cross_section_non_resonator)

    coupling_wg = straight(
        length=coupling_straight_length,
        cross_section=cross_section_obj,
    )
    coupling_ref = c.add_ref(coupling_wg)

    # Position coupling waveguide parallel to resonator with specified gap
    coupling_ref.movey(coupling_gap + cross_section_obj.width)

    coupling_ref.xmin = resonator_ref["o1"].x  # Align left edges

    for port in resonator_ref.ports:
        port_type = (
            "placement"
            if ((port.name == "o1" and open_start) or (port.name == "o2" and open_end))
            else "optical"
        )
        c.add_port(f"resonator_{port.name}", port=port, port_type=port_type)

    for port in coupling_ref.ports:
        c.add_port(f"coupling_{port.name}", port=port)

    c.info += resonator_ref.cell.info
    c.info["coupling_length"] = coupling_straight_length
    c.info["coupling_gap"] = coupling_gap

    return c


@gf.cell
def quarter_wave_resonator_coupled(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    open_start: bool = True,
    open_end: bool = False,
    cross_section_non_resonator: CrossSectionSpec = "cpw",
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 20.0,
) -> Component:
    """Creates a quarter-wave resonator with a coupling waveguide.

    Uses :func:`~qpdk.cells.resonator.resonator_coupled` as the basis but
    removes the shorted end port from the output ports.

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        start_with_bend: If True, starts the resonator with a bend.
        end_with_bend: If True, ends the resonator with a bend.
        open_start: If True, adds an etch section at the start of the resonator.
        open_end: If True, adds an etch section at the end of the resonator.
        cross_section_non_resonator: Cross-section specification for the coupling waveguide.
        coupling_straight_length: Length of the coupling waveguide section in μm.
        coupling_gap: Gap between the resonator and coupling waveguide in μm.
    """
    c = Component()

    res_ref = c << resonator_coupled(
        length=length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        start_with_bend=start_with_bend,
        end_with_bend=end_with_bend,
        open_start=open_start,
        open_end=open_end,
        cross_section_non_resonator=cross_section_non_resonator,
        coupling_straight_length=coupling_straight_length,
        coupling_gap=coupling_gap,
    )
    movement = np.array(res_ref.ports["coupling_o1"].center)
    res_ref.move(tuple(-movement))

    for port in res_ref.ports:
        if port.name != "resonator_o2":  # Skip the shorted end port
            c.add_port(port=port)

    return c


if __name__ == "__main__":
    show_components(
        resonator,
        resonator_quarter_wave,
        resonator_half_wave,
        resonator_quarter_wave_bend_start,
        resonator_quarter_wave_bend_both,
        resonator_coupled,
        partial(
            resonator_coupled,
            length=2000,
            meanders=4,
            open_start=False,
            open_end=True,
        ),
    )
