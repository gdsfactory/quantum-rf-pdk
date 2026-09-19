"""Resonator components."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells._schematic import (
    quarter_wave_resonator_coupled_schematic,
    resonator_coupled_schematic,
    resonator_half_wave_schematic,
    resonator_quarter_wave_schematic,
    resonator_schematic,
)
from qpdk.cells.resonator_sections import resonator_meander
from qpdk.cells.waveguides import (
    bend_circular,
    coupler_straight,
    open_2_port,
    short,
    straight,
)
from qpdk.tech import get_etch_section


def _resonator_straight(
    length: float,
    meanders: int,
    bend: Component,
    cross_section: CrossSectionSpec,
    *,
    start_with_bend: bool,
    end_with_bend: bool,
) -> tuple[int, Component | None, float]:
    """Return the snapped straight section and realized resonator length."""
    straight_count = meanders + 1 - int(start_with_bend) - int(end_with_bend)
    if straight_count < 0:
        raise ValueError(
            "Cannot have fewer than 0 straight sections. Reduce meanders or adjust bend start/end settings."
        )

    bend_length = meanders * bend.info["length"]
    if straight_count == 0:
        return straight_count, None, bend_length

    straight_length = (length - bend_length) / straight_count
    if straight_length <= 0:
        raise ValueError(
            f"Resonator length {length} is too short for {meanders} meanders with current bend spec {bend}. "
            "Increase length, reduce meanders, or change the bend spec."
        )

    straight_comp = straight(length=straight_length, cross_section=cross_section)
    actual_length = bend_length + straight_count * straight_comp.info["length"]
    return straight_count, straight_comp, actual_length


@gf.cell(tags=["resonators"], schematic_function=resonator_schematic)
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

    Raises:
        ValueError: If length is too short for the requested meanders.
    """
    c = Component()
    cross_section = gf.get_cross_section(cross_section)
    bend = gf.get_component(
        bend_spec, cross_section=cross_section, angle=180, angular_step=4
    )

    _, straight_comp, actual_length = _resonator_straight(
        length,
        meanders,
        bend,
        cross_section,
        start_with_bend=start_with_bend,
        end_with_bend=end_with_bend,
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
                if previous_port is None:
                    raise ValueError("previous_port is required but not initialized.")
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
            # Keep open-end ports optical so recursive circuit netlists retain them.
            c.add_port(output_port, port=open_etch.ports[output_port])

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
    c.info["resonator_type"] = (
        "half_wave" if open_start and open_end else "quarter_wave"
    )
    c.info["cross_section_name"] = cross_section.name
    # c.info["frequency_estimate"] = (
    #     3e8 / (4 * length * 1e-6) / 1e9
    # )  # GHz, rough estimate

    return c


@gf.cell(tags=["resonators"], schematic_function=resonator_quarter_wave_schematic)
def resonator_quarter_wave(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    open_start: bool = False,
    open_end: bool = True,
) -> Component:
    """Return a quarter-wave resonator with its schematic and model boundary.

    Args:
        length: Total resonator length in µm.
        meanders: Number of meander bends.
        bend_spec: Component used for each bend.
        cross_section: Resonator cross section.
        start_with_bend: Start the path with a bend.
        end_with_bend: End the path with a bend.
        open_start: Leave the start open instead of shorting it.
        open_end: Leave the end open instead of shorting it.
    """
    return resonator(
        length=length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        start_with_bend=start_with_bend,
        end_with_bend=end_with_bend,
        open_start=open_start,
        open_end=open_end,
    )


@gf.cell(tags=["resonators"], schematic_function=resonator_half_wave_schematic)
def resonator_half_wave(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    open_start: bool = True,
    open_end: bool = True,
) -> Component:
    """Return a half-wave resonator with its schematic and model boundary.

    Args:
        length: Total resonator length in µm.
        meanders: Number of meander bends.
        bend_spec: Component used for each bend.
        cross_section: Resonator cross section.
        start_with_bend: Start the path with a bend.
        end_with_bend: End the path with a bend.
        open_start: Leave the start open instead of shorting it.
        open_end: Leave the end open instead of shorting it.
    """
    return resonator(
        length=length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        start_with_bend=start_with_bend,
        end_with_bend=end_with_bend,
        open_start=open_start,
        open_end=open_end,
    )


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


@gf.cell(
    tags=["resonators", "couplers"],
    schematic_function=resonator_coupled_schematic,
)
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

    .. svgbob::

        coupling_o1 ─────────────── coupling_o2
                     coupling_gap
        resonator_o1 ───────┐
                            │
        ┌───────────────────┘
        │
        └───────────────────┐
                            │
        ┌───────────────────┘
        │
        └──── resonator_o2

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
        c.add_port(f"resonator_{port.name}", port=port)

    for port in coupling_ref.ports:
        c.add_port(f"coupling_{port.name}", port=port)

    c.info += resonator_ref.cell.info
    c.info["coupling_length"] = coupling_straight_length
    c.info["coupling_gap"] = coupling_gap

    return c


@gf.cell(
    tags=["resonators", "couplers"],
    schematic_function=quarter_wave_resonator_coupled_schematic,
)
def quarter_wave_resonator_coupled(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    start_with_bend: bool = False,
    end_with_bend: bool = False,
    cross_section_non_resonator: CrossSectionSpec = "cpw",
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 20.0,
) -> Component:
    """Creates a quarter-wave resonator with a coupling waveguide.

    Composes a straight coupler, the meander, and explicit open and short
    terminations. The meander starts at the coupler output so that the trace
    leaves the coupling gap at ``coupling_straight_length``. Layouts with no
    usable first straight section retain the unsplit coupled-resonator cell.

    The terminations are fixed: the coupled end is open and the far end is
    shorted. Use :func:`~qpdk.cells.resonator.resonator_coupled` if other
    terminations are needed.

    .. svgbob::

        coupling_o1 ─────────────── coupling_o2
                     coupling_gap
        resonator_o1 ───────┐
                            │
        ┌───────────────────┘
        │
        └───────────────────┐
                            │
        ┌───────────────────┘
        │
        └──── (shorted, no port)

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        start_with_bend: If True, starts the resonator with a bend.
        end_with_bend: If True, ends the resonator with a bend.
        cross_section_non_resonator: Cross-section specification for the coupling waveguide.
        coupling_straight_length: Length of the coupling waveguide section in μm.
        coupling_gap: Gap between the resonator and coupling waveguide in μm.

    Returns:
        The coupled quarter-wave resonator component.

    Raises:
        ValueError: If the requested length is too short for its meanders.
    """
    c = Component()
    cross_section_obj = gf.get_cross_section(cross_section)
    bend = gf.get_component(
        bend_spec,
        cross_section=cross_section_obj,
        angle=180,
        angular_step=4,
    )
    first_straight_length = 0.0
    actual_length = length
    if not start_with_bend and meanders > 0:
        _, first_straight, actual_length = _resonator_straight(
            length,
            meanders,
            bend,
            cross_section_obj,
            start_with_bend=False,
            end_with_bend=end_with_bend,
        )
        if first_straight is None:
            raise ValueError("A coupling section requires a first straight section.")
        first_straight_length = gf.snap.snap_to_grid(first_straight.info["length"])

    if (
        start_with_bend
        or meanders == 0
        or coupling_straight_length >= first_straight_length
    ):
        res_ref = c << resonator_coupled(
            length=length,
            meanders=meanders,
            bend_spec=bend_spec,
            cross_section=cross_section,
            start_with_bend=start_with_bend,
            end_with_bend=end_with_bend,
            open_start=True,
            open_end=False,
            cross_section_non_resonator=cross_section_non_resonator,
            coupling_straight_length=coupling_straight_length,
            coupling_gap=coupling_gap,
        )
        res_ref.move(tuple(-np.array(res_ref.ports["coupling_o1"].center)))
        short_ref = c << short()
        short_ref.connect("o1", res_ref.ports["resonator_o2"])
        for port in res_ref.ports:
            if port.name != "resonator_o2":
                c.add_port(port=port)
        c.info += res_ref.cell.info
        return c

    coupling_ref = c << coupler_straight(
        length=coupling_straight_length,
        gap=coupling_gap,
        cross_section=cross_section,
        cross_section_non_resonator=cross_section_non_resonator,
        resonator_offset=get_etch_section(cross_section_obj).width,
    )
    coupling_ref.move(tuple(-np.array(coupling_ref.ports["o2"].center)))
    open_ref = c << open_2_port(cross_section=cross_section)
    open_ref.connect("o1", coupling_ref.ports["o1"])
    # A straight lead-out here would run parallel to the probeline and make
    # the coupled length longer than ``coupling_straight_length``.
    meander_ref = c << resonator_meander(
        length=actual_length - coupling_straight_length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        end_with_bend=end_with_bend,
    )
    meander_ref.connect("o1", coupling_ref.ports["o4"])

    short_ref = c << short()
    short_ref.connect("o1", meander_ref.ports["o2"])
    c.add_port("coupling_o1", port=coupling_ref.ports["o2"])
    c.add_port("coupling_o2", port=coupling_ref.ports["o3"])
    c.add_port("resonator_o1", port=open_ref.ports["o2"])
    c.info["length"] = coupling_straight_length + meander_ref.cell.info["length"]
    c.info["resonator_type"] = "quarter_wave"
    c.info["cross_section_name"] = gf.get_cross_section(cross_section).name
    c.info["coupling_length"] = coupling_straight_length
    c.info["coupling_gap"] = coupling_gap

    return c
