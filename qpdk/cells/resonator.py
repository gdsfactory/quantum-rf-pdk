"""Resonator components."""

from __future__ import annotations

from functools import partial
from typing import TypedDict

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.waveguides import bend_circular, straight


class ResonatorParams(TypedDict):
    """Parameters for the resonator component."""

    length: float
    meanders: int
    bend_spec: ComponentSpec
    cross_section: CrossSectionSpec
    open_start: bool
    open_end: bool


@gf.cell_with_module_name
def resonator(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    open_start: bool = False,
    open_end: bool = False,
) -> Component:
    """Creates a meandering coplanar waveguide resonator.

    Changing `open_start` and `open_end` appropriately allows creating
    a shorted quarter-wave resonator or an open half-wave resonator.

    See :cite:`m.pozarMicrowaveEngineering2012` for details

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        open_start: If True, adds an etch section at the start of the resonator.
        open_end: If True, adds an etch section at the end of the resonator.

    Returns:
        Component: A gdsfactory component with meandering resonator geometry.
    """
    c = Component()
    cross_section = gf.get_cross_section(cross_section)
    bend = gf.get_component(bend_spec, cross_section=cross_section, angle=180)
    length_per_one_straight = (length - meanders * bend.info["length"]) / (meanders + 1)

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
    for i in range(meanders):
        straight_ref = c.add_ref(straight_comp)
        bend_ref = c.add_ref(bend)

        if i == 0:
            first_straight_ref = straight_ref
        else:  # i > 0
            straight_ref.connect("o1", previous_port)

        if i % 2 == 0:
            bend_ref.mirror()
            bend_ref.rotate(90)

        bend_ref.connect("o1", straight_ref.ports["o2"])
        previous_port = bend_ref.ports["o2"]

    # Final straight section
    final_straight = straight(
        length=length_per_one_straight,
        cross_section=cross_section,
    )
    final_straight_ref = c.add_ref(final_straight)
    final_straight_ref.connect("o1", previous_port)

    # Etch at the open end
    if open_end or open_start:
        cross_section_etch_section = next(
            s
            for s in gf.get_cross_section(cross_section).sections
            if "etch_offset" in s.name
        )

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
            c.add_port(output_port, port=open_etch.ports[output_port])

        if open_end:
            _add_etch_at_port("o1", final_straight_ref.ports["o2"], "o2")
        if open_start:
            _add_etch_at_port("o2", first_straight_ref.ports["o1"], "o1")

    if not open_end:
        c.add_port("o2", port=final_straight_ref.ports["o2"])

    if not open_start:
        c.add_port("o1", port=first_straight_ref.ports["o1"])

    # Add metadata
    c.info["length"] = length
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


# Reuse the existing ResonatorParams TypedDict by inheriting from it.
# This keeps common resonator fields defined in ResonatorParams and adds coupling-specific fields.
class ResonatorCoupledParams(ResonatorParams):
    """Parameters for the coupled resonator component.

    Inherits all fields from :class:`~ResonatorParams` and adds:
      - coupling_straight_length: float
      - coupling_gap: float
    """

    coupling_straight_length: float
    coupling_gap: float


@gf.cell_with_module_name
def resonator_coupled(
    resonator_params: ResonatorParams | None = None,  # pyright: ignore[reportRedeclaration]
    cross_section_non_resonator: CrossSectionSpec = "cpw",
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 12.0,
) -> Component:
    """Creates a meandering coplanar waveguide resonator with a coupling waveguide.

    This component combines a resonator with a parallel coupling waveguide placed
    at a specified gap for proximity coupling. Similar to the design described in
    :cite:`besedinQualityFactorTransmission2018a`.

    Args:
        resonator_params: Parameters for the resonator component. If None, defaults will be used.
        cross_section_non_resonator: Cross-section specification for the coupling waveguide.
        coupling_straight_length: Length of the coupling waveguide section in μm.
        coupling_gap: Gap between the resonator and coupling waveguide in μm.
            Measured from edges of the center conductors.

    Returns:
        Component: A gdsfactory component with meandering resonator and coupling waveguide.
    """
    c = Component()
    resonator_params: ResonatorParams = resonator_params or {}

    resonator_ref = c.add_ref(resonator(**resonator_params))

    cross_section_obj = gf.get_cross_section(cross_section_non_resonator)

    coupling_wg = straight(
        length=coupling_straight_length,
        cross_section=cross_section_obj,
    )
    coupling_ref = c.add_ref(coupling_wg)

    # Position coupling waveguide parallel to resonator with specified gap
    coupling_ref.movey(coupling_gap + cross_section_obj.width)

    coupling_ref.xmin = resonator_ref["o1"].x  # Align left edges

    for comp, prefix in ((resonator_ref, "resonator"), (coupling_ref, "coupling")):
        for port in comp.ports:
            c.add_port(f"{prefix}_{port.name}", port=port)

    c.info += resonator_ref.cell.info
    c.info["coupling_length"] = coupling_straight_length
    c.info["coupling_gap"] = coupling_gap

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = gf.Component()
    for i, component in enumerate(
        (
            resonator(),
            resonator_quarter_wave(),
            resonator_half_wave(),
            resonator_coupled(),
            resonator_coupled(
                ResonatorParams(
                    length=2000, meanders=4, open_start=False, open_end=True
                )
            ),
        ),
    ):
        (c << component).move((i * 700, 0))
    c.show()
