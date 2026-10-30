"""Su-Schrieffer-Heeger (SSH) coupled-resonator chain."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.waveguides import nxn, rectangle, straight, taper_cross_section
from qpdk.tech import get_etch_section

__all__ = ["ssh_resonator_chain"]

_SIDES = {0: "east", 90: "north", 180: "west", 270: "south"}


@gf.cell(tags=["resonators", "metamaterials"])
def ssh_resonator_chain(
    n_cells: int = 4,
    site_spacing: float = 300.0,
    resonator: ComponentSpec = "lumped_element_resonator",
    resonator_cross_section: CrossSectionSpec = "meander_inductor_cross_section",
    intra_cell_fingers: int = 8,
    inter_cell_fingers: int = 4,
    port_fingers: int = 4,
    tap_fingers: int = 2,
    finger_length: float = 20.0,
    tap_sites: tuple[int, ...] = (),
    tap_length: float = 100.0,
    port_length: float = 50.0,
    resonator_taper_length: float = 10.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""SSH chain of capacitively coupled lumped-element LC resonators.

    :math:`N` unit cells of two sites (A, B) sit on a straight line. Each
    site is a junction on the ``cross_section`` line with a
    ``resonator`` hanging below it; the resonator's far port ends at the
    etch edge and so touches the ground plane, making each site an LC
    resonator to ground. Interdigital capacitors between neighbouring sites
    alternate between ``intra_cell_fingers`` (A→B, coupling :math:`C_v`) and
    ``inter_cell_fingers`` (B→A, coupling :math:`C_w`). The end sites couple to
    ports ``o1`` and ``o2`` through ``port_fingers`` capacitors.

    .. svgbob::

                    tap         tap
                     |           |
        o1 ─||─┬─|||─┬─||─┬─|||─┬─||─ o2
               │     │    │     │
              LC    LC   LC    LC
              1,A   1,B  2,A   2,B

    More intra-cell than inter-cell fingers gives :math:`C_v > C_w`, the
    trivial phase (:math:`\delta > 0`). Swapping them gives the topological
    phase, with edge modes on the end sites
    :cite:`kimQuantumElectrodynamicsTopological2021`.

    Sites listed in ``tap_sites`` get an extra branch above the line, ending
    in a ``tap_fingers`` capacitor and a port (``o3``, ``o4``, … in the order
    given), e.g. for coupling a qubit to that site.

    The circuit model is :func:`qpdk.models.ssh.ssh_resonator_chain`. It
    takes lumped values, not this geometry.

    Args:
        n_cells: Number of unit cells :math:`N`.
        site_spacing: Centre-to-centre distance between neighbouring sites in
            µm, i.e. half the lattice constant.
        resonator: Two-port resonator placed at each site. Its ``o1`` faces the
            line and its ``o2`` is grounded.
        resonator_cross_section: Cross-section of the resonator's ``o1`` port,
            the far end of the taper from the line.
        intra_cell_fingers: Fingers of the intra-cell (A→B) capacitors.
        inter_cell_fingers: Fingers of the inter-cell (B→A) capacitors.
        port_fingers: Fingers of the capacitors to ports ``o1`` and ``o2``.
        tap_fingers: Fingers of the tap capacitors.
        finger_length: Finger length of all coupling capacitors in µm.
        tap_sites: Indices of the sites with a tap, from 0 to :math:`2N-1`
            ordered :math:`(1,\text{A}), (1,\text{B}), \ldots`.
        tap_length: Length of the line from a site to its tap capacitor in µm.
        port_length: Length of the line from each port capacitor to the
            ports in µm.
        resonator_taper_length: Length of the taper from the line to the
            resonator cross-section in µm.
        cross_section: Cross-section of the line and the capacitor ports.

    Returns:
        Component: The chain with ports ``o1``, ``o2`` and one per tap.

    Raises:
        ValueError: If ``n_cells`` < 1, ``tap_sites`` holds duplicate or
            out-of-range indices, or ``site_spacing`` is too short for the
            junctions and capacitors.
    """
    n_sites = 2 * n_cells
    if n_cells < 1:
        msg = f"n_cells must be at least 1, got {n_cells}."
        raise ValueError(msg)
    if len(set(tap_sites)) != len(tap_sites) or any(
        not 0 <= site < n_sites for site in tap_sites
    ):
        msg = (
            f"tap_sites must be unique site indices in [0, {n_sites}), got {tap_sites}."
        )
        raise ValueError(msg)

    xs = gf.get_cross_section(cross_section)
    etch_width = get_etch_section(xs).width
    lead = straight(length=etch_width, cross_section=xs)
    closure = rectangle(
        size=(xs.width, etch_width),
        layer=get_etch_section(xs).layer,
        centered=True,
        port_type=None,
        port_orientations=None,
    )

    def capacitor(fingers: int) -> Component:
        return interdigital_capacitor(
            fingers=fingers, finger_length=finger_length, cross_section=xs
        )

    couplers = [
        capacitor(intra_cell_fingers if site % 2 == 0 else inter_cell_fingers)
        for site in range(n_sites - 1)
    ]
    junction_length = xs.width + 2 * etch_width
    shortest_gap = (
        site_spacing
        - junction_length
        - max(
            (abs(cap.ports["o2"].x - cap.ports["o1"].x) for cap in couplers), default=0
        )
    )
    if shortest_gap < 0:
        msg = (
            f"site_spacing={site_spacing} µm is too short for the junctions and "
            f"coupling capacitors (needs {site_spacing - shortest_gap} µm)."
        )
        raise ValueError(msg)

    c = Component()
    sites = []
    for site in range(n_sites):
        tapped = site in tap_sites
        node = c << nxn(
            xsize=xs.width,
            ysize=xs.width,
            wg_width=xs.width,
            layer=xs.layer,
            north=int(tapped),
            cross_section=xs,
        )
        node.center = (site * site_spacing, 0)
        sides = {}
        for port in node.ports:
            ref = c << lead
            ref.connect("o1", port)
            sides[_SIDES[round(port.orientation) % 360]] = ref.ports["o2"]
        if not tapped:
            # Close the etch gap on the north side of the junction.
            stub = c << closure
            stub.center = (node.x, node.ymax + etch_width / 2)
        sites.append(sides)

        taper = c << taper_cross_section(
            cross_section1=xs,
            cross_section2=resonator_cross_section,
            length=resonator_taper_length,
        )
        taper.connect("o1", sides["south"])
        site_resonator = c << gf.get_component(resonator)
        site_resonator.connect("o1", taper.ports["o2"])

    # Coupling capacitors between neighbouring sites, centred between them.
    for site, coupler in enumerate(couplers):
        cap = c << coupler
        cap.center = ((site + 0.5) * site_spacing, 0)
        for start, end in (
            (sites[site]["east"], cap.ports["o1"]),
            (cap.ports["o2"], sites[site + 1]["west"]),
        ):
            if (length := round(end.x - start.x, 3)) > 0:
                wire = c << straight(length=length, cross_section=xs)
                wire.connect("o1", start)

    def add_port_branch(name: str, start: gf.Port, fingers: int, length: float) -> None:
        wire = c << straight(length=length, cross_section=xs)
        wire.connect("o1", start)
        cap = c << capacitor(fingers)
        cap.connect("o1", wire.ports["o2"])
        c.add_port(name, port=cap.ports["o2"])

    add_port_branch("o1", sites[0]["west"], port_fingers, port_length)
    add_port_branch("o2", sites[-1]["east"], port_fingers, port_length)
    for index, site in enumerate(tap_sites):
        add_port_branch(f"o{3 + index}", sites[site]["north"], tap_fingers, tap_length)

    c.info["n_sites"] = n_sites
    c.info["lattice_constant"] = 2 * site_spacing
    return c
