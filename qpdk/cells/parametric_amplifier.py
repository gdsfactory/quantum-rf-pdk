r"""Flux-driven Josephson parametric amplifier (JPA) components.

A flux-driven JPA is a :math:`\lambda/4` CPW resonator whose open end is
capacitively coupled to a single feed line, used in reflection, and whose
far end is shorted to ground through a DC SQUID. The SQUID inductance
:math:`L_\text{s}(\Phi)` tunes the resonance. A pump at twice the signal
frequency, applied through a separate CPW line that is shorted to ground next
to the SQUID loop, modulates :math:`\Phi` and gives three-wave mixing gain.

References:
    - :cite:`yamamotoFluxdrivenJosephsonParametric2008`
    - :cite:`linJosephsonParametricPhaselocked2014`
    - :cite:`zhongSqueezingFluxdrivenJosephson2013`
    - :cite:`pogorzalekHystereticFluxResponse2017`
"""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from kfactory import kdb
from kfactory.conf import CheckInstances

from qpdk.cells._schematic import attach_schematic, squid_termination_schematic
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.junction import josephson_junction, squid_junction
from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, straight, straight_shorted
from qpdk.tech import LAYER, get_etch_section

__all__ = ["cpw_jpa", "squid_termination"]


@gf.cell(
    check_instances=CheckInstances.IGNORE,
    tags=["junctions", "terminations"],
    schematic_function=squid_termination_schematic,
)
def squid_termination(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = partial(
        squid_junction,
        junction_spec=partial(
            josephson_junction,
            junction_overlap_displacement=1.8,
            wide_straight_length=4.0,
            narrow_straight_length=0.5,
            taper_length=4,
        ),
    ),
) -> Component:
    r"""CPW lead shorted to the ground plane through a DC SQUID.

    The lead ends in an etched pocket that separates the centre conductor from
    the ground plane. The SQUID bridges the pocket, so the centre conductor
    reaches ground only through the SQUID, as in the flux-driven JPA of
    :cite:`yamamotoFluxdrivenJosephsonParametric2008`. The pocket length
    follows from the SQUID: its edges run through the centres of the
    ``JJ_PATCH`` contact patches, so each patch lands half on the lead and
    half on the ground plane.

    .. svgbob::

         ──────────┬──────┐
                   │      │
        o1 ────────│  X   │  ground
                   │ SQUID│
         ──────────┴──────┘
          CPW lead   pocket

    Args:
        length: Length of the CPW lead in µm.
        cross_section: Cross-section of the CPW lead. The pocket spans the
            full CPW width, centre conductor plus both gaps.
        junction_spec: Component specification for the SQUID.

    Returns:
        Component: The SQUID termination with CPW port ``o1`` and a
        ``squid`` placement port at the SQUID centre.

    Raises:
        ValueError: If the SQUID has no ``JJ_PATCH`` contacts on one side, or
            its lead-side contacts overhang the centre conductor.
    """
    c = Component()
    xs = gf.get_cross_section(cross_section)
    etch_section = get_etch_section(xs)
    pocket_width = xs.width + 2 * etch_section.width

    lead_ref = c << straight(length=length, cross_section=cross_section)

    junction = gf.get_component(junction_spec)
    squid_ref = c << junction
    squid_ref.dcplx_trans *= kdb.DCplxTrans(1, 45, False, 0, 0)
    squid_ref.dcenter = (0.0, 0.0)

    # The lead has no JJ_PATCH shapes, so these are the SQUID contacts.
    patches = kdb.Region(c.begin_shapes_rec(gf.get_layer(LAYER.JJ_PATCH)))
    boxes = [patch.bbox().to_dtype(c.kcl.dbu) for patch in patches.merged()]
    lead_side = [box.center().x for box in boxes if box.center().x < 0]
    ground_side = [box.center().x for box in boxes if box.center().x > 0]
    if not lead_side or not ground_side:
        raise ValueError(
            "junction_spec must have JJ_PATCH contacts on both sides of its "
            "centre after the 45° rotation."
        )
    if any(
        box.center().x < 0 and max(abs(box.top), abs(box.bottom)) > xs.width / 2
        for box in boxes
    ):
        raise ValueError(
            "The lead-side JJ_PATCH contacts of junction_spec extend beyond the "
            f"centre conductor width ({xs.width} µm) and would not all reach it."
        )
    lead_contact_x = sum(lead_side) / len(lead_side)
    junction_gap = gf.snap.snap_to_grid(
        sum(ground_side) / len(ground_side) - lead_contact_x
    )
    squid_ref.dmove((length - lead_contact_x, 0.0))

    pocket = gf.c.rectangle(size=(junction_gap, pocket_width), layer=etch_section.layer)
    pocket_ref = c << pocket
    pocket_ref.dmove((length, -pocket_width / 2))

    # Port widths must be an even number of database units.
    port_width = 2 * round(squid_ref.size_info.height / (2 * c.kcl.dbu)) * c.kcl.dbu
    c.add_port("o1", port=lead_ref.ports["o1"])
    c.add_port(
        name="squid",
        center=squid_ref.dcenter,
        width=port_width,
        orientation=0,
        layer=LAYER.JJ_AREA,
        port_type="placement",
    )
    c.info["length"] = length
    c.info["pocket_width"] = pocket_width
    c.info["junction_gap"] = junction_gap
    c.info["loop_area"] = junction.settings.get("loop_area")
    return c


attach_schematic(squid_termination, squid_termination_schematic)


@gf.cell(check_instances=CheckInstances.IGNORE, tags=["resonators", "junctions"])
def cpw_jpa(
    length: float = 4680.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    coupling_capacitor_spec: ComponentSpec = partial(
        interdigital_capacitor, fingers=8, finger_length=50.0
    ),
    termination_spec: ComponentSpec = squid_termination,
    pump_length: float = 200.0,
    pump_gap: float = 5.0,
    pump_offset: float = 5.0,
) -> Component:
    r"""Flux-driven :math:`\lambda/4` CPW Josephson parametric amplifier.

    The signal port ``o1`` couples through ``coupling_capacitor_spec`` into a
    meandered :math:`\lambda/4` resonator. The resonator ends in
    ``termination_spec``, which shorts it to ground through a DC SQUID. The
    pump line ``o2`` is a shorted CPW whose short sits ``pump_gap`` from the
    SQUID pocket on the side away from the meander. The pump current returns
    through the ground plane past the SQUID loop and couples to it through a
    mutual inductance :math:`M`
    :cite:`yamamotoFluxdrivenJosephsonParametric2008,linJosephsonParametricPhaselocked2014`.
    The pump-to-loop distance is not reported in these sources, so ``pump_gap``
    and ``pump_offset`` are design variables, and :math:`M` has to be extracted
    by EM simulation. The SAX models do not include :math:`M`. The static flux
    bias comes from an off-chip coil.

    With the SQUID inductance :math:`L_\text{s}` the resonance is
    :cite:`yamamotoFluxdrivenJosephsonParametric2008`:

    .. math::

        \omega_\text{r}(\Phi) \approx \frac{\omega_{\lambda/4}}{1 + L_\text{s}(\Phi) / L_\text{r}},
        \qquad
        L_\text{s}(\Phi) = \frac{\Phi_0}{2 \pi I_\text{c,tot} |\cos(\pi \Phi / \Phi_0)|},

    where :math:`L_\text{r}` is the total inductance of the :math:`\lambda/4`
    line. The default coupling capacitor is about :math:`28\,\text{fF}`,
    close to the :math:`30\,\text{fF}` of the WMI devices with
    :math:`Q_\text{ext} \approx 300` :cite:`pogorzalekHystereticFluxResponse2017`.
    With the default :math:`I_\text{c,tot} = 4.8\,\text{µA}` of
    :func:`~qpdk.models.junction.squid_termination`, the SAX model of the
    default layout resonates at :math:`6.07\,\text{GHz}` at zero flux and
    :math:`5.93\,\text{GHz}` at :math:`0.3\,\Phi_0`, with
    :math:`Q_\text{ext} \approx 290`.

    .. svgbob::

                     ┌──────────────┐
                     │  meander     │
        o1 ──║║──────┘  λ/4 line    │
            Cc                      │
                     ┌──────────────┘
                     └──────── X  ground
                               ╷
                               ╵ pump short
                               │
                               o2

    Args:
        length: Realized :math:`\lambda/4` line length in µm, from the
            coupling capacitor to the SQUID pocket. Includes the
            termination's CPW lead.
        meanders: Number of meander sections.
        bend_spec: Specification for the meander bends.
        cross_section: Cross-section of the resonator and the pump line.
        coupling_capacitor_spec: Two-port coupling capacitor between ``o1``
            and the open end of the resonator.
        termination_spec: One-port SQUID termination, see
            :func:`squid_termination`. It must provide a ``squid`` port and
            the ``length``, ``pocket_width`` and ``loop_area`` info entries.
        pump_length: Length of the pump CPW line in µm.
        pump_gap: Ground-plane distance in µm between the pump short and the
            edge of the SQUID pocket.
        pump_offset: Lateral offset in µm of the pump line axis from the SQUID
            centre, along the resonator line.

    Returns:
        Component: The JPA with signal port ``o1`` and pump port ``o2``.

    Raises:
        ValueError: If ``length`` does not exceed the termination lead length.
    """
    c = Component()
    xs = gf.get_cross_section(cross_section)

    termination = gf.get_component(termination_spec)
    resonator_length = length - termination.info["length"]
    if resonator_length <= 0:
        raise ValueError(
            f"length ({length}) must exceed the termination lead length "
            f"({termination.info['length']})."
        )

    capacitor_ref = c << gf.get_component(coupling_capacitor_spec)
    resonator_ref = c << resonator(
        length=resonator_length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        open_start=False,
        open_end=False,
    )
    termination_ref = c << termination

    resonator_ref.connect("o1", capacitor_ref.ports["o2"])
    termination_ref.connect("o1", resonator_ref.ports["o2"])

    # The pump approaches the pocket from the side away from the meander,
    # which lies above the termination.
    squid_port = termination_ref.ports["squid"]
    pump_ref = c << straight_shorted(length=pump_length, cross_section=cross_section)
    pump_ref.dcplx_trans = kdb.DCplxTrans(
        1,
        90,
        False,
        squid_port.dx + pump_offset,
        squid_port.dy - termination.info["pocket_width"] / 2 - pump_gap,
    ) * kdb.DCplxTrans(-pump_length, 0)

    c.add_port("o1", port=capacitor_ref.ports["o1"])
    c.add_port("o2", port=pump_ref.ports["o1"])

    capacitor_settings = capacitor_ref.cell.settings
    c.info["length"] = length
    c.info["cross_section"] = xs.name
    c.info["resonator_type"] = "quarter_wave"
    c.info["coupling_fingers"] = capacitor_settings.get("fingers")
    c.info["coupling_finger_length"] = capacitor_settings.get("finger_length")
    c.info["coupling_finger_gap"] = capacitor_settings.get("finger_gap")
    c.info["coupling_thickness"] = capacitor_settings.get("thickness")
    c.info["squid_loop_area"] = termination.info["loop_area"]
    c.info["pump_gap"] = pump_gap
    c.info["pump_offset"] = pump_offset
    return c
