r"""Microwave-to-optical quantum transducer layout components.

This module provides layout components for the microwave side of
superconducting-to-optical quantum transducers.  Two dominant physical
mechanisms for microwave-to-optical transduction exist in the literature:

1. **Cavity electro-optic (CEO) transduction** – a microwave LC resonator
   is capacitively coupled to optical modes in a material with a
   :math:`\chi^{(2)}` nonlinearity (e.g. thin-film lithium niobate).  See
   :cite:`warnerCoherentControlSuperconducting2023,lauksTransducerCoupling2020`.

2. **Piezo-optomechanical transduction** – a superconducting qubit or
   microwave resonator is piezoelectrically coupled to a mechanical mode,
   which is in turn optomechanically coupled to an optical cavity.  See
   :cite:`mirhosseiniQuantumTransductionOptical2020,delaneyNondestructiveOpticalReadout2022`.

The layout components in this module represent the **microwave side** of
these hybrid transducer systems – i.e. the structures that would be
fabricated on a superconducting chip and subsequently integrated with an
optical or piezoelectric chip via flip-chip bonding, wire bonding, or
coaxial cable.

References:
    - :cite:`mirhosseiniQuantumTransductionOptical2020`
    - :cite:`warnerCoherentControlSuperconducting2023`
    - :cite:`delaneyNondestructiveOpticalReadout2022`
    - :cite:`lauksTransducerCoupling2020`
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.capacitor import plate_capacitor
from qpdk.cells.inductor import meander_inductor
from qpdk.cells.waveguides import straight
from qpdk.helper import show_components
from qpdk.tech import LAYER


@gf.cell
def electro_optic_transducer(
    inductor_n_turns: int = 8,
    inductor_turn_length: float = 200.0,
    capacitor_length: float = 100.0,
    capacitor_width: float = 100.0,
    capacitor_gap: float = 5.0,
    coupling_pad_size: tuple[float, float] = (200.0, 100.0),
    coupling_pad_gap: float = 20.0,
    feedline_length: float = 100.0,
    cross_section: CrossSectionSpec = "cpw",
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_eo_coupler: LayerSpec = LAYER.IND,
) -> Component:
    r"""Microwave LC resonator for cavity electro-optic transduction.

    Creates the microwave side of a cavity electro-optic
    microwave-to-optical quantum transducer (CEO-MOQT).  The layout
    consists of:

    * A lumped-element LC resonator (meander inductor in parallel with a
      plate capacitor) whose resonance frequency matches the target
      microwave transition, typically :math:`\omega_m / 2\pi \sim 5\text{--}8\;\text{GHz}`.
    * A coupling capacitor pad pair placed adjacent to the LC resonator.
      This pad is intended for capacitive coupling to a photonic chip
      carrying optical ring resonators in thin-film lithium niobate (TFLN).
    * CPW feedline connections for external microwave drive and readout.

    The transduction process relies on the :math:`\chi^{(2)}`
    nonlinearity of TFLN to mediate energy exchange between the
    microwave LC mode and two hybridised optical modes separated by the
    microwave frequency :math:`\omega_m`.

    .. svgbob::

            CPW feedline
          o1 ──────────┐
                       │  "coupling cap pads"
                   ┌───┤───┐
                   │   │   │ ← to optical chip (TFLN)
                   └───┤───┘
                       │
                 ┌─────┤─────┐
                 │  meander   │
                 │  inductor  │
                 │     ║      │
                 │  plate cap │
                 └─────┤─────┘
                       │
          o2 ──────────┘
            CPW feedline

    See :cite:`warnerCoherentControlSuperconducting2023` for experimental
    details and :cite:`lauksTransducerCoupling2020` for theory.

    Args:
        inductor_n_turns: Number of meander turns for the inductor.
        inductor_turn_length: Length of each meander turn in µm.
        capacitor_length: Length of the plate capacitor pads in µm.
        capacitor_width: Width of the plate capacitor pads in µm.
        capacitor_gap: Gap between the plate capacitor pads in µm.
        coupling_pad_size: (width, height) of each EO coupling pad in µm.
        coupling_pad_gap: Gap between the two EO coupling pads in µm.
        feedline_length: Length of CPW feedline sections in µm.
        cross_section: Cross-section specification for CPW feedlines.
        layer_metal: Metal layer for all superconducting structures.
        layer_eo_coupler: Layer for the electro-optic coupling pads
            (typically the indium bump layer for flip-chip integration).

    Returns:
        Component with ports ``o1`` and ``o2`` (CPW feedline) and
        ``eo_coupler`` (placement port at the EO coupling pad centre).
    """
    c = Component()

    # --- Plate capacitor forms the main resonator capacitance ---
    cap = c.add_ref(
        plate_capacitor(
            length=capacitor_length, width=capacitor_width, gap=capacitor_gap
        )
    )

    # --- Meander inductor in parallel with capacitor ---
    ind = c.add_ref(
        meander_inductor(n_turns=inductor_n_turns, turn_length=inductor_turn_length)
    )
    # Position inductor above the capacitor
    ind.dmove((
        cap.dcenter[0] - ind.dcenter[0],
        cap.dbbox().top + 20 - ind.dcenter[1],
    ))

    # --- EO coupling pads for flip-chip integration ---
    pad_w, pad_h = coupling_pad_size
    # Left pad
    left_pad = c.add_ref(
        gf.components.rectangle(size=(pad_w, pad_h), layer=layer_eo_coupler)
    )
    left_pad.dmove((
        -pad_w - coupling_pad_gap / 2,
        ind.dbbox().top + 20,
    ))
    # Right pad
    right_pad = c.add_ref(
        gf.components.rectangle(size=(pad_w, pad_h), layer=layer_eo_coupler)
    )
    right_pad.dmove((
        coupling_pad_gap / 2,
        ind.dbbox().top + 20,
    ))

    # --- CPW feedlines connected to capacitor ports ---
    feed_left = c.add_ref(straight(length=feedline_length, cross_section=cross_section))
    feed_left.connect("o2", cap.ports["o1"])

    feed_right = c.add_ref(
        straight(length=feedline_length, cross_section=cross_section)
    )
    feed_right.connect("o2", cap.ports["o2"])

    # --- Ports ---
    c.add_port(name="o1", port=feed_left.ports["o1"])
    c.add_port(name="o2", port=feed_right.ports["o1"])

    # Placement port at EO coupler centre
    eo_center_x = (left_pad.dcenter[0] + right_pad.dcenter[0]) / 2
    eo_center_y = (left_pad.dcenter[1] + right_pad.dcenter[1]) / 2
    c.add_port(
        name="eo_coupler",
        center=(eo_center_x, eo_center_y),
        width=pad_w * 2 + coupling_pad_gap,
        orientation=90,
        layer=layer_eo_coupler,
        port_type="placement",
    )

    # --- Metadata ---
    c.info["inductor_n_turns"] = inductor_n_turns
    c.info["inductor_turn_length"] = inductor_turn_length
    c.info["capacitor_length"] = capacitor_length
    c.info["capacitor_width"] = capacitor_width
    c.info["capacitor_gap"] = capacitor_gap
    c.info["transducer_type"] = "electro_optic"

    return c


@gf.cell
def piezo_transducer_coupler(
    pad_size: tuple[float, float] = (300.0, 200.0),
    pad_gap: float = 30.0,
    feedline_length: float = 200.0,
    cross_section: CrossSectionSpec = "cpw",
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_piezo: LayerSpec = LAYER.IND,
) -> Component:
    r"""Piezoelectric coupling pad for piezo-optomechanical transduction.

    Creates the microwave side of a piezo-optomechanical transducer.
    The layout consists of two large interdigitated metal pads that
    generate a strong electric field across a piezoelectric material
    (e.g. AlN or LiNbO₃) placed on top or bonded via flip-chip.  The
    electric field drives a mechanical resonance that is simultaneously
    coupled to an optical cavity via radiation pressure.

    .. svgbob::

         CPW feedline
        o1 ───────────┐
                      │
                ┌─────┴─────┐
                │           │
                │   pad 1   │  ← metal on SC chip
                │           │
                ├───gap─────┤  ← piezoelectric
                │           │
                │   pad 2   │  ← metal on SC chip
                │           │
                └─────┬─────┘
                      │
        o2 ───────────┘
         CPW feedline

    See :cite:`mirhosseiniQuantumTransductionOptical2020` for experimental
    details of a piezo-optomechanical quantum transducer.

    Args:
        pad_size: (width, height) of each metal pad in µm.
        pad_gap: Gap between the two pads (piezoelectric region) in µm.
        feedline_length: Length of CPW feedline sections in µm.
        cross_section: Cross-section specification for CPW feedlines.
        layer_metal: Metal layer for the coupling pads.
        layer_piezo: Layer to mark the piezoelectric coupling region.

    Returns:
        Component with ports ``o1``, ``o2`` (CPW feedline) and
        ``piezo_coupler`` (placement port at the piezo gap centre).
    """
    c = Component()

    pad_w, pad_h = pad_size

    # --- Top metal pad ---
    top_pad = c.add_ref(gf.components.rectangle(size=(pad_w, pad_h), layer=layer_metal))
    top_pad.dmove((-pad_w / 2, pad_gap / 2))

    # --- Bottom metal pad ---
    bot_pad = c.add_ref(gf.components.rectangle(size=(pad_w, pad_h), layer=layer_metal))
    bot_pad.dmove((-pad_w / 2, -pad_gap / 2 - pad_h))

    # --- Piezo coupling region marker ---
    piezo_region = c.add_ref(
        gf.components.rectangle(size=(pad_w, pad_gap), layer=layer_piezo)
    )
    piezo_region.dmove((-pad_w / 2, -pad_gap / 2))

    # --- CPW feedlines extending horizontally from pad sides ---
    feed_left = c.add_ref(straight(length=feedline_length, cross_section=cross_section))
    feed_left.dmove((
        top_pad.dbbox().right - feed_left.ports["o1"].dcenter[0],
        top_pad.dcenter[1] - feed_left.ports["o1"].dcenter[1],
    ))

    feed_right = c.add_ref(
        straight(length=feedline_length, cross_section=cross_section)
    )
    feed_right.drotate(180)
    feed_right.dmove((
        bot_pad.dbbox().left - feed_right.ports["o1"].dcenter[0],
        bot_pad.dcenter[1] - feed_right.ports["o1"].dcenter[1],
    ))

    # --- Ports ---
    c.add_port(name="o1", port=feed_left.ports["o2"])
    c.add_port(name="o2", port=feed_right.ports["o2"])

    c.add_port(
        name="piezo_coupler",
        center=(0, 0),
        width=pad_w,
        orientation=90,
        layer=layer_piezo,
        port_type="placement",
    )

    # --- Metadata ---
    c.info["pad_width"] = pad_w
    c.info["pad_height"] = pad_h
    c.info["pad_gap"] = pad_gap
    c.info["transducer_type"] = "piezo_optomechanical"

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    show_components(electro_optic_transducer, piezo_transducer_coupler)
