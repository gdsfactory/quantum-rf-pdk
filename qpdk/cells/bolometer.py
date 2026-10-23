r"""Normal-metal hot-electron nanobolometer and calorimeter cells.

The absorber is a normal-metal nanowire on ``NM_DRAW`` between two clean
superconducting Al leads on ``JJ_AREA``. One lead connects the absorber to the
open end of a coplanar-waveguide input line and the other connects it to
ground, so the absorber terminates the line as a resistive load. Al probes
crossing the wire on ``JJ_AREA`` form normal-metal-insulator-superconductor
(NIS) tunnel junctions for thermometry. As in
:cite:`gasparinettiFastElectronThermometry2015`, they are each wired to an
isolated pad.

The layout cannot tell the transparent NS contacts of the leads from the
oxidised NIS probe junctions apart. Both are Al on ``JJ_AREA`` overlapping
``NM_DRAW``, and the oxidation step is set by the process, not the drawing.

Normal metal beneath and beyond the leads is drawn explicitly. Its volume is
reported in ``info``. The two-reservoir thermal model
:func:`qpdk.models.bolometer.nanobolometer_thermal_modes` takes the same
geometry parameters and derives the same volumes.
"""

from __future__ import annotations

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from klayout.db import DBox, Region

from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER

__all__ = ["nanobolometer", "nanocalorimeter"]

_LEAD_MARGIN = 0.1
_PATCH_MARGIN = 0.5


def _box(
    c: Component, layer: LayerSpec, x0: float, y0: float, x1: float, y1: float
) -> None:
    """Add an axis-aligned rectangle with corners ``(x0, y0)`` and ``(x1, y1)``."""
    c.add_polygon(
        [
            (min(x0, x1), min(y0, y1)),
            (max(x0, x1), min(y0, y1)),
            (max(x0, x1), max(y0, y1)),
            (min(x0, x1), max(y0, y1)),
        ],
        layer=layer,
    )


def _landing(
    c: Component,
    x: float,
    y: float,
    size: float,
    layer_lead: LayerSpec,
    layer_patch: LayerSpec,
) -> None:
    """Draw an Al landing pad centred at (x, y) with a galvanic-contact patch."""
    _box(c, layer_lead, x - size / 2, y - size / 2, x + size / 2, y + size / 2)
    p = size / 2 + _PATCH_MARGIN
    _box(c, layer_patch, x - p, y - p, x + p, y + p)


@gf.cell(tags=["detectors", "bolometers"])
def nanobolometer(
    absorber_length: float = 1.0,
    absorber_width: float = 0.135,
    normal_metal_thickness: float = 0.03,
    contact_overlap: float = 0.3,
    extension_length: float = 0.2,
    lead_width: float = 0.3,
    lead_length: float = 5.0,
    probe_positions: Sequence[float] = (-0.25, 0.25),
    probe_width: float = 0.1,
    pad_size: float = 4.0,
    pad_spacing: float = 4.0,
    landing_size: float = 2.0,
    cpw_length: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
    layer_normal_metal: LayerSpec = LAYER.NM_DRAW,
    layer_lead: LayerSpec = LAYER.JJ_AREA,
    layer_patch: LayerSpec = LAYER.JJ_PATCH,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    r"""Creates a normal-metal hot-electron nanobolometer.

    A normal-metal nanowire along :math:`y` absorbs the RF power delivered by
    the input line and is read out through Al tunnel probes. The default
    geometry follows the AuPd nanowire bolometers of
    :cite:`kokkoniemiNanobolometerUltralowNoise2019`,
    :cite:`girardCryogenicSensorEnabling2023` and
    :cite:`gunyhoSingleshotReadoutSuperconducting2024`, with a
    :math:`1\,\text{µm}` absorber and a :math:`30\,\text{nm}` film. Those papers
    do not report the wire width. The :math:`135\,\text{nm}` default is chosen to
    reproduce the volume :math:`V_0 \approx (240\,\text{nm})^3` of
    :cite:`goveniusDetectionZeptojouleMicrowave2016`. Their SNS thermometer islands are not drawn; NIS
    probes stand in for them.

    .. svgbob::

                          ┌─────────────┐
         o1 ══ CPW ══╗    │ M1_ETCH     │
                     ╚═══ lead (S) ═╦   │
                          │   NM    ║━━━━━┓  ┌──┐
                          │ absorber║━━━━━╋━━┤e1│
                          │         ║     ┗━━┤e2│
                          │  ┏━ lead (S)    └──┘
                          │  ┃          │
                          └──┸──────────┘
                            ground

    The normal metal extends ``contact_overlap`` under each lead and
    ``extension_length`` beyond it. That metal is a second electron reservoir
    thermally linked to the absorber, which sets the layout-resolved time
    constants of :func:`qpdk.models.bolometer.nanobolometer_thermal_modes`.

    Args:
        absorber_length: Normal-metal length between the leads in µm.
        absorber_width: Normal-metal wire width in µm.
        normal_metal_thickness: Normal-metal film thickness in µm. It is not drawn
            and is only used for the volumes in ``info``.
        contact_overlap: Length of normal metal under each lead (the NS contact) in µm.
        extension_length: Length of normal metal beyond each lead in µm.
        lead_width: Width of the Al lead wiring in µm.
        lead_length: Distance from the wire to the open end of the input line in µm.
        probe_positions: Positions of the NIS probes along the absorber, as
            fractions of ``absorber_length`` measured from its centre. Each must lie
            in :math:`(-1/2, 1/2)`.
        probe_width: Width of each NIS probe (its overlap along the wire) in µm.
        pad_size: Side length of the square probe pads in µm.
        pad_spacing: Gap between adjacent probe pads in µm.
        landing_size: Side length of the Al landing pads on the metal layer in µm.
        cpw_length: Length of the coplanar-waveguide input stub in µm.
        cross_section: Cross-section of the input line.
        layer_normal_metal: Layer of the normal-metal absorber.
        layer_lead: Layer of the Al leads and probes.
        layer_patch: Layer marking galvanic Al-to-metal contacts.
        layer_metal: Layer of the base metal pads.
        layer_etch: Layer of the etched pocket around the device.

    Returns:
        Component with the CPW input port ``o1`` and electrical probe ports
        ``e1``, ``e2``, ... ordered by probe position.

    Raises:
        ValueError: If a dimension is not positive or a probe lies outside the absorber.
    """
    for name, value in {
        "absorber_length": absorber_length,
        "absorber_width": absorber_width,
        "normal_metal_thickness": normal_metal_thickness,
        "contact_overlap": contact_overlap,
        "lead_width": lead_width,
        "lead_length": lead_length,
        "probe_width": probe_width,
    }.items():
        if value <= 0:
            raise ValueError(f"{name}={value} must be positive.")
    if extension_length < 0:
        raise ValueError(f"extension_length={extension_length} must be non-negative.")
    half_length = absorber_length / 2
    probe_ys = sorted(p * absorber_length for p in probe_positions)
    for y in probe_ys:
        if abs(y) + probe_width / 2 >= half_length:
            raise ValueError(
                f"Probe at y={y} µm with width {probe_width} µm does not fit on the "
                f"{absorber_length} µm absorber."
            )

    c = Component()
    # Wire spans x in [0, absorber_width] so its edges stay on the database grid
    x_wire = absorber_width
    wire_end = half_length + contact_overlap + extension_length
    _box(c, layer_normal_metal, 0, -wire_end, x_wire, wire_end)

    # Hot lead: NS contact at the top of the absorber, wired to the CPW centre conductor
    y_hot = half_length + contact_overlap / 2
    x_tip = -lead_length
    _box(
        c,
        layer_lead,
        x_tip - landing_size / 2,
        half_length,
        x_wire + _LEAD_MARGIN,
        half_length + contact_overlap,
    )
    _landing(c, x_tip - landing_size / 2, y_hot, landing_size, layer_lead, layer_patch)
    cpw = c << straight(length=cpw_length, cross_section=cross_section)
    cpw.dmove((x_tip - cpw_length, y_hot))
    xs = gf.get_cross_section(cross_section)
    cpw_half_extent = max(
        (abs(s.offset) + s.width / 2 for s in xs.sections), default=xs.width / 2
    )

    # Probes: route right, then fan out to pads stacked along y
    n_probes = len(probe_ys)
    pad_pitch = pad_size + pad_spacing
    pad_ys = [(k - (n_probes - 1) / 2) * pad_pitch for k in range(n_probes)]
    down = [k for k in range(n_probes) if pad_ys[k] < probe_ys[k]]
    up = [k for k in range(n_probes) if pad_ys[k] >= probe_ys[k]]
    rank = {k: i for i, k in enumerate(down)}
    rank |= {k: i for i, k in enumerate(reversed(up))}
    column_pitch = max(lead_width, probe_width) + 1.0
    x_column0 = x_wire + 1.0
    x_pad = x_column0 + n_probes * column_pitch + landing_size
    for k, (y_probe, y_pad) in enumerate(zip(probe_ys, pad_ys, strict=True)):
        x_column = x_column0 + rank[k] * column_pitch
        w = probe_width / 2
        _box(c, layer_lead, -_LEAD_MARGIN, y_probe - w, x_column + w, y_probe + w)
        _box(c, layer_lead, x_column - w, y_probe, x_column + w, y_pad)
        _box(
            c, layer_lead, x_column - w, y_pad - w, x_pad + landing_size / 2, y_pad + w
        )
        _box(
            c,
            layer_metal,
            x_pad,
            y_pad - pad_size / 2,
            x_pad + pad_size,
            y_pad + pad_size / 2,
        )
        _landing(
            c, x_pad + landing_size / 2, y_pad, landing_size, layer_lead, layer_patch
        )
        c.add_port(
            name=f"e{k + 1}",
            center=(x_pad + pad_size, y_pad),
            width=pad_size,
            orientation=0,
            layer=layer_metal,
            port_type="electrical",
        )

    # Etched pocket enclosing the device, open towards the CPW
    margin = pad_spacing + landing_size
    x_right = x_pad + pad_size + margin if n_probes else x_wire + margin
    pad_extent = max((abs(y) + pad_size / 2 for y in pad_ys), default=0)
    y_top = max(y_hot + cpw_half_extent, wire_end + margin, pad_extent + margin)
    y_bottom = min(y_hot - cpw_half_extent, -wire_end - margin, -pad_extent - margin)

    # Cold lead: NS contact at the bottom of the absorber, wired down to ground
    x_cold = -1.0 - lead_width / 2
    _box(
        c,
        layer_lead,
        x_cold - lead_width / 2,
        -half_length - contact_overlap,
        x_wire + _LEAD_MARGIN,
        -half_length,
    )
    _box(
        c,
        layer_lead,
        x_cold - lead_width / 2,
        -half_length,
        x_cold + lead_width / 2,
        y_bottom - landing_size / 2,
    )
    _landing(
        c, x_cold, y_bottom - landing_size / 2, landing_size, layer_lead, layer_patch
    )

    pocket = Region(DBox(x_tip, y_bottom, x_right, y_top).to_itype(c.kcl.dbu))
    pads = Region(c.begin_shapes_rec(gf.get_layer(layer_metal)))
    c.shapes(gf.get_layer(layer_etch)).insert(pocket - pads)

    c.add_port(port=cpw.ports["o1"], name="o1")

    side_length = contact_overlap + extension_length
    cross_section_area = absorber_width * normal_metal_thickness
    c.info["absorber_volume"] = absorber_length * cross_section_area
    c.info["covered_volume"] = 2 * contact_overlap * cross_section_area
    c.info["extension_volume"] = 2 * extension_length * cross_section_area
    c.info["reservoir_volume"] = 2 * side_length * cross_section_area
    c.info["link_length"] = side_length / 2
    c.info["link_cross_section"] = cross_section_area
    c.info["absorber_squares"] = absorber_length / absorber_width
    c.info["probe_junction_area"] = probe_width * absorber_width
    return c


@gf.cell(tags=["detectors", "bolometers"])
def nanocalorimeter(
    absorber_length: float = 20.0,
    absorber_width: float = 0.1,
    normal_metal_thickness: float = 0.025,
    contact_overlap: float = 0.2,
    extension_length: float = 0.2,
    probe_positions: Sequence[float] = (0.0,),
) -> Component:
    r"""Creates a normal-metal hot-electron calorimeter.

    The calorimeter is a :func:`nanobolometer` with the parameters of the Cu
    nanowire calorimeter of :cite:`gasparinettiFastElectronThermometry2015`.
    The wire is :math:`20\,\text{µm} \times 100\,\text{nm} \times 25\,\text{nm}`
    with a single NIS probe at its centre. That paper does not report the
    overlap of the clean NS contacts, so the overlap and extension are
    placeholders.

    Args:
        absorber_length: Normal-metal length between the leads in µm.
        absorber_width: Normal-metal wire width in µm.
        normal_metal_thickness: Normal-metal film thickness in µm.
        contact_overlap: Length of normal metal under each lead in µm.
        extension_length: Length of normal metal beyond each lead in µm.
        probe_positions: Probe positions as fractions of ``absorber_length``.

    Returns:
        Component with the CPW input port ``o1`` and one electrical port per probe.
    """
    c = Component()
    ref = c << nanobolometer(
        absorber_length=absorber_length,
        absorber_width=absorber_width,
        normal_metal_thickness=normal_metal_thickness,
        contact_overlap=contact_overlap,
        extension_length=extension_length,
        probe_positions=probe_positions,
    )
    c.add_ports(ref.ports)
    c.info.update(ref.cell.info.model_dump())
    return c
