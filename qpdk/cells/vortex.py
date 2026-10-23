"""Resonator geometries that control where Abrikosov vortices are trapped.

Two layouts from the literature are provided:

* :func:`neck_termination` follows Shulga *et al.*
  :cite:`shulgaObservationIndividualVortex2026`: the grounded end of a
  quarter-wave resonator is a linear taper into a narrow "neck" inside an
  etched, flux-focusing window, with an optional ring of flux-trapping holes
  around the window.
* :func:`cpw_bulge` follows Nsanzineza and Plourde
  :cite:`nsanzinezaTrappingSingleVortex2014`: a short, wider section
  of centre conductor at the current antinode of a half-wave resonator lowers
  the local trapping threshold so that vortices are trapped there first.

:func:`resonator_quarter_wave_neck` and :func:`resonator_half_wave_bulge`
place these structures at the current antinode of a meandered resonator.
"""

from __future__ import annotations

import math
from typing import Literal

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.resonator import resonator
from qpdk.cells.waveguides import bend_circular, straight, taper_cross_section
from qpdk.logger import logger
from qpdk.tech import LAYER, coplanar_waveguide, get_etch_section

__all__ = [
    "cpw_bulge",
    "neck_termination",
    "resonator_half_wave_bulge",
    "resonator_quarter_wave_neck",
    "vortex_trapping_threshold",
]

_FLUX_QUANTUM = 2.067833848e-15
"""Magnetic flux quantum :math:`\\Phi_0 = h / (2 e)` in Wb (exact in the 2019 SI).

Defined here rather than imported from :mod:`qpdk.models.constants`, because
cells must import without the optional ``models`` dependencies.
"""


def vortex_trapping_threshold(width: float, prefactor: float = 1.65) -> float:
    r"""Return the field above which vortices are trapped in a strip of given width.

    Field-cooled threshold of a thin-film strip in a perpendicular field,

    .. math::

        B_\text{th} = \alpha \frac{\Phi_0}{w^2}.

    The default :math:`\alpha = 1.65` is the vortex-antivortex energy model of
    Kuit *et al.* :cite:`kuitVortexTrappingExpulsion2008`. Stan, Field and
    Martinis :cite:`stanCriticalFieldComplete2004` measured
    :math:`\alpha \approx 1` in Nb strips. Measured thresholds in wider strips
    fall between the two :cite:`nsanzinezaTrappingSingleVortex2014`.
    Use the value to compare widths, not as an absolute prediction. It does not
    apply to zero-field cooling, where entry is limited by the edge barrier and
    depends on history, nor to strong microwave drive, which lowers the entry
    field :cite:`shulgaObservationIndividualVortex2026`.

    Args:
        width: Strip width in µm.
        prefactor: Dimensionless prefactor :math:`\alpha`.

    Returns:
        Threshold field in T.

    Raises:
        ValueError: If ``width`` or ``prefactor`` is not positive.
    """
    if not width > 0:
        raise ValueError(f"width must be positive, got {width}.")
    if not prefactor > 0:
        raise ValueError(f"prefactor must be positive, got {prefactor}.")
    return prefactor * _FLUX_QUANTUM / (width * 1e-6) ** 2


def _cpw_dimensions(cross_section: CrossSectionSpec) -> tuple[gf.CrossSection, float]:
    """Return the cross-section and its etch gap in µm."""
    xs = gf.get_cross_section(cross_section)
    return xs, get_etch_section(xs).width


@gf.cell(tags=["resonators", "waveguides"])
def neck_termination(
    cross_section: CrossSectionSpec = "cpw",
    taper_length: float = 100.0,
    taper_shape: Literal["linear", "smooth"] = "linear",
    neck_width: float = 2.0,
    neck_length: float = 325.0,
    window_width: float = 380.0,
    trap_hole_rows: int = 4,
    trap_hole_size: float = 4.0,
    trap_hole_pitch: float = 5.5,
    trap_hole_offset: float = 6.0,
    trap_hole_keepout: float = 24.0,
) -> Component:
    r"""Shorted CPW termination through a taper and a narrow neck.

    Geometry of the grounded end of the resonators in Shulga *et al.*
    :cite:`shulgaObservationIndividualVortex2026`. The centre conductor tapers
    from the CPW width to ``neck_width`` and continues as a straight neck that
    joins the ground plane at the far edge of a rectangular etched window, so
    the neck is a strip in a ground opening rather than a CPW with a fixed
    gap. The ground plane around the window focuses the applied field
    unevenly onto the taper and neck, and the neck sits at the
    current antinode, so a single trapped vortex shifts the resonance by a
    measurable amount. An optional ring of square etched holes along the two
    long sides and the far end of the window traps vortices in the ground
    plane, with a hole-free column where the neck current enters the ground.
    The ring is left open at the port end, where the CPW enters. Holes in the
    rest of the ground plane are left to
    :func:`qpdk.utils.fill_magnetic_vortices`.

    The paper states the 325 µm neck length and the 1-10 µm neck widths: 1 µm
    showed single-vortex frequency steps, 5 µm did not, and a 2 µm neck stayed
    free of vortices after cooling in 113 µT. The taper length (about
    100 µm) and the window size (about 380 µm by 425 µm) are estimated from
    its figures, and the paper does not describe the taper profile. In the
    paper vortices sit preferentially in the taper. Sharp corners crowd the
    current and lower the vortex-entry current
    :cite:`clemGeometrydependentCriticalCurrents2011`. The default linear
    taper only has shallow kinks at its ends (about 2° for the defaults), so
    the effect is small; ``taper_shape="smooth"`` removes them with a
    raised-cosine profile, which matters for short tapers.

    The hole defaults follow the usual rules for flux-trap arrays: the
    superconducting web between holes (1.5 µm) is narrower than the neck, so
    the holes trap before the neck does
    :cite:`nsanzinezaTrappingSingleVortex2014`, and the first row
    is 6 µm from the window edge. Chiaro *et al.* found that holes 6 µm or
    more from a CPW gap edge add no measurable dielectric loss
    :cite:`chiaroDielectricSurfaceLoss2016`. The window sits near the voltage
    node of the resonator, so its edges see a weaker electric field than a
    CPW gap at the open end, and the same margin is conservative here. At the
    far end the ground return current crowds around the
    junction with the neck, so the hole-free column ``trap_hole_keepout``
    must stay wider than the neck. For 1 µm necks use a narrower web
    (``trap_hole_pitch - trap_hole_size``); the cell logs a warning when the
    web is not narrower than the neck.

    Thresholds from :func:`vortex_trapping_threshold` are field-cooled
    estimates. Zero-field-cooled entry and microwave drive can trap vortices
    at much lower fields, and similar frequency jumps have been seen in
    resonators without a neck :cite:`kalashnikovEquidistantResonanceJumps2026`,
    so measure a resonator without a neck on the same chip as a control.

    .. svgbob::

                    ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫
                    ┌─────────────────────────┐ ▫ ▫
                    │ ─────┐                  │ ▫ ▫
        o1 ─────────┤      ├──────────────────┤  (keepout)
                    │ ─────┘   neck           │ ▫ ▫
                    │ taper                   │ ▫ ▫
                    └─────────────────────────┘ ▫ ▫
                    ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫

    Args:
        cross_section: CPW cross-section at port ``o1``.
        taper_length: Length of the taper in µm.
        taper_shape: ``"linear"`` for a straight-sided taper,
            ``"smooth"`` for a raised-cosine width profile with no corners.
        neck_width: Width of the neck in µm.
        neck_length: Length of the neck in µm.
        window_width: Width of the etched window across the line in µm.
            The window length is ``taper_length + neck_length``.
        trap_hole_rows: Number of rows in the ring of flux-trapping holes
            around the window. Zero disables the holes.
        trap_hole_size: Side of each square hole in µm.
        trap_hole_pitch: Centre-to-centre hole spacing in µm.
        trap_hole_offset: Width of the ground strip between the window and the
            first row of holes in µm.
        trap_hole_keepout: Minimum width of the hole-free column centred on
            the neck at the far end of the window in µm. The grid can make the
            column wider.

    Returns:
        Component with a single port ``o1``.

    Raises:
        ValueError: If a dimension is out of range.
    """
    xs, gap = _cpw_dimensions(cross_section)
    width = xs.width
    for name, value in {
        "taper_length": taper_length,
        "neck_width": neck_width,
        "neck_length": neck_length,
    }.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if neck_width >= width:
        raise ValueError(
            f"neck_width ({neck_width}) must be smaller than the CPW width ({width})."
        )
    if window_width < width + 2 * gap:
        raise ValueError(
            f"window_width ({window_width}) must be at least the CPW footprint "
            f"({width + 2 * gap})."
        )
    if taper_shape not in {"linear", "smooth"}:
        raise ValueError(
            f"taper_shape must be 'linear' or 'smooth', got {taper_shape!r}."
        )
    if trap_hole_rows < 0:
        raise ValueError(f"trap_hole_rows must be >= 0, got {trap_hole_rows}.")
    if trap_hole_rows and not 0 < trap_hole_size < trap_hole_pitch:
        raise ValueError(
            "trap_hole_size must be positive and smaller than trap_hole_pitch."
        )
    if trap_hole_rows and (trap_hole_offset < 0 or trap_hole_keepout <= neck_width):
        raise ValueError(
            "trap_hole_offset must be >= 0 and trap_hole_keepout must exceed "
            "neck_width."
        )
    if trap_hole_rows and trap_hole_pitch - trap_hole_size >= neck_width:
        logger.warning(
            "Flux-trap web ({} µm) is not narrower than the neck ({} µm), so "
            "the ground plane may trap vortices before the neck does.",
            trap_hole_pitch - trap_hole_size,
            neck_width,
        )

    c = Component()
    window_length = taper_length + neck_length
    c.add_polygon(
        [
            (0, -window_width / 2),
            (window_length, -window_width / 2),
            (window_length, window_width / 2),
            (0, window_width / 2),
        ],
        layer=LAYER.M1_ETCH,
    )
    if taper_shape == "linear":
        profile = [(0.0, width), (taper_length, neck_width)]
    else:
        n_points = 41
        profile = [
            (
                taper_length * i / (n_points - 1),
                neck_width
                + (width - neck_width)
                * (1 + math.cos(math.pi * i / (n_points - 1)))
                / 2,
            )
            for i in range(n_points)
        ]
    profile.append((window_length, neck_width))
    conductor = [(x, -w / 2) for x, w in profile]
    conductor += [(x, w / 2) for x, w in reversed(profile)]
    c.add_polygon(conductor, layer=LAYER.M1_DRAW)
    c.add_polygon(conductor, layer=LAYER.WG)

    if trap_hole_rows:
        # One square grid, anchored so that the innermost rows sit
        # ``trap_hole_offset`` from the far end and from both long sides.
        half = trap_hole_size / 2
        x0 = window_length + trap_hole_offset + half
        y0 = window_width / 2 + trap_hole_offset + half
        n_inner_x = math.floor((x0 - half) / trap_hole_pitch)
        n_inner_y = math.floor((y0 - half) / trap_hole_pitch)
        xs_grid = [
            (i, x0 + i * trap_hole_pitch) for i in range(-n_inner_x, trap_hole_rows)
        ]
        ys_grid = [
            (k, y0 + k * trap_hole_pitch) for k in range(-n_inner_y, trap_hole_rows)
        ]
        for i, x in xs_grid:
            for k, y_abs in ys_grid:
                if i < 0 and k < 0:
                    continue  # inside the window's footprint
                if i >= 0 and y_abs < (trap_hole_keepout + trap_hole_size) / 2:
                    continue  # current path from the neck into the ground
                for y in (y_abs, -y_abs):
                    c.add_polygon(
                        [
                            (x - half, y - half),
                            (x + half, y - half),
                            (x + half, y + half),
                            (x - half, y + half),
                        ],
                        layer=LAYER.M1_ETCH,
                    )

    c.add_port(
        name="o1",
        center=(0, 0),
        width=width,
        orientation=180,
        layer=LAYER.M1_DRAW,
        port_type="optical",
        cross_section=xs,
    )
    c.info["taper_length"] = taper_length
    c.info["taper_shape"] = taper_shape
    c.info["neck_width"] = neck_width
    c.info["neck_length"] = neck_length
    c.info["window_width"] = window_width
    c.info["window_length"] = window_length
    c.info["neck_trapping_threshold"] = vortex_trapping_threshold(neck_width)
    return c


@gf.cell(tags=["waveguides"])
def cpw_bulge(
    cross_section: CrossSectionSpec = "cpw",
    bulge_width: float = 25.0,
    bulge_length: float = 50.0,
    taper_length: float = 20.0,
) -> Component:
    r"""CPW section with a locally widened centre conductor.

    Following Nsanzineza and Plourde
    :cite:`nsanzinezaTrappingSingleVortex2014`, the wide section
    has a lower trapping threshold
    (:func:`vortex_trapping_threshold` scales as :math:`1/w^2`) so that,
    placed at the current antinode of a resonator, it is where vortices are
    trapped first. The gap is kept constant, so the ground-plane edges flare
    with the conductor; the paper does not state how its gap behaves.

    The paper used a 3 µm line widened to 8 µm over 50 µm. It observed the
    first vortex at 42 µT and then one more for every 5 µT, which matches one
    flux quantum per bulge area. The cell reports this step as
    ``bulge_flux_step`` :math:`= \Phi_0 / (w_\text{b} l_\text{b})`. The taper
    length is not given in the paper. The default ``bulge_width`` is close
    to the paper's width ratio (8/3) on the PDK's 10 µm CPW. To reproduce the paper use
    ``cross_section=coplanar_waveguide(width=3, gap=2)`` and ``bulge_width=8``.

    For the bulge to trap first it must be the widest conductor in the
    resonator: in the paper the 6 µm coupler trapped next, at 72 µT. Ground
    planes need flux-trap holes whose web is narrower than the line, for
    example :func:`qpdk.utils.fill_magnetic_vortices`.

    .. svgbob::

                    ┌────────┐
        o1 ────────/          \──────── o2
                   \          /
                    └────────┘

    Args:
        cross_section: CPW cross-section at both ports.
        bulge_width: Width of the widened centre conductor in µm.
        bulge_length: Length of the full-width section in µm.
        taper_length: Length of each linear taper in µm.

    Returns:
        Component with ports ``o1`` and ``o2``.

    Raises:
        ValueError: If a dimension is out of range.
    """
    xs, gap = _cpw_dimensions(cross_section)
    if bulge_width <= xs.width:
        raise ValueError(
            f"bulge_width ({bulge_width}) must exceed the CPW width ({xs.width})."
        )
    for name, value in {
        "bulge_length": bulge_length,
        "taper_length": taper_length,
    }.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    xs_bulge = coplanar_waveguide(width=bulge_width, gap=gap)
    c = Component()
    taper_in = c << taper_cross_section(
        cross_section1=xs,
        cross_section2=xs_bulge,
        length=taper_length,
        npoints=2,
        linear=True,
        width_type="linear",
    )
    middle = c << straight(length=bulge_length, cross_section=xs_bulge)
    taper_out = c << taper_cross_section(
        cross_section1=xs_bulge,
        cross_section2=xs,
        length=taper_length,
        npoints=2,
        linear=True,
        width_type="linear",
    )
    middle.connect("o1", taper_in.ports["o2"])
    taper_out.connect("o1", middle.ports["o2"])
    c.add_port("o1", port=taper_in.ports["o1"])
    c.add_port("o2", port=taper_out.ports["o2"])
    c.flatten()

    c.info["length"] = bulge_length + 2 * taper_length
    c.info["bulge_width"] = bulge_width
    c.info["bulge_length"] = bulge_length
    c.info["taper_length"] = taper_length
    c.info["bulge_trapping_threshold"] = vortex_trapping_threshold(bulge_width)
    c.info["line_trapping_threshold"] = vortex_trapping_threshold(xs.width)
    c.info["bulge_flux_step"] = _FLUX_QUANTUM / (bulge_width * bulge_length * 1e-12)
    return c


@gf.cell(tags=["resonators"])
def resonator_quarter_wave_neck(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    taper_length: float = 100.0,
    taper_shape: Literal["linear", "smooth"] = "linear",
    neck_width: float = 2.0,
    neck_length: float = 325.0,
    window_width: float = 380.0,
    trap_hole_rows: int = 4,
    trap_hole_size: float = 4.0,
    trap_hole_pitch: float = 5.5,
    trap_hole_offset: float = 6.0,
    trap_hole_keepout: float = 24.0,
) -> Component:
    """Quarter-wave resonator grounded through a :func:`neck_termination`.

    The open end carries the coupling port ``o1``. At the grounded end, the
    current antinode, the line turns away from the meander with a 90° bend and
    ends in the taper, neck and etched window of
    :cite:`shulgaObservationIndividualVortex2026`, so the window does not
    overlap the meander.

    Args:
        length: CPW length from the open end to the start of the taper,
            including the 90° bend, in µm.
        meanders: Number of meander sections.
        bend_spec: Bend used in the meanders and for the turn into the window.
        cross_section: CPW cross-section of the resonator.
        taper_length: See :func:`neck_termination`.
        taper_shape: See :func:`neck_termination`.
        neck_width: See :func:`neck_termination`.
        neck_length: See :func:`neck_termination`.
        window_width: See :func:`neck_termination`.
        trap_hole_rows: See :func:`neck_termination`.
        trap_hole_size: See :func:`neck_termination`.
        trap_hole_pitch: See :func:`neck_termination`.
        trap_hole_offset: See :func:`neck_termination`.
        trap_hole_keepout: See :func:`neck_termination`.

    Returns:
        Component with the open-end port ``o1``.
    """
    c = Component()
    xs = gf.get_cross_section(cross_section)
    turn = gf.get_component(bend_spec, cross_section=xs, angle=90)
    meander_length = length - turn.info["length"]
    body = c << resonator(
        length=meander_length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=xs,
        open_start=False,
        open_end=True,
    )
    short_end = body.ports["o1"]
    turn_ref = c << turn
    # Turn away from the meander, which lies on the other side of the line.
    meander_below = body.dbbox().center().y < short_end.dcenter[1]
    turn_ref.connect("o1", short_end, mirror=meander_below)
    neck = c << neck_termination(
        cross_section=xs,
        taper_length=taper_length,
        taper_shape=taper_shape,
        neck_width=neck_width,
        neck_length=neck_length,
        window_width=window_width,
        trap_hole_rows=trap_hole_rows,
        trap_hole_size=trap_hole_size,
        trap_hole_pitch=trap_hole_pitch,
        trap_hole_offset=trap_hole_offset,
        trap_hole_keepout=trap_hole_keepout,
    )
    neck.connect("o1", turn_ref.ports["o2"])
    c.add_port("o1", port=body.ports["o2"])

    c.info["length"] = body.cell.info["length"] + turn.info["length"]
    c.info["resonator_type"] = "quarter_wave"
    c.info["cross_section_name"] = xs.name
    c.info.update(neck.cell.info.model_dump())
    return c


@gf.cell(tags=["resonators"])
def resonator_half_wave_bulge(
    length: float = 8000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    bulge_width: float = 25.0,
    bulge_length: float = 50.0,
    taper_length: float = 20.0,
) -> Component:
    """Half-wave resonator with a :func:`cpw_bulge` at its centre.

    The centre of an open-ended half-wave resonator is the current antinode of
    the fundamental mode (and a current node of the first harmonic), as in
    :cite:`nsanzinezaTrappingSingleVortex2014`. The bulge sits in
    the middle of the central meander row, so ``meanders`` must be even.

    Args:
        length: Total centre-line length, including bends and the bulge, in µm.
        meanders: Number of meander bends. Must be even.
        bend_spec: Bend used in the meanders.
        cross_section: CPW cross-section of the resonator.
        bulge_width: See :func:`cpw_bulge`.
        bulge_length: See :func:`cpw_bulge`.
        taper_length: See :func:`cpw_bulge`.

    Returns:
        Component with open-end ports ``o1`` and ``o2``.

    Raises:
        ValueError: If ``meanders`` is odd or ``length`` is too short.
    """
    if meanders < 0 or meanders % 2:
        raise ValueError(f"meanders must be even and non-negative, got {meanders}.")
    xs, _ = _cpw_dimensions(cross_section)
    bend = gf.get_component(bend_spec, cross_section=xs, angle=180, angular_step=4)
    bulge = cpw_bulge(
        cross_section=xs,
        bulge_width=bulge_width,
        bulge_length=bulge_length,
        taper_length=taper_length,
    )
    n_rows = meanders + 1
    row_length = (length - meanders * bend.info["length"]) / n_rows
    side_length = (row_length - bulge.info["length"]) / 2
    if side_length <= 0:
        raise ValueError(
            f"length ({length}) is too short for {meanders} meanders and a "
            f"{bulge.info['length']} µm bulge."
        )
    row = straight(length=row_length, cross_section=xs)
    side = straight(length=side_length, cross_section=xs)

    centre_row = meanders // 2
    chain: list[tuple[Component, bool]] = []
    for i in range(n_rows):
        if i == centre_row:
            chain += [(side, False), (bulge, False), (side, False)]
        else:
            chain.append((row, False))
        if i < meanders:
            chain.append((bend, i % 2 == 0))

    c = Component()
    refs = []
    for component, mirror in chain:
        ref = c << component
        if refs:
            ref.connect("o1", refs[-1].ports["o2"], mirror=mirror)
        refs.append(ref)

    etch = get_etch_section(xs)
    open_etch = gf.c.rectangle(
        size=(etch.width, 2 * etch.width + xs.width),
        layer=etch.layer,
        centered=True,
        port_type="optical",
        port_orientations=(0, 180),
    )
    start_etch = c << open_etch
    start_etch.connect(
        "o2", refs[0].ports["o1"], allow_width_mismatch=True, allow_layer_mismatch=True
    )
    end_etch = c << open_etch
    end_etch.connect(
        "o1", refs[-1].ports["o2"], allow_width_mismatch=True, allow_layer_mismatch=True
    )
    c.add_port("o1", port=start_etch.ports["o1"])
    c.add_port("o2", port=end_etch.ports["o2"])

    c.info["length"] = (
        meanders * bend.info["length"]
        + (n_rows - 1) * row.info["length"]
        + 2 * side.info["length"]
        + bulge.info["length"]
    )
    c.info["resonator_type"] = "half_wave"
    c.info["cross_section_name"] = xs.name
    c.info.update({k: v for k, v in bulge.info.model_dump().items() if k != "length"})
    return c
