r"""Giant-atom transmons coupled to a CPW at several points.

A giant atom couples to a waveguide at points spaced by a distance comparable
to the wavelength :cite:`kockumDesigningFrequencydependentRelaxation2014`.
These cells draw a grounded transmon whose island reaches a coplanar
waveguide through :math:`K` coupling arms. The waveguide between neighbouring
coupling points is meandered to the requested length, after the devices of
:cite:`kannanWaveguideQuantumElectrodynamics2020` and
:cite:`vadirajEngineeringLevelStructure2021`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec
from kfactory import kdb
from kfactory.conf import CheckInstances
from klayout.db import Region

from qpdk.cells.airbridge import airbridge
from qpdk.cells.junction import squid_junction
from qpdk.tech import LAYER

__all__ = ["braided_giant_atoms", "giant_atom_transmon"]

_PRECISION = 3  # Decimal places (nm grid) for drawn coordinates in µm.


def _meander_points(
    start: float,
    stop: float,
    path_length: float,
    side: int,
    bend_radius: float,
    meander_pitch: float,
) -> tuple[list[tuple[float, float]], float]:
    """Return the corner points of a one-sided serpentine along ``y = 0``.

    The serpentine runs from ``(start, 0)`` to ``(stop, 0)`` with U-shaped
    excursions to ``y = side * depth``. With quarter-circle bends of radius
    ``bend_radius`` at every corner, the smoothed path is ``path_length`` long.

    Returns:
        The corner points, excluding the end points, and the excursion depth.

    Raises:
        ValueError: If the length does not fit between ``start`` and ``stop``.
    """
    span = stop - start
    extra = path_length - span
    if abs(extra) < 10 ** (-_PRECISION):
        return [], 0.0
    if extra < 0:
        msg = (
            f"Waveguide length {path_length} µm between coupling points is shorter "
            f"than their spacing {span} µm."
        )
        raise ValueError(msg)

    bend_correction = (2 - math.pi / 2) * bend_radius  # Manhattan minus arc length
    n_excursions = math.floor(
        (span - 2 * bend_radius + meander_pitch) / (2 * meander_pitch)
    )
    while n_excursions > 0:
        depth = (extra + 4 * n_excursions * bend_correction) / (2 * n_excursions)
        if depth >= 2 * bend_radius:
            break
        n_excursions -= 1
    if n_excursions <= 0:
        msg = (
            f"Cannot meander {path_length} µm into a spacing of {span} µm with "
            f"bend_radius={bend_radius} µm and meander_pitch={meander_pitch} µm. "
            "Increase point_spacing or the segment length."
        )
        raise ValueError(msg)

    depth = round(depth, _PRECISION)
    points: list[tuple[float, float]] = []
    for j in range(n_excursions):
        x_down = start + bend_radius + 2 * j * meander_pitch
        x_up = x_down + meander_pitch
        points += [
            (x_down, 0.0),
            (x_down, side * depth),
            (x_up, side * depth),
            (x_up, 0.0),
        ]
    return points, depth


def _giant_atoms(
    coupler_lengths: Sequence[float],
    segment_lengths: Sequence[float],
    meander_sides: Sequence[int],
    atoms: Sequence[tuple[str, Sequence[int], int]],
    point_spacing: float,
    coupler_gap: float,
    pad_width: float,
    arm_length: float,
    arm_width: float,
    island_width: float,
    island_extension: float,
    gap_width: float,
    end_length: float,
    bend_radius: float,
    meander_pitch: float,
    cross_section: CrossSectionSpec,
    junction_spec: ComponentSpec,
    layer_metal: LayerSpec,
    layer_etch: LayerSpec,
    airbridges: bool,
) -> Component:
    """Draw transmons that share one meandered CPW.

    Coupling point ``k`` is centred at ``x = k * point_spacing`` on the
    waveguide at ``y = 0``. Each entry of ``atoms`` is ``(name, points,
    side)``: the qubit couples at ``points`` and sits above (``side=1``) or
    below (``side=-1``) the waveguide.

    Returns:
        Component with the waveguide ports ``o1`` and ``o2`` and one junction
        port per atom.

    Raises:
        ValueError: If the inputs are inconsistent or the meander does not fit.
    """
    n_points = len(coupler_lengths)
    if len(segment_lengths) != n_points - 1:
        msg = (
            f"Got {n_points} coupler lengths and {len(segment_lengths)} segment "
            "lengths; there must be one segment fewer than coupling points."
        )
        raise ValueError(msg)
    if meander_pitch < 2 * bend_radius:
        msg = f"meander_pitch={meander_pitch} must be at least 2 * bend_radius."
        raise ValueError(msg)
    for k in range(n_points - 1):
        if (coupler_lengths[k] + coupler_lengths[k + 1]) / 2 >= point_spacing:
            msg = (
                f"Coupling pads {k} and {k + 1} overlap; point_spacing="
                f"{point_spacing} µm must exceed half their summed lengths."
            )
            raise ValueError(msg)
    bridge = None
    if airbridges and any(len(points) > 1 for _, points, _ in atoms):
        bridge = airbridge(bridge_length=arm_width + 2 * gap_width + 10.0)
        bridge_pad = bridge.dbbox().width()  # Landing-pad extent along the arm.
        if arm_length < bridge_pad + 2 * gap_width:
            msg = (
                f"arm_length={arm_length} µm is too short for the airbridges; it "
                f"must be at least {bridge_pad + 2 * gap_width} µm."
            )
            raise ValueError(msg)

    xs = gf.get_cross_section(cross_section)
    line_half_extent = max(abs(s.offset) + s.width / 2 for s in xs.sections)
    positions = [k * point_spacing for k in range(n_points)]

    # Waveguide: straight coupler sections joined by serpentines.
    points = [(positions[0] - coupler_lengths[0] / 2 - end_length, 0.0)]
    depths = []
    for k in range(n_points - 1):
        start = positions[k] + coupler_lengths[k] / 2
        stop = positions[k + 1] - coupler_lengths[k + 1] / 2
        path_length = (
            segment_lengths[k] - (coupler_lengths[k] + coupler_lengths[k + 1]) / 2
        )
        meander, depth = _meander_points(
            start, stop, path_length, meander_sides[k], bend_radius, meander_pitch
        )
        points += meander
        depths.append(depth)
    points.append((positions[-1] + coupler_lengths[-1] / 2 + end_length, 0.0))

    if len(points) == 2:
        path = gf.path.straight(length=points[1][0] - points[0][0])
        path.move(points[0])
    else:
        path = gf.path.smooth(np.asarray(points), radius=bend_radius, bend=gf.path.arc)
    expected_length = (
        sum(segment_lengths)
        + (coupler_lengths[0] + coupler_lengths[-1]) / 2
        + 2 * end_length
    )
    # Each discretised bend is shorter than its arc by well under 1 nm.
    if not math.isclose(
        path.length(), expected_length, abs_tol=1e-2 + 1e-3 * len(points)
    ):
        msg = (
            f"Drawn waveguide is {path.length():.3f} µm long, expected "
            f"{expected_length:.3f} µm."
        )
        raise ValueError(msg)

    c = Component()
    line_ref = c.add_ref(gf.path.extrude(path, cross_section=xs))
    qubit_metal = Region()  # Qubit metal only; the CPW keeps its own gap.
    bridge_positions: list[tuple[float, float]] = []

    junction_positions: dict[str, tuple[float, float, int]] = {}
    pad_inner = xs.width / 2 + coupler_gap
    for name, atom_points, side in atoms:
        x_min = min(positions[k] for k in atom_points)
        x_max = max(positions[k] for k in atom_points)
        # Lengthen the arms past any meander that lies between this qubit's
        # own coupling points on its side of the waveguide.
        enclosed = [
            depths[k]
            for k in range(n_points - 1)
            if meander_sides[k] == side and x_min <= positions[k] < x_max
        ]
        island_x = (
            x_min - island_extension - gap_width,
            x_max + island_extension + gap_width,
        )
        for k in range(n_points - 1):
            meander_x = (
                positions[k] + coupler_lengths[k] / 2,
                positions[k + 1] - coupler_lengths[k + 1] / 2,
            )
            if (
                meander_sides[k] == side
                and depths[k] > 0
                and not x_min <= positions[k] < x_max
                and meander_x[0] < island_x[1]
                and island_x[0] < meander_x[1]
            ):
                msg = (
                    f"The meander of segment {k} lies under the island of "
                    f"{name!r}. Reduce island_extension or increase "
                    "point_spacing."
                )
                raise ValueError(msg)
        arm = arm_length
        if enclosed:
            clearance = max(enclosed) + line_half_extent + 2 * gap_width
            arm = max(arm, clearance - pad_inner - pad_width)
        island_inner = pad_inner + pad_width + arm

        rectangles = [
            # Island bar spanning all arms.
            (
                x_min - island_extension,
                x_max + island_extension,
                island_inner,
                island_inner + island_width,
            )
        ]
        for k in atom_points:
            rectangles += [
                # Coupling pad parallel to the waveguide.
                (
                    positions[k] - coupler_lengths[k] / 2,
                    positions[k] + coupler_lengths[k] / 2,
                    pad_inner,
                    pad_inner + pad_width,
                ),
                # Arm from the pad to the island.
                (
                    positions[k] - arm_width / 2,
                    positions[k] + arm_width / 2,
                    pad_inner + pad_width,
                    island_inner,
                ),
            ]
        for x0, x1, y0, y1 in rectangles:
            ys = sorted((side * y0, side * y1))
            c.add_polygon(
                [(x0, ys[0]), (x1, ys[0]), (x1, ys[1]), (x0, ys[1])],
                layer=layer_metal,
            )
            qubit_metal.insert(kdb.DBox(x0, ys[0], x1, ys[1]).to_itype(c.kcl.dbu))
        if len(atom_points) > 1:
            # The arms and the waveguide enclose a ground patch; bridge each
            # arm to tie it to the surrounding ground plane.
            arm_centre = side * (pad_inner + pad_width + arm / 2)
            bridge_positions += [(positions[k], arm_centre) for k in atom_points]
        junction_positions[name] = (
            (x_min + x_max) / 2,
            side * (island_inner + island_width + gap_width / 2),
            side,
        )

    c.add_ports(line_ref.ports)
    c.flatten(merge=True)

    # Qubit etch: grow only the qubit metal by gap_width, then keep every
    # drawn metal shape. The waveguide keeps the gap of its cross-section.
    etch_region = gf.component.size(qubit_metal, gap_width)
    # Merge with the waveguide etch and clear any etch left on drawn metal,
    # including sub-grid slivers where the bend discretisations differ.
    metal_index = c.kcl.layout.layer(*gf.get_layer_tuple(layer_metal))
    etch_index = c.kcl.layout.layer(*gf.get_layer_tuple(layer_etch))
    draw_region = Region(c.shapes(metal_index)).merged()
    etch_region = (Region(c.shapes(etch_index)) + etch_region).merged()
    # Close sub-grid ground slivers left between the qubit etch and the
    # discretised bends of the waveguide, then clear the drawn metal.
    closing = 10  # Database units.
    etch_region = (etch_region + draw_region).sized(closing).sized(-closing)
    etch_region -= draw_region
    c.shapes(etch_index).clear()
    c.shapes(etch_index).insert(etch_region)

    if bridge is not None:
        for x, y in bridge_positions:
            bridge_ref = c.add_ref(bridge)
            bridge_ref.rotate(90)
            bridge_ref.dcenter = (x, y)

    for name, (x, y, side) in junction_positions.items():
        junction_ref = c.add_ref(gf.get_component(junction_spec))
        junction_ref.rotate(135 if side > 0 else -45)
        junction_ref.dcenter = (x, y)
        c.add_port(
            name=name,
            center=junction_ref.dcenter,
            width=gf.snap.snap_to_grid(junction_ref.size_info.height, grid_factor=2),
            orientation=270 if side > 0 else 90,
            layer=LAYER.JJ_AREA,
            port_type="placement",
        )

    c.info["meander_depths"] = depths
    c.info["line_length"] = round(path.length(), _PRECISION)
    return c


@gf.cell(check_instances=CheckInstances.IGNORE, tags=["qubits", "transmons"])
def giant_atom_transmon(
    coupler_lengths: tuple[float, ...] = (160.0, 160.0),
    segment_lengths: tuple[float, ...] = (13100.0,),
    point_spacing: float = 1000.0,
    coupler_gap: float = 8.0,
    pad_width: float = 10.0,
    arm_length: float = 200.0,
    arm_width: float = 10.0,
    island_width: float = 30.0,
    island_extension: float = 50.0,
    gap_width: float = 10.0,
    end_length: float = 100.0,
    bend_radius: float = 50.0,
    meander_pitch: float = 120.0,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = squid_junction,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
    airbridges: bool = True,
) -> Component:
    r"""Grounded transmon coupled capacitively to a CPW at :math:`K` points.

    The island is a bar above the waveguide with :math:`K` arms. Each arm ends
    in a pad that runs parallel to the waveguide for ``coupler_lengths[k]``
    at an edge-to-edge distance ``coupler_gap``. Between neighbouring coupling
    points the waveguide meanders below the qubit, so that the length along
    the waveguide between the centres of points :math:`k` and :math:`k+1` is
    ``segment_lengths[k]``. The junction grounds the far edge of the island.

    .. svgbob::

                  ┌──────────── island ────────────┐
                  └──┬──────────────────────────┬──┘
                     │ arm                      │
                  ┌──┴──┐ pad                ┌──┴──┐
        o1 ───────┴─────┴──┐  ┌──┐  ┌──┐  ┌─┴─────┴─────── o2
                           │  │  │  │  │  │
                           └──┘  └──┘  └──┘
                          meander, segment_lengths[0]

    The phase between points is :math:`\varphi = \omega \ell / v`, where
    :math:`\ell` is the segment length and :math:`v` the phase velocity of the
    CPW. The defaults space the two points by :math:`\lambda/2` at
    :math:`4.645\,\text{GHz}` on the ``cpw`` cross-section, the frequency at
    which the giant atoms of :cite:`kannanWaveguideQuantumElectrodynamics2020`
    decouple from the waveguide.

    For a rough coupling capacitance :math:`C_k` of each pad for
    :func:`~qpdk.models.giant_atom.giant_atom_transmon`, call
    :func:`~qpdk.models.couplers.cpw_cpw_coupling_capacitance` with ``f``,
    ``length=coupler_lengths[k]``, ``gap=coupler_gap`` and
    ``cross_section``. It models two parallel CPWs rather than a pad with
    etch on one side, so it is an estimate only. The default
    :math:`160\,\text{µm}` pad at :math:`8\,\text{µm}` gives about
    :math:`8\,\text{fF}`. The island capacitance to ground depends on the
    geometry and is not extracted here.

    For :math:`K \ge 2` the arms and the waveguide enclose a ground patch,
    which includes the inside of each meander turn. With ``airbridges=True``
    one bridge per arm ties it to the ground plane. No bridges are drawn
    across the meander itself.

    Args:
        coupler_lengths: Length of each coupling pad along the waveguide in µm.
            Sets the number of coupling points :math:`K`.
        segment_lengths: Waveguide length between the centres of consecutive
            coupling points in µm, :math:`K - 1` values. A segment must equal
            the straight distance between the pads or exceed it by enough for
            at least one meander excursion of depth ``2 * bend_radius``;
            lengths in between raise ``ValueError``.
        point_spacing: Straight-line distance between consecutive coupling
            points in µm.
        coupler_gap: Edge-to-edge distance between pad and waveguide centre
            conductor in µm.
        pad_width: Width of the coupling pads in µm.
        arm_length: Length of the arms between pads and island in µm.
        arm_width: Width of the arms in µm.
        island_width: Width of the island bar in µm.
        island_extension: Length of the island beyond the outer arms in µm.
        gap_width: Width of the etched gap around the qubit metal in µm.
        end_length: Straight waveguide before the first and after the last
            coupling pad in µm.
        bend_radius: Bend radius of the meander in µm.
        meander_pitch: Distance between neighbouring meander legs in µm.
        cross_section: Cross-section of the waveguide.
        junction_spec: Component specification of the Josephson junction.
        layer_metal: Layer for the qubit metal.
        layer_etch: Layer for the etched gap around the qubit.
        airbridges: Bridge each arm with an
            :func:`~qpdk.cells.airbridge.airbridge` to tie the ground patch
            that the arms and the waveguide enclose to the ground plane.

    Returns:
        Component: Ports ``o1`` and ``o2`` at the waveguide ends and a
        ``junction`` placement port.
    """
    return _giant_atoms(
        coupler_lengths=coupler_lengths,
        segment_lengths=segment_lengths,
        meander_sides=[-1] * len(segment_lengths),
        atoms=[("junction", range(len(coupler_lengths)), 1)],
        point_spacing=point_spacing,
        coupler_gap=coupler_gap,
        pad_width=pad_width,
        arm_length=arm_length,
        arm_width=arm_width,
        island_width=island_width,
        island_extension=island_extension,
        gap_width=gap_width,
        end_length=end_length,
        bend_radius=bend_radius,
        meander_pitch=meander_pitch,
        cross_section=cross_section,
        junction_spec=junction_spec,
        layer_metal=layer_metal,
        layer_etch=layer_etch,
        airbridges=airbridges,
    )


@gf.cell(check_instances=CheckInstances.IGNORE, tags=["qubits", "transmons"])
def braided_giant_atoms(
    coupler_lengths: tuple[float, float] = (160.0, 160.0),
    segment_length: float = 6550.0,
    point_spacing: float = 1000.0,
    coupler_gap: float = 8.0,
    pad_width: float = 10.0,
    arm_length: float = 200.0,
    arm_width: float = 10.0,
    island_width: float = 30.0,
    island_extension: float = 50.0,
    gap_width: float = 10.0,
    end_length: float = 100.0,
    bend_radius: float = 50.0,
    meander_pitch: float = 120.0,
    cross_section: CrossSectionSpec = "cpw",
    junction_spec: ComponentSpec = squid_junction,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
    airbridges: bool = True,
) -> Component:
    r"""Two giant transmons in the braided configuration.

    Qubit a couples at points 1 and 3 and qubit b at points 2 and 4 of one
    waveguide, ordered :math:`x_{\text{a}1} < x_{\text{b}1} < x_{\text{a}2} <
    x_{\text{b}2}` with equal waveguide length ``segment_length`` between
    neighbouring points :cite:`kannanWaveguideQuantumElectrodynamics2020`.
    Qubit a sits above the waveguide and qubit b below. The arms of qubit b
    are lengthened past the meander that lies between its coupling points;
    with the defaults that meander is about :math:`1\,\text{mm}` deep, so
    qubit b has :math:`1\,\text{mm}` arms and a :math:`2.1\,\text{mm}`
    island. A larger ``point_spacing`` gives shallower meanders and shorter
    arms. The island geometry, and so its capacitance, is not matched to the
    ``qubit_capacitance`` of :func:`~qpdk.models.giant_atom.giant_atom_transmon`.

    The default ``segment_length`` is :math:`\lambda/4` at
    :math:`4.645\,\text{GHz}` on the ``cpw`` cross-section, so each qubit is
    decoherence-free there while the qubits stay coupled through the
    waveguide :cite:`kannanWaveguideQuantumElectrodynamics2020`. Each qubit
    on its own is a :func:`giant_atom_transmon` with
    ``segment_lengths=(2 * segment_length,)``, which for the defaults of both
    cells is the same :math:`13100\,\text{µm}`.

    Args:
        coupler_lengths: Pad length of qubit a and of qubit b in µm, the same
            at both coupling points of a qubit.
        segment_length: Waveguide length between neighbouring coupling points
            in µm.
        point_spacing: Straight-line distance between neighbouring coupling
            points in µm.
        coupler_gap: Edge-to-edge distance between pad and waveguide centre
            conductor in µm.
        pad_width: Width of the coupling pads in µm.
        arm_length: Minimum length of the arms between pads and island in µm.
        arm_width: Width of the arms in µm.
        island_width: Width of the island bars in µm.
        island_extension: Length of the islands beyond the outer arms in µm.
        gap_width: Width of the etched gap around the qubit metal in µm.
        end_length: Straight waveguide before the first and after the last
            coupling pad in µm.
        bend_radius: Bend radius of the meanders in µm.
        meander_pitch: Distance between neighbouring meander legs in µm.
        cross_section: Cross-section of the waveguide.
        junction_spec: Component specification of the Josephson junctions.
        layer_metal: Layer for the qubit metal.
        layer_etch: Layer for the etched gaps around the qubits.
        airbridges: Bridge each arm with an
            :func:`~qpdk.cells.airbridge.airbridge` to tie the ground patches
            that the arms and the waveguide enclose to the ground plane.

    Returns:
        Component: Ports ``o1`` and ``o2`` at the waveguide ends and the
        placement ports ``junction_a`` and ``junction_b``.
    """
    return _giant_atoms(
        coupler_lengths=(*coupler_lengths, *coupler_lengths),
        segment_lengths=(segment_length,) * 3,
        meander_sides=[-1, -1, 1],
        atoms=[("junction_a", (0, 2), 1), ("junction_b", (1, 3), -1)],
        point_spacing=point_spacing,
        coupler_gap=coupler_gap,
        pad_width=pad_width,
        arm_length=arm_length,
        arm_width=arm_width,
        island_width=island_width,
        island_extension=island_extension,
        gap_width=gap_width,
        end_length=end_length,
        bend_radius=bend_radius,
        meander_pitch=meander_pitch,
        cross_section=cross_section,
        junction_spec=junction_spec,
        layer_metal=layer_metal,
        layer_etch=layer_etch,
        airbridges=airbridges,
    )
