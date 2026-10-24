r"""Hyperbolic lattices of coplanar-waveguide resonators.

Layout for the edge-resonator scheme of Kollár et al.
:cite:`kollarHyperbolicLatticesCircuit2019`: identical CPW resonators sit on
the edges of a {p,q} tiling of the Poincaré disk (see
:mod:`qpdk.hyperbolic`) and meet at ``q``-pin capacitive couplers on its
vertices. Photons hop between resonators that share a coupler, so the lattice
realises a tight-binding model on the line graph of the tiling.

The tiling is drawn with straight chords, scaled so the unit disk has radius
``disk_radius``. Each resonator meanders inside the quadrilateral spanned by
its two end vertices and the centroids of the two faces next to it, so
resonators that do not share a coupler never come closer than twice the
``margin``.

Only the couplers and resonators are instances, placed by on-grid
translation; their rotation is a cell parameter because gdsfactory rejects
off-grid instances.

The vertex-resonator scheme of Xu et al.
:cite:`xuHyperbolicSuperconductingResonator2025` shares the tiling
(:meth:`qpdk.hyperbolic.HyperbolicTiling.distance_weighted_adjacency`) but
has no layout cell yet.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from itertools import pairwise, starmap

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec
from klayout import db as kdb

from qpdk.cells.launcher import launcher
from qpdk.hyperbolic import hyperbolic_tiling
from qpdk.tech import LAYER, get_etch_section

__all__ = [
    "hyperbolic_edge_resonator",
    "hyperbolic_lattice",
    "hyperbolic_vertex_coupler",
]

_GRID = 1e-3  # µm, the database unit


_SAGITTA = 0.01  # µm, maximum deviation of polygonised arcs


def _arc(radius: float, angle: float, **kwargs: float) -> gf.Path:
    """Circular arc polygonised to within :data:`_SAGITTA` of the true curve."""
    step = 2 * math.degrees(math.acos(1 - _SAGITTA / radius))
    return gf.path.arc(
        radius=radius,
        angle=angle,
        npoints=max(2, math.ceil(abs(angle) / step) + 1),
        **kwargs,
    )


def _snap(x: float) -> float:
    """Round ``x`` to the layout grid."""
    return round(x / _GRID) * _GRID


def _unit(angle_deg: float) -> np.ndarray:
    """Unit vector at ``angle_deg`` degrees."""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def _strip(start: np.ndarray, end: np.ndarray, width: float) -> kdb.DPolygon:
    """Rectangle of ``width`` centred on the segment ``start``–``end``."""
    d = end - start
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * width / 2
    pts = [start + n, end + n, end - n, start - n]
    return kdb.DPolygon(list(starmap(kdb.DPoint, pts)))


def _disk(radius: float, npoints: int = 128) -> kdb.DPolygon:
    """Disk of ``radius`` centred on the origin."""
    return kdb.DPolygon.ellipse(kdb.DBox(-radius, -radius, radius, radius), npoints)


def _insert(c: Component, layer: tuple[int, int], region: kdb.Region) -> None:
    """Insert ``region`` into ``c`` on ``layer``."""
    c.kdb_cell.shapes(c.kcl.layer(*layer)).insert(region)


def _vertex_coupler_port_offsets(
    q: int, rotation: float, pin_radius: float, lead_length: float
) -> list[tuple[float, float]]:
    """On-grid port centres of :func:`hyperbolic_vertex_coupler` pins."""
    reach = pin_radius + lead_length
    out = []
    for k in range(q):
        d = _unit(rotation + 360 * k / q) * reach
        out.append((_snap(d[0]), _snap(d[1])))
    return out


@gf.cell(tags=["couplers", "capacitors"])
def hyperbolic_vertex_coupler(
    q: int = 3,
    rotation: float = 0.0,
    pin_radius: float = 40.0,
    pin_gap: float = 4.0,
    ground_gap: float = 6.0,
    lead_length: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Return a ``q``-pin capacitive coupler for a hyperbolic lattice vertex.

    A metal disk of radius ``pin_radius`` is cut into ``q`` pie-slice pins by
    radial gaps of width ``pin_gap`` and isolated from ground by a ring of
    width ``ground_gap``. Each pin feeds a short CPW lead, so the resonators
    of the ``q`` lattice edges meeting at the vertex couple through the gaps
    :cite:`kollarHyperbolicLatticesCircuit2019`.

    .. svgbob::

              o2
              │
           .─────.
          ( \ 1 / )
         (   \ /   )
        ( 2   │   0 )── o1
         (    │    )
          `───┴───'
              │
              o3

    Pin ``k`` points along ``rotation + 360 k / q`` degrees. Its port sits at
    the end of the lead, snapped to the 1 nm grid, and faces away from the
    disk centre along the snapped direction (generally not Manhattan).
    The SAX model is :func:`qpdk.models.hyperbolic.hyperbolic_vertex_coupler`.

    Args:
        q: Number of pins (edges meeting at the vertex).
        rotation: Direction of pin 0 in degrees.
        pin_radius: Radius of the metal disk in µm.
        pin_gap: Width of the radial gaps between pins in µm.
        ground_gap: Clearance between the disk and ground in µm.
        lead_length: Length of the CPW lead beyond the disk in µm.
        cross_section: CPW cross-section of the leads.

    Returns:
        Component with ports ``o1`` … ``oq``.

    Raises:
        ValueError: If ``q < 2`` or the pins are too narrow for the leads.
    """
    if q < 2:
        raise ValueError(f"q must be at least 2, got {q}.")
    xs = gf.get_cross_section(cross_section)
    etch = get_etch_section(xs)
    width, gap = xs.width, etch.width
    # Lead must land on its pin: chord across the pin at the rim > lead width.
    pin_chord = 2 * pin_radius * math.sin(math.pi / q) - pin_gap
    if pin_chord <= width:
        raise ValueError(
            f"Pins of a {q}-pin coupler with pin_radius={pin_radius} µm are "
            f"narrower ({pin_chord:.1f} µm) than the lead ({width} µm)."
        )

    c = Component()
    dbu = c.kcl.dbu
    metal = kdb.Region(_disk(pin_radius).to_itype(dbu))
    for k in range(q):
        d = _unit(rotation + 360 * (k + 0.5) / q)
        metal -= kdb.Region(
            _strip(np.zeros(2), d * 2 * pin_radius, pin_gap).to_itype(dbu)
        )
    clearance = kdb.Region(_disk(pin_radius + ground_gap).to_itype(dbu))
    offsets = _vertex_coupler_port_offsets(q, rotation, pin_radius, lead_length)
    for k, (x, y) in enumerate(offsets):
        end = np.array([x, y])
        d = end / np.linalg.norm(end)
        start = d * (pin_radius - width)
        metal += kdb.Region(_strip(start, end, width).to_itype(dbu))
        clearance += kdb.Region(_strip(start, end, width + 2 * gap).to_itype(dbu))
        c.add_port(
            name=f"o{k + 1}",
            center=(x, y),
            orientation=math.degrees(math.atan2(y, x)),
            cross_section=xs,
        )
    metal.merge()
    _insert(c, LAYER.M1_DRAW, metal)
    _insert(c, LAYER.M1_ETCH, (clearance - metal).merged())
    c.info["q"] = q
    return c


def _half_planes(polygon: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Outward normals ``n`` and offsets ``b`` with ``n·x <= b`` inside."""
    area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - np.roll(polygon[:, 0], -1) * polygon[:, 1]
    )
    sign = 1.0 if area > 0 else -1.0
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        d = (b - a) * sign
        n = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        out.append((n, float(n @ a)))
    return out


def _reach(
    planes: list[tuple[np.ndarray, float]],
    origin: np.ndarray,
    direction: np.ndarray,
    margin: float,
) -> float:
    """Distance from ``origin`` along ``direction`` to the shrunk polygon edge.

    Returns:
        The distance in µm, or ``0`` if ``origin`` lies outside the polygon
        shrunk by ``margin``.
    """
    t_max = math.inf
    for n, b in planes:
        slack = b - margin - n @ origin
        if slack < 0:
            return 0.0
        rate = n @ direction
        if rate > 1e-12:
            t_max = min(t_max, slack / rate)
    return t_max


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in radians between directions ``a`` and ``b``."""
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.acos(max(-1.0, min(1.0, cos)))


def _smoothed_length(points: np.ndarray, radius: float) -> float:
    """Length of ``points`` after rounding every corner with a circular arc."""
    seg = np.diff(points, axis=0)
    length = float(np.linalg.norm(seg, axis=1).sum())
    for a, b in pairwise(seg):
        theta = _turn(a, b)
        length -= 2 * radius * math.tan(theta / 2) - radius * theta
    return length


def _meander_points(
    *,
    length: float,
    end: np.ndarray,
    start_dir: np.ndarray,
    end_dir: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
    bend_radius: float,
    pitch: float,
    margin: float,
    end_straight: float,
) -> np.ndarray:
    """Waypoints of a serpentine of smoothed ``length`` from the origin to ``end``.

    The path leaves the origin along ``start_dir``, runs a serpentine with
    legs perpendicular to the axis between the two lead ends and enters
    ``end`` along ``end_dir``. Legs alternate sides and are clipped to the
    convex region ``planes`` shrunk by ``margin``; a common height cap is
    found by bisection so the smoothed path has the requested length.

    Returns:
        Waypoints, shape ``(N, 2)``, for :func:`gdsfactory.path.smooth`.

    Raises:
        ValueError: If the region cannot hold a path of this length.
    """
    r = bend_radius
    p1 = start_dir * end_straight
    q1 = end - end_dir * end_straight
    axis = q1 - p1
    span = float(np.linalg.norm(axis))
    ux = axis / span
    uy = np.array([-ux[1], ux[0]])
    # Room needed between a lead corner and the first leg.
    u_lo = r * math.tan(_turn(start_dir, ux) / 2) + r + 1e-3
    u_hi = span - r * math.tan(_turn(ux, end_dir) / 2) - r - 1e-3
    lead_room = end_straight - r * math.tan(
        max(_turn(start_dir, ux), _turn(ux, end_dir)) / 2
    )

    def polyline(us: Sequence[float], ys: Sequence[float]) -> np.ndarray:
        pts = [np.zeros(2), p1]
        for j, y in enumerate(ys):
            pts.extend((
                p1 + ux * us[j] + uy * (ys[j - 1] if j else 0.0),
                p1 + ux * us[j] + uy * y,
            ))
        if ys:
            pts.extend((p1 + ux * us[len(ys)] + uy * ys[-1], p1 + ux * us[len(ys)]))
        pts += [q1, end]
        return np.array(pts)

    direct = polyline([], [])
    base = _smoothed_length(direct, r)
    if lead_room < 0:
        raise ValueError("end_straight is too short for the turn onto the axis.")
    if math.isclose(base, length, abs_tol=1e-6):
        return direct
    if base > length:
        raise ValueError(
            f"Endpoints are {base:.1f} µm apart along the path; cannot fit "
            f"length={length} µm."
        )

    n_slots = math.floor((u_hi - u_lo) / pitch) + 1 if u_hi > u_lo else 0
    best: tuple[float, list[float], list[float]] | None = None
    for m in range(n_slots - 1, 0, -1):
        offset = u_lo + ((u_hi - u_lo) - m * pitch) / 2
        options = []
        for start in (1.0, -1.0):
            us = [offset + j * pitch for j in range(m + 1)]
            caps = []
            for j in range(m):
                side = start * (-1) ** j
                caps.append(
                    side
                    * min(
                        _reach(planes, p1 + ux * us[j], uy * side, margin),
                        _reach(planes, p1 + ux * us[j + 1], uy * side, margin),
                    )
                )
            if all(abs(y) >= 2 * r + 1e-3 for y in caps):
                options.append((_smoothed_length(polyline(us, caps), r), us, caps))
        if not options:
            continue
        top = max(options, key=operator.itemgetter(0))
        lowest = [math.copysign(2 * r + 1e-3, y) for y in top[2]]
        if _smoothed_length(polyline(top[1], lowest), r) <= length <= top[0]:
            best = top
            break
        if best is None or top[0] > best[0]:
            best = top
    if best is None or best[0] < length:
        room = 0.0 if best is None else best[0]
        raise ValueError(
            f"Region holds at most {room:.0f} µm of meander, need {length} µm; "
            "enlarge the region (e.g. a larger disk_radius) or reduce the pitch."
        )
    _, us, caps = best

    def capped(h: float) -> np.ndarray:
        ys = [math.copysign(min(abs(y), h), y) for y in caps]
        return polyline(us, ys)

    lo, hi = 2 * r + 1e-3, max(abs(y) for y in caps)
    if _smoothed_length(capped(lo), r) > length:
        raise ValueError(
            f"Cannot shorten the meander to length={length} µm with pitch={pitch} µm."
        )
    for _ in range(80):
        mid = (lo + hi) / 2
        if _smoothed_length(capped(mid), r) < length:
            lo = mid
        else:
            hi = mid
    return capped((lo + hi) / 2)


@gf.cell(tags=["resonators"])
def hyperbolic_edge_resonator(
    length: float = 4000.0,
    end: tuple[float, float] = (1000.0, 0.0),
    start_angle: float = 0.0,
    end_angle: float = 0.0,
    region: Sequence[tuple[float, float]] | None = None,
    bend_radius: float = 25.0,
    pitch: float = 60.0,
    margin: float = 25.0,
    end_straight: float = 60.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Return a meandered CPW resonator of exact length between two points.

    The resonator starts at port ``o1`` at the origin travelling along
    ``start_angle`` and ends at port ``o2`` at ``end`` travelling along
    ``end_angle``, so both ends can meet couplers at arbitrary angles. In
    between it meanders with legs perpendicular to the line between its two
    end straights, alternating sides and staying at least ``margin`` from
    the edges of the convex ``region``. The leg height is solved so the
    centreline length equals ``length``; ``info["length"]`` holds the
    extruded length.

    Both ends are left as plain CPW ends: in
    :func:`hyperbolic_lattice` they connect galvanically to coupler pins,
    which set the open-end boundary condition. The SAX model is
    :func:`qpdk.models.hyperbolic.hyperbolic_edge_resonator`.

    Args:
        length: Centreline length in µm.
        end: Position of ``o2`` relative to ``o1`` in µm.
        start_angle: Travel direction at ``o1`` in degrees.
        end_angle: Travel direction at ``o2`` in degrees.
        region: Convex polygon the meander must stay inside, relative to
            ``o1``. Defaults to a square with ``o1`` and ``end`` on a diagonal.
        bend_radius: Radius of the meander bends in µm.
        pitch: Spacing of the meander legs in µm.
        margin: Minimum distance from the centreline to the ``region`` edge in µm.
        end_straight: Straight run at each end before the first bend in µm.
        cross_section: CPW cross-section.

    Returns:
        Component with ports ``o1`` and ``o2``.

    Raises:
        ValueError: If ``region`` cannot hold a resonator of this length.
    """
    if pitch <= 2 * bend_radius:
        raise ValueError(
            f"pitch ({pitch} µm) must exceed twice bend_radius ({bend_radius} µm)."
        )
    end_xy = np.asarray(end, dtype=float)
    if region is None:
        mid = end_xy / 2
        half = np.array([-end_xy[1], end_xy[0]]) / 2
        poly = np.array([[0.0, 0.0], mid + half, end_xy, mid - half])
    else:
        poly = np.asarray(region, dtype=float)
    planes = _half_planes(poly)

    def build(target: float) -> gf.Path:
        pts = _meander_points(
            length=target,
            end=end_xy,
            start_dir=_unit(start_angle),
            end_dir=_unit(end_angle),
            planes=planes,
            bend_radius=bend_radius,
            pitch=pitch,
            margin=margin,
            end_straight=end_straight,
        )
        return gf.path.smooth(pts, radius=bend_radius, bend=_arc)

    # Arcs are polygonised, so correct the analytic length once.
    path = build(length)
    path = build(length + (length - path.length()))

    c = Component()
    ref = c << gf.path.extrude(path, cross_section=cross_section, all_angle=False)
    ref.flatten()
    xs = gf.get_cross_section(cross_section)
    c.add_port(
        name="o1", center=(0.0, 0.0), orientation=start_angle + 180, cross_section=xs
    )
    c.add_port(
        name="o2",
        center=(float(end_xy[0]), float(end_xy[1])),
        orientation=end_angle,
        cross_section=xs,
    )
    c.info["length"] = path.length()
    return c


def _edge_faces(faces: Sequence[Sequence[int]], i: int, j: int) -> list[int]:
    """Indices of the faces that contain the edge between vertices ``i`` and ``j``."""
    out = []
    for f, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            if {face[k], face[(k + 1) % n]} == {i, j}:
                out.append(f)
                break
    return out


@gf.cell(tags=["resonators", "lattices", "chips"])
def hyperbolic_lattice(
    p: int = 7,
    q: int = 3,
    shells: int = 1,
    disk_radius: float = 6000.0,
    resonator_length: float = 7500.0,
    stub_length: float = 400.0,
    probe_pins: Sequence[int] = (11, 0),
    pin_radius: float = 40.0,
    pin_gap: float = 4.0,
    ground_gap: float = 6.0,
    lead_length: float = 20.0,
    bend_radius: float = 25.0,
    pitch: float = 60.0,
    margin: float = 25.0,
    boundary_apex: float = 0.6,
    feed_length: float = 1000.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Return a Kollár-type hyperbolic lattice of CPW edge resonators.

    Every edge of the {p,q} tiling from :func:`qpdk.hyperbolic.hyperbolic_tiling`
    carries a :func:`hyperbolic_edge_resonator` and every vertex a
    :func:`hyperbolic_vertex_coupler`
    :cite:`kollarHyperbolicLatticesCircuit2019`. The line between two pin
    rims is ``resonator_length`` long, so the couplers' leads count towards
    it. In full-wave operation the lattice realises line-graph tight binding
    with :math:`t < 0` and a flat band at :math:`\omega_0 - 2|t|`; see
    :func:`qpdk.models.hyperbolic.hyperbolic_lattice` for the circuit model.

    Free coupler pins on the boundary get a shorted CPW stub of
    ``stub_length`` (a :math:`\lambda/4` resonator far above the band), so
    boundary resonators see the same end loading as interior ones. The free
    pins listed in ``probe_pins`` instead route to launchers, exposed as
    ports ``o1``, ``o2``, … in that order.

    Each resonator meanders inside the quadrilateral of its two vertices and
    the centroids of the two adjacent faces. A boundary edge has only one
    face; its outer apex is that centroid mirrored across the edge and
    pulled in by ``boundary_apex``.

    Args:
        p: Polygon side count of the tiling.
        q: Polygons per vertex, i.e. coupler pins.
        shells: Polygon rings around the central polygon.
        disk_radius: Radius of the Poincaré disk on the chip in µm.
        resonator_length: Pin-rim to pin-rim length of each edge resonator in µm.
        stub_length: Pin-rim to short length of each boundary stub in µm.
        probe_pins: Indices into :attr:`qpdk.hyperbolic.HyperbolicTiling.free_pins`
            routed to launchers.
        pin_radius: Coupler disk radius in µm.
        pin_gap: Coupler radial gap in µm.
        ground_gap: Coupler ground clearance in µm.
        lead_length: Coupler lead length in µm.
        bend_radius: Meander bend radius in µm.
        pitch: Meander leg spacing in µm.
        margin: Minimum distance from a resonator centreline to its region edge in µm.
        boundary_apex: Scale of the mirrored outer apex of boundary edges.
        feed_length: Straight run from a probe pin before turning to its launcher in µm.
        cross_section: CPW cross-section.

    Returns:
        Component with one port per probe pin.

    Raises:
        ValueError: If ``probe_pins`` is invalid or a resonator does not fit.
    """
    tiling = hyperbolic_tiling(p, q, shells)
    free = tiling.free_pins
    probe_pins = tuple(probe_pins)
    if len(set(probe_pins)) != len(probe_pins) or not all(
        0 <= k < len(free) for k in probe_pins
    ):
        raise ValueError(
            f"probe_pins must be distinct indices below {len(free)}, got {probe_pins}."
        )
    xs = gf.get_cross_section(cross_section)

    c = Component()
    centres = [
        np.array([_snap(z.real * disk_radius), _snap(z.imag * disk_radius)])
        for z in tiling.vertices
    ]
    couplers = []
    for v, (rotation, _) in enumerate(tiling.pins):
        ref = c << hyperbolic_vertex_coupler(
            q=q,
            rotation=math.degrees(rotation),
            pin_radius=pin_radius,
            pin_gap=pin_gap,
            ground_gap=ground_gap,
            lead_length=lead_length,
            cross_section=cross_section,
        )
        ref.name = f"c{v}"
        ref.dmove(tuple(centres[v]))
        couplers.append(ref)

    def pin_port(v: int, k: int) -> gf.Port:
        return couplers[v].ports[f"o{k + 1}"]

    edge_length = resonator_length - 2 * lead_length
    for e, (i, j) in enumerate(tiling.edges.tolist()):
        ki = tiling.pins[i][1].index(e)
        kj = tiling.pins[j][1].index(e)
        a, b = pin_port(i, ki), pin_port(j, kj)
        origin = np.array(a.center)
        apexes = [
            np.mean([centres[w] for w in tiling.faces[f]], axis=0)
            for f in _edge_faces(tiling.faces, i, j)
        ]
        if len(apexes) == 1:
            mid = (centres[i] + centres[j]) / 2
            chord = centres[j] - centres[i]
            n = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
            inward = (apexes[0] - mid) @ n
            apexes.append(mid - n * inward * boundary_apex)
        region = [centres[i], apexes[0], centres[j], apexes[1]]
        try:
            resonator = hyperbolic_edge_resonator(
                length=edge_length,
                end=(b.center[0] - origin[0], b.center[1] - origin[1]),
                start_angle=a.orientation,
                end_angle=b.orientation + 180,
                region=tuple(
                    (round(float(x - origin[0]), 3), round(float(y - origin[1]), 3))
                    for x, y in region
                ),
                bend_radius=bend_radius,
                pitch=pitch,
                margin=margin,
                end_straight=2 * bend_radius + 10,
                cross_section=cross_section,
            )
        except ValueError as err:
            raise ValueError(f"Edge {e} between vertices {i} and {j}: {err}") from err
        ref = c << resonator
        ref.name = f"r{e}"
        ref.dmove(tuple(origin))

    probe_index = {free[k]: n for n, k in enumerate(probe_pins)}
    for v, k in free:
        port = pin_port(v, k)
        start = np.array(port.center)
        d = _unit(port.orientation)
        if (v, k) not in probe_index:
            stub = gf.Path(np.array([start, start + d * (stub_length - lead_length)]))
            ref = c << gf.path.extrude(stub, cross_section=cross_section)
            ref.flatten()
            continue
        heading = round(port.orientation / 90) * 90
        m = _unit(heading)
        radius = xs.radius or 100.0
        corner = start + d * feed_length
        tip = corner + m * 3 * radius
        tip = np.array([_snap(tip[0]), _snap(tip[1])])
        lead_in = start + d * 2 * radius
        points = [start, lead_in, tip - m * 2 * radius, tip]
        feed = gf.path.smooth(np.array(points), radius=radius, bend=_arc)
        ref = c << gf.path.extrude(feed, cross_section=cross_section)
        ref.flatten()
        pad = c << launcher(cross_section_small=cross_section)
        pad.name = f"launcher{probe_index[v, k] + 1}"
        pad.drotate(heading + 180)
        pad.dmove((
            float(tip[0] - pad.ports["o1"].center[0]),
            float(tip[1] - pad.ports["o1"].center[1]),
        ))
        c.add_port(name=f"o{probe_index[v, k] + 1}", port=pad.ports["waveport"])

    c.info["num_resonators"] = tiling.num_edges
    c.info["num_couplers"] = tiling.num_vertices
    return c
