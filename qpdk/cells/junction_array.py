"""Josephson junction array (junction chain) components."""

from __future__ import annotations

import math
from typing import Literal

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from klayout import db as kdb

from qpdk.models.constants import Φ_0
from qpdk.tech import LAYER

__all__ = ["josephson_junction_array", "josephson_junction_array_compact"]

LayoutStyle = Literal["dolan", "manhattan"]


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> kdb.DPolygon:
    """Axis-aligned rectangle spanning the given x and y ranges."""
    return kdb.DPolygon(kdb.DBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


def _dolan_island(
    island_length: float, island_width: float, *, first: bool, last: bool
) -> list[kdb.DPolygon]:
    """Island of a Dolan chain, centred at the origin, chain along +x.

    End islands are drawn as interior islands. The leads are added separately.

    Returns:
        The island polygons.
    """
    del first, last
    return [
        _rectangle(
            -island_length / 2, island_length / 2, -island_width / 2, island_width / 2
        )
    ]


def _manhattan_island(
    index: int,
    pitch: float,
    island_length: float,
    island_width: float,
    tip_width: float,
    *,
    first: bool,
    last: bool,
) -> list[kdb.DPolygon]:
    """Island of a Manhattan chain, centred at the origin, chain along +x.

    The island is a bar at +45° (even ``index``) or -45° (odd ``index``) to the
    chain axis. Its wide body has length ``island_length`` and width
    ``island_width``. Narrow fingers of width ``tip_width`` run from the body to
    the neighbouring junction and overshoot it by ``tip_width / 2``, so the two
    perpendicular fingers of adjacent islands overlap in a
    ``tip_width × tip_width`` square at the junction position.
    End islands omit the finger that would point away from the chain.

    Returns:
        The island polygons, rotated into the chain frame.
    """
    half_axis = pitch / math.sqrt(2)  # centre to junction, along the bar axis
    polygons = [
        _rectangle(
            -island_length / 2, island_length / 2, -island_width / 2, island_width / 2
        )
    ]
    # In the bar frame the +u finger points towards the next junction for even
    # islands (+45°: towards (+p/2, +p/2)) and the -u finger towards the previous.
    # For odd islands (-45°) the same holds after the rotation.
    if not last:
        polygons.append(
            _rectangle(
                island_length / 2,
                half_axis + tip_width / 2,
                -tip_width / 2,
                tip_width / 2,
            )
        )
    if not first:
        polygons.append(
            _rectangle(
                -half_axis - tip_width / 2,
                -island_length / 2,
                -tip_width / 2,
                tip_width / 2,
            )
        )
    angle = 45.0 if index % 2 == 0 else -45.0
    rotation = kdb.DCplxTrans(1.0, angle, False, 0.0, 0.0)
    return [rotation * polygon for polygon in polygons]


def _junctions_per_row(n_junctions: int, n_rows: int) -> list[int]:
    """Split ``n_junctions`` over ``n_rows`` rows, earlier rows taking the remainder."""
    base, extra = divmod(n_junctions, n_rows)
    return [base + (1 if row < extra else 0) for row in range(n_rows)]


@gf.cell(tags=["junctions", "inductors"])
def josephson_junction_array(
    n_junctions: int = 43,
    junction_area: float = 0.5,
    island_length: float = 0.3,
    island_width: float = 2.0,
    pitch: float = 0.465,
    layout_style: LayoutStyle = "dolan",
    n_folds: int = 0,
    fold_spacing: float = 4.0,
    lead_length: float = 1.0,
    ground_plane_distance: float | None = None,
    critical_current_density: float = 0.09,
    layer_junction: LayerSpec = LAYER.JJ_AREA,
    layer_patch: LayerSpec | None = LAYER.JJ_PATCH,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    r"""Creates a chain of Josephson junctions in series between two ports.

    The array is used as a superinductance, for example in the fluxonium
    :cite:`manucharyan_fluxonium_2009,maslukMicrowaveCharacterizationJosephson2012`,
    or as a junction-chain transmission line. Island :math:`i` (:math:`i = 0, \dots, N`)
    sits at :math:`i p` along the chain and junction :math:`k` sits between islands
    :math:`k` and :math:`k + 1`.

    .. svgbob::

        o1 ┌────┐ ┌─┐ ┌─┐ ┌─┐       ┌─┐ ┌────┐ o2
        ───┤lead├─┤ ├─┤ ├─┤ ├─ ... ─┤ ├─┤lead├───
           └────┘ └─┘ └─┘ └─┘       └─┘ └────┘
                  │<p>│
                  island_length

    Layout styles:
        - ``"dolan"``: collinear islands of ``island_length`` × ``island_width``
          separated by the bridge gap :math:`p - l`. This is the mask of a
          shadow-evaporated bridge chain. The junction overlap,
          ``junction_area / island_width``, comes from the evaporation angle and
          is not drawn.
        - ``"manhattan"``: overlap junctions on a straight chain. Islands are bars
          alternating at ±45° to the chain axis, i.e. along the two evaporation
          axes, so adjacent narrow tips of width :math:`\sqrt{A}` cross at right
          angles and draw the junction overlap :math:`A`.

    With ``n_folds > 0`` the chain zig-zags in ``n_folds + 1`` rows spaced by
    ``fold_spacing``. The junctions are split as evenly as possible over the rows,
    and the island at each row end is joined to the first island of the next row
    by a vertical connector.

    The ``info`` dictionary reports the number of junctions, the total junction
    area :math:`N A`, the area and perimeter of one interior island (as drawn), the
    chain length :math:`N p`, and the estimated inductances
    :math:`L_\text{J} = \Phi_0 / (2 \pi J_\text{c} A)` and
    :math:`L_\text{array} = N L_\text{J}`, which neglect the kinetic and geometric
    inductance of the islands.

    Defaults reproduce the array of :cite:`manucharyan_fluxonium_2009`:
    :math:`N = 43` junctions of :math:`0.25 \times 2\,\mu\text{m}^2` in a
    :math:`20\,\mu\text{m}` chain, with :math:`J_\text{c}` set so that
    :math:`L_\text{array}` matches the reported :math:`E_L = 0.52\,\text{GHz}`.

    Args:
        n_junctions: Number of junctions :math:`N` in the chain.
        junction_area: Area :math:`A` of each junction in µm².
        island_length: Length :math:`l` of each island along the chain (Dolan) or
            of the wide body of each bar (Manhattan) in µm.
        island_width: Width of each island in µm. For Dolan this is also the
            junction width.
        pitch: Island-to-island pitch :math:`p` along the chain in µm.
        layout_style: ``"dolan"`` or ``"manhattan"``.
        n_folds: Number of folds. ``0`` draws a straight chain.
        fold_spacing: Centre-to-centre spacing between rows in µm.
        lead_length: Length of the contact lead at each end in µm.
        ground_plane_distance: If given, draw a ground-plane clearance on
            ``layer_etch`` that extends this far beyond the chain bounding box in µm.
        critical_current_density: Critical current density :math:`J_\text{c}`
            in µA/µm², used only for the ``info`` inductance estimate.
        layer_junction: Layer for the islands and leads.
        layer_patch: Layer for the contact patches on the leads, or ``None``.
        layer_etch: Layer for the optional ground-plane clearance.

    Returns:
        The junction array component with ports ``o1`` and ``o2``.

    Raises:
        ValueError: If the geometry is inconsistent.
    """
    if n_junctions < 1:
        raise ValueError(f"n_junctions must be at least 1, got {n_junctions}")
    if n_folds < 0:
        raise ValueError(f"n_folds must be non-negative, got {n_folds}")
    n_rows = n_folds + 1
    if n_junctions < n_rows:
        raise ValueError(
            f"n_junctions={n_junctions} is smaller than the number of rows {n_rows}"
        )
    if min(junction_area, island_length, island_width, pitch, lead_length) <= 0:
        raise ValueError(
            "junction_area, island_length, island_width, pitch and lead_length "
            "must be positive"
        )

    tip_width = math.sqrt(junction_area)
    match layout_style:
        case "dolan":
            if island_length >= pitch:
                raise ValueError(
                    f"island_length={island_length} must be smaller than pitch={pitch} "
                    "to leave a bridge gap"
                )
            if junction_area / island_width > island_length:
                raise ValueError(
                    f"junction overlap junction_area / island_width = "
                    f"{junction_area / island_width:g} µm exceeds island_length"
                )
        case "manhattan":
            clearance = (math.sqrt(2) * pitch - island_length) / 2
            if clearance <= max(island_width, tip_width) / 2:
                raise ValueError(
                    "Manhattan geometry overlaps: need "
                    "(sqrt(2) * pitch - island_length) / 2 > "
                    "max(island_width, sqrt(junction_area)) / 2, got "
                    f"{clearance:g} µm <= {max(island_width, tip_width) / 2:g} µm"
                )
        case _:
            raise ValueError(f"Unknown layout_style {layout_style!r}")

    def island(index: int, *, first: bool, last: bool) -> list[kdb.DPolygon]:
        if layout_style == "dolan":
            return _dolan_island(island_length, island_width, first=first, last=last)
        return _manhattan_island(
            index, pitch, island_length, island_width, tip_width, first=first, last=last
        )

    # Lateral extent of an island along the chain axis, used to place the leads.
    interior = island(1, first=False, last=False)
    island_half_extent = max(
        abs(point.x)
        for polygon in island(0, first=False, last=False)
        for point in polygon.each_point_hull()
    )
    row_height = max(
        polygon.bbox().height() for polygon in island(0, first=False, last=False)
    )
    if n_folds and fold_spacing <= max(row_height, island_width):
        raise ValueError(
            f"fold_spacing={fold_spacing} must exceed the row height "
            f"{max(row_height, island_width):g} µm"
        )

    polygons: list[kdb.DPolygon] = []
    patches: list[kdb.DPolygon] = []
    island_index = 0
    x_start = 0.0
    port_specs: list[tuple[str, float, float, float]] = []
    connector_width = island_length if layout_style == "dolan" else island_width

    for row, n_row in enumerate(_junctions_per_row(n_junctions, n_rows)):
        direction = 1 if row % 2 == 0 else -1
        y = -row * fold_spacing
        # Odd rows run in -x: mirror about the y axis (mirror about x, then 180°).
        mirror = kdb.DCplxTrans(
            1.0, 0.0 if direction > 0 else 180.0, direction < 0, x_start, y
        )
        for local in range(n_row + 1):
            global_index = island_index + local
            is_first = global_index == 0
            is_last = global_index == n_junctions
            # Row-end islands are drawn in both rows. Only the chain ends omit
            # the outward finger; row turns keep only the finger inside the row.
            first = local == 0
            last = local == n_row
            shift = kdb.DCplxTrans(local * pitch, 0.0)
            polygons.extend(
                mirror * shift * polygon
                for polygon in island(global_index, first=first, last=last)
            )
            if is_first or is_last:
                outward = -1 if is_first else 1
                x_island = local * pitch
                x_outer = x_island + outward * (island_half_extent + lead_length)
                lead = _rectangle(
                    x_island, x_outer, -island_width / 2, island_width / 2
                )
                polygons.append(mirror * lead)
                patch = _rectangle(
                    x_outer - outward * lead_length,
                    x_outer + outward * lead_length / 4,
                    -island_width / 2 - lead_length / 4,
                    island_width / 2 + lead_length / 4,
                )
                patches.append(mirror * patch)
                x_port = x_start + direction * x_outer
                orientation = 0.0 if direction * outward > 0 else 180.0
                port_specs.append(("o1" if is_first else "o2", x_port, y, orientation))
        x_end = x_start + direction * n_row * pitch
        if row < n_rows - 1:
            polygons.append(
                _rectangle(
                    x_end - connector_width / 2,
                    x_end + connector_width / 2,
                    y,
                    y - fold_spacing,
                )
            )
        island_index += n_row
        x_start = x_end

    c = Component()
    dbu = c.kcl.dbu
    region = kdb.Region()
    for polygon in polygons:
        region.insert(polygon.to_itype(dbu))
    region.merge()
    c.add_polygon(region, layer=layer_junction)

    if layer_patch is not None:
        patch_region = kdb.Region()
        for polygon in patches:
            patch_region.insert(polygon.to_itype(dbu))
        patch_region.merge()
        c.add_polygon(patch_region, layer=layer_patch)

    for name, x, y, orientation in sorted(port_specs):
        c.add_port(
            name=name,
            center=(x, y),
            width=island_width,
            orientation=orientation,
            layer=layer_junction,
        )

    if ground_plane_distance is not None:
        if ground_plane_distance <= 0:
            raise ValueError(
                f"ground_plane_distance must be positive, got {ground_plane_distance}"
            )
        bbox = c.dbbox().enlarged(ground_plane_distance, ground_plane_distance)
        c.add_polygon(kdb.DPolygon(bbox), layer=layer_etch)

    interior_region = kdb.Region()
    for polygon in interior:
        interior_region.insert(polygon.to_itype(dbu))
    interior_region.merge()
    island_area = interior_region.area() * dbu**2
    island_perimeter = interior_region.perimeter() * dbu

    critical_current = critical_current_density * 1e-6 * junction_area  # A
    junction_inductance = Φ_0 / (2 * math.pi * critical_current)

    c.info["n_junctions"] = n_junctions
    c.info["layout_style"] = layout_style
    c.info["junction_area"] = junction_area
    c.info["total_junction_area"] = n_junctions * junction_area
    c.info["island_area"] = round(island_area, 6)
    c.info["island_perimeter"] = round(island_perimeter, 6)
    c.info["chain_length"] = round(n_junctions * pitch, 6)
    c.info["critical_current"] = critical_current
    c.info["junction_inductance"] = junction_inductance
    c.info["array_inductance"] = n_junctions * junction_inductance
    return c


@gf.cell(tags=["junctions", "inductors"])
def josephson_junction_array_compact(
    n_junctions: int = 43,
    junction_area: float = 0.5,
    island_length: float = 0.3,
    island_width: float = 2.0,
    pitch: float = 0.465,
    layout_style: LayoutStyle = "dolan",
    n_folds: int = 2,
    fold_spacing: float = 4.0,
    lead_length: float = 1.0,
    ground_plane_distance: float | None = None,
    critical_current_density: float = 0.09,
    layer_junction: LayerSpec = LAYER.JJ_AREA,
    layer_patch: LayerSpec | None = LAYER.JJ_PATCH,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    r"""Creates a folded Josephson junction array that fits a compact bounding box.

    Same as :func:`josephson_junction_array` with ``n_folds`` defaulting to 2, so
    the chain zig-zags in three rows and both ports face away from each other.

    .. svgbob::

        o1 ──┤├┤├┤├┤├┤├─┐
                         │
             ┌─┤├┤├┤├┤├─┘
             │
             └─┤├┤├┤├┤├┤├── o2

    See :cite:`manucharyan_fluxonium_2009,maslukMicrowaveCharacterizationJosephson2012`.

    Args:
        n_junctions: Number of junctions :math:`N` in the chain.
        junction_area: Area :math:`A` of each junction in µm².
        island_length: Length of each island along the chain in µm.
        island_width: Width of each island in µm.
        pitch: Island-to-island pitch along the chain in µm.
        layout_style: ``"dolan"`` or ``"manhattan"``.
        n_folds: Number of folds.
        fold_spacing: Centre-to-centre spacing between rows in µm.
        lead_length: Length of the contact lead at each end in µm.
        ground_plane_distance: Optional ground-plane clearance in µm.
        critical_current_density: Critical current density in µA/µm².
        layer_junction: Layer for the islands and leads.
        layer_patch: Layer for the contact patches on the leads, or ``None``.
        layer_etch: Layer for the optional ground-plane clearance.

    Returns:
        The folded junction array component.
    """
    c = Component()
    array = josephson_junction_array(
        n_junctions=n_junctions,
        junction_area=junction_area,
        island_length=island_length,
        island_width=island_width,
        pitch=pitch,
        layout_style=layout_style,
        n_folds=n_folds,
        fold_spacing=fold_spacing,
        lead_length=lead_length,
        ground_plane_distance=ground_plane_distance,
        critical_current_density=critical_current_density,
        layer_junction=layer_junction,
        layer_patch=layer_patch,
        layer_etch=layer_etch,
    )
    ref = c << array
    c.add_ports(ref.ports)
    c.info.update(array.info.model_dump())
    c.flatten()
    return c
