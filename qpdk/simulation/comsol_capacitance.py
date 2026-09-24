r"""Add a qubit capacitance study to a COMSOL model that uses metal sheets.

The model comes from
:func:`~qpdk.simulation.comsol_sheet.build_comsol_sheet_model`, where the metal
is a set of faces on the silicon/air interface. This module picks three of those
faces out with points inside them, drives the left pad with a voltage terminal,
grounds the right pad and the ground plane, and adds a mesh sequence and a
stationary study. Nothing is solved here.

A solve gives electrostatic capacitance: ``es.C11`` is the capacitance of the
left pad to the grounded rest of the chip, and the stored energy agrees with it,
since :math:`2 W_e / V^2` equals ``es.C11``. It is not an RF Josephson
eigenmode, and turning it into a qubit frequency needs an :math:`L_J` picked
outside COMSOL: that LC estimate is not an eigensolve.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from shapely.geometry import Point as ShapelyPoint, Polygon

from qpdk.simulation.comsol import _format_number
from qpdk.simulation.comsol_sheet import SILICON_SELECTION

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol_layout import ComsolLayout, Point

#: Face selection tags, one per conductor, matching the sheet model's plane.
LEFT_PAD_SELECTION = "pad_l"
RIGHT_PAD_SELECTION = "pad_r"
GROUND_SELECTION = "gnd"

#: Half-size in µm of the boxes that pick one metal face out of the plane. The
#: metal lives on a single plane, so the box straddles ``z = 0``.
_FACE_HALF_SIZE_UM = 0.01


def _require_point(point: Point, name: str) -> None:
    """Raise unless ``point`` is a finite (x, y) pair in µm.

    Raises:
        ValueError: If the point is not a finite (x, y) pair.
    """
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must be a finite (x, y) pair in µm, got {point!r}")


def _add_face_selection(component: Any, tag: str, point: Point) -> tuple[int, ...]:
    """Add a Box selection over the metal face around ``point`` and read it back.

    Returns:
        The entities the selection resolved to, as COMSOL reports them.
    """
    component.selection().create(tag, "Box")
    selection = component.selection(tag)
    selection.set("entitydim", "2")
    selection.set("condition", "intersects")
    selection.set("zmin", _format_number(-_FACE_HALF_SIZE_UM))
    selection.set("zmax", _format_number(_FACE_HALF_SIZE_UM))
    for axis, value in zip("xy", point):
        selection.set(f"{axis}min", _format_number(value - _FACE_HALF_SIZE_UM))
        selection.set(f"{axis}max", _format_number(value + _FACE_HALF_SIZE_UM))
    return tuple(selection.entities())


def _select_metal_faces(
    component: Any, layout: ComsolLayout, conductors: tuple[tuple[str, Point], ...]
) -> None:
    """Add one face selection per conductor and check what they resolved to.

    Every box has to hold exactly one face, and no two conductors the same one,
    or a point that misses the metal would put a terminal or a ground on the
    wrong face and the solve would return a capacitance for the wrong geometry.

    Raises:
        ValueError: If a selection holds a number of faces other than one, or if
            two selections hold the same face.
    """
    metal_regions = [
        Polygon(polygon.outline, polygon.holes) for polygon in layout.polygons
    ]
    for tag, point in conductors:
        if sum(region.contains(ShapelyPoint(point)) for region in metal_regions) != 1:
            raise ValueError(
                f"the {tag} point must lie inside exactly one metal polygon"
            )

    faces = {
        tag: _add_face_selection(component, tag, point) for tag, point in conductors
    }
    for tag, entities in faces.items():
        if len(entities) != 1:
            raise ValueError(
                f"the {tag} point selects {len(entities)} faces, expected one: "
                "pass a point well inside the metal face it names"
            )
    if len(set(faces.values())) != len(faces):
        raise ValueError(
            f"the conductor points must select three different faces, got {faces}"
        )


def _add_electrostatics(component: Any, voltage_v: float) -> None:
    """Add Electrostatics with a voltage terminal and two grounded faces.

    The interface covers every domain, so the default Free Space feature keeps
    the air, and a Charge Conservation feature on silicon uses that domain's
    material permittivity through the default ``epsilonr_mat``.
    """
    physics = component.physics().create("es", "Electrostatics", "geom1")
    physics.create("ccSi", "ChargeConservation", 3)
    physics.feature("ccSi").selection().named(SILICON_SELECTION)

    physics.create("term1", "Terminal", 2)
    terminal = physics.feature("term1")
    terminal.selection().named(LEFT_PAD_SELECTION)
    # The default TerminalType is Charge, which would leave the pad floating.
    terminal.set("TerminalType", "Voltage")
    terminal.set("V0", f"{_format_number(voltage_v)}[V]")

    for tag, selection in (
        ("gnd1", RIGHT_PAD_SELECTION),
        ("gnd2", GROUND_SELECTION),
    ):
        physics.create(tag, "Ground", 2)
        physics.feature(tag).selection().named(selection)


def add_qubit_capacitance_study(
    model: mph.Model,
    layout: ComsolLayout,
    *,
    left_pad_point: Point,
    right_pad_point: Point,
    ground_point: Point,
    voltage_v: float = 1.0,
    mesh_size: int = 7,
) -> mph.Model:
    """Add an unsolved electrostatic capacitance study to a sheet model.

    Args:
        model: The MPh model from
            :func:`~qpdk.simulation.comsol_sheet.build_comsol_sheet_model`.
        layout: The metal polygons used to build that model.
        left_pad_point: Point in µm on the sheet plane, inside the pad to drive.
            It centres a small box, so it has to sit clear of the pad's edges.
        right_pad_point: Point in µm inside the pad to ground.
        ground_point: Point in µm inside the ground plane.
        voltage_v: Terminal voltage in V, positive.
        mesh_size: COMSOL mesh size, an integer from 1 (finest) to 9 (coarsest).

    Returns:
        The same model, with the selections, physics, mesh, and study added and
        nothing solved. Once the caller runs them, ``es.C11`` agrees with
        ``2*es.intWe/voltage_v**2``.

    Raises:
        ValueError: If the voltage is not positive and finite, if the mesh size
            is not an integer in 1 to 9, if a point is not a finite (x, y) pair,
            or if a point does not select exactly one distinct metal face.
    """
    if not math.isfinite(voltage_v) or voltage_v <= 0.0:
        raise ValueError(f"voltage_v must be positive and finite, got {voltage_v!r}")
    if not isinstance(mesh_size, int) or not 1 <= mesh_size <= 9:
        raise ValueError(
            "mesh_size must be an integer from 1 (finest) to 9 (coarsest), "
            f"got {mesh_size!r}"
        )
    _require_point(left_pad_point, "left_pad_point")
    _require_point(right_pad_point, "right_pad_point")
    _require_point(ground_point, "ground_point")

    component = model.java.component("comp1")
    _select_metal_faces(
        component,
        layout,
        (
            (LEFT_PAD_SELECTION, left_pad_point),
            (RIGHT_PAD_SELECTION, right_pad_point),
            (GROUND_SELECTION, ground_point),
        ),
    )
    _add_electrostatics(component, voltage_v)

    mesh = component.mesh().create("mesh1", "geom1")
    mesh.autoMeshSize(mesh_size)
    study = model.java.study().create("std1")
    study.create("stat", "Stationary")

    return model
