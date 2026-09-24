"""Tests for the COMSOL sheet-model builder.

MPh and COMSOL are not installed here, so a small fake of the Java tree stands
in for them. The geometry, selection, and material assertions use real
:class:`~qpdk.simulation.comsol_layout.ComsolLayout` input, so the builder's own
arithmetic is checked rather than re-implemented.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from qpdk.simulation.comsol_layout import (
    ComsolBoundingBox,
    ComsolLayout,
    ComsolPolygon,
)
from qpdk.simulation.comsol_sheet import build_comsol_sheet_model

_LAYOUT = ComsolLayout(
    polygons=(
        ComsolPolygon(
            outline=((-10.0, -5.0), (10.0, -5.0), (10.0, 5.0), (-10.0, 5.0)),
            holes=(((-2.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (-2.0, 1.0)),),
        ),
    ),
    feed_ports=(),
    bbox=ComsolBoundingBox(xmin=-10.0, ymin=-5.0, xmax=10.0, ymax=5.0),
)
_EMPTY_LAYOUT = ComsolLayout(
    polygons=(),
    feed_ports=(),
    bbox=ComsolBoundingBox(xmin=-10.0, ymin=-5.0, xmax=10.0, ymax=5.0),
)


class _Node:
    """Stands in for a feature node, recording properties and selections."""

    def __init__(self) -> None:
        self.properties: dict[str, Any] = {}
        self.named_tags: list[str] = []
        self.references: list[str] = []
        self.children: dict[str, _Node] = {}
        self.planes: dict[str, _Group] = {}

    def child(self, tag: str) -> _Node:
        """Return the child filed under ``tag``, creating it if needed."""
        return self.children.setdefault(tag, _Node())

    def selection(self, tag: str | None = None) -> _Node:
        """Return the node's own selection, or a child selection by name."""
        if tag is None:
            return self
        return self.child(f"selection:{tag}")

    def propertyGroup(self, tag: str) -> _Node:  # ruff: ignore[invalid-function-name]
        """Return a material property group by tag."""
        return self.child(f"group:{tag}")

    def geom(self) -> _Group:
        """Return the work plane's own geometry sequence."""
        return self.planes.setdefault("wpgeom", _Group())

    def set(self, *args: Any) -> None:
        """Record ``set(name, value)`` properties and ``set(name)`` references."""
        if len(args) == 1:
            self.references.append(args[0])
        else:
            name, value = args
            self.properties[name] = value

    def named(self, tag: str) -> None:
        """Record an assignment to a named selection."""
        self.named_tags.append(tag)

    def entities(self) -> list[int]:
        """Resolve a domain probe by which side of z = 0 it occupies."""
        if "zmin" not in self.properties:
            return []
        zmin, zmax = (float(self.properties[key]) for key in ("zmin", "zmax"))
        if zmin >= 0:
            return [1]
        if zmax <= 0:
            return [2]
        return [1, 2]


class _Group:
    """Stands in for a COMSOL collection: ``create`` makes children."""

    def __init__(self, factory: Any = _Node) -> None:
        self.created: list[tuple[Any, ...]] = []
        self.children: dict[str, _Node] = {}
        self._factory = factory

    def __call__(self, tag: str | None = None) -> Any:
        """Return this collection, or the child node filed under ``tag``."""
        if tag is None:
            return self
        return self.children.setdefault(tag, self._factory())

    def create(self, tag: str, *args: Any) -> _Node:
        """Record a create call and return the new child node."""
        self.created.append((tag, *args))
        return self(tag)

    def feature(self, tag: str) -> _Node:
        """Return a child feature by tag."""
        return self(tag)


class _Geometry(_Group):
    """Stands in for the geometry sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.unit: str | None = None
        self.runs = 0

    def lengthUnit(self, unit: str) -> None:  # ruff: ignore[invalid-function-name]
        """Record the length unit."""
        self.unit = unit

    def run(self) -> None:
        """Record a geometry run."""
        self.runs += 1


class _Component:
    """Stands in for ``comp1``: its geometry, selections, and materials."""

    def __init__(self) -> None:
        self.geometries = _Group(_Geometry)
        self.selections = _Group()
        self.materials = _Group()

    def geom(self, tag: str | None = None) -> Any:
        """Return the geometry collection, or one geometry by tag."""
        return self.geometries if tag is None else self.geometries(tag)

    def selection(self, tag: str | None = None) -> Any:
        """Return the selection collection, or one selection by tag."""
        return self.selections if tag is None else self.selections(tag)

    def material(self, tag: str | None = None) -> Any:
        """Return the material collection, or one material by tag."""
        return self.materials if tag is None else self.materials(tag)


class _FakeJava:
    """The fake ``model.java`` tree."""

    def __init__(self) -> None:
        self.component = _Group(_Component)
        self.mesh = _Group()
        self.study = _Group()


def _client() -> MagicMock:
    """Return a client whose created model carries the fake Java tree.

    Returns:
        The client double.
    """
    client = MagicMock()
    client.create.return_value.java = _FakeJava()
    return client


def _build(
    client: MagicMock, layout: ComsolLayout = _LAYOUT, **overrides: Any
) -> _Component:
    """Build a sheet model with the fake client and return its component.

    Returns:
        The fake ``comp1``.
    """
    build_comsol_sheet_model(client, layout, "qubit", **overrides)
    return client.create.return_value.java.component("comp1")


def test_blocks_touch_at_the_interface_and_span_the_bbox():
    """Air sits on silicon at z = 0, both exactly the layout's bounding box."""
    component = _build(_client())
    geometry = component.geom("geom1")

    assert geometry.unit == "um"
    assert [args[0] for _, *args in geometry.created] == [
        "Block",
        "Block",
        "WorkPlane",
    ]
    assert [float(value) for value in geometry.feature("air").properties["size"]] == [
        20.0,
        10.0,
        200.0,
    ]
    assert [float(value) for value in geometry.feature("air").properties["pos"]] == [
        -10.0,
        -5.0,
        0.0,
    ]
    assert [float(value) for value in geometry.feature("si").properties["size"]] == [
        20.0,
        10.0,
        200.0,
    ]
    assert [float(value) for value in geometry.feature("si").properties["pos"]] == [
        -10.0,
        -5.0,
        -200.0,
    ]


def test_thicknesses_and_margin_are_configurable():
    """The blocks follow the requested thicknesses and lateral margin."""
    component = _build(
        _client(),
        substrate_thickness_um=50.0,
        air_height_um=30.0,
        lateral_margin_um=5.0,
    )
    geometry = component.geom("geom1")

    assert [float(value) for value in geometry.feature("air").properties["size"]] == [
        30.0,
        20.0,
        30.0,
    ]
    assert [float(value) for value in geometry.feature("si").properties["pos"]] == [
        -15.0,
        -10.0,
        -50.0,
    ]


def test_thin_slabs_keep_domain_probes_on_opposite_sides_of_interface():
    component = _build(_client(), substrate_thickness_um=0.01, air_height_um=0.01)
    assert float(component.selection("air").properties["zmin"]) > 0
    assert float(component.selection("si").properties["zmax"]) < 0


def test_metal_is_a_work_plane_polygon_with_the_hole_subtracted():
    """The sheet keeps the layout's outline and subtracts its holes."""
    component = _build(_client())
    work_plane = component.geom("geom1").feature("wp1").geom()

    assert work_plane.created == [
        ("pol0", "Polygon"),
        ("hole0_0", "Polygon"),
        ("dif0_0", "Difference"),
    ]
    assert work_plane.feature("pol0").properties["x"] == "-10,10,10,-10"
    assert work_plane.feature("pol0").properties["y"] == "-5,-5,5,5"
    assert work_plane.feature("hole0_0").properties["x"] == "-2,-1,-1,-2"
    difference = work_plane.feature("dif0_0")
    assert difference.selection("input").references == ["pol0"]
    assert difference.selection("input2").references == ["hole0_0"]


def test_geometry_is_unioned_and_run_once():
    """The finalization is a union, the sequence runs once, nothing extrudes."""
    geometry = _build(_client()).geom("geom1")

    assert geometry.feature("fin").properties["action"] == "union"
    assert geometry.runs == 1
    assert "Extrude" not in {args[0] for _, *args in geometry.created if args}


def test_dielectric_domains_are_selected_and_materialized():
    """Each domain gets a Box selection and a Common material with epsilon r."""
    component = _build(_client())

    assert component.selections.created == [("air", "Box"), ("si", "Box")]
    air = component.selection("air").properties
    assert air["entitydim"] == "3"
    assert air["condition"] == "intersects"
    assert 0.0 < float(air["zmin"]) < float(air["zmax"])
    for axis in "xy":
        assert float(air[f"{axis}min"]) < 0.0 < float(air[f"{axis}max"])
    silicon = component.selection("si").properties
    assert float(silicon["zmin"]) < float(silicon["zmax"]) < 0.0

    assert component.materials.created == [("matAir", "Common"), ("matSi", "Common")]
    air_material = component.material("matAir")
    silicon_material = component.material("matSi")
    assert air_material.named_tags == ["air"]
    assert silicon_material.named_tags == ["si"]
    assert air_material.propertyGroup("def").properties["relpermittivity"] == "1"
    assert silicon_material.propertyGroup("def").properties["relpermittivity"] == "11.7"
    for material in (air_material, silicon_material):
        properties = material.propertyGroup("def").properties
        assert properties["relpermeability"] == "1"
        assert properties["electricconductivity"] == "0"
    assert "epsilonr" not in silicon_material.propertyGroup("def").properties


def test_builder_adds_no_physics_mesh_or_study():
    """The returned model is geometry, selections, and materials only."""
    client = _client()
    _build(client)
    java = client.create.return_value.java
    model = client.create.return_value

    assert java.mesh.created == []
    assert java.study.created == []
    assert java.component.created == [("comp1",)]
    model.mesh.assert_not_called()
    model.solve.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"substrate_thickness_um": 0.0}, "substrate_thickness_um"),
        ({"substrate_thickness_um": -200.0}, "substrate_thickness_um"),
        ({"substrate_thickness_um": float("nan")}, "substrate_thickness_um"),
        ({"air_height_um": float("inf")}, "air_height_um"),
        ({"lateral_margin_um": -1.0}, "lateral_margin_um"),
        ({"lateral_margin_um": float("nan")}, "lateral_margin_um"),
    ],
)
def test_rejects_bad_thicknesses_before_touching_comsol(
    overrides: dict[str, Any], match: str
):
    """Invalid lengths are refused before the client is used."""
    client: Any = None
    with pytest.raises(ValueError, match=match):
        build_comsol_sheet_model(client, _LAYOUT, "qubit", **overrides)


def test_rejects_layout_without_polygons():
    """A layout with no metal cannot make sheets."""
    with pytest.raises(ValueError, match="no metal polygons"):
        build_comsol_sheet_model(None, _EMPTY_LAYOUT, "qubit")
