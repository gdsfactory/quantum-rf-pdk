"""Tests for the shared COMSOL sheet mesh helpers.

MPh and COMSOL are not installed here, so small fakes of the Java mesh tree
record the sequence calls and report a fixed element count. The assertions stay
on behaviour: for the refinement helper, that the mesh is run once before the
refine feature is added, and that the requested passes and the layout bounding
box end up on the refine feature; for the absolute-size helper, that the sizes
land on the right features in the order a ``Size`` feature needs, that the
generated generator is swept, and that anything else about the sequence is
refused rather than meshed. Bad arguments are refused before COMSOL is touched
in both.
"""

from __future__ import annotations

from typing import Any, override

import pytest

from qpdk import simulation
from qpdk.simulation import pin_absolute_mesh_sizes, refine_metal_plane_mesh
from qpdk.simulation.comsol_capacitance import (
    GROUND_SELECTION,
    LEFT_PAD_SELECTION,
    RIGHT_PAD_SELECTION,
)
from qpdk.simulation.comsol_layout import ComsolBoundingBox, ComsolLayout
from qpdk.simulation.comsol_mesh import (
    DEFAULT_SIZE_TAG,
    FREE_TET_TAG,
    GENERATED_FREE_TET_TAG,
    pin_absolute_mesh_sizes as absolute_mesh_helper,
    refine_metal_plane_mesh as comsol_mesh_helper,
)


class _Node:
    """Stands in for a created mesh feature, recording its properties."""

    def __init__(self) -> None:
        self.properties: dict[str, str] = {}

    def set(self, name: str, value: str) -> None:
        """Record a property assignment."""
        self.properties[name] = value


class _Mesh:
    """Stands in for ``mesh1``: its sequence, run order, and element count."""

    def __init__(self, elements: float = 4321.0) -> None:
        self.elements = elements
        self.created: list[tuple[str, str]] = []
        self.features: dict[str, _Node] = {}
        # Each run records how many features existed when it was called, so the
        # order of the two runs against the generator is visible in the result.
        self.runs: list[int] = []

    def create(self, tag: str, feature_type: str) -> _Node:
        """Record a create call and return the new feature node."""
        self.created.append((tag, feature_type))
        return self.features.setdefault(tag, _Node())

    def run(self) -> None:
        """Record that the mesh was run."""
        self.runs.append(len(self.created))

    def getNumElem(self) -> float:  # ruff: ignore[invalid-function-name]
        """Return the element count COMSOL would report."""
        return self.elements


class _Component:
    """Stands in for ``comp1``, holding its one mesh."""

    def __init__(self, mesh: _Mesh) -> None:
        self._mesh = mesh

    def mesh(self, tag: str) -> _Mesh:
        """Return the mesh for the tag asked for."""
        assert tag == "mesh1"
        return self._mesh


class _FakeJava:
    """The fake ``model.java`` tree."""

    def __init__(self, mesh: _Mesh) -> None:
        self._mesh = mesh

    def component(self, tag: str) -> _Component:
        """Return the fake component for the tag asked for."""
        assert tag == "comp1"
        return _Component(self._mesh)


class _Model:
    """Stands in for an MPh model holding a physics-controlled mesh."""

    def __init__(self, mesh: _Mesh) -> None:
        self.java = _FakeJava(mesh)


class _RejectingModel:
    """A model whose Java tree raises if anything reaches for it."""

    @property
    def java(self) -> Any:
        """Fail the test if the helper touches COMSOL.

        Raises:
            AssertionError: Always, when the Java tree is reached for.
        """
        raise AssertionError("COMSOL was touched before the arguments were validated")


def _layout(bbox: ComsolBoundingBox | None = None) -> ComsolLayout:
    """Return a layout whose bounding box is the CPW's, with no metal.

    Returns:
        The layout.
    """
    return ComsolLayout(
        polygons=(),
        feed_ports=(),
        bbox=bbox or ComsolBoundingBox(xmin=-300.0, ymin=-50.0, xmax=900.0, ymax=50.0),
    )


def test_zero_passes_meshes_at_the_physics_controlled_size():
    """Pass 0 runs the mesh twice with only a generator added between them."""
    mesh = _Mesh()
    elements = refine_metal_plane_mesh(_Model(mesh), _layout(), 0)

    assert mesh.created == [("ftet_refine", "FreeTet")]
    assert mesh.runs == [0, 1]
    assert "ref1" not in mesh.features
    assert elements == 4321


def test_three_passes_set_the_box_from_the_layout_and_the_z_half_height():
    """Three passes refine the longest edges inside the layout's bounding box."""
    mesh = _Mesh()
    refine_metal_plane_mesh(_Model(mesh), _layout(), 3)

    assert mesh.created == [("ftet_refine", "FreeTet"), ("ref1", "Refine")]
    assert mesh.runs == [0, 2]
    assert mesh.features["ref1"].properties == {
        "numrefine": "3",
        "rmethod": "longest",
        "boxcoord": "on",
        "xmin": "-300[um]",
        "xmax": "900[um]",
        "ymin": "-50[um]",
        "ymax": "50[um]",
        "zmin": "-20[um]",
        "zmax": "20[um]",
    }


def test_the_z_half_height_is_a_keyword_only_override():
    """A custom z half-height sets both z bounds."""
    mesh = _Mesh()
    refine_metal_plane_mesh(_Model(mesh), _layout(), 1, z_half_um=2.5)

    refine = mesh.features["ref1"].properties
    assert refine["zmin"] == "-2.5[um]"
    assert refine["zmax"] == "2.5[um]"


def test_a_smaller_refine_box_narrows_the_x_and_y_bounds():
    """A supplied box refines only the device inside the larger layout."""
    mesh = _Mesh()
    box = ComsolBoundingBox(xmin=-500.0, ymin=-250.0, xmax=500.0, ymax=250.0)
    refine_metal_plane_mesh(_Model(mesh), _layout(), 2, refine_box=box)

    refine = mesh.features["ref1"].properties
    assert (refine["xmin"], refine["xmax"]) == ("-500[um]", "500[um]")
    assert (refine["ymin"], refine["ymax"]) == ("-250[um]", "250[um]")
    # The z half-height still splits around the metal plane.
    assert (refine["zmin"], refine["zmax"]) == ("-20[um]", "20[um]")


@pytest.mark.parametrize(
    ("passes", "z_half_um", "match"),
    [
        (-1, 20.0, "passes"),
        (1.5, 20.0, "passes"),
        ("3", 20.0, "passes"),
        (None, 20.0, "passes"),
        (True, 20.0, "passes"),
        (3, 0.0, "z_half_um"),
        (3, -20.0, "z_half_um"),
        (3, float("nan"), "z_half_um"),
        (3, float("inf"), "z_half_um"),
    ],
)
def test_rejects_bad_arguments_before_touching_comsol(
    passes: Any, z_half_um: float, match: str
):
    """Invalid arguments are refused before any Java call is made."""
    with pytest.raises(ValueError, match=match):
        refine_metal_plane_mesh(
            _RejectingModel(), _layout(), passes, z_half_um=z_half_um
        )


def test_the_package_export_resolves_to_the_helper():
    """The lazy package export is the helper, not a re-implementation."""
    assert simulation.refine_metal_plane_mesh is comsol_mesh_helper
    assert "refine_metal_plane_mesh" in simulation.__all__


class _Selection:
    """Stands in for a feature selection, recording the named selection used."""

    def __init__(self) -> None:
        self.named_tags: list[str] = []

    def named(self, tag: str) -> None:
        """Record an assignment to a named selection."""
        self.named_tags.append(tag)


class _MeshFeature:
    """Stands in for a created mesh feature, recording its type and properties."""

    def __init__(self, feature_type: str) -> None:
        self.feature_type = feature_type
        self.properties: dict[str, str] = {}
        self.selection_node = _Selection()

    def set(self, name: str, value: str) -> None:
        """Record a property assignment."""
        self.properties[name] = value

    def selection(self) -> _Selection:
        """Return the feature's selection."""
        return self.selection_node


class _FeatureList:
    """Stands in for ``mesh1.feature()``: the ordered tags of the sequence."""

    def __init__(self, mesh: _SequenceMesh) -> None:
        self._mesh = mesh

    def tags(self) -> list[str]:
        """Return the feature tags in sequence order."""
        return list(self._mesh.order)

    def remove(self, tag: str) -> None:
        """Drop a feature from the sequence."""
        self._mesh.order.remove(tag)
        self._mesh.features.pop(tag, None)


class _SequenceMesh:
    """Stands in for ``mesh1``: a feature sequence with a current feature.

    The first run materialises the ordinary Size and FreeTet features of a
    physics-controlled sequence, and every feature created afterwards lands
    directly after the current one and becomes current itself.
    """

    def __init__(
        self,
        *,
        elements: Any = 623831.0,
        automatic: bool = False,
        generated: tuple[str, ...] = (DEFAULT_SIZE_TAG, GENERATED_FREE_TET_TAG),
    ) -> None:
        self.elements = elements
        self.automatic = automatic
        self.generated = generated
        self.order: list[str] = []
        self.features: dict[str, _MeshFeature] = {}
        # Each run records the sequence as it stood when it was called, so the
        # construction order is visible in the result.
        self.runs: list[tuple[str, ...]] = []
        self._cursor: str | None = None

    def run(self) -> None:
        """Materialise the physics-controlled features, then record the order."""
        if not self.order:
            self.order = list(self.generated)
            self.features = {
                tag: _MeshFeature("FreeTet" if tag.startswith("ftet") else "Size")
                for tag in self.order
            }
            self._cursor = self.order[-1] if self.order else None
        self.runs.append(tuple(self.order))

    def feature(self, tag: str | None = None) -> _FeatureList | _MeshFeature:
        """Return the feature list, or the one feature with this tag."""
        if tag is None:
            return _FeatureList(self)
        return self.features[tag]

    def current(self, tag: str) -> None:
        """Park the cursor on a feature."""
        self._cursor = tag

    def create(self, tag: str, feature_type: str) -> _MeshFeature:
        """Create a feature after the current one, which becomes current."""
        feature = _MeshFeature(feature_type)
        self.features[tag] = feature
        self.order.insert(self._insert_at(), tag)
        self._cursor = tag
        return feature

    def _insert_at(self) -> int:
        """Return the index a new feature lands at."""
        if self._cursor is None:
            return len(self.order)
        return self.order.index(self._cursor) + 1

    def getNumElem(self) -> Any:  # ruff: ignore[invalid-function-name]
        """Return the element count COMSOL would report."""
        return self.elements

    def isAutomatic(self) -> bool:  # ruff: ignore[invalid-function-name]
        """Return whether the sequence is still physics-controlled."""
        return self.automatic


class _MisorderingMesh(_SequenceMesh):
    """A sequence that inserts every new feature right after the default size."""

    def _insert_at(self) -> int:
        """Return the index just after the default size feature."""
        return self.order.index(DEFAULT_SIZE_TAG) + 1


class _UnreadableAutomaticMesh(_SequenceMesh):
    """A sequence whose ``isAutomatic`` getter is not available."""

    @override
    def isAutomatic(self) -> bool:
        """Fail the way a Java getter that is not there would.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("no such method")


def _pin(mesh: Any, **overrides: Any) -> int:
    """Pin the absolute sizes on a model built over this fake mesh.

    Returns:
        The element count the helper returned.
    """
    sizes: dict[str, Any] = {
        "global_hmax_um": 1000.0,
        "global_hmin_um": 2.0,
        "pad_hmax_um": 5.0,
        "pad_hmin_um": 0.5,
        "ground_hmax_um": 10.0,
        "ground_hmin_um": 1.0,
    }
    return pin_absolute_mesh_sizes(_Model(mesh), **(sizes | overrides))


def test_the_sequence_is_left_in_the_order_the_sizes_need():
    """The conductor sizes sit between the default size and the one generator."""
    mesh = _SequenceMesh()
    elements = _pin(mesh)

    assert mesh.order == [
        DEFAULT_SIZE_TAG,
        "size_pad_l",
        "size_pad_r",
        "size_gnd",
        FREE_TET_TAG,
    ]
    assert mesh.runs == [
        (DEFAULT_SIZE_TAG, GENERATED_FREE_TET_TAG),
        (DEFAULT_SIZE_TAG, "size_pad_l", "size_pad_r", "size_gnd", FREE_TET_TAG),
    ]
    assert elements == 623831


def test_the_generated_generator_is_swept_so_one_generator_is_left():
    """COMSOL's own FreeTet is gone and the created one is the only generator."""
    mesh = _SequenceMesh()
    _pin(mesh)

    assert GENERATED_FREE_TET_TAG not in mesh.features
    assert mesh.features[FREE_TET_TAG].feature_type == "FreeTet"


def test_the_default_size_is_pinned_and_made_custom_first():
    """The global sizes go on the default size as absolute µm values."""
    mesh = _SequenceMesh()
    _pin(mesh, global_hmax_um=1000.0, global_hmin_um=2.0)

    assert mesh.features[DEFAULT_SIZE_TAG].properties == {
        "custom": "on",
        "hmax": "1000[um]",
        "hmin": "2[um]",
    }


def test_the_grading_parameters_go_on_the_global_size_only():
    """hgrad, hcurve and hnarrow are set where they were given."""
    mesh = _SequenceMesh()
    _pin(mesh, hgrad=1.4, hcurve=0.5, hnarrow=0.7)

    assert mesh.features[DEFAULT_SIZE_TAG].properties == {
        "custom": "on",
        "hmax": "1000[um]",
        "hmin": "2[um]",
        "hgrad": "1.4",
        "hcurve": "0.5",
        "hnarrow": "0.7",
    }
    assert "hgrad" not in mesh.features["size_pad_l"].properties


def test_each_conductor_gets_its_own_selection_and_size():
    """The pads share one size and the ground its own, on the study's selections."""
    mesh = _SequenceMesh()
    _pin(mesh)

    assert mesh.features["size_pad_l"].selection_node.named_tags == [LEFT_PAD_SELECTION]
    assert mesh.features["size_pad_r"].selection_node.named_tags == [
        RIGHT_PAD_SELECTION
    ]
    assert mesh.features["size_gnd"].selection_node.named_tags == [GROUND_SELECTION]
    for tag, hmax, hmin in (
        ("size_pad_l", "5[um]", "0.5[um]"),
        ("size_pad_r", "5[um]", "0.5[um]"),
        ("size_gnd", "10[um]", "1[um]"),
    ):
        assert mesh.features[tag].properties["custom"] == "on"
        assert mesh.features[tag].properties["hmax"] == hmax
        assert mesh.features[tag].properties["hmin"] == hmin


def test_an_unreadable_automatic_getter_is_not_fatal():
    """A sequence whose isAutomatic cannot be read still gets meshed."""
    mesh = _UnreadableAutomaticMesh()
    assert _pin(mesh) == 623831


def test_a_sequence_left_physics_controlled_is_refused():
    """A sequence that would re-derive the sizing is not meshed."""
    with pytest.raises(RuntimeError, match="still physics-controlled"):
        _pin(_SequenceMesh(automatic=True))


def test_a_sequence_in_another_order_is_refused():
    """Sizes that would not apply to the features after them are refused."""
    with pytest.raises(RuntimeError, match="came out as"):
        _pin(_MisorderingMesh())


def test_a_build_without_a_default_size_feature_is_refused():
    """A physics-controlled build with nothing to edit is refused."""
    with pytest.raises(RuntimeError, match="no default sizing"):
        _pin(_SequenceMesh(generated=(GENERATED_FREE_TET_TAG,)))


@pytest.mark.parametrize(
    "elements",
    [0.0, -1.0, None, "many"],
)
def test_an_unusable_element_count_is_refused(elements: Any):
    """A mesh that reports no usable element count is not returned as one."""
    with pytest.raises(RuntimeError, match=r"element count|elements"):
        _pin(_SequenceMesh(elements=elements))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("global_hmax_um", 0.0),
        ("global_hmin_um", -2.0),
        ("pad_hmax_um", float("nan")),
        ("pad_hmin_um", float("inf")),
        ("ground_hmax_um", "10"),
        ("ground_hmin_um", None),
        ("ground_hmin_um", True),
        ("hgrad", 0.0),
        ("hcurve", -1.0),
        ("hnarrow", float("nan")),
    ],
)
def test_rejects_bad_sizes_before_touching_comsol(name: str, value: Any):
    """An invalid size is refused before any Java call is made."""
    with pytest.raises(ValueError, match=name):
        _pin(_RejectingModel(), **{name: value})


def test_the_package_export_resolves_to_the_absolute_size_helper():
    """The lazy package export is the helper, not a re-implementation."""
    assert simulation.pin_absolute_mesh_sizes is absolute_mesh_helper
    assert "pin_absolute_mesh_sizes" in simulation.__all__
