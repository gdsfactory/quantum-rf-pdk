"""Tests for the QPDK HFSS, Q3D and Q2D wrappers over gplugins.ansys.

The generic AEDT logic is tested in gplugins; these cover what QPDK adds: its
default layer stack and materials, the singleton wrappers and
``Q2D.create_2d_from_cross_section``. They need the ``hfss`` extra and are skipped
without it. The PyAEDT stand-ins below are kept in-repo rather than imported from
gplugins' private test helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gdsfactory as gf
import pytest

pytest.importorskip("gplugins.ansys", reason="needs the hfss extra")

from gdsfactory.technology import LayerStack
from numpy.testing import assert_allclose

from qpdk import LAYER, LAYER_STACK
from qpdk.models.cpw import get_cpw_dimensions
from qpdk.simulation import (
    HFSS,
    Q2D,
    Q3D,
    AEDTBase,
    add_materials_to_aedt,
    layer_stack_to_gds_mapping,
    lumped_port_rectangle_from_cpw,
    object_names_to_materials,
)
from qpdk.singleton import SingletonMeta
from qpdk.tech import (
    LAYER_STACK_FLIP_CHIP,
    coplanar_waveguide,
    material_properties,
)

WRAPPERS = (HFSS, Q3D, Q2D)


@dataclass
class MockAEDTObject:
    """Minimal stand-in for a PyAEDT modeler object."""

    name: str


class MockModeler:
    """Minimal stand-in for a PyAEDT modeler."""

    def __init__(self) -> None:
        """Initialize with no objects."""
        self.object_names: list[str] = []
        self._objects: dict[str, MockAEDTObject] = {}

    def __getitem__(self, name: str) -> MockAEDTObject:
        """Return the modeler object with the given name."""
        return self._objects[name]


@dataclass
class MockMaterial:
    """Minimal stand-in for a PyAEDT material."""

    permittivity: float | None = None
    conductivity: float | None = None


class MockMaterials:
    """Minimal stand-in for a PyAEDT materials manager."""

    def __init__(self) -> None:
        """Initialize an empty materials database."""
        self.material_names: list[str] = []

    def exists_material(self, name: str) -> bool:
        """Return whether the material exists in the project."""
        return name in self.material_names

    def add_material(self, name: str) -> MockMaterial:
        """Add a material to the project."""
        self.material_names.append(name)
        return MockMaterial()


class MockQ3dApp:
    """Minimal stand-in for a PyAEDT Q3d application."""

    def __init__(self, object_names: list[str]):
        """Initialize with the names of objects created by the GDS import."""
        self.imported_object_names = list(object_names)
        self.modeler = MockModeler()
        self.materials = MockMaterials()
        self.assigned_materials: dict[str, str] = {}
        self.import_kwargs: dict[str, Any] = {}

    def import_gds_3d(self, **kwargs: Any) -> bool:
        """Simulate a successful 3D GDS import creating the objects."""
        self.import_kwargs = kwargs
        self.modeler.object_names.extend(self.imported_object_names)
        for name in self.imported_object_names:
            self.modeler._objects[name] = MockAEDTObject(name)
        return True

    def assign_material(self, assignment: list, material: str) -> None:
        """Record a material assignment."""
        for obj in assignment:
            self.assigned_materials[str(obj)] = material


@dataclass
class MockRectangle:
    """Minimal stand-in for a PyAEDT rectangle."""

    name: str
    origin: list[float]
    sizes: list[float]
    material: str


class MockQ2dModeler:
    """Minimal stand-in for a PyAEDT 2D modeler."""

    def __init__(self) -> None:
        """Initialize with no rectangles drawn."""
        self.model_units: str | None = None
        self.rectangles: dict[str, MockRectangle] = {}

    def create_rectangle(
        self, name: str, origin: list[float], sizes: list[float], material: str
    ) -> MockRectangle:
        """Draw a rectangle."""
        rectangle = MockRectangle(name, origin, sizes, material)
        self.rectangles[name] = rectangle
        return rectangle


class MockMesh:
    """Minimal stand-in for a PyAEDT mesh manager."""

    def __init__(self) -> None:
        """Initialize with no mesh operations."""
        self.length_meshes: list[dict] = []

    def assign_length_mesh(self, **kwargs: Any) -> None:
        """Record a length-based mesh operation."""
        self.length_meshes.append(kwargs)


class MockQ2dApp:
    """Minimal stand-in for a PyAEDT Q2d application."""

    def __init__(self) -> None:
        """Initialize an empty 2D design."""
        self.modeler = MockQ2dModeler()
        self.materials = MockMaterials()
        self.mesh = MockMesh()
        self.conductors: dict[str, dict] = {}

    def assign_single_conductor(self, name: str, **kwargs: Any) -> None:
        """Record a conductor assignment."""
        self.conductors[name] = kwargs


def test_layer_stack_to_gds_mapping():
    """Test generating GDS mapping from a LayerStack."""
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Without a stack, QPDK's LAYER_STACK is used
    assert layer_stack_to_gds_mapping() == mapping

    # Check that it returns a dictionary
    assert isinstance(mapping, dict)

    # Check a known layer from qpdk
    # For example, layer 1 should be in the mapping
    # The structure is {layer_number: (elevation, thickness)}
    assert 1 in mapping
    assert isinstance(mapping[1], tuple)
    assert len(mapping[1]) == 2

    elevation, thickness = mapping[1]
    assert isinstance(elevation, float)
    assert isinstance(thickness, float)


def test_layer_stack_to_gds_mapping_default_stack_collision_free():
    """Test that the default LAYER_STACK maps every level to a distinct entry.

    pyaedt's ``import_gds_3d`` keys on GDS layer number only, so the Vacuum
    level (which shares ``LAYER.SIM_AREA`` (98, 0) with Substrate) must be
    skipped, and the contiguous Nb spans of Airbridge (10, 0) and
    Airbridge_Via (10, 1) must merge into a single box.
    """
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Vacuum (98, 0) is skipped, so Substrate keeps layer 98 at its own
    # elevation/thickness instead of being overwritten by vacuum values.
    assert mapping[98] == pytest.approx((-500.0, 500.0))

    # Airbridge (0.3-0.5 um) and Airbridge_Via (0.2-0.3 um) merge to one box.
    assert mapping[10] == pytest.approx((0.2, 0.3))

    # Every non-vacuum level is mapped exactly once: M1 (derived from M1_DRAW, 1),
    # NbTiN 25, Substrate 98, Airbridge and Airbridge_Via merged on 10, JJ 20,
    # TSV 31 and indium bumps 30.
    assert set(mapping) == {1, 10, 20, 25, 30, 31, 98}

    # Sheets mode keeps the lowest elevation for colliding levels.
    sheets_mapping = layer_stack_to_gds_mapping(LAYER_STACK, thickness_override=0.0)
    assert sheets_mapping[10] == pytest.approx((0.2, 0.0))
    assert sheets_mapping[98] == pytest.approx((-500.0, 0.0))


def test_layer_stack_to_gds_mapping_flip_chip_raises():
    """Test that the flip-chip stack raises on its Substrate collision.

    ``LAYER_STACK_FLIP_CHIP`` has ``Substrate`` (Si, span [-500, 0] um) and
    ``Substrate_top`` (Si, span [10.2, 510.2] um) sharing GDS layer 98 with a
    10.2 um vacuum gap between the chips. The two disjoint bodies cannot be
    represented by the single (elevation, thickness) box pyaedt's
    ``import_gds_3d`` allows per layer number, and merging would fill the
    inter-chip gap with silicon — so the mapping raises instead of silently
    producing wrong geometry. Flip-chip EM import is not currently supported;
    this test documents that contract.
    """
    with pytest.raises(
        ValueError, match=r"'Substrate' and 'Substrate_top' share GDS layer 98"
    ):
        layer_stack_to_gds_mapping(LAYER_STACK_FLIP_CHIP)


def test_object_names_to_materials():
    """Test mapping imported object names to materials from the layer stack."""
    # Layer 1 -> M1 (Nb, a metal -> pec); layer 98 -> Substrate (Si); layer 31 -> TSV (TiN)
    result = object_names_to_materials(
        ["signal1", "signal25", "signal98", "signal31", "Substrate", "M1_offset"],
        LAYER_STACK,
    )

    # Metals are PEC
    assert result["signal1"] == "pec"
    assert result["signal25"] == "pec"  # NbTiN
    assert result["signal31"] == "pec"
    assert result["M1_offset"] == "pec"
    # Substrate and other dielectrics get their real material, not pec
    assert result["signal98"] == "Si"
    assert result["Substrate"] == "Si"
    assert result["Substrate"] != "pec"

    # Objects that cannot be resolved to a layer level fail closed
    with pytest.raises(ValueError, match="Could not resolve"):
        object_names_to_materials(["signal7", "unknown_object"], LAYER_STACK)


@pytest.mark.usefixtures("isolated_wrapper_cache")
def test_q3d_import_assigns_materials_from_layer_stack(tmp_path):
    """Q3D import must not blanket-assign pec: substrate gets its real material."""
    comp = gf.components.rectangle(size=(10, 10), layer=LAYER.M1_DRAW)

    # Simulate a GDS import that creates the M1 metal and the Substrate objects
    app = MockQ3dApp(["signal1", "signal98"])
    sim = Q3D(app)

    renamed = sim.import_component(comp, gds_path=tmp_path / "comp.gds")

    # Only conductor objects are returned for downstream net assignment
    assert renamed == ["M1"]
    # Metal level becomes PEC, substrate gets silicon from the layer stack
    assert app.assigned_materials == {"M1": "pec", "Substrate": "Si"}
    # The import maps layers through QPDK's stack, not the active PDK's
    assert app.import_kwargs["mapping_layers"] == layer_stack_to_gds_mapping()
    # QPDK's materials are added to the project
    assert set(material_properties) <= set(app.materials.material_names)


@pytest.mark.usefixtures("isolated_wrapper_cache")
@pytest.mark.parametrize("cls", WRAPPERS)
def test_wrappers_default_to_qpdk_stack_and_materials(cls: type):
    """The wrappers use QPDK's layer stack and materials unless told otherwise."""
    sim = cls(object())

    assert sim.layer_stack is LAYER_STACK
    assert sim.material_properties is material_properties
    assert sim.resolve_layer_stack() is LAYER_STACK
    assert isinstance(sim, AEDTBase)


@pytest.mark.usefixtures("isolated_wrapper_cache")
@pytest.mark.parametrize("cls", WRAPPERS)
def test_wrappers_accept_a_custom_stack_and_materials(cls: type):
    """Explicit arguments still override the QPDK defaults."""
    stack = LayerStack(layers={})
    table = {"Nb": {"relative_permittivity": float("inf")}}

    sim = cls(object(), layer_stack=stack, material_properties=table)

    assert sim.layer_stack is stack
    assert sim.material_properties is table


def test_add_materials_to_aedt_adds_qpdk_materials():
    """The module-level helper registers every QPDK material."""

    class App:
        materials = MockMaterials()

    app = App()
    add_materials_to_aedt(app)

    assert app.materials.material_names == list(material_properties)


def test_lumped_port_rectangle_from_cpw_is_exported():
    """The lumped port helper stays importable from ``qpdk.simulation``."""
    result = lumped_port_rectangle_from_cpw(
        center=(10.0, 20.0, 0.0), orientation=0, cpw_gap=6.0, cpw_width=2.0
    )

    assert_allclose(result["origin"], [10.0, 19.0, 0.0])
    assert_allclose(result["sizes"], [6.0, 2.0])


@pytest.mark.usefixtures("isolated_wrapper_cache")
def test_create_2d_from_cross_section_uses_qpdk_levels():
    """A QPDK CPW cross-section is drawn on the Substrate and M1 levels."""
    app = MockQ2dApp()
    width, gap = get_cpw_dimensions(coplanar_waveguide(width=10, gap=6))

    names = Q2D(app).create_2d_from_cross_section(
        coplanar_waveguide(width=10, gap=6), ground_width=30
    )

    rectangles = app.modeler.rectangles
    assert names == {n: n for n in ("signal", "gnd_left", "gnd_right", "substrate")}
    assert rectangles["signal"].origin == [30 + gap, 0, 0]
    # The 0.2 um M1 film is thickened to 2 um for Q2D stability
    assert rectangles["signal"].sizes == [width, 2.0]
    assert rectangles["signal"].material == LAYER_STACK.layers["M1"].material
    assert rectangles["substrate"].material == LAYER_STACK.layers["Substrate"].material
    assert rectangles["substrate"].sizes[1] == LAYER_STACK.layers["Substrate"].thickness
    assert app.conductors["signal"]["conductor_type"] == "SignalLine"


@pytest.mark.usefixtures("isolated_wrapper_cache")
def test_create_2d_from_cross_section_rejects_other_units():
    """Only micrometres are supported, checked before anything is drawn."""
    app = MockQ2dApp()

    with pytest.raises(ValueError, match="units='um'"):
        Q2D(app).create_2d_from_cross_section(coplanar_waveguide(), units="mm")

    assert app.modeler.rectangles == {}


@pytest.mark.parametrize("cls", WRAPPERS)
def test_singleton_refuses_a_different_configuration(cls: type):
    """A later call asking for another stack or materials raises, not returns stale.

    Uses a throwaway subclass, which gets its own singleton slot, so the result
    does not depend on what earlier tests left in the shared cache.
    """
    fresh = type(f"Fresh{cls.__name__}", (cls,), {})
    stack = LayerStack(layers={})
    table = {"Nb": {"relative_permittivity": float("inf")}}
    try:
        sim = fresh(object(), layer_stack=stack, material_properties=table)
        assert sim.layer_stack is stack
        assert sim.material_properties is table

        # The same configuration hands back the same instance
        assert fresh(object(), layer_stack=stack, material_properties=table) is sim

        # Leaving the arguments out asks for QPDK's defaults, which differ
        with pytest.raises(ValueError, match="different layer_stack"):
            fresh(object())
        with pytest.raises(ValueError, match="different material_properties"):
            fresh(object(), layer_stack=stack)
        with pytest.raises(ValueError, match="different layer_stack"):
            fresh(object(), layer_stack=LAYER_STACK_FLIP_CHIP)

        # The refused calls did not touch the instance
        assert sim.layer_stack is stack
        assert sim.material_properties is table
    finally:
        SingletonMeta._instances.pop(fresh, None)


@pytest.mark.parametrize("cls", WRAPPERS)
def test_singleton_with_defaults_refuses_a_custom_stack(cls: type):
    """A default-configured instance refuses a later custom stack."""
    fresh = type(f"Fresh{cls.__name__}", (cls,), {})
    try:
        sim = fresh(object())
        assert fresh(object()) is sim
        assert fresh(object(), layer_stack=LAYER_STACK) is sim
        with pytest.raises(ValueError, match="different layer_stack"):
            fresh(object(), layer_stack=LayerStack(layers={}))
    finally:
        SingletonMeta._instances.pop(fresh, None)


def test_module_helpers_accept_a_custom_material_table():
    """The module-level helpers take the material table a wrapper was built with."""
    table = {"Nb": {"relative_permittivity": float("inf")}, "Si": {}}

    class App:
        materials = MockMaterials()

    app = App()
    add_materials_to_aedt(app, table)
    assert app.materials.material_names == list(table)

    with pytest.raises(ValueError, match="TiN"):
        object_names_to_materials(["signal31"], LAYER_STACK, table)


@pytest.mark.usefixtures("isolated_wrapper_cache")
def test_create_2d_from_cross_section_missing_level_raises_key_error():
    """A stack without the Substrate or M1 level still raises ``KeyError``."""
    app = MockQ2dApp()

    with pytest.raises(KeyError, match="Substrate"):
        Q2D(app).create_2d_from_cross_section(
            coplanar_waveguide(), layer_stack=LayerStack(layers={})
        )
    assert app.modeler.rectangles == {}
