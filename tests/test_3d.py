"""The default process stack supports both rendering and physical extraction."""

import runpy
from io import BytesIO
from math import dist
from pathlib import Path

import gdsfactory as gf
import pytest
import trimesh
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ComponentFactory
from klayout.db import Box, Region

from qpdk import PDK, cells
from qpdk.samples.sample_to_3d import sample_to_3d
from qpdk.simulation.layout import prepare_metal_layout
from qpdk.tech import LAYER, LAYER_STACK, LAYER_STACK_FLIP_CHIP


@pytest.mark.parametrize("metal", ["M1", "M2"])
@pytest.mark.parametrize("with_sim_area", [False, True])
def test_ground_and_additive_metal_are_disjoint(
    metal: str, with_sim_area: bool
) -> None:
    """Rendering and simulation masks both preserve additive priority."""
    component = gf.Component()
    draw = getattr(LAYER, f"{metal}_DRAW")
    etch = getattr(LAYER, f"{metal}_ETCH")
    drawn = Region(Box(2000, 2000, 12000, 12000))
    etched = Region(Box(0, 0, 8000, 8000))
    component.shapes(draw).insert(drawn)
    component.shapes(etch).insert(etched)
    domain = Region(Box(-1000, -1000, 15000, 15000))
    if with_sim_area:
        component.shapes(gf.get_layer(LAYER.SIM_AREA)).insert(domain)
    before = component.get_polygons()

    stack = LAYER_STACK_FLIP_CHIP
    physical = stack.get_component_with_derived_layers(component)
    positive = physical.get_region(draw)
    ground = physical.get_region(etch)
    assert (positive & ground).is_empty()
    expected = domain - (etched - drawn) if with_sim_area else drawn
    assert ((positive | ground) ^ expected).is_empty()

    scene = component.to_3d(layer_stack=stack)
    volume = sum(
        mesh.volume
        for name, mesh in scene.geometry.items()
        if name.startswith(f"{metal}_")
    )
    assert volume / stack.layers[metal].thickness == pytest.approx(
        expected.area() * component.kcl.dbu**2
    )
    assert component.get_polygons() == before


@pytest.mark.parametrize("stack", [LAYER_STACK, LAYER_STACK_FLIP_CHIP])
def test_backgrounds_keep_physical_stack_dimensions(stack: LayerStack) -> None:
    """Rendering backgrounds share the film and substrate process dimensions."""
    for name, level in stack.layers.items():
        if name.endswith("_ground"):
            film = stack.layers[name.removesuffix("_ground")]
            assert not level.background
            assert level.thickness == film.thickness
            assert level.zmin == film.zmin
            assert level.material == film.material
        elif name.startswith("Substrate"):
            assert level.background
    assert (
        not PDK
        .get_layer_views()
        .get_from_tuple(gf.get_layer_tuple(LAYER.SIM_AREA))
        .visible
    )


@pytest.mark.parametrize("factory", [cells.straight, cells.double_pad_transmon])
def test_ordinary_layout_has_a_3d_preview(factory: ComponentFactory) -> None:
    """Ordinary components render through the standard gdsfactory API."""
    component = factory()
    assert gf.get_layer(LAYER.SIM_AREA) not in component.get_polygons()
    scene = component.to_3d()
    assert any(name.startswith("M1_") for name in scene.geometry)
    assert any(name.startswith("Substrate_") for name in scene.geometry)
    assert not any(name.startswith("Vacuum_") for name in scene.geometry)


def test_prepared_solver_geometry_keeps_its_gaps() -> None:
    """3D viewing does not fill the etch gaps of an already inverted mask."""
    component = prepare_metal_layout(cells.straight(), margin_draw=10)
    metal = component.get_region(LAYER.M1_DRAW)
    assert metal.area() < Region(component.ibbox()).area()
    scene = component.to_3d()
    volume = sum(
        mesh.volume for name, mesh in scene.geometry.items() if name.startswith("M1_")
    )
    assert volume / LAYER_STACK.layers["M1"].thickness == pytest.approx(
        metal.area() * component.kcl.dbu**2
    )


def test_standalone_3d_sample_exports_a_complete_scene() -> None:
    """The sample's default-stack mesh survives a glTF viewer round trip."""
    component = sample_to_3d()
    scene = component.to_3d()
    assert any(name.startswith("M1_ground_") for name in scene.geometry)
    assert any(name.startswith("Substrate_") for name in scene.geometry)
    assert not any(name.startswith("Vacuum_") for name in scene.geometry)
    restored = trimesh.load_scene(
        BytesIO(scene.export(file_type="glb")), file_type="glb"
    )
    assert set(restored.geometry) == set(scene.geometry)
    assert restored.bounds == pytest.approx(scene.bounds)


def test_standalone_sample_camera_contains_the_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Micrometre chip coordinates must fit inside the viewer clipping planes."""
    scenes: list[trimesh.Scene] = []

    def capture(scene: trimesh.Scene) -> None:
        scenes.append(scene)

    monkeypatch.setattr(trimesh.Scene, "show", capture)
    sample = Path(__file__).parents[1] / "qpdk/samples/sample_to_3d.py"
    runpy.run_path(str(sample), run_name="__main__")
    (scene,) = scenes
    eye = scene.camera_transform[:3, 3]
    distances = [dist(eye, vertex) for vertex in scene.bounding_box.vertices]
    assert 0 < scene.camera.z_near < min(distances)
    assert scene.camera.z_far > max(distances)
