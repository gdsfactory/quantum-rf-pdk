"""Background-based 3D previews without changing the simulation PDK."""

import gdsfactory as gf
import pytest
from gdsfactory.technology import DerivedLayer, LayerStack
from gdsfactory.typings import ComponentFactory
from klayout.db import Box, Region

from qpdk import PDK, cells
from qpdk.tech import LAYER, LAYER_STACK, LAYER_STACK_FLIP_CHIP
from qpdk.viewer import PDK as VIEWER_PDK, get_layer_stack


def test_viewer_pdk_activation_keeps_the_simulation_stack() -> None:
    """Both PDKs can be selected without replacing the physical stack."""
    VIEWER_PDK.activate()
    assert gf.get_active_pdk() is VIEWER_PDK
    assert VIEWER_PDK.cells is PDK.cells
    assert VIEWER_PDK.models is PDK.models
    assert VIEWER_PDK.layer_stack is not LAYER_STACK
    PDK.activate()
    assert gf.get_active_pdk() is PDK
    assert PDK.get_layer_stack() is LAYER_STACK


@pytest.mark.parametrize("physical", [LAYER_STACK, LAYER_STACK_FLIP_CHIP])
def test_viewer_stack_keeps_process_values(physical: LayerStack) -> None:
    """Preview edits do not mutate levels or rendering fields in the source stack."""
    original = physical.model_dump()
    viewer = get_layer_stack(physical)
    assert "Vacuum" not in viewer.layers
    for name, level in physical.layers.items():
        if name == "Vacuum":
            continue
        preview = viewer.layers[name]
        assert preview.thickness == level.thickness
        assert preview.zmin == level.zmin
        assert preview.material == level.material
        if name.startswith("Substrate"):
            assert preview.background
    assert physical.model_dump() == original


@pytest.mark.parametrize("metal", ["M1", "M2"])
@pytest.mark.parametrize("with_sim_area", [False, True])
def test_viewer_metal_preserves_additive_priority(
    metal: str, with_sim_area: bool
) -> None:
    """Native background extrusion subtracts etch and restores additive metal."""
    component = gf.Component()
    draw = getattr(LAYER, f"{metal}_DRAW")
    etch = getattr(LAYER, f"{metal}_ETCH")
    drawn = Region(Box(2000, 2000, 12000, 12000))
    etched = Region(Box(0, 0, 8000, 8000))
    component.shapes(draw).insert(drawn)
    component.shapes(etch).insert(etched)
    sim_area = Region(Box(0, 0, 6000, 6000))
    if with_sim_area:
        component.shapes(gf.get_layer(LAYER.SIM_AREA)).insert(sim_area)
    before = component.get_polygons()
    physical = LAYER_STACK_FLIP_CHIP.layers[metal]
    assert isinstance(physical.layer, DerivedLayer)
    expected_physical = sim_area - (etched - drawn) if with_sim_area else Region()
    assert (physical.layer.get_shapes(component) ^ expected_physical).is_empty()

    viewer = get_layer_stack(LAYER_STACK_FLIP_CHIP)
    scene = component.to_3d(layer_stack=viewer)
    metal_meshes = {
        name: mesh
        for name, mesh in scene.geometry.items()
        if name.startswith(f"{metal}_")
    }
    assert metal_meshes
    volume = sum(mesh.volume for mesh in metal_meshes.values())
    expected_preview = Region(component.ibbox()) - (etched - drawn)
    assert volume / physical.thickness == pytest.approx(
        expected_preview.area() * component.kcl.dbu**2
    )
    assert component.get_polygons() == before


@pytest.mark.parametrize("factory", [cells.straight, cells.double_pad_transmon])
def test_ordinary_layout_has_a_3d_preview(factory: ComponentFactory) -> None:
    """Layouts without a simulation area render using the native display stack."""
    component = factory()
    assert gf.get_layer(LAYER.SIM_AREA) not in component.get_polygons()
    scene = component.to_3d(layer_stack=VIEWER_PDK.layer_stack)
    assert any(name.startswith("M1_additive_") for name in scene.geometry)
    assert any(name.startswith("Substrate_") for name in scene.geometry)
