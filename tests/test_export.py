"""Physical 3D export of QPDK's negative-tone masks."""

import gdsfactory as gf
import pytest

import qpdk
from qpdk.cells import quarter_wave_resonator_coupled
from qpdk.tech import LAYER, LAYER_STACK

pytest.importorskip("trimesh")


def test_component_to_3d_without_simulation_area() -> None:
    """The standard gdsfactory method works for an ordinary QPDK cell."""
    component = quarter_wave_resonator_coupled()
    original_layers = component.layers

    scene = component.to_3d()

    assert {"M1_0", "Substrate_0"} <= scene.geometry.keys()
    assert component.layers == original_layers
    assert LAYER.SIM_AREA not in component.layers


def test_to_3d_resonator_without_simulation_area() -> None:
    """A normal locked cell exports metal without modifying its mask."""
    component = quarter_wave_resonator_coupled()
    original_layers = component.layers
    original_area = component.area(LAYER.M1_ETCH)

    scene = qpdk.to_3d(component)

    assert {name.split("_")[0] for name in scene.geometry} == {"M1", "Substrate"}
    assert all(mesh.volume > 0 for mesh in scene.geometry.values())
    assert component.layers == original_layers
    assert component.area(LAYER.M1_ETCH) == original_area
    assert LAYER.SIM_AREA not in component.layers


def test_to_3d_preserves_explicit_area_and_additive_metal() -> None:
    """Etch removes metal, draw restores it, and existing chip bounds win."""
    component = gf.Component()
    component.add_polygon([(0, 0), (20, 0), (20, 20), (0, 20)], LAYER.SIM_AREA)
    component.add_polygon([(4, 4), (12, 4), (12, 12), (4, 12)], LAYER.M1_ETCH)
    component.add_polygon([(6, 6), (8, 6), (8, 8), (6, 8)], LAYER.M1_DRAW)

    scene = qpdk.to_3d(component, margin=100)

    metal_volume = sum(
        mesh.volume for name, mesh in scene.geometry.items() if name.startswith("M1_")
    )
    assert metal_volume == pytest.approx(68)  # (400 - 64 + 4) µm² × 0.2 µm
    assert scene.geometry["Substrate_0"].volume == pytest.approx(200_000)
    assert scene.bounds[0].tolist() == pytest.approx([0, 0, -500])
    assert scene.bounds[1].tolist() == pytest.approx([20, 20, 0.2])


def test_to_3d_uses_flip_chip_stack_for_m2() -> None:
    """M2 geometry exports at its flip-chip height rather than disappearing."""
    component = gf.Component()
    component.add_polygon([(0, 0), (20, 0), (20, 20), (0, 20)], LAYER.M2_ETCH)
    component.add_polygon([(4, 4), (12, 4), (12, 12), (4, 12)], LAYER.M2_DRAW)

    scene = qpdk.to_3d(component, margin=0)

    assert scene.geometry["M2_0"].volume == pytest.approx(12.8)
    assert scene.geometry["M2_0"].bounds[:, 2].tolist() == [10, 10.2]
    assert "Substrate_top_0" in scene.geometry


def test_to_3d_accepts_explicit_stack() -> None:
    """An explicit simulation stack can include the vacuum volume."""
    scene = qpdk.to_3d(quarter_wave_resonator_coupled(), layer_stack=LAYER_STACK)
    assert "Vacuum_0" in scene.geometry


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf")])
def test_to_3d_rejects_invalid_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        qpdk.to_3d(quarter_wave_resonator_coupled(), margin=margin)
