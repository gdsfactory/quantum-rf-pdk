"""Tests for qpdk.tech module."""

import inspect
import json
from collections.abc import Callable

import gdsfactory as gf
import hypothesis.strategies as st
import pytest
from gdsfactory.technology import LayerStack
from hypothesis import given, settings

from qpdk.cells import waveguides as waveguide_cells
from qpdk.models import waveguides as waveguide_models
from qpdk.tech import (
    LAYER_STACK,
    LAYER_STACK_FLIP_CHIP,
    LAYER_STACK_FLIP_CHIP_NO_VACUUM,
    LAYER_STACK_GATEMON,
    LAYER_STACK_GATEMON_NO_VACUUM,
    LAYER_STACK_NO_VACUUM,
    _route_component,
    get_gatemon_layer_stack,
    get_layer_material_properties,
    material_properties,
)

# Expected (thickness, zmin) in µm for each level of LAYER_STACK.
# Written as plain µm numbers (not the ``Xe-9 * 1e6`` idiom used in qpdk/tech.py)
# so a transcription slip in either file shows up as a mismatch.
_EXPECTED_GEOMETRY_UM: dict[str, tuple[float, float]] = {
    "M1": (0.2, 0.0),  # 200 nm Nb film
    "NbTiN": (0.015, 0.0),  # 15 nm
    "Substrate": (500, -500),
    "Vacuum": (500, 0.2),  # 500 microns of vacuum above metal
    "Airbridge": (0.2, 0.3),
    "Airbridge_Via": (0.1, 0.2),
    "JosephsonJunction": (0.07, 0.0),  # 70 nm AlOx/Al film
    "TSV": (500, -500),
    "IndiumBump": (10, 0.2),
}

# Levels unique to the flip-chip stack (the rest are shared with LAYER_STACK).
_FLIP_CHIP_EXTRA_GEOMETRY_UM: dict[str, tuple[float, float]] = {
    "M2": (0.2, 10.0),
    "Substrate_top": (500, 10.2),
}


@pytest.mark.parametrize(
    ("layer_stack", "expected"),
    [
        (LAYER_STACK, _EXPECTED_GEOMETRY_UM),
        (
            LAYER_STACK_FLIP_CHIP,
            {**_EXPECTED_GEOMETRY_UM, **_FLIP_CHIP_EXTRA_GEOMETRY_UM},
        ),
    ],
    ids=["default", "flip_chip"],
)
def test_layer_stack_geometry(
    layer_stack: LayerStack, expected: dict[str, tuple[float, float]]
) -> None:
    """Every layer thickness and zmin matches its expected value in µm.

    Guards against SI-metre vs. µm unit slips (gdsfactory LayerStack thickness
    is in µm), e.g. ``70e-9`` instead of ``70e-9 * 1e6``, as well as dropping
    the ``* 1e6`` factor entirely.
    """
    assert set(layer_stack.layers) == set(expected)
    for name, level in layer_stack.layers.items():
        thickness, zmin = expected[name]
        assert level.thickness == pytest.approx(thickness), name
        assert level.zmin == pytest.approx(zmin), name


@pytest.mark.parametrize(
    "layer_stack",
    [LAYER_STACK, LAYER_STACK_FLIP_CHIP, LAYER_STACK_GATEMON],
    ids=["default", "flip_chip", "gatemon"],
)
def test_every_layer_material_has_properties(layer_stack: LayerStack) -> None:
    """Every level's material resolves to its ``material_properties`` entry."""
    for name, level in layer_stack.layers.items():
        assert level.material is not None, name
        assert (
            get_layer_material_properties(name, layer_stack)
            is material_properties[level.material]
        ), name


# Boolean operation names the GDSFactory+ 3D viewer (kwasm) accepts when it
# deserialises the layer stack. gdsfactory also allows the symbolic aliases
# ("&", "|", "^", "-"), but the viewer rejects the whole stack if any appear.
_VIEWER_BOOLEAN_OPERATIONS = {"and", "or", "xor", "not"}


def _derived_layer_operations(spec: object) -> list[str]:
    if not isinstance(spec, dict):
        return []
    operations = [spec["operation"]] if "operation" in spec else []
    for key in ("layer1", "layer2"):
        operations += _derived_layer_operations(spec.get(key))
    return operations


@pytest.mark.parametrize(
    "layer_stack",
    [
        LAYER_STACK,
        LAYER_STACK_NO_VACUUM,
        LAYER_STACK_FLIP_CHIP,
        LAYER_STACK_FLIP_CHIP_NO_VACUUM,
        LAYER_STACK_GATEMON,
        LAYER_STACK_GATEMON_NO_VACUUM,
    ],
    ids=[
        "default",
        "no_vacuum",
        "flip_chip",
        "flip_chip_no_vacuum",
        "gatemon",
        "gatemon_no_vacuum",
    ],
)
def test_layer_stack_loads_in_3d_viewer(layer_stack: LayerStack) -> None:
    """Derived layers use operation names the GDSFactory+ 3D viewer can parse."""
    dumped = json.loads(json.dumps(layer_stack.model_dump()))
    for name, level in dumped["layers"].items():
        for operation in _derived_layer_operations(level["layer"]):
            assert operation in _VIEWER_BOOLEAN_OPERATIONS, (name, operation)


def test_layer_material_properties_rejects_unknown_material() -> None:
    stack = LAYER_STACK.model_copy(deep=True)
    stack.layers["Substrate"].material = "unobtainium"
    with pytest.raises(KeyError, match="unobtainium"):
        get_layer_material_properties("Substrate", stack)


def test_gatemon_stack_isolates_gate_from_contacts() -> None:
    """HBN separates the two contact films from the gate, with no PEC tube."""
    stack = get_gatemon_layer_stack(hbn_thickness=0.04)
    layers = stack.layers
    assert layers["ContactNb"].thickness == pytest.approx(0.035)
    assert layers["ContactAu"].thickness == pytest.approx(0.01)
    assert layers["TopGate"].thickness == pytest.approx(0.1)
    assert layers["Reconnection"].thickness == pytest.approx(0.1)
    assert layers["ContactNb"].bounds[1] == pytest.approx(layers["ContactAu"].zmin)
    assert layers["ContactAu"].bounds[1] == pytest.approx(layers["hBN"].zmin)
    assert layers["hBN"].thickness == pytest.approx(0.04)
    assert layers["hBN"].bounds[1] == pytest.approx(layers["TopGate"].zmin)
    assert layers["Oxide"].bounds[1] == pytest.approx(layers["M1"].zmin)
    assert layers["Substrate"].bounds[1] == pytest.approx(layers["Oxide"].zmin)
    assert "JosephsonJunction" not in layers
    assert "CNT" not in layers
    assert LAYER_STACK.layers["M1"].thickness == pytest.approx(0.2)
    assert LAYER_STACK_GATEMON.layers["hBN"].thickness == pytest.approx(0.03)


def test_gatemon_stack_rejects_missing_dielectric() -> None:
    with pytest.raises(ValueError, match="positive"):
        get_gatemon_layer_stack(hbn_thickness=0)


@pytest.mark.parametrize(
    "factory",
    [
        waveguide_cells.straight,
        waveguide_cells.straight_open,
        waveguide_cells.straight_double_open,
        waveguide_cells.bend_circular,
        waveguide_cells.bend_s,
        waveguide_cells.straight_all_angle,
        waveguide_cells.bend_euler_all_angle,
        waveguide_cells.bend_circular_all_angle,
        waveguide_models.straight,
        waveguide_models.straight_all_angle,
        waveguide_models.straight_shorted,
        waveguide_models.straight_open,
        waveguide_models.straight_double_open,
        waveguide_models.bend_circular,
        waveguide_models.bend_circular_all_angle,
        waveguide_models.bend_euler,
        waveguide_models.bend_euler_all_angle,
        waveguide_models.bend_s,
    ],
)
def test_waveguide_width_is_a_cross_section_setting(
    factory: Callable[..., object],
) -> None:
    """Keep width out of the public waveguide cell and model APIs."""
    assert "width" not in inspect.signature(factory).parameters


@given(width_units=st.integers(min_value=500, max_value=10_000))
@settings(deadline=None)
def test_routing_width_round_trips_in_cross_section(width_units: int) -> None:
    """Preserve the routing callback width without a cell-level override."""
    width = width_units * 0.002
    component = _route_component(
        "straight",
        length=10,
        width=width,
        cross_section="coplanar_waveguide",
    )
    serialized_settings = component.settings.model_dump()

    assert "width" not in serialized_settings
    rebuilt = gf.get_component("straight", settings=serialized_settings)
    assert all(port.width == pytest.approx(width) for port in rebuilt.ports)
