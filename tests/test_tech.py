"""Tests for qpdk.tech module."""

import inspect
from collections.abc import Callable

import gdsfactory as gf
import hypothesis.strategies as st
import pytest
from gdsfactory.technology import LayerStack
from hypothesis import given, settings

from qpdk.cells import waveguides as waveguide_cells
from qpdk.models import waveguides as waveguide_models
from qpdk.tech import LAYER_STACK, LAYER_STACK_FLIP_CHIP, _route_component

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
