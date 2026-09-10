"""Tests for qpdk.tech module."""

import pytest
from gdsfactory.technology import LayerStack

from qpdk.tech import LAYER_STACK, LAYER_STACK_FLIP_CHIP

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
