"""The boxed cell's FEM adapter keeps the bonded island floating internally."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from kfactory import LayerEnum

from qpdk import PDK
from qpdk.models.datasets import Dataset, GridInterpolator
from qpdk.models.flipmon import flipmon_with_bbox_fem
from qpdk.tech import LAYER


@pytest.mark.parametrize(
    ("inductance_nh", "junction_capacitance_ff", "z_ref"),
    [(7.0, 0.0, 50.0), (10.0, 2.0, 75.0)],
)
def test_native_one_port_and_layout_binding(
    inductance_nh: float, junction_capacitance_ff: float, z_ref: float
) -> None:
    data = Dataset("flipmon_bonded_palace")
    assert not data.metadata.synthetic
    assert data.metadata.provenance["validation"]["passed"]
    capacitance = GridInterpolator(
        data.grid("maxwell_capacitance", bond="physical_bump", allow_missing=False)
    )(chip_gap=10.0, alignment=0.0, inner_radius=60.0, top_radius=110.0)
    frequency = jnp.linspace(0.5e9, 15e9, 151)
    omega = 2 * jnp.pi * frequency
    y = 1j * omega[:, None, None] * capacitance + (
        1 / (1j * omega * inductance_nh * 1e-9)
        + 1j * omega * junction_capacitance_ff * 1e-15
    )[:, None, None] * jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    effective = y[:, 1, 1] - y[:, 1, 0] * y[:, 0, 1] / y[:, 0, 0]
    expected = (1 - z_ref * effective) / (1 + z_ref * effective)
    result = jax.jit(PDK.models["flipmon_with_bbox"])(
        f=frequency,
        inductance_nh=inductance_nh,
        junction_capacitance_ff=junction_capacitance_ff,
        z_ref=z_ref,
    )
    assert set(result) == {("outer_ring_outside", "outer_ring_outside")}
    reflection = result["outer_ring_outside", "outer_ring_outside"]
    np.testing.assert_allclose(reflection, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(jnp.abs(reflection), 1, atol=1e-10)


@pytest.mark.parametrize(
    "settings",
    [
        {"outer_ring_radius": 120.0},
        {"outer_ring_width": 50.0},
        {"m1_etch_extension_gap": 25.0},
        {"m2_etch_extension_gap": 35.0},
        {"chip_gap": 4.0},
        {"alignment": 20.0},
        {"inner_circle_radius": 45.0},
        {"top_circle_radius": 125.0},
        {"inductance_nh": 0.0},
        {"junction_capacitance_ff": -1.0},
        {"z_ref": -1.0},
        {"f": 0.0},
    ],
)
def test_outside_physical_scope_returns_nan(settings: dict[str, float]) -> None:
    assert all(
        bool(jnp.all(jnp.isnan(value)))
        for value in jax.jit(flipmon_with_bbox_fem)(**settings).values()
    )


def test_junction_batch_and_geometry_derivative() -> None:
    def response(gap: float) -> jax.Array:
        return jnp.imag(
            flipmon_with_bbox_fem(f=5e9, chip_gap=gap)[
                "outer_ring_outside", "outer_ring_outside"
            ]
        )

    derivative = jax.jit(jax.grad(response))(10.3)
    finite = (response(10.301) - response(10.299)) / 0.002
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, finite, rtol=1e-5)
    frequency = jnp.linspace(3e9, 8e9, 17)
    inductance = jnp.asarray([7.0, 10.0])
    batch = jax.jit(flipmon_with_bbox_fem)(
        f=frequency[None, :], inductance_nh=inductance[:, None]
    )
    expected = jax.vmap(
        lambda value: flipmon_with_bbox_fem(f=frequency, inductance_nh=value)
    )(inductance)
    for ports in batch:
        np.testing.assert_allclose(
            batch[ports], expected[ports], rtol=1e-10, atol=1e-11
        )


@pytest.mark.parametrize(
    ("setting", "index", "pair", "name"),
    [
        ("layer_metal", int(cast(LayerEnum, LAYER.M1_DRAW)), (1, 0), "M1_DRAW"),
        ("layer_metal_top", int(cast(LayerEnum, LAYER.M2_DRAW)), (2, 0), "M2_DRAW"),
        ("layer_etch", int(cast(LayerEnum, LAYER.M1_ETCH)), (1, 1), "M1_ETCH"),
        ("layer_etch_top", int(cast(LayerEnum, LAYER.M2_ETCH)), (2, 1), "M2_ETCH"),
    ],
)
def test_layer_selectors_preserve_supported_representations(
    setting: str, index: int, pair: tuple[int, int], name: str
) -> None:
    frequency = jnp.asarray([3e9, 5e9, 8e9])
    ports = ("outer_ring_outside", "outer_ring_outside")
    expected = flipmon_with_bbox_fem(f=frequency)[ports]
    for value in (index, pair, list(pair)):
        settings: dict[str, Any] = {setting: value}
        result = jax.jit(flipmon_with_bbox_fem)(f=frequency, **settings)
        np.testing.assert_allclose(result[ports], expected, atol=1e-12)
    named_settings: dict[str, Any] = {setting: name}
    named = jax.jit(lambda f: flipmon_with_bbox_fem(f=f, **named_settings))(frequency)
    np.testing.assert_allclose(named[ports], expected, atol=1e-12)
    for value in (99999, (77, 0), [77, 0], [1, 0, 0]):
        settings = {setting: value}
        result = jax.jit(flipmon_with_bbox_fem)(f=frequency, **settings)
        assert bool(jnp.all(jnp.isnan(result[ports])))
    unsupported_settings: dict[str, Any] = {setting: "unsupported"}
    result = flipmon_with_bbox_fem(f=frequency, **unsupported_settings)
    assert bool(jnp.all(jnp.isnan(result[ports])))


@pytest.mark.parametrize("names", [False, True])
def test_outer_sax_jit_preserves_layer_settings(names: bool) -> None:
    settings: dict[str, Any] = (
        dict(
            layer_metal="M1_DRAW",
            layer_metal_top="M2_DRAW",
            layer_etch="M1_ETCH",
            layer_etch_top="M2_ETCH",
        )
        if names
        else dict(
            layer_metal=[1, 0],
            layer_metal_top=[2, 0],
            layer_etch=[1, 1],
            layer_etch_top=[2, 1],
        )
    )
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": {"component": "guarded", "settings": settings}},
            "connections": {},
            "ports": {"ring": "qubit,outer_ring_outside"},
        },
        models={"guarded": flipmon_with_bbox_fem},
    )
    frequency = jnp.asarray([3e9, 5e9, 8e9])
    response = jax.jit(circuit)(f=frequency)
    expected = flipmon_with_bbox_fem(f=frequency)
    np.testing.assert_allclose(
        response["ring", "ring"],
        expected["outer_ring_outside", "outer_ring_outside"],
        atol=1e-12,
    )
    invalid = jax.jit(circuit)(f=frequency, qubit={"layer_metal": [77, 0]})
    assert bool(jnp.all(jnp.isnan(invalid["ring", "ring"])))


def test_external_jit_requires_numeric_or_static_layer_names() -> None:
    with pytest.raises(TypeError, match="array"):
        jax.jit(flipmon_with_bbox_fem)(f=5e9, layer_metal="M1_DRAW")
