"""Pad charging and linearized circuits retain the full extracted matrix."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk import PDK, SAX_MODEL_ALIASES, _get_cell_schematic
from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.models import (
    boxed_transmon_fem,
    double_pad_transmon_capacitance,
    double_pad_transmon_fem_model,
)
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell


def test_grid_and_reference_values() -> None:
    data = Dataset("double_pad_transmon_palace")
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (3, 3, 5, 3, 2, 2)
    check_maxwell(grid.values)
    lookup = GridInterpolator(grid)
    validation = data.metadata.provenance["validation"]
    assert validation["passed"]
    assert validation["admission_status"] == "validated_bare_electrostatic_model"
    assert validation["coverage"]["passed"]
    assert validation["coverage"]["interior_cells_tested"] == 32
    assert len(validation["coverage"]["strict_faces"]) == 8
    assert all(
        axis.validated == (float(coords[0]), float(coords[-1]))
        for axis, coords in zip(grid.axes, grid.coords, strict=True)
    )
    samples = data.metadata.provenance["previous_validation"]["samples"]
    assert len(samples) == 10
    assert {sample["reference_settings"]["order"] for sample in samples} == {3, 4}
    for sample in samples:
        actual = lookup(**sample["point"])
        reference = jnp.asarray(sample["reference_maxwell_F"])
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=1e-17)
        np.testing.assert_allclose(
            actual.sum(axis=-1), reference.sum(axis=-1), rtol=0.02, atol=1e-17
        )


def test_charging_capacitance_and_jax() -> None:
    point = dict(
        pad_width=200.0, pad_height=325.0, pad_gap=15.0, ground_clearance=150.0
    )
    capacitance = GridInterpolator(
        Dataset("double_pad_transmon_palace").grid("maxwell_capacitance")
    )(**point)
    ground = capacitance.sum(axis=-1)
    expected = -capacitance[0, 1] + ground[0] * ground[1] / ground.sum()
    value = jax.jit(double_pad_transmon_capacitance)(**point)
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
    assert 1e-15 < value < 1e-12

    def evaluate(width: float) -> jax.Array:
        return double_pad_transmon_capacitance(**{**point, "pad_width": width})

    assert jnp.isfinite(jax.jit(jax.grad(evaluate))(200.0))
    assert jax.jit(jax.vmap(evaluate))(jnp.asarray([200.0, 300.0])).shape == (2,)
    assert jnp.isnan(evaluate(149.0))


def test_linearized_junction_and_all_traces() -> None:
    inductance, junction_capacitance = 10e-9, 2e-15
    model = double_pad_transmon_fem_model(
        inductance=inductance, junction_capacitance=junction_capacitance
    )
    f = jnp.asarray([1e9, 5e9, 10e9])
    values = jax.jit(model)(f=f, z_ref=75.0)
    ports = ("left_pad", "right_pad")
    assert set(values) == {(a, b) for a in ports for b in ports}
    scattering = jnp.asarray([[values[a, b] for b in ports] for a in ports]).transpose(
        2, 0, 1
    )
    identity = jnp.broadcast_to(jnp.eye(2), scattering.shape)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    capacitance = GridInterpolator(
        Dataset("double_pad_transmon_palace").grid("maxwell_capacitance")
    )(pad_width=250.0, pad_height=400.0, pad_gap=20.0, ground_clearance=200.0)
    omega = 2 * jnp.pi * f
    expected = 1j * omega[:, None, None] * capacitance
    expected += (1 / (1j * omega * inductance) + 1j * omega * junction_capacitance)[
        :, None, None
    ] * jnp.asarray([[1, -1], [-1, 1]])
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scattering, scattering.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        scattering @ scattering.conj().swapaxes(-1, -2), identity, atol=1e-12
    )


@pytest.mark.parametrize(
    ("pad_size", "pad_gap", "clearance", "inductance_nh", "junction_capacitance_ff"),
    [((250.0, 400.0), 15.0, 200.0, 7.0, 0.0), ((200.0, 325.0), 12.5, 150.0, 10.0, 2.0)],
)
def test_boxed_layout_uses_native_matrix(
    pad_size: tuple[float, float],
    pad_gap: float,
    clearance: float,
    inductance_nh: float,
    junction_capacitance_ff: float,
) -> None:
    settings = dict(
        pad_size=pad_size,
        pad_gap=pad_gap,
        bbox_extension=clearance,
        inductance_nh=inductance_nh,
        junction_capacitance_ff=junction_capacitance_ff,
    )
    component = double_pad_transmon_with_bbox(**settings)
    for key, value in settings.items():
        assert component.settings[key] == value
    schematic = _get_cell_schematic(
        "double_pad_transmon_with_bbox", double_pad_transmon_with_bbox
    )
    assert schematic is not None
    assert schematic.info["models"][0]["name"] == "boxed_transmon_fem"
    assert SAX_MODEL_ALIASES["double_pad_transmon_with_bbox"] == "boxed_transmon_fem"
    model = PDK.models["double_pad_transmon_with_bbox"]
    assert model is boxed_transmon_fem
    frequencies = jnp.asarray([1e9, 5e9, 10e9])
    actual = jax.jit(model)(f=frequencies, z_ref=75.0, **settings)
    scattering = jnp.stack(
        [
            jnp.stack([actual[a, b] for b in ("left_pad", "right_pad")], -1)
            for a in ("left_pad", "right_pad")
        ],
        -2,
    )
    identity = jnp.broadcast_to(jnp.eye(2), scattering.shape)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    matrix = GridInterpolator(
        Dataset("double_pad_transmon_palace").grid("maxwell_capacitance")
    )(
        pad_width=pad_size[0],
        pad_height=pad_size[1],
        pad_gap=pad_gap,
        ground_clearance=clearance,
    )
    omega = 2 * jnp.pi * frequencies
    branch = (
        1 / (1j * omega * inductance_nh * 1e-9)
        + 1j * omega * junction_capacitance_ff * 1e-15
    )
    expected = 1j * omega[:, None, None] * matrix + branch[:, None, None] * jnp.asarray([
        [1, -1],
        [-1, 1],
    ])
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)


def test_boxed_model_geometry_and_junction_derivatives() -> None:
    def response(width: float, inductance: float) -> jax.Array:
        return (
            jnp.abs(
                boxed_transmon_fem(
                    f=5e9, pad_size=(width, 325.0), inductance_nh=inductance
                )["left_pad", "right_pad"]
            )
            ** 2
        )

    gradients = jax.jit(jax.grad(response, argnums=(0, 1)))(200.0, 8.0)
    for i, step in enumerate((0.01, 0.001)):
        before, after = [200.0, 8.0], [200.0, 8.0]
        before[i] -= step
        after[i] += step
        expected = (response(*after) - response(*before)) / (2 * step)
        np.testing.assert_allclose(gradients[i], expected, rtol=1e-5, atol=1e-12)
    values = jax.jit(jax.vmap(response, in_axes=(0, None)))(
        jnp.asarray([200.0, 300.0]), 8.0
    )
    assert jnp.all(jnp.isfinite(values))
    assert not jnp.isclose(values[0], values[1])
    assert jnp.isnan(response(149.0, 8.0))
    assert jnp.isnan(response(200.0, -1.0))
