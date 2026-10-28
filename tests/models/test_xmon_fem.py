"""Four Xmon arms share one FEM capacitance and one grounded junction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from jax.typing import ArrayLike
from sax.models.rf import electrical_open

from qpdk import PDK
from qpdk.models.datasets import Dataset, GridInterpolator
from qpdk.models.xmon import xmon_fem

ARMS = ("top_arm", "right_arm", "bottom_arm", "left_arm")


def _open(f: ArrayLike = 5e9) -> sax.SDict:
    return electrical_open(f=f, n_ports=1)


@pytest.mark.parametrize(
    ("inductance_nh", "junction_capacitance_ff", "z_ref"),
    [(7.0, 0.0, 50.0), (10.0, 2.0, 75.0)],
)
def test_native_capacitance_and_four_arm_response(
    inductance_nh: float, junction_capacitance_ff: float, z_ref: float
) -> None:
    data = Dataset("xmon_palace")
    assert not data.metadata.synthetic
    assert data.metadata.provenance["validation"]["passed"]
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.values.shape == (5, 5, 1, 1)
    capacitance = GridInterpolator(grid)(arm_scale=1.0, gap_width=10.0)[0, 0]
    frequency = jnp.linspace(1e9, 15e9, 101)
    settings = dict(
        inductance_nh=inductance_nh,
        junction_capacitance_ff=junction_capacitance_ff,
        z_ref=z_ref,
    )
    response = jax.jit(PDK.models["xmon_transmon"])(f=frequency, **settings)
    omega = 2 * jnp.pi * frequency
    admittance = 1j * omega * (capacitance + junction_capacitance_ff * 1e-15) + 1 / (
        1j * omega * inductance_nh * 1e-9
    )
    common = 2 / (4 + z_ref * admittance)
    for i, p in enumerate(ARMS):
        for j, q in enumerate(ARMS):
            np.testing.assert_allclose(
                response[p, q], common - (i == j), atol=1e-11, rtol=1e-9
            )
    wave = jnp.stack(
        [jnp.stack([response[p, q] for q in ARMS], axis=-1) for p in ARMS], axis=-2
    )
    np.testing.assert_allclose(
        wave.conj().swapaxes(-1, -2) @ wave,
        jnp.broadcast_to(jnp.eye(4), wave.shape),
        atol=1e-10,
    )
    np.testing.assert_allclose(wave, wave.swapaxes(-1, -2), atol=1e-11)
    opened, _ = sax.circuit(
        {
            "instances": {
                "xmon": "xmon",
                "top": "open",
                "right": "open",
                "bottom": "open",
            },
            "connections": {
                f"xmon,{arm}": f"{name},o1"
                for arm, name in zip(ARMS[:3], ("top", "right", "bottom"), strict=True)
            },
            "ports": {"left_arm": "xmon,left_arm"},
        },
        models={"xmon": xmon_fem, "open": _open},
        ignore_impossible_connections=False,
    )
    reflection = jax.jit(opened)(f=frequency, xmon=settings)["left_arm", "left_arm"]
    np.testing.assert_allclose(
        reflection,
        (1 - z_ref * admittance) / (1 + z_ref * admittance),
        rtol=1e-9,
        atol=1e-11,
    )


def test_geometry_derivative_vmap_and_junction_batch() -> None:
    def response(gap: float) -> jax.Array:
        return jnp.imag(xmon_fem(f=5e9, gap_width=gap)["left_arm", "left_arm"])

    derivative = jax.jit(jax.grad(response))(10.3)
    difference = (response(10.301) - response(10.299)) / 0.002
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, difference, rtol=1e-5)
    assert bool(
        jnp.all(jnp.isfinite(jax.jit(jax.vmap(response))(jnp.asarray([9.5, 10.5]))))
    )
    frequency = jnp.linspace(3e9, 8e9, 17)
    batched = jax.jit(xmon_fem)(
        f=frequency[None, :], inductance_nh=jnp.asarray([7.0, 10.0])[:, None]
    )["left_arm", "left_arm"]
    expected = jax.vmap(
        lambda inductance: xmon_fem(f=frequency, inductance_nh=inductance)[
            "left_arm", "left_arm"
        ]
    )(jnp.asarray([7.0, 10.0]))
    np.testing.assert_allclose(batched, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "settings",
    [
        {"arm_width": (31.0, 20.0, 30.0, 20.0)},
        {"arm_lengths": (160.0, 130.0, 160.0, 120.0)},
        {"arm_lengths": (192.000001, 144.00000075, 192.000001, 144.00000075)},
        {"arm_lengths": (127.999999, 95.99999925, 127.999999, 95.99999925)},
        {"gap_width": 7.0},
        {"inductance_nh": 0.0},
        {"junction_capacitance_ff": -1.0},
        {"z_ref": -1.0},
    ],
)
def test_unsupported_dimensions_and_circuit_values_return_nan(settings: dict) -> None:
    assert all(
        bool(jnp.all(jnp.isnan(value)))
        for value in jax.jit(xmon_fem)(**settings).values()
    )


def test_full_native_grid_remains_a_lossless_reciprocal_network() -> None:
    data = Dataset("xmon_palace")
    grid = data.grid("maxwell_capacitance")
    scale, gap = jnp.meshgrid(*grid.coords, indexing="ij")
    points = jnp.stack((scale.ravel(), gap.ravel()), axis=-1)
    frequency = jnp.linspace(1e9, 15e9, 31)

    def evaluate(point: jax.Array) -> sax.SDict:
        return xmon_fem(
            f=frequency,
            arm_lengths=jnp.asarray([160, 120, 160, 120]) * point[0],
            gap_width=point[1],
        )

    response = jax.jit(jax.vmap(evaluate))(points)
    wave = jnp.stack(
        [jnp.stack([response[p, q] for q in ARMS], axis=-1) for p in ARMS], axis=-2
    )
    assert wave.shape == (25, 31, 4, 4)
    assert bool(jnp.isfinite(wave).all())
    np.testing.assert_allclose(wave, wave.swapaxes(-1, -2), atol=1e-10)
    np.testing.assert_allclose(
        wave.conj().swapaxes(-1, -2) @ wave,
        jnp.broadcast_to(jnp.eye(4), wave.shape),
        atol=1e-10,
    )


def test_uniform_arm_scale_derivative() -> None:
    def response(scale: float) -> jax.Array:
        return jnp.imag(
            xmon_fem(
                f=5e9,
                arm_lengths=jnp.asarray([160.0, 120.0, 160.0, 120.0]) * scale,
            )["left_arm", "left_arm"]
        )

    derivative = jax.jit(jax.grad(response))(1.03)
    difference = (response(1.03001) - response(1.02999)) / 0.00002
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, difference, rtol=1e-5)
