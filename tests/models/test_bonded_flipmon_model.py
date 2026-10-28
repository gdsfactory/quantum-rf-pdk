"""Keep the finite bump's Maxwell matrix when adding the Josephson branch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from jax.typing import ArrayLike
from sax.models.rf import electrical_open

from qpdk.models import flipmon_bonded_model
from qpdk.models.datasets import Dataset, GridInterpolator


def _open(f: ArrayLike = 5e9) -> sax.SDict:
    return electrical_open(f=f, n_ports=1)


@pytest.mark.parametrize(
    ("inductance", "junction_capacitance", "z_ref"),
    [(7e-9, 0.0, 50.0), (10e-9, 2e-15, 75.0)],
)
def test_native_matrix_and_floating_island(
    inductance: float, junction_capacitance: float, z_ref: float
) -> None:
    data = Dataset("flipmon_bonded_palace")
    assert not data.metadata.synthetic
    assert data.metadata.provenance["validation"]["passed"]
    grid = data.grid("maxwell_capacitance", bond="physical_bump", allow_missing=False)
    assert grid.values.shape == (7, 3, 3, 3, 2, 2)
    dimensions = dict(chip_gap=10.0, alignment=0.0, inner_radius=60.0, top_radius=110.0)
    capacitance = GridInterpolator(grid)(**dimensions)
    frequency = jnp.linspace(0.5e9, 15e9, 151)
    omega = 2 * jnp.pi * frequency
    branch = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    admittance = (
        1j * omega[:, None, None] * capacitance
        + (1 / (1j * omega * inductance) + 1j * omega * junction_capacitance)[
            :, None, None
        ]
        * branch
    )
    identity = jnp.eye(2)
    expected = jnp.linalg.solve(
        identity + z_ref * admittance, identity - z_ref * admittance
    )
    model = flipmon_bonded_model(
        inductance=inductance, junction_capacitance=junction_capacitance
    )
    response = jax.jit(model)(f=frequency, z_ref=z_ref, **dimensions)
    ports = ("island", "ring")
    values = jnp.stack(
        [jnp.stack([response[p, q] for q in ports], axis=-1) for p in ports], axis=-2
    )
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(values, values.swapaxes(-1, -2), atol=1e-11)
    np.testing.assert_allclose(
        values.conj().swapaxes(-1, -2) @ values,
        jnp.broadcast_to(identity, values.shape),
        atol=1e-10,
    )
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": "qubit", "open": "open"},
            "connections": {"qubit,island": "open,o1"},
            "ports": {"ring": "qubit,ring"},
        },
        models={"qubit": model, "open": _open},
        ignore_impossible_connections=False,
    )
    opened = jax.jit(circuit)(f=frequency, qubit=dimensions | {"z_ref": z_ref})
    reduced = admittance[:, 1, 1] - (
        admittance[:, 1, 0] * admittance[:, 0, 1] / admittance[:, 0, 0]
    )
    np.testing.assert_allclose(
        opened["ring", "ring"],
        (1 - z_ref * reduced) / (1 + z_ref * reduced),
        rtol=1e-9,
        atol=1e-10,
    )


def test_native_geometry_derivative_and_broadcast() -> None:
    model = flipmon_bonded_model(inductance=7e-9)

    def response(gap: float) -> jax.Array:
        return jnp.imag(model(f=5e9, chip_gap=gap)["ring", "ring"])

    derivative = jax.jit(jax.grad(response))(10.3)
    difference = (response(10.301) - response(10.299)) / 0.002
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, difference, rtol=1e-5)
    assert bool(
        jnp.all(jnp.isfinite(jax.jit(jax.vmap(response))(jnp.asarray([7.0, 12.0]))))
    )
    frequency = jnp.linspace(3e9, 8e9, 17)
    inductance = jnp.asarray([7e-9, 10e-9])
    batched = jax.jit(model)(f=frequency[None, :], inductance=inductance[:, None])
    sequential = jax.vmap(lambda value: model(f=frequency, inductance=value))(
        inductance
    )
    for ports in batched:
        np.testing.assert_allclose(
            batched[ports], sequential[ports], rtol=1e-10, atol=1e-11
        )


@pytest.mark.parametrize("inductance", [0.0, -7e-9])
def test_nonphysical_inductance_is_rejected(inductance: float) -> None:
    with pytest.raises(ValueError, match="positive inductance"):
        flipmon_bonded_model(inductance=inductance)


def test_nonphysical_junction_capacitance_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative junction capacitance"):
        flipmon_bonded_model(inductance=7e-9, junction_capacitance=-1e-15)


@pytest.mark.parametrize("z_ref", [0.0, -50.0, float("inf"), float("nan")])
def test_invalid_reference_impedance_returns_nan(z_ref: float) -> None:
    model = flipmon_bonded_model(inductance=7e-9)
    result = jax.jit(model)(f=jnp.asarray([3e9, 5e9, 8e9]), z_ref=z_ref)
    assert all(bool(jnp.all(jnp.isnan(value))) for value in result.values())


def test_reference_impedance_batch_preserves_valid_rows() -> None:
    model = flipmon_bonded_model(inductance=7e-9)
    frequency = jnp.asarray([3e9, 5e9, 8e9])
    impedances = jnp.asarray([50.0, 0.0, -50.0, jnp.inf, jnp.nan, 75.0])
    result = jax.jit(model)(f=frequency[None, :], z_ref=impedances[:, None])
    valid = jax.vmap(lambda z_ref: model(f=frequency, z_ref=z_ref))(
        jnp.asarray([50.0, 75.0])
    )
    for ports, values in result.items():
        assert bool(jnp.all(jnp.isnan(values[1:5])))
        assert bool(jnp.all(jnp.isfinite(values[jnp.asarray([0, 5])])))
        np.testing.assert_allclose(
            values[jnp.asarray([0, 5])], valid[ports], rtol=1e-10, atol=1e-11
        )
