"""Coupling branches obey the documented zero-value convention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sax import SDictModel

from qpdk.models.generic import lc_resonator_coupled
from qpdk.models.qubit import fluxonium_coupled, transmon_coupled


@pytest.mark.parametrize(
    ("capacitance", "inductance"),
    [(5e-15, 0.0), (0.0, 3e-9), (5e-15, 3e-9)],
)
@pytest.mark.parametrize("grounded", [False, True])
def test_series_coupling_matches_circuit_impedances(
    capacitance: float, inductance: float, grounded: bool
) -> None:
    f = jnp.linspace(1.1e9, 9.3e9, 31)
    omega = 2 * jnp.pi * f
    admittance = 1j * omega * capacitance
    if inductance:
        admittance += 1 / (1j * omega * inductance)
    qubit_admittance = 1j * omega * 100e-15 + 1 / (1j * omega * 7e-9)
    impedance = 1 / admittance + 1 / qubit_admittance
    expected_reflection = impedance / (100 + impedance)
    expected_transmission = 100 / (100 + impedance)
    actual = jax.jit(lc_resonator_coupled, static_argnames=("grounded",))(
        f=f,
        grounded=grounded,
        capacitance=100e-15,
        inductance=7e-9,
        coupling_capacitance=capacitance,
        coupling_inductance=inductance,
    )
    if grounded:
        expected_reflection = (impedance - 50) / (impedance + 50)
        expected_transmission = jnp.zeros_like(f)
    np.testing.assert_allclose(actual["o1", "o1"], expected_reflection, atol=1e-12)
    np.testing.assert_allclose(
        actual["o2", "o2"], -1.0 if grounded else expected_reflection, atol=1e-12
    )
    np.testing.assert_allclose(actual["o1", "o2"], expected_transmission, atol=1e-12)
    np.testing.assert_allclose(actual["o2", "o1"], expected_transmission, atol=1e-12)
    np.testing.assert_allclose(
        jnp.abs(actual["o1", "o1"]) ** 2 + jnp.abs(actual["o2", "o1"]) ** 2,
        1.0,
        atol=1e-12,
    )


def test_no_coupling_is_open() -> None:
    response = lc_resonator_coupled(
        f=jnp.linspace(1e9, 10e9, 101),
        coupling_capacitance=0.0,
        coupling_inductance=0.0,
    )
    np.testing.assert_allclose(response["o1", "o1"], 1.0, atol=1e-12)
    np.testing.assert_allclose(response["o1", "o2"], 0.0, atol=1e-20)


@pytest.mark.parametrize("model", [transmon_coupled, fluxonium_coupled])
def test_qubit_readout_changes_with_capacitive_coupling(model: SDictModel) -> None:
    f = jnp.linspace(2e9, 9e9, 101)
    weak = model(f=f, coupling_capacitance=2e-15)
    strong = model(f=f, coupling_capacitance=50e-15)
    assert float(jnp.max(jnp.abs(weak["o1", "o2"] - strong["o1", "o2"]))) > 0.1


def test_capacitive_coupling_derivative_is_finite() -> None:
    def response(capacitance: float | jax.Array) -> jax.Array:
        return jnp.imag(
            lc_resonator_coupled(f=5e9, coupling_capacitance=capacitance)["o1", "o2"]
        )

    derivative = jax.jit(jax.grad(response))(10e-15)
    assert jnp.isfinite(derivative)
    assert abs(float(derivative)) > 1e10
    step = 1e-19
    expected = (response(10e-15 + step) - response(10e-15 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, expected, rtol=1e-7)


def test_disabled_inductor_has_finite_derivative() -> None:
    def response(inductance: float | jax.Array) -> jax.Array:
        return jnp.imag(
            lc_resonator_coupled(f=5e9, coupling_inductance=inductance)["o1", "o2"]
        )

    assert jnp.isfinite(jax.jit(jax.grad(response))(0.0))
