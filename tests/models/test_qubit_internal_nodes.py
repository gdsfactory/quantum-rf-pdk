"""Local coupling models keep qubit electrodes open inside the circuit boundary."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    junction_model,
    sweep,
)
from qpdk.models.datasets.generate import write
from qpdk.models.generic import open as open_circuit


@pytest.fixture
def coupled_qubit(tmp_path: Path) -> Dataset:
    metadata = DatasetMetadata(
        name="local_qubit_algebra",
        description="Synthetic circuit fixture, not FEM evidence.",
        synthetic=True,
        axes=(Axis(name="coupling", unit="F"),),
        terminals=("island", "return", "readout"),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
    )

    def solve(coupling: float) -> dict:
        return {
            "maxwell_capacitance": jnp.asarray([
                [80e-15 + coupling, -30e-15, -coupling],
                [-30e-15, 70e-15, 0],
                [-coupling, 0, 10e-15 + coupling],
            ])
        }

    return write(
        tmp_path, metadata, sweep(metadata, solve, {"coupling": [1e-15, 10e-15]})
    )


@pytest.mark.parametrize("grounded", [False, True])
def test_internal_nodes_match_explicit_open_loads(
    coupled_qubit: Dataset, grounded: bool
) -> None:
    junction = ("island", None if grounded else "return")
    full = junction_model(coupled_qubit, junction=junction, inductance=10e-9)
    reduced = junction_model(
        coupled_qubit,
        junction=junction,
        inductance=10e-9,
        exposed_ports=("readout",),
    )
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": "full", "a": "open", "b": "open"},
            "connections": {"qubit,island": "a,o1", "qubit,return": "b,o1"},
            "ports": {"readout": "qubit,readout"},
        },
        models={"full": full, "open": open_circuit},
    )
    f = jnp.linspace(1e9, 10e9, 1001)
    actual = jax.jit(reduced)(f=f)["readout", "readout"]
    np.testing.assert_allclose(
        actual, circuit(f=f)["readout", "readout"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(jnp.abs(actual), 1, atol=1e-12)
    if grounded:
        omega = 2 * jnp.pi * f
        coupling = 10e-15
        island_y = 1j * omega * (80e-15 - (30e-15) ** 2 / 70e-15 + coupling) + 1 / (
            1j * omega * 10e-9
        )
        readout_y = (
            1j * omega * (10e-15 + coupling) - (1j * omega * coupling) ** 2 / island_y
        )
        expected = (1 - 50 * readout_y) / (1 + 50 * readout_y)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    matched = full(f=f)["readout", "readout"]
    assert float(jnp.max(jnp.abs(actual - matched))) > 0.5

    def response(coupling: float) -> jax.Array:
        return jnp.imag(reduced(f=5e9, coupling=coupling)["readout", "readout"])

    derivative = jax.jit(jax.grad(response))(5e-15)
    step = 1e-19
    numerical = (response(5e-15 + step) - response(5e-15 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, numerical, rtol=1e-6)
    assert jax.jit(jax.vmap(response))(jnp.asarray([3e-15, 7e-15])).shape == (2,)


def test_grounded_junction_and_external_port_order(coupled_qubit: Dataset) -> None:
    model = junction_model(
        coupled_qubit,
        junction=("island", None),
        inductance=10e-9,
        exposed_ports=("readout", "return"),
    )
    result = jax.jit(model)(f=jnp.asarray([2e9, 5e9]))
    matrix = jnp.stack(
        [
            jnp.stack([result[a, b] for b in ("readout", "return")], axis=-1)
            for a in ("readout", "return")
        ],
        axis=-2,
    )
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(
        matrix @ matrix.conj().swapaxes(-1, -2),
        jnp.broadcast_to(jnp.eye(2), matrix.shape),
        atol=1e-12,
    )


@pytest.mark.parametrize("external", [None, ("readout",)])
@pytest.mark.parametrize("z_ref", [0.0, -50.0, jnp.inf, jnp.nan])
def test_invalid_reference_impedance(
    coupled_qubit: Dataset, external: tuple[str, ...] | None, z_ref: float
) -> None:
    model = junction_model(
        coupled_qubit,
        junction=("island", None),
        inductance=10e-9,
        exposed_ports=external,
    )
    result = jax.jit(model)(f=jnp.asarray([2e9, 5e9]), z_ref=z_ref)
    assert all(bool(jnp.isnan(value).all()) for value in result.values())


@pytest.mark.parametrize("ports", [(), ("missing",), ("readout", "readout")])
def test_bad_external_ports(coupled_qubit: Dataset, ports: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="Exposed ports"):
        junction_model(
            coupled_qubit,
            junction=("island", "return"),
            inductance=10e-9,
            exposed_ports=ports,
        )
