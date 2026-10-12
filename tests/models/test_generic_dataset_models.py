"""N-port dataset models retain port ordering and JAX transformations."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from qpdk.models.datasets import (
    Axis,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    capacitance_model,
    s_parameters_model,
    sweep,
)
from qpdk.models.datasets.generate import write


@pytest.mark.parametrize("ports", [1, 3, 5])
def test_lumped_capacitance_ports(ports: int, tmp_path: Path) -> None:
    terminals = tuple(f"p{i}" for i in range(ports))
    metadata = DatasetMetadata(
        name="algebra",
        synthetic=True,
        axes=(Axis(name="width", unit="um"),),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=terminals,
    )

    def solve(width: float) -> dict:
        return {"maxwell_capacitance": jnp.eye(ports) * width * 1e-15}

    data = write(tmp_path, metadata, sweep(metadata, solve, {"width": [2.0, 10.0]}))
    model = capacitance_model(data)
    result = jax.jit(model)(f=jnp.array([1e9, 5e9]), width=6.0)
    s = jnp.stack(
        [jnp.stack([result[a, b] for b in terminals], axis=-1) for a in terminals],
        axis=-2,
    )
    normalized = 2j * jnp.pi * jnp.array([1e9, 5e9]) * 50 * 6e-15
    np.testing.assert_allclose(
        jnp.diagonal(s, axis1=-2, axis2=-1),
        jnp.broadcast_to(((1 - normalized) / (1 + normalized))[:, None], (2, ports)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        s @ jnp.swapaxes(s.conj(), -1, -2),
        jnp.broadcast_to(jnp.eye(ports), s.shape),
        atol=1e-12,
    )

    def value(width: float) -> jax.Array:
        return jnp.imag(model(f=5e9, width=width)[terminals[0], terminals[0]])

    assert jnp.isfinite(jax.jit(jax.grad(value))(6.0))
    assert jax.jit(jax.vmap(value))(jnp.array([3.0, 6.0])).shape == (2,)
    assert jnp.isnan(model(width=11.0)[terminals[0], terminals[0]])
    circuit, _ = sax.circuit(
        {
            "instances": {
                "device": {"component": "lookup", "settings": {"width": 6.0}}
            },
            "connections": {},
            "ports": {name: f"device,{name}" for name in terminals},
        },
        models={"lookup": model},
    )
    composed = jax.jit(circuit)(f=jnp.array([1e9, 5e9]))
    for key in result:
        np.testing.assert_allclose(composed[key], result[key], atol=1e-12)


def test_stored_three_port_scattering(tmp_path: Path) -> None:
    terminals = ("left", "right", "monitor")
    metadata = DatasetMetadata(
        name="stored_scattering",
        synthetic=True,
        axes=(Axis(name="frequency", unit="Hz"), Axis(name="width", unit="um")),
        quantities=(
            Quantity(
                name="s_parameters",
                kind=QuantityKind.S_PARAMETERS,
                unit="1",
                complex=True,
            ),
        ),
        terminals=terminals,
    )

    def solve(frequency: float, width: float) -> dict:
        return {
            "s_parameters": jnp.eye(3, dtype=complex)
            * (frequency / 1e10 + 1j * width / 100)
            + jnp.arange(9).reshape(3, 3) * 0.01
        }

    data = write(
        tmp_path,
        metadata,
        sweep(metadata, solve, {"frequency": [1e9, 5e9], "width": [2.0, 10.0]}),
    )
    model = s_parameters_model(data)
    result = jax.jit(model)(f=jnp.array([2e9, 4e9]), width=6.0)
    assert set(result) == {(a, b) for a in terminals for b in terminals}
    np.testing.assert_allclose(
        result["monitor", "monitor"],
        jnp.array([0.28 + 0.06j, 0.48 + 0.06j]),
        atol=1e-12,
    )
    np.testing.assert_allclose(result["left", "monitor"], 0.02, atol=1e-12)
    np.testing.assert_allclose(result["monitor", "left"], 0.06, atol=1e-12)
    assert result["left", "monitor"].shape == (2,)
    circuit, _ = sax.circuit(
        {
            "instances": {
                "device": {"component": "lookup", "settings": {"width": 6.0}}
            },
            "connections": {},
            "ports": {name: f"device,{name}" for name in terminals},
        },
        models={"lookup": model},
    )
    composed = jax.jit(circuit)(f=jnp.array([2e9, 4e9]))
    for key in result:
        np.testing.assert_allclose(composed[key], result[key], atol=1e-12)
    assert float(
        jax.jit(
            jax.grad(lambda width: jnp.imag(model(f=2e9, width=width)["left", "left"]))
        )(6.0)
    ) == pytest.approx(0.01)
    assert jnp.isnan(model(f=6e9, width=6.0)["left", "left"])
    with pytest.raises(ValueError, match="frequency axis"):
        s_parameters_model(data, frequency_axis="f")
    with pytest.raises(ValueError, match="Maxwell"):
        capacitance_model(data, quantity="s_parameters")


def test_mutual_branches_and_unequal_self_capacitances(tmp_path: Path) -> None:
    terminals = ("a", "b", "c")
    metadata = DatasetMetadata(
        name="three_terminal_capacitor",
        synthetic=True,
        axes=(Axis(name="width", unit="um"),),
        terminals=terminals,
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
    )
    capacitance = (
        jnp.array([[30.0, -4.0, -2.0], [-4.0, 40.0, -6.0], [-2.0, -6.0, 50.0]]) * 1e-15
    )
    data = write(
        tmp_path,
        metadata,
        sweep(
            metadata,
            lambda **_: {"maxwell_capacitance": capacitance},
            {"width": [2.0, 10.0]},
        ),
    )
    model = capacitance_model(data)
    result = jax.jit(model)(f=5e9, z_ref=75.0, width=6.0)
    scattering = jnp.array([[result[a, b] for b in terminals] for a in terminals])
    identity = jnp.eye(3)
    admittance = jnp.linalg.solve(identity + scattering, identity - scattering) / 75.0
    np.testing.assert_allclose(admittance, 2j * jnp.pi * 5e9 * capacitance, atol=1e-15)
    np.testing.assert_allclose(scattering, scattering.T, atol=1e-12)
    np.testing.assert_allclose(scattering @ scattering.conj().T, identity, atol=1e-12)
    dc = model(f=0.0, width=6.0)
    np.testing.assert_array_equal(
        jnp.array([[dc[a, b] for b in terminals] for a in terminals]), identity
    )
