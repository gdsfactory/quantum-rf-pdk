"""Numerical acceptance and independent study points for component recipes."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import (
    Axis,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    junction_model,
)
from qpdk.models.datasets.capacitance import differential_capacitance
from qpdk.models.datasets.generate import sweep, write


def test_differential_reduction_floats_common_mode_and_other_islands() -> None:
    # Unequal island-ground branches are in series across a floating qubit.
    matrix = jnp.array([[8.0, -2.0], [-2.0, 5.0]]) * 1e-15
    value = jax.jit(differential_capacitance)(matrix)
    expected = (2 + 6 * 3 / (6 + 3)) * 1e-15
    assert float(value) == pytest.approx(expected, rel=1e-12, abs=0)
    assert float(
        jax.grad(lambda scale: differential_capacitance(matrix * scale))(1.0)
    ) == pytest.approx(expected, rel=1e-12, abs=0)
    np.testing.assert_allclose(
        jax.vmap(differential_capacitance)(jnp.stack([matrix, 2 * matrix])),
        [expected, 2 * expected],
        atol=0,
    )


@pytest.mark.parametrize(("positive", "negative"), [(0, 0), (0, 3), (7, 1), (-1, 1)])
def test_differential_rejects_invalid_terminal_indices(
    positive: int, negative: int
) -> None:
    with pytest.raises(ValueError, match="two distinct terminal indices"):
        jax.jit(differential_capacitance, static_argnums=(1, 2))(
            jnp.eye(3) * 1e-15, positive, negative
        )


def test_junction_model_bonds_electrodes_without_dropping_ground(
    tmp_path: Path,
) -> None:

    matrix = (
        jnp.array([[8.0, -2.0, -1.0], [-2.0, 7.0, -3.0], [-1.0, -3.0, 6.0]]) * 1e-15
    )
    metadata = DatasetMetadata(
        name="bonded-algebra",
        synthetic=True,
        axes=(Axis(name="gap", unit="um"),),
        terminals=("inner", "ring", "top"),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
    )
    data = write(
        tmp_path,
        metadata,
        sweep(
            metadata,
            lambda gap: {"maxwell_capacitance": matrix * gap},
            {"gap": [1.0, 2.0]},
        ),
    )
    model = junction_model(
        data,
        junction=("inner", "ring"),
        inductance=1e-8,
        junction_capacitance=2e-15,
        terminal_groups=(("inner", "top"), ("ring",)),
    )
    result = jax.jit(model)(f=5e9, gap=1.5)
    scattering = jnp.array([
        [result[a, b] for b in ("inner", "ring")] for a in ("inner", "ring")
    ])
    recovered = jnp.linalg.solve(jnp.eye(2) + scattering, jnp.eye(2) - scattering) / 50
    omega = 2 * jnp.pi * 5e9
    # After bonding, diagonal 00 keeps both ground branches and their mutual term cancels.
    expected_c = jnp.array([[12.0, -5.0], [-5.0, 7.0]]) * 1.5e-15
    expected_y = 1j * omega * expected_c + (
        1j * omega * 2e-15 + 1 / (1j * omega * 1e-8)
    ) * jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(recovered, expected_y, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scattering @ scattering.conj().T, jnp.eye(2), atol=1e-12)
    assert jnp.isfinite(
        jax.grad(lambda gap: jnp.imag(model(f=5e9, gap=gap)["inner", "inner"]))(1.5)
    )
    assert jnp.isnan(model(gap=3.0)["inner", "inner"])
    unbonded = junction_model(data, junction=("inner", "ring"), inductance=1e-8)
    values = jax.jit(unbonded)(f=5e9, gap=1.5)
    assert len(values) == 9
    names = ("inner", "ring", "top")
    scattering = jnp.array([[values[a, b] for b in names] for a in names])
    admittance = jnp.linalg.solve(jnp.eye(3) + scattering, jnp.eye(3) - scattering) / 50
    difference = jnp.array([1.0, -1.0, 0.0])
    expected = 1.5j * omega * matrix + jnp.outer(difference, difference) / (
        1j * omega * 1e-8
    )
    np.testing.assert_allclose(admittance, expected, rtol=1e-10, atol=1e-15)
    with pytest.raises(ValueError, match="partition"):
        junction_model(
            data,
            junction=("inner", "ring"),
            inductance=1e-8,
            terminal_groups=(("inner", "top"), ("ring", "top")),
        )
    with pytest.raises(ValueError, match="partition"):
        junction_model(
            data, junction=("inner", "ring"), inductance=1e-8, terminal_groups=()
        )
    for junction in (("inner", "inner"), ("inner", "missing"), ("top", "ring")):
        with pytest.raises(ValueError, match="distinct grouped ports"):
            junction_model(
                data,
                junction=junction,
                inductance=1e-8,
                terminal_groups=(("inner", "top"), ("ring",)),
            )


@pytest.mark.parametrize(
    ("inductance", "capacitance"), [(0.0, 0.0), (-1e-8, 0.0), (1e-8, -1e-15)]
)
def test_junction_rejects_nonphysical_inputs(
    inductance: float, capacitance: float
) -> None:
    with pytest.raises(ValueError, match="positive inductance"):
        junction_model(
            "unused",
            junction=("a", "b"),
            inductance=inductance,
            junction_capacitance=capacitance,
        )


def test_junction_rejects_scattering_data(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        name="scattering-algebra",
        synthetic=True,
        axes=(Axis(name="gap", unit="um"),),
        terminals=("a", "b"),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.S_PARAMETERS,
                unit="1",
                complex=True,
            ),
        ),
    )
    data = write(
        tmp_path,
        metadata,
        sweep(
            metadata,
            lambda **_point: {"maxwell_capacitance": jnp.eye(2, dtype=complex)},
            {"gap": [1.0]},
        ),
    )
    with pytest.raises(ValueError, match="real Maxwell"):
        junction_model(data, junction=("a", "b"), inductance=1e-8)
