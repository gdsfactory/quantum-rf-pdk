"""Stored RF networks must not invent loss when their junction ports are closed."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    s_parameters_model,
    sweep,
)
from qpdk.models.datasets.generate import write

PORTS = ("readout", "left", "right")


def _admittance(
    frequency: jax.Array | float, coupling_ff: float, conductance: float = 0.0
) -> jax.Array:
    """Two shorted 3 mm arms and a separate capacitive readout pin."""
    ω = 2 * jnp.pi * jnp.asarray(frequency)
    arm = -1j / (50 * jnp.tan(ω * 0.003 / 1.2e8))
    coupling = 1j * ω * coupling_ff * 1e-15
    zero = jnp.zeros_like(ω)
    return jnp.stack(
        [
            jnp.stack([conductance + 1j * ω * 5e-15 + coupling, -coupling, zero], -1),
            jnp.stack([-coupling, arm + coupling, zero], -1),
            jnp.stack([zero, zero, arm], -1),
        ],
        -2,
    )


def _scattering(admittance: jax.Array) -> jax.Array:
    identity = jnp.eye(admittance.shape[-1])
    return jnp.linalg.solve(identity + 50 * admittance, identity - 50 * admittance)


@pytest.fixture
def data(tmp_path: Path) -> Dataset:
    metadata = DatasetMetadata(
        name="synthetic_distributed_qubit",
        description="Analytical circuit control, not FEM admission evidence.",
        synthetic=True,
        axes=(Axis(name="frequency", unit="Hz"), Axis(name="coupling_ff", unit="fF")),
        quantities=(
            Quantity(
                name="s_parameters",
                kind=QuantityKind.S_PARAMETERS,
                unit="1",
                complex=True,
            ),
        ),
        terminals=PORTS,
        provenance={"reference_impedance_ohm": 50.0},
    )

    def solve(frequency: float, coupling_ff: float) -> dict:
        return {"s_parameters": _scattering(_admittance(frequency, coupling_ff))}

    return write(
        tmp_path,
        metadata,
        sweep(
            metadata,
            solve,
            {
                "frequency": jnp.linspace(0.5e9, 15e9, 161).tolist(),
                "coupling_ff": [1.0, 2.0],
            },
        ),
    )


def _matrix(response: dict) -> jax.Array:
    return jnp.stack(
        [jnp.stack([response[a, b] for b in PORTS], -1) for a in PORTS], -2
    )


def _close_junction(scattering: jax.Array, frequency: jax.Array) -> jax.Array:
    """Connect a passive 7 nH junction with 5 fF across the two arm ports."""
    ω = 2 * jnp.pi * frequency
    branch = 1 / (1j * ω * 7e-9) + 1j * ω * 5e-15
    junction = _scattering(branch[..., None, None] * jnp.array([[1, -1], [-1, 1]]))
    inward = jnp.linalg.solve(
        jnp.eye(2) - junction @ scattering[..., 1:, 1:],
        junction @ scattering[..., 1:, :1],
    )
    return scattering[..., 0, 0] + (scattering[..., :1, 1:] @ inward)[..., 0, 0]


def test_closed_high_q_qubit_does_not_acquire_artificial_loss(data: Dataset) -> None:
    """Complex S interpolation falsely absorbs power in a lossless circuit."""
    frequency = jnp.linspace(11.0637e9, 11.0678e9, 4001)
    admittance = s_parameters_model(data, interpolation="admittance")
    scattering = s_parameters_model(data)
    corrected = _close_junction(
        _matrix(jax.jit(admittance)(f=frequency, coupling_ff=1.0)), frequency
    )
    default = _close_junction(
        _matrix(scattering(f=frequency, coupling_ff=1.0)), frequency
    )
    exact = _close_junction(_scattering(_admittance(frequency, 1.0)), frequency)
    np.testing.assert_allclose(jnp.abs(exact), 1, atol=1e-8)
    np.testing.assert_allclose(jnp.abs(corrected), 1, atol=1e-8)
    assert float(jnp.max(1 - jnp.abs(default) ** 2)) > 0.1
    peaks = [
        frequency[
            jnp.argmax(
                jnp.abs(jnp.diff(jnp.unwrap(jnp.angle(value))) / jnp.diff(frequency))
            )
        ]
        for value in (exact, corrected)
    ]
    np.testing.assert_allclose(peaks[0], peaks[1], rtol=1e-4)


def test_roundtrip_geometry_gradient_and_domain(data: Dataset) -> None:
    """The transformed lookup retains native samples, JAX and domain checks."""
    model = s_parameters_model(data, interpolation="admittance")
    frequency = jnp.linspace(0.5e9, 15e9, 161)
    actual = _matrix(jax.jit(model)(f=frequency, coupling_ff=1.5))
    np.testing.assert_allclose(
        actual, _scattering(_admittance(frequency, 1.5)), atol=1e-12
    )
    off_grid = _matrix(model(f=jnp.array([4.123e9, 9.876e9]), coupling_ff=1.5))
    identity = jnp.broadcast_to(jnp.eye(3), off_grid.shape)
    np.testing.assert_allclose(
        off_grid.conj().swapaxes(-1, -2) @ off_grid, identity, atol=1e-12
    )
    np.testing.assert_allclose(off_grid, off_grid.swapaxes(-1, -2), atol=1e-12)

    def reflection(coupling: jax.Array | float) -> jax.Array:
        return jnp.imag(model(f=5.123e9, coupling_ff=coupling)["readout", "readout"])

    derivative = jax.jit(jax.grad(reflection))(1.5)
    assert jnp.isfinite(derivative)
    assert abs(float(derivative)) > 1e-8
    difference = (reflection(1.5001) - reflection(1.4999)) / 0.0002
    np.testing.assert_allclose(derivative, difference, rtol=1e-6)
    assert jax.jit(jax.vmap(reflection))(jnp.array([1.2, 1.8])).shape == (2,)
    assert jnp.isnan(model(f=16e9)["readout", "readout"])
    assert jnp.isnan(reflection(2.1))


def test_admittance_interpolation_retains_physical_loss(
    data: Dataset, tmp_path: Path
) -> None:
    """A real shunt conductance must survive conversion and interpolation."""

    def solve(frequency: float, coupling_ff: float) -> dict:
        return {"s_parameters": _scattering(_admittance(frequency, coupling_ff, 1e-5))}

    lossy = write(
        tmp_path / "lossy",
        data.metadata,
        sweep(
            data.metadata, solve, {"frequency": [4e9, 6e9], "coupling_ff": [1.0, 2.0]}
        ),
    )
    model = s_parameters_model(lossy, interpolation="admittance")
    scattering = _matrix(model(f=5e9, coupling_ff=1.5))
    recovered = jnp.linalg.solve(jnp.eye(3) + scattering, jnp.eye(3) - scattering) / 50
    np.testing.assert_allclose(
        recovered.real, jnp.diag(jnp.array([1e-5, 0, 0])), atol=1e-14
    )
    assert float(jnp.max(jnp.linalg.svd(scattering, compute_uv=False))) <= 1 + 1e-12
    assert float(1 - jnp.sum(jnp.abs(scattering[:, 0]) ** 2)) > 1e-4


def test_invalid_interpolation_and_admittance_pole(
    data: Dataset, tmp_path: Path
) -> None:
    """A sample at an admittance pole cannot use this representation."""
    with pytest.raises(ValueError, match="Interpolation"):
        s_parameters_model(data, interpolation="unknown")  # type: ignore[arg-type]

    def solve(**_params: float) -> dict:
        return {"s_parameters": -jnp.eye(3, dtype=complex)}

    pole = write(
        tmp_path / "pole",
        data.metadata,
        sweep(
            data.metadata, solve, {"frequency": [1e9, 2e9], "coupling_ff": [1.0, 2.0]}
        ),
    )
    with pytest.raises(ValueError, match="finite admittance"):
        s_parameters_model(pole, interpolation="admittance")
