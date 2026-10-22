"""Gatemon dataset lookups and linearised circuit on a synthetic matrix."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.constants import e, h
from qpdk.models.datasets import (
    Axis,
    Dataset,
    DatasetMetadata,
    Quantity,
    QuantityKind,
    sweep,
)
from qpdk.models.datasets.generate import write
from qpdk.models.gatemon import (
    gatemon_capacitances,
    gatemon_fem_model,
    gatemon_frequency_and_anharmonicity,
    gatemon_josephson_energy,
)
from qpdk.models.qubit import ej_to_inductance

GRID = {
    "island_size": [100.0, 200.0],
    "pocket_gap": [30.0, 90.0],
    "pocket_gap_top": [20.0, 60.0],
    "gate_tip_gap": [2.0, 10.0],
}
POINT = {
    "island_size": 150.0,
    "pocket_gap": 60.0,
    "pocket_gap_top": 40.0,
    "gate_tip_gap": 5.0,
}
PORTS = ("island", "gate", "readout")


def metadata(name: str, terminals: tuple[str, ...] = PORTS) -> DatasetMetadata:
    return DatasetMetadata(
        name=name,
        synthetic=True,
        axes=tuple(Axis(name=axis, unit="um") for axis in GRID),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=terminals,
    )


def maxwell(
    island_size: float, pocket_gap: float, pocket_gap_top: float, gate_tip_gap: float
) -> np.ndarray:
    """Multilinear in every axis, so interpolation is exact."""
    c_ground = (40 + 0.3 * island_size - 0.1 * pocket_gap) * 1e-15
    c_g = (0.5 - 0.02 * gate_tip_gap) * 1e-15
    c_c = (8 - 0.1 * pocket_gap_top) * 1e-15
    c_gate, c_readout = 20e-15, 60e-15
    return np.array([
        [c_ground + c_g + c_c, -c_g, -c_c],
        [-c_g, c_gate + c_g, 0.0],
        [-c_c, 0.0, c_readout + c_c],
    ])


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A labelled algebra fixture, not a simulated design dataset."""
    meta = metadata("test_gatemon")
    return write(
        tmp_path_factory.mktemp("gatemon"),
        meta,
        sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
    )


def test_capacitances(dataset: Dataset) -> None:
    c_sigma, c_g, c_c = gatemon_capacitances(**POINT, dataset=dataset)
    expected = maxwell(**POINT)
    np.testing.assert_allclose(
        [c_sigma, c_g, c_c],
        [expected[0, 0], -expected[0, 1], -expected[0, 2]],
        rtol=1e-6,
    )


def test_capacitances_by_path(tmp_path: Path) -> None:
    meta = metadata("test_gatemon_path")
    write(
        tmp_path,
        meta,
        sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
    )
    c_sigma, *_ = gatemon_capacitances(**POINT, dataset=tmp_path)
    np.testing.assert_allclose(c_sigma, maxwell(**POINT)[0, 0], rtol=1e-6)


def test_out_of_range_is_nan(dataset: Dataset) -> None:
    values = gatemon_capacitances(**{**POINT, "island_size": 300.0}, dataset=dataset)
    assert all(np.isnan(value) for value in values)


def test_transformations(dataset: Dataset) -> None:
    def c_sigma(size: jax.Array) -> jax.Array:
        return gatemon_capacitances(**{**POINT, "island_size": size}, dataset=dataset)[
            0
        ]

    sizes = jnp.array([120.0, 150.0, 180.0])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(c_sigma))(sizes),
        [maxwell(**{**POINT, "island_size": float(s)})[0, 0] for s in sizes],
        rtol=1e-6,
    )
    np.testing.assert_allclose(jax.grad(c_sigma)(150.0), 0.3e-15, rtol=1e-6)


def test_charging_energy_feeds_spectrum(dataset: Dataset) -> None:
    c_sigma, *_ = gatemon_capacitances(**POINT, dataset=dataset)
    ec_ghz = e**2 / (2 * c_sigma) / h / 1e9
    f01, alpha = gatemon_frequency_and_anharmonicity(ec_ghz=ec_ghz)
    assert 1 < float(f01) < 20
    assert float(alpha) < 0


def test_fem_model_matches_admittance(dataset: Dataset) -> None:
    delta_ghz, transmissions, c_j, f, z_ref = 45.0, (0.6, 0.3), 1e-15, 5e9, 50.0
    model = gatemon_fem_model(
        delta_ghz=delta_ghz,
        transmissions=transmissions,
        junction_capacitance=c_j,
        dataset=dataset,
    )
    result = model(f=f, z_ref=z_ref, **POINT)
    s = np.array([[result[a, b] for b in PORTS] for a in PORTS])

    omega = 2 * np.pi * f
    inductance = float(
        ej_to_inductance(gatemon_josephson_energy(delta_ghz, transmissions))
    )
    y = 1j * omega * maxwell(**POINT)
    y[0, 0] += 1j * omega * c_j + 1 / (1j * omega * inductance)
    eye = np.eye(3)
    expected = np.linalg.solve(eye + z_ref * y, eye - z_ref * y)
    np.testing.assert_allclose(s, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(s, s.T, atol=1e-7)
    np.testing.assert_allclose(s.conj().T @ s, eye, atol=1e-6)


def test_fem_model_gate_tuning(dataset: Dataset) -> None:
    """Lower transmission raises L_J, shifting the island resonance down."""
    model = gatemon_fem_model(dataset=dataset)
    f = jnp.linspace(1e9, 15e9, 2801)

    def resonance(transmissions: tuple[float, ...]) -> float:
        # The parallel LC opens the island port: S11 swings towards +1
        s11 = model(f=f, transmissions=transmissions, **POINT)["island", "island"]
        return float(f[jnp.argmax(jnp.real(s11))])

    c_sigma = maxwell(**POINT)[0, 0]
    for tau in (0.8, 0.4):
        inductance = float(ej_to_inductance(gatemon_josephson_energy(50.0, (tau,))))
        assert resonance((tau,)) == pytest.approx(
            1 / (2 * np.pi * np.sqrt(inductance * c_sigma)), rel=0.05
        )
    assert resonance((0.4,)) < resonance((0.8,))


def test_fem_model_jit(dataset: Dataset) -> None:
    model = gatemon_fem_model(dataset=dataset)
    s11 = jax.jit(lambda size: model(island_size=size)["island", "island"])(150.0)
    assert np.isfinite(s11)
    assert np.isnan(model(f=0.0)["island", "island"])


def test_rejects_wrong_terminals(tmp_path: Path) -> None:
    meta = metadata("test_wrong", ("a", "b", "c"))
    data = write(
        tmp_path,
        meta,
        sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
    )
    with pytest.raises(ValueError, match="terminals"):
        gatemon_capacitances(dataset=data)
