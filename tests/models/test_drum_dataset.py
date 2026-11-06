"""Drum dataset lookups and FEM-backed resonator on a synthetic matrix."""

import warnings
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
    sweep,
)
from qpdk.models.datasets.generate import write
from qpdk.models.drum import (
    VACUUM_GAP,
    drum_capacitance_reduction,
    drum_capacitances,
    drum_lc_resonator,
    drum_lc_resonator_fem_model,
    vacuum_gap_capacitance_analytical,
)

GRID = {
    "coupling_gap": [3.0, 12.0],
    "coupling_pad_size": [10.0, 40.0],
    "radius": [5.0, 10.0],
    "turn_length": [200.0, 300.0],
}
POINT = {
    "coupling_gap": 6.0,
    "coupling_pad_size": 20.0,
    "radius": 7.5,
    "turn_length": 300.0,
}
TERMINALS = ("node", "membrane", "feed", "meander_1", "meander_2")
WEIGHTS = [1.0, 0.0, 0.0, 0.25, 0.75]


def metadata(
    name: str,
    terminals: tuple[str, ...] = TERMINALS,
    weights: list[float] | None = WEIGHTS,
) -> DatasetMetadata:
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
        provenance={} if weights is None else {"terminal_weights": weights},
    )


def coupling(
    coupling_gap: float, coupling_pad_size: float, radius: float, turn_length: float
) -> dict[str, float]:
    """Multilinear mutual capacitances, so interpolation is exact."""
    return {
        "node_feed": (2.0 - 0.1 * coupling_gap + 0.02 * coupling_pad_size) * 1e-15,
        "node_membrane": (20.0 + 2.0 * radius) * 1e-15,
        "node_ground": (5.0 + 0.1 * coupling_pad_size) * 1e-15,
        "meander_ground": (0.02 * turn_length) * 1e-15,
        "meander_feed": 0.1e-15,
        "chain": 3e-15,
    }


def maxwell(**point: float) -> np.ndarray:
    """Build a Maxwell matrix from mutual and ground capacitances."""
    c = coupling(**point)
    mutual = np.zeros((5, 5))
    ground = np.array([
        c["node_ground"],
        30e-15,
        60e-15,
        c["meander_ground"],
        c["meander_ground"],
    ])

    def link(a: int, b: int, value: float) -> None:
        mutual[a, b] = mutual[b, a] = value

    link(0, 2, c["node_feed"])
    link(0, 1, c["node_membrane"])
    link(3, 4, c["chain"])
    link(4, 0, c["chain"])
    link(3, 2, c["meander_feed"])
    link(4, 2, c["meander_feed"])
    return np.diag(ground + mutual.sum(1)) - mutual


def expected(**point: float) -> tuple[float, float, float]:
    """Energy bookkeeping of the same circuit, written out by hand."""
    c = coupling(**point)
    v = WEIGHTS
    c_c = c["node_feed"] + c["meander_feed"] * (v[3] + v[4])
    c_d = c["node_membrane"]
    c_p = (
        c["node_ground"]
        + c["meander_ground"] * (v[3] ** 2 + v[4] ** 2)
        + c["chain"] * ((v[4] - v[3]) ** 2 + (1 - v[4]) ** 2)
        # Meander-to-feed coupling stores v^2 of energy but adds v to C_c
        + c["meander_feed"] * (v[3] ** 2 + v[4] ** 2 - v[3] - v[4])
    )
    return c_c, c_p, c_d


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A labelled algebra fixture, not a simulated design dataset."""
    meta = metadata("test_drum")
    return write(
        tmp_path_factory.mktemp("drum"),
        meta,
        sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
    )


class TestDrumCapacitances:
    """Maxwell-matrix reduction and dataset lookup."""

    @staticmethod
    def test_reduction() -> None:
        np.testing.assert_allclose(
            drum_capacitance_reduction(maxwell(**POINT), np.asarray(WEIGHTS)),
            expected(**POINT),
            rtol=1e-9,
        )

    @staticmethod
    def test_lookup(dataset: Dataset) -> None:
        np.testing.assert_allclose(
            drum_capacitances(**POINT, dataset=dataset), expected(**POINT), rtol=1e-6
        )

    @staticmethod
    def test_by_path(tmp_path: Path) -> None:
        meta = metadata("test_drum_path")
        write(
            tmp_path,
            meta,
            sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
        )
        c_c, *_ = drum_capacitances(**POINT, dataset=tmp_path)
        np.testing.assert_allclose(c_c, expected(**POINT)[0], rtol=1e-6)

    @staticmethod
    def test_out_of_range_is_nan(dataset: Dataset) -> None:
        values = drum_capacitances(**{**POINT, "radius": 20.0}, dataset=dataset)
        assert all(np.isnan(value) for value in values)

    @staticmethod
    def test_missing_bundled_dataset_is_nan() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = drum_capacitances(radius=jnp.array([5.0, 7.5]))
        if not np.all(np.isnan(values)):
            pytest.skip("The bundled drum dataset exists")
        assert all(np.shape(value) == (2,) for value in values)

    @staticmethod
    def test_missing_explicit_path_raises(tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            drum_capacitances(dataset=tmp_path / "absent")

    @staticmethod
    @pytest.mark.parametrize(
        ("terminals", "weights", "match"),
        [
            (
                ("membrane", "node", "feed", "meander_1", "meander_2"),
                WEIGHTS,
                "terminals",
            ),
            (TERMINALS, None, "terminal_weights"),
            (TERMINALS, WEIGHTS[:-1], "terminal_weights"),
        ],
    )
    def test_rejects_bad_metadata(
        tmp_path: Path,
        terminals: tuple[str, ...],
        weights: list[float] | None,
        match: str,
    ) -> None:
        meta = metadata("test_drum_bad", terminals, weights)
        data = write(
            tmp_path,
            meta,
            sweep(meta, lambda **p: {"maxwell_capacitance": maxwell(**p)}, GRID),
        )
        with pytest.raises(ValueError, match=match):
            drum_capacitances(dataset=data)

    @staticmethod
    def test_transformations(dataset: Dataset) -> None:
        def c_d(radius: jax.Array) -> jax.Array:
            return drum_capacitances(**{**POINT, "radius": radius}, dataset=dataset)[2]

        radii = jnp.array([6.0, 7.5, 9.0])
        np.testing.assert_allclose(
            jax.jit(jax.vmap(c_d))(radii),
            [expected(**{**POINT, "radius": float(r)})[2] for r in radii],
            rtol=1e-6,
        )
        np.testing.assert_allclose(jax.grad(c_d)(7.5), 2e-15, rtol=1e-6)


class TestDrumFemModel:
    """Lumped resonator driven by extracted capacitances."""

    @staticmethod
    def test_matches_lumped_model(dataset: Dataset) -> None:
        model = drum_lc_resonator_fem_model(dataset=dataset)
        f = jnp.linspace(4e9, 9e9, 11)
        c_c, c_p, c_d = expected(**POINT)
        parasitic = (
            c_p
            + c_d
            - float(
                vacuum_gap_capacitance_analytical(
                    radius=POINT["radius"], gap=VACUUM_GAP
                )
            )
        )
        reference = drum_lc_resonator(
            f=f, coupling_capacitance=c_c, parasitic_capacitance=parasitic
        )
        result = model(f=f, **POINT)
        for key in reference:
            np.testing.assert_allclose(
                result[key], reference[key], rtol=1e-5, atol=1e-9
            )

    @staticmethod
    def test_gap_still_tunes(dataset: Dataset) -> None:
        """A smaller gap raises the drum capacitance and lowers the dip."""
        model = drum_lc_resonator_fem_model(dataset=dataset)
        f = jnp.linspace(1e9, 20e9, 4001)

        def dip(gap: float) -> float:
            s21 = model(f=f, gap=gap)["o1", "o2"]
            return float(f[jnp.argmin(jnp.abs(s21))])

        assert dip(0.9 * VACUUM_GAP) < dip(VACUUM_GAP)

    @staticmethod
    def test_jit_and_nan_without_data() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = drum_lc_resonator_fem_model()
            s21 = jax.jit(lambda r: model(radius=r)["o1", "o2"])(7.5)
        if np.isfinite(s21):
            pytest.skip("The bundled drum dataset exists")
        assert np.isnan(s21)
