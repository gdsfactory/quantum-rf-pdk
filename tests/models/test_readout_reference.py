"""Independent numerical regression for the readout assembly models.

Rebuild the qubit load from the raw FEM Maxwell matrix, solve the coupled
island/readout circuit, terminate the CPW coupler with an independently solved
scattering connection, and compare the public model two-port S matrix.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
import sax
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from qpdk.models import (
    flipmon_with_resonator_and_probeline,
    transmon_with_resonator_and_probeline,
)
from qpdk.models.couplers import coupler_straight
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.datasets import Dataset, Grid
from qpdk.models.waveguides import straight

_FREQUENCIES = jnp.asarray([1e9, 4.67e9, 6.7e9, 10.2e9, 14.53e9, 18.7e9])
_RESONATOR_LENGTH = 5000.0
_COUPLER = ("o1", "o4", "o2", "o3")
_READOUT = ("coupling_o1", "coupling_o2")
# Model, coupon, exact terminals, and the shipped pad length, pad offset and
# nominal route plus coupling span in um.
_KINDS = (
    (
        transmon_with_resonator_and_probeline,
        "transmon_readout_coupon_palace",
        ("island_a", "island_b", "readout", "feedline"),
        (394.0, 45.0, 1742.5795478),
    ),
    (
        flipmon_with_resonator_and_probeline,
        "flipmon_readout_coupon_palace",
        ("island_a", "island_b", "readout", "feedline", "top_pad"),
        (58.0, 10.0, 1900.0795478),
    ),
)


def _reference(
    f: NDArray[np.float64],
    grid: Grid,
    n_terminals: int,
    pad: tuple[float, float, float],
    coupling_length: float,
    open_end: bool,
) -> NDArray[np.complex128]:
    """Build the expected two-port S matrix from the raw coupon matrix.

    The coupon interpolation always uses the nominal 16 um gap and 800 um
    coupling length; only the distributed coupler gets the requested length.

    Returns:
        The two-port S matrix with shape ``(frequency, response, excitation)``.
    """
    z_ref = float(cpw_z0_from_cross_section("cpw"))
    raw = RegularGridInterpolator(
        tuple(map(np.asarray, grid.coords)), np.asarray(grid.values)
    )((*pad[:2], 16.0, 800.0)).reshape(n_terminals, -1)
    if n_terminals == 5:  # Flipmon top pad is bonded to island_a.
        bond = np.vstack([np.eye(4), np.eye(4)[0]])
        raw = bond.T @ raw @ bond
    # Keep the island branches and island ground; feedline branches are in the
    # distributed sections.
    ycap = np.zeros((3, 3))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        branch = np.eye(3)[i] - np.eye(3)[j]
        ycap -= raw[i, j] * np.outer(branch, branch)
    ycap[np.diag_indices(2)] += raw[:2].sum(axis=1)
    omega = 2 * np.pi * f
    drive = np.array([1.0, -1.0, 0.0])
    y = 1j * omega[:, None, None] * ycap
    y += np.outer(drive, drive) / (1j * omega[:, None, None] * 7e-9)
    # Islands are internal; drive the readout electrode at one volt.
    islands = np.linalg.solve(y[:, :2, :2], -y[:, :2, 2, None])
    current = (y[:, 2:3, :2] @ islands)[:, 0, 0] + y[:, 2, 2]
    gamma_plate = (1 - z_ref * current) / (1 + z_ref * current)

    span = pad[2]
    coupler = coupler_straight(
        f=f, length=coupling_length, gap=16.0, cross_section="cpw"
    )
    coupler_s = np.stack(
        [np.asarray(coupler[p, q]) for p in _COUPLER for q in _COUPLER], axis=-1
    ).reshape(-1, 4, 4)
    meander = np.asarray(
        straight(f=f, length=_RESONATOR_LENGTH - span)["o1", "o2"]
    ) ** 2 * (1.0 if open_end else -1.0)
    route = (
        np.asarray(straight(f=f, length=span - coupling_length)["o1", "o2"]) ** 2
        * gamma_plate
    )
    loads = np.stack((meander, route), axis=-1)[:, :, None] * np.eye(2)
    expected = coupler_s[:, :2, :2] + coupler_s[:, :2, 2:] @ loads @ np.linalg.solve(
        np.eye(2) - coupler_s[:, 2:, 2:] @ loads, coupler_s[:, 2:, :2]
    )
    return np.asarray(expected, dtype=np.complex128)


@pytest.mark.parametrize(
    ("coupling_length", "open_end"),
    [(800.0, True), (400.0, False)],
    ids=["800-open", "400-short"],
)
@pytest.mark.parametrize(
    ("model", "dataset", "terminals", "pad"), _KINDS, ids=["transmon", "flipmon"]
)
def test_readout_assembly_matches_independent_reference(
    model: Callable[..., sax.SDict],
    dataset: str,
    terminals: tuple[str, ...],
    pad: tuple[float, float, float],
    coupling_length: float,
    open_end: bool,
) -> None:
    """The public S matrix equals a reference built from the raw coupon matrix."""
    grid = Dataset(dataset).grid("maxwell_capacitance")
    assert tuple(grid.terminals) == terminals
    f = np.asarray(_FREQUENCIES, dtype=float)
    expected = _reference(f, grid, len(terminals), pad, coupling_length, open_end)

    result = model(
        f=_FREQUENCIES,
        resonator_length=_RESONATOR_LENGTH,
        probeline_coupling_length=coupling_length,
        probeline_coupling_gap=16.0,
        resonator_open_end=open_end,
    )
    actual = np.stack(
        [np.asarray(result[p, q]) for p in _READOUT for q in _READOUT], axis=-1
    ).reshape(-1, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
