"""Reference check for the readout assembly FEM load and termination.

The FEM-to-admittance reduction, the nodal island solve and the load
termination are independent. The reference reuses QPDK's distributed CPW
straight and coupler primitives and its characteristic impedance.
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
import sax
from gdsfactory.component import Component
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from qpdk import PDK
from qpdk.models import (
    flipmon_with_resonator_and_probeline,
    transmon_with_resonator_and_probeline,
)
from qpdk.models.couplers import coupler_straight
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.datasets import Dataset, Grid
from qpdk.models.readout_assembly import _COUPONS
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
_LAYOUTS: tuple[tuple[str, Literal["double_pad", "flipmon"]], ...] = (
    ("double_pad_transmon_with_resonator_and_probeline", "double_pad"),
    ("flipmon_with_resonator_and_probeline", "flipmon"),
)


def _reference(
    f: NDArray[np.float64],
    grid: Grid,
    n_terminals: int,
    pad: tuple[float, float, float],
    coupling_length: float,
    open_end: bool,
    junction_capacitance: float,
) -> NDArray[np.complex128]:
    """Build the expected S matrix from the raw coupon matrix.

    The coupon interpolation uses the fixed nominal 16 um gap and 800 um
    coupling length; only the distributed coupler gets the requested length.

    Args:
        f: Frequency in Hz.
        grid: Raw Maxwell capacitance grid for the coupon.
        n_terminals: Number of coupon terminals before bonding.
        pad: Pad length, pad offset and nominal route plus coupling span in um.
        coupling_length: Distributed coupling length in um.
        open_end: Open the meander far end when true, short it when false.
        junction_capacitance: Junction capacitance in F.

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
    # The junction capacitance shunts the same island pair as the inductance.
    y += np.outer(drive, drive) * (
        1 / (1j * omega[:, None, None] * 7e-9)
        + 1j * omega[:, None, None] * junction_capacitance
    )
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


def _drawn_span(cell: Component) -> float:
    """Return the route plus coupling span drawn in a readout cell."""
    meander = next(inst for inst in cell.insts if "resonator_type" in inst.cell.info)
    return float(cell.info["length"]) - float(meander.cell.info["length"])


@pytest.mark.parametrize(
    ("coupling_length", "open_end", "junction_capacitance"),
    [(800.0, True, 0.0), (400.0, False, 0.0), (800.0, True, 3e-15)],
    ids=["800-open", "400-short", "800-open-junction-capacitance"],
)
@pytest.mark.parametrize(
    ("model", "dataset", "terminals", "pad"), _KINDS, ids=["transmon", "flipmon"]
)
def test_fem_load_and_termination_match_public_model(
    model: Callable[..., sax.SDict],
    dataset: str,
    terminals: tuple[str, ...],
    pad: tuple[float, float, float],
    coupling_length: float,
    open_end: bool,
    junction_capacitance: float,
) -> None:
    """The independent FEM load solve and termination match the public model."""
    grid = Dataset(dataset).grid("maxwell_capacitance")
    assert tuple(grid.terminals) == terminals
    f = np.asarray(_FREQUENCIES, dtype=float)
    expected = _reference(
        f, grid, len(terminals), pad, coupling_length, open_end, junction_capacitance
    )

    result = model(
        f=_FREQUENCIES,
        resonator_length=_RESONATOR_LENGTH,
        probeline_coupling_length=coupling_length,
        probeline_coupling_gap=16.0,
        resonator_open_end=open_end,
        junction_capacitance=junction_capacitance,
    )
    actual = np.stack(
        [np.asarray(result[p, q]) for p in _READOUT for q in _READOUT], axis=-1
    ).reshape(-1, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("coupling_length", [400.0, 800.0], ids=["400", "800"])
@pytest.mark.parametrize(("cell_name", "kind"), _LAYOUTS, ids=["double-pad", "flipmon"])
def test_drawn_span_matches_nominal_coupon_span(
    cell_name: str,
    kind: Literal["double_pad", "flipmon"],
    coupling_length: float,
) -> None:
    """The drawn route plus coupling span equals the model's coupon span."""
    cell = PDK.cells[cell_name](probeline_coupling_length=coupling_length)
    span = _COUPONS[kind].span
    assert _drawn_span(cell) == pytest.approx(span, abs=1e-3)
