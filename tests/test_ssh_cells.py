"""Tests for the SSH coupled-resonator chain cell."""

import pytest

from qpdk.cells.ssh import ssh_resonator_chain


def test_ports_and_lattice() -> None:
    """End ports sit on the chain axis and taps above their sites."""
    c = ssh_resonator_chain(n_cells=3, site_spacing=300.0, tap_sites=(4, 1))
    ports = {port.name: port for port in c.ports}
    assert set(ports) == {"o1", "o2", "o3", "o4"}
    assert ports["o1"].orientation == 180
    assert ports["o2"].orientation == 0
    assert ports["o1"].y == ports["o2"].y == 0
    # Taps are numbered in the order given.
    assert ports["o3"].x == pytest.approx(4 * 300.0)
    assert ports["o4"].x == pytest.approx(300.0)
    assert ports["o3"].orientation == ports["o4"].orientation == 90
    assert c.info["n_sites"] == 6
    assert c.info["lattice_constant"] == pytest.approx(600.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cells": 0},
        {"n_cells": 2, "tap_sites": (4,)},
        {"n_cells": 2, "tap_sites": (1, 1)},
        {"site_spacing": 50.0},
    ],
)
def test_invalid(kwargs: dict) -> None:
    """Invalid cell counts, taps and spacings raise ValueError."""
    with pytest.raises(ValueError, match=r"n_cells|tap_sites|site_spacing"):
        ssh_resonator_chain(**kwargs)
