"""Tests for the Touchstone exporter (qpdk/models/touchstone.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpdk import PDK
from qpdk.models.resonator import quarter_wave_resonator_coupled
from qpdk.models.touchstone import (
    sdict_to_array,
    sdict_to_network,
    write_touchstone,
)
from qpdk.models.waveguides import straight

skrf = pytest.importorskip("skrf", reason="models extra not installed")

PDK.activate()

FREQUENCIES = np.linspace(4e9, 8e9, 21)


@pytest.fixture(scope="module")
def cpw_sdict() -> dict:
    """S-parameters of a 1 mm CPW line, the canonical two-port example."""
    return straight(f=FREQUENCIES, length=1000)


def test_sdict_to_array_shape_and_port_order(cpw_sdict: dict) -> None:
    """A two-port model densifies to ``(k, 2, 2)`` with named ports."""
    s_array, ports = sdict_to_array(cpw_sdict)
    assert s_array.shape == (FREQUENCIES.size, 2, 2)
    assert ports == ("o1", "o2")
    np.testing.assert_allclose(s_array[:, 0, 1], s_array[:, 1, 0], rtol=1e-12)


def test_sdict_to_array_respects_explicit_port_order(cpw_sdict: dict) -> None:
    """Reordering the ports permutes both axes of the S-matrix."""
    default, _ = sdict_to_array(cpw_sdict)
    swapped, ports = sdict_to_array(cpw_sdict, ports=("o2", "o1"))
    assert ports == ("o2", "o1")
    np.testing.assert_allclose(swapped, default[:, ::-1, ::-1], rtol=1e-12)


def test_sdict_to_array_rejects_unknown_ports(cpw_sdict: dict) -> None:
    """A port list that is not a permutation of the model's ports is an error."""
    with pytest.raises(ValueError, match="not a permutation"):
        sdict_to_array(cpw_sdict, ports=("o1", "nonexistent"))


def test_sdict_to_network_rejects_mismatched_frequencies(cpw_sdict: dict) -> None:
    """Passing a different frequency vector than the model saw is an error."""
    with pytest.raises(ValueError, match="pass the same frequency array"):
        sdict_to_network(cpw_sdict, FREQUENCIES[:-1])


def test_network_round_trips_through_touchstone(
    cpw_sdict: dict, tmp_path: Path
) -> None:
    """Writing and re-reading a ``.s2p`` preserves frequencies and S-parameters."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p")
    assert path.exists()

    expected, _ = sdict_to_array(cpw_sdict)
    network = skrf.Network(str(path))
    assert network.nports == 2
    np.testing.assert_allclose(network.f, FREQUENCIES, rtol=1e-12)
    np.testing.assert_allclose(network.s, expected, atol=1e-9)
    np.testing.assert_allclose(network.z0, 50.0)


def test_touchstone_header_records_port_names(cpw_sdict: dict, tmp_path: Path) -> None:
    """The port-order comment survives, since Touchstone only stores indices."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p")
    header = path.read_text(encoding="utf-8")
    assert "Port[1] = o1" in header
    assert "Port[2] = o2" in header
    assert header.startswith("# Hz S RI R 50")


def test_three_port_model_exports(tmp_path: Path) -> None:
    """A three-port model such as the coupled resonator writes a ``.s3p``."""
    sdict = quarter_wave_resonator_coupled(f=FREQUENCIES)
    path = write_touchstone(sdict, FREQUENCIES, tmp_path / "resonator.s3p")
    network = skrf.Network(str(path))
    assert network.nports == 3
    assert not (tmp_path / "resonator.s3p.s3p").exists()


def test_reference_impedance_is_configurable(cpw_sdict: dict, tmp_path: Path) -> None:
    """``z0`` lands in the Touchstone option line."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p", z0=75.0)
    assert path.read_text(encoding="utf-8").startswith("# Hz S RI R 75")
