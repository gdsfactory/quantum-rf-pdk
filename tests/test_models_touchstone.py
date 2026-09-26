"""Tests for the Touchstone reader/writer (qpdk/models/touchstone.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpdk import PDK
from qpdk.models.resonator import quarter_wave_resonator_coupled
from qpdk.models.touchstone import (
    array_to_sdict,
    format_touchstone,
    parse_touchstone,
    read_touchstone,
    sdict_to_array,
    write_touchstone,
)
from qpdk.models.waveguides import straight

skrf = pytest.importorskip("skrf", reason="scikit-rf is only used to cross-check")

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


def test_array_to_sdict_rejects_mismatched_matrix() -> None:
    """A matrix whose shape disagrees with the port list would drop entries."""
    with pytest.raises(ValueError, match=r"expected a \(k, 1, 1\) array"):
        array_to_sdict(np.zeros((3, 2, 2)), ("o1",))


def test_array_to_sdict_rejects_duplicate_ports() -> None:
    """Duplicate port names would silently overwrite dictionary keys."""
    with pytest.raises(ValueError, match="duplicate names"):
        array_to_sdict(np.zeros((3, 2, 2)), ("o1", "o1"))


def test_format_rejects_mismatched_frequencies(cpw_sdict: dict) -> None:
    """Passing a different frequency vector than the model saw is an error."""
    s_array, ports = sdict_to_array(cpw_sdict)
    with pytest.raises(ValueError, match="pass the same frequency array"):
        format_touchstone(s_array, FREQUENCIES[:-1], ports)


def test_round_trips_through_touchstone(cpw_sdict: dict, tmp_path: Path) -> None:
    """Writing and re-reading a ``.s2p`` preserves frequencies and S-parameters."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p")
    assert path.exists()

    frequency, sdict = read_touchstone(path)
    np.testing.assert_allclose(frequency, FREQUENCIES, rtol=1e-12)
    assert set(sdict) == set(cpw_sdict)
    for key, value in cpw_sdict.items():
        np.testing.assert_allclose(sdict[key], np.asarray(value), atol=1e-11)


def test_two_port_column_order_matches_the_spec(cpw_sdict: dict) -> None:
    """The two-port layout is ``f S11 S21 S12 S22``, not row-major."""
    s_array, ports = sdict_to_array(cpw_sdict)
    row = format_touchstone(s_array, FREQUENCIES, ports).splitlines()[3].split()
    written = np.array([float(value) for value in row[1:]]).reshape(4, 2)
    expected = s_array[0].T.reshape(-1)
    np.testing.assert_allclose(written[:, 0] + 1j * written[:, 1], expected, atol=1e-11)


def test_touchstone_header_records_port_names(cpw_sdict: dict, tmp_path: Path) -> None:
    """The port-order comment survives, since Touchstone only stores indices."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "! ports: o1, o2" in lines
    assert lines[2] == "# Hz S RI R 50"


def test_missing_option_line_applies_the_standard_defaults(tmp_path: Path) -> None:
    """A file without an option line is read with the ``GHz S MA R 50`` defaults."""
    content = "! no option line\n1 0.5 0.1\n2 0.4 0.2\n"
    path = tmp_path / "default.s1p"
    path.write_text(content)

    frequency, s_array, _ports, z0 = parse_touchstone(content, n_ports=1)
    np.testing.assert_allclose(frequency, [1e9, 2e9], rtol=1e-12)
    np.testing.assert_allclose(
        s_array[:, 0, 0], [0.5, 0.4] * np.exp(1j * np.deg2rad([0.1, 0.2])), atol=1e-12
    )
    assert z0 == pytest.approx(50.0)

    # ...and scikit-rf, as the independent implementation, applies the same defaults.
    network = skrf.Network(str(path))
    np.testing.assert_allclose(network.f, frequency, rtol=1e-12)
    np.testing.assert_allclose(network.s[:, 0, 0], s_array[:, 0, 0], atol=1e-12)


def test_three_port_model_round_trips(tmp_path: Path) -> None:
    """A three-port model such as the coupled resonator writes a ``.s3p``."""
    sdict = quarter_wave_resonator_coupled(f=FREQUENCIES)
    expected, ports = sdict_to_array(sdict)
    path = write_touchstone(sdict, FREQUENCIES, tmp_path / "resonator.s3p")
    assert not (tmp_path / "resonator.s3p.s3p").exists()

    frequency, s_array, file_ports, z0 = parse_touchstone(path.read_text())
    assert s_array.shape == (FREQUENCIES.size, 3, 3)
    assert file_ports == ports
    assert z0 == pytest.approx(50.0)
    np.testing.assert_allclose(frequency, FREQUENCIES, rtol=1e-12)
    np.testing.assert_allclose(s_array, expected, atol=1e-11)


def test_reference_keyword_overrides_the_option_line() -> None:
    """A uniform v2 ``[Reference]`` overrides the option line's R."""
    content = (
        "# Hz S RI R 50\n"
        "[Number of Ports] 2\n"
        "[Reference] 75\n"
        "[Network Data]\n"
        "1e9 0 0 0 0 0 0 0 0"
    )
    _, _, _, z0 = parse_touchstone(content)
    assert z0 == pytest.approx(75.0)


def test_per_port_reference_is_rejected() -> None:
    """The reader reports a single z0, so a per-port [Reference] is an error."""
    content = "[Number of Ports] 2\n[Reference] 50 75\n[Network Data]"
    with pytest.raises(ValueError, match="per-port reference"):
        parse_touchstone(content)


def test_reference_impedance_is_configurable(cpw_sdict: dict, tmp_path: Path) -> None:
    """``z0`` lands in the Touchstone option line and comes back out."""
    path = write_touchstone(cpw_sdict, FREQUENCIES, tmp_path / "cpw.s2p", z0=75.0)
    assert "# Hz S RI R 75" in path.read_text(encoding="utf-8")
    assert parse_touchstone(path.read_text())[3] == pytest.approx(75.0)


@pytest.mark.parametrize("frequency_unit", ["Hz", "MHz", "GHz"])
def test_frequency_unit_only_changes_the_numbers(
    cpw_sdict: dict, frequency_unit: str
) -> None:
    """Frequencies are always Hz in Python, whatever unit the file uses."""
    s_array, ports = sdict_to_array(cpw_sdict)
    content = format_touchstone(
        s_array, FREQUENCIES, ports, frequency_unit=frequency_unit
    )
    np.testing.assert_allclose(parse_touchstone(content)[0], FREQUENCIES, rtol=1e-12)


@pytest.mark.parametrize("n_ports", [1, 2, 3, 5])
def test_scikit_rf_reads_what_we_write(n_ports: int, tmp_path: Path) -> None:
    """Cross-check the hand-rolled writer against another Touchstone reader.

    scikit-rf is not a dependency of :mod:`qpdk.models.touchstone`; it is used
    here only as an independent implementation of the same specification, which
    is what makes the port-index conventions worth trusting.
    """
    rng = np.random.default_rng(n_ports)
    shape = (FREQUENCIES.size, n_ports, n_ports)
    expected = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    path = tmp_path / f"random.s{n_ports}p"
    path.write_text(format_touchstone(expected, FREQUENCIES))

    network = skrf.Network(str(path))
    assert network.nports == n_ports
    np.testing.assert_allclose(network.f, FREQUENCIES, rtol=1e-12)
    np.testing.assert_allclose(network.s, expected, atol=1e-11)

    # ...and that we read back what it writes, including its port-name comments.
    written = tmp_path / f"skrf.s{n_ports}p"
    network.write_touchstone(str(written.with_suffix("")), form="ri", write_z0=False)
    _, s_array, _, _ = parse_touchstone(written.read_text())
    np.testing.assert_allclose(s_array, expected, atol=1e-11)
