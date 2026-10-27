"""Check both complete, license-free test-chip circuit representations."""

import subprocess
import sys
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import find_peaks

from .gatemon_test_chip import _circuit, _netlist


def _settings(
    netlist: dict, component: str, **settings: float | tuple[float, ...]
) -> dict:
    return {
        name: settings
        for name, instance in netlist["instances"].items()
        if instance["component"] == component
    }


def test_extracted_capacitances_keep_si_precision() -> None:
    """Netlist serialization must retain the actual coupling in farads."""
    netlist = _netlist("layout")
    couplers = [
        instance
        for instance in netlist["instances"].values()
        if instance["component"] == "gatemon_coupler"
    ]
    assert len(couplers) == 2
    assert all(
        instance["settings"]["coupling_capacitance"]
        == pytest.approx(5.4e-15, rel=1e-12, abs=0)
        for instance in couplers
    )


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_complete_chip_is_passive_reciprocal_and_independent(
    source: Literal["schematic", "layout"],
) -> None:
    """Account for the matched gate ports and the CPW dielectric loss."""
    circuit, netlist = _circuit(source)
    ports = tuple(netlist["ports"])
    assert set(ports) == {"o1", "o2", "o3", "o4", "gate_a", "gate_b"}
    response = circuit(f=jnp.linspace(1e9, 15e9, 101))
    matrix = jnp.stack(
        [jnp.stack([response[a, b] for b in ports], -1) for a in ports], -2
    )
    assert bool(jnp.all(jnp.isfinite(matrix)))
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-10)
    assert float(jnp.max(jnp.linalg.svd(matrix, compute_uv=False))) <= 1 + 1e-10
    np.testing.assert_allclose(response["o3", "o1"], 0, atol=1e-12)
    np.testing.assert_allclose(response["gate_b", "o1"], 0, atol=1e-12)
    np.testing.assert_allclose(response["o2", "o1"], response["o4", "o3"], atol=1e-10)


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_complete_chip_cavity_and_dispersive_shift(
    source: Literal["schematic", "layout"],
) -> None:
    """The open-junction cavity is at 5.056 GHz and shifts with Josephson energy."""
    circuit, netlist = _circuit(source)
    f = jnp.linspace(5.05e9, 5.07e9, 4001)
    peaks = []
    for τ in (0.0, 0.8):
        response = circuit(f=f, **_settings(netlist, "gatemon", transmissions=(τ,)))
        peaks.append(float(f[jnp.argmax(jnp.abs(response["o2", "o1"]))]))
    assert peaks[0] == pytest.approx(5.056e9, abs=2e4)
    assert peaks[1] - peaks[0] == pytest.approx(4.324e6, abs=2e4)


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_complete_chip_qubit_feature_tunes(
    source: Literal["schematic", "layout"],
) -> None:
    """The broad gate-transfer feature follows the linear Josephson frequency."""
    circuit, netlist = _circuit(source)
    f = jnp.linspace(3.3e9, 4.75e9, 2901)
    for τ, expected in ((0.5, 3.606e9), (0.8, 4.559e9)):
        response = circuit(f=f, **_settings(netlist, "gatemon", transmissions=(τ,)))
        peaks, _ = find_peaks(np.asarray(jnp.abs(response["gate_a", "o1"])))
        assert len(peaks) == 1
        assert float(f[peaks[0]]) == pytest.approx(expected, abs=1e6)


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_complete_chip_has_two_hybrid_modes(
    source: Literal["schematic", "layout"],
) -> None:
    """Near the crossing the splitting is consistent with g about 50 MHz."""
    circuit, netlist = _circuit(source)
    f = jnp.linspace(4.95e9, 5.16e9, 4201)
    response = circuit(f=f, **_settings(netlist, "gatemon", transmissions=(0.98,)))
    peaks, _ = find_peaks(np.asarray(jnp.abs(response["o2", "o1"])), prominence=1e-5)
    assert len(peaks) == 2
    assert float(f[peaks[1]] - f[peaks[0]]) == pytest.approx(100e6, abs=5e6)


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_zero_coupling_removes_the_qubit_signature(
    source: Literal["schematic", "layout"],
) -> None:
    """An isolated qubit cannot move the readout or transfer power from RF to gate."""
    circuit, netlist = _circuit(source)
    f = jnp.linspace(3e9, 6e9, 301)
    settings = _settings(netlist, "gatemon_coupler", coupling_capacitance=0.0)
    on = circuit(f=f, **settings)
    off = circuit(
        f=f, **(settings | _settings(netlist, "gatemon", transmissions=(0.0,)))
    )
    np.testing.assert_allclose(on["gate_a", "o1"], 0, atol=1e-12)
    np.testing.assert_allclose(on["o2", "o1"], off["o2", "o1"], atol=1e-12)


@pytest.mark.parametrize("source", ["schematic", "layout"])
def test_gate_load_is_at_the_external_reference_plane(
    source: Literal["schematic", "layout"],
) -> None:
    """Changing the measurement termination must leave the on-chip line intact."""
    circuit, netlist = _circuit(source, "short")
    assert set(netlist["ports"]) == {"o1", "o2", "o3", "o4"}
    target = (
        "launcher_gate_a,waveport" if source == "schematic" else "launcher,waveport"
    )
    assert netlist["connections"][target] == "load_gate_a,o1"
    result = circuit(f=jnp.linspace(4e9, 6e9, 101))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in result.values())


def test_standalone_path_without_gdsfactoryplus() -> None:
    """Both sources work when the optional editor package cannot be imported."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['gdsfactoryplus'] = None; "
                "from models.gatemon_test_chip import _circuit; "
                "assert all(len(_circuit(source)[0](f=5.06e9)) == 36 "
                "for source in ('schematic', 'layout'))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
