"""Check the loaded transmission resonators in the declarative gatemon sample."""

from functools import cache
from pathlib import Path

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
import yaml

from qpdk.models.capacitor import interdigital_capacitor
from qpdk.models.gatemon import gatemon_coupler, gatemon_island
from qpdk.models.waveguides import straight

SAMPLE = Path(__file__).parents[2] / "qpdk/samples/gatemon_test_chip.pic.yml"


@cache
def _readout_circuit(suffix: str) -> sax.SDictModel:
    document = yaml.safe_load(SAMPLE.read_text())
    instances = document["instances"]
    length = sum(
        gf.get_component(
            instances[name]["component"], **instances[name]["settings"]
        ).info["length"]
        for name in (f"resonator_{suffix}", f"tail_{suffix}")
    )
    circuit, _ = sax.circuit(
        {
            "instances": {
                "input": {
                    "component": "capacitor",
                    "settings": instances[f"cap_in_{suffix}"]["settings"],
                },
                "coupling": {
                    "component": "coupling",
                    "settings": {
                        "length": instances[f"coupler_{suffix}"]["settings"]["length"]
                    },
                },
                "line": {"component": "line", "settings": {"length": length}},
                "output": {
                    "component": "capacitor",
                    "settings": instances[f"cap_out_{suffix}"]["settings"],
                },
                "qubit": {"component": "qubit"},
            },
            "connections": {
                "input,o2": "coupling,o1",
                "coupling,o2": "line,o1",
                "line,o2": "output,o1",
                "coupling,qubit": "qubit,readout",
            },
            "ports": {"o1": "input,o1", "o2": "output,o2", "gate": "qubit,gate"},
        },
        models={
            "capacitor": interdigital_capacitor,
            "coupling": gatemon_coupler,
            "line": straight,
            "qubit": gatemon_island,
        },
        ignore_impossible_connections=False,
    )
    return circuit


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_loaded_readout_targets_reported_device_a_frequency(suffix: str) -> None:
    """Include both coupling capacitors and the complete CPW electrical length."""
    document = yaml.safe_load(SAMPLE.read_text())
    circuit = _readout_circuit(suffix)
    target = document["info"]["readout_target_hz"]
    frequencies = jnp.linspace(target - 5e6, target + 5e6, 2001)
    power = (
        jnp.abs(circuit(f=frequencies, qubit={"transmissions": (0.0,)})["o2", "o1"])
        ** 2
    )
    peak = int(jnp.argmax(power))
    assert 0 < peak < len(frequencies) - 1
    assert float(frequencies[peak]) == pytest.approx(5.056e9, abs=2e4)
    assert float(power[peak]) > 0.8
    assert float(power[peak]) > 10 * float(jnp.maximum(power[0], power[-1]))


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_qubit_changes_the_readout_response(suffix: str) -> None:
    """The drawn capacitive branch must affect RF transmission and gate transfer."""
    circuit = _readout_circuit(suffix)
    f = jnp.linspace(5.02e9, 5.09e9, 1401)
    off = circuit(f=f, qubit={"transmissions": (0.0,)})
    on = circuit(f=f, qubit={"transmissions": (0.8,)})
    peaks = [float(f[jnp.argmax(jnp.abs(s["o2", "o1"]) ** 2)]) for s in (off, on)]
    assert abs(peaks[0] - peaks[1]) > 1e6
    assert float(jnp.max(jnp.abs(on["gate", "o1"]))) > 1e-5
    decoupled = circuit(f=f, coupling={"coupling_capacitance": 0.0})
    assert float(jnp.max(jnp.abs(decoupled["gate", "o1"]))) < 1e-12


@pytest.mark.parametrize("capacitance", [0.0, 5.4e-15])
def test_capacitive_coupler_is_passive_reciprocal_and_jittable(
    capacitance: float,
) -> None:
    f = jnp.linspace(1e9, 10e9, 21)
    result = jax.jit(lambda c: gatemon_coupler(f=f, coupling_capacitance=c))(
        capacitance
    )
    ports = ("o1", "o2", "qubit")
    matrix = jnp.stack(
        [jnp.stack([result[a, b] for b in ports], -1) for a in ports], -2
    )
    assert bool(jnp.all(jnp.isfinite(matrix)))
    np.testing.assert_allclose(matrix, jnp.swapaxes(matrix, -1, -2), atol=1e-12)
    assert float(jnp.max(jnp.linalg.svd(matrix, compute_uv=False))) <= 1 + 1e-10
    if capacitance == 0:
        np.testing.assert_allclose(
            result["o2", "o1"], straight(f=f, length=600)["o2", "o1"], atol=1e-10
        )
        np.testing.assert_allclose(result["qubit", "qubit"], 1, atol=1e-12)
