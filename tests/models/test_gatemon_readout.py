"""Check the loaded transmission resonators in the declarative gatemon sample."""

import json
from functools import cache
from inspect import signature
from pathlib import Path

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
import yaml

from qpdk.models.capacitor import interdigital_capacitor
from qpdk.models.constants import e, h, ε_0
from qpdk.models.gatemon import gatemon_coupler, gatemon_island
from qpdk.models.generic import short
from qpdk.models.waveguides import straight
from qpdk.tech import LAYER, get_gatemon_layer_stack, material_properties

SAMPLE = Path(__file__).parents[2] / "qpdk/samples/gatemon_test_chip.pic.yml"


@cache
def _readout_circuit(suffix: str, *, grounded_gate: bool = False) -> sax.SDictModel:
    document = yaml.safe_load(SAMPLE.read_text())
    instances = document["instances"]
    length = sum(
        gf.get_component(
            instances[name]["component"], **instances[name]["settings"]
        ).info["length"]
        for name in (f"resonator_{suffix}", f"tail_{suffix}")
    )
    netlist = {
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
    }
    if grounded_gate:
        netlist["instances"]["gate_ground"] = {"component": "short"}
        netlist["connections"]["qubit,gate"] = "gate_ground,o1"
        del netlist["ports"]["gate"]
    circuit, _ = sax.circuit(
        netlist,
        models={
            "capacitor": interdigital_capacitor,
            "coupling": gatemon_coupler,
            "line": straight,
            "qubit": gatemon_island,
            "short": short,
        },
        ignore_impossible_connections=False,
    )
    return circuit


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_loaded_readout_targets_reported_device_a_frequency(suffix: str) -> None:
    """Include both coupling capacitors and the complete CPW electrical length."""
    document = yaml.safe_load(SAMPLE.read_text())
    circuit = _readout_circuit(suffix, grounded_gate=True)
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


def test_nominal_readout_coupling_matches_device_a_target() -> None:
    """The minimum mode splitting is twice the linear coupling strength."""
    document = yaml.safe_load(SAMPLE.read_text())
    target = document["info"]["readout_target_hz"]
    frequencies = jnp.linspace(target - 150e6, target + 150e6, 6001)
    circuit = _readout_circuit("a", grounded_gate=True)
    response = jax.jit(
        jax.vmap(
            lambda transmission: circuit(
                f=frequencies, qubit={"transmissions": (transmission,)}
            )
        )
    )(jnp.linspace(0.97, 0.995, 11))
    power = jnp.abs(response["o2", "o1"]) ** 2
    midpoint = len(frequencies) // 2
    lower = frequencies[jnp.argmax(power[:, :midpoint], axis=1)]
    upper = frequencies[midpoint + 1 + jnp.argmax(power[:, midpoint + 1 :], axis=1)]
    coupling = float(jnp.min(upper - lower) / 2)
    assert coupling == pytest.approx(document["info"]["coupling_target_hz"], rel=0.05)


def test_nominal_gate_capacitances_include_contact_overlap() -> None:
    """The thin hBN overlap sets the gate-capacitance scale."""
    device = gf.get_component("gatemon")
    contacts = sorted(
        device.get_region(LAYER.CNT_CONTACT).merged().each(),
        key=lambda polygon: polygon.bbox().top,
    )
    assert len(contacts) == 2
    gate = device.get_region(LAYER.TOP_GATE)
    hbn = device.get_region(LAYER.HBN)
    thickness = get_gatemon_layer_stack().layers["hBN"].thickness * 1e-6
    permittivity = material_properties["hBN"]["relative_permittivity"]
    overlaps = [
        (gf.kdb.Region(contact) & gate & hbn).area() * device.kcl.dbu**2 * 1e-12
        for contact in contacts
    ]
    local = [ε_0 * permittivity * area / thickness for area in overlaps]
    evidence = json.loads(
        (
            SAMPLE.parent.parent
            / "models/datasets/data/gatemon_palace.convergence.json"
        ).read_text()
    )
    matrix = evidence["cases"]["nominal"]["profiles"]["fine"]["maxwell_capacitance_f"]
    defaults = signature(gatemon_island).parameters
    mutual = defaults["gate_capacitance"].default
    assert mutual == pytest.approx(local[1] - matrix[0][1], rel=0.1)
    assert defaults["gate_ground_capacitance"].default == pytest.approx(
        local[0] + sum(matrix[1]), rel=0.1
    )
    readout = signature(gatemon_coupler).parameters["coupling_capacitance"].default
    total = defaults["capacitance"].default + mutual + readout
    target = yaml.safe_load(SAMPLE.read_text())["info"]["charging_energy_target_hz"]
    assert e**2 / (2 * total * h) == pytest.approx(target, rel=0.01)


def test_reflective_gate_avoids_matched_gate_loss() -> None:
    """A gate filter must be explicit when comparing readout to the paper."""
    frequencies = jnp.linspace(4.98e9, 5.13e9, 6001)
    grounded = _readout_circuit("a", grounded_gate=True)(
        f=frequencies, qubit={"transmissions": (0.9825,)}
    )
    matched = _readout_circuit("a")(f=frequencies, qubit={"transmissions": (0.9825,)})
    grounded_power = float(jnp.max(jnp.abs(grounded["o2", "o1"]) ** 2))
    matched_power = float(jnp.max(jnp.abs(matched["o2", "o1"]) ** 2))
    assert grounded_power > 0.8
    assert grounded_power > 10 * matched_power


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
