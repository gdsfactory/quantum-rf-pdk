"""Tests for resonator test-chip samples."""

from pathlib import Path

import numpy as np
import sax
import yaml

from qpdk.models import models

YAML_SAMPLE = (
    Path(__file__).parents[1] / "qpdk/samples/resonator_test_chip_yaml.pic.yml"
)


def _simulate_netlist(
    netlist: dict,
    frequencies: sax.FloatArrayLike,
) -> sax.SDict:
    """Evaluate a chip netlist from the registered component models."""
    circuit, _ = sax.circuit(
        netlist,
        models=models,
        ignore_impossible_connections=False,
    )
    return circuit(f=frequencies)


def _simulate_yaml_chip(frequencies: sax.FloatArrayLike) -> sax.SDict:
    """Evaluate the declarative sample without a whole-chip model."""
    document = yaml.safe_load(YAML_SAMPLE.read_text())
    netlist = {key: document[key] for key in ("instances", "connections", "ports")}
    return _simulate_netlist(netlist, frequencies)


def test_resonator_test_chip_sax_model_is_reciprocal_and_passive() -> None:
    """Check basic physical constraints across the intended RF band."""
    frequencies = np.linspace(4e9, 10e9, 101)
    s_params = _simulate_yaml_chip(frequencies)
    port_names = ("o1", "o2", "o3", "o4")
    matrix = np.zeros((len(frequencies), 4, 4), dtype=complex)

    for row, output_port in enumerate(port_names):
        for column, input_port in enumerate(port_names):
            value = s_params.get((output_port, input_port))
            if value is not None:
                matrix[:, row, column] = value

    np.testing.assert_allclose(matrix, matrix.transpose(0, 2, 1), atol=1e-12)
    assert np.linalg.svd(matrix, compute_uv=False).max() <= 1 + 1e-6


def test_resonator_test_chip_has_distinct_resonances() -> None:
    """Keep all eight designed resonances distinct on each probeline."""
    frequencies = np.linspace(4e9, 10e9, 4001)
    s_params = _simulate_yaml_chip(frequencies)

    for probe_ports in (("o1", "o2"), ("o3", "o4")):
        s21_db = 20 * np.log10(np.abs(np.asarray(s_params[probe_ports])) + 1e-30)
        dips = [
            index
            for index in range(1, len(s21_db) - 1)
            if s21_db[index] <= s21_db[index - 1]
            and s21_db[index] < s21_db[index + 1]
            and s21_db[index] < -0.3
        ]
        assert len(dips) == 8, (
            f"Probeline {probe_ports}: expected 8 distinct resonances, got {len(dips)}"
        )


def test_resonator_test_chip_yaml_simulates_from_leaf_models() -> None:
    """Keep the YAML sample usable without a registered chip model."""
    s_params = _simulate_yaml_chip([7e9])

    # sax.circuit emits the full 4x4 key set; the cross-probeline entries
    # (e.g. ("o1", "o3")) are zero because the two probelines are independent.
    assert len(s_params) == 16
    assert {port for key in s_params for port in key} == {
        "o1",
        "o2",
        "o3",
        "o4",
    }
    top_line = {"o1", "o2"}
    bottom_line = {"o3", "o4"}
    for (port_a, port_b), raw_value in s_params.items():
        value = np.asarray(raw_value)
        if not ({port_a, port_b} <= top_line or {port_a, port_b} <= bottom_line):
            np.testing.assert_allclose(value, 0.0, atol=1e-9)

    for key in (("o1", "o2"), ("o2", "o1"), ("o3", "o4"), ("o4", "o3")):
        assert np.abs(np.asarray(s_params[key])).min() > 0.9
