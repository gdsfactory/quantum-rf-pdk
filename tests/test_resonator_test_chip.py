"""Tests for resonator test-chip samples."""

from pathlib import Path

import gdsfactory as gf
import numpy as np
import sax
import yaml

from qpdk import PDK
from qpdk.models import models
from qpdk.samples.resonator_test_chip import resonator_test_chip_python

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


def _simulate_python_chip(frequencies: sax.FloatArrayLike) -> sax.SDict:
    """Evaluate the Python sample without a whole-chip model."""
    netlist = resonator_test_chip_python().get_netlist(on_dangling_port="ignore")
    return _simulate_netlist(netlist, frequencies)


def _simulate_yaml_chip(frequencies: sax.FloatArrayLike) -> sax.SDict:
    """Evaluate the declarative sample without a whole-chip model."""
    document = yaml.safe_load(YAML_SAMPLE.read_text())
    netlist = {key: document[key] for key in ("instances", "connections", "ports")}
    return _simulate_netlist(netlist, frequencies)


def test_resonator_test_chip_resolves_from_the_cells_module() -> None:
    """Register the sample through the normal PDK cell collection."""
    PDK.activate()

    component = gf.get_component("resonator_test_chip_python")
    schematic = resonator_test_chip_python.schematic_function()

    assert component.function_name == "resonator_test_chip_python"
    assert schematic.info["models"] == []
    assert "resonator_test_chip_python" not in models


def test_resonator_test_chip_exposes_launcher_waveports() -> None:
    """Expose both ends of both resonator-test-chip probelines."""
    component = resonator_test_chip_python()

    assert {port.name for port in component.ports} == {"o1", "o2", "o3", "o4"}
    assert component.ports["o1"].center == (0.0, 1000.0)
    assert component.ports["o2"].center == (9000.0, 1000.0)
    assert component.ports["o3"].center == (0.0, 0.0)
    assert component.ports["o4"].center == (9000.0, 0.0)


def test_resonator_test_chip_has_readable_instance_names() -> None:
    """Keep the extracted schematic stable and human-readable."""
    netlist = resonator_test_chip_python().get_netlist(on_dangling_port="ignore")
    expected_names = {
        f"resonator_{probeline}_{index}"
        for probeline in ("bot", "top")
        for index in range(1, 9)
    }
    expected_names |= {
        f"probeline_straight_{probeline}_{index}"
        for probeline in ("bot", "top")
        for index in range(1, 8)
    }
    expected_names |= {
        f"{instance}_{side}_{probeline}"
        for instance in ("probe", "probeline_sbend", "probeline_straight")
        for side in ("west", "east")
        for probeline in ("bot", "top")
    }

    assert set(netlist["instances"]) == expected_names


def test_resonator_test_chip_yaml_netlist_matches_python() -> None:
    """Keep the checked-in netlist in sync with the Python chip.

    The ``.pic.yml`` hardcodes what ``resonator_test_chip_python`` derives, so
    without this check changing the Python netlist silently diverges the
    sample. Only stdlib ``yaml`` is used, so the file stays covered in the
    non-gfp test runs.
    """
    document = yaml.safe_load(YAML_SAMPLE.read_text())
    netlist = resonator_test_chip_python().get_netlist(on_dangling_port="ignore")

    assert set(document["instances"]) == set(netlist["instances"])
    for name, instance in netlist["instances"].items():
        assert document["instances"][name]["component"] == instance["component"]
        # YAML turns tuples into lists, e.g. "size: (100, 0)" -> [100, 0].
        settings = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in instance["settings"].items()
        }
        assert document["instances"][name]["settings"] == settings
        assert document["instances"][name]["info"] == instance["info"]

    assert document["ports"] == netlist["ports"]

    yaml_nets = {
        frozenset(connection) for connection in document["connections"].items()
    }
    python_nets = {frozenset((net["p1"], net["p2"])) for net in netlist["nets"]}
    assert len(document["connections"]) == len(netlist["nets"])
    assert yaml_nets == python_nets


def test_resonator_test_chip_yaml_matches_python() -> None:
    """Materialize the YAML netlist and match the Python chip exactly."""
    PDK.activate()
    python_component = resonator_test_chip_python()
    yaml_component = gf.read.from_yaml(
        YAML_SAMPLE,
        name="resonator_test_chip_yaml_parity",
        label_instance_function=lambda **_kwargs: None,
    )

    assert {port.name for port in yaml_component.ports} == {
        port.name for port in python_component.ports
    }
    for name in ("o1", "o2", "o3", "o4"):
        yaml_port = yaml_component.ports[name]
        python_port = python_component.ports[name]
        assert yaml_port.center == python_port.center
        assert yaml_port.orientation == python_port.orientation
        assert yaml_port.width == python_port.width

    assert yaml_component.dbbox() == python_component.dbbox()
    assert set(yaml_component.layers) == set(python_component.layers)
    for layer in python_component.layers:
        layer_index = gf.get_layer(layer)
        python_region = gf.kdb.Region(python_component.begin_shapes_rec(layer_index))
        yaml_region = gf.kdb.Region(yaml_component.begin_shapes_rec(layer_index))

        assert (python_region ^ yaml_region).is_empty()


def test_resonator_test_chip_uses_registered_cross_sections() -> None:
    """Keep serialized SAX settings resolvable by the active PDK."""
    component = resonator_test_chip_python()
    netlist = component.get_netlist(on_dangling_port="ignore")
    resonators = [
        instance
        for instance in netlist["instances"].values()
        if instance["component"] == "quarter_wave_resonator_coupled"
    ]

    assert len(resonators) == 16
    assert {instance["settings"]["cross_section"] for instance in resonators} == {
        "coplanar_waveguide"
    }
    assert {
        instance["settings"]["cross_section_non_resonator"] for instance in resonators
    } == {"coplanar_waveguide"}
    assert len({instance["settings"]["length"] for instance in resonators}) == 16


def test_recursive_sax_netlist_builds_without_cross_section_shadowing() -> None:
    """Exercise recursive construction without claiming physical equivalence.

    Capacitive coupling requires the registered resonator leaf models; this
    low-level build only guards against unresolvable cross-section metadata.
    """
    PDK.activate()
    netlist = resonator_test_chip_python().get_netlist(
        recursive=True,
        on_dangling_port="ignore",
    )

    circuit, _ = sax.circuit(
        netlist,
        models=models,
        ignore_impossible_connections=False,
    )
    s_params = circuit(f=[7e9])

    assert {port for key in s_params for port in key} == {"o1", "o2", "o3", "o4"}


def test_python_and_yaml_chip_simulations_match() -> None:
    """Solve both chip definitions recursively from the same leaf models."""
    PDK.activate()
    frequencies = np.linspace(4e9, 10e9, 31)
    actual = _simulate_python_chip(frequencies)
    expected = _simulate_yaml_chip(frequencies)
    zero = np.zeros_like(frequencies, dtype=complex)

    for key in actual.keys() | expected.keys():
        np.testing.assert_allclose(
            actual.get(key, zero),
            expected.get(key, zero),
            rtol=1e-10,
            atol=1e-12,
        )


def test_resonator_test_chip_sax_model_is_reciprocal_and_passive() -> None:
    """Check basic physical constraints across the intended RF band."""
    frequencies = np.linspace(4e9, 10e9, 101)
    s_params = _simulate_python_chip(frequencies)
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
    """Regression for issue #678: each probeline must show distinct resonances.

    Exercises the extracted chip netlist directly, so a hierarchy-flattening
    regression that collapses the 16 distinct resonator lengths back into one
    merged resonance is caught without a whole-chip model.
    """
    frequencies = np.linspace(4e9, 10e9, 4001)
    s_params = _simulate_python_chip(frequencies)

    for probe_ports in (("o1", "o2"), ("o3", "o4")):
        s21_db = 20 * np.log10(np.abs(np.asarray(s_params[probe_ports])) + 1e-30)
        # Resonance dips: local minima below a threshold (<= on one side is
        # robust to flat minima from finite floating-point precision).
        dips = [
            index
            for index in range(1, len(s21_db) - 1)
            if s21_db[index] <= s21_db[index - 1]
            and s21_db[index] < s21_db[index + 1]
            and s21_db[index] < -0.3
        ]
        # Eight resonators per probeline; all should resolve distinctly.
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
