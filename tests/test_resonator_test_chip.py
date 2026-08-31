"""Tests for resonator test-chip samples."""

from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest
import sax

from qpdk import PDK
from qpdk.models import models
from qpdk.models.resonator import (
    resonator_test_chip_python as resonator_test_chip_python_model,
    resonator_test_chip_yaml,
)
from qpdk.samples.resonator_test_chip import resonator_test_chip_python

YAML_SAMPLE = (
    Path(__file__).parents[1] / "qpdk/samples/resonator_test_chip_yaml.pic.yml"
)


@pytest.mark.parametrize(
    "component_name",
    [
        "resonator_test_chip_python",
        "qpdk.samples.resonator_test_chip.resonator_test_chip_python",
    ],
)
def test_resonator_test_chip_resolves_in_active_pdk(component_name: str) -> None:
    """Resolve and directly simulate both identifiers used by the editor."""
    PDK.activate()

    component = gf.get_component(component_name)

    assert component.function_name == "resonator_test_chip_python"
    assert resonator_test_chip_python.schematic_function is not None
    assert PDK.models is not None
    s_params = PDK.models[component_name](f=[7e9])
    assert {port for key in s_params for port in key} == {"o1", "o2", "o3", "o4"}


def test_resonator_test_chip_runs_through_layout_simulation_server() -> None:
    """Exercise the editor's layout-driven SAX path without a .gsch workaround."""
    sax_server = pytest.importorskip("gdsfactoryplus.serve.sax")
    PDK.activate()
    result = sax_server._run_simulation(
        "qpdk.samples.resonator_test_chip.resonator_test_chip_python",
        {
            # The server API represents 4--10 GHz as wavelengths in µm.
            "wl_min": 299_792.458 / 10,
            "wl_max": 299_792.458 / 4,
            "wl_num": 3,
            "layout": {},
        },
    )

    assert len(result["wavelengths"]) == 3
    assert set(result["sdict"]) == {"o1", "o2", "o3", "o4"}


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


def test_resonator_test_chip_yaml_matches_python() -> None:
    """Keep the YAML as a visible wrapper with identical Python geometry."""
    python_component = resonator_test_chip_python()
    yaml_component = gf.read.from_yaml(
        YAML_SAMPLE,
        name="resonator_test_chip_yaml_parity",
        label_instance_function=lambda **_kwargs: None,
    )

    yaml_netlist = yaml_component.get_netlist(on_dangling_port="ignore")
    assert set(yaml_netlist["instances"]) == {"resonator_test_chip"}
    assert yaml_netlist["instances"]["resonator_test_chip"]["component"] == (
        "resonator_test_chip_python"
    )
    assert yaml_netlist["ports"] == {
        port_name: f"resonator_test_chip,{port_name}"
        for port_name in ("o1", "o2", "o3", "o4")
    }
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

    # This direct netlist keeps the sample factory name as its top-level
    # circuit name. Exclude the public top-level model to avoid SAX treating
    # that circuit itself as a model. The app netlist uses top name ``t`` and
    # therefore exercises the registered sample model.
    simulation_models = {
        name: model
        for name, model in models.items()
        if name != "resonator_test_chip_python"
    }

    sax.circuit(
        netlist,
        models=simulation_models,
        ignore_impossible_connections=False,
    )


def test_resonator_test_chip_sax_model_matches_physical_netlist() -> None:
    """Keep analytical launcher and route parameters aligned with the layout."""
    PDK.activate()
    frequencies = np.linspace(4e9, 10e9, 31)
    netlist = resonator_test_chip_python().get_netlist(
        on_dangling_port="ignore",
    )
    circuit, _ = sax.circuit(
        netlist,
        models=models,
        ignore_impossible_connections=False,
    )
    actual = circuit(f=frequencies)
    expected = resonator_test_chip_python_model(f=frequencies)
    zero = np.zeros_like(frequencies, dtype=complex)

    for key in actual.keys() | expected.keys():
        np.testing.assert_allclose(
            actual.get(key, zero),
            expected.get(key, zero),
            rtol=1e-10,
            atol=1e-12,
        )


def test_resonator_test_chip_can_be_placed_and_simulated() -> None:
    """Simulate the chip as an instance inside a schematic."""
    PDK.activate()
    schematic = gf.Component("placed_resonator_test_chip")
    chip = schematic.add_ref(
        gf.get_component("resonator_test_chip_python"),
        name="resonator_test_chip_python",
    )
    schematic.add_ports(chip.ports)
    netlist = schematic.get_netlist(recursive=True, on_dangling_port="ignore")

    circuit, _ = sax.circuit(
        netlist,
        models=models,
        ignore_impossible_connections=False,
    )
    frequencies = np.linspace(4e9, 10e9, 31)
    s_params = circuit(f=frequencies)
    expected = resonator_test_chip_python_model(f=frequencies)

    assert {port for key in s_params for port in key} == {"o1", "o2", "o3", "o4"}
    zero = np.zeros_like(frequencies, dtype=complex)
    for key, value in s_params.items():
        np.testing.assert_allclose(value, expected.get(key, zero), rtol=0, atol=0)


def test_resonator_test_chip_sax_model_is_reciprocal_and_passive() -> None:
    """Check basic physical constraints across the intended RF band."""
    frequencies = np.linspace(4e9, 10e9, 101)
    s_params = resonator_test_chip_python_model(f=frequencies)
    port_names = ("o1", "o2", "o3", "o4")
    matrix = np.zeros((len(frequencies), 4, 4), dtype=complex)

    for row, output_port in enumerate(port_names):
        for column, input_port in enumerate(port_names):
            value = s_params.get((output_port, input_port))
            if value is not None:
                matrix[:, row, column] = value

    np.testing.assert_allclose(matrix, matrix.transpose(0, 2, 1), atol=1e-12)
    assert np.linalg.svd(matrix, compute_uv=False).max() <= 1 + 1e-6


def test_resonator_test_chip_yaml_has_top_level_sax_model() -> None:
    """Keep the YAML sample usable by legacy recursive-netlist simulation."""
    assert models["resonator_test_chip_yaml"] is resonator_test_chip_yaml

    s_params = resonator_test_chip_yaml(f=[7e9])

    # Two probelines are independent, so SAX returns four entries per line.
    assert len(s_params) == 8
    assert {port for key in s_params for port in key} == {
        "o1",
        "o2",
        "o3",
        "o4",
    }
