"""Tests for resonator test-chip samples."""

import sax

from qpdk import PDK
from qpdk.models import models
from qpdk.models.resonator import resonator_test_chip_yaml
from qpdk.samples.resonator_test_chip import resonator_test_chip_python


def test_resonator_test_chip_exposes_launcher_waveports() -> None:
    """Expose both ends of both resonator-test-chip probelines."""
    component = resonator_test_chip_python()

    assert {port.name for port in component.ports} == {"o1", "o2", "o3", "o4"}
    assert component.ports["o1"].center == (0.0, 1000.0)
    assert component.ports["o2"].center == (9000.0, 1000.0)
    assert component.ports["o3"].center == (0.0, 0.0)
    assert component.ports["o4"].center == (9000.0, 0.0)


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


def test_resonator_test_chip_recursive_sax_netlist_is_valid() -> None:
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
