"""Tests for resonator test-chip samples."""

import json
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np
import pytest
import sax
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.models import models
from qpdk.models.resonator import (
    resonator_test_chip_python as resonator_test_chip_python_model,
    resonator_test_chip_yaml,
)
from qpdk.samples.resonator_test_chip import resonator_test_chip_python

GSCH_SAMPLE = Path(__file__).parents[1] / "qpdk/samples/resonator_test_chip_yaml.gsch"

_SPEED_OF_LIGHT_UM_PER_S = 299_792_458_000_000.0


@pytest.fixture(scope="module")
def mosaic_project_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a self-contained project root for the Mosaic SAX pipeline.

    The pipeline resolves the ``.gsch`` and ``models.nyanlib`` through
    nyancad's ``FileAPI`` relative to ``project_root``. Pointing at the
    repository root would depend on the gitignored
    ``build/models.nyanlib``, which the ``gfp_server`` fixture deletes and
    regenerates on a separate xdist worker.

    Returns:
        Path to the temporary project root.
    """
    root = tmp_path_factory.mktemp("sample_mosaic")
    (root / GSCH_SAMPLE.name).write_text(GSCH_SAMPLE.read_text())
    nyanlib = {
        f"models:{qualname}": {"name": name, "type": "ckt", "tags": ["qpdk"]}
        for qualname, name in (
            ("qpdk.cells.launcher.launcher", "launcher"),
            (
                "qpdk.cells.resonator.quarter_wave_resonator_coupled",
                "quarter_wave_resonator_coupled",
            ),
            ("qpdk.cells.waveguides.straight", "straight"),
        )
    }
    (root / "models.nyanlib").write_text(json.dumps(nyanlib, indent=2))
    return root


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


@pytest.mark.gfp
def test_resonator_test_chip_yaml_matches_python() -> None:
    """Materialize the ``.gsch`` sample and match the Python chip exactly."""
    nyancir = import_gfp_module("gdsfactoryplus.nyancir_to_dschematic")
    dschematic = import_gfp_module("gdsfactoryplus.dschematic_to_gds")

    result = nyancir.nyancir_to_dschematic(str(GSCH_SAMPLE), "qpdk.PDK")
    cell, errors = dschematic.materialize_dschematic(
        result.top,
        kcl=result.kcl,
        schem_name=result.schem_name,
        factories=result.factories,
    )
    assert not errors, errors

    python_component = resonator_test_chip_python()

    assert {port.name for port in cell.ports} == {
        port.name for port in python_component.ports
    }
    dbu = cell.kcl.dbu
    for name in ("o1", "o2", "o3", "o4"):
        gsch_port = cell.ports[name]
        python_port = python_component.ports[name]
        assert not gsch_port.trans.is_mirror()
        assert (
            gsch_port.trans.disp.x * dbu,
            gsch_port.trans.disp.y * dbu,
        ) == tuple(python_port.center)
        assert gsch_port.angle * 90 == python_port.orientation
        # kf3 port width is already in microns, unlike trans.disp.
        assert gsch_port.width == python_port.width

    def layer_regions(component: Any) -> dict[Any, gf.kdb.Region]:
        layout = component.kcl.layout
        regions: dict[Any, gf.kdb.Region] = {}
        for layer_index in layout.layer_indexes():
            info = layout.get_info(layer_index)
            region = gf.kdb.Region(component.begin_shapes_rec(layer_index))
            if not region.is_empty():
                regions[info] = region
        return regions

    gsch_regions = layer_regions(cell)
    python_regions = layer_regions(python_component)
    assert set(gsch_regions) == set(python_regions)
    for info in gsch_regions:
        assert (gsch_regions[info] ^ python_regions[info]).is_empty(), str(info)


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
    expected = sax.sdict(sax.sdense(resonator_test_chip_python_model(f=frequencies)))

    assert {port for key in s_params for port in key} == {"o1", "o2", "o3", "o4"}
    assert s_params.keys() == expected.keys()
    for key in s_params:
        np.testing.assert_allclose(
            s_params[key],
            expected[key],
            rtol=1e-10,
            atol=1e-12,
        )


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
    """Keep the ``.gsch`` sample usable via its registered SAX model."""
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


@pytest.mark.gfp
def test_resonator_test_chip_yaml_mosaic_sax_solves(
    mosaic_project_root: Path,
) -> None:
    """Solve the SAX circuit gfp builds from the ``.gsch`` sample's nets."""
    gfp_sax = import_gfp_module("gdsfactoryplus.sim.sax")

    wl_num = 5
    result = gfp_sax.simulate_mosaic_sax(
        str(mosaic_project_root / GSCH_SAMPLE.name),
        "qpdk.PDK",
        # The 4--10 GHz design band, as wavelengths in µm.
        _SPEED_OF_LIGHT_UM_PER_S / 10e9,
        _SPEED_OF_LIGHT_UM_PER_S / 4e9,
        wl_num,
        str(mosaic_project_root),
        sweep_frequency=True,
    )

    # Each probeline couples only its own two launchers.
    expected_keys = {
        f"o{out},o{inp}"
        for out, inp in (
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 3),
            (3, 4),
            (4, 3),
            (4, 4),
        )
    }
    assert set(result["sdict"]) == expected_keys
    for value in result["sdict"].values():
        assert len(value["real"]) == wl_num
        assert len(value["imag"]) == wl_num
        assert np.isfinite(value["real"]).all()
        assert np.isfinite(value["imag"]).all()
