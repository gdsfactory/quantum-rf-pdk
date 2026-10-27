"""Build the gatemon schematic through GF+ and compare its hybrid layers."""

import json
from collections.abc import Iterator
from pathlib import Path

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import import_gfp_module
from models.gatemon_test_chip import _circuit

from qpdk import PDK
from qpdk.models.constants import c_0
from qpdk.tech import LAYER

converter = import_gfp_module("gdsfactoryplus.nyancir_to_dschematic")
materializer = import_gfp_module("gdsfactoryplus.dschematic_to_gds")
sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
xs_default = import_gfp_module("gdsfactoryplus.xs_default")
SCHEMATIC = Path(__file__).resolve().parents[2] / "qpdk/samples/gatemon_test_chip.gsch"


@pytest.fixture(autouse=True)
def _restore_layout_cache() -> Iterator[None]:
    """GFP renames existing top cells during layout extraction."""
    before = {cell.cell_index(): cell.name for cell in gf.kcl.layout.each_cell()}
    yield
    for index, name in before.items():
        cell = gf.kcl.layout.cell(index)
        if cell is None or cell.name == name:
            continue
        replacement = gf.kcl.layout.cell(name)
        if replacement is not None:
            replacement.name = gf.kcl.layout.unique_cell_name(f"{name}_gfp_test")
        cell.name = name


@pytest.mark.gfp
def test_gfp_default_cross_section_is_json_serializable() -> None:
    """The layer RPC must return a named cross-section instead of a callable."""
    value = xs_default.infer_default_cross_section(PDK, "qpdk.PDK")
    assert value == "coplanar_waveguide"
    assert json.loads(json.dumps({"inferred_default_cross_section": value})) == {
        "inferred_default_cross_section": "coplanar_waveguide"
    }


@pytest.mark.gfp
def test_gatemon_schematic_builds_with_fallback_geometry() -> None:
    """Routes build and both mirrored hybrid devices agree with the YAML layout."""
    PDK.activate()
    converted = converter.nyancir_to_dschematic(str(SCHEMATIC), "qpdk.PDK")
    cell, warnings = materializer.materialize_dschematic(
        converted.top,
        kcl=converted.kcl,
        schem_name="gatemon_schematic_parity",
        factories=converted.factories,
    )
    assert warnings == []
    assert {port.name for port in cell.ports} == {
        "o1",
        "o2",
        "o3",
        "o4",
        "gate_a",
        "gate_b",
    }
    fallback = gf.get_component("gatemon_test_chip")
    for port in fallback.ports:
        assert cell.ports[port.name].center == pytest.approx(port.center)
    for layer in (
        LAYER.M1_DRAW,
        LAYER.M1_ETCH,
        LAYER.CNT_CONTACT,
        LAYER.HBN,
        LAYER.CNT,
        LAYER.TOP_GATE,
        LAYER.JJ_PATCH,
    ):
        index = cell.kcl.layout.layer(*layer)
        actual = gf.kdb.Region(cell.begin_shapes_rec(index))
        expected = fallback.get_region(layer)
        assert not actual.is_empty(), layer
        assert (actual ^ expected).is_empty(), layer
    assert gf.kdb.Region(
        cell.begin_shapes_rec(cell.kcl.layout.layer(*LAYER.ERROR_PATH))
    ).is_empty()


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["mosaic", "layout"])
def test_gatemon_schematic_simulates_capacitive_readout(
    source: str, tmp_path: Path
) -> None:
    """Both editor simulation paths must include the island coupling branch."""
    if source == "mosaic":
        document = json.loads(SCHEMATIC.read_text())
        path = tmp_path / SCHEMATIC.name
        path.write_text(json.dumps(document))
        ids = {entry["model"] for entry in document.values() if "model" in entry}
        (tmp_path / "models.nyanlib").write_text(
            json.dumps({
                "models:" + name: {
                    "name": name.rsplit(".", 1)[-1],
                    "type": "ckt",
                    "tags": ["qpdk"],
                }
                for name in ids
            })
        )
        info = sax_sim.inspect_mosaic_sax_models(str(path), "qpdk.PDK", str(tmp_path))
        simulate = sax_sim.simulate_mosaic_sax
        kwargs = {"project_root": str(tmp_path)}
    else:
        path = SCHEMATIC
        info = sax_sim.inspect_layout_sax_models(str(path), "qpdk.PDK")
        simulate = sax_sim.simulate_layout_sax
        kwargs = {}
    assert info["missing_models"] == []
    assert {"gatemon", "gatemon_coupler"} <= set(info["required_models"])
    result = simulate(
        str(path),
        "qpdk.PDK",
        wl_min=299792458e6 / 5.09e9,
        wl_max=299792458e6 / 5.02e9,
        wl_num=101,
        sweep_axis="f",
        **kwargs,
    )
    circuit, _ = _circuit("schematic" if source == "mosaic" else "layout")
    frequencies = c_0 * 1e6 / jnp.asarray(result["wavelengths"])
    standalone = circuit(f=frequencies)
    assert set(result["sdict"]) == {f"{a},{b}" for a, b in standalone}
    for (a, b), expected in standalone.items():
        entry = result["sdict"][f"{a},{b}"]
        actual = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
    for gate, rf in [("gate_a", "o1"), ("gate_b", "o3")]:
        entry = result["sdict"][f"{gate},{rf}"]
        values = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        assert bool(jnp.all(jnp.isfinite(values)))
        assert float(jnp.max(jnp.abs(values))) > 1e-5
