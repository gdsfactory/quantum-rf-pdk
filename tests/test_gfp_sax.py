"""gdsfactoryplus v2 SAX integration tests for the real test-chip schematic.

Both simulation paths must compose the chip from component models. The chip
itself intentionally has no SAX model: unmodeled assemblies expand until a
modeled component boundary, such as ``quarter_wave_resonator_coupled``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import sax
import yaml
from conftest import import_gfp_module
from hypothesis import given, settings, strategies as st

from qpdk import PDK
from qpdk.models import _PDK_MODEL_OVERRIDES, models

sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
factory_metadata = import_gfp_module("gdsfactoryplus.factory_metadata")

if TYPE_CHECKING:
    from collections.abc import Callable

_SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
_GSCH_PATH = _SAMPLE_DIR / "resonator_test_chip_yaml.gsch"
_PIC_YAML_PATH = _SAMPLE_DIR / "resonator_test_chip_yaml.pic.yml"
_SPEED_OF_LIGHT_UM_PER_S = 299_792_458_000_000.0
_F_MIN_HZ = 4e9
_F_MAX_HZ = 10e9

_COUPLING_MODEL_BOUNDARIES = {
    "quarter_wave_resonator_coupled",
}


@pytest.mark.gfp
@pytest.mark.parametrize(
    "component",
    ["open", "short", "straight_open", "straight_shorted"],
)
def test_one_port_termination_models_resolve(component: str) -> None:
    """Resolve each termination as a one-port model through GFP metadata."""
    model = factory_metadata.resolve_factory_model(component, "sax")

    assert model is not None
    assert set(model(f=np.asarray([5e9]))) == {("o1", "o1")}


@pytest.mark.gfp
def test_metadata_resolved_models_model_their_layout_ports() -> None:
    """Models resolved through v2 metadata must match their layout cell ports.

    ``resolve_factory_model`` returns ``None`` for cells gdsfactoryplus has no
    SAX binding for, so only the names it does resolve are compared. Those are
    the names whose schematic declares a model.
    """
    resolved: set[str] = set()

    for name in sorted(PDK.cells.keys() & PDK.models.keys()):
        model = factory_metadata.resolve_factory_model(name, "sax")
        if model is None:
            continue
        resolved.add(name)

        layout_ports = {
            port.name
            for port in PDK.cells[name]().ports
            if port.port_type != "placement"
        }
        s_params = model(f=np.asarray([5e9]))
        model_ports = {port for pair in s_params for port in pair}

        assert model_ports == layout_ports, (
            f"{name}: model ports {sorted(model_ports)} do not match layout "
            f"ports {sorted(layout_ports)}"
        )

    unresolved_overrides = set(_PDK_MODEL_OVERRIDES) - resolved
    assert not unresolved_overrides, (
        "v2 metadata did not resolve PDK model overrides: "
        f"{sorted(unresolved_overrides)}"
    )


@cache
def _reference_chip_circuit() -> Callable[..., sax.SDict]:
    """Build the flat declarative chip circuit from registered leaf models."""
    document = yaml.safe_load(_PIC_YAML_PATH.read_text())
    netlist = {key: document[key] for key in ("instances", "connections", "ports")}
    circuit, _ = sax.circuit(
        netlist,
        models=models,
        ignore_impossible_connections=False,
    )
    return circuit


def _assert_model_boundaries(info: dict[str, Any]) -> None:
    """Require coupled resonators to remain leaves during chip expansion."""
    required = set(info["required_models"])

    assert info["missing_models"] == []
    assert required >= _COUPLING_MODEL_BOUNDARIES


def _assert_sweep_matches_reference(result: dict[str, Any]) -> None:
    """Compare a gdsfactoryplus sweep with the flat leaf-model circuit."""
    wavelengths = np.asarray(result["wavelengths"], dtype=float)
    assert len(wavelengths) == 3
    assert wavelengths[0] == pytest.approx(299_792.458 / 4)
    assert wavelengths[-1] == pytest.approx(299_792.458 / 10)

    sdict = result["sdict"]
    ports = {name for key in sdict for name in key.split(",")}
    assert ports == {"o1", "o2", "o3", "o4"}

    frequencies = _SPEED_OF_LIGHT_UM_PER_S / wavelengths
    expected = _reference_chip_circuit()(f=frequencies)
    zero = np.zeros(len(wavelengths), dtype=complex)
    simulated_keys = {tuple(key.split(",")) for key in sdict}

    for key in set(expected) | simulated_keys:
        entry = sdict.get(f"{key[0]},{key[1]}")
        expected_value = expected.get(key, zero)
        if entry is None:
            assert np.abs(np.asarray(expected_value)).max() < 1e-5, (
                f"S[{key}] is missing but is not negligible in the reference"
            )
            continue

        actual = np.asarray(entry["real"]) + 1j * np.asarray(entry["imag"])
        np.testing.assert_allclose(
            actual,
            expected_value,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"S-parameters disagree for {key}",
        )


def _write_sample_mosaic_project(root: Path) -> Path:
    """Copy the sample with the minimal model library its topology needs."""
    document = json.loads(_GSCH_PATH.read_text())
    schematic_path = root / _GSCH_PATH.name
    schematic_path.write_text(json.dumps(document, indent=2))
    model_ids = sorted({
        device["model"]
        for device in document.values()
        if isinstance(device, dict) and device.get("model")
    })
    (root / "models.nyanlib").write_text(
        json.dumps(
            {
                f"models:{model_id}": {
                    "name": model_id.rsplit(".", maxsplit=1)[-1],
                    "type": "ckt",
                    "tags": ["qpdk"],
                }
                for model_id in model_ids
            },
            indent=2,
        )
    )
    return schematic_path


@pytest.mark.gfp
@settings(max_examples=10, deadline=None)
@given(
    frequency=st.floats(min_value=1e9, max_value=12e9),
    length=st.floats(min_value=1.0, max_value=50_000.0),
)
def test_straight_descriptor_preserves_nondefault_settings(
    frequency: float,
    length: float,
) -> None:
    """Resolve the primitive model through v2 metadata without losing settings."""
    model = factory_metadata.resolve_factory_model("straight", "sax")

    assert model is not None
    settings = {
        "f": np.array([frequency]),
        "length": length,
        "cross_section": "coplanar_waveguide",
    }
    actual = model(**settings)
    expected = PDK.models["straight"](**settings)

    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["mosaic", "layout"])
def test_resonator_test_chip_sax_simulation(source: str, tmp_path: Path) -> None:
    """Match both simulation sources to the same component-level circuit."""
    if source == "mosaic":
        schematic_path = _write_sample_mosaic_project(tmp_path)
        info = sax_sim.inspect_mosaic_sax_models(
            str(schematic_path),
            "qpdk.PDK",
            str(tmp_path),
        )
        result = sax_sim.simulate_mosaic_sax(
            str(schematic_path),
            "qpdk.PDK",
            wl_min=_SPEED_OF_LIGHT_UM_PER_S / _F_MAX_HZ,
            wl_max=_SPEED_OF_LIGHT_UM_PER_S / _F_MIN_HZ,
            wl_num=3,
            project_root=str(tmp_path),
            sweep_frequency=True,
        )
    else:
        info = sax_sim.inspect_layout_sax_models(str(_GSCH_PATH), "qpdk.PDK")
        result = sax_sim.simulate_layout_sax(
            str(_GSCH_PATH),
            "qpdk.PDK",
            wl_min=_SPEED_OF_LIGHT_UM_PER_S / _F_MAX_HZ,
            wl_max=_SPEED_OF_LIGHT_UM_PER_S / _F_MIN_HZ,
            wl_num=3,
            sweep_frequency=True,
        )

    _assert_model_boundaries(info)
    _assert_sweep_matches_reference(result)
