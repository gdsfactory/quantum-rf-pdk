"""gdsfactoryplus v2 SAX integration tests for the real test-chip schematic.

Both simulation paths must compose the chip from component models. The chip
itself intentionally has no SAX model: unmodeled assemblies expand until a
modeled component boundary, such as ``quarter_wave_resonator_coupled``.
"""

from __future__ import annotations

import inspect
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
from qpdk.models import models

sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
factory_metadata = import_gfp_module("gdsfactoryplus.factory_metadata")

if TYPE_CHECKING:
    from collections.abc import Callable

_SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
_GSCH_PATH = _SAMPLE_DIR / "resonator_test_chip_yaml.gsch"
_PIC_YAML_PATH = _SAMPLE_DIR / "resonator_test_chip_yaml.pic.yml"
_CHIP_QUALNAME = "qpdk.samples.resonator_test_chip.resonator_test_chip_python"

_SPEED_OF_LIGHT_UM_PER_S = 299_792_458_000_000.0
_F_MIN_HZ = 4e9
_F_MAX_HZ = 10e9

_COUPLING_MODEL_BOUNDARIES = {
    "quarter_wave_resonator_coupled",
}
_COUPLING_INTERNAL_COMPONENTS = {
    "bend_circular",
    "rectangle",
    "taper_cross_section",
}

_MODEL_PARAMS: list[tuple[str, Callable[..., Any]]] = sorted(PDK.models.items())
_F_CONTRACT_EXEMPT: frozenset[str] = frozenset()


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
    assert not required & _COUPLING_INTERNAL_COMPONENTS


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


def _write_nested_chip_schematic(root: Path) -> Path:
    """Write a schematic whose only component is the unmodeled Python chip."""
    ports = ("o1", "o2", "o3", "o4")
    document: dict[str, Any] = {
        "chip:X1": {
            "type": "ckt",
            "model": _CHIP_QUALNAME,
            "name": "X1",
            "transform": [1, 0, 0, 1, 0, 0],
            "x": 0,
            "y": 0,
            "props": {},
            "nets": {port: f"net_{port}" for port in ports},
        }
    }
    for port in ports:
        document[f"chip:{port}"] = {
            "type": "port",
            "name": port,
            "x": 0,
            "y": 0,
            "nets": {"P": f"net_{port}"},
        }

    schematic_path = root / "nested_resonator_test_chip.gsch"
    schematic_path.write_text(json.dumps(document, indent=2))
    (root / "models.nyanlib").write_text(
        json.dumps(
            {
                f"models:{_CHIP_QUALNAME}": {
                    "name": "resonator_test_chip_python",
                    "type": "ckt",
                    "tags": ["qpdk"],
                }
            },
            indent=2,
        )
    )
    return schematic_path


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
def test_nested_python_chip_stops_at_coupling_models(tmp_path: Path) -> None:
    """Resolve models after expanding each level of the Python chip hierarchy."""
    schematic_path = _write_nested_chip_schematic(tmp_path)
    info = sax_sim.inspect_mosaic_sax_models(
        str(schematic_path),
        "qpdk.PDK",
        str(tmp_path),
    )
    _assert_model_boundaries(info)

    result = sax_sim.simulate_mosaic_sax(
        str(schematic_path),
        "qpdk.PDK",
        wl_min=_SPEED_OF_LIGHT_UM_PER_S / _F_MAX_HZ,
        wl_max=_SPEED_OF_LIGHT_UM_PER_S / _F_MIN_HZ,
        wl_num=3,
        project_root=str(tmp_path),
        sweep_frequency=True,
    )
    _assert_sweep_matches_reference(result)


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


@pytest.mark.gfp
@pytest.mark.parametrize(("model_name", "model"), _MODEL_PARAMS)
def test_pdk_model_accepts_f_in_hz(model_name: str, model: Callable) -> None:
    """Require every registered SAX model to support a frequency sweep."""
    if model_name in _F_CONTRACT_EXEMPT:
        pytest.skip(f"{model_name} is exempt from the f-in-Hz contract")

    try:
        parameters = inspect.signature(model).parameters
    except (TypeError, ValueError) as exc:
        pytest.fail(f"{model_name} has no inspectable signature: {exc}")
        raise AssertionError("unreachable") from None
    if "f" not in parameters:
        pytest.fail(f"{model_name} does not take an 'f' parameter")
    required_beyond_f = [
        name
        for name, parameter in parameters.items()
        if name != "f"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    if required_beyond_f:
        pytest.fail(
            f"{model_name} requires {required_beyond_f} without defaults; "
            "it cannot be swept by sax.circuit"
        )

    s_params = model(f=np.linspace(_F_MIN_HZ, _F_MAX_HZ, 3))

    assert isinstance(s_params, dict), f"{model_name} returned {type(s_params)}"
    assert s_params, f"{model_name} returned an empty S-parameter dict"
    for key, value in s_params.items():
        arr = np.broadcast_to(np.asarray(value, dtype=complex), (3,))
        assert np.isfinite(arr).all(), f"{model_name} S[{key}] is not finite"
