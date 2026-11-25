"""End-to-end gdsfactoryplus SAX simulations of the readout test chips.

The composite qubit-resonator-probeline boundary is the leaf model. Both the
schematic and layout paths must resolve it with no missing models, and each
saved resonator length must change the chip's complex S21.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest
from conftest import import_gfp_module

sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")

_SAMPLE_DIR = Path(__file__).parents[2] / "qpdk/samples"
_SPEED_OF_LIGHT_UM_PER_S = 299_792_458_000_000.0
_F_MIN_HZ = 3e9
_F_MAX_HZ = 20e9
_WL_NUM = 301
_LENGTH_DELTA_UM = 200.0
_MIN_PEAK_S21 = 1e-5
_MIN_S21_CHANGE = 1e-6

_CHIPS = ("qubit_test_chip", "flipmon_test_chip")
_SOURCES = ("mosaic", "layout")


def _write_project(
    root: Path, source: Path, perturbed_device: str | None = None
) -> Path:
    """Copy a schematic plus the minimal model library its topology needs."""
    document = json.loads(source.read_text(encoding="utf-8"))
    if perturbed_device is not None:
        document[perturbed_device]["props"]["resonator_length"] += _LENGTH_DELTA_UM
    root.mkdir(parents=True, exist_ok=True)
    schematic_path = root / source.name
    schematic_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    models = {
        f"models:{device['model']}": {
            "name": device["model"].rsplit(".", maxsplit=1)[-1],
            "type": "ckt",
            "tags": ["qpdk"],
        }
        for device in document.values()
        if isinstance(device, dict) and isinstance(device.get("model"), str)
    }
    (root / "models.nyanlib").write_text(json.dumps(models, indent=2), encoding="utf-8")
    return schematic_path


def _readout_devices(document: dict[str, Any]) -> list[str]:
    return [
        name
        for name, device in document.items()
        if isinstance(device, dict)
        and isinstance(device.get("model"), str)
        and "with_resonator_and_probeline" in device["model"]
    ]


def _simulate(source: str, path: Path, root: Path) -> dict[str, Any]:
    kwargs = {
        "wl_min": _SPEED_OF_LIGHT_UM_PER_S / _F_MAX_HZ,
        "wl_max": _SPEED_OF_LIGHT_UM_PER_S / _F_MIN_HZ,
        "wl_num": _WL_NUM,
        "sweep_axis": "f",
    }
    if source == "mosaic":
        return sax_sim.simulate_mosaic_sax(
            str(path), "qpdk.PDK", project_root=str(root), **kwargs
        )
    return sax_sim.simulate_layout_sax(str(path), "qpdk.PDK", **kwargs)


def _s21(result: dict[str, Any]) -> jnp.ndarray:
    entry = result["sdict"]["o1,o2"]
    return jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])


@pytest.mark.gfp
@pytest.mark.parametrize("chip", _CHIPS)
@pytest.mark.parametrize("source", _SOURCES)
def test_readout_chip_length_perturbations(
    chip: str, source: str, tmp_path: Path
) -> None:
    """Simulate a chip once, then verify each saved length moves its S21."""
    source_path = _SAMPLE_DIR / f"{chip}.gsch"
    document = json.loads(source_path.read_text())
    devices = _readout_devices(document)
    assert len(devices) == 4, f"{chip}: expected four readout blocks, got {devices}"

    baseline_root = tmp_path / "baseline"
    baseline_path = _write_project(baseline_root, source_path)
    report = (
        sax_sim.inspect_mosaic_sax_models(
            str(baseline_path), "qpdk.PDK", str(baseline_root)
        )
        if source == "mosaic"
        else sax_sim.inspect_layout_sax_models(str(baseline_path), "qpdk.PDK")
    )
    assert report["missing_models"] == []
    required = set(report["required_models"])
    if source == "mosaic":
        boundary = document[devices[0]]["model"].rsplit(".", maxsplit=1)[-1]
        assert boundary in required
    else:
        assert any("with_resonator_and_probeline" in name for name in required), (
            f"layout extraction lost the readout boundary: {sorted(required)}"
        )

    baseline = _s21(_simulate(source, baseline_path, baseline_root))
    assert bool(jnp.isfinite(baseline).all())
    assert float(jnp.max(jnp.abs(baseline))) > _MIN_PEAK_S21, (
        f"{chip} {source}: baseline S21 is effectively zero"
    )

    for device in devices:
        root = tmp_path / f"perturbed-{device}"
        path = _write_project(root, source_path, perturbed_device=device)
        changed = _s21(_simulate(source, path, root))
        difference = float(jnp.max(jnp.abs(changed - baseline)))
        assert difference > _MIN_S21_CHANGE, (
            f"{chip} {source} {device}: resonator_length + {_LENGTH_DELTA_UM} um "
            f"changed max |S21| by only {difference:.3g}"
        )
