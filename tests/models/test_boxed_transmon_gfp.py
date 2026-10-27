"""Boxed-transmon circuit settings reach all three GFP simulation sources."""

import json
from pathlib import Path

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.models import boxed_transmon_fem


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["component", "schematic", "layout"])
def test_boxed_transmon_simulation(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = import_gfp_module("gdsfactoryplus.sim.sax")
    settings = dict(
        pad_size=(200.0, 325.0),
        pad_gap=12.5,
        bbox_extension=150.0,
        inductance_nh=12.0,
        junction_capacitance_ff=1.5,
    )
    component = double_pad_transmon_with_bbox(**settings)

    @gf.cell
    def boxed_network() -> gf.Component:
        result = gf.Component()
        qubit = result.add_ref(component, name="qubit")
        for name in ("left_pad", "right_pad"):
            result.add_port(name, port=qubit.ports[name])
        return result

    monkeypatch.setitem(PDK.cells, "boxed_network", boxed_network)
    sweep = dict(
        wl_min=299792458e6 / 10e9, wl_max=299792458e6 / 1e9, wl_num=101, sweep_axis="f"
    )
    if source == "component":
        result = sdk.simulate_component_sax("boxed_network", "qpdk.PDK", **sweep)
    else:
        model = "qpdk.cells.transmon.double_pad_transmon_with_bbox"
        bounds = component.dbbox()
        center = bounds.center()
        document = {
            "qubit": {
                "type": "ckt",
                "name": "qubit",
                "model": model,
                "x": 0,
                "y": 0,
                "props": dict(component.settings),
                "nets": {name: name for name in ("left_pad", "right_pad")},
                "bbox_width": bounds.width(),
                "bbox_height": bounds.height(),
                "bbox_xmin": bounds.left,
                "bbox_ymin": bounds.bottom,
                "irl_x": center.x,
                "irl_y": -center.y,
                "irl_orientation": 0,
                "irl_flip": False,
                "ports": [
                    {
                        "name": p.name,
                        "irl_rx": p.dx - center.x,
                        "irl_ry": center.y - p.dy,
                        "orientation": p.orientation,
                        "width": p.width,
                        "port_type": p.port_type,
                        "layer": list(gf.get_layer_tuple(p.layer)),
                    }
                    for p in component.ports
                ],
            }
        }
        for i, name in enumerate(("left_pad", "right_pad")):
            port = component.ports[name]
            document[name] = {
                "type": "port",
                "name": name,
                "x": -2 if i == 0 else 2,
                "y": 0,
                "nature": "photonic",
                "nets": {"P": name},
                "attached_port": {"component_id": "qubit", "port_name": name},
                "irl_x": port.dx,
                "irl_y": -port.dy,
                "port_type": "optical",
            }
        path = tmp_path / "boxed.gsch"
        path.write_text(json.dumps(document))
        (tmp_path / "models.nyanlib").write_text(
            json.dumps({
                f"models:{model}": {
                    "name": "double_pad_transmon_with_bbox",
                    "type": "ckt",
                    "tags": ["qpdk"],
                }
            })
        )
        if source == "schematic":
            result = sdk.simulate_mosaic_sax(
                str(path), "qpdk.PDK", project_root=str(tmp_path), **sweep
            )
        else:
            result = sdk.simulate_layout_sax(str(path), "qpdk.PDK", **sweep)
    frequencies = 299792458e6 / jnp.asarray(result["wavelengths"])
    expected = boxed_transmon_fem(f=frequencies, **settings)
    assert set(result["sdict"]) == {f"{a},{b}" for a, b in expected}
    for (a, b), reference in expected.items():
        entry = result["sdict"][f"{a},{b}"]
        actual = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
