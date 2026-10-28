"""The packaged Xmon FEM model agrees across GFP's simulation paths."""

import json
from functools import partial
from pathlib import Path

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import import_gfp_module
from gdsfactory.typings import LayerSpec

from qpdk import PDK, cells
from qpdk.models.xmon import xmon_fem
from qpdk.tech import LAYER

sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
_SETTINGS = {"inductance_nh": 12.0, "junction_capacitance_ff": 1.5}
_LAYERS: tuple[tuple[LayerSpec, LayerSpec, bool], ...] = (
    (LAYER.M1_DRAW, LAYER.M1_ETCH, True),
    ("M1_DRAW", "M1_ETCH", True),
    ((1, 0), (1, 1), True),
    ((77, 0), LAYER.M1_ETCH, False),
    (LAYER.M1_DRAW, (77, 0), False),
    ("M2_DRAW", LAYER.M1_ETCH, False),
    (LAYER.M1_DRAW, "M2_ETCH", False),
)


@gf.cell
def _xmon_fem_network(layer_case: int = 0) -> gf.Component:
    c = gf.Component()
    metal, etch, _ = _LAYERS[layer_case]
    qubit = c.add_ref(
        cells.xmon_transmon(layer_metal=metal, layer_etch=etch, **_SETTINGS),
        name="qubit",
    )
    for port in qubit.ports:
        if port.port_type != "placement":
            c.add_port(port.name, port=port)
    return c


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["component", "mosaic", "layout"])
@pytest.mark.parametrize(
    "layer_case",
    range(len(_LAYERS)),
    ids=["enum", "name", "pair", "metal77", "etch77", "wrong-metal", "wrong-etch"],
)
def test_xmon_fem_simulation(
    source: str, layer_case: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    PDK.activate()
    monkeypatch.setitem(
        PDK.cells, "xmon_fem_network", partial(_xmon_fem_network, layer_case=layer_case)
    )
    network = _xmon_fem_network(layer_case)
    qubit = next(iter(network.insts))
    _, _, supported = _LAYERS[layer_case]
    if layer_case in {3, 4}:
        assert network.get_polygons(by="tuple")[77, 0]
    box = qubit.cell.dbbox()
    centre = qubit.dbbox().center()
    local = box.center()
    factory = "qpdk.cells.transmon.xmon_transmon"
    document = {
        "qubit": {
            "type": "ckt",
            "model": factory,
            "name": "qubit",
            "x": 0,
            "y": 0,
            "props": qubit.cell.settings.model_dump(mode="json"),
            "bbox_width": box.width(),
            "bbox_height": box.height(),
            "bbox_xmin": box.left,
            "bbox_ymin": box.bottom,
            "irl_x": centre.x,
            "irl_y": -centre.y,
            "irl_orientation": -qubit.dcplx_trans.angle,
            "irl_flip": qubit.dcplx_trans.is_mirror(),
            "nets": {p.name: f"net_{p.name}" for p in network.ports},
            "ports": [
                {
                    "name": p.name,
                    "irl_rx": p.dx - local.x,
                    "irl_ry": local.y - p.dy,
                    "orientation": p.orientation,
                    "width": p.width,
                    "port_type": p.port_type,
                    "layer": list(gf.get_layer_tuple(p.layer)),
                }
                for p in qubit.cell.ports
            ],
        }
    }
    for i, port in enumerate(network.ports):
        document[f"port_{i}"] = {
            "type": "port",
            "name": port.name,
            "x": i + 1,
            "y": 2,
            "nature": "photonic",
            "nets": {"P": f"net_{port.name}"},
            "attached_port": {"component_id": "qubit", "port_name": port.name},
            "irl_x": port.dx,
            "irl_y": -port.dy,
            "port_type": "optical",
        }
    path = tmp_path / "xmon_fem.gsch"
    path.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "models.nyanlib").write_text(
        json.dumps({
            f"models:{factory}": {
                "name": "xmon_transmon",
                "type": "ckt",
                "tags": ["qpdk"],
            }
        }),
        encoding="utf-8",
    )
    sweep = dict(
        wl_min=299792458e6 / 8e9,
        wl_max=299792458e6 / 3e9,
        wl_num=101,
        sweep_axis="f",
    )
    match source:
        case "component":
            result = sax_sim.simulate_component_sax(
                "xmon_fem_network", "qpdk.PDK", **sweep
            )
        case "mosaic":
            result = sax_sim.simulate_mosaic_sax(
                str(path), "qpdk.PDK", project_root=str(tmp_path), **sweep
            )
        case "layout":
            result = sax_sim.simulate_layout_sax(str(path), "qpdk.PDK", **sweep)
        case _:
            raise AssertionError(source)
    frequency = 299792458e6 / jnp.asarray(result["wavelengths"])
    expected = xmon_fem(f=frequency, **_SETTINGS)
    assert set(result["sdict"]) == {f"{a},{b}" for a, b in expected}
    for (a, b), value in expected.items():
        entry = result["sdict"][f"{a},{b}"]
        actual = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        if supported:
            assert bool(jnp.isfinite(actual).all())
            np.testing.assert_allclose(actual, value, rtol=1e-6, atol=1e-6)
        else:
            assert bool(jnp.isnan(actual).all())
