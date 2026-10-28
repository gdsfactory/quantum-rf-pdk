"""The boxed flipmon has one FEM ring port in GFP component, Mosaic and layout simulations."""

import json
from pathlib import Path
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from conftest import import_gfp_module
from sax.models.rf import tee

from qpdk import PDK, cells
from qpdk.models.flipmon import flipmon_with_bbox_fem


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["component", "mosaic", "layout"])
@pytest.mark.parametrize(
    ("reflection", "layer_case"),
    [(False, 0), *((True, case) for case in range(7))],
)
def test_gfp_component_simulation(
    source: str,
    reflection: bool,
    layer_case: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "flipmon_with_bbox"
    sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
    if reflection and getattr(sax_sim, "_MIN_PORTS", 1) > 1:
        pytest.xfail("This GFP SDK requires two external ports for SAX simulation")
    base_settings: dict[str, Any] = {
        "inductance_nh": 12.0,
        "junction_capacitance_ff": 1.5,
        "chip_gap": 10.0,
        "alignment": 0.0,
    }
    layer_cases: list[dict[str, Any]] = [
        {},
        dict(
            layer_metal="M1_DRAW",
            layer_metal_top="M2_DRAW",
            layer_etch="M1_ETCH",
            layer_etch_top="M2_ETCH",
        ),
        dict(
            layer_metal=[1, 0],
            layer_metal_top=[2, 0],
            layer_etch=[1, 1],
            layer_etch_top=[2, 1],
        ),
        dict(layer_metal=[77, 0]),
        dict(layer_metal_top=[77, 0]),
        dict(layer_etch=[77, 0]),
        dict(layer_etch_top=[77, 0]),
    ]
    qubit_settings: dict[str, Any] = base_settings | layer_cases[layer_case]

    @gf.cell
    def boxed_flipmon_network(
        qubit_name: str = name,
        reflection: bool = reflection,
        layer_case: int = layer_case,
    ) -> gf.Component:
        settings: dict[str, Any] = base_settings | layer_cases[layer_case]
        c = gf.Component()
        if reflection:
            qubit = c.add_ref(getattr(cells, qubit_name)(**settings), name="qubit")
            c.add_port("outer_ring_outside", port=qubit.ports["outer_ring_outside"])
            return c
        splitter = c.add_ref(
            cells.tee(
                cross_section={"cross_section": "cpw", "settings": {"width": 60.0}}
            ),
            name="tee",
        )
        qubit = c.add_ref(getattr(cells, qubit_name)(**settings), name="qubit")
        qubit.connect("outer_ring_outside", splitter.ports["o3"])
        c.add_port("o1", port=splitter.ports["o1"])
        c.add_port("o2", port=splitter.ports["o2"])
        return c

    monkeypatch.setitem(PDK.cells, "boxed_flipmon_network", boxed_flipmon_network)
    sweep_args = dict(
        wl_min=299792458e6 / 8e9, wl_max=299792458e6 / 3e9, wl_num=101, sweep_axis="f"
    )
    if source == "component":
        result = sax_sim.simulate_component_sax(
            "boxed_flipmon_network", "qpdk.PDK", **sweep_args
        )
    else:
        qubit_model = f"qpdk.cells.transmon.{name}"
        tee_model = "qpdk.cells.waveguides.tee"
        document: dict[str, dict[str, Any]] = {
            "tee": {
                "type": "ckt",
                "model": tee_model,
                "x": 0,
                "y": 0,
                "props": {},
                "nets": {"o1": "left", "o2": "right", "o3": "branch"},
            },
            "qubit": {
                "type": "ckt",
                "model": qubit_model,
                "x": 4,
                "y": 0,
                "props": {},
                "nets": {"outer_ring_outside": "branch"},
            },
            "in": {
                "type": "port",
                "name": "o1",
                "x": -2,
                "y": 0,
                "nature": "photonic",
                "nets": {"P": "left"},
            },
            "out": {
                "type": "port",
                "name": "o2",
                "x": 8,
                "y": 0,
                "nature": "photonic",
                "nets": {"P": "right"},
            },
        }
        if reflection:
            del document["tee"], document["out"]
            document["in"].update({
                "name": "outer_ring_outside",
                "nets": {"P": "branch"},
            })
        network = boxed_flipmon_network()
        for instance in network.insts:
            entity = document[instance.name]
            bounds = instance.cell.dbbox()
            center = instance.dbbox().center()
            local_center = bounds.center()
            entity.update({
                "name": instance.name,
                "props": dict(instance.cell.settings) | layer_cases[layer_case],
                "bbox_width": bounds.width(),
                "bbox_height": bounds.height(),
                "bbox_xmin": bounds.left,
                "bbox_ymin": bounds.bottom,
                "irl_x": center.x,
                "irl_y": -center.y,
                "irl_orientation": -instance.dcplx_trans.angle,
                "irl_flip": instance.dcplx_trans.is_mirror(),
                "ports": [
                    {
                        "name": port.name,
                        "irl_rx": port.dx - local_center.x,
                        "irl_ry": local_center.y - port.dy,
                        "orientation": port.orientation,
                        "width": port.width,
                        "port_type": port.port_type,
                        "layer": list(gf.get_layer_tuple(port.layer)),
                    }
                    for port in instance.cell.ports
                ],
            })
        markers = (
            (("in", "outer_ring_outside"),)
            if reflection
            else (("in", "o1"), ("out", "o2"))
        )
        for marker, pin in markers:
            port = network.ports[pin]
            document[marker].update({
                "attached_port": {
                    "component_id": "qubit" if reflection else "tee",
                    "port_name": pin,
                },
                "irl_x": port.dx,
                "irl_y": -port.dy,
                "port_type": "optical",
            })
        path = tmp_path / "boxed_flipmon_simulation.gsch"
        path.write_text(json.dumps(document), encoding="utf-8")
        library = {
            f"models:{key}": {
                "name": key.rsplit(".", 1)[-1],
                "type": "ckt",
                "tags": ["qpdk"],
            }
            for key in (qubit_model, tee_model)
        }
        (tmp_path / "models.nyanlib").write_text(json.dumps(library), encoding="utf-8")
        if source == "mosaic":
            result = sax_sim.simulate_mosaic_sax(
                str(path), "qpdk.PDK", project_root=str(tmp_path), **sweep_args
            )
        else:
            result = sax_sim.simulate_layout_sax(str(path), "qpdk.PDK", **sweep_args)
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": "qubit", "tee": "tee"},
            "connections": {"qubit,outer_ring_outside": "tee,o3"},
            "ports": {"o1": "tee,o1", "o2": "tee,o2"},
        },
        models={"qubit": flipmon_with_bbox_fem, "tee": tee},
    )
    f = 299792458e6 / jnp.asarray(result["wavelengths"])
    expected = (
        flipmon_with_bbox_fem(f=f, **qubit_settings)
        if reflection
        else circuit(f=f, qubit=qubit_settings)
    )
    assert set(result["sdict"]) == {f"{a},{b}" for a, b in expected}
    for (a, b), reference in expected.items():
        entry = result["sdict"][f"{a},{b}"]
        actual = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        if layer_case >= 3:
            assert bool(jnp.all(jnp.isnan(actual)))
            assert bool(jnp.all(jnp.isnan(reference)))
        else:
            assert bool(jnp.all(jnp.isfinite(actual)))
            np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
