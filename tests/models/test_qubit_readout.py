"""Native Xmon capacitances reach the external readout circuit and GFP."""

import json
from pathlib import Path
from typing import Any

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from conftest import import_gfp_module
from sax.models.rf import tee

from qpdk import PDK, cells
from qpdk.models import qubit_readout as model_module
from qpdk.models.datasets import Dataset, GridInterpolator


@pytest.mark.gfp
@pytest.mark.parametrize("source", ["component", "mosaic", "layout"])
@pytest.mark.parametrize("reflection", [False, True])
def test_gfp_component_simulation(
    source: str,
    reflection: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "xmon_readout"
    sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")
    qubit_settings = {
        "inductance_nh": 12.0,
        "junction_capacitance_ff": 1.5,
        "coupling_gap": 7.5,
        "paddle_length": 60.0,
    }

    @gf.cell
    def qubit_readout_network(
        qubit_name: str = name, reflection: bool = reflection
    ) -> gf.Component:
        c = gf.Component()
        if reflection:
            qubit = c.add_ref(
                getattr(cells, qubit_name)(**qubit_settings), name="qubit"
            )
            c.add_port("readout", port=qubit.ports["readout"])
            return c
        splitter = c.add_ref(cells.tee(), name="tee")
        qubit = c.add_ref(getattr(cells, qubit_name)(**qubit_settings), name="qubit")
        qubit.connect("readout", splitter.ports["o3"])
        c.add_port("o1", port=splitter.ports["o1"])
        c.add_port("o2", port=splitter.ports["o2"])
        return c

    monkeypatch.setitem(PDK.cells, "qubit_readout_network", qubit_readout_network)
    sweep_args = dict(
        wl_min=299792458e6 / 8e9, wl_max=299792458e6 / 3e9, wl_num=101, sweep_axis="f"
    )
    if source == "component":
        result = sax_sim.simulate_component_sax(
            "qubit_readout_network", "qpdk.PDK", **sweep_args
        )
    else:
        qubit_model = f"qpdk.cells.qubit_readout.{name}"
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
                "nets": {"readout": "branch"},
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
            document["in"].update({"name": "readout", "nets": {"P": "branch"}})
        network = qubit_readout_network()
        for instance in network.insts:
            entity = document[instance.name]
            bounds = instance.cell.dbbox()
            center = instance.dbbox().center()
            local_center = bounds.center()
            entity.update({
                "name": instance.name,
                "props": dict(instance.cell.settings),
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
        markers = (("in", "readout"),) if reflection else (("in", "o1"), ("out", "o2"))
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
        path = tmp_path / "readout.gsch"
        path.write_text(json.dumps(document))
        library = {
            f"models:{key}": {
                "name": key.rsplit(".", 1)[-1],
                "type": "ckt",
                "tags": ["qpdk"],
            }
            for key in (qubit_model, tee_model)
        }
        (tmp_path / "models.nyanlib").write_text(json.dumps(library))
        if source == "mosaic":
            result = sax_sim.simulate_mosaic_sax(
                str(path), "qpdk.PDK", project_root=str(tmp_path), **sweep_args
            )
        else:
            result = sax_sim.simulate_layout_sax(str(path), "qpdk.PDK", **sweep_args)
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": "qubit", "tee": "tee"},
            "connections": {"qubit,readout": "tee,o3"},
            "ports": {"o1": "tee,o1", "o2": "tee,o2"},
        },
        models={"qubit": getattr(model_module, name), "tee": tee},
    )
    f = 299792458e6 / jnp.asarray(result["wavelengths"])
    expected = (
        getattr(model_module, name)(f=f, **qubit_settings)
        if reflection
        else circuit(f=f, qubit=qubit_settings)
    )
    assert set(result["sdict"]) == {f"{a},{b}" for a, b in expected}
    for (a, b), reference in expected.items():
        entry = result["sdict"][f"{a},{b}"]
        actual = jnp.asarray(entry["real"]) + 1j * jnp.asarray(entry["imag"])
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("inductance_nh", "junction_capacitance_ff"), [(10.0, 0.0), (15.0, 2.0)]
)
def test_native_matrix_and_feedline_mode(
    inductance_nh: float, junction_capacitance_ff: float
) -> None:
    geometry = dict(arm_scale=1.0, gap_width=10.0, coupling_gap=7.5, paddle_length=60.0)
    matrix = GridInterpolator(
        Dataset("xmon_readout_palace").grid("maxwell_capacitance")
    )(**geometry)
    frequency = 1 / (
        2
        * jnp.pi
        * jnp.sqrt(
            inductance_nh * 1e-9 * (matrix[0, 0] + junction_capacitance_ff * 1e-15)
        )
    )
    f = frequency * jnp.linspace(0.95, 1.05, 201)
    settings = geometry | dict(
        inductance_nh=inductance_nh, junction_capacitance_ff=junction_capacitance_ff
    )
    # Avoid the internal admittance pole when comparing the nodal elimination.
    omega = 2 * jnp.pi * f[jnp.asarray([0, 50, 150, 200])]
    admittance = 1j * omega[:, None, None] * matrix
    branch = (
        1 / (1j * omega * inductance_nh * 1e-9)
        + 1j * omega * junction_capacitance_ff * 1e-15
    )
    effective = admittance[:, 1, 1] - admittance[:, 1, 0] * admittance[:, 0, 1] / (
        admittance[:, 0, 0] + branch
    )
    expected = (1 - 50 * effective) / (1 + 50 * effective)
    response = jax.jit(model_module.xmon_readout)(f=omega / (2 * jnp.pi), **settings)
    np.testing.assert_allclose(
        response["readout", "readout"], expected, rtol=1e-10, atol=1e-12
    )
    circuit, _ = sax.circuit(
        {
            "instances": {"qubit": "qubit", "tee": "tee"},
            "connections": {"qubit,readout": "tee,o3"},
            "ports": {"o1": "tee,o1", "o2": "tee,o2"},
        },
        models={"qubit": model_module.xmon_readout, "tee": tee},
    )
    scattering = jax.jit(circuit)(f=f, qubit=settings)
    transmission = jnp.abs(scattering["o2", "o1"])
    reflection = jnp.abs(scattering["o1", "o1"])
    assert int(jnp.argmin(transmission)) == 100
    assert float(transmission[100]) < 1e-9
    assert float(transmission[0]) > 0.1
    np.testing.assert_allclose(transmission**2 + reflection**2, 1, atol=1e-12)
    np.testing.assert_allclose(
        scattering["o1", "o2"], scattering["o2", "o1"], atol=1e-12
    )


def test_xmon_geometry_derivative_and_domain() -> None:
    def evaluate(gap: float) -> jax.Array:
        return jnp.imag(
            model_module.xmon_readout(f=5e9, coupling_gap=gap)["readout", "readout"]
        )

    derivative = jax.jit(jax.grad(evaluate))(6.0)
    difference = (evaluate(6.001) - evaluate(5.999)) / 0.002
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, difference, rtol=1e-5)
    assert jnp.all(jnp.isfinite(jax.jit(jax.vmap(evaluate))(jnp.asarray([6.0, 8.0]))))
    assert jnp.isnan(evaluate(0.1))

    batched = model_module.xmon_readout(
        f=5e9,
        inductance_nh=jnp.asarray([10.0, 12.0]),
        junction_capacitance_ff=jnp.asarray([1.5, 2.5]),
    )["readout", "readout"]
    expected = jax.vmap(
        lambda inductance, capacitance: model_module.xmon_readout(
            f=5e9, inductance_nh=inductance, junction_capacitance_ff=capacitance
        )["readout", "readout"]
    )(jnp.asarray([10.0, 12.0]), jnp.asarray([1.5, 2.5]))
    np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-12)
