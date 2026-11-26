"""Cell models share one callable and port contract across simulation consumers."""

from __future__ import annotations

import importlib
from typing import Any

import gdsfactory as gf
import jax
import jax.numpy as jnp
import pytest
import sax

from qpdk import PDK
from qpdk.models import models
from qpdk.models.inductor import lumped_element_resonator


@pytest.mark.parametrize(
    ("name", "ports", "settings"),
    [
        ("double_pad_transmon_with_bbox", {"left_pad", "right_pad"}, {}),
        ("flipmon", {"outer_ring_outside"}, {}),
        ("flipmon_with_bbox", {"outer_ring_outside"}, {}),
        ("fluxonium", {"left_pad", "right_pad"}, {}),
        ("fluxonium_with_bbox", {"left_pad", "right_pad"}, {}),
        ("josephson_junction", {"left_wide", "right_wide"}, {}),
        ("unimon_coupled", {"coupling_o3"}, {"cross_section": "coplanar_waveguide"}),
        ("xmon_transmon", {"top_arm", "right_arm", "bottom_arm", "left_arm"}, {}),
    ],
)
def test_cell_model_has_one_jittable_port_contract(
    name: str, ports: set[str], settings: dict[str, Any]
) -> None:
    """Schematic resolution and the PDK use the public model without wrappers."""
    info: Any = gf.kcl.factories[name].get_schematic().info
    descriptor = info["models"][0]
    module = importlib.import_module(descriptor["module"])
    model = getattr(module, descriptor["qualname"])
    assert model is models[descriptor["qualname"]] is PDK.models[name]
    assert set(descriptor["port_order"]) == ports

    frequencies = jnp.linspace(4e9, 8e9, 5)
    actual = jax.block_until_ready(
        jax.jit(lambda f: model(f=f, **settings))(frequencies)
    )
    assert set(actual) == {(p1, p2) for p1 in ports for p2 in ports}


def test_lumped_element_resonator_rejects_traced_grounded() -> None:
    """``grounded`` selects the circuit topology and must be fixed when tracing."""

    @jax.jit
    def response(f: jax.Array, grounded: jax.Array) -> dict:
        return lumped_element_resonator(f=f, grounded=grounded)

    with pytest.raises(
        jax.errors.ConcretizationTypeError, match="grounded must be fixed"
    ):
        response(jnp.asarray([5e9]), jnp.asarray(False))


@pytest.mark.parametrize("name", ["flipmon", "flipmon_with_bbox", "xmon_transmon"])
def test_placed_qubit_builds_sax_circuit(name: str) -> None:
    """A parent exposing the qubit's circuit ports simulates through its model."""
    parent = gf.Component()
    ref = parent << gf.get_component(name)
    parent.add_ports([p for p in ref.ports if p.port_type != "placement"])

    circuit, _ = sax.circuit(parent.get_netlist(), models=PDK.models)

    assert set(circuit(f=jnp.array([5e9]))) == {
        (p1.name, p2.name) for p1 in parent.ports for p2 in parent.ports
    }
