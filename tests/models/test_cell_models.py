"""Cell models share one callable and port contract across simulation consumers."""

from __future__ import annotations

import importlib
from typing import Any

import gdsfactory as gf
import jax
import jax.numpy as jnp
import pytest

from qpdk import PDK
from qpdk.models import models


@pytest.mark.parametrize(
    ("name", "ports", "settings"),
    [
        ("double_pad_transmon_with_bbox", {"left_pad", "right_pad"}, {}),
        ("fluxonium", {"left_pad", "right_pad"}, {}),
        ("fluxonium_with_bbox", {"left_pad", "right_pad"}, {}),
        ("josephson_junction", {"left_wide", "right_wide"}, {}),
        ("unimon_coupled", {"coupling_o3"}, {"cross_section": "coplanar_waveguide"}),
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
    assert model is models[descriptor["name"]] is PDK.models[name]
    assert set(descriptor["port_order"]) == ports

    frequencies = jnp.linspace(4e9, 8e9, 5)
    actual = jax.block_until_ready(
        jax.jit(lambda f: model(f=f, **settings))(frequencies)
    )
    assert set(actual) == {(p1, p2) for p1 in ports for p2 in ports}
