"""Tests for the private PDK/layout model bindings.

The PDK registers the port-renamed wrappers from
:mod:`qpdk.models.pdk_bindings` for the cells whose layout ports differ from the
analytical catalog's port names. The wrappers must stay numerically identical to
the analytical models they wrap, and the analytical models must keep their own
ports for direct scientific use.
"""

from __future__ import annotations

import importlib
from operator import itemgetter
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from qpdk import PDK
from qpdk.models import _PDK_MODEL_OVERRIDES, models as sax_models, pdk_bindings

BINDING_MODULE = "qpdk.models.pdk_bindings"

# (name shared by cell and model, analytical port -> layout port, settings)
BINDINGS: list[tuple[str, dict[str, str], dict[str, Any]]] = [
    ("double_pad_transmon_with_bbox", {"o1": "left_pad", "o2": "right_pad"}, {}),
    ("fluxonium", {"o1": "left_pad", "o2": "right_pad"}, {}),
    ("fluxonium_with_bbox", {"o1": "left_pad", "o2": "right_pad"}, {}),
    ("josephson_junction", {"o1": "left_wide", "o2": "right_wide"}, {}),
    # Schematics and netlists pass the string-valued cross-section explicitly.
    (
        "unimon_coupled",
        {"o1": "coupling_o3"},
        {"cross_section": "coplanar_waveguide"},
    ),
]

FREQUENCIES = jnp.linspace(4e9, 8e9, 5)


def _descriptor(name: str) -> dict[str, Any]:
    """Return the SAX model descriptor the same-named cell declares."""
    schematic_function = cast(Any, PDK.cells[name]).schematic_function
    return schematic_function().info["models"][0]


@pytest.mark.parametrize("binding", BINDINGS, ids=itemgetter(0))
def test_wrapper_only_renames_the_analytical_model_ports(
    binding: tuple[str, dict[str, str], dict[str, Any]],
) -> None:
    """A wrapper must be its analytical model under the layout port mapping."""
    name, port_map, settings = binding
    reference = sax_models[name](f=FREQUENCIES, **settings)
    actual = getattr(pdk_bindings, name)(f=FREQUENCIES, **settings)

    assert set(actual) == {(port_map[p1], port_map[p2]) for p1, p2 in reference}
    for (p1, p2), value in reference.items():
        assert_allclose(actual[port_map[p1], port_map[p2]], value, atol=1e-12)


@pytest.mark.parametrize("binding", BINDINGS, ids=itemgetter(0))
def test_analytical_model_keeps_its_own_ports(
    binding: tuple[str, dict[str, str], dict[str, Any]],
) -> None:
    """The public catalog keeps the port names it has always exposed."""
    name, port_map, settings = binding
    ports = set(port_map)

    s_params = sax_models[name](f=FREQUENCIES, **settings)

    assert set(s_params) == {(p1, p2) for p1 in ports for p2 in ports}


@pytest.mark.parametrize("binding", BINDINGS, ids=itemgetter(0))
def test_wrapper_jits_with_traced_frequency(
    binding: tuple[str, dict[str, str], dict[str, Any]],
) -> None:
    """The registered model must trace via jit with its own settings."""
    name, port_map, settings = binding
    model = PDK.models[name]
    layout_ports = set(port_map.values())
    jitted = jax.jit(lambda f: model(f=f, **settings))

    # block_until_ready surfaces errors raised by asynchronous dispatch
    result = jax.block_until_ready(jitted(FREQUENCIES))

    assert set(result) == {(p1, p2) for p1 in layout_ports for p2 in layout_ports}


@pytest.mark.parametrize("binding", BINDINGS, ids=itemgetter(0))
def test_descriptor_resolves_to_the_pdk_bound_wrapper(
    binding: tuple[str, dict[str, str], dict[str, Any]],
) -> None:
    """Schematic descriptors must name the wrapper the PDK binds, not the model."""
    name, _port_map, _settings = binding
    descriptor = _descriptor(name)

    assert descriptor["name"] == name
    assert descriptor["module"] == BINDING_MODULE
    module = importlib.import_module(descriptor["module"])
    model = getattr(module, descriptor["qualname"])

    assert model is PDK.models[name]
    assert model is _PDK_MODEL_OVERRIDES[name]
    assert model is not sax_models[name]
