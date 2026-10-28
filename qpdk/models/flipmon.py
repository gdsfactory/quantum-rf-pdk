"""Boxed flipmon layouts expose the ring while the bonded island floats."""

from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import LayerSpec
from jax.typing import ArrayLike
from kfactory import LayerEnum
from sax.models.rf import electrical_open

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.datasets.models import _matches_layer
from qpdk.models.qubit import flipmon_bonded_model
from qpdk.tech import LAYER


def _open(f: ArrayLike = DEFAULT_FREQUENCY) -> sax.SDict:
    """Keep the static port count outside SAX's traced settings."""
    return electrical_open(f=f, n_ports=1)


@cache
def _flipmon_fem_circuit() -> sax.SDictModel:
    """An explicit open applies zero injected current at the internal island."""
    model, _ = sax.circuit(
        {
            "instances": {"qubit": "qubit", "open": "open"},
            "connections": {"qubit,island": "open,o1"},
            "ports": {"outer_ring_outside": "qubit,ring"},
        },
        models={"qubit": flipmon_bonded_model(inductance=7e-9), "open": _open},
        ignore_impossible_connections=False,
    )
    return model


def flipmon_with_bbox_fem(
    f: ArrayLike = DEFAULT_FREQUENCY,
    inner_circle_radius: ArrayLike = 60.0,
    outer_ring_radius: ArrayLike = 110.0,
    outer_ring_width: ArrayLike = 60.0,
    top_circle_radius: ArrayLike = 110.0,
    m1_etch_extension_gap: ArrayLike = 30.0,
    m2_etch_extension_gap: ArrayLike = 40.0,
    chip_gap: ArrayLike = 10.0,
    alignment: ArrayLike = 0.0,
    inductance_nh: ArrayLike = 7.0,
    junction_capacitance_ff: ArrayLike = 0.0,
    z_ref: ArrayLike = 50.0,
    layer_metal: LayerSpec | ArrayLike = LAYER.M1_DRAW,
    layer_metal_top: LayerSpec | ArrayLike = LAYER.M2_DRAW,
    layer_etch: LayerSpec | ArrayLike = LAYER.M1_ETCH,
    layer_etch_top: LayerSpec | ArrayLike = LAYER.M2_ETCH,
) -> sax.SDict:
    """One ring port on the finite-bump flipmon with two chip ground frames.

    The inner disk and shifted top pad form one floating island. Dimensions
    use um, frequency Hz, inductance nH and extra junction capacitance fF.
    The 800 um square ground frames, two 300 um silicon chips and 0.2 um
    metal match the dataset. Ring radius and width stay 110 and 60 um,
    with 30 and 40 um ground clearances on the two chips.

    Unsupported layers, ring dimensions, clearances or out-of-grid geometry
    return NaN. The model assumes the standard QPDK materials. Layer names
    can be Python literals; external JIT arguments must use numeric layer
    indices or GDS pairs, or close over the names as static values.
    It excludes readout paddles, routes, bump inductance and nonlinear spectra.
    Connecting a distributed line directly to this ring port differs from
    placing a capacitive readout paddle next to it.

    Returns:
        Jittable one-port response named ``outer_ring_outside``.
    """
    supported = jnp.isclose(outer_ring_radius, 110.0, rtol=0, atol=1e-12)
    supported &= jnp.isclose(outer_ring_width, 60.0, rtol=0, atol=1e-12)
    supported &= jnp.isclose(m1_etch_extension_gap, 30.0, rtol=0, atol=1e-12)
    supported &= jnp.isclose(m2_etch_extension_gap, 40.0, rtol=0, atol=1e-12)
    supported &= jnp.asarray(z_ref) > 0
    supported &= _matches_layer(
        layer_metal, int(cast(LayerEnum, LAYER.M1_DRAW)), (1, 0), "M1_DRAW"
    )
    supported &= _matches_layer(
        layer_metal_top, int(cast(LayerEnum, LAYER.M2_DRAW)), (2, 0), "M2_DRAW"
    )
    supported &= _matches_layer(
        layer_etch, int(cast(LayerEnum, LAYER.M1_ETCH)), (1, 1), "M1_ETCH"
    )
    supported &= _matches_layer(
        layer_etch_top, int(cast(LayerEnum, LAYER.M2_ETCH)), (2, 1), "M2_ETCH"
    )
    with jax.ensure_compile_time_eval():
        model = _flipmon_fem_circuit()
    result = model(
        f=f,
        qubit={
            "chip_gap": chip_gap,
            "alignment": alignment,
            "inner_radius": inner_circle_radius,
            "top_radius": top_circle_radius,
            "inductance": jnp.asarray(inductance_nh) * 1e-9,
            "junction_capacitance": jnp.asarray(junction_capacitance_ff) * 1e-15,
            "z_ref": z_ref,
        },
    )
    return {
        ports: jnp.where(supported, values, jnp.nan) for ports, values in result.items()
    }
