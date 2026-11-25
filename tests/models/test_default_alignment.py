"""Regression tests that SAX model defaults match the layout cell defaults.

When a schematic omits a property, the simulated circuit must use the same
default the layout does. See issue #822.

Cases are derived from the full analytical catalog, not the PDK registry, so
de-registering a model cannot silently drop its guard.
"""

import inspect
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from kfactory.serialization import clean_value

from qpdk import PDK
from qpdk.models import models as sax_models
from qpdk.models.readout_assembly import (
    _NOMINAL_MEANDER_START,
    _coupon_for,
    _nominal_plate_string,
)

# The readout models keep ``None`` sentinels for settings whose concrete
# default follows the selected qubit coupon.
_READOUT_MODELS = {
    "transmon_with_resonator_and_probeline",
    "double_pad_transmon_with_resonator_and_probeline",
    "flipmon_with_resonator_and_probeline",
}

_READOUT_SENTINELS = frozenset({
    "resonator_meander_start",
    "coupler",
    "coupler_offset",
    "coupler_port",
    "qubit_rotation",
})


def _model_for(component: str) -> Callable[..., Any]:
    """Return the registered model, or the analytical catalog entry for it.

    De-registered models keep their default-alignment guard through the
    analytical catalog.
    """
    return PDK.models[component] if component in PDK.models else sax_models[component]


def _shared_defaults() -> list[tuple[str, str]]:
    pairs = []
    components = PDK.cells.keys() & (PDK.models.keys() | sax_models.keys())
    for component in sorted(components):
        layout_parameters = inspect.signature(PDK.cells[component]).parameters
        model_parameters = inspect.signature(_model_for(component)).parameters
        shared = (layout_parameters.keys() & model_parameters.keys()) - {"f"}
        pairs.extend((component, parameter) for parameter in sorted(shared))
    return pairs


SHARED_DEFAULTS = _shared_defaults()


def _default[F: Callable[..., Any]](factory: F, parameter: str) -> Any:
    """Return the default value of *parameter* in *factory*'s signature."""
    return inspect.signature(factory).parameters[parameter].default


def _comparable_default(parameter: str, value: Any) -> Any:
    match parameter, value:
        case str(name), default if (
            name.startswith("cross_section") and default is not None
        ):
            return PDK.get_cross_section(default).model_dump(exclude_none=True)
        case "coupler", default:
            return clean_value(default)
        case _:
            return value


def _readout_semantic_default(component: str, parameter: str) -> Any:
    """Resolve a readout sentinel through the coupon's modeled values."""
    coupon = _coupon_for(_default(_model_for(component), "qubit"))
    return {
        "resonator_meander_start": _NOMINAL_MEANDER_START,
        "coupler": _nominal_plate_string(coupon),
        "coupler_offset": coupon.plate_offset,
        "coupler_port": coupon.coupler_port,
        "qubit_rotation": coupon.rotation,
    }[parameter]


def _model_default(component: str, parameter: str) -> Any:
    """Return the model default, resolving readout sentinels semantically."""
    value = _default(_model_for(component), parameter)
    if (
        value is None
        and parameter in _READOUT_SENTINELS
        and component in _READOUT_MODELS
    ):
        value = _readout_semantic_default(component, parameter)
    return _comparable_default(parameter, value)


def _shared_default_id(shared_default: tuple[str, str]) -> str:
    return f"{shared_default[0]}.{shared_default[1]}"


@pytest.mark.parametrize(
    "shared_default",
    SHARED_DEFAULTS,
    ids=_shared_default_id,
)
def test_layout_model_default_alignment(shared_default: tuple[str, str]) -> None:
    """A shared setting must default to the layout value in the SAX model."""
    component, parameter = shared_default
    layout_default = _comparable_default(
        parameter, _default(PDK.cells[component], parameter)
    )
    model_default = _model_default(component, parameter)
    assert model_default == layout_default, (
        f"{component}.{parameter}: model default {model_default!r} does not match "
        f"layout default {layout_default!r}"
    )


@pytest.mark.parametrize("component", sorted(_READOUT_MODELS))
def test_readout_omitted_sentinel_matches_explicit_layout_value(
    component: str,
) -> None:
    """Omitting a readout sentinel behaves like passing the layout default."""
    model = _model_for(component)
    layout_parameters = inspect.signature(PDK.cells[component]).parameters
    explicit = {
        parameter: layout_parameters[parameter].default
        for parameter in _READOUT_SENTINELS
        if parameter in layout_parameters
    }
    frequency = jnp.asarray((5e9,))
    omitted = model(f=frequency)
    given = model(f=frequency, **explicit)
    assert set(omitted) == set(given)
    for key in omitted:
        np.testing.assert_array_equal(omitted[key], given[key])
