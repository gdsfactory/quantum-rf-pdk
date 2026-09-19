"""Regression tests that SAX model defaults match the layout cell defaults.

When a schematic omits a property, the simulated circuit must use the same
default the layout does. See issue #822.

Cases are derived from the full analytical catalog, not the PDK registry, so
de-registering a model cannot silently drop its guard.
"""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from qpdk import PDK
from qpdk.models import models as sax_models


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
        case _:
            return value


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
    model_default = _comparable_default(
        parameter, _default(_model_for(component), parameter)
    )
    assert model_default == layout_default, (
        f"{component}.{parameter}: model default {model_default!r} does not match "
        f"layout default {layout_default!r}"
    )
