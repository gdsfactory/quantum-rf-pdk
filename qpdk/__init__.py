"""Quantum pdk."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from operator import attrgetter
from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk
from gdsfactory.read.from_yaml_template import cell_from_yaml_template
from gdsfactory.typings import ComponentFactory

import qpdk.samples
from qpdk import cells, config, helper, tech
from qpdk.cells import _schematic
from qpdk.config import PATH
from qpdk.logger import logger
from qpdk.singleton import SingletonMeta
from qpdk.tech import (
    LAYER,
    LAYER_CONNECTIVITY,
    LAYER_STACK,
    LAYER_VIEWS,
    routing_strategies,
)

gf.CONF.layer_error_path = LAYER.ERROR_PATH


def _get_cell_schematic(name: str, cell: Callable[..., Any]) -> Any:
    """Read a cell schematic without constructing its layout."""
    settings = cell.keywords if isinstance(cell, partial) else {}
    function = cell.func if isinstance(cell, partial) else cell
    provider = getattr(_schematic, f"{name}_schematic", None)
    if isinstance(cell, partial) and provider is not None:
        return provider(**settings)
    factory = gf.kcl.factories.get_by_qualified_name(
        f"{function.__module__}.{function.__qualname__}"
    )
    if factory is not None and factory.schematic_driven():
        return factory.get_schematic(**settings)
    # Virtual cells and partial aliases cannot declare schematic_function.
    return provider(**settings) if provider is not None else None


def _schematic_model_bindings(
    cell_factories: Mapping[str, Callable[..., Any]],
) -> dict[str, dict[str, Any]]:
    """Read declared SAX boundaries without constructing layouts or evaluating models."""
    bindings = {}
    for name, cell in cell_factories.items():
        schematic = _get_cell_schematic(name, cell)
        if schematic is None:
            continue
        info: Any = schematic.info
        for descriptor in info.get("models", []):
            if descriptor["language"] == "sax":
                bindings[name] = descriptor
                break
    return bindings


def _register_sax_models(
    catalog: Mapping[str, Callable[..., Any]],
    cell_factories: Mapping[str, Callable[..., Any]],
    bindings: Mapping[str, dict[str, Any]],
) -> dict[str, Callable[..., Any]]:
    """Combine circuit primitives with declared layout models."""
    registered = {
        name: model for name, model in catalog.items() if name not in cell_factories
    }
    for name, descriptor in bindings.items():
        module = importlib.import_module(descriptor["module"])
        model = attrgetter(descriptor["qualname"])(module)
        registered[descriptor["name"]] = model
        cell = cell_factories[name]
        if isinstance(cell, partial):
            parameters = inspect.signature(model).parameters
            settings = {
                key: value
                for key, value in cell.keywords.items()
                if key in parameters and value != parameters[key].default
            }
            if settings:
                model = partial(model, **settings)
        registered[name] = model
    return registered


_cells = get_cells(cells)
_cells.update({
    sample_name: cell_from_yaml_template(
        sample_path, name=sample_name, routing_strategies=routing_strategies
    )
    for sample_path in PATH.samples.glob("*.pic.yml")
    if (sample_name := sample_path.name.removesuffix(".pic.yml")) not in _cells
})
_model_bindings = _schematic_model_bindings(_cells)
SAX_MODEL_ALIASES = {
    name: descriptor["name"]
    for name, descriptor in _model_bindings.items()
    if name != descriptor["name"]
}

try:
    from .models import models as _analytical_models
except ImportError as e:
    logger.warning(
        f"QPDK models could not be loaded ({e}). "
        "Ensure dependencies are installed with `pip install qpdk[models]`."
    )
    _models = {}
else:
    _models = _register_sax_models(_analytical_models, _cells, _model_bindings)

_cross_sections = get_cross_sections(tech)


class _QPdkMeta(SingletonMeta, type(Pdk)):
    """Singleton semantics layered on pydantic's model metaclass."""


class QPdk(Pdk, metaclass=_QPdkMeta):
    """Pdk subclass whose every construction returns the same instance."""


def get_pdk() -> Pdk:
    """Return Quantum PDK."""
    return QPdk(
        name="qpdk",
        cells=_cells,
        cross_sections=_cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
        connectivity=LAYER_CONNECTIVITY,
    )


PDK = get_pdk()


@lru_cache(maxsize=1)
def get_sample_functions() -> dict[str, ComponentFactory]:
    """Lazily discover and return all sample component functions.

    Walks ``qpdk.samples`` sub-modules and collects every public function
    and :class:`~functools.partial` whose defining module matches the
    discovered module.  Results are cached so the cost is paid at most once.

    Returns:
        A mapping from qualified names to component factory callables.
    """
    return {
        f"{modname}.{name}": obj
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            qpdk.samples.__path__, qpdk.samples.__name__ + "."
        )
        for name, obj in inspect.getmembers(importlib.import_module(modname))
        if (inspect.isfunction(obj) or isinstance(obj, partial))
        and not name.startswith("_")
        # Compare .func if exists (for partials), otherwise obj itself
        and getattr(obj, "func", obj).__module__ == modname
    }


__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "SAX_MODEL_ALIASES",
    "cells",
    "config",
    "get_sample_functions",
    "helper",
    "logger",
    "tech",
]
__version__ = "0.4.1"
