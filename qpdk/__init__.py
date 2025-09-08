"""Quantum pdk."""

import importlib
import inspect
import pkgutil
from functools import lru_cache

from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

import qpdk.samples
from qpdk import cells, config, tech
from qpdk.config import PATH

# from qpdk.models import get_models
from qpdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

# _models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)


@lru_cache
def get_pdk() -> Pdk:
    """Return Quantum PDK."""
    return Pdk(
        name="qpdk",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        # models=_models,
        routing_strategies=routing_strategies,
    )


PDK = get_pdk()

# Get all functions from qpdk.samples module
sample_functions = {
    f"{modname}.{name}": obj
    for importer, modname, ispkg in pkgutil.walk_packages(
        qpdk.samples.__path__, qpdk.samples.__name__ + "."
    )
    for name, obj in inspect.getmembers(importlib.import_module(modname))
    if inspect.isfunction(obj)
    and not name.startswith("_")
    and obj.__module__ == modname
}

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
__version__ = "0.0.2"
