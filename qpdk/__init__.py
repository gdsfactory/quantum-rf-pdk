"""Quantum pdk."""

import importlib
import inspect
import pkgutil
from functools import lru_cache, partial
from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

import qpdk.samples
from qpdk import cells, config, helper, tech
from qpdk.config import PATH
from qpdk.logger import logger
from qpdk.tech import (
    LAYER,
    LAYER_CONNECTIVITY,
    LAYER_STACK,
    LAYER_VIEWS,
    routing_strategies,
)

gf.CONF.layer_error_path = LAYER.ERROR_PATH

try:
    from .models import models as _models
except ImportError as e:
    logger.warning(
        f"QPDK models could not be loaded ({e}). "
        "Ensure dependencies are installed with `pip install qpdk[models]`."
    )
    _models = {}

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
        models=_models,
        routing_strategies=routing_strategies,
        connectivity=LAYER_CONNECTIVITY,
    )


PDK = get_pdk()


@lru_cache(maxsize=1)
def get_sample_functions() -> dict[str, Any]:
    """Lazily discover and return all sample component functions.

    Walks ``qpdk.samples`` sub-modules and collects every public function
    and :class:`~functools.partial` whose defining module matches the
    discovered module.  Results are cached so the cost is paid at most once.
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
    "cells",
    "config",
    "get_sample_functions",
    "helper",
    "logger",
    "tech",
]
__version__ = "0.3.6"
