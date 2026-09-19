"""Quantum pdk."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk
from gdsfactory.typings import ComponentFactory

import qpdk.samples
from qpdk import cells, config, helper, tech
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

# Add a cell factory here when it uses another cell's SAX model unchanged.
SAX_MODEL_ALIASES = {
    "resonator_quarter_wave_bend_start": "resonator_quarter_wave",
    "resonator_quarter_wave_bend_end": "resonator_quarter_wave",
    "resonator_quarter_wave_bend_both": "resonator_quarter_wave",
    "resonator_half_wave_bend_start": "resonator_half_wave",
    "resonator_half_wave_bend_end": "resonator_half_wave",
    "resonator_half_wave_bend_both": "resonator_half_wave",
}


# Analytical models whose simulation ports do not correspond to the ports of
# their same-named layout cell. They stay importable from ``qpdk.models.models``
# but are left out of the PDK registry. ``interdigital_capacitor`` is here
# because one factory name has two port contracts, chosen by its ``half``
# setting, so no single boundary can describe it.
MODELS_WITHOUT_LAYOUT_PORTS = frozenset({
    "airbridge",
    "double_pad_transmon_with_resonator",
    "flipmon",
    "flipmon_with_bbox",
    "flipmon_with_resonator",
    "indium_bump",
    "interdigital_capacitor",
    "rectangle",
    "squid_junction",
    "straight_double_open",
    "transmon_with_resonator",
    "tsv",
    "xmon_transmon",
})


def _build_pdk_models(
    models: Mapping[str, Callable[..., Any]],
    overrides: Mapping[str, Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    registered = {
        name: model
        for name, model in models.items()
        if name not in MODELS_WITHOUT_LAYOUT_PORTS
    }
    registered.update(overrides)
    registered.update({
        alias: registered[model_name] for alias, model_name in SAX_MODEL_ALIASES.items()
    })
    return registered


try:
    from .models import _PDK_MODEL_OVERRIDES, models as _models
except ImportError as e:
    logger.warning(
        f"QPDK models could not be loaded ({e}). "
        "Ensure dependencies are installed with `pip install qpdk[models]`."
    )
    _models = {}
else:
    _models = _build_pdk_models(_models, _PDK_MODEL_OVERRIDES)

_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)


# Under the pre-commit hook's --ignore missing-import, pyrefly cannot resolve
# type(Pdk) and rejects it as a base class
class _QPdkMeta(SingletonMeta, type(Pdk)):  # pyrefly: ignore[invalid-inheritance]
    """Singleton semantics layered on pydantic's model metaclass."""


class QPdk(Pdk, metaclass=_QPdkMeta):
    """Pdk subclass whose every construction returns the same instance."""


def get_pdk() -> Pdk:
    """Return Quantum PDK."""
    return QPdk(
        name="qpdk",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore[arg-type]
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
    "MODELS_WITHOUT_LAYOUT_PORTS",
    "PATH",
    "SAX_MODEL_ALIASES",
    "cells",
    "config",
    "get_sample_functions",
    "helper",
    "logger",
    "tech",
]
__version__ = "0.4.0"
