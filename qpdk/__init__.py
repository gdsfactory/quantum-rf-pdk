"""Quantum pdk."""

from functools import lru_cache

from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from qpdk import cells, config, tech
from qpdk.config import PATH

# from qpdk.models import get_models
from qpdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

# _models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)


@lru_cache
def get_pdk() -> Pdk:
    """Return Cornerstone PDK."""
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
