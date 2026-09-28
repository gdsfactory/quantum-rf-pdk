"""QPDK Ansys Electronics Desktop (AEDT) simulations built on PyAEDT.

:mod:`qpdk.simulation.ansys.base` holds the helpers shared by every AEDT
design: component preparation, layer-stack mapping, materials, and the
:class:`~qpdk.simulation.ansys.base.AEDTBase` wrapper.
:mod:`qpdk.simulation.ansys.hfss` wraps HFSS eigenmode and driven modal
designs, and :mod:`qpdk.simulation.ansys.q3d` wraps Q3D Extractor and 2D
Extractor (Q2D).

Note:
    Running a design needs the optional ``hfss`` extra
    (``uv sync --extra hfss``) and a local AEDT installation. PyAEDT is only
    imported for type checking and inside the methods that talk to AEDT, and
    the design wrappers (which also need polars) are exposed lazily, so
    importing this package and :mod:`~qpdk.simulation.ansys.base` works
    without the extra.

See the `PyAEDT documentation <https://aedt.docs.pyansys.com/>`_.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qpdk.simulation.ansys.base import (
    AEDTBase,
    add_materials_to_aedt,
    detach_desktop_logging,
    fit_view,
    layer_stack_to_gds_mapping,
    object_names_to_materials,
    prepare_component_for_aedt,
)

if TYPE_CHECKING:
    from qpdk.simulation.ansys.hfss import HFSS, lumped_port_rectangle_from_cpw
    from qpdk.simulation.ansys.q3d import Q2D, Q3D

_LAZY_IMPORTS: dict[str, str] = {
    "HFSS": "qpdk.simulation.ansys.hfss",
    "lumped_port_rectangle_from_cpw": "qpdk.simulation.ansys.hfss",
    "Q2D": "qpdk.simulation.ansys.q3d",
    "Q3D": "qpdk.simulation.ansys.q3d",
}

__all__ = [
    "HFSS",
    "Q2D",
    "Q3D",
    "AEDTBase",
    "add_materials_to_aedt",
    "detach_desktop_logging",
    "fit_view",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "object_names_to_materials",
    "prepare_component_for_aedt",
]


def __getattr__(name: str) -> Any:
    """Import a design wrapper only when it is asked for.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
