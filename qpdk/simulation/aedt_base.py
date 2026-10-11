"""Base AEDT simulation utilities, with QPDK defaults.

The generic AEDT code lives in :mod:`gplugins.ansys`. This module wraps it so that
layer stacks default to QPDK's :data:`~qpdk.tech.LAYER_STACK` and materials to
:data:`~qpdk.tech.material_properties`, keeps the wrapper classes singletons, and
holds the QPDK-specific :func:`prepare_component_for_aedt`.

gplugins is imported on first use, so :func:`prepare_component_for_aedt` works
without the ``hfss`` extra installed. Using anything that needs gplugins without
it raises :class:`ImportError` saying how to install it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qpdk import LAYER_STACK
from qpdk.simulation.layout import prepare_metal_layout
from qpdk.tech import material_properties as _qpdk_material_properties

if TYPE_CHECKING:
    from types import ModuleType

    from ansys.aedt.core import Hfss, Q2d
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack
    from gplugins.ansys.base import (
        MaterialProperties,
        detach_desktop_logging,
        export_component_to_gds_temp,
        fit_view,
        rename_imported_objects,
    )

    from qpdk.simulation._aedt import AEDTBase

__all__ = [
    "AEDTBase",
    "add_materials_to_aedt",
    "detach_desktop_logging",
    "export_component_to_gds_temp",
    "fit_view",
    "layer_stack_to_gds_mapping",
    "object_names_to_materials",
    "prepare_component_for_aedt",
    "rename_imported_objects",
]

MISSING_HFSS_EXTRA = (
    "The QPDK AEDT wrappers need the gplugins Ansys plugin. Install it with "
    "`uv sync --extra hfss` (or `pip install qpdk[hfss]`)."
)

_LAZY_IMPORTS: dict[str, str] = {
    "AEDTBase": "qpdk.simulation._aedt",
    "detach_desktop_logging": "gplugins.ansys.base",
    "export_component_to_gds_temp": "gplugins.ansys.base",
    "fit_view": "gplugins.ansys.base",
    "rename_imported_objects": "gplugins.ansys.base",
}


def __getattr__(name: str) -> Any:
    """Import a gplugins-backed public name on first access.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(_import_gplugins_backed(module_name), name)


def _import_gplugins_backed(module_name: str) -> ModuleType:
    """Import a module that needs gplugins, with an actionable error if it is missing.

    Returns:
        The imported module.

    Raises:
        ImportError: If gplugins or its Ansys plugin is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise ImportError(MISSING_HFSS_EXTRA) from error


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(set(globals()) | set(__all__))


def layer_stack_to_gds_mapping(
    layer_stack: LayerStack | None = None,
    thickness_override: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Convert a LayerStack to an HFSS/Q3D GDS import mapping.

    See :func:`gplugins.ansys.layer_stack_to_gds_mapping`; this only defaults
    ``layer_stack`` to QPDK's ``LAYER_STACK``.

    Returns:
        Dictionary mapping layer number to (elevation, thickness) tuple.
    """
    gp_base = _import_gplugins_backed("gplugins.ansys.base")
    return gp_base.layer_stack_to_gds_mapping(
        LAYER_STACK if layer_stack is None else layer_stack,
        thickness_override=thickness_override,
    )


def prepare_component_for_aedt(
    component: Component,
    margin_draw: float = 0.0,
    margin_etch: float = 0.0,
    *,
    name: str | None = None,
) -> Component:
    """Prepare a component for AEDT simulation export.

    Thin backward-compatible wrapper around
    :func:`qpdk.simulation.layout.prepare_metal_layout`, keeping the historical
    ``<component>_aedt`` name for the initial preparation cell.

    Returns:
        A copy of the component prepared for simulation.
    """
    return prepare_metal_layout(
        component,
        margin_draw=margin_draw,
        margin_etch=margin_etch,
        name=name or f"{component.name}_aedt",
    )


def object_names_to_materials(
    object_names: list[str],
    layer_stack: LayerStack,
    material_properties: MaterialProperties | None = None,
) -> dict[str, str]:
    """Map imported object names to AEDT material names.

    See :func:`gplugins.ansys.object_names_to_materials`; this only defaults
    ``material_properties`` to QPDK's. To match a wrapper built with its own
    table, pass that wrapper's ``material_properties``.

    Returns:
        Dictionary mapping object names to AEDT material names.
    """
    gp_base = _import_gplugins_backed("gplugins.ansys.base")
    return gp_base.object_names_to_materials(
        object_names,
        layer_stack,
        _qpdk_material_properties
        if material_properties is None
        else material_properties,
    )


def add_materials_to_aedt(
    app: Hfss | Q2d | Q3d,
    material_properties: MaterialProperties | None = None,
) -> None:
    """Add materials to the PyAEDT application.

    See :func:`gplugins.ansys.add_materials_to_aedt`; this only defaults
    ``material_properties`` to QPDK's. To match a wrapper built with its own
    table, pass that wrapper's ``material_properties``.
    """
    gp_base = _import_gplugins_backed("gplugins.ansys.base")
    gp_base.add_materials_to_aedt(
        app,
        _qpdk_material_properties
        if material_properties is None
        else material_properties,
    )
