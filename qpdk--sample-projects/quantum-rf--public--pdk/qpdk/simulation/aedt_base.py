"""Base AEDT simulation utilities using PyAEDT.

This module provides shared helper functions and a base class for AEDT
simulations (HFSS, Q3D, Q2D) from gdsfactory components.
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gdsfactory as gf
from gdsfactory.technology.layer_stack import LayerLevel

from qpdk import LAYER_STACK
from qpdk.cells.helpers import (
    add_margin_to_layer,
    apply_additive_metals,
    invert_mask_polarity,
    remove_metadata_layers,
)
from qpdk.tech import LAYER

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss, Q2d
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack


def _get_layer_number_from_level(layer_level: LayerLevel) -> int | None:
    """Extract layer number from a LayerLevel's layer definition."""
    if hasattr(layer_level, "derived_layer") and layer_level.derived_layer is not None:
        derived = layer_level.derived_layer
        if hasattr(derived, "layer"):
            inner = derived.layer
            if hasattr(inner, "layer"):
                val = inner.layer
                if isinstance(val, tuple) and len(val) >= 1:
                    return int(val[0])
                return int(val)
            if isinstance(inner, tuple) and len(inner) >= 1:
                return int(inner[0])

    layer = layer_level.layer
    if isinstance(layer, tuple) and len(layer) >= 1:
        return int(layer[0])
    if hasattr(layer, "layer"):
        inner = layer.layer
        if isinstance(inner, tuple) and len(inner) >= 1:
            return int(inner[0])
        if hasattr(inner, "layer"):
            val = inner.layer
            if isinstance(val, tuple) and len(val) >= 1:
                return int(val[0])
            return int(val)
        return int(inner)
    return None


def layer_stack_to_gds_mapping(
    layer_stack: LayerStack | None = None,
    thickness_override: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Convert a LayerStack to HFSS/Q3D GDS import mapping dictionary."""
    if layer_stack is None:
        layer_stack = LAYER_STACK

    mapping: dict[int, tuple[float, float]] = {}

    for layer_level in layer_stack.layers.values():
        layer_number = _get_layer_number_from_level(layer_level)
        if layer_number is None:
            continue

        elevation = layer_level.zmin if layer_level.zmin is not None else 0.0
        thickness = (
            thickness_override
            if thickness_override is not None
            else (layer_level.thickness if layer_level.thickness else 0.0)
        )
        mapping[layer_number] = (elevation, thickness)

    return mapping


def prepare_component_for_aedt(
    component: Component,
    margin_draw: float = 0.0,
    margin_etch: float = 0.0,
) -> Component:
    """Prepare a component for AEDT simulation export."""
    c = gf.Component(name=f"{component.name}_aedt")
    c << component.copy()
    if margin_etch > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_ETCH, margin_etch),
                (LAYER.M2_ETCH, margin_etch),
            ],
        )
    c = apply_additive_metals(c)
    c = invert_mask_polarity(c)
    if margin_draw > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_DRAW, margin_draw),
                (LAYER.M2_DRAW, margin_draw),
            ],
        )
    c = c.remove_layers(layer for layer in LAYER if str(layer).endswith("_ETCH"))  # type: ignore[attr-defined]
    c = remove_metadata_layers(c)
    c.add_ports(component.ports)
    return c


@contextlib.contextmanager
def export_component_to_gds_temp(
    component: Component,
    gds_path: str | Path | None = None,
    prefix: str = "qpdk_aedt_",
) -> Generator[Path, None, None]:
    """Context manager for exporting a component to a temporary GDS file."""
    if gds_path is not None:
        path = Path(gds_path)
        component.write_gds(str(path))
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            path = Path(temp_dir) / "component.gds"
            component.write_gds(str(path))
            yield path


def rename_imported_objects(
    app: Any, new_objects: list[str], layer_stack: LayerStack
) -> list[str]:
    """Rename imported GDS objects based on the layer stack."""
    num_to_name = {}
    for name, level in layer_stack.layers.items():
        layer_num = _get_layer_number_from_level(level)
        if layer_num is not None and layer_num not in num_to_name:
            num_to_name[layer_num] = name

    renamed_objects = []
    for obj_name in new_objects:
        match = re.match(r"^signal(\d+)(_.*)?$", obj_name)
        new_name = obj_name
        if match:
            layer_num = int(match.group(1))
            suffix = match.group(2) or ""
            if layer_num in num_to_name:
                layer_name = num_to_name[layer_num]
                new_name = f"{layer_name}{suffix}"
                try:
                    app.modeler[obj_name].name = new_name
                except Exception:
                    new_name = obj_name
        renamed_objects.append(new_name)

    return renamed_objects


def add_materials_to_aedt(app: Hfss | Q2d | Q3d) -> None:
    """Add QPDK materials to the PyAEDT application."""
    from qpdk.tech import material_properties

    for name, props in material_properties.items():
        if app.materials.exists_material(name):
            continue

        mat = app.materials.add_material(name)

        for prop_name, prop_value in props.items():
            if prop_value == float("inf"):
                if prop_name == "relative_permittivity":
                    mat.conductivity = 1e30
                continue

            if prop_name == "relative_permittivity":
                mat.permittivity = prop_value
            elif prop_name == "conductivity":
                mat.conductivity = prop_value


class AEDTBase:
    """Base class for AEDT simulations."""

    def __init__(self, app: Hfss | Q2d | Q3d):
        """Initialize the AEDT base class.

        Args:
            app: The PyAEDT application instance.
        """
        self.app = app

    @property
    def modeler(self):
        """Return the AEDT modeler instance."""
        return self.app.modeler

    def add_materials(self) -> None:
        """Add QPDK materials to the AEDT project."""
        add_materials_to_aedt(self.app)

    def add_substrate(
        self,
        component: Component,
        thickness: float = 500.0,
        material: str = "silicon",
        name: str = "Substrate",
    ) -> str:
        """Add a substrate box below the component geometry."""
        bounds = component.bbox()
        x_min, y_min = bounds.p1.x, bounds.p1.y
        dx, dy = bounds.p2.x - x_min, bounds.p2.y - y_min

        substrate = self.modeler.create_box(
            origin=[x_min, y_min, -thickness],
            sizes=[dx, dy, thickness],
            name=name,
            material=material,
        )
        substrate.mesh_order = 4
        return substrate.name

    def save(self) -> None:
        """Save the AEDT project."""
        self.app.save_project()
