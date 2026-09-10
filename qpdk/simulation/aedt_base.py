"""Base AEDT simulation utilities using PyAEDT.

This module provides shared helper functions and a base class for AEDT
simulations (HFSS, Q3D, Q2D) from gdsfactory components.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gdsfactory as gf
from gdsfactory.technology.layer_stack import LayerLevel

from qpdk import LAYER_STACK, logger
from qpdk.tech import LAYER, material_properties
from qpdk.utils import (
    add_margin_to_layer,
    apply_additive_metals,
    invert_mask_polarity,
    remove_metadata_layers,
)

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss, Q2d
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack


def _get_layer_number_from_level(layer_level: LayerLevel) -> int | None:
    """Extract layer number from a LayerLevel's layer definition.

    Returns:
        The GDS layer number if available, else None.
    """
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
    """Convert a LayerStack to HFSS/Q3D GDS import mapping dictionary.

    Vacuum levels (``material == "vacuum"``) are skipped: the AEDT background
    region is already vacuum and any vacuum box is added explicitly with
    :meth:`qpdk.simulation.hfss.HFSS.add_air_region`, so importing vacuum as
    GDS geometry is redundant. This also resolves the Substrate/Vacuum
    collision on ``LAYER.SIM_AREA`` (98, 0).

    pyaedt's ``import_gds_3d`` keys the mapping on GDS layer number only — the
    GDS datatype is not part of the key — so two levels sharing a layer number
    (e.g. ``LAYER.AB_DRAW`` = (10, 0) and ``LAYER.AB_VIA`` = (10, 1)) are merged
    into a single entry when they share a material and their z-spans join into
    one box; otherwise an error is raised rather than silently importing wrong
    geometry. Levels sharing a layer number are processed in ascending
    ``zmin`` order so the result does not depend on the layer stack's
    insertion order.

    Returns:
        Dictionary mapping layer number to (elevation, thickness) tuple.

    Raises:
        ValueError: If two levels share a GDS layer number but cannot be merged
            into a single (elevation, thickness) entry because their materials
            differ (regardless of ``thickness_override``) or, with
            ``thickness_override=None``, their z-spans do not join into a
            single box.
    """
    if layer_stack is None:
        layer_stack = LAYER_STACK

    # Collect levels per layer number first, then process each number's levels
    # in zmin order so the result is independent of the layer stack's
    # insertion order.
    levels_by_number: dict[int, list[tuple[str, LayerLevel, float, float]]] = {}
    for name, layer_level in layer_stack.layers.items():
        if layer_level.material == "vacuum":
            logger.info(
                f"Skipping layer level {name!r} in GDS import mapping: vacuum is"
                " already the AEDT background region."
            )
            continue

        layer_number = _get_layer_number_from_level(layer_level)
        if layer_number is None:
            continue

        elevation = layer_level.zmin if layer_level.zmin is not None else 0.0
        thickness = (
            thickness_override
            if thickness_override is not None
            else (layer_level.thickness or 0.0)
        )
        levels_by_number.setdefault(layer_number, []).append((
            name,
            layer_level,
            elevation,
            thickness,
        ))

    mapping: dict[int, tuple[float, float]] = {}
    for layer_number, levels in levels_by_number.items():
        levels.sort(key=itemgetter(2))  # ascending zmin

        prev_name, prev_level, prev_elevation, prev_thickness = levels[0]
        mapping[layer_number] = (prev_elevation, prev_thickness)

        for name, layer_level, elevation, thickness in levels[1:]:
            if prev_level.material != layer_level.material:
                raise ValueError(
                    f"Layer levels {prev_name!r} ({prev_level.material!r}) and"
                    f" {name!r} ({layer_level.material!r}) share GDS layer"
                    f" {layer_number} but have different materials. pyaedt"
                    f" import_gds_3d keys the mapping on GDS layer number only"
                    f" (the datatype is discarded), so both would be imported as"
                    f" one object with a single material. Give them distinct GDS"
                    f" layer numbers or pass a custom layer_stack to"
                    f" import_component."
                )

            # Union of the two z-spans (um)
            span_low = min(prev_elevation, elevation)
            span_high = max(prev_elevation + prev_thickness, elevation + thickness)
            # Overlapping or touching intervals join into a single box
            max_low = max(prev_elevation, elevation)
            min_high = min(prev_elevation + prev_thickness, elevation + thickness)

            if thickness_override is not None:
                # Sheets/forced thickness: colliding levels cannot keep distinct
                # z-spans, so collapse to the lowest elevation and warn loudly.
                logger.warning(
                    f"Layer levels {prev_name!r} and {name!r} share GDS layer"
                    f" {layer_number} (pyaedt import_gds_3d keys on layer number"
                    f" only): collapsing to a single entry at elevation"
                    f" {span_low} um; the higher level's geometry is not"
                    f" represented separately."
                )
                mapping[layer_number] = (span_low, thickness)
            elif max_low > min_high + 1e-6:  # 1 pm tolerance in um
                raise ValueError(
                    f"Layer levels {prev_name!r} and {name!r} share GDS layer"
                    f" {layer_number} but their z-spans [{prev_elevation},"
                    f" {prev_elevation + prev_thickness}] and [{elevation},"
                    f" {elevation + thickness}] do not join into a single box."
                    f" pyaedt import_gds_3d keys the mapping on GDS layer number"
                    f" only (the datatype is discarded), so both would be imported"
                    f" with a single elevation and thickness. Give them distinct"
                    f" GDS layer numbers or pass a custom layer_stack to"
                    f" import_component."
                )
            else:
                logger.info(
                    f"Merging layer levels {prev_name!r} and {name!r}: both use"
                    f" GDS layer {layer_number} and their z-spans join into a"
                    f" single box (elevation {span_low} um, thickness"
                    f" {span_high - span_low} um)."
                )
                mapping[layer_number] = (span_low, span_high - span_low)

            prev_name = name
            prev_level = layer_level
            prev_elevation, prev_thickness = mapping[layer_number]

    return mapping


def prepare_component_for_aedt(
    component: Component,
    margin_draw: float = 0.0,
    margin_etch: float = 0.0,
) -> Component:
    """Prepare a component for AEDT simulation export.

    Returns:
        A copy of the component prepared for simulation.
    """
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


@contextmanager
def export_component_to_gds_temp(
    component: gf.Component,
    gds_path: str | Path | None = None,
    prefix: str = "qpdk_aedt_",
) -> Generator[Path, None, None]:
    """Context manager for exporting a component to a temporary GDS file.

    Yields:
        Path to the exported GDS file.
    """
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
    app: Any,
    new_objects: list[str],
    layer_stack: LayerStack,
) -> list[str]:
    """Rename imported GDS objects based on the layer stack.

    Returns:
        List of renamed object names.
    """
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
        """The AEDT modeler instance."""
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
        """Add a substrate box below the component geometry.

        Returns:
            Name of the created substrate object.
        """
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
