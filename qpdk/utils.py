"""Helper functions for QPDK cells."""

from collections.abc import Iterable, Sequence
from itertools import starmap

import gdsfactory as gf
import klayout.db as kdb
from gdsfactory.component import Component
from gdsfactory.typings import Layer, LayerSpec
from klayout.db import DCplxTrans, Region

from qpdk.logger import logger
from qpdk.tech import LAYER, NON_METADATA_LAYERS


def transform_component(component: gf.Component, transform: DCplxTrans) -> gf.Component:
    """Applies a complex transformation to a component.

    For use with :func:`~gdsfactory.container`.

    Args:
        component: The component to transform.
        transform: The complex transformation to apply.

    Returns:
        The transformed component.
    """
    component.transform(transform)
    return component


def add_rect(
    c: Component,
    layer: LayerSpec,
    *,
    x0: float | None = None,
    x1: float | None = None,
    y0: float | None = None,
    y1: float | None = None,
    x_center: float | None = None,
    y_center: float | None = None,
    width: float | None = None,
    height: float | None = None,
) -> None:
    """Add a rectangle to component *c* using flexible coordinates.

    Coordinates can be specified using either (x0, x1) or (x_center, width),
    and similarly for y.

    Args:
        c: Component to add the rectangle to.
        layer: Layer specification for the rectangle.
        x0: Left x-coordinate.
        x1: Right x-coordinate.
        y0: Bottom y-coordinate.
        y1: Top y-coordinate.
        x_center: Center x-coordinate.
        y_center: Center y-coordinate.
        width: Width of the rectangle.
        height: Height of the rectangle.

    Raises:
        ValueError: If coordinate specification is incomplete or ambiguous.
    """
    if x0 is not None and x1 is not None:
        x_lo, x_hi = min(x0, x1), max(x0, x1)
    elif x_center is not None and width is not None:
        x_lo, x_hi = x_center - width / 2, x_center + width / 2
    else:
        raise ValueError("Provide (x0, x1) or (x_center, width)")

    if y0 is not None and y1 is not None:
        y_lo, y_hi = min(y0, y1), max(y0, y1)
    elif y_center is not None and height is not None:
        y_lo, y_hi = y_center - height / 2, y_center + height / 2
    else:
        raise ValueError("Provide (y0, y1) or (y_center, height)")

    c.add_polygon([(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)], layer=layer)


_EXCLUDE_LAYERS_DEFAULT_M1 = [
    (LAYER.M1_ETCH, 80),
    (LAYER.M1_DRAW, 80),
    (LAYER.WG, 80),
]
_EXCLUDE_LAYERS_DEFAULT_M2 = [
    (LAYER.M2_ETCH, 80),
    (LAYER.M2_DRAW, 80),
]


def fill_magnetic_vortices(
    component: Component | None = None,
    rectangle_size: tuple[float, float] = (15.0, 15.0),
    gap: float | tuple[float, float] = 15.0,
    stagger: float | tuple[float, float] = 3.0,
    exclude_layers: Iterable[tuple[LayerSpec, float]] | None = None,
    fill_layer: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    """Fill a component with small rectangles to trap magnetic vortices.

    This function fills the bounding box area of a given component with small etch
    rectangles in an array placed with specified gaps. The purpose is to trap
    local magnetic vortices in superconducting quantum circuits.

    This is a simple wrapper over :func:`~gdsfactory.Component.fill` which itself wraps
    the fill function from kfactory.

    Args:
        component: The component to fill with vortex trapping rectangles.
            If None, a default straight waveguide (:func:`gf.components.straight`)
            with length 100 µm is used.
        rectangle_size: Size of the fill rectangles in µm (width, height).
        gap: Gap between rectangles in µm.
            A tuple (x_gap, y_gap) can be provided for different gaps in x and y directions.
        stagger: Amount of staggering in µm to apply to pattern.
            A tuple (x_stagger, y_stagger) can be provided for different staggering in x and y.
        exclude_layers: Layers to ignore. Tuples of layer and keepout in µm.
            Defaults to M1_ETCH, M1_DRAW, and WG layers with 80 µm keepout.
        fill_layer: Layer for the fill rectangles.

    Returns:
        A new component with the original component plus fill rectangles.

    Example:
        >>> from qpdk.cells.resonator import resonator_quarter_wave
        >>> from qpdk.utils import fill_magnetic_vortices
        >>> resonator = resonator_quarter_wave()
        >>> filled_resonator = fill_magnetic_vortices(resonator)
        >>> # Or use with default component
        >>> filled_default = fill_magnetic_vortices()
    """
    # Use a default component if none is provided
    if component is None:
        component = gf.components.straight(length=100.0)

    c = gf.Component()
    c.add_ref(component)

    exclude_layers = exclude_layers or _EXCLUDE_LAYERS_DEFAULT_M1

    # Create the fill rectangle cell
    fill_cell = gf.components.rectangle(
        size=rectangle_size,
        layer=fill_layer,
    )

    gap_x, gap_y = (gap, gap) if isinstance(gap, int | float) else gap
    stagger_x, stagger_y = (
        (stagger, stagger) if isinstance(stagger, int | float) else stagger
    )

    c.fill(
        fill_cell=fill_cell,
        fill_regions=[
            (
                Region(c.bbox().to_itype(dbu=c.kcl.dbu)),
                0,
            )
        ],  # Fill the entire bounding box area
        exclude_layers=exclude_layers,
        row_step=gf.kf.kdb.DVector(rectangle_size[0] + gap_x, stagger_y),
        col_step=gf.kf.kdb.DVector(-stagger_x, rectangle_size[1] + gap_y),
    )

    return c


def merge_layers_with_etch(
    component: Component,
    draw_layer: LayerSpec,
    wg_layer: LayerSpec,
    etch_layer: LayerSpec | None,
) -> Component:
    """Merge waveguide marker layer with draw layer and create an etch negative.

    This function:

    1. Merges the waveguide (WG) marker layer shapes with the draw layer
       via boolean OR, producing a unified additive component.
    2. If `etch_layer` is provided, subtracts the merged additive shapes
       from the etch layer to produce a clean etch negative.
    3. Returns a fresh component containing the merged layers.

    This is used in capacitor components to combine the CPW cross-section
    waveguide markers with the capacitor metal draw layer and generate
    the corresponding etch layer.

    Args:
        component: The component containing both draw and WG layer shapes.
        draw_layer: The additive metal layer (e.g., M1_DRAW).
        wg_layer: The waveguide marker layer to merge into the draw layer.
        etch_layer: Optional etch layer for the negative mask.

    Returns:
        A new component with merged draw and (optionally) etch layers.
    """
    c_additive = gf.boolean(
        A=component,
        B=component,
        operation="or",
        layer=draw_layer,
        layer1=draw_layer,
        layer2=wg_layer,
    )
    result = gf.Component()
    result.absorb(result << c_additive)

    if etch_layer is not None:
        c_negative = gf.boolean(
            A=component,
            B=c_additive,
            operation="A-B",
            layer=etch_layer,
            layer1=etch_layer,
            layer2=draw_layer,
        )
        result.absorb(result << c_negative)

    return result


def subtract_draw_from_etch(
    component: Component,
    etch_shape: Component,
    etch_layer: LayerSpec,
    draw_layer: LayerSpec,
) -> None:
    """Subtract draw layer from an etch shape and absorb the result into a component.

    This is commonly used to create etch regions around qubit components where
    metal is preserved wherever the draw layer defines features, and the remaining
    area is etched away.

    Args:
        component: The target component to absorb the result into.
            Its draw layer shapes are subtracted from the etch shape.
        etch_shape: The component defining the full etch area (e.g., a bounding box).
        etch_layer: The etch layer for the result.
        draw_layer: The draw layer to subtract from the etch shape.
    """
    result = gf.boolean(
        A=etch_shape,
        B=component,
        operation="-",
        layer=etch_layer,
        layer1=etch_layer,
        layer2=draw_layer,
    )
    component.absorb(component.add_ref(result))


def apply_additive_metals(component: Component) -> Component:
    """Apply additive metal layers and remove them.

    Removes additive metal layers from etch layers, leading to a negative mask.

    TODO: Implement without flattening. Maybe with a KLayout dataprep script?

    Args:
        component: The component to apply additive metals to.

    Returns:
        Component with additive metals applied.
    """
    for additive, etch in (
        (LAYER.M1_DRAW, LAYER.M1_ETCH),
        (LAYER.M2_DRAW, LAYER.M2_ETCH),
    ):
        component_etch_only = gf.boolean(
            A=component,
            B=component,
            operation="-",
            layer=etch,
            layer1=etch,
            layer2=additive,
        )
        component.flatten()
        component.remove_layers([etch, additive])
        component << component_etch_only
    return component


def invert_mask_polarity(component: Component) -> Component:
    """Invert mask polarity of a component.

    Converts DRAW layers to ETCH layers by subtracting the DRAW layer from the
    component's bounding box, and similarly converts ETCH layers to DRAW layers.
    This is applied to M1 and M2 layers. All other layers are copied intact.

    Args:
        component: The component to invert.

    Returns:
        A new component with inverted mask polarity.
    """
    c = gf.Component()

    # Bounding box of the component defines the outer boundary for inversion
    bbox_region = Region(component.bbox().to_itype(component.kcl.dbu))

    affected_layers: set[int] = set()

    for additive, etch in (
        (LAYER.M1_DRAW, LAYER.M1_ETCH),
        (LAYER.M2_DRAW, LAYER.M2_ETCH),
    ):
        # Determine the layer indices in the layout object
        add_layer_index = component.kcl.layer(*additive)
        etch_layer_index = component.kcl.layer(*etch)
        affected_layers.update([add_layer_index, etch_layer_index])

        # Extract the shapes of the old component on these layers as regions
        add_region = Region(component.begin_shapes_rec(add_layer_index))
        etch_region = Region(component.begin_shapes_rec(etch_layer_index))

        # Skip if both regions are empty (no shapes on these layers)
        if add_region.is_empty() and etch_region.is_empty():
            logger.debug(
                "Skipping empty layers: {}, {} in component {}",
                additive,
                etch,
                component.name,
            )
            continue

        # Invert the polarities using the bounding box
        new_add_region = bbox_region - etch_region
        new_etch_region = bbox_region - add_region

        # Insert the inverted regions into the new component
        c.shapes(add_layer_index).insert(new_add_region)
        c.shapes(etch_layer_index).insert(new_etch_region)

    # Copy all other layers intact
    for layer_index in set(component.kcl.layer_indices()) - affected_layers:
        other_region = Region(component.begin_shapes_rec(layer_index))
        if not other_region.is_empty():
            c.shapes(layer_index).insert(other_region)

    return c


def add_margin_to_layer(
    component: Component, layer_margins: Sequence[tuple[Layer, float]]
) -> Component:
    """Increase the component bounding box by adding a margin to given draw layers.

    For each specified layer in the component, it is extended outwards
    by the specified margin. This effectively increases the bounding box of the
    component which can be useful to define the simulation area in HFSS.

    Args:
        component: The component to modify.
        layer_margins: Sequence of tuples containing the layer to modify and the margin to add in µm.

    Returns:
        A new component with extended layers.
    """
    c = gf.Component()

    bbox = component.bbox()

    # Identify layer indices for the specified margins
    layer_indices_margins = {
        component.kcl.layer(*layer): margin for layer, margin in layer_margins
    }

    # Copy existing shapes and track which specified layers are present
    present_layer_indices = set()
    for layer_index in component.kcl.layer_indices():
        region = Region(component.begin_shapes_rec(layer_index))
        if not region.is_empty():
            c.shapes(layer_index).insert(region)
            if layer_index in layer_indices_margins:
                present_layer_indices.add(layer_index)

    # Add margins to the layers that are present in the component
    bbox_itype = bbox.to_itype(component.kcl.dbu)
    bbox_region = Region(bbox_itype)

    for layer_index in present_layer_indices:
        margin = layer_indices_margins[layer_index]
        margin_dbu = int(margin / component.kcl.dbu)
        new_bbox = kdb.Box(
            bbox_itype.left - margin_dbu,
            bbox_itype.bottom - margin_dbu,
            bbox_itype.right + margin_dbu,
            bbox_itype.top + margin_dbu,
        )
        new_bbox_region = Region(new_bbox)
        margin_region = new_bbox_region - bbox_region
        c.shapes(layer_index).insert(margin_region)

    return c


def remove_metadata_layers(component: Component) -> Component:
    """Remove metadata layers from a component.

    Retains only physical and base layers:
    M1_DRAW, M1_ETCH, M2_DRAW, M2_ETCH, AB_DRAW, AB_VIA,
    JJ_AREA, JJ_PATCH, IND, TSV, DICE, ALN_TOP, ALN_BOT.

    All other layers are stripped out.

    Args:
        component: The component to clean.

    Returns:
        A new component with metadata layers removed.
    """
    # Convert allowed layers into kcl layer indices
    allowed_indices = set(starmap(component.kcl.layer, NON_METADATA_LAYERS))

    c = gf.Component()

    for layer_index in component.kcl.layer_indices():
        if layer_index in allowed_indices:
            other_region = Region(component.begin_shapes_rec(layer_index))
            if not other_region.is_empty():
                c.shapes(layer_index).insert(other_region)

    return c
