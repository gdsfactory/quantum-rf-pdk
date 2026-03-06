"""Helper functions for QPDK cells."""

from collections.abc import Iterable

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from klayout.db import DCplxTrans, Region

from qpdk.tech import LAYER


def transform_component(component: gf.Component, transform: DCplxTrans) -> gf.Component:
    """Applies a complex transformation to a component.

    For use with :func:`~gdsfactory.container`.
    """
    component.transform(transform)
    return component


_EXCLUDE_LAYERS_DEFAULT_M1 = [
    (LAYER.M1_ETCH, 80),
    (LAYER.M1_DRAW, 80),
    (LAYER.WG, 80),
]
_EXCLUDE_LAYERS_DEFAULT_M2 = [
    (LAYER.M2_ETCH, 80),
    (LAYER.M2_DRAW, 80),
]


@gf.cell
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
        >>> from qpdk.cells.helpers import fill_magnetic_vortices
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


def apply_additive_metals(component: Component) -> Component:
    """Apply additive metal layers and remove them.

    Removes additive metal layers from etch layers, leading to a negative mask.

    TODO: Implement without flattening. Maybe with a KLayout dataprep script?
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


@gf.cell
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

    affected_layers = []

    for additive, etch in (
        (LAYER.M1_DRAW, LAYER.M1_ETCH),
        (LAYER.M2_DRAW, LAYER.M2_ETCH),
    ):
        # Determine the layer indices in the layout object
        add_layer_index = component.kcl.layer(*additive)
        etch_layer_index = component.kcl.layer(*etch)
        affected_layers.extend([add_layer_index, etch_layer_index])

        # Extract the shapes of the old component on these layers as regions
        add_region = Region(component.begin_shapes_rec(add_layer_index))
        etch_region = Region(component.begin_shapes_rec(etch_layer_index))

        # Invert the polarities using the bounding box
        new_add_region = bbox_region - etch_region
        new_etch_region = bbox_region - add_region

        # Insert the inverted regions into the new component
        c.shapes(add_layer_index).insert(new_add_region)
        c.shapes(etch_layer_index).insert(new_etch_region)

    # Copy all other layers intact
    for layer_index in component.kcl.layer_indices():
        if layer_index not in affected_layers:
            other_region = Region(component.begin_shapes_rec(layer_index))
            if not other_region.is_empty():
                c.shapes(layer_index).insert(other_region)

    return c


@gf.cell
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
    allowed_layers = {
        LAYER.M1_DRAW,
        LAYER.M1_ETCH,
        LAYER.M2_DRAW,
        LAYER.M2_ETCH,
        LAYER.AB_DRAW,
        LAYER.AB_VIA,
        LAYER.JJ_AREA,
        LAYER.JJ_PATCH,
        LAYER.IND,
        LAYER.TSV,
        LAYER.DICE,
        LAYER.ALN_TOP,
        LAYER.ALN_BOT,
    }

    # Convert allowed layers into kcl layer indices
    allowed_indices = {component.kcl.layer(*layer) for layer in allowed_layers}

    c = gf.Component()

    for layer_index in component.kcl.layer_indices():
        if layer_index in allowed_indices:
            other_region = Region(component.begin_shapes_rec(layer_index))
            if not other_region.is_empty():
                c.shapes(layer_index).insert(other_region)

    return c
