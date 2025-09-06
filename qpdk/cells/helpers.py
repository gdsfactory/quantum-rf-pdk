"""Helper functions for QPDK cells."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from klayout.db import DCplxTrans

from qpdk.tech import LAYER


def transform_component(component: gf.Component, transform: DCplxTrans) -> gf.Component:
    """Applies a complex transformation to a component.

    For use with :func:`~gdsfactory.container`.
    """
    return component.transform(transform)


def fill_magnetic_vortices(
    component: Component,
    rectangle_size: tuple[float, float] = (15.0, 15.0),
    gap: float = 15.0,
    keepout_margin: float = 80.0,
    fill_layer: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    """Fill a component with small rectangles to trap magnetic vortices.

    This function fills the bounding box area of a given component with small
    rectangles placed with specified gaps. The purpose is to trap local magnetic
    vortices in superconducting quantum circuits.

    Args:
        component: The component to fill with vortex trapping rectangles.
        rectangle_size: Size of the fill rectangles in µm (width, height).
            Defaults to (15.0, 15.0).
        gap: Gap between rectangles in µm. Defaults to 15.0.
        keepout_margin: Margin around existing features (WG, M1_DRAW, M1_ETCH)
            in µm. Defaults to 80.0.
        fill_layer: Layer for the fill rectangles. Defaults to M1_ETCH.

    Returns:
        A new component with the original component plus fill rectangles.

    Example:
        >>> from qpdk.cells.resonator import resonator_quarter_wave
        >>> from qpdk.cells.helpers import fill_magnetic_vortices
        >>> resonator = resonator_quarter_wave()
        >>> filled_resonator = fill_magnetic_vortices(resonator)
    """
    # Create a new component for the fill operation
    fill_container = Component()

    # Add the original component
    fill_container << component

    # Create the fill rectangle cell
    fill_cell = gf.components.rectangle(
        size=rectangle_size,
        layer=fill_layer,
    )

    # Define layers to exclude with keepout margins
    # Include WG (waveguide), M1_DRAW (positive metal), and M1_ETCH (negative etch)
    exclude_layers = [
        (LAYER.WG, keepout_margin),
        (LAYER.M1_DRAW, keepout_margin),
        (LAYER.M1_ETCH, keepout_margin),
    ]

    # Get the bounding box and create a large rectangle to define fill area
    bbox = component.bbox()
    margin = 50.0  # Extra margin around the component

    # Create a rectangle that defines the fill area using TEXT layer temporarily
    fill_area = gf.components.rectangle(
        size=(bbox.width() + 2 * margin, bbox.height() + 2 * margin),
        layer=LAYER.TEXT,
    )

    # Create a separate component just for the fill operation
    temp_fill_component = Component()
    temp_fill_component << fill_area
    temp_fill_component << component

    # Define the fill layers
    fill_layers = [
        (LAYER.TEXT, 0),  # Fill the text layer area
    ]

    # Apply the fill operation to the temporary component
    temp_fill_component.fill(
        fill_cell=fill_cell,
        fill_layers=fill_layers,
        exclude_layers=exclude_layers,
        x_space=gap,
        y_space=gap,
    )

    # Create the final component
    result = Component()

    # Add the original component
    result << component

    # Extract only the fill rectangles and add them to the result
    fill_polygons = temp_fill_component.get_polygons()
    if isinstance(fill_polygons, dict):
        # Polygons are returned as a dictionary with layer keys
        for layer_key, polygon_list in fill_polygons.items():
            if layer_key == fill_layer or (
                hasattr(fill_layer, "__iter__") and layer_key in fill_layer
            ):
                for polygon in polygon_list:
                    result.add_polygon(polygon, layer=fill_layer)
    else:
        # Handle if polygons are returned as a list
        for polygon in fill_polygons:
            if hasattr(polygon, "layer") and polygon.layer == fill_layer:
                result.add_polygon(polygon.points, layer=fill_layer)

    # Copy ports from the original component
    for port in component.ports:
        result.add_port(name=port.name, port=port)

    return result
