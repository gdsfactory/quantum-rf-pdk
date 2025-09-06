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
    keepout_margin: float = 20.0,  # Reduced from 80.0 to be more reasonable
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
            in µm. Defaults to 20.0.
        fill_layer: Layer for the fill rectangles. Defaults to M1_ETCH.

    Returns:
        A new component with the original component plus fill rectangles.

    Example:
        >>> from qpdk.cells.resonator import resonator_quarter_wave
        >>> from qpdk.cells.helpers import fill_magnetic_vortices
        >>> resonator = resonator_quarter_wave()
        >>> filled_resonator = fill_magnetic_vortices(resonator)
    """
    # Create the fill rectangle cell
    fill_cell = gf.components.rectangle(
        size=rectangle_size,
        layer=fill_layer,
    )

    # Define layers to exclude with keepout margins
    # Only exclude M1_DRAW (positive metal) by default to allow fill around waveguides
    exclude_layers = [
        (LAYER.M1_DRAW, keepout_margin),
    ]

    # Get the component's bounding box
    bbox = component.bbox()
    margin = 50.0  # Extra margin around the component

    # Create a rectangle that defines the fill area
    # Position it at the origin first, we'll position it later
    fill_area = gf.components.rectangle(
        size=(bbox.width() + 2 * margin, bbox.height() + 2 * margin),
        layer=LAYER.TEXT,
    )

    # Create a temporary component for ONLY the fill operation
    # We need to position everything correctly relative to the original component
    temp_fill_component = Component()
    
    # Add the fill area, but position it so it's centered on the component's bbox
    center_point = bbox.center()
    # Calculate where to put the fill area so it centers on the component
    fill_area_x = center_point.x - (bbox.width() + 2 * margin) / 2
    fill_area_y = center_point.y - (bbox.height() + 2 * margin) / 2
    
    fill_area_ref = temp_fill_component.add_ref(fill_area)
    fill_area_ref.move((fill_area_x, fill_area_y))
    
    # Add the original component at its exact position for exclusion zones
    original_ref = temp_fill_component.add_ref(component)

    # Define the fill layers - we want to fill the TEXT layer area
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

    # Now create the final result component
    result = Component()

    # Add the original component first
    result.add_ref(component)

    # Now we need to extract only the fill rectangles from temp_fill_component
    # After the fill operation, temp_fill_component contains:
    # 1. The fill area rectangle (on TEXT layer) - we don't want this
    # 2. The original component (on various layers) - we don't want this 
    # 3. The fill rectangles (on fill_layer) - we want only these
    
    temp_polygons = temp_fill_component.get_polygons()
    
    if isinstance(temp_polygons, dict) and fill_layer in temp_polygons:
        all_fill_layer_polygons = temp_polygons[fill_layer]
        
        # Get original component polygons on this layer for comparison
        original_polygons = component.get_polygons()
        original_layer_polygons = original_polygons.get(fill_layer, []) if isinstance(original_polygons, dict) else []
        
        # Simple approach: the fill rectangles should be exactly the size we specified
        # and positioned in a grid pattern. Original component polygons will have different sizes.
        target_width, target_height = rectangle_size
        # Convert to database units (typically 1 µm = 1000 database units)
        target_width_db = target_width * 1000
        target_height_db = target_height * 1000
        tolerance = 1.0  # 1 database unit tolerance
        
        for polygon in all_fill_layer_polygons:
            # Check if this polygon is the right size to be a fill rectangle
            if hasattr(polygon, 'bbox'):
                poly_bbox = polygon.bbox()
                poly_width = poly_bbox.width()
                poly_height = poly_bbox.height()
                
                # Check if this polygon matches our fill rectangle size
                is_fill_rect = (
                    abs(poly_width - target_width_db) < tolerance and 
                    abs(poly_height - target_height_db) < tolerance
                )
                
                if is_fill_rect:
                    result.add_polygon(polygon, layer=fill_layer)

    # Copy ports from the original component
    for port in component.ports:
        result.add_port(name=port.name, port=port)

    return result
