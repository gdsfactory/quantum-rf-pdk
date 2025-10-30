"""Helper functions for the qpdk package."""

from collections.abc import Sequence
from typing import cast

from gdsfactory import Component, ComponentAllAngle, LayerEnum, get_component
from gdsfactory.technology import LayerViews
from gdsfactory.typings import ComponentAllAngleSpec, ComponentSpec, Layer


def denest_layerviews_to_layer_tuples(
    layer_views: LayerViews,
) -> dict[str, tuple[int, int]]:
    """De-nest LayerViews into a flat dictionary of layer names to layer tuples.

    Args:
        layer_views: LayerViews object containing the layer views.

    Returns:
        Dictionary mapping layer names to their corresponding (layer, datatype) tuples.
    """

    def denest_layer_dict_recursive(items: dict) -> dict:
        """Recursively denest layer views to any depth.

        Args:
            items: Dictionary of layer view items to process

        Returns:
            Dictionary mapping layer names to layer objects
        """
        layers = {}

        for key, value in items.items():
            if value.group_members:
                # Recursively process nested group members and merge results
                nested_layers = denest_layer_dict_recursive(value.group_members)
                layers.update(nested_layers)
            else:
                # Base case: add the layer to our dictionary
                if hasattr(value, "layer"):
                    layers[key] = value.layer

        return layers

    # Start the recursive denesting process and return the result
    return denest_layer_dict_recursive(layer_views.layer_views)


def show_components(
    *args: ComponentSpec | ComponentAllAngleSpec,
    spacing: int = 200,
) -> Sequence[Component]:
    """Show sequence of components in a single layout in a line.

    The components are spaced based on the maximum width and height of the components.

    Args:
        *args: Component specifications to show.
        spacing: Extra spacing between components.

    Returns:
        Components after :func:`gdsfactory.get_component`.
    """
    from qpdk import PDK

    PDK.activate()

    components = [get_component(component_spec) for component_spec in args]
    any_all_angle = any(
        isinstance(component, ComponentAllAngle) for component in components
    )

    c = ComponentAllAngle() if any_all_angle else Component()

    max_component_width = max(component.size_info.width for component in components)
    max_component_height = max(component.size_info.height for component in components)
    if max_component_width > max_component_height:
        shift = (0, max_component_height + spacing)
    else:
        shift = (max_component_width + spacing, 0)

    for i, component in enumerate(components):
        (c << component).move(
            (
                shift[0] * i,
                shift[1] * i,
            )
        )
        label_offset = (
            shift[0] * i + (component.size_info.width / 2),
            shift[1] * i + (component.size_info.height / 2),
        )
        label_text = component.name if hasattr(component, "name") else f"component_{i}"
        c.add_label(
            text=label_text,
            position=label_offset,
            layer=cast(LayerEnum, PDK.layers).TEXT,
        )
    c.show()

    return components


def layerenum_to_tuple(layerenum: LayerEnum) -> Layer:
    """Convert a LayerEnum object to a tuple containing layer and datatype values.

    Args:
        layerenum: The LayerEnum object to convert.
    """
    return layerenum.layer, layerenum.datatype
