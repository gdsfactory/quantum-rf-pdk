"""Helper functions for the qpdk package."""

from collections.abc import Sequence

from gdsfactory.technology import LayerViews
from gdsfactory.typings import ComponentAllAngleSpec, ComponentSpec

from gdsfactory import Component, ComponentAllAngle, get_component


def denest_layerviews_to_layer_tuples(
    layer_views: LayerViews,
) -> dict[str, tuple[int, int]]:
    """De-nest LayerViews into a flat dictionary of layer names to layer tuples.

    Args:
        layer_views: LayerViews object containing the layer views.

    Returns:
        Dictionary mapping layer names to their corresponding (layer, datatype) tuples.
    """

    def _denest_recursive(items: dict) -> dict:
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
                nested_layers = _denest_recursive(value.group_members)
                layers.update(nested_layers)
            else:
                # Base case: add the layer to our dictionary
                if hasattr(value, "layer"):
                    layers[key] = value.layer

        return layers

    # Start the recursive denesting process and return the result
    return _denest_recursive(layer_views.layer_views)


def show_components(
    *args: ComponentSpec | ComponentAllAngleSpec,
) -> Sequence[Component]:
    """Show sequence of components in a single layot in a line.

    The components are spaced based on the maximum width and height of the components.

    Args:
        *args: Component specifications to show.

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
        spacing = (0, max_component_height + 200)
    else:
        spacing = (max_component_width + 200, 0)

    for i, component in enumerate(components):
        (c << component).move(
            (
                spacing[0] * i,
                spacing[1] * i,
            )
        )
    c.show()

    return components
