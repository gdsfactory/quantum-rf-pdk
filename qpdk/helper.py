"""Helper functions for the qpdk package."""

from gdsfactory.technology import LayerViews


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
