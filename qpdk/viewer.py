"""QPDK display stack for GDSFactory+, independent of simulation geometry."""

from gdsfactory.technology import LayerStack, LogicalLayer

from qpdk import PDK as _PDK
from qpdk.tech import LAYER, LAYER_STACK


def get_layer_stack(layer_stack: LayerStack = LAYER_STACK) -> LayerStack:
    """Use layout bounds for the substrate and etched metal in 3D previews.

    Vacuum is omitted so it does not obscure the device. The original stack
    and its simulation-area boundaries are unchanged.

    Returns:
        A copy of the process stack with background extrusion for display.
    """
    stack = layer_stack.model_copy(deep=True)
    stack.layers.pop("Vacuum", None)
    for name, level in stack.layers.items():
        if name.startswith("Substrate"):
            level.background = True
    for metal, draw, etch in (
        ("M1", LAYER.M1_DRAW, LAYER.M1_ETCH),
        ("M2", LAYER.M2_DRAW, LAYER.M2_ETCH),
    ):
        if metal not in stack.layers:
            continue
        ground = stack.layers[metal]
        ground.layer = LogicalLayer(layer=draw)
        ground.derived_layer = None
        ground.background = True
        ground.background_exclude_layers = (etch, draw)
        # Disjoint regions avoid duplicate faces where additive metal fills an etch.
        stack.layers[f"{metal}_additive"] = ground.model_copy(
            update={
                "name": f"{metal}_additive",
                "background": False,
                "background_exclude_layers": (),
            }
        )
    return stack


# gdsfactory uses the name to decide whether activating another PDK is a no-op.
PDK = _PDK.model_copy(
    update={"name": f"{_PDK.name}_viewer", "layer_stack": get_layer_stack()}
)
