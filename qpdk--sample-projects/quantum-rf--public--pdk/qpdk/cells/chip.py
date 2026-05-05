"""Chip-related finishing touches."""

import gdsfactory as gf
from gdsfactory.typings import LayerSpec

from qpdk.cells.waveguides import rectangle
from qpdk.helper import show_components


@gf.cell
def chip_edge(
    size: tuple[float, float] = (10000.0, 10000.0),
    width: float = 200.0,
    layer: LayerSpec = "M1_ETCH",
) -> gf.Component:
    """Returns a chip edge component with hollow rectangle frame.

    Creates a rectangular frame (hollow rectangle) on an etched metal layer,
    typically used to define chip edge regions for proper fabrication.

    Args:
        size: (tuple) Width and height of the chip edge area.
        width: Width/thickness of the etched frame border.
        layer: Layer to put the etched frame on.
    """
    c = gf.Component()

    # Create the hollow rectangle frame using four rectangles
    rect_configs = [
        {"size": (size[0], width), "position": (0, size[1] - width)},  # Top edge
        {"size": (size[0], width), "position": (0, 0)},  # Bottom edge
        {"size": (width, size[1]), "position": (0, 0)},  # Left edge
        {"size": (width, size[1]), "position": (size[0] - width, 0)},  # Right edge
    ]

    common_params = {
        "layer": layer,
        "centered": False,
        "port_type": None,
    }

    for config in rect_configs:
        rect = c << rectangle(size=config["size"], **common_params)
        rect.move(config["position"])

    return c


if __name__ == "__main__":
    show_components(
        chip_edge,
        spacing=50,
    )
