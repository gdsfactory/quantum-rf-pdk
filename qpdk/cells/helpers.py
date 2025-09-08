"""Helper functions for QPDK cells."""

from collections.abc import Iterable

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from kfactory import kdb
from klayout.db import DCplxTrans

from qpdk.tech import LAYER


def transform_component(component: gf.Component, transform: DCplxTrans) -> gf.Component:
    """Applies a complex transformation to a component.

    For use with :func:`~gdsfactory.container`.
    """
    return component.transform(transform)


@gf.cell()
def fill_magnetic_vortices(
    component: Component,
    rectangle_size: tuple[float, float] = (15.0, 15.0),
    gap: float = 15.0,
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
        rectangle_size: Size of the fill rectangles in µm (width, height).
        gap: Gap between rectangles in µm.
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
    """
    exclude_layers = exclude_layers or [
        (LAYER.M1_ETCH, 80),
        (LAYER.M1_DRAW, 80),
        (LAYER.WG, 80),
    ]
    c = gf.Component()
    c.add_ref(component)

    # Create the fill rectangle cell
    fill_cell = gf.components.rectangle(
        size=rectangle_size,
        layer=fill_layer,
    )

    c.fill(
        fill_cell=fill_cell,
        fill_regions=[
            (
                kdb.Region(c.bbox().to_itype(dbu=c.kcl.dbu)),
                0,
            )
        ],  # Fill the entire bounding box area
        exclude_layers=exclude_layers,
        x_space=gap,
        y_space=gap,
    )

    return c
