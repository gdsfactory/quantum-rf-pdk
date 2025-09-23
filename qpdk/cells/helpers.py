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
    component: Component,
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
    """
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
