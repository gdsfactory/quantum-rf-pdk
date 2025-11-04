"""Transmon qubit components."""

from __future__ import annotations

import operator
from functools import partial, reduce

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import rectangle
from gdsfactory.export import to_stl
from gdsfactory.typings import ComponentSpec, LayerSpec
from kfactory import kdb
from klayout.db import DCplxTrans, Region

from qpdk.cells.bump import indium_bump
from qpdk.cells.helpers import transform_component
from qpdk.cells.junction import josephson_junction, squid_junction
from qpdk.helper import show_components
from qpdk.tech import LAYER, LAYER_STACK_FLIP_CHIP


@gf.cell(check_instances=False)
def double_pad_transmon(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 15.0,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Creates a double capacitor pad transmon qubit with Josephson junction.

    A transmon qubit consists of two capacitor pads connected by a Josephson junction.
    The junction creates an anharmonic oscillator that can be used as a qubit.

    .. svgbob::

        +---------+         +---------+
        |         |         |         |
        |  pad1   |   XX    |  pad2   |
        |         |   XX    |         |
        +---------+ junct.  +---------+
         left_pad             right_pad

    See :cite:`kochChargeinsensitiveQubitDesign2007a` for details.

    Args:
        pad_size: (width, height) of each capacitor pad in μm.
        pad_gap: Gap between the two capacitor pads in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.

    Returns:
        Component: A gdsfactory component with the transmon geometry.
    """
    c = Component()

    pad_width, pad_height = pad_size

    def create_capacitor_pad(x_offset: float) -> gf.ComponentReference:
        pad = gf.components.rectangle(
            size=pad_size,
            layer=layer_metal,
        )
        pad_ref = c.add_ref(pad)
        pad_ref.move((x_offset, -pad_height / 2))
        return pad_ref

    create_capacitor_pad(-pad_width - pad_gap / 2)
    create_capacitor_pad(pad_gap / 2)

    # Create Josephson junction
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    junction_ref.rotate(45)
    # Center the junction between the pads
    junction_ref.dcenter = c.dcenter  # move((-junction_height / 2, 0))
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Add ports for connections
    c.add_port(
        name="left_pad",
        center=(-pad_width - pad_gap / 2, 0),
        width=pad_height,
        orientation=180,
        layer=layer_metal,
    )
    c.add_port(
        name="right_pad",
        center=(pad_width + pad_gap / 2, 0),
        width=pad_height,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
    )

    # Add metadata
    c.info["qubit_type"] = "transmon"

    return c


@gf.cell
def double_pad_transmon_with_bbox(
    bbox_extension: float = 200.0,
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 15.0,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
) -> Component:
    """Creates a double capacitor pad transmon qubit with Josephson junction and an etched bounding box.

    See :func:`~double_pad_transmon` for more details.

    Args:
        bbox_extension: Extension size for the bounding box in μm.
        pad_size: (width, height) of each capacitor pad in μm.
        pad_gap: Gap between the two capacitor pads in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.

    Returns:
        Component: A gdsfactory component with the transmon geometry and etched box.
    """
    c = gf.Component()
    double_pad_ref = c << double_pad_transmon(
        pad_size=pad_size,
        pad_gap=pad_gap,
        junction_spec=junction_spec,
        junction_displacement=junction_displacement,
        layer_metal=layer_metal,
    )
    double_pad_size = (double_pad_ref.size_info.width, double_pad_ref.size_info.height)
    bbox_size = (
        double_pad_size[0] + 2 * bbox_extension,
        double_pad_size[1] + 2 * bbox_extension,
    )

    bbox = gf.container(
        partial(
            gf.components.rectangle,
            size=bbox_size,
            layer=LAYER.M1_ETCH,
        ),
        # Center the bbox around the double pad
        partial(
            transform_component, transform=DCplxTrans(*(-e / 2 for e in bbox_size))
        ),
    )
    # Remove additive metal from etch
    bbox = gf.boolean(
        A=bbox,
        B=c,
        operation="-",
        layer=LAYER.M1_ETCH,
        layer1=LAYER.M1_ETCH,
        layer2=LAYER.M1_DRAW,
    )
    bbox_ref = c.add_ref(bbox)
    c.absorb(bbox_ref)

    c.add_ports(double_pad_ref.ports)
    return c


@gf.cell(check_instances=False)
def flipmon(
    inner_circle_radius: float = 60.0,
    outer_ring_radius: float = 110.0,
    outer_ring_width: float = 60.0,
    top_circle_radius: float = 110.0,
    junction_spec: ComponentSpec = partial(
        squid_junction,
        junction_spec=partial(josephson_junction, wide_straight_length=12),
    ),
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_metal_top: LayerSpec = LAYER.M2_DRAW,
) -> Component:
    """Creates a circular transmon qubit with `flipmon` geometry.

    A circular variant of the transmon qubit with another circle as the inner pad.

    See :cite:`liVacuumgapTransmonQubits2021,liCosmicrayinducedCorrelatedErrors2025`
    for details about the `flipmon` design.

    Args:
        inner_circle_radius: Central radius of the inner circular capacitor pad in μm.
        outer_ring_radius: Central radius of the outer circular capacitor pad in μm.
        outer_ring_width: Width of the outer circular capacitor pad in μm.
        top_circle_radius: Central radius of the top circular capacitor pad in μm.
            There is no separate width as the filled circle is not a ring.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.
        layer_metal_top: Layer for the other metal layer pad for flip-chip.

    Returns:
        Component: A gdsfactory component with the circular transmon geometry.
    """
    c = Component()

    _ = c << gf.c.circle(
        radius=inner_circle_radius,
        layer=layer_metal,
    )

    _ = c << gf.c.ring(
        radius=outer_ring_radius,
        width=outer_ring_width,
        layer=layer_metal,
    )

    # Create Josephson junction
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    # Center the junction between the pads
    junction_ref.dcenter = c.dcenter  # move((-junction_height / 2, 0))
    junction_ref.dcplx_trans *= reduce(
        operator.mul,
        (
            DCplxTrans(1, 45, False, 0, 0),
            DCplxTrans(
                (inner_circle_radius + outer_ring_radius - outer_ring_width / 2) / 2,
                0,
            ),
            DCplxTrans(0, -junction_ref.size_info.height),
        ),
    )
    junction_ref.y = 0

    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Create top circular pad for flip-chip
    top_circle = gf.components.circle(
        radius=top_circle_radius,
        layer=layer_metal_top,
    )
    top_circle_ref = c.add_ref(top_circle)
    top_circle_ref.dcenter = c.dcenter

    # Add indium bump to flip-chip
    bump = gf.get_component(indium_bump)
    bump_ref = c.add_ref(bump)
    bump_ref.dcenter = c.dcenter
    c.add_ports(bump_ref.ports)

    # Add ports for connections
    c.add_port(
        name="inner_ring_near_junction",
        center=(inner_circle_radius / 2, 0),
        width=outer_ring_width,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="outer_ring_near_junction",
        center=(outer_ring_radius - outer_ring_width / 2, 0),
        width=outer_ring_width,
        orientation=180,
        layer=layer_metal,
    )
    c.add_port(
        name="outer_ring_outside",
        center=(outer_ring_radius + outer_ring_width / 2, 0),
        width=outer_ring_width,
        orientation=0,
        layer=layer_metal,
    )
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
    )

    return c


@gf.cell
def flipmon_with_bbox(
    inner_circle_radius: float = 60.0,
    outer_ring_radius: float = 110.0,
    outer_ring_width: float = 60.0,
    top_circle_radius: float = 110.0,
    junction_spec: ComponentSpec = partial(
        squid_junction,
        junction_spec=partial(josephson_junction, wide_straight_length=12),
    ),
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_metal_top: LayerSpec = LAYER.M2_DRAW,
    m1_etch_extension_gap: float = 30.0,
    m2_etch_extension_gap: float = 40.0,
) -> Component:
    """Creates a circular transmon qubit with `flipmon` geometry and a circular etched bounding box.

    See :func:`~flipmon` for more details.

    Args:
        inner_circle_radius: Central radius of the inner circular capacitor pad in μm.
        outer_ring_radius: Central radius of the outer circular capacitor pad in μm.
        outer_ring_width: Width of the outer circular capacitor pad in μm.
        top_circle_radius: Central radius of the top circular capacitor pad in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.
        layer_metal_top: Layer for the other metal layer pad for flip-chip.
        m1_etch_extension_gap: Radius extension length for the M1 etch bounding box in μm.
        m2_etch_extension_gap: Radius extension length for the M2 etch bounding box in μm.

    Returns:
        Component: A gdsfactory component with the flipmon geometry and etched box.
    """
    c = gf.Component()
    flipmon_ref = c << flipmon(
        inner_circle_radius=inner_circle_radius,
        outer_ring_radius=outer_ring_radius,
        outer_ring_width=outer_ring_width,
        top_circle_radius=top_circle_radius,
        junction_spec=junction_spec,
        junction_displacement=junction_displacement,
        layer_metal=layer_metal,
        layer_metal_top=layer_metal_top,
    )
    m1_bbox_radius = outer_ring_radius + outer_ring_width / 2 + m1_etch_extension_gap
    m2_bbox_radius = top_circle_radius + m2_etch_extension_gap

    for etch_layer, draw_layer, bbox_radius in [
        (LAYER.M1_ETCH, LAYER.M1_DRAW, m1_bbox_radius),
        (LAYER.M2_ETCH, LAYER.M2_DRAW, m2_bbox_radius),
    ]:
        bbox_comp = gf.boolean(
            A=gf.components.circle(
                radius=bbox_radius,
                layer=etch_layer,
            ),
            B=c,
            operation="-",
            layer=etch_layer,
            layer1=etch_layer,
            layer2=draw_layer,
        )
        c.absorb(c.add_ref(bbox_comp))

    c.add_ports(flipmon_ref.ports)
    return c


@gf.cell(check_instances=False)
def xmon_transmon(
    arm_width: tuple[float, float, float, float] = (30.0, 20.0, 30.0, 20.0),
    arm_lengths: tuple[float, float, float, float] = (160.0, 120.0, 160.0, 120.0),
    gap_width: float = 10.0,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    """Creates an Xmon style transmon qubit with cross-shaped geometry.

    An Xmon transmon consists of a cross-shaped capacitor pad with four arms
    extending from a central region, connected by a Josephson junction at the center.
    The design provides better control over the coupling to readout resonators
    and neighboring qubits through the individual arm geometries.

    See :cite:`barendsCoherentJosephsonQubit2013a` for details about the Xmon design.

    Args:
        arm_width: Tuple of (top, right, bottom, left) arm widths in μm.
        arm_lengths: Tuple of (top, right, bottom, left) arm lengths in μm.
            Computed from center to end of each arm.
        gap_width: Width of the etched gap around arms in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.
        layer_etch: Layer for the etched regions.

    Returns:
        Component: A gdsfactory component with the Xmon transmon geometry.
    """
    c = Component()

    arm_width_top, arm_width_right, arm_width_bottom, arm_width_left = arm_width
    arm_length_top, arm_length_right, arm_length_bottom, arm_length_left = arm_lengths

    # Define arm configurations: (size, move_offset)
    arm_configs = [
        ((arm_width_top, arm_length_top), (-arm_width_top / 2, arm_length_top * 0)),
        (
            (arm_length_right, arm_width_right),
            (arm_length_right * 0, -arm_width_right / 2),
        ),
        (
            (arm_width_bottom, arm_length_bottom),
            (-arm_width_bottom / 2, -arm_length_bottom),
        ),
        ((arm_length_left, arm_width_left), (-arm_length_left, -arm_width_left / 2)),
    ]

    # Create the four arms extending from the center
    for size, move_offset in arm_configs:
        arm = gf.components.rectangle(
            size=size,
            layer=layer_metal,
        )
        arm_ref = c.add_ref(arm)
        arm_ref.move(move_offset)
        c.absorb(arm_ref)
        c.flatten(merge=True)

    # Create etch by sizing drawn metal
    etch_region = gf.component.size(
        Region(
            kdb.RecursiveShapeIterator(
                c.kcl.layout,
                c._base.kdb_cell,  # pyright: ignore[reportPrivateUsage]
                layer_metal,
            )
        ),
        gap_width,
    )
    etch_component = gf.Component()
    etch_component.add_polygon(etch_region, layer=layer_etch)

    # Remove additive metal from etch
    etch_component = gf.boolean(
        A=etch_component,
        B=c,
        operation="-",
        layer=LAYER.M1_ETCH,
        layer1=LAYER.M1_ETCH,
        layer2=LAYER.M1_DRAW,
    )
    etch_ref = c.add_ref(etch_component)
    c.absorb(etch_ref)

    # Create and place Josephson junction at the y-center of the gap
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    junction_ref.rotate(-45)
    junction_ref.dcenter = (0, c.ymin + gap_width / 2)
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Add ports at the ends of each arm for connectivity
    for name, width, center, orientation in zip(
        ["top_arm", "right_arm", "bottom_arm", "left_arm"],
        arm_width,
        [
            (0, arm_length_top),
            (arm_length_right, 0),
            (0, -arm_length_bottom),
            (-arm_length_left, 0),
        ],
        [90, 0, 270, 180],
    ):
        c.add_port(
            name=name,
            center=center,
            width=width,
            orientation=orientation,
            layer=layer_metal,
        )

    # Add junction port
    c.add_port(
        name="junction",
        center=junction_ref.dcenter,
        width=junction_ref.size_info.height,
        orientation=90,
        layer=LAYER.JJ_AREA,
    )

    # Add metadata
    c.info["qubit_type"] = "xmon"

    return c


if __name__ == "__main__":
    show_components(
        double_pad_transmon,
        partial(double_pad_transmon, junction_displacement=DCplxTrans(0, 150)),
        double_pad_transmon_with_bbox,
        flipmon,
        flipmon_with_bbox,
        xmon_transmon,
    )

    # Visualize flip-chip flipmon
    c = gf.Component()
    (c << flipmon_with_bbox()).move((0, 0))
    c << rectangle(size=(500, 500), layer=LAYER.SIM_AREA, centered=True)
    to_stl(c, "flipmon.stl", layer_stack=LAYER_STACK_FLIP_CHIP)
    scene = c.to_3d(layer_stack=LAYER_STACK_FLIP_CHIP)
    scene.show()
