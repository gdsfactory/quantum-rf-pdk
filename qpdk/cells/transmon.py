"""Transmon qubit components."""

from __future__ import annotations

import operator
from functools import partial, reduce
from typing import TypedDict, Unpack

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec
from klayout.db import DCplxTrans

from qpdk.cells.bump import indium_bump
from qpdk.cells.helpers import transform_component
from qpdk.cells.junction import squid_junction
from qpdk.tech import LAYER


class DoublePadTransmonParams(TypedDict):
    """Parameters for double pad transmon qubit.

    Keyword Args:
        pad_size: (width, height) of each capacitor pad in μm.
        pad_gap: Gap between the two capacitor pads in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.
    """

    pad_size: tuple[float, float]
    pad_gap: float
    junction_spec: ComponentSpec
    junction_displacement: DCplxTrans | None
    layer_metal: LayerSpec


_double_pad_transmon_default_params = DoublePadTransmonParams(
    pad_size=(250.0, 400.0),
    pad_gap=15.0,
    junction_spec=squid_junction,
    junction_displacement=None,
    layer_metal=LAYER.M1_DRAW,
)


@gf.cell(check_instances=False)
def double_pad_transmon(**kwargs: Unpack[DoublePadTransmonParams]) -> Component:
    """Creates a double capacitor pad transmon qubit with Josephson junction.

    A transmon qubit consists of two capacitor pads connected by a Josephson junction.
    The junction creates an anharmonic oscillator that can be used as a qubit.

    See :cite:`kochChargeinsensitiveQubitDesign2007a` for details.

    Args:
        **kwargs: :class:`~DoublePadTransmonParams` for the transmon qubit.

    Returns:
        Component: A gdsfactory component with the transmon geometry.
    """
    c = Component()
    params = _double_pad_transmon_default_params | kwargs
    # Extract wire parameters using dictionary unpacking
    pad_size, pad_gap, junction_spec, junction_displacement, layer_metal = (
        params[key]
        for key in [
            "pad_size",
            "pad_gap",
            "junction_spec",
            "junction_displacement",
            "layer_metal",
        ]
    )
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
    **kwargs: Unpack[DoublePadTransmonParams],
) -> Component:
    """Creates a double capacitor pad transmon qubit with Josephson junction and an etched bounding box.

    See :func:`~double_pad_transmon` for more details.

    Args:
        bbox_extension: Extension size for the bounding box in μm.
        **kwargs: :class:`~DoublePadTransmonParams` for the transmon qubit.

    Returns:
        Component: A gdsfactory component with the transmon geometry and etched box.
    """
    c = gf.Component()
    double_pad_ref = c << double_pad_transmon(**kwargs)
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
    inner_ring_radius: float = 50,
    inner_ring_width: float = 30,
    outer_ring_radius: float = 110,
    outer_ring_width: float = 60,
    top_circle_radius: float = 110,
    junction_spec: ComponentSpec = squid_junction,
    junction_displacement: DCplxTrans | None = None,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_metal_top: LayerSpec = LAYER.M2_DRAW,
) -> Component:
    """Creates a circular transmon qubit with `flipmon` geometry.

    A circular variant of the transmon qubit with another circle as the inner pad.

    See :cite:`liVacuumgapTransmonQubits2021,liCosmicrayinducedCorrelatedErrors2025`
    for details about the `flipmon` design.

    Args:
        inner_ring_radius: Central radius of the inner circular capacitor pad in μm.
        inner_ring_width: Width of the inner circular capacitor pad in μm.
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

    def create_circular_pad(radius: float, width: float) -> gf.ComponentReference:
        pad = gf.c.ring(
            radius=radius,
            width=width,
            layer=layer_metal,
        )
        return c.add_ref(pad)

    create_circular_pad(inner_ring_radius, inner_ring_width)
    create_circular_pad(outer_ring_radius, outer_ring_width)

    # Create Josephson junction
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    # Center the junction between the pads
    # junction_ref.rotate(45)
    junction_ref.dcenter = c.dcenter  # move((-junction_height / 2, 0))
    junction_ref.dcplx_trans *= reduce(
        operator.mul,
        (
            DCplxTrans(
                (
                    inner_ring_radius
                    + inner_ring_width / 2
                    + outer_ring_radius
                    - outer_ring_width / 2
                )
                / 2,
                0,
            ),
            DCplxTrans(1, 45, False, 0, 0),
            # DCplxTrans(1, 45, False, 0, 0),
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
        center=(inner_ring_radius + inner_ring_width / 2, 0),
        width=inner_ring_width,
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


class XmonTransmonParams(TypedDict):
    """Parameters for Xmon style transmon qubit.

    Keyword Args:
        center_width: Width of the central cross intersection in μm.
        center_height: Height of the central cross intersection in μm.
        arm_width: Width of each arm extending from center in μm.
        arm_lengths: Tuple of (top, right, bottom, left) arm lengths in μm.
        gap_width: Width of the etched gap around arms in μm.
        junction_spec: Component specification for the Josephson junction component.
        junction_displacement: Optional complex transformation to apply to the junction.
        layer_metal: Layer for the metal pads.
        layer_etch: Layer for the etched regions.
    """

    center_width: float
    center_height: float
    arm_width: float
    arm_lengths: tuple[float, float, float, float]  # top, right, bottom, left
    gap_width: float
    junction_spec: ComponentSpec
    junction_displacement: DCplxTrans | None
    layer_metal: LayerSpec
    layer_etch: LayerSpec


_xmon_transmon_default_params = XmonTransmonParams(
    center_width=50.0,
    center_height=50.0,
    arm_width=20.0,
    arm_lengths=(80.0, 80.0, 80.0, 80.0),  # top, right, bottom, left
    gap_width=10.0,
    junction_spec=squid_junction,
    junction_displacement=None,
    layer_metal=LAYER.M1_DRAW,
    layer_etch=LAYER.M1_ETCH,
)


@gf.cell(check_instances=False)
def xmon_transmon(**kwargs: Unpack[XmonTransmonParams]) -> Component:
    """Creates an Xmon style transmon qubit with cross-shaped geometry.

    An Xmon transmon consists of a cross-shaped capacitor pad with four arms
    extending from a central region, connected by a Josephson junction at the center.
    The design provides better control over the coupling to readout resonators
    and neighboring qubits through the individual arm geometries.

    See :cite:`barends2013coherent,kelly2015state` for details about the Xmon design.

    Args:
        **kwargs: :class:`~XmonTransmonParams` for the Xmon transmon qubit.

    Returns:
        Component: A gdsfactory component with the Xmon transmon geometry.
    """
    c = Component()
    params = _xmon_transmon_default_params | kwargs

    # Extract parameters
    (
        center_width,
        center_height,
        arm_width,
        arm_lengths,
        gap_width,
        junction_spec,
        junction_displacement,
        layer_metal,
        layer_etch,
    ) = (
        params[key]
        for key in [
            "center_width",
            "center_height",
            "arm_width",
            "arm_lengths",
            "gap_width",
            "junction_spec",
            "junction_displacement",
            "layer_metal",
            "layer_etch",
        ]
    )

    arm_length_top, arm_length_right, arm_length_bottom, arm_length_left = arm_lengths

    # Create the central cross intersection
    center_pad = gf.components.rectangle(
        size=(center_width, center_height),
        layer=layer_metal,
    )
    center_ref = c.add_ref(center_pad)
    center_ref.dcenter = (0, 0)

    # Create the four arms extending from the center
    # Top arm
    if arm_length_top > 0:
        top_arm = gf.components.rectangle(
            size=(arm_width, arm_length_top),
            layer=layer_metal,
        )
        top_arm_ref = c.add_ref(top_arm)
        top_arm_ref.move((-arm_width / 2, center_height / 2))

    # Right arm
    if arm_length_right > 0:
        right_arm = gf.components.rectangle(
            size=(arm_length_right, arm_width),
            layer=layer_metal,
        )
        right_arm_ref = c.add_ref(right_arm)
        right_arm_ref.move((center_width / 2, -arm_width / 2))

    # Bottom arm
    if arm_length_bottom > 0:
        bottom_arm = gf.components.rectangle(
            size=(arm_width, arm_length_bottom),
            layer=layer_metal,
        )
        bottom_arm_ref = c.add_ref(bottom_arm)
        bottom_arm_ref.move((-arm_width / 2, -(center_height / 2 + arm_length_bottom)))

    # Left arm
    if arm_length_left > 0:
        left_arm = gf.components.rectangle(
            size=(arm_length_left, arm_width),
            layer=layer_metal,
        )
        left_arm_ref = c.add_ref(left_arm)
        left_arm_ref.move((-(center_width / 2 + arm_length_left), -arm_width / 2))

    # Calculate total bounding box dimensions for the etch layer
    # Include gaps around all arms
    max_arm_right = max(arm_length_right, 0)
    max_arm_left = max(arm_length_left, 0)
    max_arm_top = max(arm_length_top, 0)
    max_arm_bottom = max(arm_length_bottom, 0)

    total_width = center_width + max_arm_left + max_arm_right + 2 * gap_width
    total_height = center_height + max_arm_top + max_arm_bottom + 2 * gap_width

    # Create the background etch layer
    etch_background = gf.components.rectangle(
        size=(total_width, total_height),
        layer=layer_etch,
    )
    etch_ref = c.add_ref(etch_background)
    etch_ref.dcenter = (0, 0)

    # Create and place Josephson junction at the center
    junction_ref = c.add_ref(gf.get_component(junction_spec))
    junction_ref.rotate(45)
    junction_ref.dcenter = (0, 0)
    if junction_displacement:
        junction_ref.transform(junction_displacement)

    # Add ports at the ends of each arm for connectivity
    if arm_length_top > 0:
        c.add_port(
            name="top_arm",
            center=(0, center_height / 2 + arm_length_top),
            width=arm_width,
            orientation=90,
            layer=layer_metal,
        )

    if arm_length_right > 0:
        c.add_port(
            name="right_arm",
            center=(center_width / 2 + arm_length_right, 0),
            width=arm_width,
            orientation=0,
            layer=layer_metal,
        )

    if arm_length_bottom > 0:
        c.add_port(
            name="bottom_arm",
            center=(0, -(center_height / 2 + arm_length_bottom)),
            width=arm_width,
            orientation=270,
            layer=layer_metal,
        )

    if arm_length_left > 0:
        c.add_port(
            name="left_arm",
            center=(-(center_width / 2 + arm_length_left), 0),
            width=arm_width,
            orientation=180,
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
    c.info["qubit_type"] = "xmon_transmon"

    return c


@gf.cell
def xmon_transmon_with_etch(**kwargs: Unpack[XmonTransmonParams]) -> Component:
    """Creates an Xmon style transmon qubit with cross-shaped geometry and etched isolation.

    This version includes properly etched gaps around the Xmon structure for isolation.
    The etch layer is created by subtracting the metal regions from a background rectangle.

    Args:
        **kwargs: :class:`~XmonTransmonParams` for the Xmon transmon qubit.

    Returns:
        Component: A gdsfactory component with the Xmon transmon geometry and etch layer.
    """
    c = gf.Component()

    # Create the basic Xmon structure
    xmon_ref = c << xmon_transmon(**kwargs)

    # Get parameters for etch calculation
    params = _xmon_transmon_default_params | kwargs
    gap_width = params["gap_width"]
    layer_etch = params["layer_etch"]
    layer_metal = params["layer_metal"]

    # Calculate the etch bounding box
    xmon_size = (xmon_ref.size_info.width, xmon_ref.size_info.height)
    etch_size = (
        xmon_size[0] + 2 * gap_width,
        xmon_size[1] + 2 * gap_width,
    )

    # Create etch background
    etch_background = gf.components.rectangle(
        size=etch_size,
        layer=layer_etch,
    )

    # Use boolean operation to subtract metal from etch
    etch_final = gf.boolean(
        A=etch_background,
        B=c,
        operation="A-B",
        layer=layer_etch,
        layer1=layer_etch,
        layer2=layer_metal,
    )

    etch_ref = c.add_ref(etch_final)
    etch_ref.dcenter = xmon_ref.dcenter

    # Add ports from the xmon
    c.add_ports(xmon_ref.ports)

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = gf.Component()
    for i, component in enumerate(
        (
            double_pad_transmon(),
            double_pad_transmon(junction_displacement=DCplxTrans(0, 150)),
            double_pad_transmon_with_bbox(),
            flipmon(),
            xmon_transmon(),
            xmon_transmon_with_etch(),
        )
    ):
        (c << component).move((0, i * 700))
    c.show()
