"""QPDK-specific inputs for the COMSOL models of :mod:`gplugins.comsol`.

The COMSOL geometry builders, studies, mesh, plotting, and result helpers, and
the chainable :class:`gplugins.comsol.COMSOL` model live in
:mod:`gplugins.comsol`; use them from there. This module holds only what depends
on the QPDK technology:

- :func:`prepare_comsol_layout` validates the M1_ETCH mask, inverts it into
  physical M1_DRAW metal with a ground margin, rejects layers the extraction
  would silently drop, and defaults to the ``coupling_o1``/``coupling_o2``
  feeds, before handing the metal to
  :func:`gplugins.comsol.layout.prepare_comsol_layout`.
- :data:`SILICON_RELATIVE_PERMITTIVITY` and :data:`AIR_RELATIVE_PERMITTIVITY`
  are the QPDK technology permittivities. The gplugins builders default to
  their own values, so pass these explicitly.

Note:
    Needs the optional ``comsol`` extra (``uv sync --extra comsol``), which
    installs ``gplugins[comsol]``. Building and solving models also needs a
    local COMSOL installation.

Example:
    >>> from gplugins.comsol import COMSOL
    >>> from qpdk.simulation.comsol import (
    ...     AIR_RELATIVE_PERMITTIVITY,
    ...     SILICON_RELATIVE_PERMITTIVITY,
    ...     prepare_comsol_layout,
    ... )
    >>> layout = prepare_comsol_layout(component)
    >>> model = COMSOL.create_sheet(
    ...     client,
    ...     layout,
    ...     "resonator",
    ...     silicon_relative_permittivity=SILICON_RELATIVE_PERMITTIVITY,
    ...     air_relative_permittivity=AIR_RELATIVE_PERMITTIVITY,
    ... )
"""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING

from gplugins.comsol import layout as gplugins_layout

from qpdk.simulation.layout import prepare_metal_layout
from qpdk.tech import LAYER, NON_METADATA_LAYERS, get_layer_material_properties

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gplugins.comsol.layout import ComsolFeedPort, ComsolLayout, CropPlanes

__all__ = [
    "AIR_RELATIVE_PERMITTIVITY",
    "SILICON_RELATIVE_PERMITTIVITY",
    "prepare_comsol_layout",
]

#: Relative permittivity of the silicon substrate, from the material of the
#: QPDK layer stack's substrate level.
SILICON_RELATIVE_PERMITTIVITY = float(
    get_layer_material_properties("Substrate")["relative_permittivity"]
)
#: Relative permittivity of the air above the chip, from the material of the
#: QPDK layer stack's vacuum level.
AIR_RELATIVE_PERMITTIVITY = float(
    get_layer_material_properties("Vacuum")["relative_permittivity"]
)

#: Input layers that the extraction understands. Everything else in
#: :data:`~qpdk.tech.NON_METADATA_LAYERS` is rejected rather than dropped.
_SUPPORTED_LAYERS: frozenset[tuple[int, ...]] = frozenset({
    tuple(LAYER.M1_DRAW),
    tuple(LAYER.M1_ETCH),
})


def prepare_comsol_layout(
    component: Component,
    feed_ports: tuple[str, str] | None = ("coupling_o1", "coupling_o2"),
    ground_margin: float = 100.0,
    *,
    crop_to_feed_ports: bool = False,
) -> ComsolLayout:
    """Extract M1_DRAW metal polygons and optional feed ports from a component.

    The M1_ETCH mask is inverted into positive M1_DRAW metal around the
    component bounding box grown by ``ground_margin``, with
    :func:`~qpdk.simulation.layout.prepare_metal_layout`, and the result is
    extracted with :func:`gplugins.comsol.layout.prepare_comsol_layout`.

    Args:
        component: The gdsfactory component to extract. It must contain an
            M1_ETCH mask; positive M1_DRAW shapes alone do not define the gaps.
            The original ports are preserved.
        feed_ports: Names of exactly two distinct ports to expose as feeds, or
            ``None`` for an unfed layout, in which case the result carries no
            feed ports. ``None`` fits layouts with no CPW feedline, e.g. an
            eigenmode qubit cell.
        ground_margin: Positive margin in µm added around the component
            bounding box to form the ground plane.
        crop_to_feed_ports: When ``True``, cut the prepared metal back to the
            two feed-port planes so each external face shows an open CPW cross
            section instead of the extended ground. Requires two feed ports
            that are axis-aligned, opposite, outward-facing, and share their
            transverse coordinate (left/right along x, or bottom/top along y).
            Cropping is refused unless every M1_DRAW and M1_ETCH polygon of the
            input component lies between the planes and the etch gaps reach
            both planes, so no resonator metal is cut and no face is shorted.
            The planes snap outwards to the database grid, so the returned
            ``bbox`` can exceed the port span by less than one database unit.
            Defaults to ``False``, which leaves the prepared ground untouched.

    Returns:
        A :class:`gplugins.comsol.layout.ComsolLayout` with metal polygons
        (holes preserved), the selected feed ports (empty when ``feed_ports``
        is ``None``), and the prepared bounding box, all in µm.

    Raises:
        ValueError: If ``ground_margin`` is not positive and finite, if
            ``feed_ports`` is supplied but does not name two distinct available
            ports, if a feed port is not cardinal or has non-finite coordinates
            or width, if the component has no M1_ETCH mask or carries geometry
            on unsupported fabrication layers, if no M1_DRAW metal remains, or
            if ``crop_to_feed_ports`` is set but the feeds are not a valid
            opposite pair or the component does not fit between the planes.
    """
    if not math.isfinite(ground_margin) or ground_margin <= 0.0:
        raise ValueError(
            f"ground_margin must be positive and finite, got {ground_margin!r}"
        )

    ports: tuple[ComsolFeedPort, ...] = ()
    if feed_ports is not None:
        if len(feed_ports) != 2:
            raise ValueError(
                "feed_ports must name exactly two ports or be None, "
                f"got {len(feed_ports)}"
            )
        if feed_ports[0] == feed_ports[1]:
            raise ValueError(
                f"feed_ports must be two distinct names, got {feed_ports!r}"
            )
        # Validate feeds before preparation: preparing registers a new cell and
        # we do not want invalid input to leave that side effect behind.
        ports = tuple(gplugins_layout.feed_port(component, name) for name in feed_ports)

    if crop_to_feed_ports and not ports:
        raise ValueError(
            "crop_to_feed_ports=True needs two feed ports to define the "
            "crop planes, but feed_ports is None"
        )
    planes = gplugins_layout.crop_planes(ports) if crop_to_feed_ports else None

    _reject_unsupported_layers(component)
    if not any(component.get_polygons(by="tuple", layers=[LAYER.M1_ETCH]).values()):
        raise ValueError(
            "COMSOL layout extraction requires an M1_ETCH mask to define the "
            "gaps; M1_DRAW shapes alone cannot be inverted into a ground plane"
        )
    if planes is not None:
        # Checked on the drawn mask: after inversion the ground spans the margin.
        _validate_crop_containment(component, planes)

    prepared = prepare_metal_layout(
        component,
        margin_draw=ground_margin,
        name=f"{component.name}_comsol_{uuid.uuid4().hex}",
    )
    return gplugins_layout.prepare_comsol_layout(
        prepared,
        LAYER.M1_DRAW,
        feed_ports,
        crop_to_feed_ports=crop_to_feed_ports,
    )


def _validate_crop_containment(component: Component, planes: CropPlanes) -> None:
    """Check the input mask fits between the crop planes, naming QPDK layers.

    :func:`gplugins.comsol.layout.validate_crop_containment` reports a layer
    as its ``(layer, datatype)`` tuple. Each layer is checked on its own so the
    error can name it as a :data:`~qpdk.tech.LAYER` member instead.

    Raises:
        ValueError: If M1_DRAW or M1_ETCH geometry extends beyond a plane, or
            the etch gaps do not reach both planes.
    """
    for layer in (LAYER.M1_DRAW, LAYER.M1_ETCH):
        try:
            gplugins_layout.validate_crop_containment(
                component, planes, layers=(layer,)
            )
        except ValueError as error:
            message = str(error).replace(f"{tuple(layer)} extends", f"{layer} extends")
            raise ValueError(message) from error
    gplugins_layout.validate_crop_containment(
        component, planes, layers=(), gap_layer=LAYER.M1_ETCH
    )


def _reject_unsupported_layers(component: Component) -> None:
    """Refuse components carrying fabrication layers this extractor ignores.

    Only M1_DRAW (and the M1_ETCH it is inverted from) is modelled. Layers
    such as M2_DRAW, airbridges, or the JJ_AREA/JJ_PATCH junction layers would
    be silently lost, making the extracted geometry wrong rather than failed,
    so they are rejected up front. A layout that wants an EM-only copy (e.g. a
    qubit cell with the junction removed) must strip those layers itself before
    extraction. Checked on the input because the AEDT preparation discards
    additive metal layers.

    Raises:
        ValueError: If any unsupported fabrication layer carries geometry.
    """
    unsupported_specs = {
        tuple(layer): str(layer)
        for layer in NON_METADATA_LAYERS
        if tuple(layer) not in _SUPPORTED_LAYERS
    }
    present = sorted(
        unsupported_specs[spec]
        for spec, shapes in component.get_polygons(by="tuple").items()
        if shapes and spec in unsupported_specs
    )
    if present:
        raise ValueError(
            "COMSOL layout extraction supports only the M1_DRAW layer, but the "
            f"component also has geometry on {present}. Remove those layers or "
            "extend the extraction before building a COMSOL model."
        )
