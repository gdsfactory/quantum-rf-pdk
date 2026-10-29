"""Technology definitions."""

# ruff: file-ignore[print]

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from functools import cache, partial, wraps
from typing import Any, Literal

import gdsfactory as gf
from gdsfactory.cross_section import (
    CrossSection,
)
from gdsfactory.technology import (
    DerivedLayer,
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import (
    ConnectivitySpec,
    CrossSectionSpec,
    Layer,
    LayerSpec,
    RoutingStrategies,
)
from kfactory.routing.generic import ManhattanRoute

from qpdk.config import PATH
from qpdk.helper import denest_layerviews_to_layer_tuples

nm = 1e-3


class LayerMapQPDK(LayerMap):
    """Layer map for QPDK technology."""

    # Base metals
    M1_DRAW: Layer = (1, 0)  # Additive metal / positive mask regions
    M1_ETCH: Layer = (1, 1)  # Subtractive etch / negative mask regions
    # Additive wins over subtractive where they overlap
    # i.e., you can draw metal over an etched region to "fill it back in"

    # flip-chip equivalents
    M2_DRAW: Layer = (2, 0)
    M2_ETCH: Layer = (2, 1)

    # Backside (side B) metal, connected to M1 through TSVs
    MB_DRAW: Layer = (3, 0)
    MB_ETCH: Layer = (3, 1)

    # Backside metal of the flip-chip top chip, connected to M2 through TSV2
    MB2_DRAW: Layer = (4, 0)
    MB2_ETCH: Layer = (4, 1)

    # Airbridges
    AB_DRAW: Layer = (10, 0)  # Bridge metal
    AB_VIA: Layer = (10, 1)  # Landing pads / contacts

    # Junctions
    JJ_AREA: Layer = (20, 0)  # Optional bridge/overlap definition
    JJ_PATCH: Layer = (20, 1)

    # Hybrid junction contacts, insulated gate and transferred materials
    CNT_CONTACT: Layer = (21, 0)  # Nb/Au contact evaporation
    TOP_GATE: Layer = (22, 0)  # Local Al gate, separate from its M1 feed
    HBN: Layer = (23, 0)  # Transferred dielectric footprint
    CNT: Layer = (24, 0)  # Nanotube placement trace, not an etched metal wire

    # Nanowire
    NbTiN: Layer = (25, 0)

    # Packaging / 3D integration / backside / misc.
    IND: Layer = (30, 0)
    TSV: Layer = (31, 0)  # Throughs / vias / backside features
    TSV2: Layer = (32, 0)  # Through-silicon vias in the flip-chip top chip
    DICE: Layer = (70, 0)  # Dicing lanes

    # Alignment / admin
    ALN_TOP: Layer = (80, 0)  # Frontside alignment
    ALN_BOT: Layer = (81, 0)  # Backside alignment

    ###################
    # Non-fabrication #
    ###################

    TEXT: Layer = (90, 0)  # Mask text / version labels

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)

    # Simulation-only helpers (never sent to fab)
    SIM_AREA: Layer = (98, 0)
    SIM_ONLY: Layer = (99, 0)

    # Marker layer for waveguides
    WG: Layer = (102, 0)

    # Error marker layer (used by gdsfactory to highlight routing/DRC errors)
    ERROR_PATH: Layer = (1000, 0)


L = LAYER = LayerMapQPDK

material_properties: dict[str, dict[str, float]] = {
    "vacuum": {"relative_permittivity": 1},
    "Nb": {"relative_permittivity": float("inf")},
    "NbTiN": {"relative_permittivity": float("inf")},
    # Loss tangent from `checchinMeasurementLowTemperatureLoss2022`
    "Si": {"relative_permittivity": 11.45, "loss_tangent": 2.7e-6},
    "AlOx/Al": {"relative_permittivity": float("inf")},
    "TiN": {"relative_permittivity": float("inf")},
    "In": {"relative_permittivity": float("inf")},
    "Au": {"relative_permittivity": float("inf")},
    "Al": {"relative_permittivity": float("inf")},
    "SiO2": {"relative_permittivity": 3.9},
    # Out-of-plane static value, https://arxiv.org/abs/2201.05826.
    "hBN": {"relative_permittivity": 3.4},
}

NON_METADATA_LAYERS = {
    LAYER.M1_DRAW,
    LAYER.M1_ETCH,
    LAYER.M2_DRAW,
    LAYER.M2_ETCH,
    LAYER.MB_DRAW,
    LAYER.MB_ETCH,
    LAYER.MB2_DRAW,
    LAYER.MB2_ETCH,
    LAYER.AB_DRAW,
    LAYER.AB_VIA,
    LAYER.JJ_AREA,
    LAYER.JJ_PATCH,
    LAYER.CNT_CONTACT,
    LAYER.TOP_GATE,
    LAYER.HBN,
    LAYER.CNT,
    LAYER.IND,
    LAYER.TSV,
    LAYER.TSV2,
    LAYER.DICE,
    LAYER.ALN_TOP,
    LAYER.ALN_BOT,
    LAYER.NbTiN,
}


@cache
def get_layer_stack() -> LayerStack:
    """Return the physical layer stack for simulations.

    The stack roughly corresponds to that of :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    return LayerStack(
        layers={
            # Base metal film (e.g., 200 nm of Nb)
            "M1": LayerLevel(
                name="M1",
                layer=L.M1_DRAW,
                thickness=200e-9 * 1e6,
                zmin=0.0,
                material="Nb",
                mesh_order=1,
            ),
            "M1_ground": LayerLevel(
                name="M1_ground",
                # An implicit ground would fill gaps in positive solver geometry.
                layer=DerivedLayer(
                    layer1=LogicalLayer(layer=L.SIM_AREA),
                    layer2=DerivedLayer(
                        layer1=LogicalLayer(layer=L.M1_ETCH),
                        layer2=LogicalLayer(layer=L.M1_DRAW),
                        operation="or",
                    ),
                    operation="not",
                ),
                # Separate outputs keep the ground and additive faces disjoint.
                derived_layer=LogicalLayer(layer=L.M1_ETCH),
                thickness=200e-9 * 1e6,
                zmin=0.0,  # top of substrate
                material="Nb",
                mesh_order=1,
            ),
            "NbTiN": LayerLevel(
                name="NbTiN",
                layer=L.NbTiN,
                thickness=15e-9 * 1e6,
                zmin=0.0,  # top of substrate
                material="NbTiN",
                mesh_order=1,
            ),
            "Substrate": LayerLevel(
                name="Substrate",
                layer=L.SIM_AREA,
                thickness=500,  # 500 microns of silicon
                zmin=-500,  # below metal
                material="Si",
                mesh_order=4,
                background=True,
            ),
            "Vacuum": LayerLevel(
                name="Vacuum",
                layer=L.SIM_AREA,
                thickness=500e-6 * 1e6,  # 500 microns of vacuum above metal
                zmin=200e-9 * 1e6,  # above metal
                material="vacuum",
                mesh_order=99,
            ),
            # Airbridge metal sitting above M1 (example: +300 nm)
            "Airbridge": LayerLevel(
                name="Airbridge",
                layer=L.AB_DRAW,
                thickness=200e-9 * 1e6,
                zmin=300e-9 * 1e6,  # stacked above via
                material="Nb",
            ),
            "Airbridge_Via": LayerLevel(
                name="Airbridge_Via",
                layer=L.AB_VIA,
                thickness=100e-9 * 1e6,
                zmin=200e-9 * 1e6,  # stacked above M1
                material="Nb",
            ),
            # JJ_AREA can be exported as a thin film if you use it in EM
            "JosephsonJunction": LayerLevel(
                name="JosephsonJunction",
                layer=L.JJ_AREA,
                thickness=70e-9 * 1e6,
                zmin=0,
                material="AlOx/Al",
                mesh_order=2,
            ),
            "TSV": LayerLevel(
                name="TSV",
                layer=L.TSV,
                thickness=500,  # full substrate thickness
                zmin=-500,  # starting at bottom
                material="TiN",
                mesh_order=3,
            ),
            "IndiumBump": LayerLevel(
                name="IndiumBump",
                layer=L.IND,
                thickness=10,  # 10 microns
                zmin=200e-9 * 1e6,  # stacked above M1
                material="In",
                mesh_order=3,
            ),
        }
    )


LAYER_STACK = get_layer_stack()


def get_gatemon_layer_stack(hbn_thickness: float = 0.03) -> LayerStack:
    """Return a planar approximation of the nanotube gatemon fabrication stack.

    Film thicknesses follow the device A recipe in Appendix I of
    :cite:`riechertCarbonNanotubeGatemon2025`. Heights are in µm, with zero
    at the oxide surface. The hBN thickness defaults to a nominal 30 nm;
    use the measured flake thickness for a particular device.

    The nanotube placement trace is deliberately not extruded as PEC metal.
    Its weak link belongs in the circuit model. The Al reconnection patches
    bridge the contact and M1 heights. Flake sag and conformal metal step
    coverage need a dedicated mesh for quantitative local electrostatics.
    The scalar hBN permittivity approximates the out-of-plane response.

    Args:
        hbn_thickness: Measured dielectric thickness in µm.

    Raises:
        ValueError: If the dielectric thickness is not positive.
    """
    if hbn_thickness <= 0:
        raise ValueError("hbn_thickness must be positive")
    return LayerStack(
        layers={
            "Substrate": LAYER_STACK.layers["Substrate"].model_copy(
                deep=True, update={"zmin": -500.15}
            ),
            "Oxide": LayerLevel(
                layer=L.SIM_AREA,
                thickness=0.15,
                zmin=-0.15,
                material="SiO2",
                mesh_order=3,
            ),
            "M1": LAYER_STACK.layers["M1"].model_copy(
                deep=True, update={"thickness": 0.155}
            ),
            "ContactNb": LayerLevel(
                layer=L.CNT_CONTACT,
                thickness=0.035,
                zmin=0,
                material="Nb",
                mesh_order=1,
            ),
            "ContactAu": LayerLevel(
                layer=L.CNT_CONTACT,
                thickness=0.01,
                zmin=0.035,
                material="Au",
                mesh_order=1,
            ),
            "hBN": LayerLevel(
                layer=L.HBN,
                thickness=hbn_thickness,
                zmin=0.045,
                material="hBN",
                mesh_order=2,
            ),
            "TopGate": LayerLevel(
                layer=L.TOP_GATE,
                thickness=0.1,
                zmin=0.045 + hbn_thickness,
                material="Al",
                mesh_order=1,
            ),
            "Reconnection": LayerLevel(
                layer=L.JJ_PATCH,
                thickness=0.1,
                zmin=0.045,
                material="Al",
                mesh_order=1,
            ),
            "Vacuum": LAYER_STACK.layers["Vacuum"].model_copy(
                deep=True, update={"zmin": 0}
            ),
        }
    )


LAYER_STACK_GATEMON = get_gatemon_layer_stack()
LAYER_STACK_GATEMON_NO_VACUUM = LayerStack(
    layers={
        name: level
        for name, level in LAYER_STACK_GATEMON.layers.items()
        if name != "Vacuum"
    }
)


def get_layer_stack_3d() -> LayerStack:
    """Return a mask preview with ground bounded by the layout extent.

    Includes the gatemon contact, dielectric and gate films at their nominal
    process heights, and the drawn M2, MB, MB2 and TSV2 shapes of the flip-chip
    and backside faces. The ground is an unconditional blanket film, even without
    an M1 etch mask. It can cover lower films and fill additive-only gaps.
    Use a physical process stack for those devices, simulations and already
    prepared positive metal geometry.
    """
    stack = LAYER_STACK.model_copy(deep=True)
    stack.layers["M1_ground"] = stack.layers["M1_ground"].model_copy(
        update={
            "layer": LogicalLayer(layer=L.M1_ETCH),
            "derived_layer": None,
            "background": True,
            "background_exclude_layers": [L.M1_ETCH, L.M1_DRAW],
        }
    )
    stack.layers.update({
        name: LAYER_STACK_GATEMON.layers[name].model_copy(deep=True)
        for name in ("ContactNb", "ContactAu", "hBN", "TopGate", "Reconnection")
    })
    # JJ_PATCH is shared by base-process junctions and gatemon contacts.
    stack.layers["Reconnection"].zmin = stack.layers["JosephsonJunction"].zmin
    # The other chip faces preview only their drawn metal: a blanket ground
    # there would appear under every layout, flip-chip or not.
    physical = LAYER_STACK_FLIP_CHIP_BACKSIDE.layers
    stack.layers.update({
        name: physical[name].model_copy(deep=True) for name in ("M2", "TSV2")
    })
    for name, draw in (("MB", L.MB_DRAW), ("MB2", L.MB2_DRAW)):
        stack.layers[name] = physical[name].model_copy(
            update={"layer": LogicalLayer(layer=draw), "derived_layer": None}
        )
    return stack


def get_layer_material_properties(
    layer_name: str, layer_stack: LayerStack | None = None
) -> dict[str, float]:
    """Returns the ``material_properties`` entry for a layer stack level's material.

    Simulation setups should read physical constants through this function so the
    numbers stay defined once, by the layer stack and ``material_properties``.

    Args:
        layer_name: Name of the level in the layer stack, e.g. ``"Substrate"``.
        layer_stack: Layer stack to look the level up in. Defaults to ``LAYER_STACK``.

    Raises:
        KeyError: If the level is not in the layer stack, or its material is not
            in ``material_properties``.
    """
    stack = layer_stack or LAYER_STACK
    material = stack.layers[layer_name].material
    if material not in material_properties:
        raise KeyError(
            f"Material {material!r} of layer {layer_name!r} is not in material_properties"
        )
    return material_properties[material]


# Nicer for 3D visualization
LAYER_STACK_NO_VACUUM = LayerStack(
    layers={
        name: level for name, level in LAYER_STACK.layers.items() if name != "Vacuum"
    }
)
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp)

LAYER_CONNECTIVITY: Sequence[ConnectivitySpec] = [
    ("M1_DRAW", "TSV", "M2_DRAW"),
    ("M1_DRAW", "TSV", "MB_DRAW"),
    ("M2_DRAW", "TSV2", "MB2_DRAW"),
    ("M1_DRAW", "IND", "M2_DRAW"),
    ("M1_DRAW", "AB_DRAW", "M1_DRAW"),
    ("M2_DRAW", "AB_DRAW", "M2_DRAW"),
    ("M1_DRAW", "JJ_PATCH", "CNT_CONTACT"),
    ("M1_DRAW", "TOP_GATE"),
]


LAYER_STACK_FLIP_CHIP = LayerStack(
    layers={
        **LAYER_STACK.layers,
        "M2": LayerLevel(
            name="M2",
            layer=L.M2_DRAW,
            thickness=0.2,
            zmin=10.0,
            material="Nb",
            mesh_order=1,
        ),
        "M2_ground": LayerLevel(
            name="M2_ground",
            layer=DerivedLayer(
                layer1=LogicalLayer(layer=L.SIM_AREA),
                layer2=DerivedLayer(
                    layer1=LogicalLayer(layer=L.M2_ETCH),
                    layer2=LogicalLayer(layer=L.M2_DRAW),
                    operation="or",
                ),
                operation="not",
            ),
            derived_layer=LogicalLayer(layer=L.M2_ETCH),
            thickness=0.2,
            zmin=10.0,
            material="Nb",
            mesh_order=1,
        ),
        "Substrate_top": LayerLevel(
            name="Substrate_top",
            layer=L.SIM_AREA,
            thickness=500,  # 500 microns of silicon
            zmin=10.2,  # below metal
            material="Si",
            mesh_order=4,
            background=True,
        ),
    }
)
LAYER_STACK_FLIP_CHIP_NO_VACUUM = LayerStack(
    layers={
        name: level
        for name, level in LAYER_STACK_FLIP_CHIP.layers.items()
        if name != "Vacuum"
    }
)


def _backside_metal_level(
    name: str, draw: Layer, etch: Layer, zmin: float
) -> LayerLevel:
    """Ground-plane style backside metal: ``SIM_AREA - (ETCH - DRAW)``."""
    return LayerLevel(
        name=name,
        layer=DerivedLayer(
            layer1=LogicalLayer(layer=L.SIM_AREA),
            layer2=DerivedLayer(
                layer1=LogicalLayer(layer=etch),
                layer2=LogicalLayer(layer=draw),
                operation="not",
            ),
            operation="not",
        ),
        derived_layer=LogicalLayer(layer=draw),
        thickness=0.2,
        zmin=zmin,
        material="Nb",
        mesh_order=1,
    )


def _without_vacuum(stack: LayerStack) -> LayerStack:
    """Copy of ``stack`` without its vacuum levels, for solvers that add their own."""
    return LayerStack(
        layers={
            name: level
            for name, level in stack.layers.items()
            if not name.startswith("Vacuum")
        }
    )


# Backside (side B) metal on the bottom face of the substrate, see
# :cite:`mallekFabricationSuperconductingThroughsilicon2021`
_BACKSIDE_LEVELS = {
    "MB": _backside_metal_level("MB", L.MB_DRAW, L.MB_ETCH, zmin=-500.2),
    "Vacuum_bottom": LayerLevel(
        name="Vacuum_bottom",
        layer=L.SIM_AREA,
        thickness=500,  # 500 microns of vacuum below backside metal
        zmin=-1000.2,
        material="vacuum",
        mesh_order=99,
    ),
}

LAYER_STACK_BACKSIDE = LayerStack(layers={**LAYER_STACK.layers, **_BACKSIDE_LEVELS})
LAYER_STACK_BACKSIDE_NO_VACUUM = _without_vacuum(LAYER_STACK_BACKSIDE)

# Both chips of the flip-chip assembly with metal on both faces: the bottom
# chip as in LAYER_STACK_BACKSIDE, and the top chip mirrored about the gap,
# with TSV2 through Substrate_top joining M2 to MB2 on its outer face.
_TOP_CHIP_OUTER_FACE = 10.2 + 500  # M2 film plus Substrate_top
LAYER_STACK_FLIP_CHIP_BACKSIDE = LayerStack(
    layers={
        **LAYER_STACK_FLIP_CHIP.layers,
        **_BACKSIDE_LEVELS,
        "TSV2": LayerLevel(
            name="TSV2",
            layer=L.TSV2,
            thickness=500,  # full top substrate thickness
            zmin=10.2,  # from M2 up through Substrate_top
            material="TiN",
            mesh_order=3,
        ),
        "MB2": _backside_metal_level(
            "MB2", L.MB2_DRAW, L.MB2_ETCH, zmin=_TOP_CHIP_OUTER_FACE
        ),
        "Vacuum_top": LayerLevel(
            name="Vacuum_top",
            layer=L.SIM_AREA,
            thickness=500,  # 500 microns of vacuum above the top chip
            zmin=_TOP_CHIP_OUTER_FACE + 0.2,
            material="vacuum",
            mesh_order=99,
        ),
    }
)
LAYER_STACK_FLIP_CHIP_BACKSIDE_NO_VACUUM = _without_vacuum(
    LAYER_STACK_FLIP_CHIP_BACKSIDE
)

LAYER_STACK_3D = get_layer_stack_3d()

############################
# Cross-sections functions
############################

cross_sections: dict[str, Callable[..., CrossSection]] = {}
_cross_section_default_names: dict[str, str] = {}


def xsection(func: Callable[..., CrossSection]) -> Callable[..., CrossSection]:
    """Returns decorated to register a cross section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

    .. code-block:: python

        @xsection
        def strip(width=TECH.width_strip, radius=TECH.radius_strip):
            return gf.cross_section.cross_section(width=width, radius=radius)
    """
    default_cross_section = func()
    _cross_section_default_names[default_cross_section.name] = func.__name__

    @wraps(func)
    def decorated_cross_section(**kwargs: Any) -> CrossSection:
        cross_section = func(**kwargs)
        if cross_section.name in _cross_section_default_names:
            # Pydantic private attributes stay writable on frozen models (gdsfactory
            # does the same); pyrefly treats every field of a frozen model as read-only.
            cross_section._name = _cross_section_default_names[  # pyrefly: ignore[read-only]
                cross_section.name
            ]
        return cross_section

    cross_sections[func.__name__] = decorated_cross_section
    return decorated_cross_section


def get_etch_sections(
    cross_section: CrossSectionSpec,
) -> Generator[gf.Section, None, None]:
    """Yields all etch sections of a cross-section.

    Args:
        cross_section: The cross-section to search for etch sections.

    Yields:
        Generator of sections with "etch" in their name.
    """
    xs = gf.get_cross_section(cross_section)
    for s in xs.sections:
        if s.name and "etch" in s.name:
            yield s


def get_etch_section(
    cross_section: CrossSectionSpec,
) -> gf.Section:
    """Returns the first etch section of a cross-section.

    Args:
        cross_section: The cross-section to search for an etch section.

    Returns:
        The first section with "etch" in its name.

    Raises:
        ValueError: If no etch section is found.
    """
    try:
        return next(get_etch_sections(cross_section))
    except StopIteration as e:
        xs = gf.get_cross_section(cross_section)
        msg = (
            f"Cross-section '{xs.name}' does not have a section with 'etch' in the name. "
            f"Found sections: {[s.name for s in xs.sections]}"
        )
        raise ValueError(msg) from e


def get_meander_wire_gap(
    cross_section: CrossSection,
    wire_gap: float | None = None,
) -> float:
    """Resolve the run-to-run gap of a meander inductor.

    An explicit gap wins. Otherwise the gap is twice the etch section width,
    so the etched regions of adjacent runs do not overlap, falling back to
    the wire width for cross-sections without an etch section.

    Shared by the meander inductor cell and its SAX model so the two cannot
    drift apart.

    Args:
        cross_section: Resolved cross-section of the meander wire.
        wire_gap: Explicit gap between adjacent meander runs in µm, if given.

    Returns:
        Gap between adjacent meander runs in µm.

    Raises:
        ValueError: If an explicit wire_gap is not positive.
    """
    if wire_gap is not None:
        # Concrete scalars only: branching on a jax tracer would break jit.
        if isinstance(wire_gap, int | float) and not wire_gap > 0:
            raise ValueError(f"wire_gap must be positive, got {wire_gap}")
        return wire_gap
    try:
        return 2 * get_etch_section(cross_section).width
    except ValueError:
        return cross_section.width


@xsection
def coplanar_waveguide(
    width: float = 10,
    gap: float = 6,
    waveguide_layer: LayerSpec = LAYER.M1_DRAW,
    etch_layer: LayerSpec = LAYER.M1_ETCH,
    radius: float | None = 100,
) -> CrossSection:
    """Return a coplanar waveguide cross_section.

    The cross_section is considered negative (etched) on the physical layer.

    Note:
        Assuming a silicon substrate thickness of 500 µm and a metal thickness of 100 nm,
        the default center conductor width and gap dimensions give a characteristic
        impedance of approximately 50 Ω.

    Args:
        width: center conductor width in µm.
        gap: gap between center conductor and ground in µm.
        waveguide_layer: for the center conductor (positive) region.
        etch_layer: for the etch (negative) region.
        radius: bend radius (if applicable).
    """
    return gf.cross_section.cross_section(
        width=width,
        layer=waveguide_layer,
        radius=radius,
        sections=(
            gf.Section(
                width=gap,
                offset=(gap + width) / 2,
                layer=etch_layer,
                name="etch_offset_pos",
            ),
            gf.Section(
                width=gap,
                offset=-(gap + width) / 2,
                layer=etch_layer,
                name="etch_offset_neg",
            ),
            gf.Section(width=width, layer=LAYER.WG, name="waveguide"),
        ),
        radius_min=(width + 2 * gap) / 2,
    )


cpw = coplanar_waveguide
etch = etch_only = partial(
    coplanar_waveguide,
    waveguide_layer=LAYER.M1_ETCH,
)


@xsection
def coplanar_waveguide_backside() -> CrossSection:
    """Return a coplanar waveguide cross-section on the backside (side B) metal.

    Same default dimensions as :func:`coplanar_waveguide`, drawn on ``MB_DRAW`` and
    ``MB_ETCH``. Connects to front-side ``M1`` through TSVs, see
    :func:`~qpdk.cells.tsv_transition_double_sided`.
    """
    return coplanar_waveguide(waveguide_layer=LAYER.MB_DRAW, etch_layer=LAYER.MB_ETCH)


cpw_backside = coplanar_waveguide_backside


@xsection
def coplanar_waveguide_top() -> CrossSection:
    """Return a coplanar waveguide cross-section on the flip-chip top chip metal.

    Same default dimensions as :func:`coplanar_waveguide`, drawn on ``M2_DRAW`` and
    ``M2_ETCH``.
    """
    return coplanar_waveguide(waveguide_layer=LAYER.M2_DRAW, etch_layer=LAYER.M2_ETCH)


cpw_top = coplanar_waveguide_top


@xsection
def coplanar_waveguide_top_backside() -> CrossSection:
    """Return a coplanar waveguide cross-section on the top chip backside metal.

    Same default dimensions as :func:`coplanar_waveguide`, drawn on ``MB2_DRAW`` and
    ``MB2_ETCH``. Connects to ``M2`` through ``TSV2``, see :data:`CHIP_LAYERS`.
    """
    return coplanar_waveguide(waveguide_layer=LAYER.MB2_DRAW, etch_layer=LAYER.MB2_ETCH)


cpw_top_backside = coplanar_waveguide_top_backside


@dataclass(frozen=True)
class ChipLayers:
    """Layers of one chip that carries metal on both faces.

    The front face looks into the flip-chip gap (``M1`` on the bottom chip,
    ``M2`` on the top chip) and the back face is the outer face of the
    substrate. The TSV layer runs through the substrate between them.
    Backside geometry is drawn in the same top-down frame as the front face.
    """

    front_draw: Layer
    front_etch: Layer
    back_draw: Layer
    back_etch: Layer
    tsv: Layer
    front_cross_section: str
    back_cross_section: str


Chip = Literal["bottom", "top"]

#: Per-chip layers for TSV cells and simulations, indexed by their ``chip``
#: argument. ``"bottom"`` is the (single or flip-chip) base chip, ``"top"`` the
#: flip-chip top chip of :data:`LAYER_STACK_FLIP_CHIP_BACKSIDE`.
CHIP_LAYERS: dict[Chip, ChipLayers] = {
    "bottom": ChipLayers(
        front_draw=LAYER.M1_DRAW,
        front_etch=LAYER.M1_ETCH,
        back_draw=LAYER.MB_DRAW,
        back_etch=LAYER.MB_ETCH,
        tsv=LAYER.TSV,
        front_cross_section="cpw",
        back_cross_section="cpw_backside",
    ),
    "top": ChipLayers(
        front_draw=LAYER.M2_DRAW,
        front_etch=LAYER.M2_ETCH,
        back_draw=LAYER.MB2_DRAW,
        back_etch=LAYER.MB2_ETCH,
        tsv=LAYER.TSV2,
        front_cross_section="cpw_top",
        back_cross_section="cpw_top_backside",
    ),
}


@xsection
def meander_inductor_cross_section() -> CrossSection:
    """Return a narrow coplanar waveguide cross-section for meander inductors.

    The default dimensions are width=2.0 µm and gap=2.0 µm.
    """
    return coplanar_waveguide(width=2.0, gap=2.0)


@xsection
def superinductor_cross_section() -> CrossSection:
    """Return an ultra-narrow coplanar waveguide cross-section for fluxonium superinductors.

    The default dimensions (width=0.04 µm, gap=0.09 µm) are based on NbTiN superinductor and SNSPD
    reports :cite:`hazardNanowireSuperinductanceFluxonium2019,yangComparisonSuperconductingNanowire2018`.
    """
    return coplanar_waveguide(
        width=0.04, gap=0.09, waveguide_layer=LAYER.NbTiN, etch_layer=LAYER.M1_ETCH
    )


@xsection
def launcher_cross_section_big() -> gf.CrossSection:
    """Return a large coplanar waveguide cross-section for a launcher.

    This cross-section is designed for the wide end of the launcher,
    providing a large area for probe pads and wirebonding.

    The default dimensions are taken from :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    return coplanar_waveguide(width=200.0, gap=110.0, etch_layer=LAYER.M1_ETCH)


@xsection
def josephson_junction_cross_section_wide() -> gf.CrossSection:
    """Return cross-section for the wide end of a Josephson junction wire.

    The default dimensions are taken from :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    return gf.cross_section.cross_section(
        width=0.2,
        layer=LAYER.JJ_AREA,
    )


@xsection
def josephson_junction_cross_section_narrow() -> gf.CrossSection:
    """Return cross-section for the narrow end of a Josephson junction wire.

    The default dimensions are taken from :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    return gf.cross_section.cross_section(
        width=0.09,
        layer=LAYER.JJ_AREA,
    )


@xsection
def microstrip(
    width: float = 10,
    layer: LayerSpec = "M1_DRAW",
) -> CrossSection:
    """Return a microstrip cross_section.

    The cross_section is considered additive (positive) on the layer.
    """
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
    )


strip = strip_metal = microstrip

############################
# Routing functions
############################


def _route_cross_section(
    cross_section: CrossSectionSpec, width: float | None
) -> CrossSectionSpec:
    """Fold the router's separate width into the cross-section specification."""
    if width is None or width == gf.get_cross_section(cross_section).width:
        return cross_section
    if isinstance(cross_section, str):
        return {"cross_section": cross_section, "settings": {"width": width}}
    if callable(cross_section) and hasattr(cross_section, "__name__"):
        return {
            "cross_section": cross_section.__name__,
            "settings": {"width": width},
        }
    if isinstance(cross_section, dict):
        settings = {**cross_section.get("settings", {}), "width": width}
        return {**cross_section, "settings": settings}
    return gf.get_cross_section(cross_section, width=width)


def _route_component(
    component: str,
    width: float | None = None,
    cross_section: CrossSectionSpec = cpw,
    **settings: Any,
) -> gf.Component | gf.ComponentAllAngle:
    """Build a routing cell with width owned by its cross-section."""
    return gf.get_cell(component)(
        cross_section=_route_cross_section(cross_section, width),
        **settings,
    )


route_bundle = route_bundle_cpw = partial(
    gf.routing.route_bundle,
    cross_section="coplanar_waveguide",
    bend=partial(_route_component, "bend_circular", allow_min_radius_violation=True),
    collision_check_layers=[LAYER.WG],
    on_collision="error",
    straight=partial(_route_component, "straight"),
    sbend=partial(_route_component, "bend_s", allow_min_radius_violation=True),
)
route_bundle_all_angle = route_bundle_all_angle_cpw = partial(
    gf.routing.route_bundle_all_angle,
    cross_section="coplanar_waveguide",
    separation=3,
    bend=partial(
        _route_component,
        "bend_circular_all_angle",
        allow_min_radius_violation=True,
    ),
    straight=partial(_route_component, "straight_all_angle"),
)
route_bundle_sbend = route_bundle_sbend_cpw = partial(
    gf.routing.route_bundle_sbend,
    cross_section="coplanar_waveguide",
    bend_s=partial(_route_component, "bend_s", allow_min_radius_violation=True),
)


def _route_single_cpw(
    component: gf.Component,
    ports1: Sequence[gf.Port],
    ports2: Sequence[gf.Port],
    cross_section: CrossSectionSpec = cpw,
    **settings: Any,
) -> list[ManhattanRoute]:
    """Place each explicit CPW path without the bundle router's fanout detours."""
    if "waypoints" in settings:
        settings["waypoints"] = [tuple(point) for point in settings["waypoints"]]
    return [
        gf.routing.route_single(
            component,
            start,
            finish,
            cross_section=cross_section,
            bend="bend_circular",
            straight="straight",
            **settings,
        )
        for start, finish in zip(ports1, ports2, strict=True)
    ]


routing_strategies: RoutingStrategies = {
    "route_bundle": route_bundle,
    "route_bundle_cpw": route_bundle_cpw,
    "route_bundle_all_angle": route_bundle_all_angle,
    "route_bundle_all_angle_cpw": route_bundle_all_angle_cpw,
    "route_bundle_sbend": route_bundle_sbend,
    "route_bundle_sbend_cpw": route_bundle_sbend_cpw,
    "route_single_cpw": _route_single_cpw,
}

if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # De-nest layers
    LAYERS_ACCORDING_TO_YAML = denest_layerviews_to_layer_tuples(LAYER_VIEWS)
    print("LAYERS_ACCORDING_TO_YAML = {")
    for yaml_layer_name, yaml_layer_tuple in LAYERS_ACCORDING_TO_YAML.items():
        print(f"\t{yaml_layer_name}: Layer = {yaml_layer_tuple}")
    print("}")

    klayout_tech = KLayoutTechnology(
        name="qpdk",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=LAYER_CONNECTIVITY,
    )
    klayout_tech.write_tech(tech_dir=PATH.klayout)
    # print(DEFAULT_CROSS_SECTION_NAMES)
    # print(strip() is strip())
    # print(strip().name, strip().name)
    # c = gf.c.bend_euler(cross_section="metal_routing")
    # c.pprint_ports()
