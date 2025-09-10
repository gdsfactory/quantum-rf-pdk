"""Technology definitions."""

from collections.abc import Callable
from functools import cache, partial, wraps
from typing import Any, cast

import gdsfactory as gf
from doroutes.bundles import add_bundle_astar
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
    Layer,
    LayerSpec,
)

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

    # flip-cihp equivalents
    M2_DRAW: Layer = (2, 0)
    M2_ETCH: Layer = (2, 1)

    # Airbridges
    AB_DRAW: Layer = (10, 0)  # Bridge metal
    AB_VIA: Layer = (10, 1)  # Landing pads / contacts

    # Junctions
    JJ_AREA: Layer = (20, 0)  # Optional bridge/overlap definition
    JJ_PATCH: Layer = (20, 1)

    # Packaging / 3D integration / backside / misc.
    IND: Layer = (30, 0)
    TSV: Layer = (31, 0)  # Throughs / vias / backside features
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


L = LAYER = LayerMapQPDK


@cache
def get_layer_stack() -> LayerStack:
    """Returns a LayerStack corresponding to the PDK.

    The stack roughly corresponds to that of :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    return LayerStack(
        layers={
            # Base metal film (e.g., 200 nm of Nb)
            "M1": LayerLevel(
                name="M1",
                layer=DerivedLayer(
                    layer1=LogicalLayer(layer=L.SIM_AREA),
                    # Drawing goes over etch
                    layer2=DerivedLayer(
                        layer1=LogicalLayer(layer=L.M1_ETCH),
                        layer2=LogicalLayer(layer=L.M1_DRAW),
                        operation="-",
                    ),
                    operation="-",
                ),
                derived_layer=LogicalLayer(layer=L.M1_DRAW),
                thickness=200e-9 * 1e6,
                zmin=0.0,  # top of substrate
                material="Nb",
                sidewall_angle=90.0,
                mesh_order=1,
            ),
            "Silicon": LayerLevel(
                name="Substrate",
                layer=LogicalLayer(layer=L.SIM_AREA),
                thickness=500e-6 * 1e6,  # 500 microns of silicon
                zmin=-500e-6 * 1e6,  # below metal
                material="Si",
                sidewall_angle=90.0,
                mesh_order=3,
            ),
            "vacuum": LayerLevel(
                name="Vacuum",
                layer=LogicalLayer(layer=L.SIM_AREA),
                thickness=500e-6 * 1e6,  # 500 microns of vacuum above metal
                zmin=200e-9 * 1e6,  # above metal
                material="vacuum",
                sidewall_angle=90.0,
                mesh_order=3,
            ),
            # Airbridge metal sitting above M1 (example: +300 nm)
            "AB_METAL": LayerLevel(
                layer=L.AB_DRAW,
                thickness=300e-9,
                zmin=200e-9,  # stacked above M1
                material="Nb",
                sidewall_angle=90.0,
                mesh_order=3,
            ),
            # JJ_AREA can be exported as a thin film if you use it in EM
            "jj_area": LayerLevel(
                layer=L.JJ_AREA,
                thickness=70e-9,
                zmin=0,
                material="AlOx/Al",
                sidewall_angle=90.0,
                mesh_order=4,
            ),
            "sim_only": LayerLevel(
                layer=L.SIM_ONLY,
                thickness=0e-9,
                zmin=0.0,
                material="vacuum",
                sidewall_angle=90.0,
                mesh_order=9,
            ),
            # You can add TSV/backside as real films if you simulate them
            # "tsv": LayerLevel(layer=L.TSV, thickness=... , zmin=... , material="Cu"),
        }
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    pass


TECH = Tech()

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
    default_xs = func()
    _cross_section_default_names[default_xs.name] = func.__name__

    @wraps(func)
    def newfunc(**kwargs: Any) -> CrossSection:
        xs = func(**kwargs)
        if xs.name in _cross_section_default_names:
            xs._name = _cross_section_default_names[xs.name]
        return xs

    cross_sections[func.__name__] = newfunc
    return newfunc


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
    )


cpw = coplanar_waveguide


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

route_single = route_single_cpw = partial(
    gf.routing.route_single,
    cross_section=cpw,
    bend="bend_circular",
)
route_bundle = route_bundle_cpw = partial(
    gf.routing.route_bundle,
    cross_section=cpw,
    bend="bend_circular",
)
route_single_sbend = route_single_sbend_cpw = partial(
    gf.routing.route_single_sbend,
    cross_section=cpw,
    bend_s="bend_s",
)
route_bundle_all_angle = route_bundle_all_angle_cpw = partial(
    gf.routing.route_bundle_all_angle,
    cross_section=cpw,
    separation=3,
    bend="bend_circular_all_angle",
    straight="straight_all_angle",
)
route_bundle_sbend = route_bundle_sbend_cpw = partial(
    gf.routing.route_bundle_sbend,
    cross_section=cpw,
    bend_s="bend_s",
)

route_astar = route_astar_cpw = partial(
    add_bundle_astar,
    layers=["M1_ETCH"],
    bend="bend_circular",
    straight="straight",
    grid_unit=500,
    spacing=3,
)
routing_strategies = dict(
    route_single=route_single,
    route_single_cpw=route_single_cpw,
    route_single_sbend=route_single_sbend,
    route_bundle=route_bundle,
    route_bundle_cpw=route_bundle_cpw,
    route_bundle_all_angle=route_bundle_all_angle,
    route_bundle_all_angle_cpw=route_bundle_all_angle_cpw,
    route_astar=route_astar,
    route_astar_cpw=route_astar_cpw,
)

if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # De-nest layers
    LAYERS_ACCORDING_TO_YAML = denest_layerviews_to_layer_tuples(LAYER_VIEWS)
    print("LAYERS_ACCORDING_TO_YAML = {")
    for yaml_layer_name, yaml_layer_tuple in LAYERS_ACCORDING_TO_YAML.items():
        print(f"\t{yaml_layer_name}: Layer = {yaml_layer_tuple}")
    print("}")

    connectivity = cast(list[ConnectivitySpec], [("M1_DRAW", "TSV", "M2_DRAW")])

    t = KLayoutTechnology(
        name="qpdk",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)
    # print(DEFAULT_CROSS_SECTION_NAMES)
    # print(strip() is strip())
    # print(strip().name, strip().name)
    # c = gf.c.bend_euler(cross_section="metal_routing")
    # c.pprint_ports()
