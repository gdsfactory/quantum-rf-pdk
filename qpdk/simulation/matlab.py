"""Geometry helpers for MATLAB field solves."""

import gdsfactory as gf
import numpy as np
from gdsfactory.pdk import Pdk

from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.tech import LAYER, material_properties


def component_to_terminal_polygons(
    component: gf.Component, layer: tuple[int, int] = LAYER.M1_DRAW
) -> list[np.ndarray]:
    """Return merged layer polygons in component port order."""
    metal = gf.kdb.Region(
        component.begin_shapes_rec(component.kcl.layer(*layer))
    ).merged()
    ports = tuple(component.ports)
    polygons: list[np.ndarray | None] = [None] * len(ports)
    for polygon in metal.each():
        hull = polygon.to_dtype(component.kcl.dbu)
        for index, port in enumerate(ports):
            if hull.bbox().enlarged(1e-3, 1e-3).contains(gf.kdb.DPoint(*port.center)):
                if polygons[index] is not None:
                    raise ValueError(f"Multiple polygons contain port {port.name}")
                polygons[index] = np.array([(p.x, p.y) for p in hull.each_point_hull()])

    if any(polygon is None for polygon in polygons):
        raise ValueError("Every port must lie on a metal polygon")
    return [polygon for polygon in polygons if polygon is not None]


def interdigital_capacitor_fem_geometry(
    pdk: Pdk,
    *,
    fingers: int,
    finger_length: float,
    finger_gap: float,
    finger_width: float,
) -> tuple[list[np.ndarray], np.ndarray, float, float, float]:
    """Return IDC conductors, bounds, film thickness, and dielectric constants."""
    idc = interdigital_capacitor(
        fingers=int(fingers),
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_width,
        etch_layer=None,
    )
    box = idc.dbbox()
    stack = pdk.layer_stack.layers
    eps_r = {
        name: props["relative_permittivity"]
        for name, props in material_properties.items()
    }
    return (
        component_to_terminal_polygons(idc),
        np.array([box.left, box.bottom, box.right, box.top]),
        float(stack["M1"].thickness),
        float(eps_r[stack["Substrate"].material]),
        float(eps_r[stack["Vacuum"].material]),
    )
