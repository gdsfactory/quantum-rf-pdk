"""The simulated ground must match the component's subtractive etch mask."""

import runpy
from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk.cells import coupler_straight
from qpdk.tech import LAYER, coplanar_waveguide

EXPERIMENT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "qpdk/models/datasets/data/cpw_coupling.py"
    )
)


@pytest.mark.parametrize(
    ("width", "cpw_gap", "gap"),
    [(10, 6, 8), (10, 6, 12), (10, 6, 16), (30, 20, 250), (2, 1, 2.5), (2, 6, 1)],
)
def test_ground_strip_matches_the_drawn_etch_mask(
    width: float, cpw_gap: float, gap: float
) -> None:
    settings = EXPERIMENT["SETTINGS"]
    device = gf.Component()
    reference = device.add_ref(
        coupler_straight(
            length=settings.slice_length_um,
            gap=gap,
            cross_section=coplanar_waveguide(width=width, gap=cpw_gap),
        )
    )
    reference.dmovey(-(width + gap) / 2)
    etch = device.get_region(LAYER.M1_ETCH) - device.get_region(LAYER.M1_DRAW)
    arguments = dict(width=width, cpw_gap=cpw_gap, gap=gap, settings=settings)
    component, bounds = EXPERIMENT["geometry"](
        **arguments, topology=EXPERIMENT["Topology"].AS_DRAWN
    )
    footprint = gf.Component()
    left, bottom, right, top = bounds
    footprint.add_polygon(
        [(left, bottom), (right, bottom), (right, top), (left, top)],
        layer=LAYER.M1_DRAW,
    )
    metal = component.get_region(LAYER.M1_DRAW)
    simulated_etch = footprint.get_region(LAYER.M1_DRAW) - metal
    assert (simulated_etch ^ etch).is_empty()
    unshielded, _ = EXPERIMENT["geometry"](**arguments)
    ground_difference = metal - unshielded.get_region(LAYER.M1_DRAW)
    assert ground_difference.area() * component.kcl.dbu**2 == pytest.approx(
        settings.slice_length_um * max(gap - 2 * cpw_gap, 0)
    )
