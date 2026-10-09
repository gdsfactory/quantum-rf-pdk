"""The simulated ground must match the component's subtractive etch mask."""

import runpy
from pathlib import Path

import pytest
from shapely import Polygon, union_all
from shapely.affinity import translate

from qpdk.cells import coupler_straight
from qpdk.tech import coplanar_waveguide

EXPERIMENT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "qpdk/models/datasets/data/cpw_coupling.py"
    )
)


@pytest.mark.parametrize(
    ("width", "cpw_gap", "gap"),
    [(10, 6, 8), (10, 6, 12), (10, 6, 16), (30, 20, 250), (2, 1, 2.5)],
)
def test_ground_strip_matches_the_drawn_etch_mask(
    width: float, cpw_gap: float, gap: float
) -> None:
    settings = EXPERIMENT["SETTINGS"]
    component = coupler_straight(
        length=settings.slice_length_um,
        gap=gap,
        cross_section=coplanar_waveguide(width=width, gap=cpw_gap),
    )
    etch = union_all([
        translate(Polygon(points), yoff=-(width + gap) / 2)
        for points in component.get_polygons_points(by="tuple")[1, 1]
    ])
    arguments = dict(width=width, cpw_gap=cpw_gap, gap=gap, settings=settings)
    sheets, footprint = EXPERIMENT["geometry"](
        **arguments, topology=EXPERIMENT["Topology"].AS_DRAWN
    )
    simulated_etch = footprint.difference(union_all(list(sheets.values())))
    assert simulated_etch.symmetric_difference(etch).area < 1e-8
    unshielded, _ = EXPERIMENT["geometry"](**arguments)
    ground_difference = sheets["ground"].difference(unshielded["ground"])
    assert ground_difference.area == pytest.approx(
        settings.slice_length_um * max(gap - 2 * cpw_gap, 0)
    )
