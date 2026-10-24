"""Tests for the Josephson junction array cells and the fluxonium array inductor."""

import math
from functools import partial
from typing import Any

import pytest
import scipy.constants
from kfactory import kdb

from qpdk.cells.fluxonium import fluxonium, fluxonium_with_bbox
from qpdk.cells.junction_array import (
    josephson_junction_array,
    josephson_junction_array_compact,
)
from qpdk.tech import LAYER

MANHATTAN: dict[str, Any] = {
    "n_junctions": 10,
    "junction_area": 0.04,
    "island_width": 0.5,
    "island_length": 1.0,
    "pitch": 1.2,
    "layout_style": "manhattan",
}


def _junction_region(component) -> kdb.Region:
    return kdb.Region(component.begin_shapes_rec(component.kcl.layer(*LAYER.JJ_AREA)))


def _port_on_polygon_edge(component, port_name: str) -> bool:
    region = _junction_region(component).merged()
    port = component.ports[port_name]
    probe = kdb.DBox(-0.01, -0.01, 0.01, 0.01).moved(*port.dcenter)
    return not (region & kdb.Region(probe.to_itype(component.kcl.dbu))).is_empty()


class TestJosephsonJunctionArrayCell:
    """Geometry and ``info`` of :func:`josephson_junction_array`."""

    @staticmethod
    def test_manucharyan_inductive_energy() -> None:
        """The default chain gives E_L ≈ 0.52 GHz as in Manucharyan et al. (2009)."""
        c = josephson_junction_array()
        e_l = (scipy.constants.hbar / (2 * scipy.constants.e)) ** 2 / (
            c.info["array_inductance"] * scipy.constants.h
        )
        assert e_l / 1e9 == pytest.approx(0.52, rel=0.01)
        assert c.info["chain_length"] == pytest.approx(20, rel=0.01)
        assert c.info["total_junction_area"] == pytest.approx(43 * 0.5)

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            MANHATTAN,
            MANHATTAN | {"n_folds": 2, "fold_spacing": 3.0},
            {"n_junctions": 1},
            {"n_junctions": 3, "n_folds": 2},
        ],
    )
    def test_ports_on_chain(kwargs: dict) -> None:
        """Both ports sit on the drawn chain and face away from it."""
        c = josephson_junction_array(**kwargs)
        assert c.ports["o1"].orientation == 180
        assert c.ports["o2"].orientation == 0
        assert _port_on_polygon_edge(c, "o1")
        assert _port_on_polygon_edge(c, "o2")

    @staticmethod
    @pytest.mark.parametrize("n_folds", [0, 2])
    def test_dolan_islands_are_separate(n_folds: int) -> None:
        """Dolan islands are separated by the bridge gap: N + 1 islands, leads merged."""
        c = josephson_junction_array(n_folds=n_folds)
        assert _junction_region(c).merged().count() == 43 + 1

    @staticmethod
    @pytest.mark.parametrize("n_folds", [0, 2])
    def test_manhattan_chain_is_connected(n_folds: int) -> None:
        """Manhattan fingers overlap at the junctions, so the chain is one polygon."""
        c = josephson_junction_array(**MANHATTAN, n_folds=n_folds, fold_spacing=3.0)
        assert _junction_region(c).merged().count() == 1

    @staticmethod
    def test_odd_folds_turn_o2_west() -> None:
        c = josephson_junction_array(n_folds=1)
        assert c.ports["o2"].orientation == 180

    @staticmethod
    def test_folding_shrinks_length() -> None:
        straight = josephson_junction_array()
        folded = josephson_junction_array(n_folds=2)
        assert folded.dxsize < straight.dxsize / 2
        assert folded.info["chain_length"] == straight.info["chain_length"]

    @staticmethod
    def test_ground_plane_distance() -> None:
        c = josephson_junction_array(ground_plane_distance=5.0)
        etch = kdb.Region(c.begin_shapes_rec(c.kcl.layer(*LAYER.M1_ETCH)))
        assert etch.count() == 1
        box = etch.bbox().to_dtype(c.kcl.dbu)
        assert box.width() == pytest.approx(c.dxsize)
        assert c.dxsize == pytest.approx(josephson_junction_array().dxsize + 10)

    @staticmethod
    def test_manhattan_island_info() -> None:
        c = josephson_junction_array(**MANHATTAN)
        tip = math.sqrt(MANHATTAN["junction_area"])
        body = MANHATTAN["island_length"] * MANHATTAN["island_width"]
        assert c.info["island_area"] > body
        assert c.info["island_area"] < body + 2 * tip * MANHATTAN["pitch"]

    @staticmethod
    def test_compact_variant() -> None:
        c = josephson_junction_array_compact()
        reference = josephson_junction_array(n_folds=2)
        assert c.dbbox() == reference.dbbox()
        assert c.info == reference.info
        assert c.ports["o2"].orientation == 0

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_junctions": 0}, "at least 1"),
            ({"n_folds": -1}, "non-negative"),
            ({"n_junctions": 2, "n_folds": 2}, "smaller than the number of rows"),
            ({"pitch": 0}, "must be positive"),
            ({"island_length": 0.5}, "bridge gap"),
            ({"junction_area": 1.0}, "exceeds island_length"),
            (MANHATTAN | {"pitch": 0.8}, "Manhattan geometry overlaps"),
            ({"layout_style": "bridge-free"}, "Unknown layout_style"),
            ({"n_folds": 2, "fold_spacing": 1.0}, "fold_spacing"),
            ({"ground_plane_distance": 0}, "ground_plane_distance"),
        ],
    )
    def test_invalid_geometry_raises(kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            josephson_junction_array(**kwargs)


class TestFluxoniumArrayInductor:
    """The ``inductor="array"`` option of :func:`fluxonium`."""

    @staticmethod
    def test_array_replaces_meander() -> None:
        meander = fluxonium()
        array = fluxonium(inductor="array")
        assert "inductor_n_turns" in meander.info
        assert array.info["inductor"] == "array"
        assert array.info["n_junctions"] == 43
        assert array.info["array_inductance"] == pytest.approx(
            josephson_junction_array().info["array_inductance"]
        )
        assert not _junction_region(array).is_empty()
        assert {p.name for p in array.ports} == {p.name for p in meander.ports}

    @staticmethod
    def test_with_bbox_passes_through() -> None:
        meander = fluxonium_with_bbox()
        array = fluxonium_with_bbox(inductor="array")
        assert _junction_region(array).count() > _junction_region(meander).count()

    @staticmethod
    def test_unknown_inductor_raises() -> None:
        with pytest.raises(ValueError, match="Unknown inductor"):
            fluxonium(inductor="coil")  # pyrefly: ignore[bad-argument-type]

    @staticmethod
    def test_odd_folds_rejected() -> None:
        with pytest.raises(ValueError, match="o1 facing west"):
            fluxonium(
                inductor="array",
                inductor_array_spec=partial(josephson_junction_array, n_folds=1),
            )

    @staticmethod
    def test_array_too_long_rejected() -> None:
        with pytest.raises(ValueError, match="does not fit"):
            fluxonium(
                inductor="array",
                inductor_array_spec=partial(josephson_junction_array, n_junctions=2000),
            )
