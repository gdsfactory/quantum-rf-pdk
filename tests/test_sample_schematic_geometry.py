"""Geometry parity between checked-in schematics and their layout fallbacks.

``.github/check-schematic-sources.py`` compares logical topology only, so a
fallback can match every instance, connection, and port while materializing
different placements or routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gdsfactory as gf
import kfactory as kf
import pytest
from conftest import import_gfp_module
from klayout.db import Region

pytestmark = pytest.mark.gfp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "qpdk/samples"

PDK_QUALIFIED_NAME = "qpdk.PDK"

type Layer = tuple[int, int]

_gfp_nyancir = import_gfp_module("gdsfactoryplus.nyancir_to_dschematic")
_gfp_dschematic = import_gfp_module("gdsfactoryplus.dschematic_to_gds")

# Pairs whose geometry parity is known-broken. strict=True fails the suite if
# one starts passing, so a fixed pair must drop its entry here.
XFAIL_REASONS = {
    # The shipped gdsfactoryplus SDK resolves a qualified model id through the
    # kfactory factory registries, which never contain bare functools.partial
    # cells, so materialize_dschematic raises KeyError before any geometry
    # exists to compare.
    "flipmon_test_chip": (
        "cannot materialize: the SDK cannot resolve "
        "flipmon_with_resonator_and_probeline, registered in PDK.cells as a "
        "bare functools.partial invisible to the factory lookup"
    ),
    # The .gsch in-real-layout coordinates hold a different placement and
    # routing of the same netlist than the hand-tuned .pic.yml layout: device
    # pitch and route lengths differ, and no rigid frame transform maps one
    # onto the other (the frame handling itself is sound; the resonator pair
    # below matches exactly).
    "qubit_test_chip": (
        "the .gsch in-real-layout placement and routing is a different "
        "arrangement of the netlist than the .pic.yml layout"
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplePair:
    """A checked-in schematic and the layout fallback that should match it."""

    stem: str
    gsch: Path
    pic_yml: Path


def _sample_pairs() -> list[SamplePair]:
    """Pair every ``.gsch`` sample with its ``.pic.yml`` fallback."""
    gsch_paths = {path.stem: path for path in SAMPLE_DIR.glob("*.gsch")}
    pic_yml_paths = {
        path.name.removesuffix(".pic.yml"): path
        for path in SAMPLE_DIR.glob("*.pic.yml")
    }
    return [
        SamplePair(stem=stem, gsch=gsch_paths[stem], pic_yml=pic_yml_paths[stem])
        for stem in sorted(gsch_paths.keys() & pic_yml_paths.keys())
    ]


SAMPLE_PAIRS = _sample_pairs()

assert SAMPLE_PAIRS, f"no .gsch/.pic.yml sample pairs under {SAMPLE_DIR}"


def _materialize_gsch(gsch: Path) -> kf.DKCell:
    """Materialize a schematic with the direct GDSFactory+ layout pipeline."""
    result = _gfp_nyancir.nyancir_to_dschematic(str(gsch), PDK_QUALIFIED_NAME)
    cell, warnings = _gfp_dschematic.materialize_dschematic(
        result.top,
        kcl=result.kcl,
        schem_name=result.schem_name,
        factories=result.factories,
    )
    # Dropped connections can make degraded topology look geometrically valid.
    assert not warnings, f"{gsch.stem}: materialization warnings: {warnings}"
    return cell


def _materialize_fallback(pic_yml: Path, *, name: str) -> kf.DKCell:
    """Build a component from the fallback netlist."""
    return gf.read.from_yaml(pic_yml, name=name)


def _layer_regions(component: kf.DKCell) -> dict[Layer, Region]:
    """Map each drawn layer to its geometry, discarding cell names."""
    layout = component.kcl.layout
    regions: dict[Layer, Region] = {}
    for index in layout.layer_indexes():
        region = Region(component.kdb_cell.begin_shapes_rec(index)).merged()
        if not region.is_empty():
            info = layout.get_info(index)
            regions[info.layer, info.datatype] = region
    return regions


def _describe(regions: dict[Layer, Region], layer: Layer) -> str:
    """Summarize one side of a comparison on a single layer."""
    if layer not in regions:
        return "nothing drawn"
    region = regions[layer]
    return f"{region.count()} polygons, {region.area()} dbu^2, bbox {region.bbox()}"


def _assert_same_geometry(
    materialized: kf.DKCell, fallback: kf.DKCell, stem: str
) -> None:
    """Require identical per-layer geometry, including placements and routing."""
    materialized_regions = _layer_regions(materialized)
    fallback_regions = _layer_regions(fallback)

    differences: list[str] = []
    for layer in sorted(materialized_regions.keys() | fallback_regions.keys()):
        # Bind before XOR: chaining the dict.get calls confuses the type checker.
        lhs = materialized_regions.get(layer) or Region()
        rhs = fallback_regions.get(layer) or Region()
        xor = lhs ^ rhs
        if xor.is_empty():
            continue
        differences.append(
            f"layer {layer}: xor {xor.count()} polygons, {xor.area()} dbu^2, "
            f"bbox {xor.bbox()}; .gsch drew {_describe(materialized_regions, layer)}, "
            f".pic.yml drew {_describe(fallback_regions, layer)}"
        )
    assert not differences, (
        f"{stem}: the materialized .gsch and {stem}.pic.yml draw different "
        f"geometry:\n" + "\n".join(differences)
    )


def _params(pairs: list[SamplePair]) -> list[Any]:
    """Wrap the pairs so known-broken stems carry their xfail reason."""
    return [
        pytest.param(
            pair,
            marks=[pytest.mark.xfail(reason=reason, strict=True)]
            if (reason := XFAIL_REASONS.get(pair.stem))
            else [],
        )
        for pair in pairs
    ]


@pytest.mark.parametrize("pair", _params(SAMPLE_PAIRS), ids=lambda pair: pair.stem)
def test_schematic_geometry_matches_fallback(pair: SamplePair) -> None:
    """Materialize both sources and compare their exact drawn geometry."""
    materialized = _materialize_gsch(pair.gsch)
    fallback = _materialize_fallback(pair.pic_yml, name=f"{pair.stem}__fallback")

    _assert_same_geometry(materialized, fallback, pair.stem)
