"""Every drum recipe point separates the resonator nets from ground."""

import itertools
import runpy
from pathlib import Path

import pytest

from qpdk.cells import drum_lc_resonator
from qpdk.tech import LAYER

SCRIPTS = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "drum.py"))


class TestDrumRecipe:
    """Drum recipe geometry without running Palace."""

    @staticmethod
    @pytest.mark.parametrize("corner", list(itertools.product((0, -1), repeat=4)))
    def test_nets(module: dict, corner: tuple[int, ...]) -> None:
        recipe = module["RECIPE"]
        point = {
            name: values[i]
            for (name, values), i in zip(recipe.grid.items(), corner, strict=True)
        }
        geometry = recipe.geometry(point, recipe.settings, 1)
        assert tuple(geometry.terminal_ports) == recipe.terminals
        # One extraction layer per terminal plus ground, one polygon each
        regions = geometry.component.get_polygons(by="tuple")
        assert len(regions) == len(recipe.terminals) + 1
        assert all(len(polygons) == 1 for polygons in regions.values())
        ports = geometry.component.ports
        assert ports["feed"].dy > ports["node"].dy

    @staticmethod
    def test_nominal_matches_default_cell(module: dict) -> None:
        """The lookup's default point is the default layout, minus the slits."""
        parts = module["layout"](module["NOMINAL"])
        cell = drum_lc_resonator()
        metal = cell.get_region(LAYER.M1_DRAW)
        electrodes = parts["electrodes"]
        devices = electrodes["node"].dup()
        for name, region in electrodes.items():
            if name.startswith("meander_"):
                devices += region
        # Every resonator net is cell metal, and the cell's resonator metal is
        # covered apart from the slits and the ground lead
        assert (devices - metal).is_empty()
        assert (electrodes["membrane"] ^ cell.get_region(LAYER.VG_TOP)).is_empty()

    @staticmethod
    def test_weights(module: dict) -> None:
        assert module["segment_runs"]() == [0, 15, 30, 45, 60, 75]
        weights = module["terminal_weights"]()
        assert weights[:3] == [1.0, 0.0, 0.0]
        assert weights[3:] == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
        assert module["RECIPE"].provenance["terminal_weights"] == weights

    @staticmethod
    def test_conductor_modes(module: dict) -> None:
        recipe = module["RECIPE"]
        sheet = recipe.geometry(module["NOMINAL"], recipe.settings, 1)
        solid = module["geometry"](
            module["NOMINAL"], recipe.settings, 1, conductors="solid"
        )
        gap = 0.07
        for geometry, bottom in ((sheet, 0.0), (solid, 0.2)):
            box = geometry.mesh_inputs["gap_refinement_box"]
            assert box["ZMin"] == pytest.approx(bottom)
            assert box["ZMax"] - box["ZMin"] == pytest.approx(gap)
            membrane = geometry.mesh_inputs["stack"]["layers"]["membrane"]
            assert membrane["zmin"] == pytest.approx(box["ZMax"])
        assert sheet.mesh_inputs["mesh"]["planar_conductors"]
        assert not solid.mesh_inputs["mesh"]["planar_conductors"]

    @staticmethod
    def test_rejects_bad_inputs(module: dict) -> None:
        recipe = module["RECIPE"]
        point = module["NOMINAL"]
        with pytest.raises(ValueError, match="positive"):
            recipe.geometry({**point, "coupling_gap": 0.0}, recipe.settings, 1)
        with pytest.raises(ValueError, match="conductors"):
            module["geometry"](point, recipe.settings, 1, conductors="thick")
        with pytest.raises(ValueError, match="drum_mesh"):
            module["geometry"](point, recipe.settings, 1, drum_mesh=5.0)
