"""The hyperbolic vertex coupler recipe builds valid, symmetric, open-ended geometry."""

import importlib.util
import json
import math
import sys
from itertools import product
from pathlib import Path
from types import ModuleType

import gdsfactory as gf
import pytest
from typer.testing import CliRunner

from qpdk.tech import LAYER

SCRIPTS = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the recipe as a module; patching its attributes affects its functions."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    name = "hyperbolic_vertex_coupler_recipe"
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / "hyperbolic_vertex_coupler.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def corners(recipe: ModuleType) -> list[dict[str, float]]:
    """Return the eight extreme points of the grid."""
    return [
        dict(zip(recipe.GRID, values, strict=True))
        for values in product(*((axis[0], axis[-1]) for axis in recipe.GRID.values()))
    ]


@pytest.mark.parametrize("q", [3, 4])
def test_corners_have_one_electrode_per_port_and_open_ends(
    recipe: ModuleType, q: int
) -> None:
    for point in corners(recipe):
        geometry = recipe.geometry(point, recipe.SETTINGS, 1, q=q)
        component = geometry.component
        dbu = component.kcl.dbu
        metal = component.get_region(LAYER.M1_DRAW).merged()
        assert metal.count() == q + 1
        assert tuple(geometry.terminal_ports) == tuple(f"o{k}" for k in range(1, q + 1))
        assert [port.name for port in component.ports] == list(geometry.terminal_ports)
        assert {port.port_type for port in component.ports} == {"electrical"}
        ground = metal
        for port in component.ports:
            x, y = (round(value / dbu) for value in port.center)
            pin = metal.interacting(
                gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2))
            )
            assert pin.count() == 1
            ground = ground - pin
        assert ground.count() == 1
        for port in component.ports:
            # Nothing of the ground within SLOT of the lead end.
            reach = recipe.SLOT - dbu
            x, y = port.center
            guard = gf.kdb.Region(
                gf.kdb.DPolygon
                .ellipse(gf.kdb.DBox(2 * reach, 2 * reach), 256)
                .moved(x, y)
                .to_itype(dbu)
            )
            assert (guard & ground).is_empty()
        # The domain is the footprint plus the pad, and centred on the cell.
        left, bottom, right, top = geometry.bounds
        radius = point["pin_radius"] + recipe.LEAD_LENGTH + recipe.SLOT
        radius += recipe.GROUND_MARGIN
        pad = recipe.SETTINGS.domain_pad
        assert (left, bottom, right, top) == pytest.approx(
            (-radius - pad, -radius - pad, radius + pad, radius + pad), abs=2 * dbu
        )


@pytest.mark.parametrize("q", [3, 4])
def test_every_grid_point_builds_and_is_symmetric(recipe: ModuleType, q: int) -> None:
    worst = 0.0
    for point in recipe.points(recipe.GRID):
        metal = (
            recipe
            .geometry(point, recipe.SETTINGS, 1, q=q)
            .component.get_region(LAYER.M1_DRAW)
            .merged()
        )
        assert metal.count() == q + 1
        rotated = metal.transformed(gf.kdb.ICplxTrans(1.0, 360.0 / q, False, 0, 0))
        worst = max(worst, (metal ^ rotated).area() / metal.area())
    # Not exactly zero: the cell's 128-gon disk is not 3-fold symmetric.
    assert worst < 1e-3


def test_missing_open_end_cut_is_rejected(recipe: ModuleType) -> None:
    # Without the slot beyond a lead, its pin would short to ground at the port.
    recipe.PDK.activate()
    device = gf.Component()
    strip = device.add_ref(gf.components.straight(length=100, cross_section="cpw"))
    ports = {"short": strip.ports["o1"]}
    dbu = device.kcl.dbu
    footprint = gf.kdb.Region(gf.kdb.DBox(-200, -100, 300, 100).to_itype(dbu))
    # The device has no ports of its own, so no slot is cut.
    with pytest.raises(ValueError, match="metal polygons"):
        recipe._assemble(
            footprint, device, ports, width=10.0, expected=2, settings=recipe.SETTINGS
        )


def test_short_slot_is_rejected(
    recipe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short_cuts(ports: list, width: float, dbu: float) -> gf.kdb.Region:
        """Slots only 2 um long, as wide as the clearance."""
        cuts = gf.kdb.Region()
        for port in ports:
            angle = math.radians(port.orientation)
            x, y = port.center
            path = gf.kdb.DPath(
                [
                    gf.kdb.DPoint(x, y),
                    gf.kdb.DPoint(x + 2 * math.cos(angle), y + 2 * math.sin(angle)),
                ],
                width + 12.0,
            )
            cuts.insert(path.polygon().to_itype(dbu))
        return cuts

    monkeypatch.setattr(recipe, "_cuts", short_cuts)
    point = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    with pytest.raises(ValueError, match="closer than"):
        recipe.geometry(point, recipe.SETTINGS, 1, q=3)


def test_coupon_has_two_strips_and_ground(recipe: ModuleType) -> None:
    geometry = recipe.coupon_geometry(recipe.SETTINGS)
    assert geometry.component.get_region(LAYER.M1_DRAW).merged().count() == 3
    assert tuple(geometry.terminal_ports) == ("short", "long")


def test_grid_and_settings_match_the_plan(recipe: ModuleType) -> None:
    assert recipe.GRID == {
        "pin_radius": [25.0, 32.0, 40.0, 50.0, 60.0, 80.0],
        "pin_gap": [2.0, 4.0, 8.0],
        "ground_gap": [3.0, 6.0, 12.0],
    }
    assert len(recipe.points(recipe.GRID)) == 54
    assert recipe.PHASE_A_GRID["pin_radius"] == [32.0, 40.0, 50.0]
    assert recipe.SETTINGS.permittivity == pytest.approx(11.45)
    assert recipe.dataset_name(4) == "hyperbolic_vertex_coupler_q4_palace"


def test_phase_a_and_b_split_the_grid(recipe: ModuleType) -> None:
    both = {
        tuple(point.values())
        for phase in ("a", "b")
        for point in recipe.points(recipe.phase_grid(phase))
    }
    assert both == {tuple(point.values()) for point in recipe.points(recipe.GRID)}
    with pytest.raises(ValueError, match="Phase"):
        recipe.phase_grid("c")


def test_job_counts_after_sharing_runs(recipe: ModuleType) -> None:
    pilot = recipe.unique_jobs(recipe.pilot_jobs())
    assert len(pilot) == 13
    seen = {job.key for job in pilot}
    new_counts = []
    for q, phase in [(3, "a"), (4, "a"), (None, "held-out"), (3, "b"), (4, "b")]:
        jobs = (
            recipe.production_jobs(3, phase) + recipe.production_jobs(4, phase)
            if q is None
            else recipe.production_jobs(q, phase)
        )
        new = [job for job in recipe.unique_jobs(jobs) if job.key not in seen]
        seen |= {job.key for job in new}
        new_counts.append(len(new))
    # The pilot already holds three phase A points and one phase B point.
    assert new_counts == [25, 26, 2, 26, 27]
    assert len(seen) == 119


def test_job_identity_ignores_grid_and_label(recipe: ModuleType) -> None:
    point = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    first = recipe.coupler_job(3, point)
    assert first.key == recipe.coupler_job(3, dict(point)).key
    assert first.key != recipe.coupler_job(4, point).key
    assert first.key != recipe.coupler_job(3, point, lead_length=60.0).key
    inputs = json.dumps(first.inputs)
    assert "grid" not in first.inputs
    for forbidden in ("Users", "alisa", "C:"):
        assert forbidden not in inputs


def test_submit_and_pilot_do_not_reach_the_cloud_without_execute(
    recipe: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Boom:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("cloud backend constructed")

    monkeypatch.setattr(sys.modules["_cloud"], "GsimBackend", Boom)
    runner = CliRunner()
    work = ["--workdir", str(tmp_path)]
    for args in (
        ["pilot", *work],
        ["submit", "--q", "3", "--phase", "a", *work],
        ["submit", "--q", "4", "--phase", "held-out", *work],
        ["dry-run", "--q", "4", "--phase", "b", *work],
    ):
        result = runner.invoke(recipe.app, args)
        assert result.exit_code == 0, result.output
        assert "batches of at most 5" in result.output
    result = runner.invoke(recipe.app, ["submit", "--q", "5", *work])
    assert result.exit_code != 0
