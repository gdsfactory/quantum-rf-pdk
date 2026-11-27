"""Meshing and config generation of the coupler recipe with the real gsim mesher.

Run with ``uv run --group palace --python 3.12 pytest -m palace``. It meshes
and validates the Palace config; it solves nothing and uploads nothing.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.palace

SCRIPTS = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("gsim.palace")
    pytest.importorskip("meshwell")
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


def test_mesh_is_repeatable_so_the_cloud_cache_can_match(
    recipe: ModuleType, tmp_path: Path
) -> None:
    settings = recipe.Settings(near_mesh=2.0, far_mesh=60.0, domain_pad=200.0, order=3)
    point = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    job = recipe.coupler_job(3, point, settings=settings)
    digests = []
    for attempt in range(2):
        prepared = job.prepare(tmp_path / str(attempt) / job.key)
        mesh = tmp_path / str(attempt) / job.key / "sim" / "palace.msh"
        digests.append((
            prepared.tetrahedra,
            hashlib.sha256(mesh.read_bytes()).hexdigest(),
        ))
    assert digests[0] == digests[1]


@pytest.mark.parametrize("q", [3, 4])
def test_coarse_mesh_has_one_terminal_per_pin_and_one_ground(
    recipe: ModuleType, tmp_path: Path, q: int
) -> None:
    settings = recipe.Settings(near_mesh=2.0, far_mesh=60.0, domain_pad=200.0, order=3)
    point = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    job = recipe.coupler_job(q, point, settings=settings)
    prepared = job.prepare(tmp_path / job.key)
    assert prepared.tetrahedra is not None
    assert prepared.tetrahedra > 0
    assert prepared.sim.validate_config().valid
    # gsim's folder holds only what is uploaded; bookkeeping sits next to it.
    uploaded = tmp_path / job.key / "sim"
    assert {"config.json", "palace.msh"} <= {p.name for p in uploaded.iterdir()}
    assert not {"inputs.json", "mesh.json", "job.json", "complete.json"} & {
        p.name for p in uploaded.iterdir()
    }
    config = json.loads((uploaded / "config.json").read_text("utf-8"))
    boundaries = config["Boundaries"]
    assert len(boundaries["Terminal"]) == q
    assert [entry["Index"] for entry in boundaries["Terminal"]] == list(range(1, q + 1))
    assert len(boundaries["Ground"]["Attributes"]) == 1
    assert config["Problem"]["Type"] == "Electrostatic"
    assert config["Solver"]["Order"] == 3
    assert config["Solver"]["Linear"]["Tol"] == pytest.approx(1e-9)
    assert not config["Solver"]["Linear"]["InitialGuess"]
    permittivities = sorted(m["Permittivity"] for m in config["Domains"]["Materials"])
    assert permittivities == pytest.approx([1.0, 11.45])
    assert [t.name for t in prepared.sim.terminals] == [
        f"o{k}" for k in range(1, q + 1)
    ]
