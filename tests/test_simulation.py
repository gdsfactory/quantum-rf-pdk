"""Tests for the qpdk.simulation FEM helpers."""

import importlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cloudpickle
import gdsfactory as gf
import klayout.db as kdb
import pytest

from qpdk.cells import coupler_straight, flipmon_with_bbox, tsv_transition_double_sided
from qpdk.simulation import (
    FEM_LAYERS,
    FLIP_CHIP_FEM_LAYERS,
    RAY_PORT,
    TSV_FEM_LAYERS,
    SlurmCluster,
    SlurmJobError,
    cluster as cluster_module,
    flip_chip_stack,
    single_chip_stack,
    study as study_module,
    to_fem_regions,
    to_flip_chip_regions,
    to_tsv_regions,
    tsv_stack,
)
from qpdk.simulation.palace_run import (
    _slurm_mpi_launcher,
    add_domain_energy_postprocessing,
    domain_loss_tangents,
    palace_command,
    verify_port_connectivity,
)
from qpdk.simulation.trial import _summarise, main as run_trial, result_path
from qpdk.tech import LAYER


def _sim_layout() -> gf.Component:
    """Return a small layout with a simulation area and feed ports."""

    @gf.cell
    def _layout() -> gf.Component:
        c = gf.Component()
        ref = c << coupler_straight(gap=16.0, length=100.0)
        c.add_ports(ref.ports)
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    return _layout()


BUMP_RADIUS = 7.5  # µm, matches the cell's own indium bump


def _bump(x: float, y: float) -> kdb.DPolygon:
    """Return a circular bump polygon of the standard radius at (x, y)."""
    return kdb.DPolygon.ellipse(
        kdb.DBox(x - BUMP_RADIUS, y - BUMP_RADIUS, x + BUMP_RADIUS, y + BUMP_RADIUS), 64
    )


def _flip_chip_layout() -> gf.Component:
    """Return a flipmon layout with corner ground bumps and a simulation area."""

    @gf.cell
    def _flipmon_layout() -> gf.Component:
        c = gf.Component()
        ref = c << flipmon_with_bbox()
        c.add_ports(ref.ports)
        # Corner bumps tie the two chips' ground planes into a single node.
        for x, y in ((-150, -150), (150, -150), (-150, 150), (150, 150)):
            c.kdb_cell.shapes(LAYER.IND).insert(_bump(x, y))
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    return _flipmon_layout()


def test_to_fem_regions_ports_and_layers():
    c = to_fem_regions(_sim_layout())

    assert sorted(p.name for p in c.ports) == ["o1", "o2", "o3", "o4"]
    layout = c.kdb_cell.layout()
    for name, layer in FEM_LAYERS.items():
        region = c.kdb_cell.begin_shapes_rec(layout.layer(*layer))
        assert not region.at_end(), f"{name} empty"


def test_to_fem_regions_conductor_excludes_etch():
    component = _sim_layout()
    c = to_fem_regions(component)

    layout_in = component.kdb_cell.layout()
    etch = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout_in.layer(*LAYER.M1_ETCH))
    ).merged()
    layout_out = c.kdb_cell.layout()
    conductor = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout_out.layer(*FEM_LAYERS["SUPERCONDUCTOR"]))
    ).merged()
    sim_area = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout_out.layer(*FEM_LAYERS["SUBSTRATE"]))
    ).merged()
    assert (conductor & etch).is_empty()
    assert not conductor.is_empty()
    assert conductor.area() < sim_area.area()


def test_to_fem_regions_draw_bridges_etched_gap():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(LAYER.M1_ETCH).insert(kdb.DBox(48, 0, 52, 100))
    component.kdb_cell.shapes(LAYER.M1_DRAW).insert(kdb.DBox(45, 48, 55, 52))

    converted = to_fem_regions(component)
    layout = converted.kdb_cell.layout()
    conductor = kdb.Region(
        converted.kdb_cell.begin_shapes_rec(layout.layer(*FEM_LAYERS["SUPERCONDUCTOR"]))
    ).merged()

    assert len(list(conductor.each())) == 1
    assert (conductor & kdb.Region(kdb.Box(49000, 49000, 51000, 51000))).area() > 0
    assert (conductor & kdb.Region(kdb.Box(49000, 10000, 51000, 20000))).is_empty()
    source_layout = component.kdb_cell.layout()
    assert not component.kdb_cell.begin_shapes_rec(
        source_layout.layer(*LAYER.M1_DRAW)
    ).at_end()


def _require_gsim() -> None:
    """Skip unless gsim imports; gmsh can raise OSError, not just ImportError."""
    try:
        importlib.import_module("gsim")
    except (ImportError, OSError):
        pytest.skip("gsim unavailable (missing package or GL system libraries)")


def test_single_chip_stack_matches_material_properties():
    _require_gsim()

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["SUPERCONDUCTOR"].thickness == 0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)
    assert stack.materials["qpdk-silicon"]["loss_tangent"] == pytest.approx(2.7e-6)


def test_layer_stacks_without_optional_gsim(monkeypatch):
    """Check both exported stacks when gsim's native libraries are missing."""
    gsim = ModuleType("gsim")
    common = ModuleType("gsim.common")
    stack_module = ModuleType("gsim.common.stack")
    materials = ModuleType("gsim.common.stack.materials")

    class Layer(SimpleNamespace):
        pass

    class LayerStack:
        def __init__(self, pdk_name):
            self.pdk_name = pdk_name
            self.layers = {}
            self.materials = {}

    stack_module.__dict__.update(Layer=Layer, LayerStack=LayerStack)
    materials.__dict__["MATERIALS_DB"] = {
        "vacuum": SimpleNamespace(to_dict=lambda: {"permittivity": 1.0})
    }
    for name, module in (
        ("gsim", gsim),
        ("gsim.common", common),
        ("gsim.common.stack", stack_module),
        ("gsim.common.stack.materials", materials),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.pdk_name == "qpdk"
    assert stack.layers["SUBSTRATE"].gds_layer == FEM_LAYERS["SUBSTRATE"]
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["SUPERCONDUCTOR"].zmin == pytest.approx(200.0)
    assert stack.layers["SUPERCONDUCTOR"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.materials["qpdk-silicon"] == {
        "permittivity": pytest.approx(11.45),
        "loss_tangent": pytest.approx(2.7e-6),
    }
    assert stack.materials["vacuum"] == {"permittivity": 1.0}

    flip_stack = flip_chip_stack(substrate_thickness=200.0, bump_thickness=10.0)
    assert flip_stack.layers["SUBSTRATE"].zmin == pytest.approx(-200.0)
    assert flip_stack.layers["VACUUM"].zmax == pytest.approx(10.0)
    assert flip_stack.layers["BUMP"].thickness == pytest.approx(10.0)
    assert flip_stack.layers["M2"].zmin == pytest.approx(10.0)
    assert flip_stack.layers["SUBSTRATE_TOP"].zmax == pytest.approx(210.0)
    assert flip_stack.layers["SUBSTRATE_TOP"].material == "qpdk-silicon"
    assert flip_stack.materials["qpdk-indium"]["conductivity"] == pytest.approx(1.16e7)


def test_to_fem_regions_requires_sim_area():
    """A component without SIM_AREA must fail loudly, not model empty space."""
    with pytest.raises(ValueError, match="SIM_AREA"):
        to_fem_regions(coupler_straight(gap=16.0, length=100.0))


def test_to_fem_regions_rejects_fully_etched_area():
    component = gf.Component()
    for layer in (LAYER.SIM_AREA, LAYER.M1_ETCH):
        component.kdb_cell.shapes(layer).insert(kdb.DBox(0, 0, 100, 100))

    with pytest.raises(ValueError, match="fully etched away"):
        to_fem_regions(component)


def test_fem_layers_do_not_collide_with_mask_layers():
    """FEM regions must not land on real mask layers (M1_DRAW is (1,0) ...)."""
    mask_layers = {tuple(layer) for layer in LAYER}
    assert mask_layers, "no mask layers found; the check would be vacuous"
    for name, layer in {**FEM_LAYERS, **FLIP_CHIP_FEM_LAYERS, **TSV_FEM_LAYERS}.items():
        assert layer not in mask_layers, f"{name} collides with a mask layer"


def test_to_flip_chip_regions_topology():
    c = to_flip_chip_regions(_flip_chip_layout())

    assert sorted(p.name for p in c.ports) == [
        "center",
        "inner_ring_near_junction",
        "junction",
        "outer_ring_near_junction",
        "outer_ring_outside",
    ]
    layout = c.kdb_cell.layout()
    regions = {
        name: kdb.Region(
            c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS[name]))
        ).merged()
        for name in FLIP_CHIP_FEM_LAYERS
    }
    # Both chips follow the subtractive convention inside their etched
    # bounding circles, so each metal splits into isolated islands plus
    # the surrounding ground plane.
    assert len(regions["M1"]) == 3  # ground plane, outer ring, inner circle
    assert len(regions["M2"]) == 2  # ground plane, top circle

    def islands(region: kdb.Region) -> kdb.Region:
        """Return the isolated islands (everything but the ground frame)."""
        ground = max(region.each(), key=lambda poly: poly.area())
        return region - kdb.Region(ground)

    # The center bump must bridge the two isolated islands (the inner
    # circle on M1 and the top circle on M2), not merely land anywhere on
    # the near-ubiquitous ground planes.
    center_bump = kdb.Region(kdb.DBox(-8, -8, 8, 8).to_itype(c.kcl.dbu))
    assert not (center_bump & islands(regions["M1"])).is_empty()
    assert not (center_bump & islands(regions["M2"])).is_empty()
    # Every bump must land on conductor of both chips to actually connect.
    assert (regions["BUMP"] - regions["M1"]).is_empty()
    assert (regions["BUMP"] - regions["M2"]).is_empty()


def test_to_flip_chip_regions_requires_sim_area():
    with pytest.raises(ValueError, match="no SIM_AREA"):
        to_flip_chip_regions(flipmon_with_bbox())


def test_to_flip_chip_regions_requires_both_metal_levels():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(LAYER.M1_ETCH).insert(kdb.DBox(40, 40, 60, 60))
    with pytest.raises(ValueError, match="metal level M2"):
        to_flip_chip_regions(component)


def test_to_flip_chip_regions_requires_bumps():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    for layer in (LAYER.M1_ETCH, LAYER.M2_ETCH):
        component.kdb_cell.shapes(layer).insert(kdb.DBox(40, 40, 60, 60))
    with pytest.raises(ValueError, match="no indium bumps"):
        to_flip_chip_regions(component)


def test_to_flip_chip_regions_bump_off_metal_detected():
    """A bump in the etched lead gap is visible in the converted regions.

    The topology test asserts every bump lands on conductor of both chips;
    this is the negative case proving that assertion has teeth: a bump
    placed in the etched junction gap (no bottom-chip metal under it — the
    M2 top circle above does not help) shows up outside the M1 conductor.
    """

    @gf.cell
    def _bad_bump_layout() -> gf.Component:
        c = gf.Component()
        ref = c << flipmon_with_bbox()
        c.add_ports(ref.ports)
        # In the 12 um etched lead gap between the inner circle and the ring.
        c.kdb_cell.shapes(LAYER.IND).insert(_bump(66.0, 0.0))
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(10, 10))
        return c

    c = to_flip_chip_regions(_bad_bump_layout())
    layout = c.kdb_cell.layout()
    bumps = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS["BUMP"]))
    ).merged()
    m1 = kdb.Region(
        c.kdb_cell.begin_shapes_rec(layout.layer(*FLIP_CHIP_FEM_LAYERS["M1"]))
    ).merged()
    assert not (bumps - m1).is_empty(), "off-metal bump not detected"


def test_flip_chip_stack_matches_conventions():
    _require_gsim()

    stack = flip_chip_stack(substrate_thickness=200.0, bump_thickness=10.0)
    assert stack.layers["SUBSTRATE"].zmin == pytest.approx(-200.0)
    assert stack.layers["M1"].zmax == pytest.approx(0.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(10.0)
    assert stack.layers["M2"].zmin == pytest.approx(10.0)
    assert stack.layers["SUBSTRATE_TOP"].zmax == pytest.approx(210.0)
    assert stack.layers["BUMP"].layer_type == "via"
    # The bump must be a real 3-D volume spanning the gap: a zero-height via
    # would degrade to a 2-D PEC sheet and connect nothing.
    assert stack.layers["BUMP"].zmin == pytest.approx(0.0)
    assert stack.layers["BUMP"].zmax == pytest.approx(10.0)
    assert stack.layers["BUMP"].thickness > 0
    # Every layer's material must resolve in the stack's own materials dict;
    # gsim has no builtin fallback and silently emits Permittivity 1.0 for
    # unknown materials (this caught the top substrate modeled as vacuum).
    stack.validate_stack()
    # Without a conductivity gsim demotes each bump to a 2-D PEC sheet at
    # its base, so the two chips would no longer connect.
    assert stack.materials["qpdk-indium"]["conductivity"] > 0.0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)


def test_slurm_cluster_ray_script_requests_the_allocation():
    cluster = SlurmCluster(
        partition="batch-a,batch-b",
        nodes=3,
        cores_per_node=32,
        solver_cores=8,
        mem_per_node="96G",
        account="proj",
    )
    script = cluster.sbatch_script("python driver.py")

    assert "#SBATCH --partition=batch-a,batch-b" in script
    assert "#SBATCH --account=proj" in script
    assert "#SBATCH --nodes=3" in script
    # One task per node: that is what starts one Ray daemon per node.
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --cpus-per-task=32" in script
    # Slurm sites default to a few hundred MB per core, so memory is required.
    assert "#SBATCH --mem=96G" in script


def test_slurm_cluster_script_starts_head_then_workers():
    cluster = SlurmCluster(nodes=2, cores_per_node=16, solver_cores=4)
    script = cluster.sbatch_script("python driver.py")

    assert script.startswith("#!/bin/bash")
    # The head must come up before the workers can join it.
    assert (
        script.index("ray start --head")
        < script.index("ray status > /dev/null")
        < script.index("ray start --address")
    )
    assert f"${RAY_PORT}" not in script  # formatted, not left as a template
    assert f":{RAY_PORT}" in script
    assert script.index("ray start --address") < script.index("ray.nodes()")
    assert script.index("ray.nodes()") < script.index("python driver.py")
    assert "Ray workers did not join the allocation" in script
    # A process pool that inherits BLAS threads would thrash a shared node.
    assert "OMP_NUM_THREADS=1" in script
    # Slurm runs a spool copy, so the submit directory has to be resolved.
    assert "SLURM_SUBMIT_DIR" in script


def test_slurm_cluster_single_node_has_no_worker_step():
    script = SlurmCluster(nodes=1).sbatch_script("true")
    assert "ray start --head" in script
    assert "ray start --address" not in script
    assert "ray.nodes()) >= 1" in script


def test_slurm_cluster_rejects_unschedulable_request():
    with pytest.raises(ValueError, match="no trial could ever run"):
        SlurmCluster(cores_per_node=4, solver_cores=8)


def test_slurm_log_path_is_absolute_and_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cluster = SlurmCluster(log_dir="logs")
    script = cluster.write_task_script(tmp_path / "jobs" / "trial.sbatch", "true")

    assert str(tmp_path / "logs") in script.read_text()
    assert (tmp_path / "logs").is_dir()


def test_domain_loss_tangents_follow_attribute_order(tmp_path):
    """Gsim emits materials out of attribute order; the lookup must sort."""
    # Materials deliberately listed with the silicon first, the way the mesh
    # generator produced them, against attributes 2 and 1 respectively.
    (tmp_path / "config.json").write_text(
        json.dumps({
            "Domains": {
                "Materials": [
                    {"Attributes": [2], "Permittivity": 11.45, "LossTan": 2.7e-6},
                    {"Attributes": [1], "Permittivity": 1.0, "LossTan": 0.0},
                ],
                "Postprocessing": {"Energy": [], "Probe": []},
            }
        })
    )

    assert add_domain_energy_postprocessing(tmp_path) == [1, 2]
    # Index 1 is the vacuum (attribute 1), index 2 the lossy silicon.
    assert domain_loss_tangents(tmp_path) == {1: 0.0, 2: 2.7e-6}

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["Domains"]["Postprocessing"]["Energy"] == [
        {"Attributes": [1], "Index": 1},
        {"Attributes": [2], "Index": 2},
    ]


def test_slurm_task_script_solves_one_trial():
    cluster = SlurmCluster(
        partition="batch-a,batch-b",
        cores_per_node=40,
        solver_cores=8,
        mem_per_task="24G",
        job_name="qpdk-trial",
    )
    script = cluster.task_sbatch_script('python -m qpdk.simulation.trial "$1"')

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=8" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --mem=24G" in script
    assert "QPDK_MPI_NODES=1" in script
    # One job per trial, so no array indexing and a plain per-job log name.
    assert "--array" not in script
    assert "%j.out" in script
    # Slurm forwards the arguments after the script name, so one script serves
    # every trial and only its parameter file differs.
    assert '"$1"' in script
    assert "OMP_NUM_THREADS=1" in script


def test_slurm_task_script_splits_a_trial_across_nodes():
    cluster = SlurmCluster(solver_cores=8, trial_nodes=2, mem_per_task="32G")
    script = cluster.task_sbatch_script("true")

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --mem=32G" in script
    assert "QPDK_MPI_NODES=2" in script
    larger_trial = SlurmCluster(cores_per_node=8, solver_cores=24, trial_nodes=3)
    assert "#SBATCH --ntasks-per-node=8" in larger_trial.task_sbatch_script("true")
    with pytest.raises(ValueError, match="Ray keeps a trial on one node"):
        larger_trial.sbatch_script("true")
    with pytest.raises(ValueError, match="divisible"):
        SlurmCluster(solver_cores=8, trial_nodes=3)


def test_driver_script_is_one_core_and_outlives_the_study():
    cluster = SlurmCluster(solver_cores=8, driver_time_limit="12:00:00")
    script = cluster.driver_sbatch_script("python driver.py")

    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --mem=2G" in script
    # The driver is not an array, and its log is not per-task.
    assert "--array" not in script
    assert "%A" not in script
    assert "python driver.py" in script


def test_slurm_scripts_survive_a_missing_submit_dir():
    """Slurm does not set SLURM_SUBMIT_DIR for jobs submitted from a compute node.

    A plain `cd "$SLURM_SUBMIT_DIR"` then quietly leaves the job in $HOME, so
    every script falls back to the working directory instead.
    """
    cluster = SlurmCluster(nodes=2, cores_per_node=16, solver_cores=4)

    for script in (
        cluster.task_sbatch_script("true"),
        cluster.driver_sbatch_script("true"),
        cluster.sbatch_script("true"),
    ):
        assert 'cd "${SLURM_SUBMIT_DIR:-$PWD}"' in script


def test_slurm_submit_reports_the_job_id(monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = "Submitted batch job 20386936\n"
        stderr = ""

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr(cluster_module.subprocess, "run", fake_run)
    assert cluster_module.submit("run.sbatch") == "20386936"
    assert captured["command"] == ["sbatch", "run.sbatch"]


def test_slurm_submit_raises_on_failure(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "Invalid account"

    monkeypatch.setattr(cluster_module.subprocess, "run", lambda *_, **__: Result())
    with pytest.raises(SlurmJobError, match="Invalid account"):
        cluster_module.submit("run.sbatch")


@pytest.mark.parametrize("stdout", ["", "Submitted batch job", "warning: retry later"])
def test_slurm_submit_rejects_missing_job_id(monkeypatch, stdout):
    class Result:
        returncode = 0
        stderr = ""

    Result.stdout = stdout
    monkeypatch.setattr(cluster_module.subprocess, "run", lambda *_, **__: Result())
    with pytest.raises(SlurmJobError, match="no job id"):
        cluster_module.submit("run.sbatch")


def test_slurm_submit_uses_numeric_id_for_federated_job(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Submitted batch job 12345;cluster\n"
        stderr = ""

    monkeypatch.setattr(cluster_module.subprocess, "run", lambda *_, **__: Result())
    assert cluster_module.submit("run.sbatch") == "12345"


def test_slurm_wait_ignores_a_single_queue_miss(monkeypatch):
    """A just-submitted job can be briefly absent from squeue."""
    states = iter(["RUNNING", None, "RUNNING", None, None])
    monkeypatch.setattr(cluster_module, "job_state", lambda _job_id: next(states))
    monkeypatch.setattr(cluster_module.time, "sleep", lambda *_: None)

    # Three real states means it must not have returned on the first miss.
    cluster_module.wait("1", poll_seconds=0)
    assert next(states, "exhausted") == "exhausted"


def test_slurm_wait_resets_completion_after_query_failure(monkeypatch):
    states = iter([None, SlurmJobError("outage"), None, "RUNNING", None, None])

    def job_state(_job_id):
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return state

    monkeypatch.setattr(cluster_module, "job_state", job_state)
    monkeypatch.setattr(cluster_module.time, "sleep", lambda *_: None)
    cluster_module.wait("1", poll_seconds=0)
    assert next(states, "exhausted") == "exhausted"


def test_slurm_job_state_distinguishes_controller_failure_from_completed_job(
    monkeypatch,
):
    monkeypatch.setattr(
        cluster_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stderr="controller unavailable", stdout=""
        ),
    )
    with pytest.raises(SlurmJobError, match="controller unavailable"):
        cluster_module.job_state("42")

    monkeypatch.setattr(
        cluster_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )
    assert cluster_module.job_state("42") is None


def test_slurm_wait_stops_after_repeated_controller_failures(monkeypatch):
    def failed_query(_job_id):
        raise SlurmJobError("controller unavailable")

    monkeypatch.setattr(cluster_module, "job_state", failed_query)
    monkeypatch.setattr(cluster_module.time, "sleep", lambda *_: None)
    with pytest.raises(SlurmJobError, match="squeue failed 3 times"):
        cluster_module.wait("42", poll_seconds=0, max_query_failures=2)


def test_trial_result_path_is_beside_the_parameters(tmp_path):
    assert (
        result_path(tmp_path / "trial_0007.json") == tmp_path / "trial_0007.result.json"
    )


@pytest.fixture(scope="module")
def transmon_notebook():
    _require_gsim()
    source_path = (
        Path(__file__).parents[1] / "notebooks/src/palace_batched_qubit_optimization.py"
    )
    setup = source_path.read_text().split("# %%\ndemo_layout =", 1)[0]
    namespace = {"__name__": "__main__"}
    exec(compile(setup, str(source_path), "exec"), namespace)  # ruff: ignore[exec-builtin]
    return namespace


@pytest.mark.parametrize("fundamental_pj", [0.3803881, 0.09766947])
def test_evaluate_simulation_picks_the_fundamental_not_the_largest(
    tmp_path, fundamental_pj: float, transmon_notebook
):
    """A higher-order mode can carry marginally more junction energy.

    On a real sweep that near-tie returned harmonics, while another physical
    fundamental had only 0.098 junction participation.
    """
    palace = tmp_path / "output" / "palace"
    palace.mkdir(parents=True)
    (tmp_path / "config.json").write_text(
        json.dumps({
            "Domains": {
                "Materials": [
                    {"Attributes": [1], "LossTan": 0.0},
                    {"Attributes": [2], "LossTan": 2.7e-6},
                ]
            }
        })
    )
    (palace / "eig.csv").write_text(
        "        m, Re{f} (GHz), Im{f} (GHz), Q\n"
        " 1.00e+00, +2.868516e+00, +3.5e-06, +4.03e+05\n"
        " 2.00e+00, +1.391516e+01, +1.7e-05, +4.01e+05\n"
        " 3.00e+00, +2.198966e+01, +2.8e-05, +3.92e+05\n"
    )
    (palace / "port-EPR.csv").write_text(
        "        m, p[1]\n"
        f" 1.00e+00, -{fundamental_pj:.7e}\n"
        " 2.00e+00, +7.385873e-03\n"
        " 3.00e+00, +3.936045e-01\n"
    )
    (palace / "domain-E.csv").write_text(
        "        m, p_elec[1], p_elec[2]\n"
        " 1.00e+00, +0.08, +0.92\n"
        " 2.00e+00, +0.08, +0.92\n"
        " 3.00e+00, +0.08, +0.92\n"
    )
    (palace / "surface-Q.csv").write_text(
        "m, p_surf[1], Q_surf[1], p_surf[2], Q_surf[2], p_surf[3], Q_surf[3]\n"
        "1, 1e-4, 1e7, 9e-5, 1.1e7, 7e-7, 1.4e9\n"
        "2, 2e-4, 5e6, 8e-5, 1.2e7, 6e-7, 1.6e9\n"
        "3, 3e-4, 3e6, 7e-5, 1.4e7, 5e-7, 2e9\n"
    )

    evaluate_simulation = transmon_notebook["evaluate_simulation"]
    row = evaluate_simulation(tmp_path)

    assert row["mode_index"] == 1
    assert row["f_linear"] == pytest.approx(2.868516e9, rel=1e-9)
    assert row["quality_factor"] == pytest.approx(1 / (0.92 * 2.7e-6))
    assert row["eigenmode_quality_factor"] == pytest.approx(4.03e5)
    assert row["T1"] == pytest.approx(
        row["quality_factor"] / (2 * math.pi * row["f_linear"])
    )
    assert row["participation_sa"] == pytest.approx(1e-4)
    assert row["participation_ms"] == pytest.approx(9e-5)
    assert row["participation_ma"] == pytest.approx(7e-7)

    (palace / "domain-E.csv").write_text("m, p_elec[1]\n1, +0.08\n2, +0.08\n3, +0.08\n")
    with pytest.raises(ValueError, match="missing a configured dielectric domain"):
        evaluate_simulation(tmp_path)


def test_evaluate_simulation_skips_modes_without_junction_energy(
    tmp_path, transmon_notebook
):
    """A packaging mode below the qubit mode is not the qubit mode."""
    palace = tmp_path / "output" / "palace"
    palace.mkdir(parents=True)
    (tmp_path / "config.json").write_text(
        json.dumps({
            "Domains": {
                "Materials": [
                    {"Attributes": [1], "LossTan": 2.7e-6},
                    {"Attributes": [2], "LossTan": 0.0},
                ]
            }
        })
    )
    (palace / "eig.csv").write_text(
        "        m, Re{f} (GHz), Im{f} (GHz), Q\n"
        " 1.00e+00, +1.500000e+00, +1.0e-06, +4.00e+05\n"
        " 2.00e+00, +4.500000e+00, +1.0e-05, +4.00e+05\n"
    )
    (palace / "port-EPR.csv").write_text(
        "        m, p[1]\n 1.00e+00, +4.0e-06\n 2.00e+00, +6.0e-01\n"
    )
    (palace / "domain-E.csv").write_text(
        "        m, p_elec[1], p_elec[2]\n"
        " 1.00e+00, +0.08, +0.92\n 2.00e+00, +0.08, +0.92\n"
    )

    row = transmon_notebook["evaluate_simulation"](tmp_path)
    assert row["mode_index"] == 2
    assert row["quality_factor"] == pytest.approx(1 / (0.08 * 2.7e-6))


def _write_config(tmp_path, boundaries):
    """Write a minimal Palace config carrying only the boundaries under test."""
    (tmp_path / "config.json").write_text(json.dumps({"Boundaries": boundaries}))
    return tmp_path


def test_verify_port_connectivity_reports_a_consumed_port(tmp_path):
    """A port whose surface the boolean pipeline ate leaves no LumpedPort entry.

    That is the exact failure this check exists for, so it must be the
    documented ValueError rather than an IndexError off an empty list.
    """
    sim_dir = _write_config(tmp_path, {"PEC": {"Attributes": [3]}, "LumpedPort": []})
    with pytest.raises(ValueError, match="no lumped port"):
        verify_port_connectivity(sim_dir)


def test_verify_port_connectivity_reports_missing_conductors(tmp_path):
    """PEC is omitted entirely when nothing reduces to a planar conductor."""
    sim_dir = _write_config(tmp_path, {"LumpedPort": [{"Index": 1}]})
    with pytest.raises(ValueError, match="no PEC conductor"):
        verify_port_connectivity(sim_dir)


def test_palace_command_defaults_to_mpirun(monkeypatch):
    """Without a container image the solver comes off PATH under mpirun."""
    monkeypatch.delenv("QPDK_PALACE_SIF", raising=False)
    assert palace_command(ranks=4) == ["mpirun", "-np", "4", "palace", "config.json"]


def test_palace_command_wraps_an_apptainer_image(monkeypatch):
    """A compute node runs the solver from a container, with its own MPI."""
    monkeypatch.setenv("QPDK_PALACE_SIF", "/images/palace.sif")
    command = palace_command(ranks=8)

    assert command[:4] == ["apptainer", "exec", "--cleanenv", "/images/palace.sif"]
    # The ranks still go to mpirun inside the image, not to apptainer.
    assert command[4:] == [
        "mpirun",
        "-np",
        "8",
        "palace-x86_64.bin",
        "config.json",
    ]


def test_slurm_mpi_launcher_maps_container_ranks_across_nodes(tmp_path, monkeypatch):
    monkeypatch.setenv("QPDK_MPI_NODES", "2")
    monkeypatch.setenv("QPDK_PALACE_SIF", "/images/palace.sif")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "csl[1-2]")
    monkeypatch.setattr(
        "qpdk.simulation.palace_run.subprocess.check_output",
        lambda _command, **_kwargs: "csl1\ncsl2\n",
    )

    options = _slurm_mpi_launcher(tmp_path, 8)
    command = palace_command(ranks=8, launcher_args=options)

    assert (tmp_path / "palace.hosts").read_text() == ("csl1 slots=4\ncsl2 slots=4\n")
    assert "ppr:4:node" in command
    assert str(tmp_path / "palace-mpi-agent") in command
    assert (
        "apptainer exec --cleanenv $quoted_sif /bin/sh -c $quoted_cmd"
        in (tmp_path / "palace-mpi-agent").read_text()
    )


def _two_port_mesh(tmp_path, *, metal_name="SUPERCONDUCTOR_pec"):
    """Write a mesh with one port touching metal and one isolated from it.

    Three coplanar rectangles: the conductor, a port sharing an edge with it
    (so the two surfaces share mesh nodes), and a port off on its own.

    Returns:
        The directory the mesh was written to.
    """
    try:
        gmsh = importlib.import_module("gmsh")
    except (ImportError, OSError):
        return pytest.skip("gmsh unavailable (missing package or GL system libraries)")
    gmsh.initialize()
    try:
        geo = gmsh.model.geo
        # Conductor (0,0)-(1,1) and an attached port (1,0)-(2,1) reusing the
        # shared edge, so the meshes are conformal across it.
        pts = {
            n: geo.addPoint(*xy, 0)
            for n, xy in {
                "a": (0, 0),
                "b": (1, 0),
                "c": (1, 1),
                "d": (0, 1),
                "e": (2, 0),
                "f": (2, 1),
                "g": (5, 0),
                "h": (6, 0),
                "i": (6, 1),
                "j": (5, 1),
            }.items()
        }
        shared = geo.addLine(pts["b"], pts["c"])
        metal = geo.addPlaneSurface([
            geo.addCurveLoop([
                geo.addLine(pts["a"], pts["b"]),
                shared,
                geo.addLine(pts["c"], pts["d"]),
                geo.addLine(pts["d"], pts["a"]),
            ])
        ])
        attached = geo.addPlaneSurface([
            geo.addCurveLoop([
                geo.addLine(pts["b"], pts["e"]),
                geo.addLine(pts["e"], pts["f"]),
                geo.addLine(pts["f"], pts["c"]),
                -shared,
            ])
        ])
        orphaned = geo.addPlaneSurface([
            geo.addCurveLoop([
                geo.addLine(pts["g"], pts["h"]),
                geo.addLine(pts["h"], pts["i"]),
                geo.addLine(pts["i"], pts["j"]),
                geo.addLine(pts["j"], pts["g"]),
            ])
        ])
        geo.synchronize()
        for surface, tag, name in (
            (metal, 3, metal_name),
            (attached, 1, "P1"),
            (orphaned, 2, "P2"),
        ):
            gmsh.model.addPhysicalGroup(2, [surface], tag)
            gmsh.model.setPhysicalName(2, tag, name)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(tmp_path / "palace.msh"))
    finally:
        gmsh.finalize()
    return tmp_path


def test_verify_port_connectivity_accepts_an_attached_port(tmp_path):
    """A port sharing an edge with the conductor shares mesh nodes with it."""
    sim_dir = _two_port_mesh(tmp_path)
    _write_config(sim_dir, {"PEC": {"Attributes": [3]}, "LumpedPort": [{"Index": 1}]})

    assert verify_port_connectivity(sim_dir)["P1"] > 0


@pytest.mark.parametrize(
    ("metal_name", "pec_attributes"),
    [("SUPERCONDUCTOR", [3]), ("SUPERCONDUCTOR_pec", [999])],
)
def test_verify_port_connectivity_finds_conductor_by_tag_or_name(
    tmp_path, metal_name, pec_attributes
):
    sim_dir = _two_port_mesh(tmp_path, metal_name=metal_name)
    _write_config(
        sim_dir,
        {"PEC": {"Attributes": pec_attributes}, "LumpedPort": [{"Index": 1}]},
    )

    assert verify_port_connectivity(sim_dir)["P1"] > 0


def test_verify_port_connectivity_rejects_an_orphaned_port(tmp_path):
    """The failure the check exists for: a port with no metal under it."""
    sim_dir = _two_port_mesh(tmp_path)
    _write_config(sim_dir, {"PEC": {"Attributes": [3]}, "LumpedPort": [{"Index": 2}]})

    with pytest.raises(ValueError, match="share no mesh nodes"):
        verify_port_connectivity(sim_dir)


def test_transmon_worker_builds_a_connected_palace_mesh(tmp_path, transmon_notebook):
    _require_gsim()
    evaluator = cloudpickle.loads(
        cloudpickle.dumps(transmon_notebook["evaluate_layout"])
    )
    build_simulation = evaluator.__globals__["build_simulation"]
    layout = evaluator.__globals__["Layout"]

    sim_dir = build_simulation(
        layout(pad_width=20, pad_height=40, pad_gap=10),
        tmp_path / "trial",
        num_modes=1,
        substrate_thickness=20,
        vacuum_thickness=20,
        lateral_margin=20,
        refined_mesh_size=1.0,
        linear_max_its=3,
    )
    config = json.loads((sim_dir / "config.json").read_text())
    assert (sim_dir / "palace.msh").stat().st_size > 0
    assert config["Solver"]["Eigenmode"]["N"] == 1
    assert config["Solver"]["Linear"]["MaxIts"] == 3
    assert config["Domains"]["Postprocessing"]["Energy"]


class _StubRunner:
    """Finishes a trial after a fixed number of polls, newest last."""

    def __init__(self, polls_needed=1, fail=()):
        self.polls_needed = polls_needed
        self.fail = set(fail)
        self.started = []
        self.polls = {}
        self.peak_in_flight = 0
        self.live = 0

    def start(self, name, params):  # ruff: ignore[unused-method-argument]
        self.started.append(name)
        self.polls[name] = 0
        self.live += 1
        self.peak_in_flight = max(self.peak_in_flight, self.live)
        return name

    def collect(self, handle):
        self.polls[handle] += 1
        if self.polls[handle] < self.polls_needed:
            return None
        self.live -= 1
        if handle in self.fail:
            return {"error": "stub failure"}
        return {"f01": 4.5e9, "T1": 1e-5, "footprint_mm2": 0.2}


def _study():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(directions=["minimize", "maximize", "minimize"])


def _suggest(trial):
    trial.suggest_float("pad_width", 100.0, 400.0)


def _objectives(row):
    return [abs(row["f01"] - 4.5e9) / 4.5e9, row["T1"], row["footprint_mm2"]]


def test_run_study_keeps_trials_in_flight_without_batch_barriers(monkeypatch):
    """Concurrency is a standing limit, not a wave that must drain first."""
    monkeypatch.setattr(study_module.time, "sleep", lambda *_: None)
    runner = _StubRunner(polls_needed=3)

    rows = study_module.run_study(
        _study(),
        runner,
        suggest=_suggest,
        objectives=_objectives,
        n_trials=10,
        max_in_flight=4,
    )

    assert len(rows) == 10
    # Never exceeds the cap, and actually uses it rather than draining first.
    assert runner.peak_in_flight == 4
    assert len(runner.started) == 10


def test_run_study_prunes_failures_and_keeps_going(monkeypatch):
    """A failed trial is pruned, not fatal, and its slot is refilled."""
    monkeypatch.setattr(study_module.time, "sleep", lambda *_: None)
    runner = _StubRunner(fail={"trial_0000", "trial_0002"})

    rows = study_module.run_study(
        _study(),
        runner,
        suggest=_suggest,
        objectives=_objectives,
        n_trials=6,
        max_in_flight=2,
    )

    assert len(rows) == 4
    assert len(runner.started) == 6


def test_run_study_raises_when_every_trial_fails(monkeypatch):
    """Nothing to analyse is an error, not an empty table."""
    monkeypatch.setattr(study_module.time, "sleep", lambda *_: None)
    runner = _StubRunner(fail={f"trial_{i:04d}" for i in range(4)})

    with pytest.raises(RuntimeError, match="all 4 trials failed"):
        study_module.run_study(
            _study(),
            runner,
            suggest=_suggest,
            objectives=_objectives,
            n_trials=4,
            max_in_flight=2,
        )


def test_slurm_runner_submits_one_job_per_trial(tmp_path, monkeypatch):
    """Each trial is its own submission of one shared script.

    Slurm forwards the arguments after the script name, so the script is
    written once and only the parameter file differs between submissions.
    """
    submitted = []
    monkeypatch.setattr(
        study_module,
        "submit",
        lambda script, *args: submitted.append((script, args)) or str(len(submitted)),
    )
    runner = study_module.SlurmRunner(
        cluster=SlurmCluster(log_dir=str(tmp_path / "logs")),
        run_root=tmp_path,
        ranks=8,
        evaluator="mysweep:evaluate_layout",
    )

    runner.start("trial_0000", {"pad_width": 200.0})
    runner.start("trial_0001", {"pad_width": 300.0})

    # One script, reused; two submissions, each with its own parameter file.
    assert len({script for script, _ in submitted}) == 1
    assert [Path(args[0]).name for _, args in submitted] == [
        "trial_0000.json",
        "trial_0001.json",
    ]
    assert json.loads((tmp_path / "params" / "trial_0000.json").read_text()) == {
        "pad_width": 200.0
    }


def test_slurm_runner_sends_notebook_evaluator_to_worker(tmp_path, monkeypatch):
    offset = 7.0

    def evaluate(params, run_root, *, ranks, sim_dir, timeout):
        assert run_root == tmp_path
        assert sim_dir == tmp_path / "params" / "trial_0000"
        assert timeout > 0
        return {"score": params["pad_width"] + offset, "ranks": ranks}

    monkeypatch.setattr(study_module, "submit", lambda *_a, **_k: "42")
    runner = study_module.SlurmRunner(
        cluster=SlurmCluster(log_dir=str(tmp_path / "logs")),
        run_root=tmp_path,
        ranks=4,
        evaluator=evaluate,
    )
    _, params_path = runner.start("trial_0000", {"pad_width": 200.0})
    evaluator_path = tmp_path / "evaluator.pkl"

    assert evaluator_path.exists()
    assert "--evaluator-file" in (tmp_path / "trial.sbatch").read_text()
    assert (
        run_trial([
            str(params_path),
            "--run-root",
            str(tmp_path),
            "--ranks",
            "4",
            "--evaluator-file",
            str(evaluator_path),
        ])
        == 0
    )
    assert json.loads(result_path(params_path).read_text()) == {
        "score": 207.0,
        "ranks": 4,
    }


def test_slurm_runner_reports_a_job_that_vanished(tmp_path, monkeypatch):
    """Gone from the queue with no result is a failure, not "still running"."""
    monkeypatch.setattr(study_module, "submit", lambda *_a, **_k: "424242")
    monkeypatch.setattr(study_module, "job_state", lambda _job_id: None)
    runner = study_module.SlurmRunner(
        cluster=SlurmCluster(log_dir=str(tmp_path / "logs")),
        run_root=tmp_path,
        ranks=1,
        evaluator="mysweep:evaluate_layout",
    )

    handle = runner.start("trial_0000", {"pad_width": 200.0})
    row = runner.collect(handle)

    assert row is not None
    assert "left the queue" in row["error"]


def test_ray_runner_requires_an_allocation(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    runner = study_module.RayRunner(tmp_path, ranks=2, evaluator=lambda *_a: {})

    with pytest.raises(RuntimeError, match="RAY_ADDRESS is not set"):
        runner.start("trial_0000", {"pad_width": 200.0})


def test_ray_runner_dispatches_and_collects_trial_results(tmp_path, monkeypatch):
    ray = ModuleType("ray")
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RAY_ADDRESS", "ray://head")
    launched = []
    monkeypatch.setattr(
        ray, "init", lambda **kwargs: launched.append(kwargs), raising=False
    )

    def remote(*, num_cpus):
        assert num_cpus == 4
        return lambda evaluate: SimpleNamespace(remote=evaluate)

    monkeypatch.setattr(ray, "remote", remote, raising=False)

    def evaluate(params, run_root, *, ranks, sim_dir):
        assert run_root == tmp_path.resolve()
        assert ranks == 4
        assert sim_dir == tmp_path / "trial_0000"
        return {"score": params["pad_width"]}

    runner = study_module.RayRunner(tmp_path, ranks=4, evaluator=evaluate)
    handle = runner.start("trial_0000", {"pad_width": 200.0})
    assert launched == [{"address": "ray://head", "ignore_reinit_error": True}]

    ready = False

    def wait(refs, *, timeout):
        assert timeout == 0
        return (refs, []) if ready else ([], refs)

    monkeypatch.setattr(ray, "wait", wait, raising=False)
    assert runner.collect(handle) is None
    ready = True
    monkeypatch.setattr(ray, "get", lambda ref: ref, raising=False)
    assert runner.collect(handle) == {"score": 200.0}

    def failed_task(_ref):
        raise RuntimeError("worker lost")

    monkeypatch.setattr(ray, "get", failed_task)
    assert runner.collect(handle) == {"error": "RuntimeError: worker lost"}

    evaluator_module = ModuleType("test_ray_evaluator")
    monkeypatch.setattr(evaluator_module, "evaluate", evaluate, raising=False)
    monkeypatch.setitem(sys.modules, evaluator_module.__name__, evaluator_module)
    runner.evaluator = "test_ray_evaluator:evaluate"
    assert runner.start("trial_0000", {"pad_width": 300.0}) == {"score": 300.0}


def test_trial_worker_writes_an_evaluator_failure(tmp_path, monkeypatch):
    evaluator_module = ModuleType("test_failing_evaluator")

    def evaluate(*_args, **_kwargs):
        raise RuntimeError("mesh failed")

    monkeypatch.setattr(evaluator_module, "evaluate", evaluate, raising=False)
    monkeypatch.setitem(sys.modules, evaluator_module.__name__, evaluator_module)
    params_path = tmp_path / "params" / "trial_0000.json"
    params_path.parent.mkdir()
    params_path.write_text(json.dumps({"pad_width": 200.0}))

    assert (
        run_trial([
            str(params_path),
            "--run-root",
            str(tmp_path),
            "--evaluator",
            "test_failing_evaluator:evaluate",
        ])
        == 1
    )
    assert json.loads(result_path(params_path).read_text()) == {
        "error": "RuntimeError: mesh failed"
    }


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        # Eigenmode, S-parameter and capacitance rows share no keys at all, so
        # the trial entry point must not assume any of them.
        ({"f01": 4.83e9, "T1": 1.33e-05}, "f01=4.83e+09  T1=1.33e-05"),
        ({"s11_db": -0.42, "s21_db": -3.1}, "s11_db=-0.42  s21_db=-3.1"),
        ({"c_matrix_ff": 143.2}, "c_matrix_ff=143.2"),
    ],
)
def test_trial_summary_does_not_assume_a_simulation_type(row, expected):
    assert _summarise(row) == expected


def test_trial_summary_survives_a_row_with_no_scalars():
    """Booleans are not results, and a row may carry only structured data."""
    assert "no scalar results" in _summarise({"converged": True, "modes": [1, 2]})


def test_trial_declares_runnable_script_metadata():
    """PEP 723 metadata lets a node with no project environment run it.

    A generated sbatch script may invoke this file directly, so the dependency
    block has to stay valid TOML and keep naming the extra that supplies gsim.
    """
    src = (
        Path(__file__).parent.parent / "qpdk" / "simulation" / "trial.py"
    ).read_text()
    block = re.search(r"^# /// script$(.*?)^# ///$", src, re.MULTILINE | re.DOTALL)
    assert block, "inline script metadata missing"
    body = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in block.group(1).splitlines(keepends=True)
    )
    meta = tomllib.loads(body)

    assert meta["dependencies"] == ["qpdk[models,optimization]"]
    # gsim pins a 3.12-only gdsfactoryplus, so a standalone run must say so.
    assert meta["requires-python"] == ">=3.12,<3.13"


@gf.cell
def _tsv_layout() -> gf.Component:
    c = gf.Component()
    ref = c << tsv_transition_double_sided()
    c.add_ports(ref.ports)
    # Enlarge only across the line: the CPW ends must reach the SIM_AREA edge,
    # otherwise the metal beyond them would short the signal to ground.
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 50))
    return c


def _tsv_regions(c: gf.Component) -> dict[str, kdb.Region]:
    layout = c.kdb_cell.layout()
    return {
        name: kdb.Region(
            c.kdb_cell.begin_shapes_rec(layout.layer(*TSV_FEM_LAYERS[name]))
        ).merged()
        for name in ("M1", "MB", "TSV")
    }


def test_to_tsv_regions_topology():
    c = to_tsv_regions(_tsv_layout())

    assert sorted(p.name for p in c.ports) == ["o1", "o2"]
    regions = _tsv_regions(c)
    # Each face is a ground plane plus the isolated signal pad/taper.
    assert len(regions["M1"]) == 2
    assert len(regions["MB"]) == 2
    # Every TSV must land on conductor of both faces to connect them.
    assert not regions["TSV"].is_empty()
    assert (regions["TSV"] - regions["M1"]).is_empty()
    assert (regions["TSV"] - regions["MB"]).is_empty()


def test_to_tsv_regions_requires_sim_area():
    with pytest.raises(ValueError, match="no SIM_AREA"):
        to_tsv_regions(tsv_transition_double_sided())


def test_to_tsv_regions_requires_backside_metal():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    component.kdb_cell.shapes(LAYER.M1_ETCH).insert(kdb.DBox(40, 40, 60, 60))
    with pytest.raises(ValueError, match="metal level MB"):
        to_tsv_regions(component)


def test_to_tsv_regions_requires_tsvs():
    component = gf.Component()
    component.kdb_cell.shapes(LAYER.SIM_AREA).insert(kdb.DBox(0, 0, 100, 100))
    for layer in (LAYER.M1_ETCH, LAYER.MB_ETCH):
        component.kdb_cell.shapes(layer).insert(kdb.DBox(40, 40, 60, 60))
    with pytest.raises(ValueError, match="no TSVs"):
        to_tsv_regions(component)


def test_tsv_stack_matches_conventions():
    _require_gsim()

    stack = tsv_stack(substrate_thickness=200.0, vacuum_thickness=300.0)
    assert stack.layers["M1"].zmin == pytest.approx(0.0)
    assert stack.layers["MB"].zmin == pytest.approx(-200.0)
    assert stack.layers["SUBSTRATE"].zmin == pytest.approx(-200.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(0.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["VACUUM_BOTTOM"].zmin == pytest.approx(-500.0)
    # The TSV must be a real volume through the whole substrate; a
    # zero-height via would degrade to a PEC sheet and connect nothing.
    assert stack.layers["TSV"].layer_type == "via"
    assert stack.layers["TSV"].zmin == pytest.approx(-200.0)
    assert stack.layers["TSV"].zmax == pytest.approx(0.0)
    stack.validate_stack()
    assert stack.materials["qpdk-tsv-lining"]["conductivity"] > 0.0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)
