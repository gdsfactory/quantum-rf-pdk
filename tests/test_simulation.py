"""Tests for the qpdk.simulation FEM helpers."""

import importlib
import json
import re
import tomllib
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb
import pytest

from qpdk.cells import coupler_straight, flipmon_with_bbox
from qpdk.simulation import (
    FEM_LAYERS,
    FLIP_CHIP_FEM_LAYERS,
    RAY_PORT,
    SlurmCluster,
    SlurmJobError,
    cluster as cluster_module,
    study as study_module,
    to_fem_regions,
    to_flip_chip_regions,
)
from qpdk.simulation.palace_run import (
    add_domain_energy_postprocessing,
    domain_loss_tangents,
    evaluate_simulation,
    palace_command,
    verify_port_connectivity,
)
from qpdk.simulation.trial import _summarise, result_path
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

    # The conductor is the simulation area minus the etch mask, so no point
    # of the input etch mask may be conductor. The etch region must come from
    # the INPUT component: the output never carries M1_ETCH, so reading it
    # from the output would make the assertion vacuous.
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


def _require_gsim() -> None:
    """Skip unless gsim imports; gmsh can raise OSError, not just ImportError."""
    try:
        importlib.import_module("gsim")
    except (ImportError, OSError):
        pytest.skip("gsim unavailable (missing package or GL system libraries)")


def test_single_chip_stack_matches_material_properties():
    _require_gsim()

    from qpdk.simulation import (  # ruff: ignore[import-outside-top-level]
        single_chip_stack,
    )

    stack = single_chip_stack(substrate_thickness=200.0, vacuum_thickness=100.0)
    assert stack.layers["SUBSTRATE"].zmax == pytest.approx(200.0)
    assert stack.layers["VACUUM"].zmax == pytest.approx(300.0)
    assert stack.layers["SUPERCONDUCTOR"].thickness == 0
    assert stack.materials["qpdk-silicon"]["permittivity"] == pytest.approx(11.45)
    assert stack.materials["qpdk-silicon"]["loss_tangent"] == pytest.approx(2.7e-6)


def test_to_fem_regions_requires_sim_area():
    """A component without SIM_AREA must fail loudly, not model empty space."""
    with pytest.raises(ValueError, match="SIM_AREA"):
        to_fem_regions(coupler_straight(gap=16.0, length=100.0))


def test_fem_layers_do_not_collide_with_mask_layers():
    """FEM regions must not land on real mask layers (M1_DRAW is (1,0) ...)."""
    mask_layers = {
        tuple(getattr(LAYER, name))
        for name in dir(LAYER)
        if not name.startswith("_") and isinstance(getattr(LAYER, name), tuple)
    }
    for name, layer in {**FEM_LAYERS, **FLIP_CHIP_FEM_LAYERS}.items():
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

    from qpdk.simulation import (  # ruff: ignore[import-outside-top-level]
        flip_chip_stack,
    )

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
    assert script.index("ray start --head") < script.index("ray start --address")
    assert f"${RAY_PORT}" not in script  # formatted, not left as a template
    assert f":{RAY_PORT}" in script
    assert "python driver.py" in script
    # A process pool that inherits BLAS threads would thrash a shared node.
    assert "OMP_NUM_THREADS=1" in script
    # Slurm runs a spool copy, so the submit directory has to be resolved.
    assert "SLURM_SUBMIT_DIR" in script


def test_slurm_cluster_single_node_has_no_worker_step():
    script = SlurmCluster(nodes=1).sbatch_script("true")
    assert "ray start --head" in script
    assert "ray start --address" not in script


def test_slurm_cluster_rejects_unschedulable_request():
    with pytest.raises(ValueError, match="no trial could ever run"):
        SlurmCluster(cores_per_node=4, solver_cores=8)


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

    # Sized on its own, so it starts on any node with room to spare.
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=24G" in script
    # One job per trial, so no array indexing and a plain per-job log name.
    assert "--array" not in script
    assert "%j.out" in script
    # Slurm forwards the arguments after the script name, so one script serves
    # every trial and only its parameter file differs.
    assert '"$1"' in script
    assert "OMP_NUM_THREADS=1" in script


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


def test_slurm_wait_ignores_a_single_queue_miss(monkeypatch):
    """A just-submitted job can be briefly absent from squeue."""
    states = iter(["RUNNING", None, "RUNNING", None, None])
    monkeypatch.setattr(cluster_module, "job_state", lambda _job_id: next(states))
    monkeypatch.setattr(cluster_module.time, "sleep", lambda *_: None)

    # Three real states means it must not have returned on the first miss.
    cluster_module.wait("1", poll_seconds=0)
    assert next(states, "exhausted") == "exhausted"


def test_trial_result_path_is_beside_the_parameters(tmp_path):
    assert (
        result_path(tmp_path / "trial_0007.json") == tmp_path / "trial_0007.result.json"
    )


def test_evaluate_simulation_picks_the_fundamental_not_the_largest(tmp_path):
    """A higher-order mode can carry marginally more junction energy.

    On a real sweep that near-tie returned harmonic modes at several times the
    qubit frequency, so the rule is the lowest mode the junction takes part in,
    not the largest participation.
    """
    palace = tmp_path / "output" / "palace"
    palace.mkdir(parents=True)
    (palace / "eig.csv").write_text(
        "        m, Re{f} (GHz), Im{f} (GHz), Q\n"
        " 1.00e+00, +2.868516e+00, +3.5e-06, +4.03e+05\n"
        " 2.00e+00, +1.391516e+01, +1.7e-05, +4.01e+05\n"
        " 3.00e+00, +2.198966e+01, +2.8e-05, +3.92e+05\n"
    )
    (palace / "port-EPR.csv").write_text(
        "        m, p[1]\n"
        " 1.00e+00, -3.803881e-01\n"
        " 2.00e+00, +7.385873e-03\n"
        " 3.00e+00, +3.936045e-01\n"
    )
    (palace / "domain-E.csv").write_text(
        "        m, p_elec[1], p_elec[2]\n"
        " 1.00e+00, +0.08, +0.92\n"
        " 2.00e+00, +0.08, +0.92\n"
        " 3.00e+00, +0.08, +0.92\n"
    )

    row = evaluate_simulation(tmp_path)

    assert row["mode_index"] == 1
    assert row["f01"] == pytest.approx(2.868516e9, rel=1e-9)


def test_evaluate_simulation_skips_modes_without_junction_energy(tmp_path):
    """A packaging mode below the qubit mode is not the qubit mode."""
    palace = tmp_path / "output" / "palace"
    palace.mkdir(parents=True)
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

    assert evaluate_simulation(tmp_path)["mode_index"] == 2


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
    assert command[4:] == ["mpirun", "-np", "8", "palace", "config.json"]


def _two_port_mesh(tmp_path):
    """Write a mesh with one port touching metal and one isolated from it.

    Three coplanar rectangles: the conductor, a port sharing an edge with it
    (so the two surfaces share mesh nodes), and a port off on its own.

    Returns:
        The directory the mesh was written to.
    """
    gmsh = pytest.importorskip("gmsh")
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
            (metal, 3, "SUPERCONDUCTOR_pec"),
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


def test_verify_port_connectivity_rejects_an_orphaned_port(tmp_path):
    """The failure the check exists for: a port with no metal under it."""
    sim_dir = _two_port_mesh(tmp_path)
    _write_config(sim_dir, {"PEC": {"Attributes": [3]}, "LumpedPort": [{"Index": 2}]})

    with pytest.raises(ValueError, match="share no mesh nodes"):
        verify_port_connectivity(sim_dir)


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

    assert meta["dependencies"] == ["qpdk[models]"]
    # gsim pins a 3.12-only gdsfactoryplus, so a standalone run must say so.
    assert meta["requires-python"] == ">=3.12,<3.13"
