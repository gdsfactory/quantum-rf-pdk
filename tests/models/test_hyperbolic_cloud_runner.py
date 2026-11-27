"""Cloud batches, lead de-embedding and publishing, with fakes and synthetic data.

No test here submits a job: the backend is a fake, and the default backend that
imports ``gsim.gcloud`` is replaced wherever the runner could construct it.
"""

import importlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qpdk.models.datasets import Dataset, check_maxwell

SCRIPTS = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"
TERMINALS = ("a", "b")


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the runner helper and make every gsim cloud call fail the test.

    The runner needs gsim only to read results. Every test passes a fake
    backend, and these guards catch any path that would reach the real one.

    Returns:
        The imported ``_cloud`` module.
    """
    pytest.importorskip("gsim.palace.capacitance")
    gcloud = pytest.importorskip("gsim.gcloud")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a test reached the real gsim cloud interface")

    for name in (
        "upload",
        "start",
        "get_status",
        "wait_for_results",
        "run_simulation",
        "upload_simulation_dir",
        "check_cache",
        "check_cache_for_dir",
    ):
        monkeypatch.setattr(gcloud, name, forbidden)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("_cloud")


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
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


def write_results(
    directory: Path, matrix: np.ndarray, *, norm: float | None = 1e-3
) -> dict[str, Path]:
    """Write the files a Palace electrostatic download holds."""
    directory.mkdir(parents=True, exist_ok=True)
    n = len(matrix)
    mutual = -matrix.copy()
    np.fill_diagonal(mutual, matrix.sum(axis=1))

    def table(values: np.ndarray) -> str:
        header = "i, " + ", ".join(f"C[{j + 1}] (F)" for j in range(n))
        rows = [
            f"{i + 1}, " + ", ".join(f"{value:.17e}" for value in row)
            for i, row in enumerate(values)
        ]
        return "\n".join([header, *rows]) + "\n"

    files = {
        "terminal-C.csv": table(matrix),
        "terminal-Cm.csv": table(mutual),
    }
    if norm is not None:
        files["error-indicators.csv"] = (
            f"Norm, Minimum, Maximum, Mean\n{norm}, 0.0, 1.0, 0.5\n"
        )
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return {name: directory / name for name in files}


GOOD = np.array([[3.0, -1.0], [-1.0, 3.0]]) * 1e-15


class FakeBackend:
    """Records submissions and collections; never leaves this process."""

    def __init__(
        self,
        *,
        fail: Sequence[str] = (),
        statuses: Mapping[str, str] | None = None,
        matrix: np.ndarray = GOOD,
        norm: float | None = 1e-3,
    ) -> None:
        """Set up empty records and the failure and status scripts."""
        self.submitted: list[str] = []
        self.inflight: set[str] = set()
        self.max_inflight = 0
        self.waits: list[list[str]] = []
        self.fail = set(fail)
        self.statuses = dict(statuses or {})
        self.matrix = matrix
        self.norm = norm

    def submit(self, _prepared: Any, run_dir: Path) -> str:
        job_id = f"fake-{len(self.submitted)}"
        self.submitted.append(run_dir.name)
        self.inflight.add(job_id)
        self.max_inflight = max(self.max_inflight, len(self.inflight))
        assert len(self.inflight) <= 5
        return job_id

    def status(self, job_id: str) -> str:
        return self.statuses.get(job_id, "running")

    def wait(
        self, job_ids: Sequence[str], parent_dir: Path
    ) -> dict[str, Mapping[str, Path] | Exception]:
        self.waits.append(list(job_ids))
        outcome: dict[str, Mapping[str, Path] | Exception] = {}
        for job_id in job_ids:
            self.inflight.discard(job_id)
            outcome[job_id] = (
                RuntimeError("solver crashed")
                if job_id in self.fail
                else write_results(
                    parent_dir / f"sim-data-{job_id}", self.matrix, norm=self.norm
                )
            )
        return outcome


def make_jobs(cloud: ModuleType, count: int, prepared: list[str] | None = None) -> list:
    def job(tag: int) -> Any:
        def prepare(run: Path) -> Any:
            (run / "sim").mkdir(parents=True, exist_ok=True)
            (run / "sim" / "palace.msh").write_text("mesh", encoding="utf-8")
            if prepared is not None:
                prepared.append(run.name)
            return cloud.Prepared(
                sim=None, tetrahedra=1000 + tag, estimated_runtime_s=1.0
            )

        return cloud.Job(
            name=f"job{tag}",
            inputs={"tag": tag},
            terminals=TERMINALS,
            tolerance=1e-9,
            prepare=prepare,
        )

    return [job(tag) for tag in range(count)]


def test_batches_hold_at_most_five_jobs(cloud: ModuleType, tmp_path: Path) -> None:
    backend = FakeBackend()
    jobs = make_jobs(cloud, 12)
    report = cloud.run_batches(jobs, workdir=tmp_path, backend=backend)
    assert [len(batch) for batch in report.batches] == [5, 5, 2]
    assert [len(ids) for ids in backend.waits] == [5, 5, 2]
    assert backend.max_inflight == 5
    assert len(report.completed) == 12
    assert not report.failed
    for job in jobs:
        record = cloud.read_complete(tmp_path, job)
        assert record is not None
        np.testing.assert_allclose(record["maxwell_capacitance"], GOOD, rtol=1e-15)
        assert record["tetrahedra"] == 1000 + job.inputs["tag"]
        assert record["diagnostics"] == {"fem_error_indicator_norm": 1e-3}
        # The mesh is deleted once the result is stored.
        assert not (tmp_path / job.key / "sim" / "palace.msh").exists()
        assert (tmp_path / job.key / "results" / "terminal-C.csv").is_file()


@pytest.mark.parametrize("limit", [0, 6])
def test_concurrency_limit_is_enforced(
    cloud: ModuleType, tmp_path: Path, limit: int
) -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        cloud.run_batches(
            make_jobs(cloud, 1),
            workdir=tmp_path,
            backend=FakeBackend(),
            max_concurrent=limit,
        )


def test_smaller_batches_are_allowed(cloud: ModuleType, tmp_path: Path) -> None:
    backend = FakeBackend()
    cloud.run_batches(
        make_jobs(cloud, 5), workdir=tmp_path, backend=backend, max_concurrent=2
    )
    assert backend.max_inflight == 2
    assert [len(ids) for ids in backend.waits] == [2, 2, 1]


def test_extending_the_grid_does_not_resubmit_finished_jobs(
    cloud: ModuleType, tmp_path: Path
) -> None:
    first = FakeBackend()
    cloud.run_batches(make_jobs(cloud, 3), workdir=tmp_path, backend=first)
    second = FakeBackend()
    prepared: list[str] = []
    report = cloud.run_batches(
        make_jobs(cloud, 6, prepared), workdir=tmp_path, backend=second
    )
    assert len(first.submitted) == 3
    assert len(second.submitted) == 3
    assert len(prepared) == 3  # finished jobs are not even meshed again
    assert sorted(report.reused) == ["job0", "job1", "job2"]
    assert sorted(report.completed) == ["job3", "job4", "job5"]


def test_identical_jobs_are_submitted_once(cloud: ModuleType, tmp_path: Path) -> None:
    jobs = make_jobs(cloud, 2)
    backend = FakeBackend()
    report = cloud.run_batches(
        [jobs[0], jobs[0], jobs[1]], workdir=tmp_path, backend=backend
    )
    assert len(backend.submitted) == 2
    assert len(report.completed) == 2


def test_resume_reattaches_a_submitted_job(cloud: ModuleType, tmp_path: Path) -> None:
    (job,) = make_jobs(cloud, 1)
    run = tmp_path / job.key
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"job_id": "earlier-7"}), encoding="utf-8")
    prepared: list[str] = []
    (job,) = make_jobs(cloud, 1, prepared)
    backend = FakeBackend(statuses={"earlier-7": "running"})
    report = cloud.run_batches([job], workdir=tmp_path, backend=backend)
    assert backend.submitted == []
    assert prepared == []
    assert backend.waits == [["earlier-7"]]
    assert report.reattached == ["job0"]
    record = cloud.read_complete(tmp_path, job)
    assert record is not None
    assert record["job_id"] == "earlier-7"


def test_reattached_jobs_count_against_the_batch_limit(
    cloud: ModuleType, tmp_path: Path
) -> None:
    jobs = make_jobs(cloud, 8)
    statuses = {}
    for number, job in enumerate(jobs[:3]):
        (tmp_path / job.key).mkdir()
        (tmp_path / job.key / "job.json").write_text(
            json.dumps({"job_id": f"old-{number}"}), encoding="utf-8"
        )
        statuses[f"old-{number}"] = "running"
    backend = FakeBackend(statuses=statuses)
    report = cloud.run_batches(jobs, workdir=tmp_path, backend=backend)
    assert [len(batch) for batch in report.batches] == [5, 3]
    assert backend.max_inflight <= 5
    assert len(backend.submitted) == 5


def test_failed_job_leaves_no_result_and_is_not_resubmitted(
    cloud: ModuleType, tmp_path: Path
) -> None:
    jobs = make_jobs(cloud, 3)
    backend = FakeBackend(fail=["fake-1"])
    report = cloud.run_batches(jobs, workdir=tmp_path, backend=backend)
    assert list(report.failed) == ["job1"]
    assert "solver crashed" in report.failed["job1"]
    assert sorted(report.completed) == ["job0", "job2"]
    run = tmp_path / jobs[1].key
    assert not (run / "complete.json").exists()
    assert json.loads((run / "failed.json").read_text("utf-8"))["job_id"] == "fake-1"
    # A second pass sees the cloud job failed; it does not submit it again...
    again = FakeBackend(statuses={"fake-1": "failed"})
    report = cloud.run_batches(jobs, workdir=tmp_path, backend=again)
    assert again.submitted == []
    assert list(report.failed) == ["job1"]
    # ...unless asked to.
    retry = FakeBackend(statuses={"fake-1": "failed"})
    report = cloud.run_batches(jobs, workdir=tmp_path, backend=retry, retry_failed=True)
    assert len(retry.submitted) == 1
    assert not report.failed
    assert (run / "complete.json").is_file()


@pytest.mark.parametrize(
    ("matrix", "norm", "message"),
    [
        (np.array([[3.0, 1.0], [1.0, 3.0]]) * 1e-15, 1e-3, "off-diagonal"),
        (np.array([[3.0, -1.0], [-0.5, 3.0]]) * 1e-15, 1e-3, "symmetric"),
        (GOOD, None, "error-indicators"),
        (GOOD, float("nan"), "error indicator"),
    ],
)
def test_unphysical_or_incomplete_results_are_rejected(
    cloud: ModuleType,
    tmp_path: Path,
    matrix: np.ndarray,
    norm: float | None,
    message: str,
) -> None:
    (job,) = make_jobs(cloud, 1)
    report = cloud.run_batches(
        [job], workdir=tmp_path, backend=FakeBackend(matrix=matrix, norm=norm)
    )
    assert message in report.failed["job0"]
    assert cloud.read_complete(tmp_path, job) is None


def test_job_identity_is_the_input_hash(cloud: ModuleType) -> None:
    key = cloud.job_key({"b": 1, "a": [1.5, 2.0]})
    assert key == cloud.job_key({"a": [1.5, 2.0], "b": 1})
    assert key != cloud.job_key({"a": [1.5, 2.0], "b": 2})
    assert len(key) == 20


def test_result_files_accepts_both_gsim_shapes(
    cloud: ModuleType, tmp_path: Path
) -> None:
    mapping = {"terminal-C.csv": tmp_path / "x.csv"}
    assert cloud.result_files(mapping) == mapping
    run_result = SimpleNamespace(files=mapping, sim_dir=tmp_path, job_name="j")
    assert cloud.result_files(run_result) == mapping
    with pytest.raises(TypeError):
        cloud.result_files(42)


class FakeGcloud:
    """Stands in for ``gsim.gcloud``."""

    def __init__(self, tmp_path: Path, fail_batches: bool = False) -> None:
        """Remember the scratch directory and whether batch calls fail."""
        self.tmp_path = tmp_path
        self.fail_batches = fail_batches
        self.calls: list[tuple[str, ...]] = []

    @staticmethod
    def get_status(_job_id: str) -> str:
        return "completed"

    def wait_for_results(self, *job_ids: str, parent_dir: Path) -> Any:
        self.calls.append(job_ids)
        if len(job_ids) > 1 and self.fail_batches:
            raise RuntimeError("one job failed")
        if job_ids == ("bad",):
            raise RuntimeError("job bad failed")
        results = [
            SimpleNamespace(files=write_results(parent_dir / job_id, GOOD))
            for job_id in job_ids
        ]
        return results[0] if len(results) == 1 else results


def test_gsim_backend_submits_without_waiting_and_uses_the_cache(
    cloud: ModuleType, tmp_path: Path
) -> None:
    sim = SimpleNamespace(calls=[])

    def run(**kwargs: Any) -> str:
        sim.calls.append(kwargs)
        return "cloud-1"

    sim.run = run
    fake = FakeGcloud(tmp_path)
    backend = cloud.GsimBackend(gcloud=fake)
    assert backend.submit(cloud.Prepared(sim=sim), tmp_path / "r") == "cloud-1"
    assert sim.calls == [
        {"parent_dir": tmp_path / "r", "wait": False, "check_cache": True}
    ]
    assert backend.status("cloud-1") == "completed"
    assert fake.calls == []  # nothing was awaited


def test_gsim_backend_waits_for_batches_and_single_jobs(
    cloud: ModuleType, tmp_path: Path
) -> None:
    fake = FakeGcloud(tmp_path)
    backend = cloud.GsimBackend(gcloud=fake)
    batch = backend.wait(["j1", "j2"], tmp_path)
    assert set(batch) == {"j1", "j2"}
    assert fake.calls == [("j1", "j2")]
    single = backend.wait(["j3"], tmp_path)  # one job is not wrapped in a list
    assert set(single) == {"j3"}
    assert "terminal-C.csv" in single["j3"]


def test_gsim_backend_collects_per_job_after_a_failed_batch(
    cloud: ModuleType, tmp_path: Path
) -> None:
    fake = FakeGcloud(tmp_path, fail_batches=True)
    outcome = cloud.GsimBackend(gcloud=fake).wait(["j1", "bad", "j2"], tmp_path)
    assert isinstance(outcome["bad"], Exception)
    assert "terminal-C.csv" in outcome["j1"]
    assert "terminal-C.csv" in outcome["j2"]
    assert fake.calls[0] == ("j1", "bad", "j2")
    assert fake.calls[1:] == [("j1",), ("bad",), ("j2",)]


# --- De-embedding ---------------------------------------------------------------

C_PRIME = 166.6e-12  # F/m, the analytic model's CPW value, used as synthetic input
C_OPEN = 0.5e-15


def coupon_matrix(recipe: ModuleType, coupling: float = 0.0) -> np.ndarray:
    short, long = (length * 1e-6 for length in recipe.COUPON_LENGTHS)
    ground = np.array([C_PRIME * short + 2 * C_OPEN, C_PRIME * long + 2 * C_OPEN])
    return np.array([
        [ground[0] + coupling, -coupling],
        [-coupling, ground[1] + coupling],
    ])


def ring(q: int, c_c: float, c_g: float, c_x: float = 0.0) -> np.ndarray:
    """Maxwell matrix of q pins on a ring, with optional cross-disk coupling."""
    mutual = np.zeros((q, q))
    for i in range(q):
        j = (i + 1) % q
        mutual[i, j] = mutual[j, i] = c_c
        if q == 4:
            j = (i + 2) % 4
            mutual[i, j] = mutual[j, i] = c_x
    return np.diag(c_g + mutual.sum(axis=1)) - mutual


def test_calibration_recovers_line_capacitance_and_open_end(recipe: ModuleType) -> None:
    calibration = recipe.calibrate(coupon_matrix(recipe), "key")
    assert calibration.c_prime == pytest.approx(C_PRIME, rel=1e-12)
    assert calibration.c_open == pytest.approx(C_OPEN, rel=1e-9)
    assert calibration.coupon_key == "key"
    assert calibration.lead(20.0) == pytest.approx(C_PRIME * 20e-6 + C_OPEN, rel=1e-12)


def test_calibration_uses_ground_capacitance_when_strips_couple_weakly(
    recipe: ModuleType,
) -> None:
    calibration = recipe.calibrate(coupon_matrix(recipe, coupling=1e-20))
    assert calibration.c_prime == pytest.approx(C_PRIME, rel=1e-9)


def test_calibration_rejects_coupled_strips_and_nonphysical_values(
    recipe: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="couple"):
        recipe.calibrate(coupon_matrix(recipe, coupling=5e-17))  # 0.3% of the diagonal
    with pytest.raises(ValueError, match="2x2"):
        recipe.calibrate(np.eye(3))
    flat = np.diag([1e-15, 1e-15])
    with pytest.raises(ValueError, match="Non-physical"):
        recipe.calibrate(flat)


def test_deembedding_subtracts_the_lead_from_each_diagonal_entry(
    recipe: ModuleType,
) -> None:
    calibration = recipe.calibrate(coupon_matrix(recipe))
    core = ring(3, 5e-15, 10e-15)
    lead = C_PRIME * recipe.LEAD_LENGTH * 1e-6 + C_OPEN
    raw = core + lead * np.eye(3)
    np.testing.assert_allclose(recipe.deembed(raw, calibration), core, atol=1e-25)
    check_maxwell(recipe.deembed(raw, calibration))
    # Off-diagonal entries are untouched, and a longer lead removes more.
    longer = recipe.deembed(raw, calibration, lead_length=60.0)
    np.testing.assert_allclose(
        longer - np.diag(np.diag(longer)), core - np.diag(np.diag(core)), atol=1e-25
    )
    assert longer[0, 0] < core[0, 0]
    np.testing.assert_allclose(
        recipe.deembed(raw[None], calibration)[0], core, atol=1e-25
    )


def test_rotation_asymmetry_measures_the_deviation(recipe: ModuleType) -> None:
    assert recipe.rotation_asymmetry(ring(3, 5e-15, 10e-15)) == pytest.approx(
        0, abs=1e-12
    )
    matrix = ring(3, 5e-15, 10e-15)
    perturbed = matrix.copy()
    perturbed[0, 0] += 0.3e-15
    expected = (
        np.abs(perturbed - np.roll(np.roll(perturbed, 1, 0), 1, 1)).max()
        / np.diag(perturbed).max()
    )
    assert recipe.rotation_asymmetry(perturbed) == pytest.approx(expected)
    assert expected > 0.01


# --- Publishing from a synthetic cache -------------------------------------------


def fill_cache(recipe: ModuleType, workdir: Path, q: int, phase: str) -> dict:
    """Write finished runs for every point of a phase; return the cores by point."""
    cores = {}
    jobs = [recipe.coupon_job()]
    matrices = [coupon_matrix(recipe)]
    lead = C_PRIME * recipe.LEAD_LENGTH * 1e-6 + C_OPEN
    for point in recipe.points(recipe.phase_grid(phase)):
        scale = (point["pin_radius"] / 40.0) * (4.0 / point["pin_gap"]) ** 0.2
        core = ring(
            q,
            5e-15 * scale,
            10e-15 * (6.0 / point["ground_gap"]) ** 0.2 * scale,
            c_x=0.3e-15 * scale,
        )
        cores[tuple(point.values())] = core
        jobs.append(recipe.coupler_job(q, point))
        matrices.append(core + lead * np.eye(q))
    for job, matrix in zip(jobs, matrices, strict=True):
        run = workdir / job.key
        run.mkdir(parents=True)
        (run / "complete.json").write_text(
            json.dumps({
                "name": job.name,
                "job_id": "synthetic",
                "terminals": list(job.terminals),
                "maxwell_capacitance": matrix.tolist(),
                "diagnostics": {"fem_error_indicator_norm": 1e-3},
                "tetrahedra": 1,
            }),
            encoding="utf-8",
        )
    return cores


@pytest.mark.parametrize("q", [3, 4])
def test_publish_writes_a_loadable_deembedded_dataset(
    recipe: ModuleType, tmp_path: Path, q: int
) -> None:
    pytest.importorskip("gsim.palace")  # provenance records the solver stack

    cores = fill_cache(recipe, tmp_path / "work", q, "a")
    dataset = recipe.publish(
        q, phase="a", workdir=tmp_path / "work", output=tmp_path / "out"
    )
    reloaded = Dataset(tmp_path / "out")
    assert reloaded.metadata.name == f"hyperbolic_vertex_coupler_q{q}_palace"
    assert reloaded.metadata.terminals == tuple(f"o{k}" for k in range(1, q + 1))
    grid = reloaded.grid(
        "maxwell_capacitance", allow_missing=False, cross_section="cpw"
    )
    assert grid.axis_names == ("pin_radius", "pin_gap", "ground_gap")
    assert grid.values.shape == (3, 3, 3, q, q)
    for i, radius in enumerate(grid.coords[0]):
        for j, gap in enumerate(grid.coords[1]):
            for k, ground in enumerate(grid.coords[2]):
                np.testing.assert_allclose(
                    grid.values[i, j, k],
                    cores[radius, gap, ground],
                    rtol=1e-9,
                    atol=1e-24,
                )
    raw = reloaded.grid(
        "maxwell_capacitance_with_leads", allow_missing=False, cross_section="cpw"
    )
    lead = C_PRIME * 20e-6 + C_OPEN
    np.testing.assert_allclose(
        raw.values - grid.values,
        lead * np.eye(q) * np.ones(grid.values.shape),
        atol=1e-24,
    )
    provenance = reloaded.metadata.provenance
    assert provenance["calibration"]["line_capacitance_F_per_m"] == pytest.approx(
        C_PRIME
    )
    assert provenance["calibration"]["lead_length_um"] == pytest.approx(20.0)
    assert provenance["reference_plane"].startswith("disk rim")
    assert provenance["issue"] == 1090
    assert provenance["validation"]["status"] == "pending"
    assert provenance["rotation_asymmetry_max"] < 1e-9
    assert dataset.metadata.quantity("mutual_capacitance").kind == "mutual_capacitance"
    assert "solver_iterations" not in {
        quantity.name for quantity in reloaded.metadata.quantities
    }
    text = json.dumps(provenance)
    for forbidden in ("Users", "alisa", "C:"):
        assert forbidden not in text


def test_publish_names_missing_points_and_a_missing_coupon(
    recipe: ModuleType, tmp_path: Path
) -> None:
    pytest.importorskip("gsim.palace")
    with pytest.raises(FileNotFoundError, match="coupon"):
        recipe.publish(
            3, phase="a", workdir=tmp_path / "empty", output=tmp_path / "out"
        )
    work = tmp_path / "work"
    fill_cache(recipe, work, 3, "a")
    victim = recipe.coupler_job(
        3, {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    )
    (work / victim.key / "complete.json").unlink()
    with pytest.raises(ValueError, match="1 points have no finished run"):
        recipe.publish(3, phase="a", workdir=work, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()
