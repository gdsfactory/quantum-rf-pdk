"""Resumable Palace electrostatic runs on the cloud, at most five jobs at a time.

A recipe describes each solve as a :class:`Job`: JSON-serialisable inputs, whose
hash names the run directory, and a callable that meshes and configures it.
:func:`run_batches` submits jobs in batches, waits for a whole batch before it
starts the next, and stores one ``complete.json`` per finished job. A job with
a ``job.json`` but no ``complete.json`` is reattached, never submitted again.

This helper is separate from ``_palace.py`` on purpose: that file's checksum is
part of the cache key of every other recipe.
"""

# Native solver imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qpdk import logger

if TYPE_CHECKING:
    from _experiment import Geometry, Settings

MAX_CONCURRENT = 5
"""Hard upper bound on cloud jobs that are submitted but not yet collected."""

RESULT_FILES = ("terminal-C.csv", "terminal-Cm.csv", "error-indicators.csv")
"""Result files kept next to ``complete.json``."""

SIM_DIR = "sim"
"""Subfolder of a run holding what gsim uploads: mesh, config and nothing else.

gsim hashes every file of its output directory for the cloud result cache and
uploads them all, so this runner's bookkeeping lives one level up.
"""


def sim_dir(run: Path) -> Path:
    """Return the folder of a run that gsim meshes into and uploads."""
    return run / SIM_DIR


def job_key(inputs: Mapping[str, Any]) -> str:
    """Return the run identity: the hash of the JSON-serialised solver inputs."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]


@dataclass(slots=True)
class Prepared:
    """A meshed and configured simulation, ready for submission."""

    sim: Any
    tetrahedra: int | None = None
    estimated_runtime_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """One electrostatic solve.

    Args:
        name: Label for logs and tables; not part of the identity.
        inputs: Everything that affects the solve, JSON-serialisable. Keep the
            sweep grid out of it so a larger grid reuses finished runs.
        terminals: Terminal names in matrix row and column order.
        tolerance: Linear-solver tolerance the run was configured with.
        prepare: Mesh and configure the job in the given run directory.
    """

    name: str
    inputs: dict[str, Any]
    terminals: tuple[str, ...]
    tolerance: float
    prepare: Callable[[Path], Prepared]

    @property
    def key(self) -> str:
        """Run directory name."""
        return job_key(self.inputs)


class Backend(Protocol):
    """What the runner needs from the cloud; tests replace it with a fake."""

    def submit(self, prepared: Prepared, run_dir: Path) -> str:
        """Upload and start (or find a cached) job and return its id."""
        ...

    def status(self, job_id: str) -> str:
        """Return ``created``, ``queued``, ``running``, ``completed`` or ``failed``."""
        ...

    def wait(
        self, job_ids: Sequence[str], parent_dir: Path
    ) -> dict[str, Mapping[str, Path] | Exception]:
        """Wait for every job; return its result files or the error it raised."""
        ...


def result_files(result: Any) -> dict[str, Path]:
    """Return ``{file name: path}`` from either shape gsim may return.

    The 0.7.0 docstring promises ``dict[str, Path]``, but an unparsed download
    is a ``RunResult`` whose ``files`` holds that mapping.

    Raises:
        TypeError: If the result has neither shape.
    """
    if isinstance(result, Mapping):
        return {str(name): Path(path) for name, path in result.items()}
    files = getattr(result, "files", None)
    if isinstance(files, Mapping):
        return {str(name): Path(path) for name, path in files.items()}
    raise TypeError(f"Unrecognised result type {type(result).__name__}")


class GsimBackend:
    """Submit and collect jobs with ``gsim.gcloud``.

    Constructing this object does not touch the network. Pass ``gcloud`` to
    use another module with the same functions.
    """

    def __init__(self, gcloud: Any = None) -> None:
        """Import ``gsim.gcloud`` unless a replacement is given."""
        if gcloud is None:
            from gsim import gcloud as imported

            gcloud = imported
        self._gcloud = gcloud

    @staticmethod
    def submit(prepared: Prepared, run_dir: Path) -> str:
        """Upload and start a job; byte-identical inputs reuse a cloud result.

        Returns:
            The cloud job id.
        """
        return str(prepared.sim.run(parent_dir=run_dir, wait=False, check_cache=True))

    def status(self, job_id: str) -> str:
        """Return the cloud status string of a job."""
        return str(self._gcloud.get_status(job_id))

    def wait(
        self, job_ids: Sequence[str], parent_dir: Path
    ) -> dict[str, Mapping[str, Path] | Exception]:
        """Wait for a batch, then for each job alone if the batch call failed.

        One failed job makes the multi-job call raise before it downloads the
        jobs that come after it, so a failure falls back to one call per job.

        Returns:
            Result files, or the exception, per job id.
        """
        ids = list(job_ids)
        try:
            raw = self._gcloud.wait_for_results(*ids, parent_dir=parent_dir)
            results = [raw] if len(ids) == 1 else list(raw)
            return {
                job_id: result_files(result)
                for job_id, result in zip(ids, results, strict=True)
            }
        except Exception as error:
            logger.warning(
                f"Batch wait failed ({type(error).__name__}); collecting per job"
            )
        outcome: dict[str, Mapping[str, Path] | Exception] = {}
        for job_id in ids:
            try:
                outcome[job_id] = result_files(
                    self._gcloud.wait_for_results(job_id, parent_dir=parent_dir)
                )
            except Exception as error:
                outcome[job_id] = error
        return outcome


def prepare_sim(
    geometry: Geometry,
    *,
    settings: Settings,
    inputs: Mapping[str, Any],
    run_dir: Path,
    threads: int = 4,
    minimum_feature_elements: float = 4,
) -> Prepared:
    """Mesh sheets on silicon, write and validate the Palace config.

    The domain is a box holding an air prism of height ``settings.domain_pad``
    over a silicon prism of the same thickness. gsim treats metal that touches
    a terminal port as that terminal and all other metal as ground.

    Returns:
        The simulation with the tetrahedron count and gsim's runtime heuristic.

    Raises:
        ValueError: If the terminals differ from the geometry or the config is
            invalid.
    """
    import gdsfactory as gf
    from gsim.palace import ElectrostaticSim

    from qpdk.tech import LAYER

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    # Start clean: everything in this folder is uploaded and hashed.
    shutil.rmtree(sim_dir(run_dir), ignore_errors=True)
    sim_dir(run_dir).mkdir()
    sim = ElectrostaticSim()
    sim.set_geometry(geometry.component)
    sim.set_output_dir(sim_dir(run_dir))
    result = sim.mesh_sheets(
        conductor_layer=gf.get_layer_tuple(LAYER.M1_DRAW),
        terminal_ports=geometry.terminal_ports,
        domain_bounds=geometry.bounds,
        height=settings.domain_pad,
        near_mesh=settings.near_mesh,
        far_mesh=settings.far_mesh,
        permittivity=settings.permittivity,
        minimum_feature_elements=minimum_feature_elements,
        num_threads=threads,
    )
    actual = [terminal.name for terminal in sim.terminals]
    if actual != list(geometry.terminal_ports):
        raise ValueError(f"Mesh terminals {actual} != {list(geometry.terminal_ports)}")
    # Warm starts change the norm used to normalize the logged residuals.
    sim.set_solver(
        order=settings.order,
        tolerance=settings.tolerance,
        max_iterations=1000,
        preconditioner="BoomerAMG",
        initial_guess=False,
    )
    sim.set_electrostatic(save_fields=0)
    sim.write_config()
    report = sim.validate_config()
    if not report.valid:
        raise ValueError(f"Invalid Palace config: {report}")
    estimate = None
    try:
        from gsim import gcloud

        estimate = float(gcloud.estimate_runtime_seconds(sim_dir(run_dir)))
    except ImportError:
        pass
    tetrahedra = result.mesh_stats.get("tetrahedra")
    return Prepared(
        sim=sim,
        tetrahedra=None if tetrahedra is None else int(tetrahedra),
        estimated_runtime_s=estimate,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_complete(workdir: Path, job: Job) -> dict[str, Any] | None:
    """Return a job's stored result, or ``None`` if it has not finished."""
    path = workdir / job.key / "complete.json"
    return _read_json(path) if path.is_file() else None


def finalize(
    job: Job, run: Path, files: Mapping[str, Path], job_id: str
) -> dict[str, Any]:
    """Check one downloaded result and write ``complete.json``.

    The Maxwell matrix must be physical. The FEM error indicator must be
    present. Solver residual and iteration counts need a verbose Palace log;
    they are recorded when the download holds one and left out otherwise.

    Returns:
        The stored record.

    Raises:
        FileNotFoundError: If a capacitance matrix or the error indicator is
            missing from the download.
        ValueError: If the matrix is not physical or the indicator is invalid.
    """
    import math

    from gsim.palace.capacitance import load_capacitance
    from gsim.palace.results import load_text_results

    from qpdk.models.datasets.capacitance import check_maxwell

    matrices = load_capacitance(dict(files), terminal_names=job.terminals)
    check_maxwell(matrices.maxwell)
    if problems := matrices.problems(rtol=1e-8):
        raise ValueError(f"Inconsistent Palace results: {problems}")
    if "error-indicators.csv" not in files:
        raise FileNotFoundError("error-indicators.csv is missing from the download")
    indicators = load_text_results({
        "error-indicators.csv": files["error-indicators.csv"]
    }).error_indicators
    if (
        indicators is None
        or not math.isfinite(indicators["norm"])
        or indicators["norm"] < 0
    ):
        raise ValueError("Invalid FEM error indicator")
    diagnostics = {"fem_error_indicator_norm": float(indicators["norm"])}
    logs = sorted(name for name in files if name.endswith(".log"))
    if logs:
        from _palace import solver_diagnostics

        text = Path(files[logs[0]]).read_text(encoding="utf-8", errors="replace")
        diagnostics |= solver_diagnostics(
            text, terminals=len(job.terminals), tolerance=job.tolerance
        )
    mesh = _read_json(run / "mesh.json") if (run / "mesh.json").is_file() else {}
    kept = run / "results"
    kept.mkdir(exist_ok=True)
    for name in (*RESULT_FILES, *logs):
        if name in files:
            shutil.copy2(files[name], kept / name)
    record = {
        "name": job.name,
        "job_id": job_id,
        "terminals": list(job.terminals),
        "maxwell_capacitance": matrices.maxwell.tolist(),
        "diagnostics": diagnostics,
        "tetrahedra": mesh.get("tetrahedra"),
    }
    _write_json(run / "complete.json", record)
    (sim_dir(run) / "palace.msh").unlink(missing_ok=True)
    return record


@dataclass(slots=True)
class Report:
    """What :func:`run_batches` did."""

    reused: list[str] = field(default_factory=list)
    reattached: list[str] = field(default_factory=list)
    submitted: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    batches: list[list[str]] = field(default_factory=list)


def unique_jobs(jobs: Sequence[Job]) -> list[Job]:
    """Drop jobs whose inputs, and so whose run, are identical."""
    seen: dict[str, Job] = {}
    for job in jobs:
        seen.setdefault(job.key, job)
    return list(seen.values())


def run_batches(
    jobs: Sequence[Job],
    *,
    workdir: Path,
    backend: Backend,
    max_concurrent: int = MAX_CONCURRENT,
    retry_failed: bool = False,
) -> Report:
    """Run jobs in batches; a batch is collected before the next is submitted.

    Jobs with ``complete.json`` are skipped. A job with ``job.json`` is
    reattached with its stored id. A job that failed is reported and left
    without ``complete.json``; it is submitted again only with
    ``retry_failed``. Meshing happens just before submission, so only one mesh
    exists on disk at a time besides those of the batch in flight.

    Returns:
        The jobs that were reused, reattached, submitted, completed or failed.

    Raises:
        ValueError: If ``max_concurrent`` is not between 1 and ``MAX_CONCURRENT``.
    """
    if not 1 <= max_concurrent <= MAX_CONCURRENT:
        raise ValueError(f"max_concurrent must be in 1..{MAX_CONCURRENT}")
    workdir.mkdir(parents=True, exist_ok=True)
    downloads = workdir / "downloads"
    report = Report()
    batch: list[tuple[Job, str]] = []

    def collect() -> None:
        if not batch:
            return
        report.batches.append([job.name for job, _ in batch])
        outcome = backend.wait([job_id for _, job_id in batch], downloads)
        for job, job_id in batch:
            run = workdir / job.key
            result = outcome.get(job_id, RuntimeError("no result returned"))
            failure = result if isinstance(result, Exception) else None
            if failure is None:
                try:
                    finalize(job, run, result, job_id)
                except Exception as error:
                    failure = error
            if failure is not None:
                report.failed[job.name] = f"{type(failure).__name__}: {failure}"
                _write_json(
                    run / "failed.json",
                    {"job_id": job_id, "error": report.failed[job.name]},
                )
                logger.error(f"{job.name} failed: {report.failed[job.name]}")
            else:
                report.completed.append(job.name)
        batch.clear()

    for job in unique_jobs(jobs):
        run = workdir / job.key
        if (run / "complete.json").is_file():
            report.reused.append(job.name)
            continue
        job_json = run / "job.json"
        if job_json.is_file():
            job_id = _read_json(job_json)["job_id"]
            if backend.status(job_id).casefold() == "failed":
                if not retry_failed:
                    report.failed[job.name] = f"cloud job {job_id} failed"
                    continue
                job_json.unlink()
            else:
                logger.info(f"Reattaching {job.name}: job {job_id}")
                report.reattached.append(job.name)
                batch.append((job, job_id))
                if len(batch) == max_concurrent:
                    collect()
                continue
        prepared = job.prepare(run)
        _write_json(
            run / "mesh.json",
            {
                "tetrahedra": prepared.tetrahedra,
                "estimated_runtime_s": prepared.estimated_runtime_s,
            },
        )
        logger.info(
            f"Submitting {job.name}: {prepared.tetrahedra} tetrahedra, "
            f"gsim heuristic {prepared.estimated_runtime_s} s"
        )
        job_id = backend.submit(prepared, run)
        _write_json(run / "job.json", {"job_id": job_id})
        (run / "failed.json").unlink(missing_ok=True)
        report.submitted.append(job.name)
        batch.append((job, job_id))
        if len(batch) == max_concurrent:
            collect()
    collect()
    return report
