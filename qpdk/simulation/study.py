"""Drive an Optuna study whose trials are solved somewhere else.

A geometry sweep is a queue of independent solver runs, so the only real
questions are where a trial runs and how many run at once. Those are separate
concerns here: a *runner* knows how to start one trial and how to collect it,
and :func:`run_study` keeps a fixed number of them in flight while feeding
results back to the sampler.

Keeping trials in flight rather than evaluating fixed batches matters twice
over. A batch can only advance when its slowest trial finishes, so a wave of
twelve solves idles most of its allocation waiting for one big mesh; and the
sampler learns nothing until the whole wave lands, when it could have been told
about each result as it arrived.

Two runners come with the module and the difference between them is only where
the solve happens:

- :class:`SlurmRunner` submits one job per trial. A small job starts on any node
  with room, so the sweep fills whatever the cluster has free.
- :class:`RayRunner` dispatches a Ray task per trial, for when the study already
  owns an allocation.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qpdk import logger
from qpdk.simulation.cluster import SlurmCluster, SlurmJobError, job_state, submit

if TYPE_CHECKING:
    import optuna

__all__ = ["RayRunner", "SlurmRunner", "TrialRunner", "run_study"]


class TrialRunner(Protocol):
    """Starts one trial somewhere and collects it when it is done."""

    def start(self, name: str, params: dict[str, float]) -> Any:
        """Begin evaluating ``params`` and return a handle to it.

        Args:
            name: Unique name for the trial, usable as a directory name.
            params: The geometry to evaluate.

        Returns:
            A handle to pass back to :meth:`collect`.
        """
        ...

    def collect(self, handle: Any) -> dict[str, Any] | None:
        """Return a trial's row, or ``None`` while it is still running.

        Args:
            handle: The value :meth:`start` returned.

        Returns:
            The result row, carrying ``error`` if the trial failed, or ``None``
            if it has not finished yet.
        """
        ...


@dataclass(slots=True)
class SlurmRunner:
    """Run each trial as its own Slurm job.

    One script serves the whole study: Slurm forwards the arguments after the
    script name to the script, so each submission differs only in the parameter
    file it is given.

    Attributes:
        cluster: Site settings the job is sized and submitted with.
        run_root: Shared directory the trials are written under.
        ranks: MPI ranks to give each solve.
        evaluator: Callable to serialize for worker jobs, or an importable
            ``module:function`` name.
        timeout: Seconds before a solve is abandoned. Keep it under the job's
            own wall clock, or Slurm kills the job first and the trial reports
            nothing.
    """

    cluster: SlurmCluster
    run_root: Path
    ranks: int
    evaluator: str | Callable[..., dict[str, Any]]
    timeout: float = 1800.0
    _script: Path | None = field(default=None, init=False)
    _missing_since: dict[str, float] = field(default_factory=dict, init=False)
    _query_failures: dict[str, tuple[int, float]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Validate the solver ranks before writing or submitting anything.

        Raises:
            ValueError: If the ranks do not fit the trial allocation.
        """
        if (
            self.ranks <= 0
            or self.ranks > self.cluster.solver_cores
            or self.ranks % self.cluster.trial_nodes
        ):
            raise ValueError(
                "ranks must be positive, fit cluster.solver_cores and divide "
                "evenly across cluster.trial_nodes"
            )

    def _ensure_script(self) -> Path:
        """Write the one-trial script once and reuse it for every submission."""
        if self._script is None:
            self.run_root.mkdir(parents=True, exist_ok=True)
            # Reusing parameters could admit results from a previous study.
            (self.run_root / "params").mkdir()
            if isinstance(self.evaluator, str):
                evaluator_arg = f"--evaluator {shlex.quote(self.evaluator)}"
            else:
                import cloudpickle

                self.run_root.mkdir(parents=True, exist_ok=True)
                evaluator_path = (self.run_root / "evaluator.pkl").resolve()
                evaluator_path.write_bytes(cloudpickle.dumps(self.evaluator))
                evaluator_arg = f"--evaluator-file {shlex.quote(str(evaluator_path))}"
            command = (
                "python -m qpdk.simulation.trial "
                f'"$1" --run-root {shlex.quote(str(self.run_root))} '
                f"--ranks {self.ranks} --timeout {self.timeout} {evaluator_arg}"
            )
            self._script = self.cluster.write_task_script(
                self.run_root / "trial.sbatch", command
            )
        return self._script

    def start(self, name: str, params: dict[str, float]) -> tuple[str, Path]:
        """Submit one job for ``params``.

        Args:
            name: Unique trial name, used for the parameter file.
            params: The geometry to evaluate.

        Returns:
            The Slurm job id and the parameter file's path.
        """
        script = self._ensure_script()
        params_dir = self.run_root / "params"
        params_path = params_dir / f"{name}.json"
        with params_path.open("x", encoding="utf-8") as stream:
            json.dump(params, stream)
        job_id = submit(script, str(params_path))
        return job_id, params_path

    def collect(self, handle: tuple[str, Path]) -> dict[str, Any] | None:
        """Read a finished job's result, if it has one yet.

        Args:
            handle: The job id and parameter path from :meth:`start`.

        Returns:
            The row, or ``None`` while the job is still queued or running.

        Raises:
            SlurmJobError: If the scheduler query fails after ten spaced retries.
        """
        job_id, params_path = handle
        result_path = params_path.with_suffix(".result.json")
        if result_path.exists():
            return json.loads(result_path.read_text())
        if job_id in self._query_failures:
            _, retry_after = self._query_failures[job_id]
            if time.monotonic() < retry_after:
                return None
        try:
            state = job_state(job_id)
        except SlurmJobError:
            self._missing_since.pop(job_id, None)
            failures = self._query_failures.get(job_id, (0, 0.0))[0] + 1
            if failures > 10:
                raise
            self._query_failures[job_id] = (failures, time.monotonic() + 20.0)
            return None
        self._query_failures.pop(job_id, None)
        if state is not None:
            self._missing_since.pop(job_id, None)
            return None
        now = time.monotonic()
        missing_since = self._missing_since.setdefault(job_id, now)
        # Other completed trials can trigger immediate polls without sleeping.
        if now - missing_since < 60.0:
            return None
        if result_path.exists():
            return json.loads(result_path.read_text())
        # Gone from the queue with nothing written: killed, out of wall clock,
        # or it never started. Distinguishable from a solver failure, which
        # writes its reason to the result file.
        return {"error": f"job {job_id} left the queue without writing a result"}


@dataclass(slots=True)
class RayRunner:
    """Run each trial as a Ray task inside an allocation the study owns.

    Attributes:
        run_root: Shared directory the trials are written under.
        ranks: MPI ranks to give each solve, and the CPUs each task reserves.
        evaluator: Callable serialized by Ray or an importable
            ``module:function`` name.
    """

    run_root: Path
    ranks: int
    evaluator: str | Callable[..., dict[str, Any]]

    def start(self, name: str, params: dict[str, float]) -> Any:
        """Dispatch one Ray task for ``params``.

        Args:
            name: Unique trial name, used for the trial's directory.
            params: The geometry to evaluate.

        Returns:
            The Ray object reference for the task.

        Raises:
            RuntimeError: If ``RAY_ADDRESS`` is unset, i.e. this is not running
                inside a Ray cluster.
        """
        import ray

        if not os.environ.get("RAY_ADDRESS"):
            # Without it ray.init() starts a local instance, where a task
            # asking for `ranks` cores on a one-core driver never becomes
            # schedulable and the study hangs instead of failing.
            msg = (
                "RAY_ADDRESS is not set, so this is not inside a Ray cluster. "
                "Submit the script written by SlurmCluster.write_sbatch_script,"
                " which starts the head and workers and exports it."
            )
            raise RuntimeError(msg)
        ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)

        run_root, ranks = self.run_root.resolve(), self.ranks
        evaluator = self.evaluator

        env_vars = {
            key: os.environ[key] for key in ("QPDK_PALACE_SIF",) if key in os.environ
        }

        @ray.remote(num_cpus=ranks, runtime_env={"env_vars": env_vars})
        def evaluate(params: dict[str, float], name: str) -> dict[str, Any]:
            if isinstance(evaluator, str):
                module_name, _, function_name = evaluator.partition(":")
                evaluate_layout = getattr(import_module(module_name), function_name)
            else:
                evaluate_layout = evaluator
            return evaluate_layout(
                params, run_root, ranks=ranks, sim_dir=run_root / name
            )

        return evaluate.remote(params, name)

    @staticmethod
    def collect(handle: Any) -> dict[str, Any] | None:
        """Return a task's row if it has finished.

        Args:
            handle: The object reference from :meth:`start`.

        Returns:
            The row, or ``None`` while the task is still running.
        """
        import ray

        ready, _ = ray.wait([handle], timeout=0)
        if not ready:
            return None
        try:
            return ray.get(handle)
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}


def run_study(
    study: optuna.Study,
    runner: TrialRunner,
    *,
    suggest: Callable[[optuna.Trial], None],
    objectives: Callable[[dict[str, Any]], Sequence[float]],
    n_trials: int,
    max_in_flight: int,
    poll_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Run ``study`` to ``n_trials``, keeping trials in flight as they finish.

    The sampler is asked for a new trial as soon as one comes back rather than
    at a batch boundary, so no part of the study waits on its slowest solve and
    every result informs the next suggestion.

    Args:
        study: The study to advance. Its sampler decides what to try next.
        runner: Starts and collects a trial.
        suggest: Fills a trial's parameters, usually with ``trial.suggest_*``.
        objectives: Maps a result row to the values ``study`` was created for.
        n_trials: Total trials to evaluate.
        max_in_flight: How many trials may be running at once. On Slurm this is
            the real concurrency; under Ray the cluster's cores also bound it.
        poll_seconds: Seconds between checks for finished trials.

    Returns:
        One row per successful trial, carrying its number and parameters.

    Raises:
        ModuleNotFoundError: If Optuna is not installed. Use ``qpdk[models]``.
        ValueError: If ``max_in_flight`` is not positive.
        RuntimeError: If every trial failed, since there would be nothing to
            analyse.
    """
    try:
        import optuna
    except ModuleNotFoundError as error:
        if error.name != "optuna":
            raise
        raise ModuleNotFoundError(
            "run_study needs Optuna; install qpdk[models]"
        ) from error

    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    rows: list[dict[str, Any]] = []
    in_flight: dict[int, tuple[optuna.Trial, Any]] = {}
    asked = failed = 0
    started = time.monotonic()

    while asked < n_trials or in_flight:
        while asked < n_trials and len(in_flight) < max_in_flight:
            trial = study.ask()
            suggest(trial)
            handle = runner.start(f"trial_{trial.number:04d}", dict(trial.params))
            in_flight[trial.number] = (trial, handle)
            asked += 1

        finished = [
            (number, trial, row)
            for number, (trial, handle) in list(in_flight.items())
            if (row := runner.collect(handle)) is not None
        ]
        for number, trial, row in finished:
            del in_flight[number]
            if "error" in row:
                failed += 1
                logger.warning(f"trial {number} failed: {row['error']}")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            study.tell(trial, objectives(row))
            rows.append(row | dict(trial.params) | {"trial": trial.number})
            logger.info(
                f"trial {number} done ({len(rows)} ok, {failed} failed, "
                f"{len(in_flight)} in flight, {time.monotonic() - started:.0f} s)"
            )

        if in_flight and not finished:
            time.sleep(poll_seconds)

    if not rows:
        msg = f"all {failed} trials failed; check the job logs under the run root"
        raise RuntimeError(msg)
    return rows
