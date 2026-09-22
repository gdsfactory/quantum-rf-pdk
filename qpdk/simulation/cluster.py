"""Launch simulation work on a Slurm cluster.

A geometry sweep is a queue of independent solver runs, and this module renders
the ``sbatch`` scripts for the two ways of handing that queue to Slurm:

- **One job per trial**, submitted as the study goes. Each is a small
  allocation that starts on any node with room, so the sweep fills whatever the
  cluster has free. How many run at once is the driver's business, not the
  scheduler's: it keeps a fixed number in flight.
- **A Ray cluster inside one allocation**, one worker per node. A good fit when
  the job owns its nodes, but it can only start once a whole node's worth of
  cores is free.

This module only turns settings into scripts and submits them; the loop that
decides what to submit lives in :mod:`qpdk.simulation.study`.

Every site-specific value is a plain string, so adapting the scripts to a given
cluster means filling in the fields rather than editing the templates.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jinja2

__all__ = [
    "RAY_PORT",
    "SlurmCluster",
    "SlurmJobError",
    "job_state",
    "submit",
    "wait",
]

# Ray's default port for the global control store on the head node.
RAY_PORT = 6379

# StrictUndefined so a template referring to a field that no longer exists is an
# error at render time rather than a silently empty line in an sbatch script.
_TEMPLATES = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
    # Shell scripts, not markup: HTML escaping would corrupt quoting.
    autoescape=False,  # ruff: ignore[jinja2-autoescape-false]
)


class SlurmJobError(RuntimeError):
    """Raised when a Slurm command fails."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlurmCluster:
    """Site settings for running batched simulation work on Slurm.

    The same settings describe both arrangements: one job per trial, and a Ray
    cluster inside a single multi-node allocation.

    Attributes:
        partition: Partition to request, or a comma-separated list of them.
            Naming several lets the scheduler start the job wherever it can
            begin earliest, which usually beats pinning one fast partition.
        scratch: Shared filesystem the job runs from, visible on every node.
            The driver writes results there and Ray stages its session under
            it.
        setup: Shell snippet run on every node before Ray starts, for loading
            environment modules, activating a virtual environment and
            redirecting package caches onto ``scratch``.
        account: Slurm billing account. Optional because on many sites the
            default account is the only one the general partitions accept.
        nodes: Number of nodes to request.
        cores_per_node: Cores per node, which also bounds how many solver
            ranks a node can run at once.
        mem_per_node: Memory per node. Sites commonly default to a few hundred
            megabytes per core, so this is worth setting explicitly.
        time_limit: Wall-clock limit, ``HH:MM:SS``.
        job_name: Slurm job name, reused in the log file names.
        log_dir: Directory for Slurm's job output, relative to the submit
            directory.
        solver_cores: Cores one solver invocation may use, and so the size of
            one trial's job. Under Ray the driver asks for this many CPUs per
            trial.
        mem_per_task: Memory for one trial's job. Separate from
            ``mem_per_node`` because such a job is sized on its own, not as a
            share of a node.
        driver_time_limit: Wall-clock limit for the driver job. The driver only
            submits and waits, so it needs one core but usually outlives each
            trial by a long way.
    """

    partition: str = "batch"
    scratch: str = "."
    setup: str = ""
    account: str | None = None
    nodes: int = 1
    cores_per_node: int = 16
    mem_per_node: str = "32G"
    time_limit: str = "04:00:00"
    job_name: str = "qpdk-ray"
    log_dir: str = "slurm-logs"
    solver_cores: int = 8
    mem_per_task: str = "24G"
    driver_time_limit: str = "24:00:00"

    def __post_init__(self) -> None:
        """Reject settings that could not schedule a single trial."""
        if self.solver_cores > self.cores_per_node:
            msg = (
                f"solver_cores={self.solver_cores} exceeds "
                f"cores_per_node={self.cores_per_node}: no trial could ever run"
            )
            raise ValueError(msg)

    def _render(
        self,
        template: str,
        command: str,
        *,
        suffix: str = "",
        time_limit: str | None = None,
    ) -> str:
        """Render one of the job templates against these settings.

        Args:
            template: Template file name under ``templates/``.
            command: The command the job runs.
            suffix: Appended to the job name, to tell job kinds apart in a log
                directory.
            time_limit: Wall clock, when it differs from :attr:`time_limit`.

        Returns:
            The script as a string.
        """
        context = asdict(self) | {
            "command": command,
            "suffix": suffix,
            "time_limit": time_limit or self.time_limit,
            # Slurm's own placeholder for the job id, left for it to expand.
            "log_id": "%j",
            # `nodes` is the bash array of hostnames in the Ray template, so the
            # count travels under its own name.
            "nodes_count": self.nodes,
            "ray_port": RAY_PORT,
        }
        return _TEMPLATES.get_template(template).render(**context)

    def _write(self, path: str | Path, text: str) -> Path:
        """Write a rendered script, creating what it needs to run.

        Slurm opens the job's output file before the script runs, so the log
        directory has to exist at submit time rather than when the script
        starts.

        Args:
            path: Destination for the script.
            text: The rendered script.

        Returns:
            The written path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def task_sbatch_script(self, command: str) -> str:
        """Render the script that solves one trial.

        Submitted once per trial. Slurm forwards any arguments after the script
        name to the script itself, so ``sbatch trial.sbatch params.json`` gives
        the command its parameter file as ``$1`` and one script serves the whole
        study.

        Args:
            command: Command the job runs, which can use ``"$1"``.

        Returns:
            The script as a string.
        """
        return self._render("trial.sh.j2", command)

    def write_task_script(self, path: str | Path, command: str) -> Path:
        """Write the one-trial script to ``path``.

        Args:
            path: Destination for the script.
            command: Command the job runs.

        Returns:
            The written path.
        """
        return self._write(path, self.task_sbatch_script(command))

    def driver_sbatch_script(self, driver: str) -> str:
        """Render the script for the driver job.

        The driver asks the optimizer for trials and submits a job per trial;
        the solving happens in those jobs, so this one needs a single core and a
        wall clock longer than the whole study.

        Args:
            driver: Command the driver job runs.

        Returns:
            The script as a string.
        """
        return self._render(
            "driver.sh.j2",
            driver,
            suffix="-driver",
            time_limit=self.driver_time_limit,
        )

    def write_driver_script(self, path: str | Path, driver: str) -> Path:
        """Write the driver script to ``path``.

        Args:
            path: Destination for the script.
            driver: Command the driver job runs.

        Returns:
            The written path.
        """
        return self._write(path, self.driver_sbatch_script(driver))

    def sbatch_script(self, driver: str) -> str:
        """Render the script that runs ``driver`` on a Ray cluster.

        Args:
            driver: Command to run on the head node once the cluster is up.

        Returns:
            The script as a string.
        """
        return self._render("ray.sh.j2", driver)

    def write_sbatch_script(self, path: str | Path, driver: str) -> Path:
        """Write the Ray cluster script to ``path``.

        Args:
            path: Destination for the script.
            driver: Command to run on the head node.

        Returns:
            The written path.
        """
        return self._write(path, self.sbatch_script(driver))


def submit(script: str | Path, *args: str) -> str:
    """Submit ``script`` to Slurm and return the job id.

    Args:
        script: Path to the ``sbatch`` script.
        args: Arguments passed to the script itself. Slurm forwards anything
            after the script name, so one script can serve many submissions
            that differ only in what they are given.

    Returns:
        The job id Slurm assigned.

    Raises:
        SlurmJobError: If ``sbatch`` fails.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["sbatch", str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"sbatch failed for {script}: {result.stderr.strip()}"
        raise SlurmJobError(msg)
    return result.stdout.strip().split()[-1]


def job_state(job_id: str) -> str | None:
    """Return a job's Slurm state, or ``None`` once it has left the queue.

    Args:
        job_id: A Slurm job id.

    Returns:
        The state Slurm reports, or ``None`` if the job is no longer queued.

    Raises:
        SlurmJobError: If ``squeue`` itself fails, which is not the same as the
            job having left the queue.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Distinct from "not queued": an unreachable or busy slurmctld also
        # prints nothing, and treating that as completion would let the driver
        # move on while the job is still running.
        msg = f"squeue failed for job {job_id}: {result.stderr.strip()}"
        raise SlurmJobError(msg)
    states = result.stdout.split()
    return states[0] if states else None


def wait(
    job_id: str, *, poll_seconds: float = 20.0, max_query_failures: int = 10
) -> None:
    """Block until ``job_id`` leaves the queue.

    A job can be briefly absent from ``squeue`` right after submission, so a
    single miss is not treated as completion.

    Args:
        job_id: A Slurm job id.
        poll_seconds: Seconds between polls.
        max_query_failures: Consecutive ``squeue`` failures tolerated before
            giving up.

    Raises:
        SlurmJobError: If ``squeue`` keeps failing, rather than silently
            reporting the job as finished.
    """
    misses = 0
    failures = 0
    while misses < 2:
        try:
            state = job_state(job_id)
        except SlurmJobError:
            # A scheduler that cannot answer says nothing about the job, so
            # retry rather than conclude anything. Give up only once it has
            # been unreachable for long enough to be a real outage.
            failures += 1
            if failures > max_query_failures:
                msg = (
                    f"squeue failed {failures} times in a row for job {job_id};"
                    " giving up rather than assuming it finished"
                )
                raise SlurmJobError(msg) from None
            time.sleep(poll_seconds)
            continue
        failures = 0
        misses = misses + 1 if state is None else 0
        if misses < 2:
            time.sleep(poll_seconds)


def _srun_ray(ray_arguments: str, *, nodelist: str, count: int) -> str:
    """Return a background ``srun`` step starting Ray on ``count`` nodes.

    The step is backgrounded so the batch script can continue; ``--block``
    keeps ``ray start`` in the foreground of the step, which is what holds the
    daemon alive for the length of the job.
    """
    return (
        f"srun -N {count} -n {count} --nodelist={nodelist} \\\n"
        f"  ray start {ray_arguments} --block &\n"
    )
