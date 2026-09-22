# /// script
# # gsim pins gdsfactoryplus, which is 3.12-only (cp312 wheels, ~=3.12.0), so a
# # standalone run needs 3.12 even though qpdk itself supports more.
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "qpdk[models]",
# ]
#
# [tool.qpdk]
# # What a cluster job runs, one process per trial. The dependency block makes it
# # runnable standalone with `uv run qpdk/simulation/trial.py ...`, which is what
# # a generated sbatch script does on a node that has no project environment; from
# # a checkout, `python -m qpdk.simulation.trial ...` uses the local qpdk instead.
# entry-point = "qpdk.simulation.trial:main"
# writes = "<params>.result.json"
# ///
"""Evaluate one trial of a parameter sweep, as its own process.

A sweep is a queue of independent solver runs, and the cheapest way to put that
queue on a cluster is one job per trial: a small allocation starts on any node
with room, so the sweep fills whatever is free. This module is what such a job
runs. It reads the trial's parameters from a JSON file, hands them to an
evaluator, and writes the result beside them, so the driver only has to read
files back.

Nothing here knows what kind of simulation the evaluator runs. ``--evaluator``
names any ``module:function`` taking ``(params, run_root)`` plus keyword
arguments and returning a row of results, so an eigenmode sweep, an S-parameter
sweep and a capacitance extraction all use the same entry point and differ only
in what they return. It is imported on the node the job lands on, so it has to
live in a module that is importable there.

Running it by hand is the same call the scheduler makes::

    python -m qpdk.simulation.trial params.json --run-root /scratch/run
        --ranks 8 --evaluator mysweep:evaluate_layout
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

__all__ = ["main"]


def _summarise(row: dict[str, object]) -> str:
    """Return a one-line summary of a result row, whatever produced it.

    The scalars are reported as they come rather than in named units, because
    an eigenmode row, an S-parameter row and a capacitance row share no keys.
    The job's log is a breadcrumb; the result file is the record.

    Args:
        row: The evaluator's result row.

    Returns:
        A compact ``key=value`` line, or a note when the row has no scalars.
    """
    scalars = [
        f"{key}={value:.4g}"
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return "  ".join(scalars) if scalars else f"no scalar results in {sorted(row)}"


def result_path(params_path: Path) -> Path:
    """Return the file the result for ``params_path`` is written to."""
    return params_path.with_suffix(".result.json")


def main(argv: list[str] | None = None) -> int:
    """Evaluate one trial and write its result next to its parameters.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success. A failed trial writes the error to the result file and
        still exits non-zero, so Slurm records the task as failed while the
        driver can report what went wrong.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params", type=Path, help="JSON file of trial parameters")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory each trial's simulation directory goes under",
    )
    parser.add_argument(
        "--ranks", type=int, default=1, help="MPI ranks to give the solver"
    )
    parser.add_argument(
        "--evaluator",
        required=True,
        help=(
            "module:function evaluating one trial, taking (params, run_root) "
            "and keyword arguments and returning a result row"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds before the solve is abandoned",
    )
    args = parser.parse_args(argv)

    module_name, _, function_name = args.evaluator.partition(":")
    evaluate_layout = getattr(import_module(module_name), function_name)

    params = json.loads(args.params.read_text())
    destination = result_path(args.params)
    try:
        # One directory per trial, keyed by the parameter file's own unique
        # path rather than by geometry: trials run concurrently and a sampler
        # does hand out duplicate parameter sets.
        sim_dir = args.run_root / args.params.parent.name / args.params.stem
        row = evaluate_layout(
            params,
            args.run_root,
            ranks=args.ranks,
            sim_dir=sim_dir,
            timeout=args.timeout,
        )
    except Exception as error:
        destination.write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}),
            encoding="utf-8",
        )
        # Echo the reason: the task's own log is where someone will look first.
        print(f"trial failed: {error}", file=sys.stderr)
        return 1
    destination.write_text(json.dumps(row), encoding="utf-8")
    print(_summarise(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
