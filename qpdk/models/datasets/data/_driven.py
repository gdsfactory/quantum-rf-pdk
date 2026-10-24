"""Resumable Palace sweeps for labelled N-port scattering datasets.

A recipe prepares the geometry, mesh, ports, sweep and config. This helper
runs one complete frequency sweep per geometry and stores all matrix entries
using the usual dataset schema. It makes no assumptions about port names,
layer stacks, or lumped versus wave ports.
"""

# Native solver imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from csv import reader
from dataclasses import dataclass, field
from functools import cache
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import numpy as np
import typer
from _palace import fingerprint, runtime

from qpdk import PDK, logger
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write

if TYPE_CHECKING:
    from gsim.palace import DrivenSim


@dataclass(frozen=True, slots=True, kw_only=True)
class Recipe:
    """Describe a passive N-port experiment and how to prepare each geometry.

    The generator runs one complete frequency sweep per geometry and stores
    every complex S-parameter in the requested terminal order. Each port needs
    its own independent excitation and a real, positive reference impedance.

    Args:
        name: Dataset identifier, also used for the default output directory.
        script: Recipe source path, usually ``Path(__file__)``. Its checksum is
            recorded with the dataset to identify the code that generated it.
        grid: Numeric geometry parameters and their sampled values. All
            combinations are simulated, with lengths in um unless overridden
            by ``axis_units``. These axes are interpolated by the dataset model.
        frequencies: Full frequency grid in Hz, finite, positive and strictly
            increasing. The prepared solver sweep must return these frequencies
            for every geometry and excitation.
        terminals: Port names in matrix row and column order. These must match
            the names configured by ``prepare`` and become the SAX model ports.
        prepare: Callable ``prepare(point, output, processes)`` returning a
            meshed ``DrivenSim`` with its ports, sweep and final config written
            in ``output``. ``point`` contains one value from each grid axis and
            variant, and ``processes`` is the requested MPI process count. The
            generator runs the returned simulation and collects its results.
        description: Short description of the device and its physical setting,
            stored with the dataset for users loading it later.
        provenance: JSON-serializable settings and assumptions, including mesh,
            materials, port reference impedances and reference planes. Include
            every setting that affects the solve. These values form part of the
            cache key, so changing them starts new simulations.
        reference_ground: Name of the common reference conductor in the dataset
            metadata. Configure the actual ground geometry and boundaries in
            ``prepare``.
        variants: Discrete choices sampled alongside the numeric grid, such as
            ``{"n_turns": ["3", "5"]}``. Values are strings and are selected
            explicitly when loading a model, rather than interpolated.
        axis_units: Unit labels for numeric grid axes, defaulting to ``"um"``.
            These labels do not convert values; ``prepare`` must interpret them
            consistently. Frequency is always stored in Hz.
        reciprocal: Whether to check that exchanging input and output ports
            leaves the response unchanged. Set False for nonreciprocal media.
            Passivity is checked in either case.
    """

    name: str
    script: Path
    grid: dict[str, list[float]]
    frequencies: list[float]
    terminals: tuple[str, ...]
    prepare: Callable[[dict[str, float | str], Path, int], DrivenSim]
    description: str
    provenance: dict[str, Any]
    reference_ground: str = "ground"
    variants: dict[str, list[str]] = field(default_factory=dict)
    axis_units: dict[str, str] = field(default_factory=dict)
    reciprocal: bool = True

    def __post_init__(self) -> None:
        """Reject an empty, repeated or unordered frequency grid."""
        if not self.frequencies or any(
            not np.isfinite(f) or f <= 0 for f in self.frequencies
        ):
            raise ValueError("Frequencies must be finite and positive")
        if any(
            a >= b for a, b in zip(self.frequencies, self.frequencies[1:], strict=False)
        ):
            raise ValueError("Frequencies must be strictly increasing")


def response(
    run: Path,
    terminals: tuple[str, ...],
    frequencies: list[float],
    *,
    reciprocal: bool = True,
) -> np.ndarray:
    """Read a complete converged sweep in the requested port order.

    Allow 0.005 absolute reciprocity error and 0.01 passivity excess. These
    are numerical sanity checks, not mesh or interpolation accuracy bounds.

    Returns:
        Complex array with frequency, output port and input port dimensions.

    Raises:
        ValueError: If convergence, port coverage, reciprocity or passivity fails.
    """
    from gsim.palace._validation_log import (
        clean_lines,
        solver_checks,
    )
    from gsim.palace.results import (
        _parse_sparam_col,
        _resolve_source,
        load_sparams,
    )

    checks, _ = solver_checks(
        clean_lines((run / "solver.log").read_text()), estimator_disabled=False
    )
    if checks.status != "passed":
        raise ValueError(f"Driven solver did not converge in {run}")
    data = load_sparams(run)
    if set(data.port_names) != set(terminals):
        raise ValueError(f"Unexpected ports in {run}: {data.port_names}")
    csv_path = cast(Path, _resolve_source(run)[0])
    with csv_path.open(newline="") as stream:
        columns = next(reader(stream))
    parts: dict[tuple[int, int], set[str]] = {}
    for column in columns:
        if (parsed := _parse_sparam_col(column.strip())) is not None:
            i, j, kind = parsed
            parts.setdefault((i, j), set()).add(kind)
    # The result reader supplies zero phase for a missing column.
    if len(parts) != len(terminals) ** 2 or any(
        kinds != {"db", "deg"} for kinds in parts.values()
    ):
        raise ValueError(f"Incomplete complex S-parameter columns in {run}")
    np.testing.assert_allclose(
        np.asarray(data.freq) * 1e9, frequencies, rtol=1e-8, atol=1e-3
    )
    matrix = np.stack(
        [np.stack([data[a, b].complex for b in terminals], axis=-1) for a in terminals],
        axis=-2,
    )
    if (
        matrix.shape != (len(frequencies), len(terminals), len(terminals))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"Incomplete or nonfinite S-parameters in {run}")
    if reciprocal and np.max(np.abs(matrix - matrix.swapaxes(-1, -2))) > 0.005:
        raise ValueError(f"Scattering reciprocity failed in {run}")
    if np.linalg.svd(matrix, compute_uv=False).max() > 1.01:
        raise ValueError(f"Scattering passivity failed in {run}")
    return matrix


def generate(
    recipe: Recipe,
    *,
    grid: dict[str, list[float]],
    output: Path,
    workdir: Path,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace-x86_64.bin",
) -> Dataset:
    """Run each geometry once and write a complete complex frequency grid."""
    PDK.activate()
    execution, files = runtime(executable, sif, container_binary)
    provenance = {
        **fingerprint(recipe.script),
        **files,
        **recipe.provenance,
        "driven_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processes": processes,
        "reciprocal": recipe.reciprocal,
        "diagnostics": "Converged at every frequency and excitation; reciprocity error <= 0.005 when requested; maximum singular value <= 1.01",
    }
    if "linear_initial_guess" not in recipe.provenance:
        provenance.pop("linear_initial_guess", None)
    metadata = DatasetMetadata(
        name=recipe.name,
        description=recipe.description,
        axes=(
            *(
                Axis(name=name, unit=recipe.axis_units.get(name, "um"))
                for name in recipe.grid
            ),
            Axis(name="frequency", unit="Hz"),
        ),
        variants=tuple(recipe.variants),
        terminals=recipe.terminals,
        reference_ground=recipe.reference_ground,
        quantities=(
            Quantity(
                name="s_parameters",
                kind=QuantityKind.S_PARAMETERS,
                unit="1",
                complex=True,
            ),
        ),
        provenance=provenance,
    )

    @cache
    def solve_geometry(values: tuple[float | str, ...]) -> np.ndarray:
        point = dict(zip((*recipe.grid, *recipe.variants), values, strict=True))
        inputs = {
            "point": point,
            "frequencies_Hz": recipe.frequencies,
            "terminals": recipe.terminals,
            "provenance": {
                name: value
                for name, value in provenance.items()
                if name != "generator_sha256"
            },
        }
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        run = workdir.resolve() / key
        complete = run / "complete.json"
        if not complete.exists():
            run.mkdir(parents=True, exist_ok=True)
            (run / "inputs.json").write_text(json.dumps(inputs, indent=2))
            sim = recipe.prepare(point, run, processes)
            sim.run_local(
                **execution,
                num_processes=processes,
                num_threads=1,
                verbose=True,
                log_path=run / "solver.log",
            )
        matrix = response(
            run, recipe.terminals, recipe.frequencies, reciprocal=recipe.reciprocal
        )
        complete.write_text(json.dumps({"terminals": recipe.terminals}))
        return matrix

    def solve(frequency: float, **point: float | str) -> dict[str, np.ndarray]:
        matrix = solve_geometry(
            tuple(point[name] for name in (*recipe.grid, *recipe.variants))
        )
        return {"s_parameters": matrix[recipe.frequencies.index(frequency)]}

    data = write(
        output,
        metadata,
        sweep(
            metadata, solve, {**grid, "frequency": recipe.frequencies}, recipe.variants
        ),
    )
    for values in product(*recipe.variants.values()):
        data.grid(
            "s_parameters",
            allow_missing=False,
            **dict(zip(recipe.variants, values, strict=True)),
        )
    return data


def run_cli(recipe: Recipe) -> None:
    """Preview, generate geometry shards, or merge their full frequency sweeps."""

    def main(
        dry_run: bool = False,
        shard: Annotated[int, typer.Option(min=0)] = 0,
        shards: Annotated[int, typer.Option(min=1)] = 1,
        merge_shards: Annotated[list[Path] | None, typer.Option()] = None,
        output: Path | None = None,
        workdir: Path | None = None,
        executable: Path | None = None,
        sif: Path | None = None,
        processes: Annotated[int, typer.Option(min=1)] = 4,
        container_binary: str = "palace-x86_64.bin",
    ) -> None:
        """Generate an N-port S-parameter dataset with Palace."""
        if executable is not None and sif is not None:
            raise typer.BadParameter("Choose --executable or --sif")
        output = output or Path("build/datasets") / recipe.name
        workdir = workdir or Path("build/palace") / recipe.name
        if merge_shards:
            merge(
                merge_shards,
                output,
                grid={**recipe.grid, "frequency": recipe.frequencies},
                variants=recipe.variants,
            )
        elif dry_run:
            count = prod(map(len, recipe.grid.values())) * prod(
                map(len, recipe.variants.values())
            )
            logger.info(
                f"{recipe.name}: {count} geometries, {len(recipe.frequencies)} frequencies, {len(recipe.terminals)} ports"
            )
        else:
            generate(
                recipe,
                grid=partition_grid(recipe.grid, shard=shard, shards=shards),
                output=output,
                workdir=workdir,
                executable=executable,
                sif=sif,
                processes=processes,
                container_binary=container_binary,
            )

    typer.run(main)
