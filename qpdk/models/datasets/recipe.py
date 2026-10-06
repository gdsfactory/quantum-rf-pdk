"""Run a Palace dataset generator from a TOML sweep recipe.

Recipes are generation inputs, not dataset manifests. Results remain
self-describing Parquet files and need no recipe when loaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shlex
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from qpdk import logger
from qpdk.models.datasets.table import Dataset
from qpdk.simulation.palace import ElectrostaticSettings, Palace


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationRecipe:
    """A generator, parameter grid and Palace runtime, ready to preview or run.

    A generator is a ``module:function`` accepting ``runner``, ``grid`` and
    ``output`` keyword arguments and returning a :class:`Dataset`. It owns
    the geometry, units, terminal conventions and dataset metadata.
    """

    generator: str
    grid: dict[str, list[float]]
    output: Path
    workdir: Path
    command: tuple[str, ...] = ("palace", "-np", "4")
    settings: ElectrostaticSettings = field(default_factory=ElectrostaticSettings)

    def __post_init__(self) -> None:
        """Reject malformed recipes before any meshing or solver execution."""
        if not isinstance(self.generator, str):
            raise TypeError("generator must be a Python module:function")
        module, separator, function = self.generator.partition(":")
        if (
            not separator
            or not function.isidentifier()
            or not all(part.isidentifier() for part in module.split("."))
        ):
            raise ValueError("generator must be a Python module:function")
        if not self.command or not all(
            isinstance(arg, str) and arg for arg in self.command
        ):
            raise ValueError("palace.command must be a nonempty array of strings")
        if (
            not isinstance(self.grid, dict)
            or not self.grid
            or any(
                not isinstance(values, list)
                or not values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    for value in values
                )
                or len(set(values)) != len(values)
                for values in self.grid.values()
            )
        ):
            raise ValueError("grid axes need nonempty arrays of unique finite numbers")

    @property
    def points(self) -> int:
        """Number of geometries in the Cartesian product of the axes."""
        return math.prod(len(values) for values in self.grid.values())

    def run(self) -> Dataset:
        """Run the generator, reusing the completed Palace geometry directories.

        Returns:
            The complete dataset written by the generator.
        """
        module, function = self.generator.split(":")
        generate = getattr(importlib.import_module(module), function)
        runner = Palace(
            workdir=self.workdir, command=self.command, settings=self.settings
        )
        return generate(runner=runner, output=self.output, grid=self.grid)


def load_recipe(path: Path | str) -> GenerationRecipe:
    """Read a TOML recipe, resolving output and work directories beside the file.

    Returns:
        Validated generation inputs. Neither Palace nor the generator is run.

    Raises:
        ValueError: If the recipe has missing, unknown or invalid fields.
        TypeError: If a field has the wrong type.
    """
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    allowed = {"generator", "grid", "output", "workdir", "palace"}
    if missing := {"generator", "grid", "output", "workdir"} - config.keys():
        raise ValueError(f"Missing recipe fields: {sorted(missing)}")
    if unknown := config.keys() - allowed:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    palace = config.get("palace", {})
    if not isinstance(palace, dict):
        raise TypeError("palace must be a TOML table")
    command = palace.get("command", ["palace", "-np", "4"])
    if not isinstance(command, list):
        raise TypeError("palace.command must be a nonempty array of strings")
    if unknown := palace.keys() - {"command", "settings"}:
        raise ValueError(f"Unknown palace fields: {sorted(unknown)}")
    return GenerationRecipe(
        generator=config["generator"],
        grid=config["grid"],
        output=(path.parent / config["output"]).resolve(),
        workdir=(path.parent / config["workdir"]).resolve(),
        command=tuple(command),
        settings=ElectrostaticSettings(**palace.get("settings", {})),
    )


def main() -> None:
    """Preview or run a recipe with optional output and runtime overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="TOML generation recipe")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without solving"
    )
    parser.add_argument("--output", type=Path, help="Override output, relative to cwd")
    parser.add_argument(
        "--workdir", type=Path, help="Override workdir, relative to cwd"
    )
    parser.add_argument("--palace-command", help="Override the MPI/container command")
    args = parser.parse_args()
    try:
        recipe = load_recipe(args.recipe)
        recipe = replace(
            recipe,
            output=args.output.resolve() if args.output is not None else recipe.output,
            workdir=args.workdir.resolve()
            if args.workdir is not None
            else recipe.workdir,
            command=tuple(shlex.split(args.palace_command))
            if args.palace_command is not None
            else recipe.command,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    logger.info(
        f"{recipe.generator}: {recipe.points} geometries\n"
        f"grid: {recipe.grid}\noutput: {recipe.output}\nworkdir: {recipe.workdir}\n"
        f"command: {shlex.join(recipe.command)}\n"
        f"settings: {json.dumps(asdict(recipe.settings), sort_keys=True)}"
    )
    if not args.dry_run:
        recipe.run()


if __name__ == "__main__":
    main()
