"""Store path."""

__all__ = ["PATH"]

import pathlib
from dataclasses import dataclass
from typing import ClassVar, final

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent


@final
@dataclass
class Path:
    """Creates object for referencing paths in repository."""

    module: ClassVar[pathlib.Path] = module
    repo: ClassVar[pathlib.Path] = repo
    build: ClassVar[pathlib.Path] = repo / "build"
    docs: ClassVar[pathlib.Path] = repo / "docs"
    gds: ClassVar[pathlib.Path] = build / "gds"
    simulation: ClassVar[pathlib.Path] = build / "simulation"
    tests: ClassVar[pathlib.Path] = repo / "tests"

    cells: ClassVar[pathlib.Path] = module / "cells"
    derived: ClassVar[pathlib.Path] = module / "derived"
    klayout: ClassVar[pathlib.Path] = module / "klayout"
    models: ClassVar[pathlib.Path] = module / "models"
    samples: ClassVar[pathlib.Path] = module / "samples"

    lyp: ClassVar[pathlib.Path] = klayout / "layers.lyp"
    lyt: ClassVar[pathlib.Path] = klayout / "tech.lyt"
    lyp_yaml: ClassVar[pathlib.Path] = module / "layers.yaml"
    tech: ClassVar[pathlib.Path] = module / "klayout" / "tech"


PATH = Path()
