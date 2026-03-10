"""Store path."""

__all__ = ["PATH"]

import pathlib
from dataclasses import dataclass
from typing import final

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent


@final
@dataclass
class Path:
    """Creates object for referencing paths in repository."""

    module = module
    repo = repo
    build = repo / "build"
    docs = repo / "docs"
    gds = module / "gds"
    simulation = build / "simulation"
    tests = repo / "tests"

    cells = module / "cells"
    derived = module / "derived"
    klayout = module / "klayout"
    models = module / "models"
    samples = module / "samples"

    lyp = klayout / "layers.lyp"
    lyt = klayout / "tech.lyt"
    lyp_yaml = module / "layers.yaml"
    tech = module / "klayout" / "tech"


PATH = Path()
