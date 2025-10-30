"""Write docs."""

import inspect
from pathlib import Path

from gdsfactory.serialization import clean_value_json
from jinja2 import Environment, FileSystemLoader

import qpdk
from qpdk import PDK
from qpdk.config import PATH

filepath_cells = PATH.repo / "docs" / "cells.rst"
filepath_samples = PATH.repo / "docs" / "samples.rst"
template_dir = PATH.repo / "docs" / "templates"

skip = {}

skip_plot: set[str] = {"transform_component"}
skip_settings: set[str] = set()

PDK.activate()
cells = PDK.cells
samples = qpdk.sample_functions

# Set up Jinja2 environment
# Note: autoescape is False because we're generating RST, not HTML
env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)


def get_kwargs(sig: inspect.Signature) -> str:
    """Extract kwargs from function signature."""
    return ", ".join(
        [
            f"{p}={clean_value_json(sig.parameters[p].default)!r}"
            for p in sig.parameters
            if isinstance(sig.parameters[p].default, int | float | str | tuple)
            and p not in skip_settings
        ]
    )


# Generate cells.rst
cells_items = []
for name in sorted(cells.keys()):
    if name in skip or name.startswith("_"):
        continue
    print(name)
    sig = inspect.signature(cells[name])
    kwargs = get_kwargs(sig)
    cells_items.append({"name": name, "kwargs": kwargs})

template = env.get_template("cells.rst.j2")
rendered = template.render(items=cells_items, skip_plot=skip_plot)

with Path(filepath_cells).open("w") as f:
    f.write(rendered)

# Generate samples.rst
samples_items = []
for name in sorted(samples.keys()):
    if name in skip or name.startswith("_"):
        continue
    print(name)
    sig = inspect.signature(samples[name])
    kwargs = get_kwargs(sig)
    import_path = name.rpartition(".")[0]
    samples_items.append({"name": name, "kwargs": kwargs, "import_path": import_path})

template = env.get_template("samples.rst.j2")
rendered = template.render(items=samples_items, skip_plot=skip_plot)

with Path(filepath_samples).open("w") as f:
    f.write(rendered)
