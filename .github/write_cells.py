"""Write docs."""

# ruff: noqa: S701, T201

import inspect
from pathlib import Path

from gdsfactory.serialization import clean_value_json
from jinja2 import Environment, FileSystemLoader

import qpdk
from qpdk.config import PATH

filepath_cells = PATH.docs / "cells.rst"
filepath_gallery = PATH.docs / "gallery.rst"
filepath_samples = PATH.docs / "samples.rst"
template_dir = PATH.docs / "templates"

skip = {}

skip_plot: set[str] = {"transform_component"}
skip_settings: set[str] = set()

qpdk.PDK.activate()
cells = qpdk.PDK.cells
samples = qpdk.get_sample_functions()

# Set up Jinja2 environment
# Note: autoescape is False because we're generating RST, not HTML
env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)


def get_kwargs(sig: inspect.Signature) -> str:
    """Extract kwargs from function signature.

    Returns:
        String of comma-separated keyword arguments.
    """
    return ", ".join([
        f"{p}={clean_value_json(sig.parameters[p].default)!r}"
        for p in sig.parameters
        if isinstance(sig.parameters[p].default, int | float | str | tuple)
        and p not in skip_settings
    ])


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

Path(filepath_cells).write_text(rendered, encoding="utf-8")

# Generate gallery.rst
template = env.get_template("gallery.rst.j2")
rendered = template.render(items=cells_items, skip_plot=skip_plot)

Path(filepath_gallery).write_text(rendered, encoding="utf-8")

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

Path(filepath_samples).write_text(rendered, encoding="utf-8")
