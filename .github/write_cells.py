"""Write docs."""

# ruff: noqa: S701, T201

import inspect
import traceback
from pathlib import Path

import kwasm.embed
import matplotlib as mpl
import matplotlib.pyplot as plt
from gdsfactory.serialization import clean_value_json
from jinja2 import Environment, FileSystemLoader

import qpdk
from qpdk.config import PATH

mpl.use("Agg")

filepath_cells = PATH.docs / "cells.rst"
filepath_samples = PATH.docs / "samples.rst"
template_dir = PATH.docs / "templates"

kwasm_dir = PATH.docs / "kwasm"
gds_dir = kwasm_dir / "gds"

skip = {}

skip_plot: set[str] = {"transform_component"}
skip_settings: set[str] = set()

qpdk.PDK.activate()
cells = qpdk.PDK.cells
samples = qpdk.get_sample_functions()

# Set up Jinja2 environment
# Note: autoescape is False because we're generating RST, not HTML
env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)


def _setup_kwasm_viewer() -> None:
    """Create the kwasm viewer HTML and GDS output directory."""
    gds_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = kwasm_dir / "viewer.html"
    if viewer_path.exists():
        return
    template = kwasm.embed._read_artifacts()
    template = template.replace("KWASM_GDS_B64", "")
    template = template.replace("KWASM_LYP_B64", "")
    template = template.replace("KWASM_LYRDB_B64", "")
    template = template.replace("KWASM_NETLIST_B64", "")
    viewer_path.write_text(template)


def _generate_artifacts(c, name: str) -> None:
    """Generate GDS and PNG plot for the component."""
    c.draw_ports()
    c.write_gds(gds_dir / f"{name}.gds")
    fig, ax = plt.subplots()
    c.plot(ax=ax)
    fig.savefig(gds_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_gds(name: str) -> bool:
    """Write GDS and PNG for a cell.

    Returns:
        True if files were written successfully, False otherwise.
    """
    sig = inspect.signature(cells[name])
    kwargs = {
        p: sig.parameters[p].default
        for p in sig.parameters
        if isinstance(sig.parameters[p].default, int | float | str | tuple)
        and p not in skip_settings
    }
    try:
        c = cells[name](**kwargs).copy()
        _generate_artifacts(c, name)
    except Exception:
        traceback.print_exc()
        return False
    else:
        return True


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


_setup_kwasm_viewer()

# Generate cells.rst
cells_items = []
for name in sorted(cells.keys()):
    if name in skip or name.startswith("_"):
        continue
    print(name)
    sig = inspect.signature(cells[name])
    kwargs = get_kwargs(sig)
    has_gds = name not in skip_plot and _write_gds(name)
    cells_items.append({"name": name, "kwargs": kwargs, "has_gds": has_gds})

template = env.get_template("cells.rst.j2")
rendered = template.render(items=cells_items, skip_plot=skip_plot)

Path(filepath_cells).write_text(rendered, encoding="utf-8")

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
