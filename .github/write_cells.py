"""Write docs."""

# ruff: file-ignore[jinja2-autoescape-false, print]

import base64
import functools
import inspect
import traceback
from pathlib import Path

import gdsfactory as gf
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

# Lives under ``docs/_extra`` which is declared as ``html_extra_path`` in
# ``docs/conf.py``, so its contents are copied verbatim to the HTML output root
# as ``kwasm/``.  Sphinx only copies files it knows about (``.. image::``
# targets and the like), so the viewer HTML and the GDS files it fetches would
# otherwise never reach the built site and the Dynamic tab would 404.
kwasm_dir = PATH.docs / "_extra" / "kwasm"
gds_dir = kwasm_dir / "gds"
# Static previews are referenced with ``.. image::`` so Sphinx copies them into
# ``_images/`` itself; keeping them out of ``_extra`` avoids shipping a second
# copy of every preview.
png_dir = PATH.docs / "_generated" / "cells"

skip = {}

# Sections of the PCells page, in display order.  Each cell is listed once,
# under the first section whose tag it carries; its other tags are shown as
# badges.  ``subsections`` are tags that get their own sub-heading inside the
# section, again first match wins.  Cells carrying none of the section tags end
# up under "Other components" and the script prints a warning.
SECTIONS: list[dict] = [
    {"tag": "qubits", "title": "Qubits", "subsections": ["transmons"]},
    {"tag": "resonators", "title": "Resonators", "subsections": []},
    {"tag": "junctions", "title": "Josephson Junctions", "subsections": []},
    {"tag": "couplers", "title": "Couplers", "subsections": []},
    {"tag": "capacitors", "title": "Capacitors", "subsections": []},
    {"tag": "inductors", "title": "Inductors", "subsections": []},
    {"tag": "detectors", "title": "Detectors", "subsections": []},
    {
        "tag": "interconnects",
        "title": "Interconnects",
        "subsections": ["3d-integration"],
    },
    {"tag": "terminations", "title": "Terminations", "subsections": []},
    {"tag": "waveguides", "title": "Waveguides", "subsections": ["bend"]},
    {"tag": "chip", "title": "Chip", "subsections": []},
]
SUBSECTION_TITLES = {
    "transmons": "Transmons",
    "3d-integration": "3D Integration",
    "bend": "Bends",
}
OTHER_SECTION = {"tag": "other", "title": "Other Components", "subsections": []}

skip_plot: set[str] = {"transform_component"}
skip_settings: set[str] = set()

qpdk.PDK.activate()
cells = {
    name: factory
    for name, factory in qpdk.PDK.cells.items()
    if getattr(getattr(factory, "func", factory), "__module__", "").startswith(
        "qpdk.cells"
    )
}
samples = qpdk.get_sample_functions()

# ``@gf.cell(tags=...)`` stores the tags on the kfactory wrapper, not on the
# function the PDK hands out, so map the undecorated function back to them.
_tags_by_function = {
    id(wrapped._f_orig): sorted(wrapped.tags)
    for factories in (gf.kcl.factories, gf.kcl.virtual_factories)
    for wrapped in factories._all
}


def cell_tags(factory) -> list[str]:
    """Return the tags of a cell, following ``functools.partial`` to its base cell.

    Returns:
        Sorted list of tags, empty if the cell has none.
    """
    while isinstance(factory, functools.partial):
        factory = factory.func
    return _tags_by_function.get(id(inspect.unwrap(factory)), [])


def _subsection_title(tag: str) -> str:
    """Return the heading for a subsection tag, title-casing tags missing a title.

    Returns:
        Heading text such as ``"Transmons"`` or ``"Flip Chip"``.
    """
    return SUBSECTION_TITLES.get(tag) or tag.replace("-", " ").title()


def group_by_section(items: list[dict]) -> list[dict]:
    """Group cell items into the sections and subsections defined in ``SECTIONS``.

    Returns:
        Non-empty sections, each with its ``groups``: a list of
        ``{"title", "cells"}`` where a ``None`` title means items listed directly
        under the section heading.
    """
    sections = [*SECTIONS, OTHER_SECTION]
    buckets: dict[str, list[dict]] = {section["tag"]: [] for section in sections}
    for item in items:
        tag = next(
            (s["tag"] for s in SECTIONS if s["tag"] in item["tags"]),
            OTHER_SECTION["tag"],
        )
        if tag == OTHER_SECTION["tag"]:
            print(f"WARNING: cell {item['name']!r} has no section tag: {item['tags']}")
        buckets[tag].append(item)

    grouped = []
    for section in sections:
        section_items = buckets[section["tag"]]
        if not section_items:
            continue
        groups = []
        remaining = section_items
        for subtag in section["subsections"]:
            matched = [item for item in remaining if subtag in item["tags"]]
            remaining = [item for item in remaining if subtag not in item["tags"]]
            if matched:
                groups.append({"title": _subsection_title(subtag), "cells": matched})
        if remaining:
            # Headings must nest consistently, so once a section has
            # sub-headings its leftover cells need one too.
            title = f"Other {section['title']}" if groups else None
            groups.append({"title": title, "cells": remaining})
        grouped.append({
            **section,
            "anchor": f"cells-{section['tag']}",
            "cells": section_items,
            "groups": groups,
        })
    return grouped


# Set up Jinja2 environment
# Note: autoescape is False because we're generating RST, not HTML
env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)


def _setup_kwasm_viewer() -> None:
    """Create the kwasm viewer HTML and GDS output directory.

    The QPDK layer properties are embedded in the viewer so the Dynamic tab
    shows the same layer names and colours as KLayout and GDSFactory+.
    """
    gds_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = kwasm_dir / "viewer.html"
    lyp_b64 = base64.b64encode(PATH.lyp.read_bytes()).decode("ascii")
    template = kwasm.embed._read_artifacts()
    template = template.replace("KWASM_GDS_B64", "")
    template = template.replace("KWASM_LYP_B64", lyp_b64)
    template = template.replace("KWASM_LYRDB_B64", "")
    template = template.replace("KWASM_NETLIST_B64", "")
    viewer_path.write_text(template, encoding="utf-8")


def _generate_artifacts(c, name: str) -> None:
    """Generate GDS and PNG plot for the component."""
    c.draw_ports()
    c.write_gds(gds_dir / f"{name}.gds")
    fig, ax = plt.subplots()
    c.plot(ax=ax)
    fig.savefig(png_dir / f"{name}.png", dpi=150, bbox_inches="tight")
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
    cells_items.append({
        "name": name,
        "kwargs": kwargs,
        "has_gds": has_gds,
        "tags": cell_tags(cells[name]),
    })

template = env.get_template("cells.rst.j2")
rendered = template.render(sections=group_by_section(cells_items), skip_plot=skip_plot)

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
