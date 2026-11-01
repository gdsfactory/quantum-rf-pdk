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
sample_gds_dir = gds_dir / "samples"
sample_png_dir = PATH.docs / "_generated" / "samples"

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

# Sections of the Samples page, in display order.  Samples are matched by their
# short name (the function name, or the file stem for schematic-only samples);
# a sample in none of them ends up under "Other samples" and the script prints
# a warning.
SAMPLE_SECTIONS: list[dict] = [
    {
        "key": "getting-started",
        "title": "Getting Started",
        "description": "Step-by-step primers on gdsfactory basics with QPDK cells.",
        "samples": [
            "sample0_hello_world",
            "sample1_connect",
            "sample2_remove_layers",
            "sample3_grid",
            "sample4_pack",
            "sample5_path",
            "sample6_cross_section",
            "all_cells",
        ],
    },
    {
        "key": "qubit-chips",
        "title": "Qubit Test Chips",
        "description": "Transmon, flipmon and gatemon chips with readout resonators.",
        "samples": ["qubit_test_chip", "flipmon_test_chip", "gatemon_test_chip"],
    },
    {
        "key": "resonator-chips",
        "title": "Resonator and Lattice Chips",
        "description": "Resonator test chips and resonator lattices.",
        "samples": [
            "resonator_test_chip_python",
            "resonator_test_chip_yaml",
            "hyperbolic_lattice_chip",
        ],
    },
    {
        "key": "3d-integration",
        "title": "3D Integration",
        "description": "Flip-chip, TSV and 3D layer stack examples.",
        "samples": ["tsv_flip_chip_test_chip", "sample_to_3d"],
    },
    {
        "key": "metal-fill",
        "title": "Metal Fill",
        "description": "Chips with ground-plane fill and flux-trapping holes.",
        "samples": [
            "filled_quarter_wave_resonator",
            "filled_qubit_test_chip",
            "filled_flipmon_test_chip",
            "filled_gatemon_test_chip",
            "filled_resonator_test_chip",
        ],
    },
    {
        "key": "simulation",
        "title": "Simulation",
        "description": "Layouts prepared for electromagnetic simulation.",
        "samples": ["resonator_simulation"],
    },
]
OTHER_SAMPLE_SECTION = {
    "key": "other",
    "title": "Other Samples",
    "description": "",
    "samples": [],
}

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


def group_samples(items: list[dict]) -> list[dict]:
    """Group sample items into the sections defined in ``SAMPLE_SECTIONS``.

    Returns:
        Non-empty sections, each with its ``samples`` in ``SAMPLE_SECTIONS`` order.

    Raises:
        ValueError: If two samples share a short name.
    """
    by_short_name = {item["short_name"]: item for item in items}
    if len(by_short_name) != len(items):
        # Short names key the section list and the GDS/PNG file names.
        msg = "two samples share a short name; rename one of them"
        raise ValueError(msg)
    grouped = []
    for section in SAMPLE_SECTIONS:
        section_items = [
            by_short_name.pop(name)
            for name in section["samples"]
            if name in by_short_name
        ]
        if section_items:
            grouped.append({
                **section,
                "anchor": f"samples-{section['key']}",
                "samples": section_items,
            })
    if by_short_name:
        print(
            f"WARNING: samples {sorted(by_short_name)} are not in any section; "
            "add them to SAMPLE_SECTIONS in .github/write_cells.py"
        )
        grouped.append({
            **OTHER_SAMPLE_SECTION,
            "anchor": f"samples-{OTHER_SAMPLE_SECTION['key']}",
            "samples": [by_short_name[name] for name in sorted(by_short_name)],
        })
    return grouped


def _setup_kwasm_viewer() -> None:
    """Create the kwasm viewer HTML and GDS output directory.

    The QPDK layer properties are embedded in the viewer so the Dynamic tab
    shows the same layer names and colours as KLayout and GDSFactory+.
    """
    for directory in (gds_dir, png_dir, sample_gds_dir, sample_png_dir):
        directory.mkdir(parents=True, exist_ok=True)
    viewer_path = kwasm_dir / "viewer.html"
    lyp_b64 = base64.b64encode(PATH.lyp.read_bytes()).decode("ascii")
    template = kwasm.embed._read_artifacts()
    template = template.replace("KWASM_GDS_B64", "")
    template = template.replace("KWASM_LYP_B64", lyp_b64)
    template = template.replace("KWASM_LYRDB_B64", "")
    template = template.replace("KWASM_NETLIST_B64", "")
    viewer_path.write_text(template, encoding="utf-8")


def _generate_artifacts(
    c, name: str, gds_out: Path = gds_dir, png_out: Path = png_dir
) -> None:
    """Generate GDS and PNG plot for the component."""
    c.draw_ports()
    c.write_gds(gds_out / f"{name}.gds")
    fig, ax = plt.subplots()
    c.plot(ax=ax)
    fig.savefig(png_out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_gds(
    name: str, factory, gds_out: Path = gds_dir, png_out: Path = png_dir
) -> bool:
    """Write GDS and PNG for a cell or sample.

    Returns:
        True if files were written successfully, False otherwise.
    """
    sig = inspect.signature(factory)
    kwargs = {
        p: sig.parameters[p].default
        for p in sig.parameters
        if isinstance(sig.parameters[p].default, int | float | str | tuple)
        and p not in skip_settings
    }
    try:
        c = factory(**kwargs).copy()
        _generate_artifacts(c, name, gds_out, png_out)
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
    has_gds = name not in skip_plot and _write_gds(name, cells[name])
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
    import_path, _, short_name = name.rpartition(".")
    has_gds = name not in skip_plot and _write_gds(
        short_name, samples[name], sample_gds_dir, sample_png_dir
    )
    samples_items.append({
        "name": name,
        "short_name": short_name,
        "kwargs": kwargs,
        "import_path": import_path,
        "has_gds": has_gds,
        "schematic": False,
    })

# YAML/schematic samples without a Python script of the same name
for path in sorted(PATH.samples.glob("*.pic.yml")):
    name = path.name.removesuffix(".pic.yml")
    if path.with_name(f"{name}.py").exists():
        continue
    print(name)
    has_gds = _write_gds(
        name,
        functools.partial(gf.get_component, name),
        sample_gds_dir,
        sample_png_dir,
    )
    samples_items.append({
        "name": name,
        "short_name": name,
        "has_gds": has_gds,
        "schematic": True,
    })

template = env.get_template("samples.rst.j2")
rendered = template.render(sections=group_samples(samples_items), skip_plot=skip_plot)

Path(filepath_samples).write_text(rendered, encoding="utf-8")
