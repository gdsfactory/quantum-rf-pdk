"""Write model documentation."""

import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files

import gdsfactory as gf
from jinja2 import Environment, FileSystemLoader, select_autoescape

import qpdk.models
from qpdk import PDK
from qpdk.config import PATH

filepath_models = PATH.docs / "models.rst"
# Staged inside the docs tree: Typst refuses image paths outside its root.
symbol_dir = PATH.docs / "_generated" / "symbols"
symbol_package = "qpdk.symbols"
# The artwork switches to light ink through prefers-color-scheme, which an
# <img> resolves from the OS rather than the docs theme toggle.
symbol_dark_query = re.compile(r"@media \(prefers-color-scheme:dark\)\{(.*)\}</style>")
template_dir = PATH.docs / "templates"

# Models that should NOT be plotted
skip_plots = {
    "gamma_0_load",
    "launcher",
    "resonator_frequency",
    "MediaCallable",
    "cross_section_to_media",
    "cpw_cpw_coupling_capacitance",
    "cpw_media_skrf",
}

# Models that should NOT be documented at all (re-exported from other packages)
skip_autodoc = {
    "admittance",
    "capacitor",
    "electrical_open",
    "electrical_short",
    "gamma_0_load",
    "impedance",
    "inductor",
    "tee",
}


@dataclass
class Category:
    """A section of the models page, grouping functions by defining module."""

    key: str
    title: str
    description: str
    modules: tuple[str, ...]
    functions: list[str] = field(default_factory=list)


# SAX models (functions that return S-parameter dictionaries), grouped by module
sax_categories = [
    Category(
        "qubits",
        "Qubits",
        "Transmon, fluxonium, flipmon, unimon and gatemon circuits, bare or coupled to a resonator.",
        ("qpdk.models.qubit", "qpdk.models.unimon", "qpdk.models.gatemon"),
    ),
    Category(
        "resonators",
        "Resonators",
        "Quarter- and half-wave transmission-line resonators and their coupled variants.",
        ("qpdk.models.resonator",),
    ),
    Category(
        "couplers",
        "Couplers",
        "Straight and ring couplers between CPW lines.",
        ("qpdk.models.couplers",),
    ),
    Category(
        "lumped",
        "Capacitors and inductors",
        "Interdigital and plate capacitors, meander inductors and lumped-element and nanowire resonators.",
        ("qpdk.models.capacitor", "qpdk.models.inductor", "qpdk.models.nanowire"),
    ),
    Category(
        "junctions",
        "Josephson junctions",
        "Single junctions and SQUIDs.",
        ("qpdk.models.junction",),
    ),
    Category(
        "waveguides",
        "Waveguides and interconnects",
        "Straights, bends, tapers, launchers, airbridges, bumps, TSVs, rectangles and generic N-port blocks.",
        ("qpdk.models.waveguides",),
    ),
    Category(
        "generic",
        "Generic circuit elements",
        "Opens, shorts, series and shunt elements and ideal LC resonators.",
        ("qpdk.models.generic",),
    ),
]

# Helpers (analytical formulas, parameter conversions, Hamiltonians), grouped by module
helper_categories = [
    Category(
        "transmission-lines",
        "Transmission-line parameters",
        "Effective permittivity, characteristic impedance and propagation of CPW and microstrip lines.",
        ("qpdk.models.cpw", "sax.models.rf"),
    ),
    Category(
        "qubit-parameters",
        "Qubit parameters",
        "Conversions between circuit energies, capacitances and inductances, "
        "and unimon and gatemon spectra.",
        ("qpdk.models.qubit", "qpdk.models.unimon", "qpdk.models.gatemon"),
    ),
    Category(
        "perturbation",
        "Perturbation theory",
        "Transmon frequency and anharmonicity, dispersive shifts, Purcell decay, linewidths and dephasing.",
        ("qpdk.models.perturbation",),
    ),
    Category(
        "component-helpers",
        "Component helpers",
        "Closed-form estimates for resonators, couplers and inductors.",
        (
            "qpdk.models.resonator",
            "qpdk.models.couplers",
            "qpdk.models.inductor",
            "qpdk.models.nanowire",
        ),
    ),
]

# Collect all public functions/classes in qpdk.models
models = {
    name: obj
    for name, obj in qpdk.models.__dict__.items()
    if callable(obj) and not name.startswith("_") and name not in skip_autodoc
}

sax_model_names = set(qpdk.models.models.keys())


def _stage_symbol(filename: str) -> None:
    """Copy a symbol into ``symbol_dir`` as a light and a ``_dark`` variant.

    The docs theme extension pairs ``<stem>.svg`` with ``<stem>_dark.svg`` and
    shows whichever matches the page theme, so each copy gets one fixed ink.

    Raises:
        ValueError: If the artwork has no ``prefers-color-scheme`` block.
    """
    svg = (files(symbol_package) / filename).read_text(encoding="utf-8")
    match = symbol_dark_query.search(svg)
    if match is None:
        raise ValueError(f"{filename}: no prefers-color-scheme:dark block to split")
    light = svg.replace(match.group(0), "</style>")
    dark = light.replace("</style>", f"{match.group(1)}</style>")
    stem = filename.removesuffix(".svg")
    (symbol_dir / filename).write_text(light, encoding="utf-8")
    (symbol_dir / f"{stem}_dark.svg").write_text(dark, encoding="utf-8")


def _model_symbols() -> dict[str, list[str]]:
    """Map each SAX model to the schematic symbols GDSFactory+ draws for it.

    GDSFactory+ links a cell to its models through the ``models`` entry of the
    cell's schematic and draws the artwork named in ``display.symbol``, so the
    same link is followed here. Only QPDK's own artwork is used, and it is
    staged into ``symbol_dir``.

    Returns:
        Model name to sorted symbol paths, relative to the docs source directory.
    """
    PDK.activate()
    shutil.rmtree(symbol_dir, ignore_errors=True)
    symbol_dir.mkdir(parents=True)
    symbols: defaultdict[str, set[str]] = defaultdict(set)
    # Virtual (all-angle) factories carry no schematic API.
    for factory in gf.kcl.factories.values():
        if not factory.schematic_driven():
            continue
        info = factory.get_schematic().info
        package, _, filename = info.get("display", {}).get("symbol", "").partition(":")
        if package != symbol_package:
            continue
        names = {m["name"] for m in info.get("models", [])} & sax_model_names
        if not names:
            continue
        _stage_symbol(filename)
        for name in names:
            symbols[name].add(f"_generated/symbols/{filename}")
    return {name: sorted(paths) for name, paths in symbols.items()}


def _assign(categories: list[Category], names: list[str], group: str) -> list[Category]:
    """Sort ``names`` into ``categories`` by module; unmatched ones go to "Other"."""
    by_module: dict[str, Category] = {
        module: category for category in categories for module in category.modules
    }
    other = Category(f"{group}-other", "Other", "", ())
    for name in sorted(names):
        module = getattr(models[name], "__module__", "")
        by_module.get(module, other).functions.append(name)
    if other.functions:
        warnings.warn(
            f"{other.functions} are not in any {group} category; "
            "add their module to a Category in .github/write_models.py",
            stacklevel=2,
        )
    return [c for c in [*categories, other] if c.functions]


sax_sections = _assign(
    sax_categories, [n for n in models if n in sax_model_names], "sax"
)
helper_sections = _assign(
    helper_categories, [n for n in models if n not in sax_model_names], "helpers"
)

# Setup Jinja2
env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())
template = env.get_template("models_static.rst.j2")

rendered = template.render(
    sections=[
        {
            "key": "sax",
            "title": "SAX models",
            "intro": "Frequency-domain models returning S-parameter dictionaries "
            "(:class:`sax.SDict`). Pass them to :func:`sax.circuit` to simulate "
            "netlists, or call them directly with a frequency array ``f``.",
            "categories": sax_sections,
        },
        {
            "key": "helpers",
            "title": "Helper functions",
            "intro": "Analytical formulas, parameter conversions and Hamiltonians "
            "used to build and design with the SAX models above.",
            "categories": helper_sections,
        },
    ],
    skip_plots=skip_plots,
    sax_models=sax_model_names,
    model_symbols=_model_symbols(),
)

with filepath_models.open("w", encoding="utf-8") as f:
    f.write(rendered)
