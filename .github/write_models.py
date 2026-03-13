"""Write model documentation."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

import qpdk.models
from qpdk.config import PATH

filepath_models = PATH.docs / "models.rst"
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

# Collect all public functions/classes in qpdk.models
if qpdk.models is not None:
    models = {
        name: obj
        for name, obj in qpdk.models.__dict__.items()
        if callable(obj) and not name.startswith("_") and name not in skip_autodoc
    }
else:
    models = {}

# SAX models (functions that return S-parameter dictionaries)
sax_model_names = set(qpdk.models.models.keys())

# Prepare items for the template
items = [
    {
        "title": "Models",
        "automodule": "qpdk.models",
        "synopsis": "Code for S-parameter and other modelling",
        "members": True,
        "undoc_members": True,
        "show_inheritance": True,
        "functions": sorted(models.keys()) if models else [],  # preserves order
        "skip_plots": skip_plots,
        "sax_models": sax_model_names,
    },
    {
        "title": "References",
        "bibliography_filter": "docname in docnames",
    },
]

# Setup Jinja2
env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())
template = env.get_template("models_static.rst.j2")

rendered = template.render(items=items)

with filepath_models.open("w") as f:
    f.write(rendered)
