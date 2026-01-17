import inspect
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import qpdk
from qpdk.config import PATH

filepath_models = PATH.docs / "models.rst"
template_dir = PATH.docs / "templates"

# Collect all public functions/classes in qpdk.models
models = {
    name: obj
    for name, obj in qpdk.models.__dict__.items()
    if callable(obj) and not name.startswith("_")
}

# Prepare items for the template
items = [
    {
        "title": "Models",
        "automodule": "qpdk.models",
        "synopsis": "Code for S-parameter and other modelling",
        "members": True,
        "undoc_members": True,
        "show_inheritance": True,
        "functions": sorted(models.keys()),
    },
    {
        "title": "References",
        "bibliography_filter": "docname in docnames",
    },
]

# Setup Jinja2
env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
template = env.get_template("models_static.rst.j2")

rendered = template.render(items=items)

with filepath_models.open("w") as f:
    f.write(rendered)
