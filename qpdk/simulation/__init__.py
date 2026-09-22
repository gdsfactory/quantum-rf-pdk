"""Simulation utilities: shared layout preparation and solver inputs.

:func:`~qpdk.simulation.layout.prepare_metal_layout` folds a QPDK mask into
positive metal geometry for a field solver. This package also holds
QPDK-specific inputs for :mod:`gplugins.comsol`.

**Ansys AEDT (HFSS, Q3D Extractor, Q2D):** use :mod:`gplugins.ansys` directly.
Activate QPDK first (``qpdk.PDK.activate()``); importing :mod:`qpdk` does not.
Pass ``layer_stack=qpdk.tech.LAYER_STACK`` explicitly. The PDK default is a
mask preview with an implicit ground plane, not a simulation stack. Also pass
``material_properties=qpdk.tech.material_properties`` to use QPDK's material
table instead of the gplugins default:

1. Prepare a component with :func:`~qpdk.simulation.layout.prepare_metal_layout`
2. Wrap a PyAEDT ``Hfss``, ``Q3d`` or ``Q2d`` app in ``gplugins.ansys.HFSS``,
   ``Q3D`` or ``Q2D`` and import the component with ``import_component``
3. Configure the setup through PyAEDT, solve, and read the results through the
   gplugins helpers. For a CPW cross-section in Q2D, pass the width and gap from
   :func:`~qpdk.models.cpw.get_cpw_dimensions` to ``Q2D.create_2d_cpw`` with
   ``substrate_level="Substrate"`` and ``conductor_level="M1"``.

**Palace workflow:**

Run ``uv run --script datasets/plate_capacitor.py`` from a checkout. The script
uses gsim's electrostatic APIs and writes a self-describing dataset for lookup.

**COMSOL workflow:**

The COMSOL models come from :mod:`gplugins.comsol`; import them from there.
:mod:`qpdk.simulation.comsol` adds only the QPDK-specific inputs:

1. Extract metal polygons and optional feed ports with
   :func:`qpdk.simulation.comsol.prepare_comsol_layout`
2. Build a model with :class:`gplugins.comsol.COMSOL`, passing
   :data:`~qpdk.simulation.comsol.SILICON_RELATIVE_PERMITTIVITY` and
   :data:`~qpdk.simulation.comsol.AIR_RELATIVE_PERMITTIVITY` for the sheet
   dielectrics, then add a study and mesh it with its methods

Note:
    :mod:`gplugins.ansys` requires ``uv sync --extra hfss``. The COMSOL module
    requires ``uv sync --extra comsol`` and, to build and solve models, a local
    COMSOL installation. It is not imported by this package, so
    :mod:`qpdk.simulation` stays importable without the extra.

    The VACASK netlist classes, runner and raw-file reader need only the base
    install and the external ``vacask`` binary. The model-conversion helpers,
    :func:`~qpdk.simulation.vacask.vector_fit` and
    :func:`~qpdk.simulation.vacask.cpw_tline_params`, also need
    ``uv sync --extra models``.

**VACASK workflow:**

1. Write the circuit and its analyses in VACASK's own netlist language and
   pass the blocks to a :class:`~qpdk.simulation.vacask.Netlist`, which fills
   them into a Jinja2 template. Loading ``josephson_junction.va`` or
   ``squid.va`` uses the models shipped with qpdk.
2. Run it with :meth:`~qpdk.simulation.vacask.Vacask.run`, which calls the
   external ``vacask`` binary and returns one
   :class:`~qpdk.simulation.vacask.RawPlot` per analysis.
3. Use :meth:`~qpdk.simulation.vacask.Vacask.run_record` with a work directory
   to retain decks, logs, raw results and input/binary/compiler hashes.
4. Optionally embed any linear SAX model as a subcircuit: fit it with
   :func:`~qpdk.simulation.vacask.vector_fit` and emit the fit with
   :func:`~qpdk.simulation.vacask.vector_fit_subckt`.

Example:
    >>> from ansys.aedt.core import Hfss
    >>> from gplugins.ansys import HFSS
    >>> from qpdk import PDK
    >>> from qpdk.cells import resonator
    >>> from qpdk.simulation.layout import prepare_metal_layout
    >>> from qpdk.tech import LAYER_STACK, material_properties
    >>> PDK.activate()
    >>> comp = prepare_metal_layout(resonator(length=4000, meanders=4))
    >>> hfss_app = Hfss(project="resonator_sim", solution_type="Eigenmode")
    >>> hfss_sim = HFSS(
    ...     hfss_app, layer_stack=LAYER_STACK, material_properties=material_properties
    ... )
    >>> hfss_sim.import_component(comp)

References:
    - PyAEDT documentation: https://aedt.docs.pyansys.com/
    - HFSS import_gds_3d: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.import_gds_3d.html
    - Q3D Extractor: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.q3d.Q3d.html
    - MPh: https://github.com/MPh-py/MPh
"""

# ruff: file-ignore[undefined-export]

import importlib
from typing import Any

# TODO(Python 3.15): `lazy from ... import` (PEP 810) - replaces _LAZY_IMPORTS, __getattr__,
# __dir__ and the undefined-export ignore. Keep them lazy, not eager.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FEM_LAYERS": ("qpdk.simulation.fem", "FEM_LAYERS"),
    "single_chip_stack": ("qpdk.simulation.fem", "single_chip_stack"),
    "to_fem_regions": ("qpdk.simulation.fem", "to_fem_regions"),
    "Netlist": ("qpdk.simulation.vacask", "Netlist"),
    "RawFile": ("qpdk.simulation.vacask", "RawFile"),
    "RawPlot": ("qpdk.simulation.vacask", "RawPlot"),
    "Vacask": ("qpdk.simulation.vacask", "Vacask"),
    "VacaskError": ("qpdk.simulation.vacask", "VacaskError"),
    "cpw_tline_params": ("qpdk.simulation.vacask", "cpw_tline_params"),
    "format_value": ("qpdk.simulation.vacask", "format_value"),
    "vector_fit": ("qpdk.simulation.vacask", "vector_fit"),
    "vector_fit_subckt": ("qpdk.simulation.vacask", "vector_fit_subckt"),
}

__all__ = [
    "FEM_LAYERS",
    "Netlist",
    "RawFile",
    "RawPlot",
    "Vacask",
    "VacaskError",
    "cpw_tline_params",
    "format_value",
    "single_chip_stack",
    "to_fem_regions",
    "vector_fit",
    "vector_fit_subckt",
]


def __getattr__(name: str) -> Any:
    """Import a public name from its module on first access.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), attribute)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
