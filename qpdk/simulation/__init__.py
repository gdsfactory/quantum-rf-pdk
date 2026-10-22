"""AEDT, COMSOL, Palace and VACASK simulation utilities.

This package provides class-based interfaces for setting up HFSS simulations
(eigenmode and driven modal) and Q3D Extractor parasitic extractions from
gdsfactory components, plus MPh-based COMSOL geometry and study builders.

The AEDT wrappers use the PyAEDT library to interface with Ansys HFSS and Q3D
Extractor.

**HFSS workflow:**

1. Prepare a component with :func:`~qpdk.simulation.aedt_base.prepare_component_for_aedt`
2. Export to GDS and import into HFSS with :meth:`qpdk.simulation.hfss.HFSS.import_component`
3. Configure simulation setup (e.g. Eigenmode or Driven) manually via PyAEDT
4. Extract results with :meth:`qpdk.simulation.hfss.HFSS.get_eigenmode_results` or :meth:`qpdk.simulation.hfss.HFSS.get_sparameter_results`

**Q3D Extractor workflow:**

1. Prepare a component with :func:`~qpdk.simulation.aedt_base.prepare_component_for_aedt`
2. Export to GDS and import into Q3D with :meth:`qpdk.simulation.q3d.Q3D.import_component`
3. Assign signal nets with :meth:`qpdk.simulation.q3d.Q3D.assign_nets_from_ports`
4. Configure Q3D setup and analyze
5. Extract capacitance matrix with :meth:`qpdk.simulation.q3d.Q3D.get_capacitance_matrix`

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
    The AEDT wrappers require ``uv sync --extra hfss``. The COMSOL module
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
    >>> from qpdk.simulation import HFSS, prepare_component_for_aedt
    >>> from qpdk.cells import resonator
    >>> comp = resonator(length=4000, meanders=4)
    >>> prepared_comp = prepare_component_for_aedt(comp)
    >>> hfss_app = Hfss(project="resonator_sim", solution_type="Eigenmode")
    >>> hfss_sim = HFSS(hfss_app)
    >>> hfss_sim.import_component(prepared_comp)

References:
    - PyAEDT documentation: https://aedt.docs.pyansys.com/
    - HFSS import_gds_3d: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.import_gds_3d.html
    - Q3D Extractor: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.q3d.Q3d.html
    - MPh: https://github.com/MPh-py/MPh
"""

# ruff: file-ignore[undefined-export]

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qpdk.simulation.hfss import HFSS
    from qpdk.simulation.q3d import Q2D, Q3D

# TODO(Python 3.15): `lazy from ... import` (PEP 810) - replaces _LAZY_IMPORTS, __getattr__,
# __dir__, the TYPE_CHECKING block and the undefined-export ignore. Keep them lazy, not eager.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AEDTBase": ("qpdk.simulation.aedt_base", "AEDTBase"),
    "add_materials_to_aedt": ("qpdk.simulation.aedt_base", "add_materials_to_aedt"),
    "detach_desktop_logging": ("qpdk.simulation.aedt_base", "detach_desktop_logging"),
    "fit_view": ("qpdk.simulation.aedt_base", "fit_view"),
    "layer_stack_to_gds_mapping": (
        "qpdk.simulation.aedt_base",
        "layer_stack_to_gds_mapping",
    ),
    "object_names_to_materials": (
        "qpdk.simulation.aedt_base",
        "object_names_to_materials",
    ),
    "prepare_component_for_aedt": (
        "qpdk.simulation.aedt_base",
        "prepare_component_for_aedt",
    ),
    "HFSS": ("qpdk.simulation.hfss", "HFSS"),
    "lumped_port_rectangle_from_cpw": (
        "qpdk.simulation.hfss",
        "lumped_port_rectangle_from_cpw",
    ),
    "Q2D": ("qpdk.simulation.q3d", "Q2D"),
    "Q3D": ("qpdk.simulation.q3d", "Q3D"),
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
    "HFSS",
    "Q2D",
    "Q3D",
    "AEDTBase",
    "Netlist",
    "RawFile",
    "RawPlot",
    "Vacask",
    "VacaskError",
    "add_materials_to_aedt",
    "cpw_tline_params",
    "detach_desktop_logging",
    "fit_view",
    "format_value",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "object_names_to_materials",
    "prepare_component_for_aedt",
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
