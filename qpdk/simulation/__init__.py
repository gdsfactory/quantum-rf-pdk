"""AEDT and COMSOL simulation utilities.

This package provides class-based interfaces for setting up HFSS simulations
(eigenmode and driven modal) and Q3D Extractor parasitic extractions from
gdsfactory components, plus a small MPh-based COMSOL geometry builder.

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

**COMSOL workflow:**

1. Extract metal polygons and optional feed ports with
   :func:`~qpdk.simulation.comsol_layout.prepare_comsol_layout`
2. Build a 3D geometry project with
   :func:`~qpdk.simulation.comsol.build_comsol_metal_model`
3. Add RF physics and a study to the returned model yourself

Note:
    The AEDT wrappers require ``uv sync --extra hfss``. The COMSOL builder
    requires ``uv sync --extra comsol`` and a local COMSOL installation.

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
from typing import Any

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
    "ComsolBoundingBox": ("qpdk.simulation.comsol_layout", "ComsolBoundingBox"),
    "ComsolFeedPort": ("qpdk.simulation.comsol_layout", "ComsolFeedPort"),
    "ComsolLayout": ("qpdk.simulation.comsol_layout", "ComsolLayout"),
    "ComsolPolygon": ("qpdk.simulation.comsol_layout", "ComsolPolygon"),
    "prepare_comsol_layout": (
        "qpdk.simulation.comsol_layout",
        "prepare_comsol_layout",
    ),
    "build_comsol_cpw_model": ("qpdk.simulation.comsol", "build_comsol_cpw_model"),
    "build_comsol_metal_model": (
        "qpdk.simulation.comsol",
        "build_comsol_metal_model",
    ),
}

__all__ = [
    "HFSS",
    "Q2D",
    "Q3D",
    "AEDTBase",
    "ComsolBoundingBox",
    "ComsolFeedPort",
    "ComsolLayout",
    "ComsolPolygon",
    "add_materials_to_aedt",
    "detach_desktop_logging",
    "fit_view",
    "build_comsol_cpw_model",
    "build_comsol_metal_model",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "object_names_to_materials",
    "prepare_component_for_aedt",
    "prepare_comsol_layout",
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
