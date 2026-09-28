"""QPDK Ansys Electronics Desktop (AEDT) simulations built on PyAEDT.

:mod:`qpdk.simulation.ansys.base` holds the helpers shared by every AEDT
design: component preparation, layer-stack mapping, materials, and the
:class:`~qpdk.simulation.ansys.base.AEDTBase` wrapper.
:mod:`qpdk.simulation.ansys.hfss` wraps HFSS eigenmode and driven modal
designs, and :mod:`qpdk.simulation.ansys.q3d` wraps Q3D Extractor and 2D
Extractor (Q2D).

Note:
    Running a design needs the optional ``hfss`` extra
    (``uv sync --extra hfss``) and a local AEDT installation. PyAEDT is only
    imported for type checking and inside the methods that talk to AEDT, so
    importing this package works without it.

See the `PyAEDT documentation <https://aedt.docs.pyansys.com/>`_.
"""

from qpdk.simulation.ansys.base import (
    AEDTBase,
    add_materials_to_aedt,
    detach_desktop_logging,
    fit_view,
    layer_stack_to_gds_mapping,
    object_names_to_materials,
    prepare_component_for_aedt,
)
from qpdk.simulation.ansys.hfss import HFSS, lumped_port_rectangle_from_cpw
from qpdk.simulation.ansys.q3d import Q2D, Q3D

__all__ = [
    "HFSS",
    "Q2D",
    "Q3D",
    "AEDTBase",
    "add_materials_to_aedt",
    "detach_desktop_logging",
    "fit_view",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "object_names_to_materials",
    "prepare_component_for_aedt",
]
