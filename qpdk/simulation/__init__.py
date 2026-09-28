"""AEDT simulation utilities using PyAEDT.

This module provides class-based interfaces for setting up HFSS simulations
(eigenmode and driven modal) and Q3D Extractor parasitic extractions
from gdsfactory components. It uses the PyAEDT library to interface
with Ansys HFSS and Q3D Extractor.

**HFSS workflow:**

1. Prepare a component with :func:`~qpdk.simulation.ansys.base.prepare_component_for_aedt`
2. Export to GDS and import into HFSS with :meth:`qpdk.simulation.ansys.hfss.HFSS.import_component`
3. Configure simulation setup (e.g. Eigenmode or Driven) manually via PyAEDT
4. Extract results with :meth:`qpdk.simulation.ansys.hfss.HFSS.get_eigenmode_results` or :meth:`qpdk.simulation.ansys.hfss.HFSS.get_sparameter_results`

**Q3D Extractor workflow:**

1. Prepare a component with :func:`~qpdk.simulation.ansys.base.prepare_component_for_aedt`
2. Export to GDS and import into Q3D with :meth:`qpdk.simulation.ansys.q3d.Q3D.import_component`
3. Assign signal nets with :meth:`qpdk.simulation.ansys.q3d.Q3D.assign_nets_from_ports`
4. Configure Q3D setup and analyze
5. Extract capacitance matrix with :meth:`qpdk.simulation.ansys.q3d.Q3D.get_capacitance_matrix`

The AEDT wrappers live in :mod:`qpdk.simulation.ansys` and are re-exported here.

Note:
    This module requires the optional ``hfss`` dependency group.
    Install with: ``uv sync --extra hfss`` or ``pip install qpdk[hfss]``

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
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qpdk.simulation.ansys import (
    AEDTBase,
    add_materials_to_aedt,
    detach_desktop_logging,
    fit_view,
    layer_stack_to_gds_mapping,
    object_names_to_materials,
    prepare_component_for_aedt,
)
from qpdk.simulation.cluster import RAY_PORT, SlurmCluster, SlurmJobError
from qpdk.simulation.fem import (
    FEM_LAYERS,
    FLIP_CHIP_FEM_LAYERS,
    flip_chip_stack,
    single_chip_stack,
    to_fem_regions,
    to_flip_chip_regions,
)
from qpdk.simulation.study import RayRunner, SlurmRunner, run_study

if TYPE_CHECKING:
    from qpdk.simulation.ansys import (
        HFSS,
        Q2D,
        Q3D,
        lumped_port_rectangle_from_cpw,
    )

_ANSYS_LAZY_IMPORTS = frozenset({
    "HFSS",
    "Q2D",
    "Q3D",
    "lumped_port_rectangle_from_cpw",
})

__all__ = [
    "FEM_LAYERS",
    "FLIP_CHIP_FEM_LAYERS",
    "HFSS",
    "Q2D",
    "Q3D",
    "RAY_PORT",
    "AEDTBase",
    "RayRunner",
    "SlurmCluster",
    "SlurmJobError",
    "SlurmRunner",
    "add_materials_to_aedt",
    "detach_desktop_logging",
    "fit_view",
    "flip_chip_stack",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "object_names_to_materials",
    "prepare_component_for_aedt",
    "run_study",
    "single_chip_stack",
    "to_fem_regions",
    "to_flip_chip_regions",
]


def __getattr__(name: str) -> Any:
    """Forward the AEDT design wrappers from :mod:`qpdk.simulation.ansys`.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    if name in _ANSYS_LAZY_IMPORTS:
        return getattr(importlib.import_module("qpdk.simulation.ansys"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
