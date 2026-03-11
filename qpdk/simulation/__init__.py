"""AEDT simulation utilities using PyAEDT.

This module provides class-based interfaces for setting up HFSS simulations
(eigenmode and driven modal) and Q3D Extractor parasitic extractions
from gdsfactory components. It uses the PyAEDT library to interface
with Ansys HFSS and Q3D Extractor.

**HFSS workflow:**

1. Prepare a component with :func:`prepare_component_for_aedt`
2. Export to GDS and import into HFSS with :meth:`qpdk.simulation.hfss.HFSS.import_component`
3. Configure simulation setup (e.g. Eigenmode or Driven) manually via PyAEDT
4. Extract results with :meth:`qpdk.simulation.hfss.HFSS.get_eigenmode_results` or :meth:`qpdk.simulation.hfss.HFSS.get_sparameter_results`

**Q3D Extractor workflow:**

1. Prepare a component with :func:`prepare_component_for_aedt`
2. Export to GDS and import into Q3D with :meth:`qpdk.simulation.q3d.Q3D.import_component`
3. Assign signal nets with :meth:`qpdk.simulation.q3d.Q3D.assign_nets_from_ports`
4. Configure Q3D setup and analyze
5. Extract capacitance matrix with :meth:`qpdk.simulation.q3d.Q3D.get_capacitance_matrix`

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

from qpdk.simulation.aedt_base import (
    AEDTBase,
    add_materials_to_aedt,
    layer_stack_to_gds_mapping,
    prepare_component_for_aedt,
)
from qpdk.simulation.hfss import HFSS, lumped_port_rectangle_from_cpw
from qpdk.simulation.q3d import Q2D, Q3D

__all__ = [
    "HFSS",
    "Q2D",
    "Q3D",
    "AEDTBase",
    "add_materials_to_aedt",
    "layer_stack_to_gds_mapping",
    "lumped_port_rectangle_from_cpw",
    "prepare_component_for_aedt",
]
