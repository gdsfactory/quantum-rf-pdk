"""QPDK wrappers around the COMSOL models of :mod:`gplugins.comsol`.

The generic COMSOL code (polygon extraction, sheet and metal builders, the RF
and electrostatic studies, the mesh, plotting, and result helpers, and the
chainable :class:`COMSOL` model) lives in :mod:`gplugins.comsol`. QPDK keeps
only what depends on its technology:

- :func:`~qpdk.simulation.comsol.layout.prepare_comsol_layout` inverts the
  M1_ETCH mask into M1_DRAW metal with a ground margin, rejects unsupported
  layers, and defaults to the ``coupling_o1``/``coupling_o2`` feeds.
- :func:`~qpdk.simulation.comsol.sheet.build_comsol_sheet_model` and
  :class:`COMSOL` default to the QPDK technology permittivities.

:mod:`~qpdk.simulation.comsol.rf`, :mod:`~qpdk.simulation.comsol.capacitance`,
:mod:`~qpdk.simulation.comsol.mesh`, :mod:`~qpdk.simulation.comsol.plotting`,
and :mod:`~qpdk.simulation.comsol.results` re-export the gplugins helpers.
Solving, saving, and evaluating are MPh's own methods.

Note:
    The builders need the optional ``comsol`` extra, which installs
    ``gplugins[comsol]``, and a local COMSOL installation. Only
    :class:`COMSOL` imports MPh, and lazily, so importing this package or the
    layout and helper modules stays possible without it.

See the `MPh repository <https://github.com/MPh-py/MPh>`_ and
`MPh documentation <https://mph.readthedocs.io/en/stable/>`_.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from gplugins.comsol.capacitance import add_capacitance_study
from gplugins.comsol.mesh import (
    pin_absolute_edge_mesh_sizes,
    pin_absolute_mesh_sizes,
    refine_metal_plane_mesh,
)
from gplugins.comsol.rf import add_cpw_rf_study

from qpdk.simulation.comsol.layout import (
    ComsolBoundingBox,
    ComsolFeedPort,
    ComsolLayout,
    ComsolPolygon,
    prepare_comsol_layout,
)
from qpdk.simulation.comsol.metal import build_comsol_metal_model
from qpdk.simulation.comsol.sheet import build_comsol_sheet_model

if TYPE_CHECKING:
    from qpdk.simulation.comsol.model import COMSOL

__all__ = [
    "COMSOL",
    "ComsolBoundingBox",
    "ComsolFeedPort",
    "ComsolLayout",
    "ComsolPolygon",
    "add_capacitance_study",
    "add_cpw_rf_study",
    "build_comsol_metal_model",
    "build_comsol_sheet_model",
    "pin_absolute_edge_mesh_sizes",
    "pin_absolute_mesh_sizes",
    "prepare_comsol_layout",
    "refine_metal_plane_mesh",
]


# TODO(Python 3.15): lazy from-import of COMSOL (PEP 810) replaces this hook.
def __getattr__(name: str) -> Any:
    """Import a public name that needs MPh only when it is asked for.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    if name == "COMSOL":
        return importlib.import_module("qpdk.simulation.comsol.model").COMSOL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public names, including the lazily imported one.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
