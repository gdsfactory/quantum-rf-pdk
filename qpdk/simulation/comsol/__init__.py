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
    Using any of these names needs the optional ``comsol`` extra, which
    installs ``gplugins[comsol]``; building and solving models also needs a
    local COMSOL installation. Every module here imports gplugins only when
    one of its names is used, so importing this package and its modules stays
    possible without the extra, and a missing extra fails with an
    :class:`ImportError` naming ``uv sync --extra comsol``.

See the `MPh repository <https://github.com/MPh-py/MPh>`_ and
`MPh documentation <https://mph.readthedocs.io/en/stable/>`_.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qpdk.simulation.comsol.capacitance import add_capacitance_study
    from qpdk.simulation.comsol.layout import (
        ComsolBoundingBox,
        ComsolFeedPort,
        ComsolLayout,
        ComsolPolygon,
        prepare_comsol_layout,
    )
    from qpdk.simulation.comsol.mesh import (
        pin_absolute_edge_mesh_sizes,
        pin_absolute_mesh_sizes,
        refine_metal_plane_mesh,
    )
    from qpdk.simulation.comsol.metal import build_comsol_metal_model
    from qpdk.simulation.comsol.model import COMSOL
    from qpdk.simulation.comsol.rf import add_cpw_rf_study
    from qpdk.simulation.comsol.sheet import build_comsol_sheet_model

_LAZY_IMPORTS: dict[str, str] = {
    "COMSOL": "model",
    "ComsolBoundingBox": "layout",
    "ComsolFeedPort": "layout",
    "ComsolLayout": "layout",
    "ComsolPolygon": "layout",
    "add_capacitance_study": "capacitance",
    "add_cpw_rf_study": "rf",
    "build_comsol_metal_model": "metal",
    "build_comsol_sheet_model": "sheet",
    "pin_absolute_edge_mesh_sizes": "mesh",
    "pin_absolute_mesh_sizes": "mesh",
    "prepare_comsol_layout": "layout",
    "refine_metal_plane_mesh": "mesh",
}

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


# TODO(Python 3.15): lazy from-imports (PEP 810) replace this hook.
def __getattr__(name: str) -> Any:
    """Import a public name, and gplugins with it, only when it is asked for.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    try:
        submodule = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
