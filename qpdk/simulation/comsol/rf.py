"""CPW RF study on a sheet model, re-exported from :mod:`gplugins.comsol.rf`.

The names load from gplugins on first use, so this module imports without the
``comsol`` extra and only using a name needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.simulation.comsol._gplugins import reexport

if TYPE_CHECKING:
    from gplugins.comsol.rf import (
        INPUT_FACES_SELECTION,
        INPUT_GAP_SELECTION,
        METAL_FACES_SELECTION,
        OUTPUT_FACES_SELECTION,
        OUTPUT_GAP_SELECTION,
        add_cpw_rf_study,
    )

__all__ = [
    "INPUT_FACES_SELECTION",
    "INPUT_GAP_SELECTION",
    "METAL_FACES_SELECTION",
    "OUTPUT_FACES_SELECTION",
    "OUTPUT_GAP_SELECTION",
    "add_cpw_rf_study",
]

__getattr__ = reexport(__name__, "rf", __all__)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
