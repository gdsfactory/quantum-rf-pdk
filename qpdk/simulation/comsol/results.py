"""Result-file helpers for the COMSOL notebooks.

Re-exported from :mod:`gplugins.comsol.results`.

The names load from gplugins on first use, so this module imports without the
``comsol`` extra and only using a name needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.simulation.comsol._gplugins import reexport

if TYPE_CHECKING:
    from gplugins.comsol.results import (
        FREQUENCY_UNITS_GHZ,
        explain_missing_results,
        exported_frequency_ghz,
        requested_frequency_grid,
        result_file,
        write_json_atomically,
    )

__all__ = [
    "FREQUENCY_UNITS_GHZ",
    "explain_missing_results",
    "exported_frequency_ghz",
    "requested_frequency_grid",
    "result_file",
    "write_json_atomically",
]

__getattr__ = reexport(__name__, "results", __all__)


def __dir__() -> list[str]:
    """List the public names, including the lazily imported ones.

    Returns:
        Sorted public attribute names.
    """
    return sorted(__all__)
