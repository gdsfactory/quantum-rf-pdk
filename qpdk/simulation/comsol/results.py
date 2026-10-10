"""Result-file helpers for the COMSOL notebooks.

Re-exported from :mod:`gplugins.comsol.results`.
"""

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
