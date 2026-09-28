"""Deprecated alias of :mod:`qpdk.simulation.ansys.base`.

Import from :mod:`qpdk.simulation.ansys` (or ``qpdk.simulation.ansys.base``)
instead. This module will be removed in a future release.
"""

import warnings
from typing import Any

from qpdk.simulation.ansys import base as _moved

warnings.warn(
    "qpdk.simulation.aedt_base is deprecated and will be removed in a future release; "
    "import from qpdk.simulation.ansys.base instead.",
    DeprecationWarning,
    skip_file_prefixes=("<frozen importlib",),
)


def __getattr__(name: str) -> Any:
    """Forward attribute access to :mod:`qpdk.simulation.ansys.base`.

    Returns:
        The attribute of the moved module.
    """
    return getattr(_moved, name)
