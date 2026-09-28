"""Deprecated alias of :mod:`qpdk.simulation.ansys.q3d`.

Import from :mod:`qpdk.simulation.ansys` (or ``qpdk.simulation.ansys.q3d``)
instead. This module will be removed in a future release.
"""

import warnings
from typing import Any

from qpdk.simulation.ansys import q3d as _moved

warnings.warn(
    "qpdk.simulation.q3d is deprecated and will be removed in a future release; "
    "import from qpdk.simulation.ansys.q3d instead.",
    DeprecationWarning,
    skip_file_prefixes=("<frozen importlib",),
)


def __getattr__(name: str) -> Any:
    """Forward attribute access to :mod:`qpdk.simulation.ansys.q3d`.

    Returns:
        The attribute of the moved module.
    """
    return getattr(_moved, name)
