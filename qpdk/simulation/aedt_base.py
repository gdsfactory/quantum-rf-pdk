"""Deprecated alias of :mod:`qpdk.simulation.ansys.base`.

Import from :mod:`qpdk.simulation.ansys` (or ``qpdk.simulation.ansys.base``)
instead. This module will be removed in a future release.
"""

import sys
import warnings

from qpdk.simulation.ansys import base

warnings.warn(
    "qpdk.simulation.aedt_base is deprecated and will be removed in a future release; "
    "import from qpdk.simulation.ansys.base instead.",
    DeprecationWarning,
    skip_file_prefixes=("<frozen importlib",),
)

sys.modules[__name__] = base
