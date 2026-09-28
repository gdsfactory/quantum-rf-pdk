"""Deprecated alias of :mod:`qpdk.simulation.ansys.hfss`.

Import from :mod:`qpdk.simulation.ansys` (or ``qpdk.simulation.ansys.hfss``)
instead. This module will be removed in a future release.
"""

import sys
import warnings

from qpdk.simulation.ansys import hfss

warnings.warn(
    "qpdk.simulation.hfss is deprecated and will be removed in a future release; "
    "import from qpdk.simulation.ansys.hfss instead.",
    DeprecationWarning,
    skip_file_prefixes=("<frozen importlib",),
)

sys.modules[__name__] = hfss
