"""Deprecated alias of :mod:`qpdk.simulation.ansys.q3d`.

Import from :mod:`qpdk.simulation.ansys` (or ``qpdk.simulation.ansys.q3d``)
instead. This module will be removed in a future release.
"""

import sys
import warnings

from qpdk.simulation.ansys import q3d

warnings.warn(
    "qpdk.simulation.q3d is deprecated and will be removed in a future release; "
    "import from qpdk.simulation.ansys.q3d instead.",
    DeprecationWarning,
    skip_file_prefixes=("<frozen importlib",),
)

sys.modules[__name__] = q3d
