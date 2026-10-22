"""Tests for the QPDK COMSOL inputs in :mod:`qpdk.simulation.comsol`.

The COMSOL model builder lives in :mod:`gplugins.comsol` and is tested there.
QPDK only supplies the layout preparation and the material constants.
"""

from __future__ import annotations

import inspect

from gplugins.comsol import COMSOL

from qpdk.simulation.comsol import (
    AIR_RELATIVE_PERMITTIVITY,
    SILICON_RELATIVE_PERMITTIVITY,
)
from qpdk.tech import material_properties


def test_permittivities_come_from_the_tech():
    """The constants are the QPDK technology values."""
    assert (
        material_properties["Si"]["relative_permittivity"]
        == SILICON_RELATIVE_PERMITTIVITY
    )
    assert (
        material_properties["vacuum"]["relative_permittivity"]
        == AIR_RELATIVE_PERMITTIVITY
    )


def test_silicon_permittivity_differs_from_the_gplugins_default():
    """The gplugins default is not the QPDK silicon, so notebooks must pass it."""
    parameters = inspect.signature(COMSOL.create_sheet).parameters
    default = parameters["silicon_relative_permittivity"].default
    assert default != SILICON_RELATIVE_PERMITTIVITY
