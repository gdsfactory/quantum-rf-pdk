"""HFSS simulation utilities, with QPDK defaults.

The generic HFSS wrapper lives in :mod:`gplugins.ansys`; :class:`HFSS` here only
defaults the layer stack and materials to QPDK's and keeps one instance per class.
"""

from __future__ import annotations

from gplugins.ansys.hfss import (
    HFSS as _HFSS,
    LumpedPortConfig,
    lumped_port_rectangle_from_cpw,
)

from qpdk.simulation._aedt import AEDTBase

__all__ = ["HFSS", "LumpedPortConfig", "lumped_port_rectangle_from_cpw"]


class HFSS(_HFSS, AEDTBase):
    """HFSS simulation wrapper.

    Provides high-level methods for importing components into HFSS,
    setting up simulation regions, and extracting results. See
    :class:`gplugins.ansys.HFSS`; the layer stack defaults to QPDK's
    ``LAYER_STACK`` and the materials to QPDK's ``material_properties``.
    """
