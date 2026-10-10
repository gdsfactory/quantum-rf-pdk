"""QPDK AEDT base class over :class:`gplugins.ansys.AEDTBase`.

Kept apart from :mod:`qpdk.simulation.aedt_base` so that importing that module, and
:func:`~qpdk.simulation.aedt_base.prepare_component_for_aedt` with it, does not need
gplugins. Import :class:`AEDTBase` from :mod:`qpdk.simulation.aedt_base`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplugins.ansys.base import AEDTBase as _AEDTBase

from qpdk import LAYER_STACK
from qpdk.singleton import SingletonMeta
from qpdk.tech import material_properties as _qpdk_material_properties

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss, Q2d
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.technology import LayerStack
    from gplugins.ansys.base import MaterialProperties

__all__ = ["AEDTBase"]


class AEDTBase(_AEDTBase, metaclass=SingletonMeta):
    """Base class for AEDT simulations, defaulting to QPDK's stack and materials."""

    def __init__(
        self,
        app: Hfss | Q2d | Q3d,
        *,
        layer_stack: LayerStack | None = None,
        material_properties: MaterialProperties | None = None,
    ):
        """Initialize the AEDT base class.

        Args:
            app: The PyAEDT application instance.
            layer_stack: Default layer stack for imports. Defaults to QPDK's
                ``LAYER_STACK``.
            material_properties: Material table. Defaults to QPDK's
                ``material_properties``.
        """
        super().__init__(
            app,
            layer_stack=LAYER_STACK if layer_stack is None else layer_stack,
            material_properties=(
                _qpdk_material_properties
                if material_properties is None
                else material_properties
            ),
        )
