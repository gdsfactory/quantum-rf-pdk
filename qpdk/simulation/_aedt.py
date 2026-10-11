"""QPDK AEDT base class over :class:`gplugins.ansys.AEDTBase`.

Kept apart from :mod:`qpdk.simulation.aedt_base` so that importing that module, and
:func:`~qpdk.simulation.aedt_base.prepare_component_for_aedt` with it, does not need
gplugins. Import :class:`AEDTBase` from :mod:`qpdk.simulation.aedt_base`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk import LAYER_STACK
from qpdk.simulation.aedt_base import MISSING_HFSS_EXTRA
from qpdk.singleton import SingletonMeta
from qpdk.tech import material_properties as _qpdk_material_properties

try:
    from gplugins.ansys.base import AEDTBase as _AEDTBase
except ModuleNotFoundError as error:
    raise ImportError(MISSING_HFSS_EXTRA) from error

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss, Q2d
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.technology import LayerStack
    from gplugins.ansys.base import MaterialProperties

__all__ = ["AEDTBase"]


class AEDTBase(_AEDTBase, metaclass=SingletonMeta):
    """Base class for AEDT simulations, defaulting to QPDK's stack and materials.

    Each subclass is a singleton: later constructor calls return the first
    instance. A later call that asks for a different ``layer_stack`` or
    ``material_properties`` than that instance has (leaving one out asks for the
    QPDK default) raises :class:`ValueError` instead of silently returning an
    instance configured differently. The ``app`` of a later call is ignored.
    """

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

    def _check_singleton_args(
        self,
        app: Hfss | Q2d | Q3d,  # ruff: ignore[unused-method-argument]
        *,
        layer_stack: LayerStack | None = None,
        material_properties: MaterialProperties | None = None,
    ) -> None:
        """Refuse a later construction that asks for a different configuration.

        Raises:
            ValueError: If ``layer_stack`` or ``material_properties`` (or the QPDK
                default, when left out) differs from the existing instance's.
        """
        requested = {
            "layer_stack": LAYER_STACK if layer_stack is None else layer_stack,
            "material_properties": (
                _qpdk_material_properties
                if material_properties is None
                else material_properties
            ),
        }
        for name, value in requested.items():
            current = getattr(self, name)
            if current is not value and current != value:
                msg = (
                    f"{type(self).__name__} is a singleton and already exists with a "
                    f"different {name}; pass the same {name} as the first call, or "
                    "reuse the existing instance."
                )
                raise ValueError(msg)
