"""Q3D and Q2D simulation utilities, with QPDK defaults.

The generic Q3D and Q2D wrappers live in :mod:`gplugins.ansys`; the classes here
default the layer stack and materials to QPDK's, keep one instance per class, and
add :meth:`Q2D.create_2d_from_cross_section` for QPDK cross-sections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qpdk.models.cpw import get_cpw_dimensions
from qpdk.simulation.aedt_base import MISSING_HFSS_EXTRA

try:
    from gplugins.ansys.q3d import Q2D as _Q2D, Q3D as _Q3D
except ModuleNotFoundError as error:
    raise ImportError(MISSING_HFSS_EXTRA) from error

from qpdk.simulation._aedt import AEDTBase

if TYPE_CHECKING:
    from gdsfactory.technology import LayerStack
    from gdsfactory.typings import CrossSectionSpec

__all__ = ["Q2D", "Q3D"]


class Q3D(_Q3D, AEDTBase):
    """Q3D Extractor simulation wrapper.

    Provides methods for importing components into Q3D, assigning signal nets, and
    extracting the capacitance matrix. See :class:`gplugins.ansys.Q3D`; the layer
    stack defaults to QPDK's ``LAYER_STACK`` and the materials to QPDK's
    ``material_properties``.
    """


class Q2D(_Q2D, AEDTBase):
    """Q2D simulation wrapper.

    Provides methods for 2D cross-sectional impedance extraction. See
    :class:`gplugins.ansys.Q2D`; the layer stack defaults to QPDK's
    ``LAYER_STACK`` and the materials to QPDK's ``material_properties``.
    """

    def create_2d_from_cross_section(
        self,
        cross_section: CrossSectionSpec,
        layer_stack: LayerStack | None = None,
        *,
        ground_width: float | None = None,
        units: str = "um",
    ) -> dict[str, str]:
        """Create a 2D model from a CPW cross-section for impedance extraction.

        Builds the cross-sectional geometry of a coplanar waveguide in Ansys Q2D
        (2D Extractor) with :meth:`gplugins.ansys.Q2D.create_2d_cpw`, on the
        ``Substrate`` and ``M1`` levels of the layer stack.

        Args:
            cross_section: A gdsfactory cross-section specification describing the CPW
                geometry (width and gap).
            layer_stack: LayerStack defining substrate and conductor properties.
                If None, uses QPDK's default ``LAYER_STACK``.
            ground_width: Width of each coplanar ground plane in µm. If None,
                defaults to 10× the CPW gap.
            units: Length units for the Q2D geometry (default ``"um"``).

        Returns:
            Dictionary with keys ``"signal"``, ``"gnd_left"``, ``"gnd_right"``,
            ``"substrate"`` mapping to the created Q2D object names.

        Raises:
            ValueError: If the units are not ``"um"``, or the cross-section is not a
                valid CPW.
            KeyError: If the layer stack has no ``Substrate`` or ``M1`` level.
        """
        if units != "um":
            raise ValueError("Q2D cross-section expects units='um'")
        cpw_width, cpw_gap = get_cpw_dimensions(cross_section)
        stack = self.resolve_layer_stack(layer_stack)
        for level in ("Substrate", "M1"):
            if level not in stack.layers:
                raise KeyError(level)
        return self.create_2d_cpw(
            cpw_width,
            cpw_gap,
            stack,
            substrate_level="Substrate",
            conductor_level="M1",
            ground_width=ground_width,
            units=units,
        )
