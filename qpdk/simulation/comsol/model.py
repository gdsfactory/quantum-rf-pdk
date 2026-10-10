"""An MPh model with QPDK layout, study, and mesh methods.

:class:`COMSOL` is :class:`gplugins.comsol.model.COMSOL` with the QPDK defaults:
the sheet builder uses the QPDK technology permittivities and the metal builder
the QPDK model name. Solving, saving, and evaluating are MPh's own methods; the
study and mesh methods are inherited unchanged. Import it from
:mod:`qpdk.simulation.comsol` or :mod:`qpdk.simulation` after installing the
optional ``comsol`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from qpdk.simulation.comsol import sheet
from qpdk.simulation.comsol._gplugins import INSTALL_HINT, is_missing

try:
    from gplugins.comsol import model as _gplugins_model
except ModuleNotFoundError as error:
    if not is_missing(error, "gplugins.comsol.model"):
        raise
    raise ImportError(f"{__name__} {INSTALL_HINT}") from error

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol.layout import ComsolLayout

__all__ = ["COMSOL"]


class COMSOL(_gplugins_model.COMSOL):
    """An MPh model holding the QPDK layout its geometry was built from.

    Use :meth:`create_sheet` or :meth:`create_metal` to build one. The study and
    mesh methods then use the stored layout. Study methods return the model for
    chaining; mesh methods return the resulting element count.
    """

    @classmethod
    def create_sheet(
        cls,
        client: mph.Client,
        layout: ComsolLayout,
        name: str,
        *,
        substrate_thickness_um: float = 200.0,
        air_height_um: float = 200.0,
        lateral_margin_um: float = 0.0,
        silicon_relative_permittivity: float = sheet.SILICON_RELATIVE_PERMITTIVITY,
        air_relative_permittivity: float = sheet.AIR_RELATIVE_PERMITTIVITY,
    ) -> Self:
        """Build a sheet model, with the metal as faces on the z = 0 interface.

        Args:
            client: A connected :class:`mph.Client`.
            layout: Extracted metal polygons and bounding box in µm.
            name: Name of the COMSOL model.
            substrate_thickness_um: Silicon thickness below the interface, in µm.
            air_height_um: Air height above the interface, in µm.
            lateral_margin_um: Margin around the layout bounding box for both
                blocks, in µm.
            silicon_relative_permittivity: Relative permittivity of the silicon
                block, positive and finite. Defaults to the QPDK technology
                value for Si.
            air_relative_permittivity: Relative permittivity of the air block,
                positive and finite. Defaults to the QPDK technology value for
                vacuum.

        Returns:
            The built model, holding the layout it was built from.
        """
        return super().create_sheet(
            client,
            layout,
            name,
            substrate_thickness_um=substrate_thickness_um,
            air_height_um=air_height_um,
            lateral_margin_um=lateral_margin_um,
            silicon_relative_permittivity=silicon_relative_permittivity,
            air_relative_permittivity=air_relative_permittivity,
        )

    @classmethod
    def create_metal(
        cls,
        client: mph.Client,
        layout: ComsolLayout,
        *,
        metal_thickness_um: float = 0.2,
        name: str = "QPDK metal",
    ) -> Self:
        """Build a metal model, with the polygons extruded to a thickness.

        Args:
            client: A connected :class:`mph.Client`.
            layout: Extracted metal polygons and feed ports in µm.
            metal_thickness_um: Extrusion height in µm, strictly positive.
            name: Name of the COMSOL model.

        Returns:
            The built model, holding the layout it was built from.
        """
        return super().create_metal(
            client, layout, metal_thickness_um=metal_thickness_um, name=name
        )
