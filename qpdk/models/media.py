"""Transmission media."""

from functools import cache, partial
from typing import Protocol, cast

import gdsfactory as gf
import skrf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import CrossSectionSpec
from skrf.media import CPW, Media

from qpdk import LAYER_STACK
from qpdk.tech import material_properties


class MediaCallable(Protocol):
    """Typing :class:`Protocol` for functions that accept a frequency keyword argument and return :class:`~Media`."""

    def __call__(self, *, frequency: skrf.Frequency) -> Media:
        """Call with frequency keyword argument and return Media object."""
        ...


@cache
def cpw_media_skrf(width: float, gap: float) -> MediaCallable:
    """Create a partial coplanar waveguide (CPW) media object using scikit-rf.

    Args:
        width: Width of the center conductor in μm.
        gap: Width of the gap between the center conductor and ground planes in μm.

    Returns:
        partial[skrf.media.CPW]: A CPW media object with specified dimensions.
    """
    kwargs = {
        "w": width * 1e-6,
        "s": gap * 1e-6,
        "h": LAYER_STACK.layers["Substrate"].thickness * 1e-6,
        "t": LAYER_STACK.layers["M1"].thickness * 1e-6,
        "ep_r": material_properties[
            cast(str, LAYER_STACK.layers["Substrate"].material)
        ]["relative_permittivity"],
        "rho": 1e-100,  # set to a very low value to avoid warnings
        "tand": 0,  # No dielectric losses for now
        "has_metal_backside": False,
    }
    return partial(CPW, **kwargs)


def cross_section_to_media(cross_section: CrossSectionSpec) -> MediaCallable:
    """Converts a layout :class:`~CrossSectionSpec` to model :class:`~MediaCallable`.

    This function assumes the cross-section to have Sections similarly
    to :func:`qpdk.tech.coplanar_waveguide`. Namely, the primary width corresponds
    to CPW width and the gap is the width of a Section that includes
    `etch_offset` in the name.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        MediaCallable: A callable that returns a skrf Media object for a given frequency.
    """
    # Convert input to CrossSection object
    xs: CrossSection
    if isinstance(cross_section, CrossSection):
        xs = cross_section
    elif callable(cross_section):
        # If it's a callable (like a partial or factory function), call it to get the CrossSection
        xs = cast(CrossSection, cross_section())
    else:
        # It's a string name, requires active PDK
        xs = gf.get_cross_section(cross_section)

    # Extract width and gap from the CrossSection
    width = xs.width
    try:
        gap = next(
            section.width
            for section in xs.sections
            if section.name and "etch_offset" in section.name
        )
    except StopIteration as e:
        msg = (
            f"Cross-section does not have a section with 'etch_offset' in the name. "
            f"Found sections: {[s.name for s in xs.sections]}"
        )
        raise ValueError(msg) from e
    return cpw_media_skrf(width=width, gap=gap)
