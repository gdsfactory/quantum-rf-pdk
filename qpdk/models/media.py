"""Models for transmission line media."""

import inspect
from functools import cache, partial
from typing import Protocol, cast

import gdsfactory as gf
import skrf
from gdsfactory.typings import CrossSectionSpec
from skrf.media import CPW, Media

from qpdk import LAYER_STACK
from qpdk.tech import coplanar_waveguide, material_properties


class MediaCallable(Protocol):
    """Typing :class:`Protocol` for functions that accept a frequency keyword argument and return :class:`~Media`."""

    def __call__(self, *, frequency: skrf.Frequency) -> Media:
        """Call with frequency keyword argument and return Media object."""
        ...


_coplanar_waveguide_xsection_signature = inspect.signature(coplanar_waveguide)


@cache
def cpw_media_skrf(
    width: float = _coplanar_waveguide_xsection_signature.parameters["width"].default,
    gap: float = _coplanar_waveguide_xsection_signature.parameters["gap"].default,
) -> MediaCallable:
    """Create a partial coplanar waveguide (CPW) media object using scikit-rf.

    Args:
        width: Width of the center conductor in μm.
        gap: Width of the gap between the center conductor and ground planes in μm.

    Returns:
        partial[skrf.media.CPW]: A CPW media object with specified dimensions.
    """
    # Convert μm to m for skrf
    return partial(
        CPW,
        w=width * 1e-6,
        s=gap * 1e-6,
        h=LAYER_STACK.layers["Substrate"].thickness * 1e-6,
        t=LAYER_STACK.layers["M1"].thickness * 1e-6,
        ep_r=material_properties[cast(str, LAYER_STACK.layers["Substrate"].material)][
            "relative_permittivity"
        ],
        # rho=1e-32,  # set to a very low value to avoid warnings
        rho=1e-100,  # set to a very low value to avoid warnings
        tand=0,  # No dielectric losses for now
        has_metal_backside=False,
    )


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
    xs = gf.get_cross_section(cross_section)
    width = xs.width
    gap = next(
        section.width for section in xs.sections if "etch_offset" in section.name
    )
    return cpw_media_skrf(width=width, gap=gap)


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    freq = skrf.Frequency(2, 8, 501, "GHz")
    media = cross_section_to_media("cpw")
    cpw = media(frequency=freq)

    print(cpw)
