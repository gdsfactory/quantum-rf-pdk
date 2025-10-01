"""Models for transmission line media."""

import inspect
from functools import cache, partial
from typing import Protocol, cast

import skrf
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
    )
