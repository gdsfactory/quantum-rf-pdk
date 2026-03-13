"""Transmission media."""

from functools import cache, partial
from typing import Protocol, cast

import gdsfactory as gf
import jax.numpy as jnp
import skrf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import (
    cpw_epsilon_eff,
    cpw_thickness_correction,
    cpw_z0,
)
from skrf.media import CPW

from qpdk import LAYER_STACK
from qpdk.helper import deprecated
from qpdk.tech import material_properties

DEPRECATION_MSG = "Prefer cpw_parameters or the functions in qpdk.models.cpw for JAX-jittable analysis."


class MediaCallable(Protocol):
    """Typing :class:`Protocol` for functions that accept a frequency keyword argument and return :class:`~CPW`."""

    def __call__(self, *, frequency: skrf.Frequency) -> CPW:
        """Call with frequency keyword argument and return CPW media object."""
        ...


@cache
@deprecated(DEPRECATION_MSG)
def cpw_media_skrf(width: float, gap: float) -> MediaCallable:
    """Create a partial coplanar waveguide (CPW) media object using scikit-rf.

    .. deprecated::
        Prefer :func:`cpw_parameters` or the functions in
        :mod:`qpdk.models.cpw` for JAX-jittable analysis.

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


def get_cpw_dimensions(cross_section: CrossSectionSpec) -> tuple[float, float]:
    """Extracts CPW width and gap from a cross-section specification.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        tuple[float, float]: Width and gap of the CPW.
    """
    xs: CrossSection
    if isinstance(cross_section, CrossSection):
        xs = cross_section
    elif callable(cross_section):
        xs = cast(CrossSection, cross_section())
    else:
        xs = gf.get_cross_section(cross_section)

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
    return width, gap


@deprecated(DEPRECATION_MSG)
def cross_section_to_media(cross_section: CrossSectionSpec) -> MediaCallable:
    """Converts a layout :class:`~CrossSectionSpec` to model :class:`~MediaCallable`.

    .. deprecated::
        Prefer :func:`cpw_parameters` or the functions in
        :mod:`qpdk.models.cpw` for JAX-jittable analysis.

    This function assumes the cross-section to have Sections similarly
    to :func:`qpdk.tech.coplanar_waveguide`. Namely, the primary width corresponds
    to CPW width and the gap is the width of a Section that includes
    `etch_offset` in the name.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        MediaCallable: A callable that returns a skrf Media object for a given frequency.
    """
    width, gap = get_cpw_dimensions(cross_section)
    return cpw_media_skrf(width=width, gap=gap)


@cache
def get_cpw_substrate_params() -> tuple[float, float, float]:
    """Extract substrate parameters from the PDK layer stack.

    Returns:
        ``(h, t, ep_r)`` — substrate height (µm), conductor thickness (µm),
        and relative permittivity.
    """
    h = LAYER_STACK.layers["Substrate"].thickness  # µm
    t = LAYER_STACK.layers["M1"].thickness  # µm
    ep_r = material_properties[cast(str, LAYER_STACK.layers["Substrate"].material)][
        "relative_permittivity"
    ]
    return float(h), float(t), float(ep_r)


@cache
def cpw_parameters(
    width: float,
    gap: float,
) -> tuple[float, float]:
    r"""Compute effective permittivity and characteristic impedance for a CPW.

    Uses the JAX-jittable functions from :mod:`qpdk.models.cpw` with the
    PDK layer stack (substrate height, conductor thickness, material
    permittivity).

    Conductor thickness corrections follow
    Gupta, Garg, Bahl & Bhartia :cite:`guptaMicrostripLinesSlotlines1996`
    (§7.3, Eqs. 7.98-7.100).

    Args:
        width: Centre-conductor width in µm.
        gap: Gap between centre conductor and ground plane in µm.

    Returns:
        ``(ep_eff, z0)`` — effective permittivity (dimensionless) and
        characteristic impedance (Ω).
    """
    h_um, t_um, ep_r = get_cpw_substrate_params()

    # Convert to SI (metres)
    w_m = width * 1e-6
    s_m = gap * 1e-6
    h_m = h_um * 1e-6
    t_m = t_um * 1e-6

    # Base (zero-thickness) quantities
    ep_eff = cpw_epsilon_eff(w_m, s_m, h_m, ep_r)

    if t_um > 0:
        ep_eff_t, z0_val = cpw_thickness_correction(w_m, s_m, t_m, ep_eff)
        return float(ep_eff_t), float(z0_val)

    z0_val = cpw_z0(w_m, s_m, ep_eff)
    return float(ep_eff), float(z0_val)


def cpw_z0_from_cross_section(
    cross_section: CrossSectionSpec,
    f: ArrayLike | None = None,
) -> jnp.ndarray:
    """Characteristic impedance of a CPW defined by a layout cross-section.

    Args:
        cross_section: A gdsfactory cross-section specification.
        f: Frequency array (Hz). Used only to determine the output shape;
           the impedance is frequency-independent in the quasi-static model.

    Returns:
        Characteristic impedance broadcast to the shape of *f* (Ω).
    """
    width, gap = get_cpw_dimensions(cross_section)
    _ep_eff, z0_val = cpw_parameters(width, gap)
    z0 = jnp.asarray(z0_val)
    if f is not None:
        f = jnp.asarray(f)
        z0 = jnp.broadcast_to(z0, f.shape)
    return z0


def cpw_ep_r_from_cross_section(
    cross_section: CrossSectionSpec,  # noqa: ARG001
) -> float:
    r"""Substrate relative permittivity for a given cross-section.

    .. note::
        The substrate permittivity is determined by the PDK layer stack
        (``LAYER_STACK["Substrate"]``), not by the cross-section geometry.
        All CPW cross-sections on the same substrate share the same
        :math:`\varepsilon_r`.  The *cross_section* parameter is accepted
        for API symmetry with :func:`cpw_z0_from_cross_section`.

    Args:
        cross_section: A gdsfactory cross-section specification.

    Returns:
        Relative permittivity of the substrate.
    """
    _h, _t, ep_r = get_cpw_substrate_params()
    return ep_r
