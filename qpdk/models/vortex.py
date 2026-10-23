r"""Models for resonators with vortex-trapping geometries.

S-parameter models for the cells in :mod:`qpdk.cells.vortex`: the taper and
neck termination of Shulga *et al.* :cite:`shulgaObservationIndividualVortex2026`
and the centre-conductor bulge of Nsanzineza and Plourde
:cite:`nsanzinezaTrappingSingleVortex2014`.

Every section is a cascade of uniform CPW segments, all referenced to the
impedance of the resonator's cross-section so that the impedance steps
between segments are included. Inside the etched window of the neck
termination the ground planes sit at the window edges, so the taper and neck are modelled as a CPW whose
gap is :math:`(W - w)/2` for window width :math:`W` and local strip width
:math:`w`. The models describe the vortex-free state: they include neither
the impedance of trapped vortices nor the kinetic inductance of the narrow
neck, which raises its inductance per unit length by a few percent for a
2 µm Nb strip.
"""

import math
from typing import Literal

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from sax.models.rf import electrical_short

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import (
    cpw_epsilon_eff,
    cpw_thickness_correction,
    get_cpw_dimensions,
    get_cpw_substrate_params,
    propagation_constant,
    transmission_line_s_params,
)
from qpdk.models.waveguides import straight

__all__ = [
    "cpw_bulge",
    "neck_termination",
    "resonator_half_wave_bulge",
    "resonator_quarter_wave_neck",
]


def _taper_widths(
    width1: float,
    width2: float,
    npoints: int,
    shape: Literal["linear", "smooth"],
) -> list[float]:
    """Return strip widths at the centres of ``npoints`` equal taper segments.

    The widths stay Python floats, so the geometry is static under
    :func:`jax.jit`.
    """
    widths = []
    for i in range(npoints):
        u = (i + 0.5) / npoints
        fraction = u if shape == "linear" else (1 - math.cos(math.pi * u)) / 2
        widths.append(width1 + (width2 - width1) * fraction)
    return widths


def _cpw_line(
    f: sax.FloatArrayLike, width: float, gap: float
) -> tuple[jax.Array, jax.Array]:
    """Return the propagation constant (1/m) and impedance (Ω) of a CPW.

    Uses the same PDK layer stack and formulae as
    :func:`~qpdk.models.waveguides.straight`.
    """
    h, t, ep_r, tand = get_cpw_substrate_params()
    w_m, s_m = width * 1e-6, gap * 1e-6
    ep_eff = cpw_epsilon_eff(w_m, s_m, h * 1e-6, ep_r)
    ep_eff_t, z0 = cpw_thickness_correction(w_m, s_m, t * 1e-6, ep_eff)
    gamma = propagation_constant(jnp.asarray(f), ep_eff_t, tand=tand, ep_r=ep_r)
    return gamma, z0


def _cascade(
    f: sax.FloatArrayLike,
    cross_section: CrossSectionSpec,
    sections: list[tuple[float, float, float]],
) -> sax.SDict:
    """Cascade CPW sections given as ``(length, width, gap)`` tuples in µm.

    :func:`~qpdk.models.waveguides.straight` references each line to its own
    characteristic impedance, which hides the reflection at an impedance
    step. Here every section is referenced to the impedance of
    ``cross_section``, so the cascade includes the steps between sections
    and a section of ``cross_section`` itself matches ``straight``.

    Returns:
        Two-port S-parameters with ports ``o1`` and ``o2``.
    """
    f = jnp.asarray(f)
    _, z_ref = _cpw_line(f, *get_cpw_dimensions(cross_section))
    instances: dict[str, sax.SType] = {}
    for i, (length, width, gap) in enumerate(sections):
        gamma, z0 = _cpw_line(f, width, gap)
        s11, s21 = transmission_line_s_params(gamma, z0, length * 1e-6, z_ref)
        instances[f"s{i}"] = sax.reciprocal({
            ("o1", "o1"): s11.reshape(f.shape),
            ("o1", "o2"): s21.reshape(f.shape),
            ("o2", "o2"): s11.reshape(f.shape),
        })
    connections = {f"s{i},o2": f"s{i + 1},o1" for i in range(len(sections) - 1)}
    ports = {"o1": "s0,o1", "o2": f"s{len(sections) - 1},o2"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


def _neck_sections(
    cross_section: CrossSectionSpec,
    taper_length: float,
    taper_shape: Literal["linear", "smooth"],
    neck_width: float,
    neck_length: float,
    window_width: float,
    npoints: int,
) -> list[tuple[float, float, float]]:
    """Return the CPW sections of the taper and neck inside the window.

    Raises:
        ValueError: If the strip does not fit inside the window.
    """
    width, _ = get_cpw_dimensions(cross_section)
    if not 0 < neck_width < width < window_width:
        raise ValueError(
            "Need 0 < neck_width < CPW width < window_width, got "
            f"{neck_width}, {width} and {window_width}."
        )
    npoints = max(npoints, 1)
    taper = [
        (taper_length / npoints, w, (window_width - w) / 2)
        for w in _taper_widths(width, neck_width, npoints, taper_shape)
    ]
    return [*taper, (neck_length, neck_width, (window_width - neck_width) / 2)]


def _bulge_sections(
    cross_section: CrossSectionSpec,
    bulge_width: float,
    bulge_length: float,
    taper_length: float,
    npoints: int,
) -> list[tuple[float, float, float]]:
    """Return the CPW sections of the bulge, with a constant gap."""
    width, gap = get_cpw_dimensions(cross_section)
    npoints = max(npoints, 1)
    widths = _taper_widths(width, bulge_width, npoints, "linear")
    taper_in = [(taper_length / npoints, w, gap) for w in widths]
    taper_out = [(taper_length / npoints, w, gap) for w in reversed(widths)]
    return [*taper_in, (bulge_length, bulge_width, gap), *taper_out]


def neck_termination(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    cross_section: CrossSectionSpec = "cpw",
    taper_length: float = 100.0,
    taper_shape: Literal["linear", "smooth"] = "linear",
    neck_width: float = 2.0,
    neck_length: float = 325.0,
    window_width: float = 380.0,
    npoints: int = 10,
) -> sax.SDict:
    r"""S-parameter model for a CPW shorted through a taper and a narrow neck.

    The taper and neck run inside an etched window of width :math:`W`, so each
    segment is a CPW of strip width :math:`w` and gap :math:`(W - w)/2`. The
    neck joins the ground plane at the far end of the window, modelled as an
    ideal short. See :func:`qpdk.cells.vortex.neck_termination` for the
    geometry and :cite:`shulgaObservationIndividualVortex2026` for the
    measurement.

    Args:
        f: Array of frequency points in Hz.
        cross_section: CPW cross-section at port ``o1``.
        taper_length: Length of the taper in µm.
        taper_shape: ``"linear"`` or ``"smooth"`` (raised-cosine) width
            profile.
        neck_width: Width of the neck in µm.
        neck_length: Length of the neck in µm.
        window_width: Width of the etched window across the line in µm.
        npoints: Number of segments the taper is divided into.

    Returns:
        sax.SDict: One-port S-parameters with port ``o1``.
    """
    sections = _neck_sections(
        cross_section,
        taper_length,
        taper_shape,
        neck_width,
        neck_length,
        window_width,
        npoints,
    )
    instances: dict[str, sax.SType] = {
        "line": _cascade(f, cross_section, sections),
        "short": electrical_short(f=f),
    }
    connections = {"line,o2": "short,o1"}
    ports = {"o1": "line,o1"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


def cpw_bulge(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    cross_section: CrossSectionSpec = "cpw",
    bulge_width: float = 25.0,
    bulge_length: float = 50.0,
    taper_length: float = 20.0,
    npoints: int = 5,
) -> sax.SDict:
    r"""S-parameter model for a CPW with a locally widened centre conductor.

    Linear tapers to and from a section of width ``bulge_width``, all with the
    gap of ``cross_section``, matching :func:`qpdk.cells.vortex.cpw_bulge`.
    See :cite:`nsanzinezaTrappingSingleVortex2014` for the measurement.

    Args:
        f: Array of frequency points in Hz.
        cross_section: CPW cross-section at both ports.
        bulge_width: Width of the widened centre conductor in µm.
        bulge_length: Length of the full-width section in µm.
        taper_length: Length of each linear taper in µm.
        npoints: Number of segments each taper is divided into.

    Returns:
        sax.SDict: Two-port S-parameters with ports ``o1`` and ``o2``.
    """
    return _cascade(
        f,
        cross_section,
        _bulge_sections(
            cross_section, bulge_width, bulge_length, taper_length, npoints
        ),
    )


def resonator_quarter_wave_neck(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 4000.0,
    cross_section: CrossSectionSpec = "cpw",
    taper_length: float = 100.0,
    taper_shape: Literal["linear", "smooth"] = "linear",
    neck_width: float = 2.0,
    neck_length: float = 325.0,
    window_width: float = 380.0,
    npoints: int = 10,
) -> sax.SDict:
    """S-parameter model for a quarter-wave resonator grounded through a neck.

    A CPW of ``length`` followed by :func:`neck_termination`. Port ``o1`` is
    the open end, as in :func:`qpdk.cells.vortex.resonator_quarter_wave_neck`.

    Args:
        f: Array of frequency points in Hz.
        length: CPW length from the open end to the start of the taper in µm.
        cross_section: CPW cross-section of the resonator.
        taper_length: See :func:`neck_termination`.
        taper_shape: See :func:`neck_termination`.
        neck_width: See :func:`neck_termination`.
        neck_length: See :func:`neck_termination`.
        window_width: See :func:`neck_termination`.
        npoints: See :func:`neck_termination`.

    Returns:
        sax.SDict: One-port S-parameters with port ``o1``.
    """
    instances: dict[str, sax.SType] = {
        "line": straight(f=f, length=length, cross_section=cross_section),
        "neck": neck_termination(
            f=f,
            cross_section=cross_section,
            taper_length=taper_length,
            taper_shape=taper_shape,
            neck_width=neck_width,
            neck_length=neck_length,
            window_width=window_width,
            npoints=npoints,
        ),
    }
    connections = {"line,o2": "neck,o1"}
    ports = {"o1": "line,o1"}
    return sax.evaluate_circuit_fg((connections, ports), instances)


def resonator_half_wave_bulge(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 8000.0,
    cross_section: CrossSectionSpec = "cpw",
    bulge_width: float = 25.0,
    bulge_length: float = 50.0,
    taper_length: float = 20.0,
    npoints: int = 5,
) -> sax.SDict:
    """S-parameter model for a half-wave resonator with a central bulge.

    Two equal CPW sections on either side of :func:`cpw_bulge`, with open
    ends at ports ``o1`` and ``o2``.

    Args:
        f: Array of frequency points in Hz.
        length: Total centre-line length, including the bulge, in µm.
        cross_section: CPW cross-section of the resonator.
        bulge_width: See :func:`cpw_bulge`.
        bulge_length: See :func:`cpw_bulge`.
        taper_length: See :func:`cpw_bulge`.
        npoints: See :func:`cpw_bulge`.

    Returns:
        sax.SDict: Two-port S-parameters with ports ``o1`` and ``o2``.

    Raises:
        ValueError: If a concrete ``length`` is shorter than the bulge.
    """
    side_length = (length - bulge_length - 2 * taper_length) / 2
    if isinstance(side_length, int | float) and side_length <= 0:
        raise ValueError(
            f"length ({length}) must exceed the bulge length "
            f"({bulge_length + 2 * taper_length})."
        )
    instances: dict[str, sax.SType] = {
        "left": straight(f=f, length=side_length, cross_section=cross_section),
        "bulge": cpw_bulge(
            f=f,
            cross_section=cross_section,
            bulge_width=bulge_width,
            bulge_length=bulge_length,
            taper_length=taper_length,
            npoints=npoints,
        ),
        "right": straight(f=f, length=side_length, cross_section=cross_section),
    }
    connections = {"left,o2": "bulge,o1", "bulge,o2": "right,o1"}
    ports = {"o1": "left,o1", "o2": "right,o2"}
    return sax.evaluate_circuit_fg((connections, ports), instances)
