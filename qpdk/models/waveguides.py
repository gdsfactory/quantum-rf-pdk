"""Waveguides."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import electrical_open, electrical_short
from skrf import Frequency

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0, π
from qpdk.models.generic import short_2_port
from qpdk.models.media import cross_section_to_media
from qpdk.tech import coplanar_waveguide


def straight(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a straight waveguide.

    See `scikit-rf <skrf>`_ for details on analytical formulæ.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary

    .. _skrf: https://scikit-rf.org/
    """
    f = jnp.asarray(f)
    f_flat = f.ravel()
    # Keep f as tuple for scikit-rf, convert to array only for final JAX operations
    media = cross_section_to_media(cross_section)
    skrf_media = media(frequency=Frequency.from_f(np.asarray(f_flat), unit="Hz"))
    transmission_line = skrf_media.line(d=np.asarray(length), unit="um")
    sdict = {
        ("o1", "o1"): jnp.array(transmission_line.s[:, 0, 0]).reshape(*f.shape),
        ("o1", "o2"): jnp.array(transmission_line.s[:, 0, 1]).reshape(*f.shape),
        ("o2", "o2"): jnp.array(transmission_line.s[:, 1, 1]).reshape(*f.shape),
    }
    return sax.reciprocal(sdict)


def straight_shorted(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a straight waveguide with one shorted end.

    This may be used to model a quarter-wave coplanar waveguide resonator.

    Note:
        The port ``o2`` is internally shorted and should not be used.
        It seems to be a Sax limitation that we need to define at least two ports.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    instances = {"straight": straight(**kwargs), "short": short_2_port(f=f)}
    connections = {
        "straight,o2": "short,o1",
    }
    ports = {
        "o1": "straight,o1",
        "o2": "short_2_port,o2",  # don't use: shorted!
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def straight_open(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a straight waveguide with one open end.

    Note:
        The port ``o2`` is internally open-circuited and should not be used.
        It is provided to match the number of ports in the layout component.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    instances = {
        "straight": straight(**kwargs),
        "open": electrical_open(f=f, n_ports=2),
    }
    connections = {
        "straight,o2": "open,o1",
    }
    ports = {
        "o1": "straight,o1",
        "o2": "open,o2",  # don't use: opened!
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def straight_double_open(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a straight waveguide with open ends.

    Note:
        Ports ``o1`` and ``o2`` are internally open-circuited and should not be used.
        They are provided to match the number of ports in the layout component.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SType: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    instances = {
        "straight": straight(**kwargs),
        "open1": electrical_open(f=f, n_ports=2),
        "open2": electrical_open(f=f, n_ports=2),
    }
    connections = {
        "straight,o1": "open1,o1",
        "straight,o2": "open2,o1",
    }
    ports = {
        "o1": "open1,o2",  # don't use: opened!
        "o2": "open2,o2",  # don't use: opened!
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def tee(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    _cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """S-parameter model for a 3-port tee junction.

    This wraps the generic tee model.

    Args:
        f: Array of frequency points in Hz.
        _cross_section: The cross-section of the waveguide (ignored for ideal model).

    Returns:
        sax.SType: S-parameters dictionary.
    """
    from qpdk.models.generic import tee as _generic_tee

    return _generic_tee(f=f)


@partial(jax.jit, static_argnames=["west", "east", "north", "south"])
def nxn(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    west: int = 1,
    east: int = 1,
    north: int = 1,
    south: int = 1,
    _cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """NxN junction model using tee components.

    This model creates an N-port divider/combiner by chaining 3-port tee
    junctions. All ports are connected to a single node.

    Args:
        f: Array of frequency points in Hz.
        west: Number of ports on the west side.
        east: Number of ports on the east side.
        north: Number of ports on the north side.
        south: Number of ports on the south side.
        _cross_section: The cross-section of the waveguide (ignored for ideal model).

    Returns:
        sax.SType: S-parameters dictionary with ports o1, o2, ..., oN.
    """
    from qpdk.models.generic import tee as _generic_tee

    f = jnp.asarray(f)
    n_ports = west + east + north + south

    if n_ports <= 0:
        raise ValueError("Total number of ports must be positive.")
    if n_ports == 1:
        return electrical_open(f=f)
    if n_ports == 2:
        return electrical_short(f=f, n_ports=2)

    instances = {f"tee_{i}": _generic_tee(f=f) for i in range(n_ports - 2)}
    connections = {f"tee_{i},o3": f"tee_{i + 1},o1" for i in range(n_ports - 3)}

    ports = {
        "o1": "tee_0,o1",
        "o2": "tee_0,o2",
    }
    for i in range(1, n_ports - 2):
        ports[f"o{i + 2}"] = f"tee_{i},o2"

    # Last tee's o3 is the last external port
    ports[f"o{n_ports}"] = f"tee_{n_ports - 3},o3"

    return sax.evaluate_circuit_fg((connections, ports), instances)


@partial(jax.jit, inline=True)
def _superconducting_airbridge_shunt(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    cpw_width: sax.Float = 10.0,
    bridge_width: sax.Float = 10.0,
    airgap_height: sax.Float = 3.0,
    loss_tangent: sax.Float = 1.2e-8,
    z0: sax.Complex = 50.0,
) -> sax.SType:
    """S-parameter model for a superconducting CPW airbridge shunt admittance.

    Modeled as a lossy shunt capacitor.
    """
    f = jnp.asarray(f)
    ω = 2 * π * f

    # Parallel plate capacitance
    c_pp = (ε_0 * cpw_width * 1e-6 * bridge_width * 1e-6) / (airgap_height * 1e-6)

    # Heuristics: fringing capacitance assumed to be 20% of the parallel plate for small bridges.
    c_bridge = c_pp * 1.2

    # 2. Admittance of the bridge (Conductance from dielectric loss + Susceptance)
    Y_bridge = ω * c_bridge * (loss_tangent + 1j)

    # Normalized admittance
    y = Y_bridge * z0

    # 3. S-parameters for a shunt element
    denom = 2.0 + y
    s11 = -y / denom
    s21 = 2.0 / denom

    return {
        ("o1", "o1"): s11,
        ("o1", "o2"): s21,
        ("o2", "o1"): s21,
        ("o2", "o2"): s11,
    }


def airbridge(
    f: ArrayLike = DEFAULT_FREQUENCY,
    bridge_length: float = 30.0,
    bridge_width: float = 8.0,
    pad_width: float = 15.0,
    loss_tangent: float = 1.2e-8,
    airgap_height: float = 3.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""S-parameter model for a superconducting CPW airbridge.

    The airbridge is modeled as a lumped lossy shunt admittance (accounting for
    dielectric loss and shunt capacitance) embedded between two sections of
    transmission line that represent the physical footprint of the bridge.

    Parallel plate capacitor model is as done in :cite:`chenFabricationCharacterizationAluminum2014`
    The default value for the loss tangent :math:`\tan\,\delta` is also taken from there.

    Args:
        f: Array of frequency points in Hz
        bridge_length: Length of the airbridge in µm.
        bridge_width: Width of the airbridge in µm.
        pad_width: Width of the landing pads in µm.
        loss_tangent: Dielectric loss tangent of the supporting layer/residues.
        airgap_height: Height of the airgap in µm.
        cross_section: The cross-section of the CPW under the bridge.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    import gdsfactory as gf

    if pad_width <= bridge_width:
        raise ValueError(
            f"pad_width ({pad_width}) must be greater than bridge_width ({bridge_width})"
        )

    # Determine CPW trace width from the provided cross_section
    if isinstance(cross_section, gf.CrossSection):
        xs = cross_section
    elif callable(cross_section):
        xs = cast(gf.CrossSection, cross_section())
    else:
        xs = gf.get_cross_section(cross_section)

    cpw_width = xs.width

    # Create the shunt component
    shunt = _superconducting_airbridge_shunt(
        f=f,
        cpw_width=cpw_width,
        bridge_width=bridge_width,
        airgap_height=airgap_height,
        loss_tangent=loss_tangent,
    )

    # Transmission line segments under the bridge
    bridge_cross_section = coplanar_waveguide(
        width=bridge_width,
        gap=(pad_width - bridge_width) / 2,
    )
    half_length = bridge_length / 2

    instances = {
        "line1": straight(f=f, length=half_length, cross_section=bridge_cross_section),
        "shunt": shunt,
        "line2": straight(f=f, length=half_length, cross_section=bridge_cross_section),
    }

    connections = {
        "line1,o2": "shunt,o1",
        "shunt,o2": "line2,o1",
    }

    ports = {
        "o1": "line1,o1",
        "o2": "line2,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def tsv(
    f: ArrayLike = DEFAULT_FREQUENCY,
    via_height: float = 1000.0,
) -> sax.SDict:
    """S-parameter model for a through-silicon via (TSV), wrapped to :func:`~straight`.

    TODO: add a constant loss channel for TSVs.

    Args:
        f: Array of frequency points in Hz
        via_height: Physical height (length) of the TSV in µm.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    return straight(f=f, length=via_height)


def indium_bump(
    f: ArrayLike = DEFAULT_FREQUENCY,
    bump_height: float = 10.0,
) -> sax.SType:
    """S-parameter model for an indium bump, wrapped to :func:`~straight`.

    TODO: add a constant loss channel for indium bumps.

    Args:
        f: Array of frequency points in Hz
        bump_height: Physical height (length) of the indium bump in µm.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return straight(f=f, length=bump_height)


def bend_circular(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a circular bend, wrapped to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)


def bend_euler(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for an Euler bend, wrapped to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)


def bend_s(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for an S-bend, wrapped to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)  # pyrefly: ignore[bad-keyword-argument]


def rectangle(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a rectangular section, wrapped to :func:`~straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section: The cross-section of the waveguide.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    kwargs = {
        "f": jnp.asarray(f),
        "length": jnp.asarray(length),
        "cross_section": cross_section,
    }
    return straight(**kwargs)  # pyrefly: ignore[bad-keyword-argument]


def taper_cross_section(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section_1: CrossSectionSpec = "cpw",
    cross_section_2: CrossSectionSpec = "cpw",
    n_points: int = 50,
) -> sax.SDict:
    """S-parameter model for a cross-section taper using linear interpolation.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        cross_section_1: Cross-section for the start of the taper.
        cross_section_2: Cross-section for the end of the taper.
        n_points: Number of segments to divide the taper into for simulation.

    Returns:
        sax.SDict: S-parameters dictionary
    """

    def get_width_gap(cs: CrossSectionSpec) -> tuple[float, float]:
        import gdsfactory as gf

        if isinstance(cs, gf.CrossSection):
            xs = cs
        elif callable(cs):
            xs = cast(gf.CrossSection, cs())
        else:
            xs = gf.get_cross_section(cs)

        # Infer from first section with "etch_offset" in the name
        width = xs.width
        try:
            gap = next(
                section.width
                for section in xs.sections
                if section.name and "etch_offset" in section.name
            )
        except StopIteration:
            raise ValueError(
                f"Cannot extract CPW gap from cross-section {cs!r}: "
                "no section with 'etch_offset' in its name found. "
                "Only coplanar_waveguide cross-sections are supported."
            ) from None
        return width, gap

    w1, g1 = get_width_gap(cross_section_1)
    w2, g2 = get_width_gap(cross_section_2)

    f = jnp.asarray(f)
    segment_length = length / n_points

    ws = jnp.linspace(w1, w2, n_points)
    gs = jnp.linspace(g1, g2, n_points)

    instances = {
        f"straight_{i}": straight(
            f=f,
            length=segment_length,
            cross_section=coplanar_waveguide(width=float(ws[i]), gap=float(gs[i])),
        )
        for i in range(n_points)
    }
    connections = {
        f"straight_{i},o2": f"straight_{i + 1},o1" for i in range(n_points - 1)
    }
    ports = {
        "o1": "straight_0,o1",
        "o2": f"straight_{n_points - 1},o2",
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)


def launcher(
    f: ArrayLike = DEFAULT_FREQUENCY,
    straight_length: sax.Float = 200.0,
    taper_length: sax.Float = 100.0,
    cross_section_big: CrossSectionSpec | None = None,
    cross_section_small: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a launcher, effectively a straight section followed by a taper.

    Args:
        f: Array of frequency points in Hz
        straight_length: Length of the straight section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_big: Cross-section for the wide section.
        cross_section_small: Cross-section for the narrow section.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f = jnp.asarray(f)
    if cross_section_big is None:
        cross_section_big = coplanar_waveguide(width=200, gap=100)

    instances = {
        "straight": straight(
            f=f,
            length=straight_length,
            cross_section=cross_section_big,
        ),
        "taper": taper_cross_section(
            f=f,
            length=taper_length,
            cross_section_1=cross_section_big,
            cross_section_2=cross_section_small,
        ),
    }
    connections = {
        "straight,o2": "taper,o1",
    }
    ports = {
        "waveport": "straight,o1",
        "o1": "taper,o2",
    }
    return sax.backends.evaluate_circuit_fg((connections, ports), instances)
