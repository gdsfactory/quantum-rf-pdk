"""Waveguides."""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import (
    coplanar_waveguide as _coplanar_waveguide,
)
from sax.models.rf import (
    electrical_open,
    electrical_short,
)
from sax.models.rf import (
    microstrip as _microstrip,
)

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0, π
from qpdk.models.generic import admittance, short_2_port
from qpdk.models.media import (
    get_cpw_dimensions,
    get_cpw_substrate_params,
)
from qpdk.tech import coplanar_waveguide as tech_coplanar_waveguide


def straight(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for a straight coplanar waveguide."""
    width, gap = get_cpw_dimensions(cross_section)
    _h, _t, ep_r = get_cpw_substrate_params()
    return _coplanar_waveguide(
        f=f,
        length=length,
        width=width,
        gap=gap,
        thickness=_t,
        substrate_thickness=_h,
        ep_r=ep_r,
        tand=0.0,
    )


def straight_microstrip(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: sax.Float = 1000,
    width: sax.Float = 10.0,
    h: sax.Float = 500.0,
    t: sax.Float = 0.2,
    ep_r: sax.Float = 11.45,
    tand: sax.Float = 0.0,
) -> sax.SDict:
    """S-parameter model for a straight microstrip line."""
    return _microstrip(
        f=f,
        length=length,
        width=width,
        substrate_thickness=h,
        thickness=t,
        ep_r=ep_r,
        tand=tand,
    )


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
    instances = {
        "straight": straight(f=f, length=length, cross_section=cross_section),
        "short": short_2_port(f=f),
    }
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
    instances = {
        "straight": straight(f=f, length=length, cross_section=cross_section),
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
    instances = {
        "straight": straight(f=f, length=length, cross_section=cross_section),
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
) -> sax.SType:
    """S-parameter model for a 3-port tee junction.

    This wraps the generic tee model.

    Args:
        f: Array of frequency points in Hz.

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
def airbridge(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    cpw_width: sax.Float = 10.0,
    bridge_width: sax.Float = 10.0,
    airgap_height: sax.Float = 3.0,
    loss_tangent: sax.Float = 1.2e-8,
) -> sax.SType:
    r"""S-parameter model for a superconducting CPW airbridge.

    The airbridge is modeled as a lumped lossy shunt admittance (accounting for
    dielectric loss and shunt capacitance) embedded between two sections of
    transmission line that represent the physical footprint of the bridge.

    Parallel plate capacitor model is as done in :cite:`chenFabricationCharacterizationAluminum2014`
    The default value for the loss tangent :math:`\tan\,\delta` is also taken from there.

    Args:
        f: Array of frequency points in Hz
        cpw_width: Width of the CPW center conductor in µm.
        bridge_width: Width of the airbridge in µm.
        airgap_height: Height of the airgap in µm.
        loss_tangent: Dielectric loss tangent of the supporting layer/residues.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    f = jnp.asarray(f)
    ω = 2 * π * f

    # Parallel plate capacitance
    c_pp = (ε_0 * cpw_width * 1e-6 * bridge_width * 1e-6) / (airgap_height * 1e-6)

    # Heuristics: fringing capacitance assumed to be 20% of the parallel plate for small bridges.
    c_bridge = c_pp * 1.2

    # Admittance of the bridge (Conductance from dielectric loss + Susceptance)
    Y_bridge = ω * c_bridge * (loss_tangent + 1j)

    return admittance(f=f, y=Y_bridge)


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
    return straight(f=f, length=length, cross_section=cross_section)


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
    return straight(f=f, length=length, cross_section=cross_section)


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
    return straight(f=f, length=length, cross_section=cross_section)


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
    return straight(f=f, length=length, cross_section=cross_section)


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
    n_points = int(n_points)
    w1, g1 = get_cpw_dimensions(cross_section_1)
    w2, g2 = get_cpw_dimensions(cross_section_2)

    f = jnp.asarray(f)
    segment_length = length / n_points

    ws = jnp.linspace(w1, w2, n_points)
    gs = jnp.linspace(g1, g2, n_points)

    instances = {
        f"straight_{i}": straight(
            f=f,
            length=segment_length,
            cross_section=tech_coplanar_waveguide(width=float(ws[i]), gap=float(gs[i])),
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
        cross_section_big = tech_coplanar_waveguide(width=200, gap=100)

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
