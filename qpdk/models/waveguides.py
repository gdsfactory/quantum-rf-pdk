"""S-parameter model for a straight waveguide."""

from typing import TypedDict, Unpack

import jax
import jax.numpy as jnp
import jax.scipy.interpolate
import sax
from jax.typing import ArrayLike
from skrf import Frequency

from qpdk.models.generic import short_2_port
from qpdk.models.media import MediaCallable, cpw_media_skrf


class StraightModelKwargs(TypedDict, total=False):
    """Type definition for straight S-parameter model keyword arguments."""

    f: ArrayLike
    length: int | float
    media: MediaCallable


# JIT disabled for now due to scikit-rf internals not being JAX-compatible
# @partial(jax.jit, static_argnames=["media"])
def straight(
    f: ArrayLike = jnp.array([5e9]),
    length: int | float = 1000,
    media: MediaCallable = cpw_media_skrf(),
) -> sax.SType:
    """S-parameter model for a straight waveguide.

    See `scikit-rf <skrf>`_ for details on analytical formulæ.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        media: Function returning a scikit-rf :class:`~Media` object after called
            with ``frequency=f``. If None, uses default CPW media.

    Returns:
        sax.SType: S-parameters dictionary

    .. _skrf: https://scikit-rf.org/
    """
    # Keep f as tuple for scikit-rf, convert to array only for final JAX operations
    skrf_media = media(frequency=Frequency.from_f(f, unit="Hz"))
    transmission_line = skrf_media.line(d=length, unit="um")
    sdict = {
        ("o1", "o1"): jnp.array(transmission_line.s[:, 0, 0]),
        ("o1", "o2"): jnp.array(transmission_line.s[:, 0, 1]),
        ("o2", "o2"): jnp.array(transmission_line.s[:, 1, 1]),
    }
    return sax.reciprocal(sdict)


def straight_shorted(
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for a straight waveguide with one shorted end.

    This may be used to model a quarter-wave coplanar waveguide resonator.

    Note:
        The port ``o2`` is internally shorted and should not be used.
        It seems to be a Sax limitation that we need to define at least two ports.
    """
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "straight": {
                    "component": "straight",
                    "settings": kwargs,
                },
                "short_2_port": "short_2_port",
            },
            "connections": {
                "straight,o2": "short_2_port,o1",
            },
            "ports": {
                "o1": "straight,o1",
                # This port should never be used since it's shorted
                "o2": "short_2_port,o2",
            },
        },
        models={
            "straight": straight,
            "short_2_port": short_2_port,
        },
    )
    return circuit(f=kwargs.get("f", jnp.array([5e9])))


def bend_circular(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for a circular bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)  # pyrefly: ignore[bad-keyword-argument]


def bend_euler(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for an Euler bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)  # pyrefly: ignore[bad-keyword-argument]


def bend_s(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for an S-bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)  # pyrefly: ignore[bad-keyword-argument]


def taper_cross_section(
    f: ArrayLike = jnp.array([5e9]),
    length: int | float = 1000,
    media_1: MediaCallable = cpw_media_skrf(),
    media_2: MediaCallable = cpw_media_skrf(),
    n_points: int = 50,
) -> sax.SType:
    """S-parameter model for a cross-section taper using linear interpolation.

    Uses jax.scipy.interpolate.RegularGridInterpolator to efficiently interpolate
    media parameters (width and gap) along the taper length.

    Args:
        f: Array of frequency points in Hz
        length: Physical length in µm
        media_1: Function returning a scikit-rf :class:`~Media` object after called
            with ``frequency=f`` for the start of the taper.
        media_2: Function returning a scikit-rf :class:`~Media` object after called
            with ``frequency=f`` for the end of the taper.
        n_points: Number of segments to divide the taper into for simulation.
    """
    # Ensure n_points is a concrete Python int
    n_points = int(n_points)

    # Get media parameters at the start and end of the taper
    dummy_freq = Frequency.from_f(f, unit="Hz")
    media_1_obj = media_1(frequency=dummy_freq)
    media_2_obj = media_2(frequency=dummy_freq)

    width_1 = media_1_obj.w
    width_2 = media_2_obj.w
    gap_1 = media_1_obj.s
    gap_2 = media_2_obj.s

    # Create interpolation grid points using physical positions
    position_grid = jnp.array([0.0, length])
    width_values = jnp.array([width_1, width_2])
    gap_values = jnp.array([gap_1, gap_2])

    # Create interpolators for width and gap
    width_interpolator = jax.scipy.interpolate.RegularGridInterpolator(
        (position_grid,), width_values, method="linear"
    )
    gap_interpolator = jax.scipy.interpolate.RegularGridInterpolator(
        (position_grid,), gap_values, method="linear"
    )

    segment_length = length / n_points
    # Compute physical positions for each segment
    positions = jnp.linspace(0, length, num=n_points)

    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                **{
                    f"straight_{i}": {
                        "component": "straight",
                        "settings": {
                            "f": f,
                            "length": segment_length,
                            "media": lambda frequency, i=i: cpw_media_skrf(
                                width=float(
                                    width_interpolator(jnp.array([positions[i]]))[0]
                                ),
                                gap=float(
                                    gap_interpolator(jnp.array([positions[i]]))[0]
                                ),
                            )(frequency=frequency),
                        },
                    }
                    for i in range(n_points)
                }
            },
            "connections": {
                **{
                    f"straight_{i},o2": f"straight_{i + 1},o1"
                    for i in range(n_points - 1)
                }
            },
            "ports": {
                "o1": "straight_0,o1",
                "o2": f"straight_{n_points - 1},o2",
            },
        },
        models={
            "straight": straight,
        },
    )

    return circuit(f=f)


def rectangle(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for a rectangular section, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)  # pyrefly: ignore[bad-keyword-argument]


def launcher(
    f: ArrayLike = jnp.array([5e9]),
    straight_length: float = 200.0,
    taper_length: float = 100.0,
    media_big: MediaCallable = cpw_media_skrf(width=200, gap=100),
    media_small: MediaCallable = cpw_media_skrf(),
) -> sax.SType:
    """S-parameter model for a launcher, effectively a straight section followed by a taper.

    Args:
        straight_length: Length of the straight section in µm.
        taper_length: Length of the taper section in µm.
        media_big: Media callable for the wide section.
        media_small: Media callable for the narrow section.

    Returns:
        sax.SType: S-parameters dictionary
    """
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "straight": {
                    "component": "straight",
                    "settings": {
                        "length": straight_length,
                        "media": media_big,
                    },
                },
                "taper": {
                    "component": "taper_cross_section",
                    "settings": {
                        "length": taper_length,
                        "media_1": media_big,
                        "media_2": media_small,
                    },
                },
            },
            "connections": {
                "straight,o2": "taper,o1",
            },
            "ports": {
                "o1": "straight,o1",
                "o2": "taper,o2",
            },
        },
        models={
            "straight": straight,
            "taper_cross_section": taper_cross_section,
        },
    )
    return circuit(f=f)


if __name__ == "__main__":
    import time

    from tqdm import tqdm

    cpw = cpw_media_skrf(width=10, gap=6)

    def straight_no_jit(
        f: ArrayLike = jnp.array([5e9]),
        length: int | float = 1000,
        media: MediaCallable = cpw_media_skrf(),
    ) -> sax.SType:
        """Version of straight without just-in-time compilation."""
        skrf_media = media(frequency=Frequency.from_f(f, unit="Hz"))
        transmission_line = skrf_media.line(d=length, unit="um")
        sdict = {
            ("o1", "o1"): jnp.array(transmission_line.s[:, 0, 0]),
            ("o1", "o2"): jnp.array(transmission_line.s[:, 0, 1]),
            ("o2", "o2"): jnp.array(transmission_line.s[:, 1, 1]),
        }
        return sax.reciprocal(sdict)

    test_freq = jnp.linspace(0.5e9, 9e9, 200001)
    test_length = 1000

    print("Benchmarking jitted vs non-jitted performance…")

    n_runs = 10

    jit_times = []
    for _ in tqdm(range(n_runs), desc="With jax.jit", ncols=80, unit="run"):
        start_time = time.perf_counter()
        S_jit = straight(f=test_freq, length=test_length, media=cpw)
        _ = S_jit["o2", "o1"].block_until_ready()
        end_time = time.perf_counter()
        jit_times.append(end_time - start_time)

    no_jit_times = []
    for _ in tqdm(range(n_runs), desc="Without jax.jit", ncols=80, unit="run"):
        start_time = time.perf_counter()
        S_no_jit = straight_no_jit(f=test_freq, length=test_length, media=cpw)
        _ = S_no_jit["o2", "o1"].block_until_ready()
        end_time = time.perf_counter()
        no_jit_times.append(end_time - start_time)

    jit_times_steady = jit_times[1:]
    avg_jit = sum(jit_times_steady) / len(jit_times_steady)
    avg_no_jit = sum(no_jit_times) / len(no_jit_times)
    speedup = avg_no_jit / avg_jit

    print(f"Jitted: {avg_jit:.4f}s avg (excl. first), {jit_times[0]:.3f}s first run")
    print(f"Non-jitted: {avg_no_jit:.4f}s avg")
    print(f"Speedup: {speedup:.1f}x")

    S_jit = straight(f=test_freq, length=test_length, media=cpw)
    S_no_jit = straight_no_jit(f=test_freq, length=test_length, media=cpw)
    max_diff = jnp.max(jnp.abs(S_jit["o2", "o1"] - S_no_jit["o2", "o1"]))
    print(f"Max absolute difference in results: {max_diff:.2e}")

    try:
        s21_array = S_jit["o2", "o1"]
        s21_gpu = jax.device_put(s21_array, jax.devices("gpu")[0])
        print(f"GPU available: {s21_gpu.device}")
    except Exception:
        print("GPU not available, using CPU")
