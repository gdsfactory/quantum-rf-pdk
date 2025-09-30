"""S-parameter model for a straight waveguide."""

from functools import partial
from typing import TypedDict, Unpack

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from skrf import Frequency

from qpdk.models.media import MediaCallable, cpw_media_skrf


class StraightModelKwargs(TypedDict, total=False):
    """Type definition for straight S-parameter model keyword arguments."""

    f: ArrayLike
    length: int | float
    media: MediaCallable


@partial(jax.jit, static_argnames=["length", "media"])
def straight(
    f: ArrayLike = jnp.array([5e9]),
    length: int | float = 1000,
    media: MediaCallable = cpw_media_skrf(),
) -> sax.SType:
    """S-parameter model for a straight waveguide.

    Args:
        f: Tuple of frequency points in Hz
        length: Physical length in µm
        media: Function returning a scikit-rf :class:`~Media` object after called
            with ``frequency=f``. If None, uses default CPW media.

    Returns:
        sax.SType: S-parameters dictionary
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


def bend_circular(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for a circular bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)


def bend_euler(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for an Euler bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)


def bend_s(
    *args: ArrayLike | int | float | MediaCallable,
    **kwargs: Unpack[StraightModelKwargs],
) -> sax.SType:
    """S-parameter model for an S-bend, wrapped to to :func:`~straight`."""
    return straight(*args, **kwargs)


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
