"""S-parameter model for a straight waveguide."""

from functools import partial
from pprint import pprint
from typing import TypedDict

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
        f: Tuple of frequency points in Hz (static for JIT)
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


if __name__ == "__main__":
    cpw = cpw_media_skrf(width=10, gap=6)
    S = straight(media=cpw)
    S = straight(f=jnp.linspace(0.5e9, 9e9, 201), media=cpw)
    pprint(S)

    # Move the S['o2','o1'] array to GPU and test that it works
    try:
        # Get the array
        s21_array = S["o2", "o1"]
        print(f"Original device: {s21_array.device}")

        # Move to GPU
        s21_gpu = jax.device_put(s21_array, jax.devices("gpu")[0])
        print(f"GPU device: {s21_gpu.device}")

        # Test that it works by doing a simple operation
        result = jnp.abs(s21_gpu) ** 2
        print(f"GPU computation result shape: {result.shape}")
        print(f"GPU computation result device: {result.device}")

    except Exception as e:
        print(f"GPU test failed: {e}")
        print("Falling back to CPU")
        s21_array = S["o2", "o1"]
        print(f"CPU device: {s21_array.device}")
