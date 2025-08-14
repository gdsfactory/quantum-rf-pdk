"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
from numpy.typing import NDArray

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            continue
        if str(sig.return_annotation).lower().split(".")[-1] == "sdict":
            models[name] = func
    return models
