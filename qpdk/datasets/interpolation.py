"""Jittable N-dimensional lookup of gridded FEM datasets.

Data is loaded, validated, and arranged once, outside tracing (see
:meth:`qpdk.datasets.Dataset.grid`). The interpolator then holds plain JAX
arrays, so calling it under :func:`jax.jit`, :func:`jax.vmap`, or
:func:`jax.grad` involves no Polars, file I/O, or data-dependent Python
control flow.

Interpolation is multilinear on the rectilinear grid, using the same
:func:`jax.scipy.ndimage.map_coordinates` core as :func:`sax.interpolate_xarray`.
Unlike that function, axes are never filled in silently and queries outside the
validated domain return NaN instead of the nearest edge value.

Values keep the dtype JAX gives float64 data: float64 when ``jax_enable_x64`` is
on (importing :mod:`sax` turns it on), float32 otherwise.
"""

from itertools import starmap
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from qpdk.datasets.table import Grid

type OutOfRange = Literal["nan", "clip"]


class GridInterpolator:
    """Multilinear interpolation of a :class:`~qpdk.datasets.table.Grid`.

    Call with one keyword argument per axis. Arguments broadcast against each
    other; the result has shape ``(*broadcast shape, *component shape)``.

    Args:
        grid: Complete grid from :meth:`qpdk.datasets.Dataset.grid`.
        out_of_range: ``"nan"`` (default) returns NaN for any query outside the
            validated domain of an axis. ``"clip"`` evaluates at the nearest point
            of the domain; use it only where a deliberate clamp is acceptable.

    Example:
        >>> grid = Dataset("plate_capacitor_synthetic").grid(
        ...     "maxwell_capacitance", cross_section="cpw"
        ... )
        >>> c = GridInterpolator(grid)(length=100.0, width=10.0, gap=5.0)  # (2, 2) in F
    """

    def __init__(self, grid: Grid, *, out_of_range: OutOfRange = "nan") -> None:
        """Move the grid to JAX arrays once, outside any trace."""
        if out_of_range not in {"nan", "clip"}:
            msg = f"out_of_range must be 'nan' or 'clip', got {out_of_range!r}."
            raise ValueError(msg)
        self.grid = grid
        self.out_of_range = out_of_range
        self.axis_names = grid.axis_names
        self.component_shape = grid.values.shape[len(grid.axes) :]
        with jax.ensure_compile_time_eval():
            self._coords = tuple(jnp.asarray(axis.values) for axis in grid.axes)
            self._domain = tuple(
                (jnp.asarray(lo), jnp.asarray(hi))
                for lo, hi in (a.domain for a in grid.axes)
            )
            self._values = jnp.asarray(
                grid.values.reshape(*grid.values.shape[: len(grid.axes)], -1)
            )

    @property
    def domain(self) -> dict[str, tuple[float, float]]:
        """Validated domain of each axis, in the axis' unit."""
        return {axis.name: axis.domain for axis in self.grid.axes}

    def __repr__(self) -> str:
        """Show the dataset, quantity, domain, and out-of-range policy."""
        return (
            f"GridInterpolator({self.grid.dataset!r}, {self.grid.quantity.name!r}, "
            f"domain={self.domain}, out_of_range={self.out_of_range!r})"
        )

    def __call__(self, **params: jax.typing.ArrayLike) -> jax.Array:
        """Interpolate at the given axis values."""
        if set(params) != set(self.axis_names):
            msg = f"Expected exactly the axes {self.axis_names}, got {tuple(params)}."
            raise TypeError(msg)
        xs = jnp.broadcast_arrays(
            *(jnp.asarray(params[name], dtype=float) for name in self.axis_names)
        )
        if self.out_of_range == "clip":
            xs = [
                jnp.clip(x, lo, hi)
                for x, (lo, hi) in zip(xs, self._domain, strict=True)
            ]
        index = list(starmap(_fractional_index, zip(self._coords, xs, strict=True)))
        flat = jax.vmap(
            lambda component: map_coordinates(
                component, index, order=1, mode="nearest"
            ),
            in_axes=-1,
            out_axes=-1,
        )(self._values)
        result = flat.reshape(*xs[0].shape, *self.component_shape)
        if self.out_of_range == "nan":
            inside = jnp.ones(xs[0].shape, dtype=bool)
            for x, (lo, hi) in zip(xs, self._domain, strict=True):
                inside &= (x >= lo) & (x <= hi)
            result = jnp.where(
                inside.reshape(*inside.shape, *(1,) * len(self.component_shape)),
                result,
                jnp.nan,
            )
        return result

    def in_domain(self, **params: np.typing.ArrayLike) -> np.ndarray:
        """Eagerly check which query points lie inside the validated domain."""
        xs = np.broadcast_arrays(
            *(np.asarray(params[name], dtype=float) for name in self.axis_names)
        )
        inside = np.ones(xs[0].shape, dtype=bool)
        for x, (lo, hi) in zip(xs, self.domain.values(), strict=True):
            inside &= (x >= lo) & (x <= hi)
        return inside


def _fractional_index(coords: jax.Array, x: jax.Array) -> jax.Array:
    """Map values to fractional grid indices, linearly within each cell."""
    n = coords.shape[0]
    if n == 1:
        return jnp.zeros_like(x)
    i = jnp.clip(jnp.searchsorted(coords, x, side="right") - 1, 0, n - 2)
    lo, hi = coords[i], coords[i + 1]
    return i + (x - lo) / (hi - lo)
