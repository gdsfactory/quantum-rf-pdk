"""Jittable N-dimensional lookup of gridded FEM datasets.

Data is loaded, validated, and arranged once, outside tracing (see
:meth:`qpdk.models.datasets.Dataset.grid`). The interpolation itself is
:class:`jax.scipy.interpolate.RegularGridInterpolator`, so calling the lookup
under :func:`jax.jit`, :func:`jax.vmap`, or :func:`jax.grad` involves no Polars,
file I/O, or data-dependent Python control flow.

:class:`GridInterpolator` only adds what a dataset lookup needs on top: keyword
arguments named after the axes, matrix-valued results, axes with a single grid
value, and NaN outside the *validated* domain rather than just outside the grid.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

from qpdk.models.datasets.table import Grid

type OutOfRange = Literal["nan", "clip"]


class GridInterpolator:
    """Multilinear interpolation of a :class:`~qpdk.models.datasets.table.Grid`.

    Call with one keyword argument per axis. Arguments broadcast against each
    other; the result has shape ``(*broadcast shape, *component shape)``.

    Args:
        grid: Complete grid from :meth:`qpdk.models.datasets.Dataset.grid`.
        out_of_range: ``"nan"`` (default) returns NaN for any query outside the
            validated domain of an axis. ``"clip"`` evaluates at the nearest point
            of the domain; use it only where a deliberate clamp is acceptable.

    Example:
        >>> grid = Dataset("plate_capacitor_palace").grid(
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
        self.domain = grid.domain
        self.component_shape = grid.values.shape[len(grid.axes) :]
        # A single-valued axis has no cell to interpolate in; drop it from the
        # interpolator and keep only its domain check.
        self._varying = tuple(i for i, c in enumerate(grid.coords) if len(c) > 1)
        values = grid.values.reshape(*grid.values.shape[: len(grid.axes)], -1)
        values = values.reshape(*(values.shape[i] for i in self._varying), -1)
        with jax.ensure_compile_time_eval():
            self._values = jnp.asarray(values)
            self._interpolate = (
                RegularGridInterpolator(
                    tuple(jnp.asarray(grid.coords[i]) for i in self._varying),
                    self._values,
                    bounds_error=False,
                    fill_value=jnp.nan,
                )
                if self._varying
                else None
            )

    def __repr__(self) -> str:
        """Show the dataset, quantity, domain, and out-of-range policy."""
        return (
            f"GridInterpolator({self.grid.dataset!r}, {self.grid.quantity.name!r}, "
            f"domain={self.domain}, out_of_range={self.out_of_range!r})"
        )

    def __call__(self, **params: jax.typing.ArrayLike) -> jax.Array:
        """Interpolate at the given axis values.

        Returns:
            Interpolated values of shape ``(*broadcast shape, *component shape)``.

        Raises:
            TypeError: unless exactly the grid axes are given.
        """
        if set(params) != set(self.axis_names):
            msg = f"Expected exactly the axes {self.axis_names}, got {tuple(params)}."
            raise TypeError(msg)
        xs = jnp.broadcast_arrays(
            *(jnp.asarray(params[name], dtype=float) for name in self.axis_names)
        )
        domain = [self.domain[name] for name in self.axis_names]
        if self.out_of_range == "clip":
            xs = [jnp.clip(x, lo, hi) for x, (lo, hi) in zip(xs, domain, strict=True)]
        shape = xs[0].shape
        if self._interpolate is None:
            flat = jnp.broadcast_to(self._values, (*shape, self._values.shape[-1]))
        else:
            points = jnp.stack([xs[i] for i in self._varying], axis=-1)
            flat = self._interpolate(points)
        result = flat.reshape(*shape, *self.component_shape)
        if self.out_of_range == "nan":
            inside = jnp.ones(shape, dtype=bool)
            for x, (lo, hi) in zip(xs, domain, strict=True):
                inside &= (x >= lo) & (x <= hi)
            result = jnp.where(
                inside.reshape(*shape, *(1,) * len(self.component_shape)),
                result,
                jnp.nan,
            )
        return result

    def in_domain(self, **params: np.typing.ArrayLike) -> np.ndarray:
        """Eagerly check which query points lie inside the validated domain.

        Returns:
            Boolean mask of the broadcast shape.
        """
        xs = np.broadcast_arrays(
            *(np.asarray(params[name], dtype=float) for name in self.axis_names)
        )
        inside = np.ones(xs[0].shape, dtype=bool)
        for x, name in zip(xs, self.axis_names, strict=True):
            lo, hi = self.domain[name]
            inside &= (x >= lo) & (x <= hi)
        return inside
