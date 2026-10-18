"""Signature-preserving wrapper around :func:`jax.jit`.

:func:`jax.jit` returns a ``JitWrapped`` whose ``__call__`` is untyped, so type
checkers infer every jitted function as returning ``Never``. That silently
satisfies any downstream annotation and hides real type errors. :func:`jit`
calls :func:`jax.jit` unchanged at runtime and keeps the wrapped function's
signature for static analysis.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import cast, overload

import jax

__all__ = ["jit"]


@overload
def jit[**P, R](fun: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def jit[**P, R](
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    inline: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def jit[**P, R](
    fun: Callable[P, R] | None = None,
    /,
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    inline: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply :func:`jax.jit` while preserving the signature of ``fun``.

    Use as ``@jit`` or ``@jit(inline=True, static_argnames=...)``.

    Args:
        fun: Function to compile. If omitted, return a decorator.
        static_argnums: Passed to :func:`jax.jit`.
        static_argnames: Passed to :func:`jax.jit`.
        inline: Passed to :func:`jax.jit`.

    Returns:
        The jitted function, typed as ``fun``, or a decorator producing it.
    """

    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        return cast(
            Callable[P, R],
            jax.jit(
                f,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                inline=inline,
            ),
        )

    return decorate if fun is None else decorate(fun)
