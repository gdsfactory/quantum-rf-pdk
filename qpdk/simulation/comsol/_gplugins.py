"""Import :mod:`gplugins.comsol` submodules with an actionable error.

``gplugins`` is not a base dependency of QPDK; it comes with the optional
``comsol`` extra. The QPDK COMSOL modules import it only when one of its names
is used, so they stay importable without the extra and fail with an install
hint instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

__all__ = ["INSTALL_HINT", "import_gplugins_comsol", "is_missing", "reexport"]

INSTALL_HINT = (
    "needs gplugins.comsol, which is not installed; "
    "install it with `uv sync --extra comsol`"
)


def is_missing(error: ModuleNotFoundError, module_name: str) -> bool:
    """Tell whether ``error`` means ``module_name`` or a parent is absent.

    Args:
        error: The error raised while importing ``module_name``.
        module_name: Dotted name of the module being imported.

    Returns:
        ``True`` if the missing module is ``module_name`` or one of its
        parent packages, ``False`` if it is a dependency of it.
    """
    return error.name is not None and (
        module_name == error.name or module_name.startswith(f"{error.name}.")
    )


def import_gplugins_comsol(submodule: str) -> ModuleType:
    """Import ``gplugins.comsol.<submodule>``.

    Args:
        submodule: Name of the submodule of :mod:`gplugins.comsol`.

    Returns:
        The imported module.

    Raises:
        ImportError: If gplugins, or a gplugins release with the COMSOL
            plugin, is not installed.
        ModuleNotFoundError: If a dependency of the submodule is missing.
    """
    module_name = f"gplugins.comsol.{submodule}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not is_missing(error, module_name):
            raise
        raise ImportError(f"{module_name} {INSTALL_HINT}") from error


def reexport(
    module_name: str, submodule: str, names: list[str]
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that loads ``names`` from gplugins.

    Args:
        module_name: ``__name__`` of the re-exporting module.
        submodule: Name of the submodule of :mod:`gplugins.comsol`.
        names: Public names re-exported from that submodule.

    Returns:
        A module-level ``__getattr__``.
    """
    exported = frozenset(names)

    def module_getattr(name: str) -> Any:
        if name in exported:
            return getattr(import_gplugins_comsol(submodule), name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return module_getattr
