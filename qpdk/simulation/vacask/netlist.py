"""VACASK netlists and the value formatting used to write them."""

from __future__ import annotations

import numbers
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import jinja2
import numpy as np

VACASK_MODELS_DIR = Path(__file__).parents[1] / "vacask_models"
VACASK_TEMPLATES_DIR = Path(__file__).parents[1] / "vacask_templates"

_ANALYSIS = re.compile(r"^\s*analysis\s+(\S+)", re.MULTILINE)


def format_value(value: Any) -> str:
    """Format a Python value as a VACASK parameter value.

    Useful for writing long vectors into netlist text, e.g.
    ``f"values={format_value(frequencies)}"``. Strings are quoted, numbers use
    the shortest exact representation, sequences become vectors ``[a, b]`` and
    sequences of sequences become lists of vectors ``{[a],[b]}``, as used by
    ``spur`` and ``outspur``.

    Returns:
        The value as VACASK netlist text.

    Raises:
        TypeError: If the value has no VACASK representation.
    """
    match value:
        case str():
            return f'"{value}"'
        case bool() | np.bool_():
            return "1" if value else "0"
        case numbers.Integral():
            return str(int(value))
        case numbers.Real():
            short = f"{float(value):g}"
            exact = float(short) == float(value)  # ruff: ignore[float-equality-comparison]
            return short if exact else repr(float(value))
        case _ if hasattr(value, "__array__") and np.ndim(value) == 0:
            return format_value(np.asarray(value).item())
        case _ if isinstance(value, Sequence) or hasattr(value, "__array__"):
            items = list(value)
            if items and all(
                (isinstance(item, Sequence) and not isinstance(item, str))
                or np.ndim(item) > 0
                for item in items
            ):
                return "{" + ",".join(format_value(item) for item in items) + "}"
            return "[" + ", ".join(format_value(item) for item in items) + "]"
        case _:
            raise TypeError(f"Cannot format {value!r} as a VACASK value")


def _dedent(text: str) -> str:
    """Remove common indentation and surrounding blank lines."""
    return textwrap.dedent(text).strip()


@cache
def _templates() -> jinja2.Environment:
    """Jinja2 environment for the netlist templates."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(VACASK_TEMPLATES_DIR),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,  # ruff: ignore[jinja2-autoescape-false]  # netlists, not HTML
    )
    env.filters["dedent"] = _dedent
    env.filters["number"] = format_value
    return env


@dataclass(frozen=True)
class Netlist:
    """A VACASK netlist, rendered from ``vacask_templates/netlist.sim.j2``.

    ``circuit``, ``control`` and ``subckts`` are VACASK netlist text and are
    inserted as-is, after removing their common indentation. The control block always
    starts with ``abort always``, so that a failed analysis gives a nonzero
    exit code.

    Attributes:
        title: First line of the netlist.
        circuit: Models and instances, in VACASK syntax.
        control: Contents of the ``control`` block: options, sweeps and
            analyses. Each ``analysis <name> ...`` line produces a result.
        loads: Files to load, e.g. ``"capacitor.osdi"`` from the VACASK
            installation or ``"josephson_junction.va"``. Verilog-A models
            shipped with qpdk are copied next to the netlist automatically.
        subckts: Subcircuit definitions, e.g. from :func:`vector_fit_subckt`.
            Each one is dedented on its own, so they can come from different
            sources.
        files: Extra files to place next to the netlist, as ``{name: content}``.
            A :class:`~pathlib.Path` value is copied; a string is written as text.
    """

    title: str
    circuit: str = ""
    control: str = ""
    loads: Sequence[str] = ()
    subckts: Sequence[str] = ()
    files: Mapping[str, str | Path] = field(default_factory=dict)

    @property
    def analyses(self) -> list[str]:
        """Names of the analyses in :attr:`control`, in order."""
        return _ANALYSIS.findall(self.control)

    def render(self) -> str:
        """Return the netlist text."""
        return _templates().get_template("netlist.sim.j2").render(**vars(self))

    def all_files(self) -> dict[str, str | Path]:
        """Files to write next to the netlist, including loaded qpdk models."""
        files: dict[str, str | Path] = {
            name: VACASK_MODELS_DIR / name
            for name in self.loads
            if (VACASK_MODELS_DIR / name).is_file()
        }
        files.update(self.files)
        return files
