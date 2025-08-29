"""Write docs."""

import inspect
import textwrap
from pathlib import Path

from qpdk import PDK
from qpdk.config import PATH

filepath = PATH.repo / "docs" / "cells.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()

PDK.activate()
cells = PDK.cells


with Path(filepath).open("w+") as f:
    f.write(
        """

Cells QPDK
=============================
"""
    )

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])
        kwargs = ", ".join(
            [
                f"{p}={sig.parameters[p].default!r}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: qpdk.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: qpdk.cells.{name}

.. plot::
  :include-source:

  from qpdk import cells, PDK

  PDK.activate()
  c = cells.{name}({kwargs}).copy()
  c.draw_ports()
  c.plot()

"""
            )

    f.write(
        textwrap.dedent("""
            .. bibliography::
               :filter: docname in docnames
               """)
    )
