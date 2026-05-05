"""Write docs."""

import inspect

from cspdk.si220.cband import _cells as cells
from cspdk.si220.cband.config import PATH

filepath = PATH.repo / "docs" / "cells_si220_cband.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


with open(filepath, "w+") as f:
    f.write(
        """

Cells Si SOI 220nm Cband
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

.. autofunction:: cspdk.si220.cband.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: cspdk.si220.cband.cells.{name}

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cband.cells.{name}({kwargs}).dup()
  c.draw_ports()
  c.plot()

"""
            )
