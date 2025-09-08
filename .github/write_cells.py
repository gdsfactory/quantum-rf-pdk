"""Write docs."""

import inspect
import textwrap
from pathlib import Path

from gdsfactory.serialization import clean_value_json

from qpdk import PDK
import qpdk
from qpdk.config import PATH

filepath_cells = PATH.repo / "docs" / "cells.rst"
filepath_samples = PATH.repo / "docs" / "samples.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()

PDK.activate()
cells = PDK.cells
samples = qpdk.sample_functions


with Path(filepath_cells).open("w+") as f:
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
                f"{p}={clean_value_json(sig.parameters[p].default)!r}"
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

with Path(filepath_samples).open("w+") as f:
    f.write(
        """
Samples
=============================
"""
    )

    for name in sorted(samples.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(samples[name])
        kwargs = ", ".join(
            [
                f"{p}={clean_value_json(sig.parameters[p].default)!r}"
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

.. autofunction:: {name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: {name}

.. plot::
  :include-source:

  import {name.rpartition(".")[0]}
  from qpdk import PDK

  PDK.activate()
  c = {name}({kwargs}).copy()
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
