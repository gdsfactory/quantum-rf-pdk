"""Run the VACASK circuit simulator as a subprocess and read its results.

`VACASK <https://codeberg.org/arpadbuermen/VACASK>`_ is an analog circuit
simulator with multi-tone harmonic balance (``hb``) and small-signal analysis
around a pumped operating point (``hbac``). Devices are written in Verilog-A
and compiled on the fly by OpenVAF-Reloaded (``openvaf-r``).

VACASK is AGPL-3.0 licensed. This package only calls the ``vacask`` binary as a
separate process and parses the SPICE raw files it writes; it does not import,
link or copy any VACASK code.

Circuits and analyses are written directly in VACASK's netlist syntax. The
package only adds what is awkward to do by hand:

- :class:`Netlist` fills raw VACASK text into a Jinja2 template and copies the
  Verilog-A models shipped with qpdk next to it.
- :class:`Vacask` runs a netlist and returns one :class:`RawPlot` per analysis.
- :class:`RawFile` reads the SPICE raw files that VACASK writes.
- :func:`vector_fit` and :func:`vector_fit_subckt` turn S-parameters into a
  VACASK subcircuit. They need the ``models`` extra.

Example:
    >>> from qpdk.simulation.vacask import Netlist, Vacask
    >>> netlist = Netlist(
    ...     "JJ resonance",
    ...     loads=["josephson_junction.va", "capacitor.osdi"],
    ...     circuit='''
    ...         model isource isource
    ...         model capacitor capacitor
    ...         model jj jj
    ...         iin (0 1) isource dc=0 mag=1n
    ...         j1 (1 0) jj ic=1u
    ...         c1 (1 0) capacitor c=2p
    ...     ''',
    ...     control='analysis ac1 ac from=1G to=10G mode="lin" points=1000',
    ... )
    >>> results = Vacask.find().run(netlist)
    >>> results["ac1"]["1"]  # complex node voltage
"""

from .errors import VacaskError
from .netlist import (
    VACASK_MODELS_DIR,
    VACASK_TEMPLATES_DIR,
    Netlist,
    _templates as _templates,
    format_value,
)
from .network import MODELS_EXTRA_HINT, cpw_tline_params, vector_fit, vector_fit_subckt
from .raw import RawFile, RawPlot
from .runner import INSTALL_HINT, SimulationResult, Vacask

__all__ = [
    "INSTALL_HINT",
    "MODELS_EXTRA_HINT",
    "VACASK_MODELS_DIR",
    "VACASK_TEMPLATES_DIR",
    "Netlist",
    "RawFile",
    "RawPlot",
    "SimulationResult",
    "Vacask",
    "VacaskError",
    "cpw_tline_params",
    "format_value",
    "vector_fit",
    "vector_fit_subckt",
]
