"""Export SAX S-parameter dictionaries as Touchstone files.

Touchstone_ is the interchange format that essentially every RF tool reads,
so writing one is the shortest path from a :mod:`sax` model to MATLAB's
`RF Toolbox`_ (``sparameters``/``nport``), Keysight ADS, Qucs-S,
scikit-rf, or a vector network analyser's own software.

The conversion itself is :func:`sax.sdense`, which densifies an
:class:`sax.SDict` into an ``(..., n, n)`` array plus a port-name to index
map.  This module adds the bookkeeping around it: a deterministic port
order, the ``Hz S RI R <z0>`` header, and the port-order comment that tells
a reader which physical port each index corresponds to (Touchstone itself
only stores numbers).

.. _Touchstone:
   https://ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf
.. _RF Toolbox: https://se.mathworks.com/help/rf/index.html

Note:
    Like the rest of :mod:`qpdk.models`, this module needs the optional
    ``models`` extra — here specifically :mod:`skrf` (scikit-rf).  Install
    with ``uv sync --extra models`` or ``pip install "qpdk[models]"``.

Example:
    >>> import numpy as np
    >>> from qpdk import PDK
    >>> from qpdk.models.touchstone import write_touchstone
    >>> from qpdk.models.waveguides import straight
    >>> PDK.activate()
    >>> f = np.linspace(2e9, 8e9, 401)
    >>> sdict = straight(f=f, length=1000)
    >>> write_touchstone(sdict, f, "cpw_1mm.s2p")  # doctest: +SKIP
    PosixPath('cpw_1mm.s2p')
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sax
import skrf

__all__ = ["sdict_to_array", "sdict_to_network", "write_touchstone"]


def sdict_to_array(
    sdict: sax.SDict,
    ports: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Densify a SAX ``SDict`` into an S-parameter array and its port order.

    Args:
        sdict: S-parameter dictionary returned by any qpdk/SAX model.
        ports: Port names in the order they should occupy in the output
            array.  Defaults to the model's own port order as reported by
            :func:`sax.sdense`, which is stable for a given model.

    Returns:
        A ``(k, n, n)`` complex array of S-parameters over ``k`` frequency
        points, and the tuple of port names giving the index order.  Models
        evaluated at a single frequency still return ``k == 1``.

    Raises:
        ValueError: If *ports* is not a permutation of the model's ports.
    """
    s_array, port_map = sax.sdense(sdict)
    model_ports = tuple(sorted(port_map, key=port_map.__getitem__))

    s_array = np.asarray(s_array, dtype=complex)
    # sax returns (..., n, n); a scalar-frequency model has no leading axis.
    if s_array.ndim == 2:
        s_array = s_array[np.newaxis, ...]

    if ports is None:
        return s_array, model_ports

    ports = tuple(ports)
    if sorted(ports) != sorted(model_ports):
        raise ValueError(
            f"ports={ports} is not a permutation of the model ports {model_ports}"
        )
    order = [port_map[port] for port in ports]
    return s_array[..., order, :][..., :, order], ports


def sdict_to_network(
    sdict: sax.SDict,
    f: Any,
    *,
    z0: float = 50.0,
    ports: Sequence[str] | None = None,
    name: str | None = None,
) -> skrf.Network:
    """Convert a SAX ``SDict`` into a :class:`skrf.Network`.

    Args:
        sdict: S-parameter dictionary returned by any qpdk/SAX model.
        f: Frequency points in Hz, the same ones the model was evaluated at.
        z0: Reference impedance in ohms.
        ports: Optional explicit port order, see :func:`sdict_to_array`.
        name: Network name, used as the default Touchstone comment.

    Returns:
        A scikit-rf network carrying the port order in ``network.port_names``.

    Raises:
        ValueError: If ``len(f)`` does not match the model's frequency axis.
    """
    s_array, port_order = sdict_to_array(sdict, ports)
    frequency = np.atleast_1d(np.asarray(f, dtype=float))
    if frequency.size != s_array.shape[0]:
        raise ValueError(
            f"f has {frequency.size} points but the model was evaluated at "
            f"{s_array.shape[0]}; pass the same frequency array used for the model"
        )

    network = skrf.Network(
        frequency=skrf.Frequency.from_f(frequency, unit="Hz"),
        s=s_array,
        z0=z0,
        name=name,
    )
    network.port_names = list(port_order)
    return network


def write_touchstone(
    sdict: sax.SDict,
    f: Any,
    path: str | Path,
    *,
    z0: float = 50.0,
    ports: Sequence[str] | None = None,
) -> Path:
    """Write a SAX ``SDict`` to a Touchstone file.

    The file extension is not rewritten, so pass ``.s2p`` for a two-port
    model, ``.s3p`` for a three-port model and so on; the port count is
    reported in the returned path's suffix only if you chose it correctly.

    Args:
        sdict: S-parameter dictionary returned by any qpdk/SAX model.
        f: Frequency points in Hz, the same ones the model was evaluated at.
        path: Destination file.  Parent directories are created if missing.
        z0: Reference impedance in ohms.
        ports: Optional explicit port order, see :func:`sdict_to_array`.

    Returns:
        The path that was written.

    Example:
        Read the result back in MATLAB with ``sparameters(path)``, or in
        Python with ``skrf.Network(path)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    network = sdict_to_network(sdict, f, z0=z0, ports=ports, name=path.stem)
    # write_touchstone() appends its own extension, so hand it the stem and
    # let the caller keep whatever suffix they asked for.
    network.write_touchstone(
        filename=str(path),
        form="ri",
        write_z0=False,
        skrf_comment=False,
    )
    written = path.with_suffix(f".s{network.nports}p")
    if written != path and written.exists():
        written.replace(path)
    return path
