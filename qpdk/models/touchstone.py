"""Read and write Touchstone files for SAX S-parameter dictionaries.

Touchstone_ is the interchange format that essentially every RF tool reads,
so it is the shortest path from a :mod:`sax` model to MATLAB's `RF Toolbox`_
(``sparameters``/``nport``), Keysight ADS, Qucs-S, or a vector network
analyser's own software — and back again.

The conversion itself is :func:`sax.sdense`, which densifies an
:class:`sax.SDict` into an ``(..., n, n)`` array plus a port-name to index
map.  This module adds the bookkeeping around it: a deterministic port
order, the ``Hz S RI R <z0>`` option line, and the ``! ports:`` comment that
records which physical port each index corresponds to (Touchstone itself only
stores numbers).

The reader and writer are implemented directly against the Touchstone v1
grammar with :mod:`numpy` and the standard library only — no scikit-rf, and no
dependency on anything in :mod:`qpdk`.  :func:`format_touchstone` and
:func:`parse_touchstone` work on plain arrays, and the ``SDict`` wrappers
around them are four lines each, so the whole module is straightforward to
upstream into :mod:`sax` (which today has a ``sax.parsers.touchstone`` that
delegates to scikit-rf).

Files written by v2 tools are read as well, as far as the v1 data section
allows: keyword lines are skipped, ``[Number of Ports]``,
``[Two-Port Data Order]`` and a uniform ``[Reference]`` are honoured.

.. _Touchstone:
   https://ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf
.. _RF Toolbox: https://se.mathworks.com/help/rf/index.html

Example:
    >>> import numpy as np
    >>> from qpdk import PDK
    >>> from qpdk.models.touchstone import read_touchstone, write_touchstone
    >>> from qpdk.models.waveguides import straight
    >>> PDK.activate()
    >>> f = np.linspace(2e9, 8e9, 401)
    >>> sdict = straight(f=f, length=1000)
    >>> write_touchstone(sdict, f, "cpw_1mm.s2p")  # doctest: +SKIP
    PosixPath('cpw_1mm.s2p')
    >>> f_back, sdict_back = read_touchstone("cpw_1mm.s2p")  # doctest: +SKIP
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sax

__all__ = [
    "array_to_sdict",
    "format_touchstone",
    "parse_touchstone",
    "read_touchstone",
    "sdict_to_array",
    "write_touchstone",
]

#: Frequency-unit keywords of the option line and their value in Hz.
FREQUENCY_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

#: Number formatting of every value written; 12 significant digits is well
#: beyond the precision of any measurement and still round-trips float32 data.
_FORMAT = "{:.12g}"

_PORTS_COMMENT = re.compile(r"^\s*!\s*ports\s*:\s*(.+)$", re.IGNORECASE)
_PORT_COMMENT = re.compile(r"^\s*!\s*Port\[(\d+)\]\s*=\s*(\S+)", re.IGNORECASE)


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


def array_to_sdict(s_array: np.ndarray, ports: Sequence[str]) -> sax.SDict:
    """Build a SAX ``SDict`` from a dense S-parameter array and its port order.

    Inverse of :func:`sdict_to_array`.  Following the SAX convention, the key
    is ``(input_port, output_port)`` and the value is ``S[..., out, in]``.

    Args:
        s_array: ``(k, n, n)`` complex array of S-parameters.  A scalar-frequency
            ``(n, n)`` array is accepted and read at a single point.
        ports: Port names, in index order.

    Returns:
        The equivalent ``SDict``, with one entry per port pair.

    Raises:
        ValueError: If the matrix is not square with one row and column per
            port, or *ports* contains duplicate names.
    """
    s_array = np.asarray(s_array, dtype=complex)
    if s_array.ndim == 2:
        s_array = s_array[np.newaxis, ...]
    ports = tuple(ports)
    if (
        s_array.ndim != 3
        or s_array.shape[1] != s_array.shape[2]
        or (s_array.shape[1] != len(ports))
    ):
        raise ValueError(
            f"expected a (k, {len(ports)}, {len(ports)}) array for ports {ports}, "
            f"got shape {s_array.shape}"
        )
    if len(set(ports)) != len(ports):
        raise ValueError(f"ports={ports} contains duplicate names")
    return {
        (p_in, p_out): s_array[..., j, i]
        for i, p_in in enumerate(ports)
        for j, p_out in enumerate(ports)
    }


def format_touchstone(
    s_array: np.ndarray,
    f: Any,
    ports: Sequence[str] | None = None,
    *,
    z0: float = 50.0,
    frequency_unit: str = "Hz",
) -> str:
    """Serialize an S-parameter array as Touchstone v1 text.

    Always writes real/imaginary pairs, which is the only lossless format of
    the three (magnitude/angle and dB/angle both go through a transcendental).

    Args:
        s_array: ``(k, n, n)`` complex array of S-parameters.
        f: The ``k`` frequency points in Hz.
        ports: Optional port names, recorded in a ``! ports:`` comment.
        z0: Reference impedance in ohms.
        frequency_unit: One of ``Hz``, ``kHz``, ``MHz``, ``GHz``; only the
            written numbers change, *f* is always in Hz.

    Returns:
        The file contents, ending in a newline.

    Raises:
        ValueError: If the array is not square, the frequency axis does not
            match, or *frequency_unit* is not a Touchstone unit.
    """
    s_array = np.asarray(s_array, dtype=complex)
    if s_array.ndim == 2:
        s_array = s_array[np.newaxis, ...]
    if s_array.ndim != 3 or s_array.shape[1] != s_array.shape[2]:
        raise ValueError(f"expected a (k, n, n) array, got shape {s_array.shape}")

    frequency = np.atleast_1d(np.asarray(f, dtype=float))
    if frequency.size != s_array.shape[0]:
        raise ValueError(
            f"f has {frequency.size} points but the S-parameters have "
            f"{s_array.shape[0]}; pass the same frequency array used for the model"
        )

    try:
        scale = FREQUENCY_UNITS[frequency_unit.lower()]
    except KeyError:
        raise ValueError(
            f"frequency_unit={frequency_unit!r} is not one of "
            f"{', '.join(FREQUENCY_UNITS)}"
        ) from None

    n = s_array.shape[1]
    lines = ["! Touchstone 1.1 file generated by qpdk.models.touchstone"]
    if ports is not None:
        if len(ports) != n:
            raise ValueError(f"got {len(ports)} port names for a {n}-port matrix")
        lines.append(f"! ports: {', '.join(ports)}")
    lines.append(f"# {frequency_unit} S RI R {_FORMAT.format(z0)}")

    for point, matrix in zip(frequency / scale, s_array, strict=True):
        if n == 2:
            # The two-port layout is the format's one irregularity: a single
            # row of S11 S21 S12 S22, i.e. the transpose of every other size.
            lines.append(
                " ".join([_FORMAT.format(point), *_pairs(matrix.T.reshape(-1))])
            )
            continue
        for row_index, row in enumerate(matrix):
            prefix = [_FORMAT.format(point)] if row_index == 0 else []
            # The spec allows at most four complex values per line.
            for chunk_start in range(0, n, 4):
                chunk = row[chunk_start : chunk_start + 4]
                lines.append(" ".join([*prefix, *_pairs(chunk)]))
                prefix = []

    return "\n".join(lines) + "\n"


def parse_touchstone(
    content: str,
    n_ports: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...] | None, float]:
    """Parse Touchstone v1 text into frequencies and an S-parameter array.

    The option line is optional: when it is absent the Touchstone defaults of
    ``GHz S MA R 50`` apply and the data section starts right after the header
    comments.

    Args:
        content: The file contents.
        n_ports: Port count, normally taken from the ``.sNp`` extension.  When
            omitted it is inferred from the first data line, which is correct
            for any file that does not wrap its rows.

    Returns:
        ``(f, s_array, ports, z0)``: frequencies in Hz, the ``(k, n, n)``
        complex S-parameters, the port names if the file carries a ``! ports:``
        or ``! Port[i] =`` comment (otherwise :obj:`None`), and the reference
        impedance in ohms.

    Raises:
        ValueError: If the option line is unsupported, ``[Reference]`` holds
            values the single-z0 contract cannot express, or the data does not
            divide evenly into frequency points.
    """
    option: list[str] | None = None
    port_names: dict[int, str] = {}
    tokens: list[str] = []
    swap_two_port = True
    reference_z0: float | None = None
    # An option line is optional; without one the standard defaults apply and
    # the v1 data section starts immediately (a keyword line ends it).
    in_data = True

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!"):
            if match := _PORTS_COMMENT.match(line):
                port_names = {
                    i: name.strip()
                    for i, name in enumerate(match.group(1).split(","), start=1)
                }
            elif match := _PORT_COMMENT.match(line):
                port_names[int(match.group(1))] = match.group(2)
            continue
        line = line.split("!", 1)[0].strip()
        if line.startswith("["):
            # Touchstone v2 keyword line; only the few that change how the
            # v1-style data section is read matter here.
            keyword, _, value = line.partition("]")
            keyword = keyword[1:].strip().lower()
            if keyword == "number of ports":
                n_ports = int(float(value.strip()))
            elif keyword == "two-port data order":
                swap_two_port = value.strip().lower() != "12_21"
            elif keyword == "reference":
                try:
                    reference = [float(word) for word in value.split()]
                except ValueError:
                    raise ValueError(
                        f"unsupported [Reference] values {value.strip()!r}; only real "
                        "impedances are supported"
                    ) from None
                # [Reference] overrides the option line's R. The reader's
                # single-z0 contract cannot express a per-port reference, so
                # only uniform values are accepted.
                if len(set(reference)) != 1:
                    raise ValueError(
                        f"per-port reference impedances {reference} are not supported; "
                        "the reader reports a single z0"
                    )
                reference_z0 = reference[0]
                in_data = False
            elif keyword == "network data":
                in_data, tokens = True, []
            elif keyword in {"end", "noise data"}:
                in_data = False
            else:
                in_data = False
            continue
        if line.startswith("#"):
            option = line[1:].split()
            in_data = True
            continue
        if in_data:
            tokens.append(line)

    frequency_unit, parameter, data_format, z0 = _parse_option(option or [])
    if reference_z0 is not None:
        z0 = reference_z0
    if parameter != "s":
        raise ValueError(f"only S-parameters are supported, the file holds {parameter}")

    values = np.fromstring(" ".join(tokens), sep=" ")
    if n_ports is None:
        n_ports = _infer_n_ports(tokens, values.size)

    record = 1 + 2 * n_ports**2
    if values.size == 0 or values.size % record:
        raise ValueError(
            f"{values.size} numbers do not divide into {record}-number records "
            f"for a {n_ports}-port file"
        )
    values = values.reshape(-1, record)

    frequency = values[:, 0] * FREQUENCY_UNITS[frequency_unit]
    pairs = values[:, 1:].reshape(-1, n_ports, n_ports, 2)
    s_array = _to_complex(pairs, data_format)
    if n_ports == 2 and swap_two_port:
        s_array = s_array.transpose(0, 2, 1)

    ports = (
        tuple(port_names[i] for i in range(1, n_ports + 1))
        if set(port_names) == set(range(1, n_ports + 1))
        else None
    )
    return frequency, s_array, ports, z0


def write_touchstone(
    sdict: sax.SDict,
    f: Any,
    path: str | Path,
    *,
    z0: float = 50.0,
    ports: Sequence[str] | None = None,
    frequency_unit: str = "Hz",
) -> Path:
    """Write a SAX ``SDict`` to a Touchstone file.

    The file extension is not rewritten, so pass ``.s2p`` for a two-port
    model, ``.s3p`` for a three-port model and so on; readers take the port
    count from it.

    Args:
        sdict: S-parameter dictionary returned by any qpdk/SAX model.
        f: Frequency points in Hz, the same ones the model was evaluated at.
        path: Destination file.  Parent directories are created if missing.
        z0: Reference impedance in ohms.
        ports: Optional explicit port order, see :func:`sdict_to_array`.
        frequency_unit: Unit the frequency column is written in.

    Returns:
        The path that was written.

    Example:
        Read the result back in MATLAB with ``sparameters(path)``, or in
        Python with :func:`read_touchstone`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s_array, port_order = sdict_to_array(sdict, ports)
    path.write_text(
        format_touchstone(s_array, f, port_order, z0=z0, frequency_unit=frequency_unit)
    )
    return path


def read_touchstone(
    path: str | Path,
    ports: Sequence[str] | None = None,
) -> tuple[np.ndarray, sax.SDict]:
    """Read a Touchstone file into frequencies and a SAX ``SDict``.

    Args:
        path: File to read.  A ``.sNp`` extension sets the port count.
        ports: Port names to use, overriding the file's ``! ports:`` comment.
            Without either, ports are named ``o1``, ``o2``, ... in file order.

    Returns:
        The frequency points in Hz and the S-parameters as an ``SDict`` whose
        values are arrays over those frequencies.

    Raises:
        ValueError: If *ports* does not have one name per port, or the file
            cannot be parsed.
    """
    path = Path(path)
    match = re.fullmatch(r"\.s(\d+)p", path.suffix, re.IGNORECASE)
    frequency, s_array, file_ports, _ = parse_touchstone(
        path.read_text(), int(match.group(1)) if match else None
    )

    n = s_array.shape[1]
    if ports is None:
        ports = file_ports or tuple(f"o{i}" for i in range(1, n + 1))
    if len(ports) != n:
        raise ValueError(f"got {len(ports)} port names for a {n}-port file")
    return frequency, array_to_sdict(s_array, ports)


def _pairs(values: np.ndarray) -> list[str]:
    """Format complex values as alternating real/imaginary strings."""
    return [
        _FORMAT.format(part) for value in values for part in (value.real, value.imag)
    ]


def _parse_option(option: Sequence[str]) -> tuple[str, str, str, float]:
    """Read frequency unit, parameter, format and impedance from an option line."""
    frequency_unit, parameter, data_format, z0 = "ghz", "s", "ma", 50.0
    words = [word.lower() for word in option]
    while words:
        word = words.pop(0)
        if word in FREQUENCY_UNITS:
            frequency_unit = word
        elif word in {"s", "y", "z", "h", "g"}:
            parameter = word
        elif word in {"ri", "ma", "db"}:
            data_format = word
        elif word == "r" and words:
            z0 = float(words.pop(0))
        else:
            raise ValueError(f"unsupported Touchstone option {word!r}")
    return frequency_unit, parameter, data_format, z0


def _infer_n_ports(lines: Sequence[str], total_values: int) -> int:
    """Infer the port count from the shape of the data section.

    A ``.sNp`` extension is the reliable source and is used when there is one.
    Failing that, the line layout pins the port count down: how many values the
    first line holds, how many lines each frequency point takes, and how many
    values a point needs all have to agree.

    Args:
        lines: The data lines of the file, comments and keywords removed.
        total_values: How many numbers those lines hold in total.

    Returns:
        The number of ports.

    Raises:
        ValueError: If no port count is consistent with the data section.
    """
    first_values = len(lines[0].split()) if lines else 0
    for n in range(1, 65):
        # One line per point up to two ports; above that one line per matrix
        # row, wrapped at the four complex values per line the spec allows.
        rows = 1 if n <= 2 else n * -(-n // 4)
        expected_first = 1 + 2 * (n**2 if n <= 2 else min(n, 4))
        record = 1 + 2 * n**2
        if (
            first_values == expected_first
            and not total_values % record
            and len(lines) == rows * total_values // record
        ):
            return n
    raise ValueError(
        f"cannot infer the port count from {len(lines)} data lines holding "
        f"{total_values} values; name the file '.sNp' or pass n_ports"
    )


def _to_complex(pairs: np.ndarray, data_format: str) -> np.ndarray:
    """Combine ``(..., 2)`` value pairs into complex numbers."""
    first, second = pairs[..., 0], pairs[..., 1]
    if data_format == "ri":
        return first + 1j * second
    magnitude = first if data_format == "ma" else 10 ** (first / 20)
    return magnitude * np.exp(1j * np.deg2rad(second))
