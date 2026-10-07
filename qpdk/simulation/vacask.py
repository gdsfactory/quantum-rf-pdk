"""Run the VACASK circuit simulator as a subprocess and read its results.

`VACASK <https://codeberg.org/arpadbuermen/VACASK>`_ is an analog circuit
simulator with multi-tone harmonic balance (``hb``) and small-signal analysis
around a pumped operating point (``hbac``). Devices are written in Verilog-A
and compiled on the fly by OpenVAF-Reloaded (``openvaf-r``).

VACASK is AGPL-3.0 licensed. This module only calls the ``vacask`` binary as a
separate process and parses the SPICE raw files it writes; it does not import,
link or copy any VACASK code.

Circuits and analyses are written directly in VACASK's netlist syntax. The
module only adds what is awkward to do by hand:

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

from __future__ import annotations

import itertools
import numbers
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import jinja2
import numpy as np

from qpdk.logger import logger
from qpdk.models.constants import c_0

if TYPE_CHECKING:
    import sax
    from gdsfactory.typings import CrossSectionSpec
    from skrf.vectorFitting import VectorFitting

VACASK_MODELS_DIR = Path(__file__).parent / "vacask_models"
VACASK_TEMPLATES_DIR = Path(__file__).parent / "vacask_templates"

INSTALL_HINT = (
    "Install VACASK from https://codeberg.org/arpadbuermen/VACASK/releases "
    "(the Linux .tar.gz/.deb and Windows .zip bundle the openvaf-r compiler; on macOS "
    "build both from source), then put 'vacask' on PATH or set the VACASK environment "
    "variable to the binary."
)

MODELS_EXTRA_HINT = (
    "Install it with `uv sync --extra models` or `pip install 'qpdk[models]'`."
)

_ANALYSIS = re.compile(r"^\s*analysis\s+(\S+)", re.MULTILINE)


class VacaskError(RuntimeError):
    """Raised when VACASK is missing, a simulation fails, or a raw file is malformed."""


# --------------------------------------------------------------------------- #
# Netlists
# --------------------------------------------------------------------------- #


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
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool | np.bool_):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        short = f"{float(value):g}"
        exact = float(short) == float(value)  # ruff: ignore[float-equality-comparison]
        return short if exact else repr(float(value))
    if hasattr(value, "__array__") and np.ndim(value) == 0:
        return format_value(np.asarray(value).item())
    if isinstance(value, Sequence) or hasattr(value, "__array__"):
        items = list(value)
        if items and all(
            (isinstance(item, Sequence) and not isinstance(item, str))
            or np.ndim(item) > 0
            for item in items
        ):
            return "{" + ",".join(format_value(item) for item in items) + "}"
        return "[" + ", ".join(format_value(item) for item in items) + "]"
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


# --------------------------------------------------------------------------- #
# Raw files
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RawPlot:
    """One plot of a SPICE raw file.

    Attributes:
        title: Circuit title.
        plotname: Analysis description written by the simulator.
        names: Variable names, in column order.
        data: Array of shape ``(points, variables)``; complex for complex plots.
    """

    title: str
    plotname: str
    names: tuple[str, ...]
    data: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        """Return the column of variable ``name``.

        Raises:
            KeyError: If the plot has no variable called ``name``.
        """
        try:
            return self.data[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"{name!r} not in {self.names}") from None

    def __contains__(self, name: object) -> bool:
        """Whether the plot has a variable called ``name``."""
        return name in self.names

    def split(self, name: str) -> list[RawPlot]:
        """Split a swept plot into one plot per value of the sweep variable ``name``.

        A new group starts wherever the sweep column changes value.

        Returns:
            One plot per sweep point, in order.
        """
        column = self[name]
        starts = [0, *(np.flatnonzero(column[1:] != column[:-1]) + 1), len(column)]
        return [
            RawPlot(self.title, self.plotname, self.names, self.data[a:b])
            for a, b in itertools.pairwise(starts)
        ]


class RawFile:
    """A SPICE raw file (binary or ASCII) as written by VACASK.

    The file may hold several plots back to back. Binary data is little-endian
    ``float64`` (``complex128`` for complex plots), stored point-major.

    Attributes:
        path: Path of the file.
        plots: The plots in file order.

    Raises:
        VacaskError: If the file is truncated or malformed.
    """

    def __init__(self, path: str | Path) -> None:
        """Read and parse ``path``."""
        self.path = Path(path)
        self.plots = self._parse(self.path.read_bytes())

    def __len__(self) -> int:
        """Number of plots."""
        return len(self.plots)

    def __getitem__(self, index: int) -> RawPlot:
        """Plot number ``index``."""
        return self.plots[index]

    def _error(self, message: str) -> VacaskError:
        """Return an error that names this raw file."""
        return VacaskError(f"{self.path}: {message}")

    @staticmethod
    def _parse_header(lines: list[str]) -> tuple[dict[str, str], list[str]]:
        """Parse the text header of one plot into its fields and variable names."""
        header: dict[str, str] = {}
        names: list[str] = []
        in_variables = False
        for line in lines:
            if in_variables and line[:1] in {"\t", " "}:
                names.append(line.split()[1])
                continue
            key, _, value = line.partition(":")
            in_variables = key == "Variables"
            header[key] = value.strip()
        return header, names

    def _parse_ascii(
        self, text: str, npoints: int, nvars: int, dtype: type
    ) -> np.ndarray:
        """Parse the ``Values:`` section of an ASCII raw file."""
        tokens = text.split()
        if len(tokens) < npoints * (nvars + 1):
            raise self._error("ASCII data is truncated")
        data = np.empty((npoints, nvars), dtype=dtype)
        pos = 0
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            for point in range(npoints):
                pos += 1  # point index
                for var in range(nvars):
                    token = tokens[pos]
                    pos += 1
                    if dtype is complex:
                        real, _, imag = token.partition(",")
                        data[point, var] = complex(float(real), float(imag or 0.0))
                    else:
                        data[point, var] = float(token)
        except ValueError as error:
            raise self._error(f"malformed ASCII value {tokens[pos - 1]!r}") from error
        return data

    def _parse(self, raw: bytes) -> list[RawPlot]:
        """Split a raw file into its plots and parse each one."""
        plots: list[RawPlot] = []
        pos = 0
        while pos < len(raw) and raw[pos:].strip():
            markers = [
                m
                for m in (raw.find(b"Binary:\n", pos), raw.find(b"Values:\n", pos))
                if m >= 0
            ]
            if not markers:
                raise self._error("no 'Binary:' or 'Values:' section")
            marker = min(markers)
            header, names = self._parse_header(raw[pos:marker].decode().splitlines())
            try:
                npoints = int(header["No. Points"])
                nvars = int(header["No. Variables"])
            except (KeyError, ValueError) as error:
                raise self._error(f"bad header field {error}") from error
            if len(names) != nvars:
                raise self._error(f"header lists {len(names)} of {nvars} variables")
            is_complex = "complex" in header.get("Flags", "")
            start = marker + len(b"Binary:\n")
            if raw.startswith(b"Binary:\n", marker):
                dtype = np.dtype("<c16" if is_complex else "<f8")
                end = start + npoints * nvars * dtype.itemsize
                if end > len(raw):
                    raise self._error("binary data is truncated")
                data = np.frombuffer(raw[start:end], dtype=dtype).reshape(
                    npoints, nvars
                )
            else:
                next_plot = raw.find(b"Title:", start)
                end = len(raw) if next_plot < 0 else next_plot
                data = self._parse_ascii(
                    raw[start:end].decode(),
                    npoints,
                    nvars,
                    complex if is_complex else float,
                )
            plots.append(
                RawPlot(
                    header.get("Title", ""),
                    header.get("Plotname", ""),
                    tuple(names),
                    data,
                )
            )
            pos = end
        return plots


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


def _find_binary(env_vars: tuple[str, ...], name: str) -> Path | None:
    """Return the first path set in ``env_vars``, else ``name`` on ``PATH``."""
    for var in env_vars:
        if value := os.environ.get(var):
            return Path(value)
    found = shutil.which(name)
    return Path(found) if found else None


@dataclass(frozen=True)
class Vacask:
    """The VACASK simulator, run as a separate process.

    Attributes:
        binary: Path to the ``vacask`` binary.
        openvaf: Path to the Verilog-A compiler, passed to VACASK as
            ``SIM_OPENVAF`` unless that is already set.
    """

    binary: Path
    openvaf: Path | None = None

    @classmethod
    def find(cls) -> Self:
        """Locate VACASK and OpenVAF.

        Checks the ``VACASK`` and ``QPDK_VACASK`` environment variables, then
        ``PATH``, for ``vacask``, and ``OPENVAF`` then ``PATH`` for ``openvaf-r``.

        Returns:
            A runner for the binaries found.

        Raises:
            VacaskError: If no ``vacask`` binary is found.
        """
        binary = _find_binary(("VACASK", "QPDK_VACASK"), "vacask")
        if binary is None:
            raise VacaskError(f"VACASK binary not found. {INSTALL_HINT}")
        return cls(binary, _find_binary(("OPENVAF",), "openvaf-r"))

    def run(
        self,
        netlist: Netlist,
        *,
        workdir: str | Path | None = None,
        timeout: float | None = None,
    ) -> dict[str, RawPlot]:
        """Run a netlist and return the results of every analysis.

        The netlist and its files are written to a scratch directory, which is
        also where VACASK compiles Verilog-A sources.

        Args:
            netlist: The circuit and analyses.
            workdir: Directory to run in. Defaults to a temporary directory that
                is removed afterwards.
            timeout: Timeout in seconds.

        Returns:
            ``{analysis name: plot}``, read from ``<analysis name>.raw``. Swept
            analyses hold all sweep points; see :meth:`RawPlot.split`.

        Raises:
            VacaskError: If VACASK exits with an error or a result is missing.
        """
        env = dict(os.environ)
        if "SIM_OPENVAF" not in env and self.openvaf is not None:
            env["SIM_OPENVAF"] = str(self.openvaf)

        with tempfile.TemporaryDirectory(prefix="qpdk_vacask_") as tmp:
            cwd = Path(workdir) if workdir is not None else Path(tmp)
            cwd.mkdir(parents=True, exist_ok=True)
            for name, content in netlist.all_files().items():
                if isinstance(content, Path):
                    shutil.copyfile(content, cwd / name)
                else:
                    (cwd / name).write_text(content)
            (cwd / "netlist.sim").write_text(netlist.render())

            logger.debug("Running {} in {}", self.binary, cwd)
            try:
                proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                    [str(self.binary), "netlist.sim"],
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise VacaskError(f"VACASK timed out after {timeout} s") from error
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            if proc.returncode != 0:
                raise VacaskError(
                    f"VACASK exited with code {proc.returncode}:\n{output}"
                )

            results: dict[str, RawPlot] = {}
            for name in netlist.analyses:
                raw_file = cwd / f"{name}.raw"
                if not raw_file.exists():
                    raise VacaskError(
                        f"Analysis {name!r} wrote no {raw_file.name}:\n{output}"
                    )
                results[name] = RawFile(raw_file)[0]
        return results


# --------------------------------------------------------------------------- #
# Model conversion (needs the models extra)
# --------------------------------------------------------------------------- #


def cpw_tline_params(
    length: float, cross_section: CrossSectionSpec = "cpw"
) -> tuple[float, float]:
    """Characteristic impedance and delay of a lossless CPW for ``tline_ideal``.

    Uses :func:`~qpdk.models.cpw.cpw_parameters` with zero loss tangent, so the
    result matches VACASK's lossless ``tline_ideal`` device. Needs the
    ``models`` extra.

    Args:
        length: Line length in µm.
        cross_section: CPW cross-section.

    Returns:
        ``(z0, td)``: characteristic impedance in Ω and propagation delay in s.

    Raises:
        ImportError: If the ``models`` extra is not installed.
    """
    try:
        from qpdk.models.cpw import (  # ruff: ignore[import-outside-top-level]
            cpw_parameters,
            get_cpw_dimensions,
        )
    except ImportError as error:
        raise ImportError(
            f"cpw_tline_params needs the qpdk CPW models. {MODELS_EXTRA_HINT}"
        ) from error

    width, gap = get_cpw_dimensions(cross_section)
    ep_eff, z0 = cpw_parameters(width, gap, tand=0.0)
    td = length * 1e-6 * np.sqrt(float(np.real(ep_eff))) / c_0
    return float(np.real(z0)), float(td)


def vector_fit(
    sdict: sax.SDict,
    f: np.ndarray,
    *,
    ports: Sequence[str] | None = None,
    z0: float = 50.0,
    n_poles_real: int = 1,
    n_poles_cmplx: int = 10,
    enforce_passivity: bool = True,
    n_samples: int = 1000,
) -> VectorFitting:
    """Fit S-parameters with :class:`skrf.vectorFitting.VectorFitting`.

    A thin wrapper that builds the :class:`skrf.Network` from an SDict and
    checks that the result can be used as a circuit: every pole stable and the
    model passive. Evaluate the fit with
    :meth:`~skrf.vectorFitting.VectorFitting.get_model_response` and turn it
    into a VACASK subcircuit with :func:`vector_fit_subckt`.

    Args:
        sdict: S-parameters on ``f``, referenced to ``z0``. Missing entries
            of a sparse SDict are zero.
        f: Frequencies in Hz. Cover every frequency the circuit will see,
            e.g. up to ``nharm`` times the pump frequency for harmonic balance.
        ports: Port order. Defaults to the order in which ``sdict`` names them.
            Stored as the network's ``port_names``.
        z0: Reference impedance in Ω.
        n_poles_real: Number of initial real poles.
        n_poles_cmplx: Number of initial complex-conjugate pole pairs.
        enforce_passivity: Make the model passive if the fit is not.
        n_samples: Frequency samples for the passivity enforcement.

    Returns:
        The fitted :class:`~skrf.vectorFitting.VectorFitting` object.

    Raises:
        ImportError: If the ``models`` extra is not installed.
        ValueError: If a port is missing from ``sdict``, the fit has an
            unstable pole, or it is still not passive after enforcement.
    """
    try:
        import sax  # ruff: ignore[import-outside-top-level]
        import skrf  # ruff: ignore[import-outside-top-level]
        from skrf.vectorFitting import (  # ruff: ignore[import-outside-top-level]
            VectorFitting,
        )
    except ImportError as error:
        raise ImportError(
            f"vector_fit needs sax and scikit-rf. {MODELS_EXTRA_HINT}"
        ) from error

    f = np.asarray(f, dtype=float)
    dense, port_map = sax.sdense({
        k: np.broadcast_to(v, f.shape) for k, v in sdict.items()
    })
    ports = tuple(ports) if ports is not None else tuple(port_map)
    missing = set(ports) - set(port_map)
    if missing:
        raise ValueError(f"Ports {sorted(missing)} are not in the SDict")
    order = [port_map[p] for p in ports]
    s = np.asarray(dense)[:, order][:, :, order]

    network = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s, z0=z0)
    network.port_names = list(ports)
    vf = VectorFitting(network)
    vf.vector_fit(n_poles_real=n_poles_real, n_poles_cmplx=n_poles_cmplx)
    if enforce_passivity and not vf.is_passive():
        vf.passivity_enforce(n_samples=n_samples, f_max=float(f.max()))
        if not vf.is_passive():
            raise ValueError(
                "Vector fit is still not passive after passivity enforcement; "
                "try fewer poles or a wider frequency range"
            )
    if np.any(vf.poles.real >= 0):
        raise ValueError("Vector fit has unstable poles")
    logger.info(
        "Vector fit: {} poles, RMS error {:.2e}", len(vf.poles), vf.get_rms_error()
    )
    return vf


def vector_fit_subckt(vf: VectorFitting, name: str) -> str:
    r"""Emit a vector fit as a VACASK subcircuit.

    The fitted S-matrix is

    .. math::

        S(s) = D + \sum_k \frac{R_k}{s - p_k},

    where each complex pole stands for itself and its conjugate.

    Each port is a resistor :math:`Z_0` to ground in parallel with a current
    source :math:`2 b / \sqrt{Z_0}`, which gives the port voltage
    :math:`\sqrt{Z_0}(a + b)`. The incident waves :math:`a` drive one
    first-order state per real pole and two per complex pair, and the
    reflected waves :math:`b` are weighted sums of the states. Waves and
    states are node voltages scaled by :math:`\sqrt{Z_0}`; each state
    capacitor is :math:`1/|p_k|`, so all conductances and gains are of order
    one.

    The subcircuit uses the builtin ``vccs`` and the ``resistor`` and
    ``capacitor`` models, so the netlist must load ``resistor.osdi`` and
    ``capacitor.osdi`` and define models of those names.

    Args:
        vf: A fit from :func:`vector_fit`, with a constant term and no
            proportional term, referenced to one real impedance.
        name: Subcircuit name.

    Returns:
        The subcircuit text, with terminals ``p1``, ``p2``, ... in port order.

    Raises:
        ValueError: If the fit has a proportional term, an unstable pole, or
            port-dependent or complex reference impedances.
    """
    network = vf.network
    n = network.nports
    z0_all = np.asarray(network.z0)
    z0 = float(z0_all.flat[0].real)
    if not np.allclose(z0_all, z0):
        raise ValueError("vector_fit_subckt needs one real reference impedance")
    if np.any(np.asarray(vf.proportional_coeff) != 0):
        raise ValueError("vector_fit_subckt does not support a proportional term")
    poles = np.asarray(vf.poles)
    if np.any(poles.real >= 0):
        raise ValueError("Vector fit has unstable poles")
    residues = np.asarray(vf.residues).reshape(n, n, -1)
    constant = np.asarray(vf.constant_coeff).reshape(n, n)

    terminals = [f"p{i + 1}" for i in range(n)]
    elements: list[tuple[str, tuple[str, ...], float]] = []
    for i in range(n):
        # Port: V = Z0 I + 2 sqrt(Z0) b
        elements += [
            ("r", (terminals[i],), z0),
            ("g", (terminals[i], f"b{i}"), 2 / z0),
            # Incident wave: sqrt(Z0) a = V - sqrt(Z0) b
            ("r", (f"a{i}",), 1.0),
            ("g", (f"a{i}", terminals[i]), 1.0),
            ("g", (f"a{i}", f"b{i}"), -1.0),
            ("r", (f"b{i}",), 1.0),
        ]
    for j in range(n):
        for k, pole in enumerate(poles):
            mag = abs(pole)
            u, v = f"x{j}_{k}", f"y{j}_{k}"
            # Scaled state |p| a / (s - p), split into real and imaginary parts
            elements += [
                ("c", (u,), 1 / mag),
                ("r", (u,), mag / -pole.real),
                ("g", (u, f"a{j}"), 1.0),
            ]
            if pole.imag != 0:
                elements += [
                    ("g", (u, v), -pole.imag / mag),
                    ("c", (v,), 1 / mag),
                    ("r", (v,), mag / -pole.real),
                    ("g", (v, u), pole.imag / mag),
                ]
            scale = 2 / mag if pole.imag != 0 else 1 / mag
            for i in range(n):
                residue = residues[i, j, k]
                elements.append(("g", (f"b{i}", u), scale * residue.real))
                if pole.imag != 0:
                    elements.append(("g", (f"b{i}", v), -scale * residue.imag))
        for i in range(n):
            elements.append(("g", (f"b{i}", f"a{j}"), float(constant[i, j].real)))

    port_names = network.port_names or terminals
    return (
        _templates()
        .get_template("subckt.sim.j2")
        .render(
            name=name,
            ports=port_names,
            terminals=terminals,
            n_poles=len(poles),
            elements=elements,
        )
    )
