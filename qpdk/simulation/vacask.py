"""Run the VACASK circuit simulator as a subprocess and read its results.

`VACASK <https://codeberg.org/arpadbuermen/VACASK>`_ is an analog circuit
simulator with multi-tone harmonic balance (``hb``) and small-signal analysis
around a pumped operating point (``hbac``). Devices are written in Verilog-A
and compiled on the fly by OpenVAF-Reloaded (``openvaf-r``).

VACASK is AGPL-3.0 licensed. This module only calls the ``vacask`` binary as a
separate process and parses the SPICE raw files it writes; it does not import,
link or copy any VACASK code.

Example:
    >>> from qpdk.simulation.vacask import run_vacask, vacask_model_path
    >>> netlist = '''JJ resonance
    ... load "josephson_junction.va"
    ... load "capacitor.osdi"
    ... model isource isource
    ... model capacitor capacitor
    ... model jj jj
    ... iin (0 1) isource dc=0 mag=1n
    ... j1 (1 0) jj ic=1u
    ... c1 (1 0) capacitor c=2p
    ... control
    ...   options tolscale=1e-6 reltol=1e-6
    ...   analysis ac1 ac from=1G to=10G mode="lin" points=1000
    ... endc
    ... '''
    >>> results = run_vacask(
    ...     netlist,
    ...     files={"josephson_junction.va": vacask_model_path("josephson_junction")},
    ... )
    >>> results["ac1"]["1"]  # complex node voltage
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qpdk.logger import logger
from qpdk.models.constants import c_0

if TYPE_CHECKING:
    import sax
    from gdsfactory.typings import CrossSectionSpec
    from skrf.vectorFitting import VectorFitting

VACASK_MODELS_DIR = Path(__file__).parent / "vacask_models"

INSTALL_HINT = (
    "Install VACASK from https://codeberg.org/arpadbuermen/VACASK/releases "
    "(the Linux .tar.gz/.deb and Windows .zip bundle the openvaf-r compiler; on macOS "
    "build both from source), then put 'vacask' on PATH or set the VACASK environment "
    "variable to the binary."
)

_ANALYSIS_RE = re.compile(r"^\s*analysis\s+(\S+)", re.MULTILINE)
_CONTROL_RE = re.compile(r"^\s*control\s*$", re.MULTILINE)
_ABORT_RE = re.compile(r"^\s*abort\s+always\b", re.MULTILINE)


class VacaskError(RuntimeError):
    """Raised when VACASK is missing or a simulation fails."""


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


def _parse_ascii_values(text: str, npoints: int, nvars: int, dtype: type) -> np.ndarray:
    """Parse the ``Values:`` section of an ASCII raw file."""
    tokens = text.split()
    data = np.empty((npoints, nvars), dtype=dtype)
    pos = 0
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
    return data


def read_raw(path: str | Path) -> list[RawPlot]:
    """Read a SPICE raw file (binary or ASCII) as written by VACASK.

    The file may hold several plots back to back. Binary data is little-endian
    ``float64`` (``complex128`` for complex plots), stored point-major.

    Args:
        path: Path to the ``.raw`` file.

    Returns:
        The plots in file order.

    Raises:
        VacaskError: If the file is truncated or malformed.
    """
    raw = Path(path).read_bytes()
    plots: list[RawPlot] = []
    pos = 0
    while pos < len(raw):
        if not raw[pos:].strip():
            break
        binary = raw.find(b"Binary:\n", pos)
        values = raw.find(b"Values:\n", pos)
        marker = (
            min(m for m in (binary, values) if m >= 0)
            if max(binary, values) >= 0
            else -1
        )
        if marker < 0:
            raise VacaskError(f"{path}: no 'Binary:' or 'Values:' section")
        header, names = _parse_header(raw[pos:marker].decode().splitlines())
        npoints = int(header["No. Points"])
        nvars = int(header["No. Variables"])
        if len(names) != nvars:
            raise VacaskError(f"{path}: header lists {len(names)} of {nvars} variables")
        is_complex = "complex" in header.get("Flags", "")
        start = marker + len(b"Binary:\n")
        if marker == binary:
            dtype = np.dtype("<c16" if is_complex else "<f8")
            end = start + npoints * nvars * dtype.itemsize
            if end > len(raw):
                raise VacaskError(f"{path}: binary data is truncated")
            data = np.frombuffer(raw[start:end], dtype=dtype).reshape(npoints, nvars)
        else:
            next_plot = raw.find(b"Title:", start)
            end = len(raw) if next_plot < 0 else next_plot
            data = _parse_ascii_values(
                raw[start:end].decode(),
                npoints,
                nvars,
                complex if is_complex else float,
            )
        plots.append(
            RawPlot(
                header.get("Title", ""), header.get("Plotname", ""), tuple(names), data
            )
        )
        pos = end
    return plots


def _find_binary(env_vars: tuple[str, ...], name: str) -> Path | None:
    """Return the first path set in ``env_vars``, else ``name`` on ``PATH``."""
    for var in env_vars:
        if value := os.environ.get(var):
            return Path(value)
    found = shutil.which(name)
    return Path(found) if found else None


def find_vacask() -> Path:
    """Locate the ``vacask`` binary.

    Checks the ``VACASK`` and ``QPDK_VACASK`` environment variables, then ``PATH``.

    Returns:
        Path to the binary.

    Raises:
        VacaskError: If no binary is found.
    """
    path = _find_binary(("VACASK", "QPDK_VACASK"), "vacask")
    if path is None:
        raise VacaskError(f"VACASK binary not found. {INSTALL_HINT}")
    return path


def vacask_model_path(name: str) -> Path:
    """Path of a Verilog-A model shipped with qpdk, e.g. ``"josephson_junction"``.

    Returns:
        Path to ``qpdk/simulation/vacask_models/<name>.va``.
    """
    return VACASK_MODELS_DIR / f"{name}.va"


def _prepare_netlist(netlist: str) -> str:
    """Check for a control block and add ``abort always`` to it if missing."""
    if not _CONTROL_RE.search(netlist):
        raise VacaskError("Netlist has no 'control' block")
    if _ABORT_RE.search(netlist):
        return netlist
    # VACASK exits 0 after a failed analysis unless told to abort
    return _CONTROL_RE.sub(lambda m: f"{m.group(0)}\n  abort always", netlist, count=1)


def run_vacask(
    netlist: str,
    files: Mapping[str, str | Path] | None = None,
    *,
    workdir: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, RawPlot]:
    """Run a VACASK netlist and return the results of every analysis.

    The netlist and ``files`` are written to a scratch directory, which is also
    where VACASK compiles Verilog-A sources. ``abort always`` is added to the
    control block if missing so that a failed analysis gives a nonzero exit code.

    Args:
        netlist: Netlist text, including a ``control`` block.
        files: Extra files to place next to the netlist, as ``{name: content}``.
            A :class:`~pathlib.Path` value is copied; a string is written as text.
        workdir: Directory to run in. Defaults to a temporary directory that is
            removed afterwards.
        timeout: Timeout in seconds.

    Returns:
        ``{analysis name: plot}``, read from ``<analysis name>.raw``.

    Raises:
        VacaskError: If VACASK exits with an error or an expected result is missing.
    """
    netlist = _prepare_netlist(netlist)
    analyses = _ANALYSIS_RE.findall(netlist.split("endc")[0])

    vacask = find_vacask()
    env = dict(os.environ)
    if "SIM_OPENVAF" not in env and (
        openvaf := _find_binary(("OPENVAF",), "openvaf-r")
    ):
        env["SIM_OPENVAF"] = str(openvaf)

    with tempfile.TemporaryDirectory(prefix="qpdk_vacask_") as tmp:
        cwd = Path(workdir) if workdir is not None else Path(tmp)
        cwd.mkdir(parents=True, exist_ok=True)
        for name, content in (files or {}).items():
            if isinstance(content, Path):
                shutil.copyfile(content, cwd / name)
            else:
                (cwd / name).write_text(content)
        (cwd / "netlist.sim").write_text(netlist)

        logger.debug("Running {} in {}", vacask, cwd)
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [str(vacask), "netlist.sim"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = f"{proc.stdout}\n{proc.stderr}".strip()
        if proc.returncode != 0:
            raise VacaskError(f"VACASK exited with code {proc.returncode}:\n{output}")

        results: dict[str, RawPlot] = {}
        for name in analyses:
            raw_file = cwd / f"{name}.raw"
            if not raw_file.exists():
                raise VacaskError(
                    f"Analysis {name!r} wrote no {raw_file.name}:\n{output}"
                )
            results[name] = read_raw(raw_file)[0]
    return results


def cpw_tline_params(
    length: float, cross_section: CrossSectionSpec = "cpw"
) -> tuple[float, float]:
    """Characteristic impedance and delay of a lossless CPW for ``tline_ideal``.

    Uses :func:`~qpdk.models.cpw.cpw_parameters` with zero loss tangent, so the
    result matches VACASK's lossless ``tline_ideal`` device.

    Args:
        length: Line length in µm.
        cross_section: CPW cross-section.

    Returns:
        ``(z0, td)``: characteristic impedance in Ω and propagation delay in s.
    """
    from qpdk.models.cpw import (  # ruff: ignore[import-outside-top-level]
        cpw_parameters,
        get_cpw_dimensions,
    )

    width, gap = get_cpw_dimensions(cross_section)
    ep_eff, z0 = cpw_parameters(width, gap, tand=0.0)
    td = length * 1e-6 * np.sqrt(float(np.real(ep_eff))) / c_0
    return float(np.real(z0)), float(td)


def _fmt(value: float) -> str:
    """Format a number for a netlist without losing precision."""
    return f"{value:.17g}"


def vector_fit_subckt(
    sdict: sax.SDict,
    f: np.ndarray,
    name: str,
    *,
    ports: Sequence[str] | None = None,
    z0: float = 50.0,
    n_poles_real: int = 1,
    n_poles_cmplx: int = 10,
    enforce_passivity: bool = True,
    n_samples: int = 1000,
) -> tuple[str, VectorFitting]:
    r"""Fit an S-parameter model with a rational function and emit a VACASK subcircuit.

    The S-matrix is fitted with :class:`skrf.vectorFitting.VectorFitting` as

    .. math::

        S(s) = D + \sum_k \frac{R_k}{s - p_k},

    with complex poles in conjugate pairs. Each port is a resistor :math:`Z_0`
    to ground in parallel with a current source :math:`2 b / \sqrt{Z_0}`, which
    gives the port voltage :math:`\sqrt{Z_0}(a + b)`. The incident waves
    :math:`a` drive one first-order state per real pole and two per complex
    pair, and the reflected waves :math:`b` are weighted sums of the states.
    Waves and states are node voltages scaled by :math:`\sqrt{Z_0}`; each state
    capacitor is :math:`1/|p_k|`, so all conductances and gains are of order one.

    The subcircuit uses the builtin ``vccs`` and the ``resistor`` and
    ``capacitor`` models, so the netlist must load ``resistor.osdi`` and
    ``capacitor.osdi``.

    Args:
        sdict: S-parameters on ``f``, referenced to ``z0``.
        f: Frequencies in Hz. Cover every frequency the circuit will see,
            e.g. up to ``nharm`` times the pump frequency for harmonic balance.
        name: Subcircuit name.
        ports: Port order. Defaults to the sorted port names of ``sdict``.
        z0: Reference impedance in Ω.
        n_poles_real: Number of initial real poles.
        n_poles_cmplx: Number of initial complex-conjugate pole pairs.
        enforce_passivity: Make the model passive if the fit is not.
        n_samples: Frequency samples for the passivity enforcement.

    Returns:
        ``(subckt, fit)``: the subcircuit text and the fitted model, whose
        :meth:`~skrf.vectorFitting.VectorFitting.get_model_response` gives the
        S-parameters the subcircuit implements.

    Raises:
        ValueError: If the fit has an unstable pole.
    """
    import skrf  # ruff: ignore[import-outside-top-level]
    from skrf.vectorFitting import (  # ruff: ignore[import-outside-top-level]
        VectorFitting,
    )

    f = np.asarray(f, dtype=float)
    ports = list(ports) if ports is not None else sorted({p for p, _ in sdict})
    n = len(ports)
    s = np.zeros((len(f), n, n), dtype=complex)
    for i, pi in enumerate(ports):
        for j, pj in enumerate(ports):
            s[:, i, j] = np.broadcast_to(np.asarray(sdict[pi, pj]), f.shape)

    network = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s, z0=z0)
    fit = VectorFitting(network)
    fit.vector_fit(n_poles_real=n_poles_real, n_poles_cmplx=n_poles_cmplx)
    if enforce_passivity and not fit.is_passive():
        fit.passivity_enforce(n_samples=n_samples, f_max=float(f.max()))
        if not fit.is_passive():
            logger.warning(
                "Vector fit of {} is still not passive; try fewer poles", name
            )
    logger.info(
        "Vector fit of {}: {} poles, RMS error {:.2e}",
        name,
        len(fit.poles),
        fit.get_rms_error(),
    )
    if np.any(fit.poles.real >= 0):
        raise ValueError("Vector fit has unstable poles")

    terminals = [f"p{i + 1}" for i in range(n)]
    lines = [
        f"// Rational model of ports {', '.join(ports)}, {len(fit.poles)} poles",
        f"subckt {name} ({' '.join(terminals)})",
        "  model vccs vccs",
    ]
    count = 0

    def vccs(node: str, control: str, gain: float) -> None:
        nonlocal count
        count += 1
        lines.append(f"  g{count} (0 {node} {control} 0) vccs gain={_fmt(gain)}")

    def resistor(node: str, r: float) -> None:
        nonlocal count
        count += 1
        lines.append(f"  r{count} ({node} 0) resistor r={_fmt(r)}")

    def capacitor(node: str, c: float) -> None:
        nonlocal count
        count += 1
        lines.append(f"  c{count} ({node} 0) capacitor c={_fmt(c)}")

    for i in range(n):
        # Port: V = Z0 I + 2 sqrt(Z0) b
        resistor(terminals[i], z0)
        vccs(terminals[i], f"b{i}", 2 / z0)
        # Incident wave: sqrt(Z0) a = V - sqrt(Z0) b
        resistor(f"a{i}", 1.0)
        vccs(f"a{i}", terminals[i], 1.0)
        vccs(f"a{i}", f"b{i}", -1.0)
        resistor(f"b{i}", 1.0)

    for j in range(n):
        for k, pole in enumerate(fit.poles):
            mag = abs(pole)
            u, v = f"x{j}_{k}", f"y{j}_{k}"
            # Scaled state |p| a / (s - p), split into real and imaginary parts
            capacitor(u, 1 / mag)
            resistor(u, mag / -pole.real)
            vccs(u, f"a{j}", 1.0)
            if pole.imag != 0:
                vccs(u, v, -pole.imag / mag)
                capacitor(v, 1 / mag)
                resistor(v, mag / -pole.real)
                vccs(v, u, pole.imag / mag)
            for i in range(n):
                residue = fit.residues[i * n + j, k]
                scale = 2 / mag if pole.imag != 0 else 1 / mag
                vccs(f"b{i}", u, scale * residue.real)
                if pole.imag != 0:
                    vccs(f"b{i}", v, -scale * residue.imag)
        for i in range(n):
            vccs(f"b{i}", f"a{j}", fit.constant_coeff[i * n + j])

    lines.append("ends")
    return "\n".join(lines) + "\n", fit
