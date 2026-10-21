"""Run the VACASK circuit simulator as a subprocess."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from qpdk.logger import logger

from .errors import VacaskError
from .netlist import Netlist
from .raw import RawFile, RawPlot

INSTALL_HINT = (
    "Install VACASK from https://codeberg.org/arpadbuermen/VACASK/releases "
    "(the Linux .tar.gz/.deb and Windows .zip bundle the openvaf-r compiler; on macOS "
    "build both from source), then put 'vacask' on PATH or set the VACASK environment "
    "variable to the binary. On Linux, OpenVAF also needs a system linker "
    "(the binutils package)."
)


def _find_binary(env_vars: tuple[str, ...], name: str) -> Path | None:
    """Return the first path set in ``env_vars``, else ``name`` on ``PATH``."""
    for var in env_vars:
        if value := os.environ.get(var):
            return Path(value)
    found = shutil.which(name)
    return Path(found) if found else None


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationResult:
    """Plots and diagnostics from one simulator invocation.

    With an explicit workdir, the deck, model files, raw results, ``vacask.log``
    and ``run.json`` remain available after success or failure. A successful
    process exit does not establish harmonic convergence or pump stability.
    """

    plots: dict[str, RawPlot]
    output: str
    elapsed: float
    binary_sha256: str


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

        See :meth:`run_record` for diagnostics and failure conditions.
        """
        return self.run_record(netlist, workdir=workdir, timeout=timeout).plots

    def run_record(
        self,
        netlist: Netlist,
        *,
        workdir: str | Path | None = None,
        timeout: float | None = None,
    ) -> SimulationResult:
        """Run a netlist with captured diagnostics and optional retained artifacts.

        Expected raw outputs from a previous invocation are removed before the
        process starts, so a reused directory cannot supply stale results.
        ``run.json`` records input and binary hashes, elapsed seconds and exit
        status. Compiler overrides and module paths are recorded from the
        effective environment; they can refer to external, unbundled files.

        Returns:
            Plots, captured output, elapsed seconds and the simulator hash.

        Raises:
            VacaskError: If analyses have duplicate or invalid names, the
                simulator cannot start, fails or times out, or an output is missing.
        """
        env = dict(os.environ)
        if "SIM_OPENVAF" not in env and self.openvaf is not None:
            env["SIM_OPENVAF"] = str(self.openvaf)

        binary = self.binary.resolve()
        try:
            binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        except OSError as error:
            raise VacaskError(
                f"Cannot read VACASK binary {binary}. {INSTALL_HINT}"
            ) from error
        with tempfile.TemporaryDirectory(prefix="qpdk_vacask_") as tmp:
            cwd = Path(workdir) if workdir is not None else Path(tmp)
            cwd.mkdir(parents=True, exist_ok=True)
            names = netlist.analyses
            if len(names) != len(set(names)) or any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names
            ):
                raise VacaskError("Analysis names must be unique simple identifiers")
            for name in names:
                (cwd / f"{name}.raw").unlink(missing_ok=True)
            inputs = {}
            for name, content in netlist.all_files().items():
                if isinstance(content, Path):
                    shutil.copyfile(content, cwd / name)
                else:
                    (cwd / name).write_text(content)
                inputs[name] = hashlib.sha256((cwd / name).read_bytes()).hexdigest()
            (cwd / "netlist.sim").write_text(netlist.render())
            inputs["netlist.sim"] = hashlib.sha256(
                (cwd / "netlist.sim").read_bytes()
            ).hexdigest()
            metadata: dict[str, Any] = {
                "binary": str(binary),
                "binary_sha256": binary_sha256,
                "inputs_sha256": inputs,
                "environment": {
                    key: env[key]
                    for key in (
                        "SIM_OPENVAF",
                        "SIM_MODULE_PATH",
                        "SIM_INCLUDE_PATH",
                        "SIM_CONFIG",
                    )
                    if key in env
                },
            }
            if compiler := shutil.which(env.get("SIM_OPENVAF", "openvaf-r")):
                metadata["compiler"] = str(Path(compiler).resolve())
                metadata["compiler_sha256"] = hashlib.sha256(
                    Path(compiler).read_bytes()
                ).hexdigest()
            started = time.perf_counter()

            logger.debug("Running {} in {}", self.binary, cwd)
            log_path = cwd / "vacask.log"
            try:
                with log_path.open("w") as stream:
                    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                        [str(binary), "netlist.sim"],
                        cwd=cwd,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired as error:
                metadata.update(elapsed=time.perf_counter() - started, status="timeout")
                (cwd / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
                raise VacaskError(f"VACASK timed out after {timeout} s") from error
            except OSError as error:
                metadata.update(
                    elapsed=time.perf_counter() - started,
                    status="launch-error",
                    error=str(error),
                )
                (cwd / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
                raise VacaskError(
                    f"Cannot launch VACASK binary {binary}: {error}"
                ) from error
            elapsed = time.perf_counter() - started
            output = log_path.read_text(errors="replace").strip()
            metadata.update(elapsed=elapsed, returncode=proc.returncode)
            (cwd / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
        return SimulationResult(
            plots=results, output=output, elapsed=elapsed, binary_sha256=binary_sha256
        )
