"""Parasitic extraction helpers using KLayout-PEX.

This module provides helper functions for running KLayout-PEX
parasitic extraction on gdsfactory components to extract
capacitance matrices.

Note:
    KLayout-PEX requires external installation of FasterCap or MAGIC
    for the actual field solving. The 2.5D engine is built-in.

See Also:
    https://github.com/iic-jku/klayout-pex for installation and documentation.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gdsfactory as gf
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PEXEngine",
    "PEXResult",
    "is_kpex_available",
    "parse_capacitance_matrix_from_log",
    "run_capacitance_extraction",
]

logger = logging.getLogger(__name__)

PEXEngine = Literal["fastercap", "2.5D", "magic"]


@dataclass
class PEXResult:
    """Result of parasitic extraction.

    Attributes:
        capacitance_matrix: Maxwell capacitance matrix in Farads.
            Rows and columns correspond to conductor nets.
        net_names: Names of conductor nets in the same order as the matrix.
        csv_netlist: Extracted parasitic netlist in CSV format.
        spice_netlist: Extracted parasitic netlist in SPICE format.
        log_output: Full log output from the extraction run.
        success: Whether the extraction completed successfully.
    """

    capacitance_matrix: NDArray[np.float64]
    net_names: list[str]
    csv_netlist: str = ""
    spice_netlist: str = ""
    log_output: str = ""
    success: bool = True
    error_message: str = ""

    def get_mutual_capacitance(self, net1: str, net2: str) -> float:
        """Get mutual capacitance between two nets.

        Args:
            net1: Name of first net.
            net2: Name of second net.

        Returns:
            Mutual capacitance in Farads. Returns negative value
            from the Maxwell capacitance matrix (off-diagonal elements).

        Raises:
            ValueError: If net names are not found.
        """
        if net1 not in self.net_names:
            msg = f"Net '{net1}' not found. Available: {self.net_names}"
            raise ValueError(msg)
        if net2 not in self.net_names:
            msg = f"Net '{net2}' not found. Available: {self.net_names}"
            raise ValueError(msg)

        i = self.net_names.index(net1)
        j = self.net_names.index(net2)
        # Off-diagonal elements in Maxwell matrix are negative mutual capacitances
        return -self.capacitance_matrix[i, j]

    def get_self_capacitance(self, net: str) -> float:
        """Get self-capacitance of a net.

        Args:
            net: Name of the net.

        Returns:
            Self-capacitance in Farads (diagonal element of Maxwell matrix).

        Raises:
            ValueError: If net name is not found.
        """
        if net not in self.net_names:
            msg = f"Net '{net}' not found. Available: {self.net_names}"
            raise ValueError(msg)

        i = self.net_names.index(net)
        return self.capacitance_matrix[i, i]

    def summary(self) -> str:
        """Return a human-readable summary of extraction results."""
        lines = ["Capacitance Matrix (fF):"]
        lines.append(f"  Nets: {', '.join(self.net_names)}")
        lines.append("")

        # Format matrix
        for i, net_i in enumerate(self.net_names):
            row_vals = [
                f"{self.capacitance_matrix[i, j] * 1e15:.3f}"
                for j in range(len(self.net_names))
            ]
            lines.append(f"  {net_i}: [{', '.join(row_vals)}]")

        lines.append("")
        lines.append("Mutual Capacitances (fF):")
        for i in range(len(self.net_names)):
            for j in range(i + 1, len(self.net_names)):
                c_mutual = -self.capacitance_matrix[i, j] * 1e15
                lines.append(
                    f"  {self.net_names[i]} <-> {self.net_names[j]}: {c_mutual:.3f} fF"
                )

        return "\n".join(lines)


def _get_kpex_prefix() -> list[str] | None:
    """Try to find the kpex command prefix.

    Returns:
        List of command parts (e.g., ["kpex"] or ["uv", "run", "kpex"]) or None if not found.
    """
    for prefix in [["kpex"], ["uv", "run", "kpex"]]:
        try:
            result = subprocess.run(
                [*prefix, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return prefix
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def is_kpex_available() -> bool:
    """Check if the kpex command is available.

    Returns:
        True if kpex is installed and callable.
    """
    return _get_kpex_prefix() is not None


def parse_capacitance_matrix_from_log(
    log_output: str,
) -> tuple[NDArray[np.float64], list[str]]:
    """Parse capacitance matrix from KLayout-PEX log output.

    The FasterCap engine outputs the Maxwell capacitance matrix in the format::

        Capacitance matrix is:
        Dimension 3 x 3
        g1_VSUBS  5.2959e-09 -4.46971e-10 -1.67304e-09
        g2_C1  -5.56106e-10 1.5383e-08 -1.47213e-08
        g3_C0  -1.69838e-09 -1.48846e-08 1.64502e-08

    Args:
        log_output: Full log output from kpex run.

    Returns:
        Tuple of (capacitance_matrix, net_names).
        Matrix values are in Farads.

    Raises:
        ValueError: If capacitance matrix cannot be parsed.
    """
    # Find the capacitance matrix section
    matrix_match = re.search(
        r"Capacitance matrix is:\s*\n"
        r"Dimension\s+(\d+)\s*x\s*(\d+)\s*\n"
        r"((?:g\d+_\S+\s+[-\d.eE+]+(?:\s+[-\d.eE+]+)*\s*\n?)+)",
        log_output,
        re.MULTILINE,
    )

    if not matrix_match:
        # Try alternative format from CSV output
        csv_match = re.search(
            r"Device;Net1;Net2;Capacitance \[fF\]\s*\n((?:Cext_\d+_\d+;[^;]+;[^;]+;[\d.]+\s*\n?)+)",
            log_output,
        )
        if csv_match:
            return _parse_csv_capacitance(csv_match.group(1))
        msg = "Could not find capacitance matrix in log output"
        raise ValueError(msg)

    dim = int(matrix_match.group(1))
    matrix_lines = matrix_match.group(3).strip().split("\n")

    net_names: list[str] = []
    matrix_values: list[list[float]] = []

    for line in matrix_lines:
        parts = line.split()
        if len(parts) < dim + 1:
            continue

        # Extract net name (e.g., "g1_VSUBS" -> "VSUBS")
        net_name = parts[0]
        if "_" in net_name:
            net_name = net_name.split("_", 1)[1]
        net_names.append(net_name)

        # Extract capacitance values
        values = [float(v) for v in parts[1 : dim + 1]]
        matrix_values.append(values)

    if len(net_names) != dim or len(matrix_values) != dim:
        msg = f"Parsed matrix dimension mismatch: expected {dim}, got {len(net_names)} nets"
        raise ValueError(msg)

    return np.array(matrix_values), net_names


def _parse_csv_capacitance(csv_content: str) -> tuple[NDArray[np.float64], list[str]]:
    """Parse capacitance from CSV netlist format.

    CSV format::

        Cext_0_1;VSUBS;C1;0.5
        Cext_0_2;VSUBS;C0;1.69
        Cext_1_2;C1;C0;14.8

    Args:
        csv_content: CSV content with capacitor definitions.

    Returns:
        Tuple of (capacitance_matrix, net_names).
    """
    # Collect all unique net names and capacitances
    net_set: set[str] = set()
    capacitances: list[tuple[str, str, float]] = []

    for line in csv_content.strip().split("\n"):
        parts = line.split(";")
        if len(parts) >= 4:
            net1, net2 = parts[1], parts[2]
            cap_ff = float(parts[3])
            net_set.add(net1)
            net_set.add(net2)
            capacitances.append((net1, net2, cap_ff * 1e-15))  # fF -> F

    net_names = sorted(net_set)
    n = len(net_names)
    matrix = np.zeros((n, n))

    # Build Maxwell capacitance matrix
    # Off-diagonal elements are negative mutual capacitances
    # Diagonal elements are sum of all capacitances connected to that net
    for net1, net2, cap in capacitances:
        i = net_names.index(net1)
        j = net_names.index(net2)
        matrix[i, j] = -cap
        matrix[j, i] = -cap
        matrix[i, i] += cap
        matrix[j, j] += cap

    return matrix, net_names


def run_capacitance_extraction(
    component: gf.Component,
    engine: PEXEngine = "2.5D",
    output_dir: Path | str | None = None,
    cell_name: str | None = None,
    blackbox_devices: bool = False,
    cleanup: bool = True,
    timeout: int = 300,
    extra_args: list[str] | None = None,
) -> PEXResult:
    """Run capacitance extraction on a gdsfactory component using KLayout-PEX.

    This function exports the component to GDS, runs the kpex command-line
    tool, and parses the resulting capacitance matrix.

    Note:
        KLayout-PEX is not specific to any PDK, but it requires technology
        definition files. For custom PDKs like QPDK, you may need to create
        a technology definition. This function currently demonstrates the
        workflow without full QPDK technology integration.

    Args:
        component: gdsfactory component to extract parasitics from.
        engine: Extraction engine to use. Options:
            - "fastercap": 3D field solver (requires FasterCap installation)
            - "2.5D": Analytical 2.5D engine (built-in, under development)
            - "magic": MAGIC wrapper (requires MAGIC installation)
        output_dir: Directory for output files. If None, uses a temp directory.
        cell_name: Cell name for extraction. Defaults to component name.
        blackbox_devices: If True, blackbox devices like MIM/MOM caps.
        cleanup: If True, clean up temporary files after extraction.
        timeout: Timeout in seconds for the extraction process.
        extra_args: Additional command-line arguments to pass to kpex.

    Returns:
        PEXResult containing the capacitance matrix and extraction details.

    Raises:
        RuntimeError: If kpex is not available or extraction fails.

    Example:
        >>> from qpdk.cells.capacitor import interdigital_capacitor
        >>> cap = interdigital_capacitor(fingers=4)
        >>> result = run_capacitance_extraction(cap, engine="2.5D")
        >>> print(result.summary())  # doctest: +SKIP
    """
    kpex_prefix = _get_kpex_prefix()
    if kpex_prefix is None:
        return PEXResult(
            capacitance_matrix=np.array([]),
            net_names=[],
            success=False,
            error_message="kpex command not found. Install with: pip install klayout-pex",
        )

    # Create output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="kpex_")
        output_path = Path(temp_dir)
        should_cleanup = cleanup
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        should_cleanup = False

    try:
        # Export component to GDS
        gds_path = output_path / f"{component.name}.gds"
        component.write_gds(gds_path)

        # Build kpex command
        cell = cell_name or component.name
        cmd = [
            *kpex_prefix,
            "--gds",
            str(gds_path),
            "--cell",
            cell,
            "--out_dir",
            str(output_path),
        ]

        # Add engine flag
        if engine == "fastercap":
            cmd.append("--fastercap")
        elif engine == "2.5D":
            cmd.append("--2.5D")
        elif engine == "magic":
            cmd.append("--magic")

        # Add blackbox option
        if blackbox_devices:
            cmd.extend(["--blackbox", "y"])
        else:
            cmd.extend(["--blackbox", "n"])

        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Running kpex: %s", " ".join(cmd))

        # Run extraction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=output_path,
        )

        log_output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            return PEXResult(
                capacitance_matrix=np.array([]),
                net_names=[],
                log_output=log_output,
                success=False,
                error_message=f"kpex exited with code {result.returncode}",
            )

        # Parse results
        try:
            matrix, net_names = parse_capacitance_matrix_from_log(log_output)
        except ValueError as e:
            return PEXResult(
                capacitance_matrix=np.array([]),
                net_names=[],
                log_output=log_output,
                success=False,
                error_message=str(e),
            )

        # Read SPICE netlist if available
        spice_path = output_path / cell / f"{cell}.pex.spice"
        spice_netlist = ""
        if spice_path.exists():
            spice_netlist = spice_path.read_text()

        # Read CSV netlist if available
        csv_netlist = ""
        csv_path = output_path / cell / f"{cell}.pex.csv"
        if csv_path.exists():
            csv_netlist = csv_path.read_text()

        return PEXResult(
            capacitance_matrix=matrix,
            net_names=net_names,
            csv_netlist=csv_netlist,
            spice_netlist=spice_netlist,
            log_output=log_output,
            success=True,
        )

    finally:
        if should_cleanup:
            import shutil

            shutil.rmtree(output_path, ignore_errors=True)


if __name__ == "__main__":
    # Simple test
    print(f"kpex available: {is_kpex_available()}")

    # Test parsing
    sample_log = """
Capacitance matrix is:
Dimension 2 x 2
g1_NET1  1.5e-14 -1.2e-14
g2_NET2  -1.2e-14 1.8e-14
"""
    matrix, nets = parse_capacitance_matrix_from_log(sample_log)
    print(f"Parsed nets: {nets}")
    print(f"Matrix:\n{matrix}")
