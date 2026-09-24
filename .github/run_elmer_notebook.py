"""Check Elmer results, execute CI smoke, or refresh high-accuracy output."""

import argparse
import hashlib
import logging
import math
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logger = logging.getLogger(__name__)

CI_MODE_MARKER = (
    "Elmer notebook run mode: CI smoke "
    "(element_order=2, tolerance=3.0 %, n_processes=1)"
)
SAVED_MODE_MARKER = (
    "Elmer notebook run mode: high accuracy "
    "(element_order=3, tolerance=0.5 %, n_processes="
)
SAVED_FINAL_MARKER = "(element_order=3, mesh factor 0.25)"
SAVED_CONVERGENCE_PATTERN = re.compile(
    r"Mesh convergence check passed: final changes ([0-9.]+) % and "
    r"([0-9.]+) % <= tolerance 0.5 %"
)
SAVED_TABLE_PATTERN = re.compile(
    r"(?m)^[ \t]*(\d+)[ \t]+(0\.\d+)[ \t]+([0-9.]+)[ \t]+(n/a|[0-9.]+)[ \t]*$"
)
# Printed by the final-mesh result cell, the convergence check and the sanity loop.
FINAL_RESULT_MARKER = "Final reported mutual capacitance"
CONVERGENCE_MARKER = "Mesh convergence check passed"
SANITY_MARKER = "All checks passed."
DOMAIN_MARKER = "Lateral-domain sensitivity check passed"
MATRIX_PATTERN = re.compile(
    r"Final Maxwell capacitance matrix \(fF\):\s*\[\[([^\]]+)\]\s*\[([^\]]+)\]\]"
)
CI_TOLERANCE_PATTERN = re.compile(
    r"(?m)^CONVERGENCE_TOLERANCE = 0\.\d+ if CI_FAST else 0\.005$"
)
HIGH_CODE_DIGEST_KEY = "qpdk_elmer_high_code_sha256"


def _high_code_digest(notebook: nbformat.NotebookNode) -> str:
    """Hash the code used for the saved high-accuracy result."""
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    code = "\0".join(sources)
    # The CI-only tolerance may change without changing the saved cubic solve.
    code, count = CI_TOLERANCE_PATTERN.subn(
        "CONVERGENCE_TOLERANCE = <ci-tolerance> if CI_FAST else 0.005", code
    )
    if count != 1:
        raise RuntimeError("Elmer notebook is missing its high-accuracy profile")
    return hashlib.sha256(code.encode()).hexdigest()


def _final_matrix(output: str) -> list[list[float]]:
    """Parse the final 2x2 Maxwell capacitance matrix, in fF, from the output."""
    match = MATRIX_PATTERN.search(output)
    if match is None:
        raise RuntimeError("notebook output missing the final 2x2 Maxwell matrix")
    matrix = [[float(value) for value in row.split()] for row in match.groups()]
    if any(len(row) != 2 for row in matrix):
        raise RuntimeError("notebook output has an invalid Maxwell matrix shape")
    return matrix


def _stream_output(notebook: nbformat.NotebookNode) -> str:
    """Collect printed output from every notebook cell."""
    return "\n".join(
        item.text
        for cell in notebook.cells
        for item in cell.get("outputs", [])
        if item.output_type == "stream"
    )


def _has_convergence_plot(notebook: nbformat.NotebookNode) -> bool:
    """Check that the convergence cell rendered its figure."""
    cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "ax_change.plot" in cell.source
    ]
    return len(cells) == 1 and any(
        "image/png" in item.get("data", {}) for item in cells[0].outputs
    )


def _check_saved_output(notebook: nbformat.NotebookNode) -> None:
    """Require a complete saved high-accuracy run alongside the CI smoke run."""
    if notebook.metadata.get(HIGH_CODE_DIGEST_KEY) != _high_code_digest(notebook):
        raise RuntimeError("saved Elmer output does not match the notebook code")
    if re.search(
        r"Aalto|Triton|/scratch/work|savolan",
        nbformat.writes(notebook),
        re.IGNORECASE,
    ):
        raise RuntimeError("saved Elmer notebook contains compute-site details")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells) or any(
        item.output_type == "error" for cell in code_cells for item in cell.outputs
    ):
        raise RuntimeError("saved Elmer notebook has missing or failed execution")
    if not _has_convergence_plot(notebook):
        raise RuntimeError("saved Elmer notebook is missing the convergence plot")

    output = _stream_output(notebook)
    if SAVED_MODE_MARKER not in output or SAVED_FINAL_MARKER not in output:
        raise RuntimeError("saved Elmer notebook is not the cubic finest-mesh run")
    match = SAVED_CONVERGENCE_PATTERN.search(output)
    if match is None or any(float(value) > 0.5 for value in match.groups()):
        raise RuntimeError(
            "saved Elmer notebook is missing the 0.5 % convergence check"
        )
    passes = [
        (int(index), float(factor))
        for index, factor, _, _ in SAVED_TABLE_PATTERN.findall(output)
    ]
    if passes != list(enumerate((0.5, 0.4, 0.35, 0.3, 0.25), start=1)):
        raise RuntimeError(
            "saved Elmer notebook has an incomplete mesh-refinement table"
        )
    for marker in (FINAL_RESULT_MARKER, DOMAIN_MARKER, SANITY_MARKER):
        if marker not in output:
            raise RuntimeError(f"saved Elmer notebook output missing {marker!r}")
    _final_matrix(output)


def _execute(notebook: nbformat.NotebookNode, timeout: int) -> None:
    """Execute every cell with no saved output available to mask a skipped cell."""
    # Clear saved output so a skipped cell cannot satisfy the result check.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    try:
        NotebookClient(
            notebook, timeout=timeout, kernel_name="python3", force_raise_errors=True
        ).execute()
    finally:
        logger.info("%s", _stream_output(notebook))


def save_high_output(path: Path) -> None:
    """Replace saved output only after a complete high-accuracy execution."""
    notebook = nbformat.read(path, as_version=4)
    _execute(notebook, timeout=43200)
    notebook.metadata[HIGH_CODE_DIGEST_KEY] = _high_code_digest(notebook)
    _check_saved_output(notebook)
    nbformat.write(notebook, path)


def main(path: Path) -> None:
    """Run the notebook and check the reported capacitance and its convergence."""
    saved_notebook = nbformat.read(path, as_version=4)
    _check_saved_output(saved_notebook)
    notebook = nbformat.read(path, as_version=4)
    _execute(notebook, timeout=2400)

    output = _stream_output(notebook)

    if any(
        cell.execution_count is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    ):
        raise RuntimeError("notebook skipped a code cell")

    if not _has_convergence_plot(notebook):
        raise RuntimeError("notebook did not render the convergence plot")

    if CI_MODE_MARKER not in output:
        raise RuntimeError(
            "notebook did not run the CI smoke profile; QPDK_ELMER_CI_FAST=1 must be "
            "set for this job"
        )

    # A failed convergence or sanity check raises in the notebook, so the markers
    # below also assert that those checks actually ran and passed.
    for marker in (
        FINAL_RESULT_MARKER,
        CONVERGENCE_MARKER,
        DOMAIN_MARKER,
        SANITY_MARKER,
    ):
        if marker not in output:
            raise RuntimeError(f"notebook output missing {marker!r}")

    matrix = _final_matrix(output)
    mutual_fF = -matrix[0][1]
    if not math.isfinite(mutual_fF) or mutual_fF <= 0:
        raise RuntimeError(f"final mutual capacitance is not positive: {mutual_fF}")
    if not 5.0 < mutual_fF < 40.0:
        raise RuntimeError(
            f"final mutual capacitance outside the 5-40 fF reference range: {mutual_fF}"
        )
    logger.info("CI numerical check: final mutual capacitance = %.3f fF", mutual_fF)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--save-high-output", action="store_true")
    args = parser.parse_args()
    if args.save_high_output:
        save_high_output(args.path)
    else:
        main(args.path)
