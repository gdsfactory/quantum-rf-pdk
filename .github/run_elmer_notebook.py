"""Execute the Elmer notebook and require a fresh capacitance result."""

import logging
import math
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logger = logging.getLogger(__name__)

# Printed by the final-mesh result cell, the convergence check and the sanity loop.
FINAL_RESULT_MARKER = "Final reported mutual capacitance"
CONVERGENCE_MARKER = "Mesh convergence check passed"
SANITY_MARKER = "All checks passed."
DOMAIN_MARKER = "Lateral-domain sensitivity check passed"
MATRIX_PATTERN = re.compile(
    r"Final Maxwell capacitance matrix \(fF\):\s*\[\[([^\]]+)\]\s*\[([^\]]+)\]\]"
)


def _final_matrix(output: str) -> list[list[float]]:
    """Parse the final 2x2 Maxwell capacitance matrix, in fF, from the output."""
    match = MATRIX_PATTERN.search(output)
    if match is None:
        raise RuntimeError("notebook output missing the final 2x2 Maxwell matrix")
    matrix = [[float(value) for value in row.split()] for row in match.groups()]
    if any(len(row) != 2 for row in matrix):
        raise RuntimeError("notebook output has an invalid Maxwell matrix shape")
    return matrix


def main(path: Path) -> None:
    """Run the notebook and check the reported capacitance and its convergence."""
    notebook = nbformat.read(path, as_version=4)

    # Clear saved output so a skipped cell cannot satisfy the result check.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    try:
        NotebookClient(
            notebook, timeout=2400, kernel_name="python3", force_raise_errors=True
        ).execute()
    finally:
        output = "\n".join(
            item.text
            for cell in notebook.cells
            for item in cell.get("outputs", [])
            if item.output_type == "stream"
        )
        logger.info("%s", output)

    if any(
        cell.execution_count is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    ):
        raise RuntimeError("notebook skipped a code cell")

    convergence_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "ax_change.plot" in cell.source
    ]
    if len(convergence_cells) != 1 or not any(
        "image/png" in item.get("data", {}) for item in convergence_cells[0].outputs
    ):
        raise RuntimeError("notebook did not render the convergence plot")

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
    main(Path(sys.argv[1]))
