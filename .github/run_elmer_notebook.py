"""Execute the Elmer notebook and require a fresh capacitance result."""

import logging
import math
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logger = logging.getLogger(__name__)


def main(path: Path) -> None:
    """Run both solves and check the extracted mutual capacitance."""
    notebook = nbformat.read(path, as_version=4)
    solve_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "run_capacitive_simulation_elmer(" in cell.source
    ]
    if len(solve_cells) != 2:
        raise ValueError(f"expected two Elmer solve cells in {path}")

    # Clear saved output so a skipped cell cannot satisfy the result check.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    try:
        NotebookClient(
            notebook, timeout=900, kernel_name="python3", force_raise_errors=True
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

    for marker in ("Maxwell capacitance matrix (fF):", "All checks passed."):
        if marker not in output:
            raise RuntimeError(f"notebook output missing {marker!r}")

    matrix_match = re.search(
        r"Maxwell capacitance matrix \(fF\):\s*\[\[([^\]]+)\]\s*\[([^\]]+)\]\]",
        output,
    )
    if matrix_match is None:
        raise RuntimeError("notebook output missing 2x2 Maxwell matrix")
    matrix = [[float(value) for value in row.split()] for row in matrix_match.groups()]
    if any(len(row) != 2 for row in matrix):
        raise RuntimeError("notebook output has invalid Maxwell matrix shape")
    mutual_fF = -matrix[0][1]
    if not math.isfinite(mutual_fF) or not 5.0 < mutual_fF < 40.0:
        raise RuntimeError(
            f"mutual capacitance outside 5–40 fF reference range: {mutual_fF}"
        )
    logger.info("CI numerical check: mutual capacitance = %.3f fF", mutual_fF)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main(Path(sys.argv[1]))
