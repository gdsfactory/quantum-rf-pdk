"""Component that instantiates and displays all available cells in the PDK.

This module provides a component that creates a grid layout containing all cells
available in qpdk.PDK.cells. This is useful for:
- Quick visualization of all available components
- Running cell-level DRC checks on all cells simultaneously
- Documentation and reference purposes
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def all_cells(
    spacing: float = 200.0,
    cells_per_row: int = 5,
) -> Component:
    """Create a component containing all cells from qpdk.PDK.cells.

    Instantiates and arranges all available cells in a grid layout. Cells that
    fail to instantiate are skipped with a warning message.

    Args:
        spacing: Spacing between cells in micrometers (default: 200.0).
        cells_per_row: Number of cells to place per row (default: 5).

    Returns:
        Component containing all successfully instantiated cells arranged in a grid.

    Example:
        >>> import qpdk
        >>> c = qpdk.cells.all_cells()
        >>> c.show()  # Display all cells in KLayout
    """
    from qpdk import PDK

    c = Component("all_cells")

    # Get all cell names, excluding all_cells itself to avoid recursion
    cell_names = sorted([name for name in PDK.cells if name != "all_cells"])

    # Grid layout parameters
    x_pos = 0.0
    y_pos = 0.0
    row_height = 0.0
    cells_in_current_row = 0

    for name in cell_names:
        try:
            # Instantiate the cell
            cell = PDK.cells[name]()

            # Add reference based on cell type
            if isinstance(cell, gf.Component):
                ref = c.add_ref(cell)
            else:  # ComponentAllAngle or other types
                ref = c.add_ref_off_grid(cell)

            # Position the reference
            ref.dmovex(x_pos)
            ref.dmovey(y_pos)

            # Get bounding box for spacing calculations
            bbox = ref.dbbox()
            width = bbox.width()
            height = bbox.height()

            # Update row height to accommodate tallest cell in row
            row_height = max(row_height, height)

            # Move to next position
            cells_in_current_row += 1
            if cells_in_current_row >= cells_per_row:
                # Start new row
                x_pos = 0.0
                y_pos += row_height + spacing
                row_height = 0.0
                cells_in_current_row = 0
            else:
                # Move to next column
                x_pos += width + spacing

        except Exception as e:
            # Log warning but continue with other cells
            import warnings

            warnings.warn(
                f"Failed to instantiate cell '{name}': {e}",
                UserWarning,
                stacklevel=2,
            )

    return c


__all__ = ["all_cells"]
