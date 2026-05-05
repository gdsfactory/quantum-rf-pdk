"""Component that instantiates and displays all available cells in the PDK.

This module provides a component that creates an efficient layout containing all cells
available in qpdk.PDK.cells. This is useful for:
- Quick visualization of all available components
- Running cell-level DRC checks on all cells simultaneously
- Documentation and reference purposes
"""

from __future__ import annotations

import inspect
import warnings

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def all_cells(
    spacing: float = 200.0,
    **kwargs,
) -> Component:
    """Create a component containing all cells from qpdk.PDK.cells.

    Instantiates and arranges all available cells efficiently. Cells that
    fail to instantiate are skipped with a warning message.

    Args:
        spacing: Spacing between cells in micrometers (default: 200.0).
        **kwargs: Additional arguments passed to gf.pack.

    Returns:
        Component containing all successfully instantiated cells.

    Example:
        >>> import qpdk
        >>> c = qpdk.cells.all_cells()
        >>> c.show()  # Display all cells in KLayout
    """
    from qpdk import PDK

    # Get all cell names, excluding all_cells itself to avoid recursion
    cell_names = sorted([name for name in PDK.cells if name != "all_cells"])

    cells = []

    for name in cell_names:
        cell_func = PDK.cells[name]
        sig = inspect.signature(cell_func)
        required_params = [
            p
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name != "kwargs"
        ]

        if required_params:
            warnings.warn(
                f"Skipping cell '{name}': requires arguments {[p.name for p in required_params]}",
                UserWarning,
                stacklevel=2,
            )
            continue

        try:
            cell = cell_func()
            if cell is None:
                continue

            if not isinstance(cell, gf.Component):
                c_wrap = gf.Component(name=f"{name}_wrap")
                c_wrap.add_ref_off_grid(cell)
                cell = c_wrap

            cells.append(cell)

        except Exception as e:
            warnings.warn(
                f"Failed to instantiate cell '{name}': {e}",
                UserWarning,
                stacklevel=2,
            )

    if not cells:
        return Component("empty_all_cells")

    kwargs.setdefault("max_size", (None, None))
    bins = gf.pack(cells, spacing=spacing, **kwargs)

    if not bins:
        return Component("empty_packed_all_cells")

    if len(bins) > 1:
        warnings.warn(
            f"Packed cells resulted in {len(bins)} bins. Returning only the first one.",
            UserWarning,
            stacklevel=2,
        )

    c = bins[0]
    c.name = "all_cells"
    return c


__all__ = ["all_cells"]

if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = all_cells()
    c.show()
