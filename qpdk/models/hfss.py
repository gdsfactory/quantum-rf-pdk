"""HFSS and Q3D simulation utilities using PyAEDT.

This module provides helper functions for setting up HFSS simulations
(eigenmode and driven modal) and Q3D Extractor parasitic extractions
from gdsfactory components. It uses the PyAEDT library to interface
with Ansys HFSS and Q3D Extractor.

**HFSS workflow:**

1. Prepare a component with :func:`prepare_component_for_hfss`
2. Export to GDS and import into HFSS with :func:`import_component_to_hfss`
3. Configure simulation setup (e.g. Eigenmode or Driven) manually via PyAEDT
4. Extract results with :func:`get_eigenmode_results` or :func:`get_sparameter_results`

**Q3D Extractor workflow:**

1. Prepare a component with :func:`prepare_component_for_hfss`
2. Export to GDS and import into Q3D with :func:`import_component_to_q3d`
3. Assign signal nets with :func:`assign_q3d_nets_from_ports`
4. Configure Q3D setup and analyze
5. Extract capacitance matrix with :func:`get_q3d_capacitance_matrix`

Note:
    This module requires the optional ``hfss`` dependency group.
    Install with: ``uv sync --extra hfss`` or ``pip install qpdk[hfss]``

Example:
    >>> from ansys.aedt.core import Hfss
    >>> from qpdk.models.hfss import import_component_to_hfss
    >>> from qpdk.cells import resonator
    >>> comp = resonator(length=4000, meanders=4)
    >>> hfss = Hfss(project="resonator_sim", solution_type="Eigenmode")
    >>> import_component_to_hfss(hfss, comp)

References:
    - PyAEDT documentation: https://aedt.docs.pyansys.com/
    - HFSS import_gds_3d: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.import_gds_3d.html
    - Q3D Extractor: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.q3d.Q3d.html
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import gdsfactory as gf
import numpy as np
import polars as pl
from gdsfactory.technology.layer_stack import LayerLevel
from gdsfactory.typings import Ports

from qpdk import LAYER_STACK
from qpdk.cells.helpers import (
    add_margin_to_layer,
    apply_additive_metals,
    invert_mask_polarity,
    remove_metadata_layers,
)
from qpdk.tech import LAYER

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.q3d import Q3d
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack


def layer_stack_to_gds_mapping(
    layer_stack: LayerStack | None = None,
    thickness_override: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Convert a LayerStack to HFSS GDS import mapping dictionary.

    Creates a mapping from GDS layer numbers to (elevation, thickness) tuples
    for use with :meth:`ansys.aedt.core.hfss.Hfss.import_gds_3d`.

    Args:
        layer_stack: LayerStack defining layers with elevation and thickness.
            If None, uses QPDK's default LAYER_STACK.
        thickness_override: Optional thickness to use for all layers.
            If None, uses the thickness from the LayerStack.

    Returns:
        Dictionary mapping GDS layer numbers to (elevation, thickness) tuples
        in micrometers.

    Example:
        >>> from qpdk.models.hfss import layer_stack_to_gds_mapping
        >>> mapping = layer_stack_to_gds_mapping()
        >>> # Returns e.g. {1: (0.0, 0.0002), 10: (0.0003, 0.0002), ...}
    """
    if layer_stack is None:
        layer_stack = LAYER_STACK

    mapping: dict[int, tuple[float, float]] = {}

    for layer_level in layer_stack.layers.values():
        # Get the layer number from the layer definition
        layer_number = _get_layer_number_from_level(layer_level)
        if layer_number is None:
            continue

        # Get elevation (zmin) and thickness
        elevation = layer_level.zmin if layer_level.zmin is not None else 0.0
        if thickness_override is not None:
            thickness = thickness_override
        else:
            thickness = layer_level.thickness if layer_level.thickness else 0.0

        # Store mapping: layer_number -> (elevation, thickness)
        mapping[layer_number] = (elevation, thickness)

    return mapping


def _get_layer_number_from_level(layer_level: LayerLevel) -> int | None:
    """Extract layer number from a LayerLevel's layer definition.

    Handles various layer definition types:
    - Direct tuple: (layer_number, datatype)
    - LogicalLayer: wraps a LayerMap enum or tuple
    - DerivedLayer: uses derived_layer attribute if available

    Args:
        layer_level: A gdsfactory LayerLevel object.

    Returns:
        Layer number (int) or None if not extractable.
    """
    # First check for derived_layer (for DerivedLayer types)
    if hasattr(layer_level, "derived_layer") and layer_level.derived_layer is not None:
        derived = layer_level.derived_layer
        # derived_layer is typically a LogicalLayer
        if hasattr(derived, "layer"):
            inner = derived.layer
            if hasattr(inner, "layer"):
                val = inner.layer
                if isinstance(val, tuple) and len(val) >= 1:
                    return int(val[0])
                return int(val)
            if isinstance(inner, tuple) and len(inner) >= 1:
                return int(inner[0])

    layer = layer_level.layer
    # Direct tuple
    if isinstance(layer, tuple) and len(layer) >= 1:
        return int(layer[0])
    # LogicalLayer with .layer attribute that is a LayerMap enum
    if hasattr(layer, "layer"):
        inner = layer.layer
        # Inner is a tuple
        if isinstance(inner, tuple) and len(inner) >= 1:
            return int(inner[0])
        # Inner is a LayerMap enum with .layer attribute
        if hasattr(inner, "layer"):
            val = inner.layer
            if isinstance(val, tuple) and len(val) >= 1:
                return int(val[0])
            return int(val)
        return int(inner)
    return None


def prepare_component_for_hfss(
    component: Component,
    margin_draw: float = 0.0,
    margin_etch: float = 0.0,
) -> Component:
    """Prepare a component for HFSS simulation export.

    You should run this before exporting a component to HFSS, as it applies
    the transformations you likely want.

    This function prepares the component by doing the following:
    1. Optionally add a margin to the etch layers (e.g. M1_ETCH, M2_ETCH)
    2. Applying additive metal operations
    3. Inverting mask polarity to positive metals
    4. Remove etch layers (e.g. M1_ETCH, M2_ETCH) that are not needed for HFSS geometry
    5. Remove metadata-like layers
    6. Optionally add a margin to the simulation bounding box by extending M1_DRAW and M2_DRAW

    Args:
        component: The gdsfactory component to prepare.
        margin_draw: The margin to add to the draw layers bounding box in µm.
        margin_etch: The margin to add to the etch layers bounding box in µm.

    Returns:
        The prepared component (may be modified in-place).

    Example:
        >>> from qpdk.cells import resonator
        >>> comp = resonator(length=4000)
        >>> prepared = prepare_component_for_hfss(comp, margin_draw=100)
    """
    c = gf.Component(name=f"{component.name}_hfss")
    c << component.copy()
    if margin_etch > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_ETCH, margin_etch),
                (LAYER.M2_ETCH, margin_etch),
            ],
        )
    c = apply_additive_metals(c)
    c = invert_mask_polarity(c)
    if margin_draw > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_DRAW, margin_draw),
                (LAYER.M2_DRAW, margin_draw),
            ],
        )
    c = c.remove_layers(layer for layer in LAYER if str(layer).endswith("_ETCH"))
    c = remove_metadata_layers(c)
    c.add_ports(component.ports)
    return c


@contextlib.contextmanager
def _export_component_to_gds_temp(
    component: Component,
    gds_path: str | Path | None = None,
    prefix: str = "qpdk_aedt_",
):
    """Context manager for exporting a component to a GDS file."""
    if gds_path is not None:
        path = Path(gds_path)
        component.write_gds(str(path))
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            path = Path(temp_dir) / "component.gds"
            component.write_gds(str(path))
            yield path


def _rename_imported_objects(
    app: Any, new_objects: list[str], layer_stack: LayerStack
) -> list[str]:
    """Rename imported GDS objects based on the layer stack."""
    num_to_name = {}
    for name, level in layer_stack.layers.items():
        layer_num = _get_layer_number_from_level(level)
        if layer_num is not None and layer_num not in num_to_name:
            num_to_name[layer_num] = name

    renamed_objects = []
    for obj_name in new_objects:
        match = re.match(r"^signal(\d+)(_.*)?$", obj_name)
        new_name = obj_name
        if match:
            layer_num = int(match.group(1))
            suffix = match.group(2) or ""
            if layer_num in num_to_name:
                layer_name = num_to_name[layer_num]
                new_name = f"{layer_name}{suffix}"
                try:
                    app.modeler[obj_name].name = new_name
                except Exception:
                    new_name = obj_name  # Fallback if rename fails
        renamed_objects.append(new_name)

    return renamed_objects


def import_component_to_hfss(
    hfss: Hfss,
    component: Component,
    layer_stack: LayerStack | None = None,
    *,
    import_as_sheets: bool = False,
    units: str = "um",
    gds_path: str | Path | None = None,
) -> bool:
    """Import a gdsfactory component into HFSS.

    Args:
        hfss: The HFSS application instance.
        component: The gdsfactory component to import.
        layer_stack: LayerStack defining thickness and elevation for each layer.
            If None, uses QPDK's default LAYER_STACK.
        import_as_sheets: If True, imports metals as 2D sheets (zero thickness)
            and assigns PerfectE boundary to them. If False, imports as 3D
            objects with thickness from layer_stack and assigns PerfectE
            boundary to their surfaces.
        units: Length units for the geometry (default: "um" for micrometers).
        gds_path: Optional path to write the GDS file. If None, uses a temporary file.

    Returns:
        True if import was successful, False otherwise.
    """
    if layer_stack is None:
        layer_stack = LAYER_STACK

    # Generate layer mapping from LayerStack
    thickness_override = 0.0 if import_as_sheets else None
    mapping_layers = layer_stack_to_gds_mapping(
        layer_stack, thickness_override=thickness_override
    )

    with _export_component_to_gds_temp(
        component, gds_path, prefix="qpdk_hfss_"
    ) as path:
        # Set modeler units
        hfss.modeler.model_units = units

        # Record existing objects
        existing_objects = set(hfss.modeler.object_names)

        # Import GDS with 3D layer mapping
        result = hfss.import_gds_3d(
            input_file=str(path),
            mapping_layers=mapping_layers,
            units=units,
            import_method=0,
        )

        if result:
            # Set all newly imported objects to PEC
            new_objects = list(set(hfss.modeler.object_names) - existing_objects)
            renamed_objects = _rename_imported_objects(hfss, new_objects, layer_stack)

            if renamed_objects:
                if import_as_sheets:
                    hfss.assign_perfecte_to_sheets(renamed_objects, name="PEC_Sheets")
                else:
                    hfss.assign_perfect_e(renamed_objects, name="PEC_3D")

    return result


def _get_bbox_dimensions(component: Component) -> tuple[float, float, float, float]:
    """Get the x_min, y_min, x_size, and y_size from a component's bounding box."""
    bounds = component.bbox()
    x_min, y_min = bounds.p1.x, bounds.p1.y
    x_max, y_max = bounds.p2.x, bounds.p2.y
    return x_min, y_min, x_max - x_min, y_max - y_min


def add_substrate_to_hfss(
    hfss: Hfss,
    component: Component,
    *,
    thickness: float = 500.0,
    material: str = "silicon",
) -> str:
    """Add a substrate box below the component geometry.

    Args:
        hfss: The HFSS application instance.
        component: The component to create substrate for (used for dimensions).
        thickness: Substrate thickness in micrometers.
        material: Substrate material name.

    Returns:
        Name of the created substrate object.
    """
    x_min, y_min, dx, dy = _get_bbox_dimensions(component)

    substrate = hfss.modeler.create_box(
        origin=[x_min, y_min, -thickness],
        sizes=[dx, dy, thickness],
        name="Substrate",
        material=material,
    )
    # Ensure substrate has higher priority than vacuum
    substrate.mesh_order = 4
    return substrate.name


def add_air_region_to_hfss(
    hfss: Hfss,
    component: Component,
    *,
    height: float = 500.0,
    substrate_thickness: float = 500.0,
    pec_boundary: bool = False,
) -> str:
    """Add an air region (vacuum box) around the component for eigenmode analysis.

    Creates a vacuum region surrounding the component. Optionally assigns PerfectE (PEC)
    boundary conditions to all outer faces, which is appropriate for closed-box
    eigenmode simulations where the structure is shielded.

    Args:
        hfss: The HFSS application instance.
        component: The component to create air region around.
        height: Height above the component in micrometers.
        substrate_thickness: Depth below surface for the region.
        pec_boundary: If True, assign PerfectE boundary conditions to outer faces.

    Returns:
        Name of the created region object.
    """
    x_min, y_min, dx, dy = _get_bbox_dimensions(component)

    region = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[dx, dy, height + substrate_thickness],
        name="AirRegion",
        material="vacuum",
    )

    # Ensure vacuum has lowest priority if it overlaps with substrate
    region.mesh_order = 99

    if pec_boundary:
        # Assign PerfectE (PEC) boundary for closed-box eigenmode analysis
        hfss.assign_perfect_e(
            assignment=[face.id for face in region.faces],
            name="PEC_Boundary",
        )

    return region.name


class LumpedPortConfig(TypedDict):
    """Configuration for defining a lumped port rectangle in HFSS."""

    origin: list[float]
    sizes: list[float]
    integration_line: list[list[float]]


def lumped_port_rectangle_from_cpw(
    center: tuple[float, float, float],
    orientation: float,
    cpw_gap: float,
    cpw_width: float,
) -> LumpedPortConfig:
    """Calculates parameters for a lumped port based on its orientation.

    Args:
        center: [x, y, z] coordinates of the port face center.
        orientation: Angle in degrees (must be a multiple of 90).
        cpw_gap: The length of the port along the axis of propagation.
        cpw_width: The width of the port perpendicular to propagation.

    Returns:
        A dictionary containing 'origin', 'sizes', and 'integration_line' for HFSS.
    """
    if orientation % 90 != 0:
        raise ValueError(f"Unsupported port orientation: {orientation}°")

    cx, cy = center[0], center[1]

    # Calculate directional unit vectors
    theta = np.deg2rad(orientation)
    c = np.round(np.cos(theta))
    s = np.round(np.sin(theta))

    # Calculate bounding box sizes
    size_x = cpw_gap * np.abs(c) + cpw_width * np.abs(s)
    size_y = cpw_width * np.abs(c) + cpw_gap * np.abs(s)

    # Calculate the literal center point of the port rectangle
    rect_cx = cx + (cpw_gap / 2) * c
    rect_cy = cy + (cpw_gap / 2) * s

    # The HFSS origin is the bottom-left corner of the bounding box
    origin = [rect_cx - size_x / 2, rect_cy - size_y / 2, 0]

    # The integration line goes from the outer edge to the original center
    int_line = [[cx + cpw_gap * c, cy + cpw_gap * s, 0], [cx, cy, 0]]

    return {"origin": origin, "sizes": [size_x, size_y], "integration_line": int_line}


def add_lumped_ports_to_hfss(
    hfss: Hfss, ports: Ports, cpw_gap: float, cpw_width: float
) -> None:
    """Add lumped ports to HFSS at given port locations.

    Args:
        hfss: The HFSS application instance.
        ports: Collection of gdsfactory ports.
        cpw_gap: The length of the port along the axis of propagation.
        cpw_width: The width of the port perpendicular to propagation.
    """
    for port in ports:
        port_rectangle_params = lumped_port_rectangle_from_cpw(
            port.center, port.orientation, cpw_gap, cpw_width
        )

        port_rect = hfss.modeler.create_rectangle(
            orientation="XY", name=f"{port.name}_face", **port_rectangle_params
        )

        hfss.lumped_port(
            assignment=port_rect.name,
            name=port.name,
            create_port_sheet=False,  # Already done
            integration_line=port_rectangle_params["integration_line"],
        )


def get_eigenmode_results(hfss: Hfss, setup_name: str = "EigenmodeSetup") -> dict:
    """Extract eigenmode simulation results.

    Args:
        hfss: The HFSS application instance.
        setup_name: Name of the setup to get results from.

    Returns:
        Dictionary containing:
        - frequencies: List of eigenmode frequencies in GHz
        - q_factors: List of Q factors for each mode
    """
    # Get frequency values
    freq_names = hfss.post.available_report_quantities(
        quantities_category="Eigen Modes"
    )
    q_names = hfss.post.available_report_quantities(quantities_category="Eigen Q")

    results = {"frequencies": [], "q_factors": [], "setup": setup_name}

    for f_name in freq_names:
        solution = hfss.post.get_solution_data(
            expressions=f_name, report_category="Eigenmode"
        )
        if solution:
            freq_hz = float(solution.data_real()[0])
            results["frequencies"].append(freq_hz / 1e9)  # Convert to GHz

    for q_name in q_names:
        solution = hfss.post.get_solution_data(
            expressions=q_name, report_category="Eigenmode"
        )
        if solution:
            q = float(solution.data_real()[0])
            results["q_factors"].append(q)

    return results


def get_sparameter_results(
    hfss: Hfss,
    setup_name: str = "DrivenSetup",
    sweep_name: str = "FrequencySweep",
) -> pl.DataFrame:
    """Extract S-parameter results from a driven simulation.

    Args:
        hfss: The HFSS application instance.
        setup_name: Name of the setup.
        sweep_name: Name of the frequency sweep.

    Returns:
        DataFrame containing a 'frequency_ghz' column and a column
        for each S-parameter trace (e.g., "S(1,1)") containing complex values.
    """
    traces = hfss.get_traces_for_plot()

    data = {}

    for trace in traces:
        solution = hfss.post.get_solution_data(
            expressions=trace,
            setup_sweep_name=f"{setup_name} : {sweep_name}",
        )
        if solution:
            if "frequency_ghz" not in data:
                data["frequency_ghz"] = np.array(solution.primary_sweep_values)

            # Get complex S-parameter data
            # Use get_expression_data to get real and imaginary parts
            _, real_data = solution.get_expression_data(formula="real")
            _, imag_data = solution.get_expression_data(formula="imag")
            complex_data = real_data + 1j * imag_data

            data[trace] = complex_data

    return pl.DataFrame(data)


def import_component_to_q3d(
    q3d: Q3d,
    component: Component,
    layer_stack: LayerStack | None = None,
    *,
    units: str = "um",
    gds_path: str | Path | None = None,
) -> list[str]:
    """Import a gdsfactory component into Q3D Extractor.

    Imports the component's GDS geometry into a Q3D Extractor project,
    mapping each GDS layer to a 3D conductor at the appropriate elevation
    and thickness from the layer stack. Imported objects are renamed based
    on the layer stack for clarity.

    This is the Q3D equivalent of :func:`import_component_to_hfss`.

    Args:
        q3d: The Q3D Extractor application instance.
        component: The gdsfactory component to import.
        layer_stack: LayerStack defining thickness and elevation for each layer.
            If None, uses QPDK's default LAYER_STACK.
        units: Length units for the geometry (default: "um" for micrometers).
        gds_path: Optional path to write the GDS file. If None, uses a
            temporary file.

    Returns:
        List of newly created conductor object names in Q3D.

    Example:
        >>> from ansys.aedt.core import Q3d
        >>> from qpdk.models.hfss import import_component_to_q3d
        >>> from qpdk.cells.capacitor import interdigital_capacitor
        >>> comp = interdigital_capacitor(fingers=6, finger_length=20)
        >>> q3d = Q3d(project="cap_q3d", solution_type="Q3DExtractor")
        >>> objects = import_component_to_q3d(q3d, comp)
    """
    if layer_stack is None:
        layer_stack = LAYER_STACK

    # Generate layer mapping from LayerStack (use real thickness for Q3D)
    mapping_layers = layer_stack_to_gds_mapping(layer_stack)

    with _export_component_to_gds_temp(component, gds_path, prefix="qpdk_q3d_") as path:
        # Set modeler units
        q3d.modeler.model_units = units

        # Record existing objects
        existing_objects = set(q3d.modeler.object_names)

        # Import GDS with 3D layer mapping
        q3d.import_gds_3d(
            input_file=str(path),
            mapping_layers=mapping_layers,
            units=units,
            import_method=0,
        )

        # Track and rename new objects
        new_objects = list(set(q3d.modeler.object_names) - existing_objects)
        renamed_objects = _rename_imported_objects(q3d, new_objects, layer_stack)

        # Assign material so Q3D identifies them as conductors
        if renamed_objects:
            q3d.assign_material(renamed_objects, "pec")

        return renamed_objects


def assign_q3d_nets_from_ports(
    q3d: Q3d,
    ports: Ports,
    conductor_objects: list[str],
) -> list[str]:
    """Assign Q3D signal nets based on gdsfactory port locations.

    For each gdsfactory port, finds the conductor object whose bounding-box
    center is nearest to the port center and assigns it as a Q3D signal net.
    Connected conductor objects sharing the same net are automatically
    included via Q3D's auto net identification.

    This is the Q3D equivalent of :func:`add_lumped_ports_to_hfss` for driven
    HFSS simulations.

    After calling this function, Q3D will include one signal net per port
    in the computed capacitance (and inductance) matrices.

    Args:
        q3d: The Q3D Extractor application instance.
        ports: Collection of gdsfactory ports defining signal locations.
        conductor_objects: List of conductor object names created by
            :func:`import_component_to_q3d`.

    Returns:
        List of assigned signal net names (one per port).

    Example:
        >>> from qpdk.models.hfss import assign_q3d_nets_from_ports
        >>> signal_nets = assign_q3d_nets_from_ports(
        ...     q3d, component.ports, conductor_objects
        ... )
    """
    # Auto-identify nets to group geometrically connected conductors
    q3d.auto_identify_nets()

    assigned_nets: list[str] = []
    used_objects: set[str] = set()

    # Find scale factor by trying powers of 10 (PyAEDT bounding_box might be in different units)
    scale_factor = 1.0
    if ports:
        px0, py0 = float(ports[0].center[0]), float(ports[0].center[1])
        for power in [-3, -2, -1, 0, 1, 2, 3, 4]:
            test_scale = 10**power
            found_contain = False
            for obj_name in conductor_objects:
                obj = q3d.modeler.get_object_from_name(obj_name)
                if obj:
                    bbox = obj.bounding_box
                    xmin, ymin = bbox[0] * test_scale, bbox[1] * test_scale
                    xmax, ymax = bbox[3] * test_scale, bbox[4] * test_scale
                    if (xmin - 1.0 <= px0 <= xmax + 1.0) and (
                        ymin - 1.0 <= py0 <= ymax + 1.0
                    ):
                        found_contain = True
                        break
            if found_contain:
                scale_factor = test_scale
                break

    print(f"Computed scale factor from Q3D units to Port units: {scale_factor}")

    for port in ports:
        px, py = float(port.center[0]), float(port.center[1])

        # Find the conductor object closest to this port
        best_obj: str | None = None
        best_metric = float("inf")

        print(f"\n--- Assigning port {port.name} at ({px}, {py}) ---")

        for obj_name in conductor_objects:
            if obj_name in used_objects:
                continue

            # Get object from modeler; skip if not found
            obj = q3d.modeler.get_object_from_name(obj_name)
            if obj is None:
                continue

            # bounding_box returns [xmin, ymin, zmin, xmax, ymax, zmax]
            bbox = obj.bounding_box
            xmin = bbox[0] * scale_factor
            ymin = bbox[1] * scale_factor
            xmax = bbox[3] * scale_factor
            ymax = bbox[4] * scale_factor

            area = (xmax - xmin) * (ymax - ymin)

            # Check if port is inside or very close to the bounding box
            tol = 1.0
            if (xmin - tol <= px <= xmax + tol) and (ymin - tol <= py <= ymax + tol):
                # Prefer the smallest area that contains the port (e.g. trace vs ground plane)
                metric = area
                print(
                    f"Object {obj_name} CONTAINS port. bbox=({xmin:.1f}, {ymin:.1f}) to ({xmax:.1f}, {ymax:.1f}). area={area:.1f}. metric={metric}"
                )
            else:
                # Fallback to distance if port is outside
                obj_cx = (xmin + xmax) / 2
                obj_cy = (ymin + ymax) / 2
                dist = ((px - obj_cx) ** 2 + (py - obj_cy) ** 2) ** 0.5
                metric = 1e9 + dist
                print(
                    f"Object {obj_name} DOES NOT contain port. bbox={bbox}, dist={dist:.1f}. metric={metric}"
                )

            if metric < best_metric:
                best_metric = metric
                best_obj = obj_name

        print(f"Selected best object for {port.name}: {best_obj}")

        if best_obj is None:
            continue

        # Find the auto-identified net that contains this object
        net_to_rename = None
        for b in q3d.boundaries:
            if b.type == "SignalNet" and best_obj in b.props.get("Objects", []):
                net_to_rename = b
                break

        if net_to_rename is not None:
            # Rename the existing net instead of creating an overlapping one
            net_to_rename.name = port.name
            assigned_nets.append(port.name)
            used_objects.add(best_obj)
        else:
            # Fallback if no net was auto-identified for some reason
            q3d.assign_net(
                assignment=[best_obj],
                net_name=port.name,
                net_type="Signal",
            )
            assigned_nets.append(port.name)
            used_objects.add(best_obj)

    return assigned_nets


def get_q3d_capacitance_matrix(
    q3d: Q3d,
    setup_name: str = "Q3DSetup",
) -> pl.DataFrame:
    """Extract the capacitance matrix from a Q3D Extractor simulation.

    Retrieves all capacitance matrix entries (e.g. ``C(o1,o1)``, ``C(o1,o2)``)
    from the solved Q3D setup.

    Args:
        q3d: The Q3D Extractor application instance.
        setup_name: Name of the analysis setup.

    Returns:
        DataFrame with one column per capacitance expression containing
        the extracted values in Farads.

    Example:
        >>> cap_df = get_q3d_capacitance_matrix(q3d, "Q3DSetup")
        >>> print(cap_df)
    """
    nets = [b.name for b in q3d.boundaries if b.type == "SignalNet"]
    expressions = [f"C({n1},{n2})" for n1 in nets for n2 in nets]

    data: dict[str, list[float]] = {}

    for expr in expressions:
        solution = q3d.post.get_solution_data(
            expressions=expr,
            setup_sweep_name=f"{setup_name} : LastAdaptive",
        )
        if solution:
            # Q3D usually returns capacitance in pF by default in reports
            val_pf = float(solution.data_real()[0])
            data[expr] = [val_pf * 1e-12]  # convert to Farads

    return pl.DataFrame(data)


if __name__ == "__main__":
    from qpdk import PDK
    from qpdk.cells.resonator import resonator

    PDK.activate()

    # Generate a resonator component
    comp = resonator(length=4000)

    # Run the preparation step that is typically done before import
    prepared_comp = prepare_component_for_hfss(comp)

    # Print the mapping layers
    mapping = layer_stack_to_gds_mapping()
    print("Mapping layers:")
    comp_layers = {layer[0] for layer in prepared_comp.layers}
    for layer_num, (elev, thick) in mapping.items():
        if layer_num in comp_layers:
            print(f"Layer {layer_num}: elevation={elev}, thickness={thick}")

    # Show the component
    prepared_comp.show()
