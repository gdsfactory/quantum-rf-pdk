"""HFSS simulation utilities using PyAEDT.

This module provides helper functions for setting up HFSS simulations
(eigenmode and driven modal) from gdsfactory components. It uses the
PyAEDT library to interface with Ansys HFSS.

The main workflow is:
1. Prepare a component with :func:`prepare_component_for_hfss`
2. Export to GDS and import into HFSS with :func:`import_component_to_hfss`
3. Configure simulation setup (e.g. Eigenmode or Driven) manually via PyAEDT
4. Extract results with :func:`get_eigenmode_results` or :func:`get_sparameter_results`

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
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import numpy as np
from gdsfactory.technology.layer_stack import LayerLevel

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
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack


# Default HFSS simulation parameters
DEFAULT_EIGENMODE_PARAMS = {
    "min_frequency_ghz": 1.0,
    "num_modes": 3,
    "max_passes": 15,
    "min_passes": 2,
    "percent_refinement": 30,
    "max_delta_freq": 2,  # Percentage
}

DEFAULT_DRIVEN_PARAMS = {
    "frequency_ghz": 5.0,
    "max_delta_s": 0.02,
    "max_passes": 10,
    "min_passes": 2,
    "percent_refinement": 30,
}

# Materials that should be treated as perfect conductors (PEC) in HFSS simulations.
# Superconducting materials at cryogenic temperatures are well-approximated by PEC.
SUPERCONDUCTING_MATERIALS = frozenset({"Nb", "Al", "TiN", "Ta", "NbN"})


def _check_pyaedt_available() -> None:
    """Check if PyAEDT is available and raise helpful error if not."""
    try:
        import ansys.aedt.core  # noqa: F401
    except ImportError as e:
        msg = (
            "PyAEDT is required for HFSS simulations. "
            "Install it with: uv sync --extra hfss or pip install qpdk[hfss]"
        )
        raise ImportError(msg) from e


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
    margin: float = 0.0,
) -> Component:
    """Prepare a component for HFSS simulation export.

    You should run this before exporting a component to HFSS, as it applies
    the transformations you likely want.

    This function prepares the component by doing the following:
    1. Applying additive metal operations
    2. Inverting mask polarity to positive metals
    3. Remove metadata-like layers
    4. Optionally add a margin to the simulation bounding box by extending M1_DRAW and M2_DRAW

    Args:
        component: The gdsfactory component to prepare.
        margin: The margin to add to the bounding box in µm.

    Returns:
        The prepared component (may be modified in-place).

    Example:
        >>> from qpdk.cells import resonator
        >>> comp = resonator(length=4000)
        >>> prepared = prepare_component_for_hfss(comp)
    """
    c = gf.Component(name=f"{component.name}_hfss")
    c << component.copy()
    c = apply_additive_metals(c)
    c = invert_mask_polarity(c)
    if margin > 0.0:
        c = add_margin_to_layer(
            c,
            layer_margins=[
                (LAYER.M1_DRAW, margin),
                (LAYER.M2_DRAW, margin),
            ],
        )
    return remove_metadata_layers(c)


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

    # Create reverse mapping from layer number to layer name for renaming
    num_to_name = {}
    for name, level in layer_stack.layers.items():
        layer_num = _get_layer_number_from_level(level)
        if layer_num is not None and layer_num not in num_to_name:
            num_to_name[layer_num] = name

    # Export component to GDS
    # Note: We use TemporaryDirectory to ensure cleanup, but need to keep it
    # alive until import is complete, so we store the reference
    temp_dir_obj = None
    if gds_path is None:
        # Use temporary directory that will be cleaned up when function returns
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="qpdk_hfss_")
        gds_path = Path(temp_dir_obj.name) / "component.gds"

    gds_path = Path(gds_path)
    component.write_gds(str(gds_path))

    # Set modeler units
    hfss.modeler.model_units = units

    # Record existing objects
    existing_objects = set(hfss.modeler.object_names)

    # Import GDS with 3D layer mapping
    result = hfss.import_gds_3d(
        input_file=str(gds_path),
        mapping_layers=mapping_layers,
        units=units,
        import_method=0,
    )

    if result:
        # Set all newly imported objects to PEC
        new_objects = list(set(hfss.modeler.object_names) - existing_objects)

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
                        hfss.modeler[obj_name].name = new_name
                    except Exception:
                        new_name = obj_name  # Fallback if rename fails
            renamed_objects.append(new_name)

        if renamed_objects:
            if import_as_sheets:
                hfss.assign_perfecte_to_sheets(renamed_objects, name="PEC_Sheets")
            else:
                hfss.assign_perfect_e(renamed_objects, name="PEC_3D")

    # Clean up temporary directory if we created one
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return result


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
    bounds = component.bbox()
    x_min, y_min = bounds.p1.x, bounds.p1.y
    x_max, y_max = bounds.p2.x, bounds.p2.y

    substrate = hfss.modeler.create_box(
        origin=[x_min, y_min, -thickness],
        sizes=[x_max - x_min, y_max - y_min, thickness],
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
) -> str:
    """Add an air region (vacuum box) around the component for eigenmode analysis.

    Creates a vacuum region surrounding the component and assigns PerfectE (PEC)
    boundary conditions to all outer faces. This is appropriate for closed-box
    eigenmode simulations where the structure is shielded.

    Args:
        hfss: The HFSS application instance.
        component: The component to create air region around.
        height: Height above the component in micrometers.
        substrate_thickness: Depth below surface for the region.

    Returns:
        Name of the created region object.
    """
    bounds = component.bbox()
    x_min, y_min = bounds.p1.x, bounds.p1.y
    x_max, y_max = bounds.p2.x, bounds.p2.y

    region = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[x_max - x_min, y_max - y_min, height + substrate_thickness],
        name="AirRegion",
        material="vacuum",
    )

    # Ensure vacuum has lowest priority if it overlaps with substrate
    region.mesh_order = 99

    # Assign PerfectE (PEC) boundary for closed-box eigenmode analysis
    hfss.assign_perfect_e(
        assignment=[face.id for face in region.faces],
        name="PEC_Boundary",
    )

    return region.name


def add_lumped_port(
    hfss: Hfss,
    port_face_id: int,
    port_name: str = "Port1",
    *,
    impedance: float = 50.0,
) -> object:
    """Add a lumped port to a face in the HFSS model.

    Args:
        hfss: The HFSS application instance.
        port_face_id: The face ID to assign the port to.
        port_name: Name for the port.
        impedance: Port impedance in ohms.

    Returns:
        The created port object.
    """
    return hfss.lumped_port(
        assignment=port_face_id,
        name=port_name,
        impedance=impedance,
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
) -> dict:
    """Extract S-parameter results from a driven simulation.

    Args:
        hfss: The HFSS application instance.
        setup_name: Name of the setup.
        sweep_name: Name of the frequency sweep.

    Returns:
        Dictionary containing:
        - frequencies: Array of frequencies in GHz
        - s_parameters: Dictionary of S-parameter arrays (e.g., "S11", "S21")
    """
    traces = hfss.get_traces_for_plot()

    results = {"frequencies": None, "s_parameters": {}}

    for trace in traces:
        solution = hfss.post.get_solution_data(
            expressions=trace,
            setup_sweep_name=f"{setup_name} : {sweep_name}",
        )
        if solution:
            if results["frequencies"] is None:
                results["frequencies"] = np.array(solution.primary_sweep_values) / 1e9

            # Get complex S-parameter data
            # data_real and data_imag give the real and imaginary parts
            real_data = np.array(solution.data_real())
            imag_data = np.array(solution.data_imag())
            complex_data = real_data + 1j * imag_data

            # Extract magnitude in dB and phase in degrees
            results["s_parameters"][trace] = {
                "magnitude_db": 20 * np.log10(np.abs(complex_data)),
                "phase_deg": np.degrees(np.angle(complex_data)),
            }

    return results


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
