"""HFSS simulation utilities using PyAEDT.

This module provides helper functions for setting up HFSS simulations
(eigenmode and driven modal) from gdsfactory components. It uses the
PyAEDT library to interface with Ansys HFSS.

The main workflow is:
1. Prepare a component with :func:`prepare_component_for_hfss`
2. Export to GDS and import into HFSS with :func:`import_component_to_hfss`
3. Configure simulation setup with :func:`setup_eigenmode_simulation` or
   :func:`setup_driven_simulation`
4. Extract results with :func:`get_eigenmode_results` or :func:`get_sparameter_results`

Note:
    This module requires the optional ``hfss`` dependency group.
    Install with: ``uv sync --extra hfss`` or ``pip install qpdk[hfss]``

Example:
    >>> from qpdk.models.hfss import import_component_to_hfss, create_hfss_project
    >>> from qpdk.cells import resonator
    >>> comp = resonator(length=4000, meanders=4)
    >>> hfss = create_hfss_project("resonator_sim", solution_type="Eigenmode")
    >>> import_component_to_hfss(hfss, comp)

References:
    - PyAEDT documentation: https://aedt.docs.pyansys.com/
    - HFSS import_gds_3d: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.import_gds_3d.html
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
SUPERCONDUCTING_MATERIALS = frozenset({"Nb", "Al", "TiN", "Ta", "NbN", "copper"})


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
) -> dict[int, tuple[float, float]]:
    """Convert a LayerStack to HFSS GDS import mapping dictionary.

    Creates a mapping from GDS layer numbers to (elevation, thickness) tuples
    for use with :meth:`ansys.aedt.core.hfss.Hfss.import_gds_3d`.

    Args:
        layer_stack: LayerStack defining layers with elevation and thickness.
            If None, uses QPDK's default LAYER_STACK.

    Returns:
        Dictionary mapping GDS layer numbers to (elevation, thickness) tuples
        in micrometers.

    Example:
        >>> from qpdk.models.hfss import layer_stack_to_gds_mapping
        >>> mapping = layer_stack_to_gds_mapping()
        >>> # Returns e.g. {1: (0.0, 0.0002), 10: (0.0003, 0.0002), ...}
    """
    from qpdk import LAYER_STACK

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
        thickness = layer_level.thickness if layer_level.thickness else 0.0

        # Store mapping: layer_number -> (elevation, thickness)
        mapping[layer_number] = (elevation, thickness)

    return mapping


def _get_layer_number_from_level(layer_level) -> int | None:
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
    *,
    apply_additive: bool = True,
) -> Component:
    """Prepare a component for HFSS simulation export.

    This function prepares the component by optionally applying additive metal
    operations to create the proper negative mask representation for simulation.

    Args:
        component: The gdsfactory component to prepare.
        apply_additive: If True, applies :func:`~qpdk.cells.helpers.apply_additive_metals`
            to properly handle additive vs subtractive mask operations.

    Returns:
        The prepared component (may be modified in-place).

    Example:
        >>> from qpdk.cells import resonator
        >>> comp = resonator(length=4000)
        >>> prepared = prepare_component_for_hfss(comp)
    """
    import gdsfactory as gf

    c = gf.Component(name=f"{component.name}_hfss")
    c << component
    component = c

    if apply_additive:
        from qpdk.cells.helpers import apply_additive_metals

        component = apply_additive_metals(component)

    return component


def draw_component_in_hfss(
    hfss: Hfss,
    component: Component,
    layer_stack: LayerStack | None = None,
    *,
    apply_additive: bool = True,
) -> bool:
    """Draw a gdsfactory component in HFSS by iterating over polygons.

    This is an alternative to :func:`import_component_to_hfss` that avoids
    native GDS import by manually drawing each polygon as a polyline and
    thickening it. This approach is often more robust in non-graphical
    environments or on Linux.

    Args:
        hfss: The HFSS application instance.
        component: The gdsfactory component to draw.
        layer_stack: LayerStack defining thickness and elevation for each layer.
            If None, uses QPDK's default LAYER_STACK.
        apply_additive: If True, applies additive metal operations before drawing.

    Returns:
        True if drawing was successful, False otherwise.
    """
    from qpdk import LAYER_STACK

    if layer_stack is None:
        layer_stack = LAYER_STACK

    # Prepare component
    prepared_component = prepare_component_for_hfss(
        component, apply_additive=apply_additive
    )

    # Set modeler units to match gdsfactory (micrometers)
    hfss.modeler.model_units = "um"

    success = True
    for layer_name, layer_level in layer_stack.layers.items():
        # Get layer number
        layer_number = _get_layer_number_from_level(layer_level)
        if layer_number is None:
            continue

        # Get elevation and thickness
        elevation = layer_level.zmin if layer_level.zmin is not None else 0.0
        thickness = layer_level.thickness if layer_level.thickness else 0.0

        if thickness == 0:
            continue

        # Get polygons for this layer
        # Note: we assume (layer_number, 0) as is common in GDS
        polys_dict = prepared_component.get_polygons(
            layers=[(layer_number, 0)], by="tuple"
        )
        if (layer_number, 0) not in polys_dict:
            continue

        polys = polys_dict[(layer_number, 0)]
        material = layer_level.material or "vacuum"
        # Superconducting materials are PEC
        if material in SUPERCONDUCTING_MATERIALS:
            material = "pec"

        for i, poly in enumerate(polys):
            # Convert kfactory polygon to micrometer points
            dpoly = poly.to_dtype(prepared_component.kcl.dbu)

            # Handle hull
            points = [[float(pt.x), float(pt.y), elevation] for pt in dpoly.each_point_hull()]
            if points[0] != points[-1]:
                points.append(points[0])

            name = f"{layer_name}_{i}"
            try:
                sheet = hfss.modeler.create_polyline(
                    points, cover_surface=True, name=name, material=material
                )
                if thickness != 0:
                    hfss.modeler.thicken_sheet(sheet.name, thickness)
                
                # Handle holes if any
                for j, hole in enumerate(dpoly.each_hole()):
                    hole_points = [[float(pt.x), float(pt.y), elevation] for pt in hole.each_point()]
                    if hole_points[0] != hole_points[-1]:
                        hole_points.append(hole_points[0])
                    
                    hole_name = f"{name}_hole_{j}"
                    hole_sheet = hfss.modeler.create_polyline(
                        hole_points, cover_surface=True, name=hole_name
                    )
                    hfss.modeler.subtract(sheet.name, [hole_sheet.name])
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to draw polygon {i} on layer {layer_name}: {e}")
                success = False

    return success


def import_component_to_hfss(
    hfss: Hfss,
    component: Component,
    layer_stack: LayerStack | None = None,
    *,
    units: str = "um",
    apply_additive: bool = True,
    gds_path: str | Path | None = None,
    import_method: int = 1,
    use_direct_draw: bool = True,
) -> bool:
    """Import a gdsfactory component into HFSS.

    By default, this uses :func:`draw_component_in_hfss` on Linux or if
    ``use_direct_draw`` is True, as native GDS import can be unstable in
    non-graphical environments.

    Args:
        hfss: The HFSS application instance.
        component: The gdsfactory component to import.
        layer_stack: LayerStack defining thickness and elevation for each layer.
            If None, uses QPDK's default LAYER_STACK.
        units: Length units for the geometry (default: "um" for micrometers).
        apply_additive: If True, applies additive metal operations before export.
            See :func:`prepare_component_for_hfss`.
        gds_path: Optional path to write the GDS file. If None, uses a temporary file.
        import_method: GDSII import method (0=script, 1=Parasolid). Default is 1.
        use_direct_draw: If True, use manual drawing instead of GDS import.

    Returns:
        True if import was successful, False otherwise.
    """
    if use_direct_draw:
        return draw_component_in_hfss(
            hfss, component, layer_stack, apply_additive=apply_additive
        )

    # Prepare component for export
    prepared_component = prepare_component_for_hfss(
        component, apply_additive=apply_additive
    )

    # Generate layer mapping from LayerStack
    mapping_layers = layer_stack_to_gds_mapping(layer_stack)

    # Export component to GDS
    # Note: We use TemporaryDirectory to ensure cleanup, but need to keep it
    # alive until import is complete, so we store the reference
    temp_dir_obj = None
    if gds_path is None:
        # Use temporary directory that will be cleaned up when function returns
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="qpdk_hfss_")
        gds_path = Path(temp_dir_obj.name) / "component.gds"

    gds_path = Path(gds_path)
    prepared_component.write_gds(str(gds_path))

    # Set modeler units
    hfss.modeler.model_units = units

    # Import GDS with 3D layer mapping
    result = hfss.import_gds_3d(
        input_file=str(gds_path),
        mapping_layers=mapping_layers,
        units=units,
        import_method=import_method,
    )

    # Clean up temporary directory if we created one
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return result


def create_hfss_project(
    project_name: str = "qpdk_simulation",
    design_name: str = "design1",
    solution_type: str = "Eigenmode",
    *,
    non_graphical: bool = True,
    aedt_version: str | None = None,
    new_desktop: bool = True,
    project_dir: str | Path | None = None,
) -> Hfss:
    """Create a new HFSS project with the specified solution type.

    Args:
        project_name: Name of the HFSS project.
        design_name: Name of the design within the project.
        solution_type: HFSS solution type. One of "Eigenmode", "DrivenModal",
            "DrivenTerminal", "Transient".
        non_graphical: If True, run HFSS without GUI. Set to False for debugging.
        aedt_version: AEDT version string (e.g., "2025.1"). If None, uses default.
        new_desktop: If True, starts a new AEDT desktop session.
        project_dir: Directory to save the project. If None, uses temp directory.

    Returns:
        An Hfss application instance.

    Example:
        >>> hfss = create_hfss_project(
        ...     project_name="resonator_sim",
        ...     solution_type="Eigenmode",
        ...     non_graphical=True
        ... )
    """
    _check_pyaedt_available()
    from ansys.aedt.core import Hfss, settings
    
    # Disable UDS to avoid hangs on Linux
    settings.use_grpc_uds = False

    project_path = None
    if project_dir is not None:
        project_path = str(Path(project_dir) / f"{project_name}.aedt")

    kwargs = {
        "project": project_path,
        "design": design_name,
        "solution_type": solution_type,
        "non_graphical": non_graphical,
        "new_desktop": new_desktop,
    }
    if aedt_version is not None:
        kwargs["version"] = aedt_version

    return Hfss(**kwargs)


def add_substrate_to_hfss(
    hfss: Hfss,
    component: Component,
    *,
    margin: float = 50.0,
    thickness: float = 500.0,
    material: str = "silicon",
) -> str:
    """Add a substrate box below the component geometry.

    Args:
        hfss: The HFSS application instance.
        component: The component to create substrate for (used for dimensions).
        margin: Extra margin around the component bounds in micrometers.
        thickness: Substrate thickness in micrometers.
        material: Substrate material name.

    Returns:
        Name of the created substrate object.
    """
    bounds = component.bbox()
    x_min, y_min = bounds.p1.x - margin, bounds.p1.y - margin
    x_max, y_max = bounds.p2.x + margin, bounds.p2.y + margin

    substrate = hfss.modeler.create_box(
        origin=[x_min, y_min, -thickness],
        sizes=[x_max - x_min, y_max - y_min, thickness],
        name="Substrate",
        material=material,
    )
    return substrate.name


def add_air_region_to_hfss(
    hfss: Hfss,
    component: Component,
    *,
    margin: float = 100.0,
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
        margin: Horizontal margin around the component in micrometers.
        height: Height above the component in micrometers.
        substrate_thickness: Depth below surface for the region.

    Returns:
        Name of the created region object.
    """
    bounds = component.bbox()
    x_min, y_min = bounds.p1.x - margin, bounds.p1.y - margin
    x_max, y_max = bounds.p2.x + margin, bounds.p2.y + margin

    region = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[x_max - x_min, y_max - y_min, height + substrate_thickness],
        name="AirRegion",
        material="vacuum",
    )

    # Assign PerfectE (PEC) boundary for closed-box eigenmode analysis
    hfss.assign_perfect_e(
        assignment=[face.id for face in region.faces],
        name="PEC_Boundary",
    )

    return region.name


def setup_eigenmode_simulation(
    hfss: Hfss,
    setup_name: str = "EigenmodeSetup",
    *,
    min_frequency_ghz: float = 1.0,
    num_modes: int = 3,
    max_passes: int = 15,
    min_passes: int = 2,
    percent_refinement: float = 30,
    max_delta_freq: float = 2,
) -> object:
    """Configure an eigenmode simulation setup.

    Args:
        hfss: The HFSS application instance (must have solution_type="Eigenmode").
        setup_name: Name for the simulation setup.
        min_frequency_ghz: Minimum frequency for mode search in GHz.
        num_modes: Number of eigenmodes to find.
        max_passes: Maximum number of adaptive passes.
        min_passes: Minimum number of adaptive passes.
        percent_refinement: Percentage of mesh refinement per pass.
        max_delta_freq: Maximum frequency change criterion (percentage).

    Returns:
        The created setup object.

    Example:
        >>> setup = setup_eigenmode_simulation(
        ...     hfss,
        ...     min_frequency_ghz=3.0,
        ...     num_modes=5
        ... )
    """
    setup = hfss.create_setup(name=setup_name)

    setup.props["MinimumFrequency"] = f"{min_frequency_ghz}GHz"
    setup.props["NumModes"] = num_modes
    setup.props["MaximumPasses"] = max_passes
    setup.props["MinimumPasses"] = min_passes
    setup.props["PercentRefinement"] = percent_refinement
    setup.props["MaxDeltaFreq"] = max_delta_freq
    setup.props["ConvergeOnRealFreq"] = True

    setup.update()
    return setup


def setup_driven_simulation(
    hfss: Hfss,
    setup_name: str = "DrivenSetup",
    *,
    frequency_ghz: float = 5.0,
    max_delta_s: float = 0.02,
    max_passes: int = 10,
    min_passes: int = 2,
    percent_refinement: float = 30,
    sweep_start_ghz: float | None = None,
    sweep_stop_ghz: float | None = None,
    sweep_points: int = 201,
) -> tuple[object, object | None]:
    """Configure a driven modal simulation setup with optional frequency sweep.

    Args:
        hfss: The HFSS application instance.
        setup_name: Name for the simulation setup.
        frequency_ghz: Solution frequency in GHz.
        max_delta_s: Maximum delta S convergence criterion.
        max_passes: Maximum number of adaptive passes.
        min_passes: Minimum number of adaptive passes.
        percent_refinement: Percentage of mesh refinement per pass.
        sweep_start_ghz: Start frequency for sweep. If None, no sweep is created.
        sweep_stop_ghz: Stop frequency for sweep. If None, no sweep is created.
        sweep_points: Number of frequency points in sweep.

    Returns:
        Tuple of (setup, sweep) where sweep may be None if not created.

    Example:
        >>> setup, sweep = setup_driven_simulation(
        ...     hfss,
        ...     frequency_ghz=5.0,
        ...     sweep_start_ghz=1.0,
        ...     sweep_stop_ghz=10.0
        ... )
    """
    setup = hfss.create_setup(
        name=setup_name,
        setup_type="HFSSDriven",
        Frequency=f"{frequency_ghz}GHz",
    )

    setup.props["MaxDeltaS"] = max_delta_s
    setup.props["MaximumPasses"] = max_passes
    setup.props["MinimumPasses"] = min_passes
    setup.props["PercentRefinement"] = percent_refinement
    setup.update()

    sweep = None
    if sweep_start_ghz is not None and sweep_stop_ghz is not None:
        sweep = setup.create_frequency_sweep(
            unit="GHz",
            name="FrequencySweep",
            start_frequency=sweep_start_ghz,
            stop_frequency=sweep_stop_ghz,
            sweep_type="Interpolating",
            num_of_freq_points=sweep_points,
        )

    return setup, sweep


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


def close_hfss(hfss: Hfss, *, save_project: bool = True) -> None:
    """Close the HFSS project and release the desktop.

    Args:
        hfss: The HFSS application instance.
        save_project: If True, saves the project before closing.
    """
    if save_project:
        hfss.save_project()
    hfss.release_desktop()
