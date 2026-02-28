"""HFSS simulation utilities using PyAEDT.

This module provides helper functions for setting up HFSS simulations
(eigenmode and driven modal) from gdsfactory components. It uses the
PyAEDT library to interface with Ansys HFSS.

Note:
    This module requires the optional ``hfss`` dependency group.
    Install with: ``uv sync --extra hfss`` or ``pip install qpdk[hfss]``

Example:
    >>> from qpdk.models.hfss import create_hfss_from_component
    >>> from qpdk.cells import resonator
    >>> comp = resonator(length=4000, meanders=4)
    >>> hfss = create_hfss_from_component(comp, solution_type="Eigenmode")

References:
    - PyAEDT documentation: https://aedt.docs.pyansys.com/
    - HFSS Examples: https://examples.aedt.docs.pyansys.com/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ansys.aedt.core import Hfss
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerLevel, LayerStack


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


def component_polygons_to_numpy(
    component: Component,
    layer: tuple[int, int],
) -> list[NDArray[np.floating]]:
    """Extract polygon coordinates from a gdsfactory component on a specific layer.

    Args:
        component: The gdsfactory component to extract polygons from.
        layer: The layer tuple (layer_number, datatype) to extract.

    Returns:
        List of numpy arrays, each containing polygon vertex coordinates
        with shape (N, 2) where N is the number of vertices.
    """
    polygons = component.get_polygons(layers=[layer], by_spec=True)
    return [np.array(poly) for poly in polygons.get(layer, [])]


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
    from ansys.aedt.core import Hfss

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


def add_component_geometry_to_hfss(
    hfss: Hfss,
    component: Component,
    layer_stack: LayerStack | None = None,
    *,
    units: str = "um",
) -> dict[str, list[str]]:
    """Add a gdsfactory component's geometry to an HFSS project.

    Extracts polygons from each layer in the component and creates
    corresponding 3D objects in HFSS based on the layer stack definitions.

    Args:
        hfss: The HFSS application instance.
        component: The gdsfactory component to add.
        layer_stack: LayerStack defining thickness and materials for each layer.
            If None, uses QPDK's default LAYER_STACK.
        units: Length units for the geometry (default: "um" for micrometers).

    Returns:
        Dictionary mapping layer names to lists of created object names.

    Example:
        >>> from qpdk.cells import resonator
        >>> comp = resonator(length=4000)
        >>> objects = add_component_geometry_to_hfss(hfss, comp)
    """
    from qpdk import LAYER_STACK

    if layer_stack is None:
        layer_stack = LAYER_STACK

    hfss.modeler.model_units = units

    created_objects: dict[str, list[str]] = {}

    # Map GDS layers to layer stack entries
    for layer_name, layer_level in layer_stack.layers.items():
        # Get the layer tuple for this layer level
        layer_tuple = _get_layer_tuple_from_level(layer_level)
        if layer_tuple is None:
            continue

        # Extract polygons from this layer
        polygons = component_polygons_to_numpy(component, layer_tuple)
        if not polygons:
            continue

        created_objects[layer_name] = []
        material = layer_level.material if layer_level.material else "pec"
        thickness = layer_level.thickness if layer_level.thickness else 0.2
        zmin = layer_level.zmin if layer_level.zmin is not None else 0.0

        for i, poly in enumerate(polygons):
            obj_name = f"{layer_name}_{i}"
            # Create polyline from polygon vertices
            points = [[float(x), float(y), float(zmin)] for x, y in poly]
            points.append(points[0])  # Close the polygon

            # Create polyline and cover to make a 2D surface
            polyline = hfss.modeler.create_polyline(
                points=points,
                cover_surface=True,
                name=obj_name,
            )

            if polyline is not None:
                # Thicken to 3D if thickness > 0
                if thickness > 0:
                    hfss.modeler.thicken_sheet(polyline, thickness)

                # Assign material (use PEC for metals in superconducting sims)
                if material in ["Nb", "Al", "TiN", "copper"]:
                    hfss.assign_perfect_conductor(polyline.name)
                else:
                    hfss.modeler.assign_material(polyline.name, material)

                created_objects[layer_name].append(polyline.name)

    return created_objects


def _get_layer_tuple_from_level(layer_level: LayerLevel) -> tuple[int, int] | None:
    """Extract layer tuple from a LayerLevel's layer definition.

    Args:
        layer_level: A gdsfactory LayerLevel object.

    Returns:
        Layer tuple (layer_number, datatype) or None if not extractable.
    """
    layer = layer_level.layer
    if isinstance(layer, tuple) and len(layer) == 2:
        return layer
    # Handle LogicalLayer
    if hasattr(layer, "layer") and isinstance(layer.layer, tuple):
        return layer.layer
    return None


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
    bounds = component.bbox
    x_min, y_min = bounds[0] - margin
    x_max, y_max = bounds[1] + margin

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

    Args:
        hfss: The HFSS application instance.
        component: The component to create air region around.
        margin: Horizontal margin around the component in micrometers.
        height: Height above the component in micrometers.
        substrate_thickness: Depth below surface for the region.

    Returns:
        Name of the created region object.
    """
    bounds = component.bbox
    x_min, y_min = bounds[0] - margin
    x_max, y_max = bounds[1] + margin

    region = hfss.modeler.create_box(
        origin=[x_min, y_min, -substrate_thickness],
        sizes=[x_max - x_min, y_max - y_min, height + substrate_thickness],
        name="AirRegion",
        material="vacuum",
    )

    # Assign radiation boundary (or PerfectE for eigenmode)
    hfss.assign_perfect_conductor(
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

            # Extract magnitude in dB
            results["s_parameters"][trace] = {
                "magnitude_db": 20 * np.log10(np.abs(solution.data_magnitude())),
                "phase_deg": np.degrees(np.angle(solution.data_magnitude())),
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
