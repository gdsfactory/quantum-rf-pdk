"""Keysight EMPro simulation utilities."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


def _setup_empro_env() -> None:
    """Automatically set up EMPro environment variables and relaunch if necessary."""
    empro_dir = os.environ.get("EMPRO_DIR")
    if not empro_dir:
        return

    linux_bin = os.path.join(empro_dir, "linux_x86_64", "bin")
    if not os.path.exists(linux_bin):
        return

    env_changed = False

    if os.environ.get("EMPROHOME") != linux_bin:
        os.environ["EMPROHOME"] = linux_bin
        os.environ["OA_PLUGIN_PATH"] = os.path.join(empro_dir, "data", "plugins")
        os.environ["QT_PLUGIN_PATH"] = os.path.join(linux_bin, "plugins", "qt")
        env_changed = True

    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    if linux_bin not in ld_lib_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{linux_bin}:{ld_lib_path}" if ld_lib_path else linux_bin
        )
        env_changed = True

    python_path = os.environ.get("PYTHONPATH", "")
    if linux_bin not in python_path.split(":"):
        os.environ["PYTHONPATH"] = (
            f"{linux_bin}:{python_path}" if python_path else linux_bin
        )
        env_changed = True

    # Do not relaunch if we're in a test runner or Jupyter notebook to avoid breaking the process
    if "pytest" in sys.modules or "ipykernel" in sys.modules:
        return

    # Only relaunch if we had to modify LD_LIBRARY_PATH (which requires a process restart to take effect for C-extensions)
    if env_changed and "EMPRO_ENV_RELAUNCHED" not in os.environ:
        os.environ["EMPRO_ENV_RELAUNCHED"] = "1"
        try:
            # Check if we can import empro already. If so, we don't need to restart.
            import empro  # noqa: F401
        except ImportError:
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


# Relaunch immediately if needed
_setup_empro_env()

from qpdk import LAYER_STACK
from qpdk.simulation.aedt_base import (
    layer_stack_to_gds_mapping,
)
from qpdk.tech import material_properties

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack
    from gdsfactory.typings import Ports


class EMPro:
    """Keysight EMPro simulation wrapper.

    Provides high-level methods for importing components into EMPro,
    setting up simulation regions, and configuring FEM simulations.
    """

    def __init__(self, project: Any = None):
        """Initialize the EMPro wrapper.

        Args:
            project: The empro.activeProject instance. If None, imports empro.
        """
        import empro  # noqa: PLC0415
        from empro import toolkit  # noqa: PLC0415

        self.project = project or empro.activeProject
        self.empro = empro
        self.toolkit = toolkit

    def add_materials(self) -> None:
        """Add QPDK materials to the EMPro project."""
        existing_names = self.project.materials.names()
        for name, props in material_properties.items():
            if name in existing_names:
                continue

            mat = self.empro.material.Material()
            mat.name = name

            for prop_name, prop_value in props.items():
                if prop_value == float("inf"):
                    if prop_name == "relative_permittivity":
                        mat.details.electricProperties.parameters.conductivity = 1e30
                    continue

                if prop_name == "relative_permittivity":
                    mat.details.electricProperties.parameters.relativePermittivity = (
                        prop_value
                    )
                elif prop_name == "conductivity":
                    mat.details.electricProperties.parameters.conductivity = prop_value

            self.project.materials.append(mat)

    def import_component(
        self,
        component: Component,
        layer_stack: LayerStack | None = None,
    ) -> list[str]:
        """Import a gdsfactory component into EMPro via extrusion.

        Args:
            component: The gdsfactory component to import.
            layer_stack: LayerStack defining thickness and elevation for each layer.
                If None, uses QPDK's default LAYER_STACK.

        Returns:
            List of names of the created objects.
        """
        if layer_stack is None:
            layer_stack = LAYER_STACK

        mapping = layer_stack_to_gds_mapping(layer_stack)
        created_objects = []

        all_polygons = component.get_polygons(by="index")
        dbu = component.kcl.dbu

        # Get polygons for each layer
        for layer_number, (elevation, thickness) in mapping.items():
            # Find layer name in layer_stack
            layer_name = None
            for name, level in layer_stack.layers.items():
                from qpdk.simulation.aedt_base import (  # noqa: PLC0415
                    _get_layer_number_from_level,
                )

                if _get_layer_number_from_level(level) == layer_number:
                    layer_name = name
                    break

            if layer_name is None:
                layer_name = f"Layer_{layer_number}"

            if layer_number not in all_polygons:
                continue

            polygons = all_polygons[layer_number]

            for i, poly in enumerate(polygons):
                # Convert to micron-scaled DPolygon and get points
                dpoly = poly.to_dtype(dbu)
                pts = [
                    self.empro.geometry.Vector2d(p.x, p.y)
                    for p in dpoly.each_point_hull()
                ]
                sketch = self.toolkit.geometry.PolySketch(pts)

                # Extrude to create Model
                model = self.toolkit.geometry.extrude(sketch, thickness)
                model.name = f"{layer_name}_{i}"

                # Set position (elevation)
                self.toolkit.geometry.translate(
                    model, self.empro.geometry.Vector3d(0, 0, elevation)
                )

                # Assign material if possible
                with contextlib.suppress(Exception):
                    mat_name = layer_stack.layers[layer_name].material
                    model.material = self.project.materials[mat_name]

                self.project.geometry.append(model)
                created_objects.append(model.name)

        return created_objects

    def add_substrate(
        self,
        component: Component,
        thickness: float = 500.0,
        material: str = "silicon",
        name: str = "Substrate",
    ) -> str:
        """Add a substrate block below the component geometry.

        Args:
            component: The component to create substrate around.
            thickness: Thickness of the substrate in micrometers.
            material: Material name for the substrate.
            name: Name of the created substrate object.

        Returns:
            Name of the created substrate object.
        """
        bounds = component.bbox()
        x_min, y_min = bounds.p1.x, bounds.p1.y
        x_max, y_max = bounds.p2.x, bounds.p2.y

        # Create EMPro Block (which returns a Model)
        v1 = self.empro.geometry.Vector3d(x_min, y_min, -thickness)
        v2 = self.empro.geometry.Vector3d(x_max, y_max, 0)
        model = self.toolkit.geometry.Block(v1, v2)
        model.name = name

        # Assign material
        with contextlib.suppress(Exception):
            model.material = self.project.materials[material]

        self.project.geometry.append(model)
        return model.name

    def add_air_region(
        self,
        component: Component,
        height: float = 500.0,
        substrate_thickness: float = 500.0,
        material: str = "Air",
        name: str = "AirRegion",
    ) -> str:
        """Add an air region (box) around the component.

        Args:
            component: The component to create air region around.
            height: Height above the component in micrometers.
            substrate_thickness: Depth below surface for the region.
            material: Material name for the region.
            name: Name of the created region object.

        Returns:
            Name of the created region object.
        """
        bounds = component.bbox()
        x_min, y_min = bounds.p1.x, bounds.p1.y
        x_max, y_max = bounds.p2.x, bounds.p2.y

        # Create EMPro Block
        v1 = self.empro.geometry.Vector3d(x_min, y_min, -substrate_thickness)
        v2 = self.empro.geometry.Vector3d(x_max, y_max, height)
        model = self.toolkit.geometry.Block(v1, v2)
        model.name = name

        # Assign material
        with contextlib.suppress(Exception):
            model.material = self.project.materials[material]

        self.project.geometry.append(model)
        return model.name

    def add_lumped_ports(self, ports: Ports, resistance: float = 50.0) -> None:
        """Add lumped ports to EMPro at given port locations.

        Args:
            ports: Collection of gdsfactory ports defining signal locations.
            resistance: Port resistance in ohms.
        """
        for port in ports:
            # Calculate tail and head based on orientation and width
            # For a coplanar waveguide port, we usually want the port to span
            # from the signal center to the ground plane.
            # Here we set the port size to match the port width to ensure it
            # connects the signal to the reference.
            theta = np.deg2rad(port.orientation)
            # Use the port width as the reference dimension for the lumped port
            gap = port.width if port.width > 0 else 1.0
            
            # The port should typically be aligned with the normal to the orientation
            # to span across the gap, or along the orientation to feed into it.
            # Usually, internal lumped ports are created along the direction of propagation
            dx = (gap / 2) * np.cos(theta)
            dy = (gap / 2) * np.sin(theta)

            tail = self.empro.geometry.Vector3d(
                port.center[0] - dx, port.center[1] - dy, 0
            )
            head = self.empro.geometry.Vector3d(
                port.center[0] + dx, port.center[1] + dy, 0
            )

            # Create Feed
            feed = self.empro.components.Feed()
            feed.impedance.resistance = resistance
            feed.impedance.inductance = 0
            feed.impedance.capacitance = 0

            # Create Circuit Component
            comp = self.empro.components.CircuitComponent()
            comp.name = port.name
            comp.definition = feed
            comp.head = head
            comp.tail = tail
            comp.port = True

            self.project.circuitComponents.append(comp)

    def setup_fem_simulation(
        self, start_freq: float, stop_freq: float, num_points: int = 101
    ) -> None:
        """Set up a FEM simulation with a frequency sweep.

        Args:
            start_freq: Start frequency in Hz.
            stop_freq: Stop frequency in Hz.
            num_points: Number of frequency points.
        """
        sim_setup = self.project.simulationSettings
        sim_setup.simulator = "com.keysight.xxpro.simulator.fem"

        # Clear existing plans and create new one
        plans = sim_setup.femFrequencyPlanList()
        plans.clear()

        plan = self.empro.simulation.FrequencyPlan()
        plan.startFrequency = start_freq
        plan.stopFrequency = stop_freq
        plan.numberOfFrequencyPoints = num_points
        plan.type = "Adaptive"

        plans.append(plan)

    def setup_boundary_conditions(self, padding: float = 1000.0) -> None:
        """Set up boundary conditions for the FEM simulation.

        Args:
            padding: Padding distance in microns for the Absorbing boundary.
        """
        bc = self.project.boundaryConditions
        bc.xLowerBoundaryType = "Absorbing"
        bc.xUpperBoundaryType = "Absorbing"
        bc.yLowerBoundaryType = "Absorbing"
        bc.yUpperBoundaryType = "Absorbing"
        bc.zLowerBoundaryType = "Absorbing"
        bc.zUpperBoundaryType = "Absorbing"

    def save_as(self, path: str | Path) -> None:
        """Save the active project to a specific path.

        Args:
            path: The file path to save the project to (should end in .ep).
        """
        self.project.saveActiveProjectTo(str(path))

    def get_simulation_log(self, sim: Any) -> str:
        """Retrieve the simulation log for a given simulation.

        Args:
            sim: The EMPro simulation object.

        Returns:
            The content of the simulation log.
        """
        # EMPro stores logs in the project directory under simulations/sim_id/log
        try:
            project_dir = Path(str(self.project.location.canonicalPathToProjectXml())).parent
        except Exception:
            project_dir = Path(str(self.project.location.directory()))
        
        log_path = project_dir / "simulations" / f"{int(sim.id()):06d}" / "log"
        if not log_path.exists():
            # Try without zero-padding
            log_path = project_dir / "simulations" / str(sim.id()) / "log"
            
        if log_path.exists():
            return log_path.read_text()
        return f"Log file not found at {log_path}"

    def run_simulation(self, wait: bool = True) -> Any:
        """Run the simulation.

        Args:
            wait: If True, blocks until simulation is complete.

        Returns:
            The simulation object.
        """
        sim = self.project.createSimulation(True)
        if wait:
            try:
                self.toolkit.simulation.wait(sim)
            except Exception as e:
                print(f"Error waiting for simulation: {e}")
        return sim
