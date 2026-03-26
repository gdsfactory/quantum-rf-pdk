"""Mock-based tests for the simulation module (HFSS, Q3D, Q2D).

These tests verify the simulation wrapper logic without requiring
a licensed Ansys installation.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from gdsfactory.component import Component

from qpdk import LAYER_STACK
from qpdk.simulation.aedt_base import (
    AEDTBase,
    _get_layer_number_from_level,
    add_materials_to_aedt,
    export_component_to_gds_temp,
    layer_stack_to_gds_mapping,
    rename_imported_objects,
)


class TestLayerStackToGdsMapping:
    """Tests for layer_stack_to_gds_mapping."""

    @staticmethod
    def test_default_layer_stack() -> None:
        """Test with default LAYER_STACK."""
        mapping = layer_stack_to_gds_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for layer_num, (elev, thick) in mapping.items():
            assert isinstance(layer_num, int)
            assert isinstance(elev, float)
            assert isinstance(thick, float)

    @staticmethod
    def test_thickness_override() -> None:
        """Test that thickness_override replaces all thicknesses."""
        mapping = layer_stack_to_gds_mapping(thickness_override=0.0)
        for _layer_num, (_, thick) in mapping.items():
            assert thick == 0.0


class TestGetLayerNumberFromLevel:
    """Tests for _get_layer_number_from_level edge cases."""

    @staticmethod
    def test_tuple_layer() -> None:
        """Test extracting layer number from a tuple."""

        class MockLevel:
            layer = (5, 0)

        assert _get_layer_number_from_level(MockLevel()) == 5

    @staticmethod
    def test_nested_layer_attr() -> None:
        """Test extracting layer number from nested .layer.layer."""

        class Inner:
            layer = (3, 0)

        class MockLevel:
            layer = Inner()

        assert _get_layer_number_from_level(MockLevel()) == 3

    @staticmethod
    def test_deeply_nested_layer() -> None:
        """Test extracting layer number from deeply nested .layer.layer.layer."""

        class DeepInner:
            layer = (7, 0)

        class Inner:
            layer = DeepInner()

        class MockLevel:
            layer = Inner()

        assert _get_layer_number_from_level(MockLevel()) == 7

    @staticmethod
    def test_derived_layer() -> None:
        """Test extracting layer number from derived_layer."""

        class InnerInner:
            layer = (2, 0)

        class LogicalLayer:
            layer = InnerInner()

        class MockLevel:
            layer = None
            derived_layer = LogicalLayer()

        assert _get_layer_number_from_level(MockLevel()) == 2

    @staticmethod
    def test_none_returns_none() -> None:
        """Test that a level with no extractable layer returns None."""

        class MockLevel:
            layer = None

        assert _get_layer_number_from_level(MockLevel()) is None

    @staticmethod
    def test_int_inner_layer() -> None:
        """Test extracting layer from inner int value."""

        class Inner:
            layer = 4

        class MockLevel:
            layer = Inner()

        assert _get_layer_number_from_level(MockLevel()) == 4


class TestExportComponentToGdsTemp:
    """Tests for the export_component_to_gds_temp context manager."""

    @staticmethod
    def test_temp_dir_mode() -> None:
        """Test that a temp file is created and cleaned up."""
        from qpdk.cells.bump import indium_bump

        comp = indium_bump()
        with export_component_to_gds_temp(comp) as path:
            assert path.exists()
            assert path.suffix == ".gds"
        # Temp dir should be cleaned up
        assert not path.exists()

    @staticmethod
    def test_explicit_path(tmp_path: Path) -> None:
        """Test writing to an explicit path."""
        from qpdk.cells.bump import indium_bump

        comp = indium_bump()
        gds_path = tmp_path / "test.gds"
        with export_component_to_gds_temp(comp, gds_path=gds_path) as path:
            assert path == gds_path
            assert path.exists()
        # Explicit path is NOT cleaned up
        assert gds_path.exists()


class TestRenameImportedObjects:
    """Tests for rename_imported_objects."""

    @staticmethod
    def test_renames_signal_objects() -> None:
        """Test that signal<N> objects get renamed to layer names."""
        mock_app = MagicMock()
        new_objects = ["signal1_0", "signal1_1"]

        renamed = rename_imported_objects(mock_app, new_objects, LAYER_STACK)
        # Should attempt to rename objects matching signal<N> pattern
        assert len(renamed) == 2
        for name in renamed:
            assert isinstance(name, str)

    @staticmethod
    def test_non_signal_objects_unchanged() -> None:
        """Test that non-signal objects pass through unchanged."""
        mock_app = MagicMock()
        new_objects = ["SomeOtherObject", "AnotherObject"]

        renamed = rename_imported_objects(mock_app, new_objects, LAYER_STACK)
        assert renamed == ["SomeOtherObject", "AnotherObject"]

    @staticmethod
    def test_rename_exception_falls_back() -> None:
        """Test that rename failure falls back to original name."""
        mock_app = MagicMock()
        # Make the modeler assignment raise
        mock_app.modeler.__getitem__().name = PropertyMock(
            side_effect=RuntimeError("rename failed")
        )
        new_objects = ["signal1_0"]
        renamed = rename_imported_objects(mock_app, new_objects, LAYER_STACK)
        assert len(renamed) == 1


class TestAddMaterialsToAedt:
    """Tests for add_materials_to_aedt."""

    @staticmethod
    def test_adds_new_materials() -> None:
        """Test that materials are added when they don't exist."""
        mock_app = MagicMock()
        mock_app.materials.exists_material.return_value = False
        mock_mat = MagicMock()
        mock_app.materials.add_material.return_value = mock_mat

        add_materials_to_aedt(mock_app)

        assert mock_app.materials.add_material.called

    @staticmethod
    def test_skips_existing_materials() -> None:
        """Test that existing materials are not re-added."""
        mock_app = MagicMock()
        mock_app.materials.exists_material.return_value = True

        add_materials_to_aedt(mock_app)

        mock_app.materials.add_material.assert_not_called()


class TestAEDTBase:
    """Tests for the AEDTBase class."""

    @staticmethod
    def test_modeler_property() -> None:
        """Test that modeler delegates to app.modeler."""
        mock_app = MagicMock()
        base = AEDTBase(mock_app)
        assert base.modeler is mock_app.modeler

    @staticmethod
    def test_add_materials() -> None:
        """Test that add_materials calls add_materials_to_aedt."""
        mock_app = MagicMock()
        mock_app.materials.exists_material.return_value = True
        base = AEDTBase(mock_app)
        base.add_materials()
        # Should have checked for material existence
        assert mock_app.materials.exists_material.called

    @staticmethod
    def test_save() -> None:
        """Test that save delegates to app.save_project."""
        mock_app = MagicMock()
        base = AEDTBase(mock_app)
        base.save()
        mock_app.save_project.assert_called_once()

    @staticmethod
    def test_add_substrate() -> None:
        """Test that add_substrate creates a box with correct parameters."""
        from qpdk.cells.bump import indium_bump

        mock_app = MagicMock()
        mock_box = MagicMock()
        mock_box.name = "Substrate"
        mock_app.modeler.create_box.return_value = mock_box

        base = AEDTBase(mock_app)
        comp = indium_bump()
        result = base.add_substrate(comp, thickness=500.0, material="silicon")

        assert result == "Substrate"
        mock_app.modeler.create_box.assert_called_once()
        call_kwargs = mock_app.modeler.create_box.call_args
        assert call_kwargs.kwargs["material"] == "silicon"
        assert call_kwargs.kwargs["name"] == "Substrate"


class TestHFSSMock:
    """Mock-based tests for HFSS wrapper."""

    @staticmethod
    def test_lumped_port_rectangle_invalid_orientation() -> None:
        """Test that non-90-degree orientation raises ValueError."""
        from qpdk.simulation.hfss import lumped_port_rectangle_from_cpw

        with pytest.raises(ValueError, match="Unsupported port orientation: 45"):
            lumped_port_rectangle_from_cpw(
                center=(0.0, 0.0, 0.0), orientation=45, cpw_gap=6.0, cpw_width=10.0
            )

    @staticmethod
    def test_hfss_init() -> None:
        """Test HFSS wrapper initialization."""
        from qpdk.simulation.hfss import HFSS

        mock_hfss = MagicMock()
        wrapper = HFSS(mock_hfss)
        assert wrapper.hfss is mock_hfss
        assert wrapper.app is mock_hfss

    @staticmethod
    def test_hfss_import_component() -> None:
        """Test HFSS import_component with mock."""
        from qpdk.simulation.hfss import HFSS

        mock_hfss = MagicMock()
        mock_hfss.modeler.object_names = ["obj1", "obj2"]
        mock_hfss.import_gds_3d.return_value = True

        wrapper = HFSS(mock_hfss)

        from qpdk.cells.bump import indium_bump

        comp = indium_bump()
        # Mock the import to return True
        result = wrapper.import_component(comp)
        assert mock_hfss.import_gds_3d.called

    @staticmethod
    def test_hfss_add_air_region() -> None:
        """Test HFSS add_air_region with mock."""
        from qpdk.simulation.hfss import HFSS

        mock_hfss = MagicMock()
        mock_region = MagicMock()
        mock_region.name = "AirRegion"
        mock_hfss.modeler.create_box.return_value = mock_region

        wrapper = HFSS(mock_hfss)

        from qpdk.cells.bump import indium_bump

        comp = indium_bump()
        result = wrapper.add_air_region(comp, height=500.0)
        assert result == "AirRegion"
        mock_hfss.modeler.create_box.assert_called_once()

    @staticmethod
    def test_hfss_add_air_region_with_pec() -> None:
        """Test HFSS add_air_region with PEC boundary."""
        from qpdk.simulation.hfss import HFSS

        mock_hfss = MagicMock()
        mock_region = MagicMock()
        mock_region.name = "AirRegion"
        mock_region.faces = [MagicMock(id=1), MagicMock(id=2)]
        mock_hfss.modeler.create_box.return_value = mock_region

        wrapper = HFSS(mock_hfss)

        from qpdk.cells.bump import indium_bump

        comp = indium_bump()
        wrapper.add_air_region(comp, pec_boundary=True)
        mock_hfss.assign_perfect_e.assert_called_once()


class TestQ3DMock:
    """Mock-based tests for Q3D wrapper."""

    @staticmethod
    def test_q3d_init() -> None:
        """Test Q3D wrapper initialization."""
        from qpdk.simulation.q3d import Q3D

        mock_q3d = MagicMock()
        wrapper = Q3D(mock_q3d)
        assert wrapper.q3d is mock_q3d
        assert wrapper.app is mock_q3d

    @staticmethod
    def test_q3d_assign_nets_empty_ports() -> None:
        """Test assign_nets_from_ports with empty ports returns empty list."""
        from qpdk.simulation.q3d import Q3D

        mock_q3d = MagicMock()
        wrapper = Q3D(mock_q3d)
        result = wrapper.assign_nets_from_ports([], ["obj1"])
        assert result == []

    @staticmethod
    def test_q3d_assign_nets_empty_conductors() -> None:
        """Test assign_nets_from_ports with empty conductors returns empty list."""
        from qpdk.simulation.q3d import Q3D

        mock_q3d = MagicMock()
        wrapper = Q3D(mock_q3d)
        mock_port = MagicMock()
        result = wrapper.assign_nets_from_ports([mock_port], [])
        assert result == []


class TestQ2DMock:
    """Mock-based tests for Q2D wrapper."""

    @staticmethod
    def test_q2d_init() -> None:
        """Test Q2D wrapper initialization."""
        from qpdk.simulation.q3d import Q2D

        mock_q2d = MagicMock()
        wrapper = Q2D(mock_q2d)
        assert wrapper.q2d is mock_q2d
        assert wrapper.app is mock_q2d

    @staticmethod
    def test_q2d_invalid_units_raises() -> None:
        """Test that non-um units raise ValueError."""
        from qpdk.simulation.q3d import Q2D

        mock_q2d = MagicMock()
        wrapper = Q2D(mock_q2d)

        from qpdk.tech import coplanar_waveguide

        xs = coplanar_waveguide(width=10, gap=6)
        with pytest.raises(ValueError, match="expects units='um'"):
            wrapper.create_2d_from_cross_section(xs, units="mm")
