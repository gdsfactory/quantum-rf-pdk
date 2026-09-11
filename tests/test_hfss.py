"""Tests for HFSS and Q3D simulation integration."""

import os
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from gdsfactory.component import Component
from gdsfactory.technology import LayerLevel, LayerStack
from numpy.testing import assert_allclose

from qpdk import LAYER_STACK, PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.resonator import resonator
from qpdk.simulation import (
    HFSS,
    Q2D,
    Q3D,
    layer_stack_to_gds_mapping,
    lumped_port_rectangle_from_cpw,
    prepare_component_for_aedt,
)
from qpdk.simulation.aedt_base import _get_layer_number_from_level
from qpdk.tech import LAYER_STACK_FLIP_CHIP, coplanar_waveguide

# Ensure Ansys path is set so PyAEDT can find it
ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
if "ANSYSEM_ROOT252" not in os.environ and Path(ansys_default_path).exists():
    os.environ["ANSYSEM_ROOT252"] = ansys_default_path


def test_layer_stack_to_gds_mapping():
    """Test generating GDS mapping from a LayerStack."""
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Check that it returns a dictionary
    assert isinstance(mapping, dict)

    # Check a known layer from qpdk
    # For example, layer 1 should be in the mapping
    # The structure is {layer_number: (elevation, thickness)}
    assert 1 in mapping
    assert isinstance(mapping[1], tuple)
    assert len(mapping[1]) == 2

    elevation, thickness = mapping[1]
    assert isinstance(elevation, float)
    assert isinstance(thickness, float)


def test_layer_stack_to_gds_mapping_default_stack_collision_free():
    """Test that the default LAYER_STACK maps every level to a distinct entry.

    pyaedt's ``import_gds_3d`` keys on GDS layer number only, so the Vacuum
    level (which shares ``LAYER.SIM_AREA`` (98, 0) with Substrate) must be
    skipped, and the contiguous Nb spans of Airbridge (10, 0) and
    Airbridge_Via (10, 1) must merge into a single box.
    """
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Vacuum (98, 0) is skipped, so Substrate keeps layer 98 at its own
    # elevation/thickness instead of being overwritten by vacuum values.
    assert mapping[98] == pytest.approx((-500.0, 500.0))

    # Airbridge (0.3-0.5 um) and Airbridge_Via (0.2-0.3 um) merge to one box.
    assert mapping[10] == pytest.approx((0.2, 0.3))

    # Every non-vacuum level with a resolvable layer number is mapped exactly
    # once (Airbridge and Airbridge_Via share one merged entry on layer 10).
    expected_layers = set()
    for level in LAYER_STACK.layers.values():
        if level.material == "vacuum":
            continue
        layer_number = _get_layer_number_from_level(level)
        if layer_number is not None:
            expected_layers.add(layer_number)
    assert set(mapping) == expected_layers

    # Sheets mode keeps the lowest elevation for colliding levels.
    sheets_mapping = layer_stack_to_gds_mapping(LAYER_STACK, thickness_override=0.0)
    assert sheets_mapping[10] == pytest.approx((0.2, 0.0))
    assert sheets_mapping[98] == pytest.approx((-500.0, 0.0))


def test_layer_stack_to_gds_mapping_unmergeable_collision_raises():
    """Test that unmergeable layer-number collisions raise a clear error."""
    stack = LayerStack(
        layers={
            "A": LayerLevel(
                name="A", layer=(7, 0), thickness=1.0, zmin=0.0, material="Nb"
            ),
            # Gap between spans: union is not a single box.
            "B": LayerLevel(
                name="B", layer=(7, 1), thickness=1.0, zmin=5.0, material="Nb"
            ),
        }
    )
    with pytest.raises(ValueError, match="do not join into a single box"):
        layer_stack_to_gds_mapping(stack)

    stack_different_material = LayerStack(
        layers={
            "A": LayerLevel(
                name="A", layer=(7, 0), thickness=1.0, zmin=0.0, material="Nb"
            ),
            "C": LayerLevel(
                name="C", layer=(7, 1), thickness=1.0, zmin=1.0, material="SiO2"
            ),
        }
    )
    with pytest.raises(ValueError, match="different materials"):
        layer_stack_to_gds_mapping(stack_different_material)

    # A material mismatch raises even in sheets mode (thickness_override set):
    # the colliding levels would still be imported as one object with a single
    # material.
    with pytest.raises(ValueError, match="different materials"):
        layer_stack_to_gds_mapping(stack_different_material, thickness_override=0.0)


def test_layer_stack_to_gds_mapping_order_independent():
    """Test that merged spans do not depend on layer stack insertion order.

    Three Nb levels on the same layer number whose z-spans chain into one
    contiguous box must merge to the same entry regardless of the order the
    levels appear in the ``LayerStack``.
    """

    def make_stack(names: Sequence[str]) -> LayerStack:
        spans = {"A": (0.0, 1.0), "B": (1.0, 3.0), "C": (3.0, 4.0)}
        return LayerStack(
            layers={
                name: LayerLevel(
                    name=name,
                    layer=(7, index),
                    thickness=spans[name][1] - spans[name][0],
                    zmin=spans[name][0],
                    material="Nb",
                )
                for index, name in enumerate(names)
            }
        )

    expected = layer_stack_to_gds_mapping(make_stack(("A", "B", "C")))
    assert expected[7] == pytest.approx((0.0, 4.0))
    assert layer_stack_to_gds_mapping(make_stack(("A", "C", "B"))) == expected
    assert layer_stack_to_gds_mapping(make_stack(("C", "B", "A"))) == expected


def test_layer_stack_to_gds_mapping_flip_chip_raises():
    """Test that the flip-chip stack raises on its Substrate collision.

    ``LAYER_STACK_FLIP_CHIP`` has ``Substrate`` (Si, span [-500, 0] um) and
    ``Substrate_top`` (Si, span [10.2, 510.2] um) sharing GDS layer 98 with a
    10.2 um vacuum gap between the chips. The two disjoint bodies cannot be
    represented by the single (elevation, thickness) box pyaedt's
    ``import_gds_3d`` allows per layer number, and merging would fill the
    inter-chip gap with silicon — so the mapping raises instead of silently
    producing wrong geometry. Flip-chip EM import is not currently supported;
    this test documents that contract.
    """
    with pytest.raises(
        ValueError, match=r"'Substrate' and 'Substrate_top' share GDS layer 98"
    ):
        layer_stack_to_gds_mapping(LAYER_STACK_FLIP_CHIP)


def test_prepare_component_for_aedt():
    """Test component preparation for AEDT."""
    comp = resonator()
    prepared = prepare_component_for_aedt(comp)

    assert isinstance(prepared, Component)
    # The prepared component should be different if additive metals are applied
    # or at least it should return a valid component
    assert prepared.name is not None


def test_prepare_component_for_aedt_margin():
    """Test component preparation for AEDT with margin."""
    comp = resonator(length=3000)

    # Original bbox
    bbox_orig = comp.bbox()
    prepared = prepare_component_for_aedt(comp, margin_draw=200)

    assert isinstance(prepared, Component)
    assert prepared.name is not None

    # Prepared bbox might be larger
    bbox_new = prepared.bbox()
    assert bbox_new.left <= bbox_orig.left
    assert bbox_new.right >= bbox_orig.right
    assert bbox_new.bottom <= bbox_orig.bottom
    assert bbox_new.top >= bbox_orig.top


def test_get_layer_number_from_level():
    """Test layer number extraction from various layer definitions."""

    # Test with a regular LayerLevel that has a direct tuple
    class MockLevelTuple:
        layer = (1, 0)

    assert _get_layer_number_from_level(MockLevelTuple()) == 1

    # Test with a derived layer structure
    class MockLogicalLayerInner:
        layer = (2, 0)

    class MockLogicalLayer:
        layer = MockLogicalLayerInner()

    class MockDerivedLevel:
        layer = None
        derived_layer = MockLogicalLayer()

    assert _get_layer_number_from_level(MockDerivedLevel()) == 2


@pytest.mark.hfss
def test_hfss_import_and_draw():
    """Test creating an HFSS project and drawing a component."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from ansys.aedt.core import Hfss, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_draw_{int(time.time())}"
    hfss_app = Hfss(
        project=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        hfss_sim = HFSS(hfss_app)
        # Use direct draw which we know is stable on Linux
        success = hfss_sim.import_component(comp)
        assert success, "Failed to draw component in HFSS"
    finally:
        hfss_app.release_desktop()


@pytest.mark.hfss
def test_hfss_eigenmode_setup():
    """Test setting up an eigenmode simulation."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from ansys.aedt.core import Hfss, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_name = f"test_eigenmode_{int(time.time())}"
    hfss_app = Hfss(
        project=project_name,
        solution_type="Eigenmode",
        non_graphical=True,
    )

    try:
        hfss_sim = HFSS(hfss_app)
        hfss_sim.import_component(comp)
        hfss_sim.add_substrate(comp)
        hfss_sim.add_air_region(comp)

        setup = hfss_app.create_setup(name="EigenmodeSetup")
        setup.props["MinimumFrequency"] = "1.0GHz"
        setup.props["NumModes"] = 1
        setup.props["MaximumPasses"] = 2
        setup.props["MinimumPasses"] = 2
        setup.props["PercentRefinement"] = 30
        setup.props["MaxDeltaFreq"] = 2.0
        setup.props["ConvergeOnRealFreq"] = True
        setup.update()

        assert setup is not None, "Failed to setup eigenmode simulation"
    finally:
        hfss_app.release_desktop()


@pytest.fixture
def mock_port_dimensions():
    """Provides a standardized set of dimensions for testing."""
    return {"center": [10.0, 20.0, 0.0], "cpw_gap": 6.0, "cpw_width": 2.0}


@pytest.mark.parametrize(
    ("orientation", "expected_origin", "expected_sizes", "expected_int_line"),
    [
        (0, [10.0, 19.0, 0.0], [6.0, 2.0], [[16.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (90, [9.0, 20.0, 0.0], [2.0, 6.0], [[10.0, 26.0, 0.0], [10.0, 20.0, 0.0]]),
        (180, [4.0, 19.0, 0.0], [6.0, 2.0], [[4.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (270, [9.0, 14.0, 0.0], [2.0, 6.0], [[10.0, 14.0, 0.0], [10.0, 20.0, 0.0]]),
    ],
)
def test_lumped_port_rectangle_from_cpw_valid_angles(
    mock_port_dimensions,
    orientation,
    expected_origin,
    expected_sizes,
    expected_int_line,
):
    """Verifies that the vectorized geometry perfectly matches the expected dictionary values."""
    result = lumped_port_rectangle_from_cpw(
        center=mock_port_dimensions["center"],
        orientation=orientation,
        cpw_gap=mock_port_dimensions["cpw_gap"],
        cpw_width=mock_port_dimensions["cpw_width"],
    )

    assert_allclose(result["origin"], expected_origin)
    assert_allclose(result["sizes"], expected_sizes)
    assert_allclose(result["integration_line"], expected_int_line)


def test_lumped_port_rectangle_from_cpw_invalid_angle(mock_port_dimensions):
    """Ensures the function throws a ValueError if passed an unaligned angle."""
    with pytest.raises(ValueError, match="Unsupported port orientation: 45°"):
        lumped_port_rectangle_from_cpw(
            center=mock_port_dimensions["center"],
            orientation=45,
            cpw_gap=mock_port_dimensions["cpw_gap"],
            cpw_width=mock_port_dimensions["cpw_width"],
        )


@pytest.mark.hfss
def test_q3d_import_and_net_assignment():
    """Test creating a Q3D project, importing a component, and assigning nets."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS/Q3D installation not found at {ansys_dir}")

    from ansys.aedt.core import Q3d, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = interdigital_capacitor(fingers=4, finger_length=20)

    project_name = f"test_q3d_{int(time.time())}"
    q3d_app = Q3d(
        project=project_name,
        solution_type="Q3DExtractor",
        non_graphical=True,
    )

    try:
        q3d_sim = Q3D(q3d_app)
        conductor_objects = q3d_sim.import_component(comp)
        assert len(conductor_objects) > 0, "No conductor objects imported"

        signal_nets = q3d_sim.assign_nets_from_ports(comp.ports, conductor_objects)
        assert len(signal_nets) > 0, "No signal nets assigned"
    finally:
        q3d_app.release_desktop()


@pytest.mark.hfss
def test_create_2d_from_cross_section():
    """Test creating a Q2D model from a CPW cross-section."""
    ansys_dir = os.environ.get("ANSYSEM_ROOT252", "/usr/ansys_inc/v252/AnsysEM")
    if not Path(ansys_dir).exists():
        pytest.skip(f"HFSS installation not found at {ansys_dir}")

    from ansys.aedt.core import Q2d, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    cross_section = coplanar_waveguide(width=10, gap=6)

    project_name = f"test_q2d_{int(time.time())}"
    q2d_app = Q2d(
        project=project_name,
        design="CPW_Test",
        non_graphical=True,
    )

    try:
        q2d_sim = Q2D(q2d_app)
        result = q2d_sim.create_2d_from_cross_section(cross_section)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "gnd_left" in result
        assert "gnd_right" in result
        assert "substrate" in result
    finally:
        q2d_app.release_desktop()
