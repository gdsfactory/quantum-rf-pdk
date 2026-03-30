"""Tests for the KLayout LVS deck rendering."""

from __future__ import annotations

from qpdk.klayout.lvs import render_lvs as _render_lvs_mod
from qpdk.klayout.lvs.render_lvs import (
    CONNECTIONS,
    DEVICES,
    MARKER_LAYERS,
    PHYSICAL_LAYERS,
    PORT_LAYERS,
    render,
)


def test_render_lvs_produces_output():
    """Render the LVS deck and verify it is non-empty."""
    result = render(write=False)
    assert len(result) > 0


def test_render_lvs_contains_layer_definitions():
    """Rendered deck must mention every physical layer."""
    result = render(write=False)
    for layer in PHYSICAL_LAYERS:
        assert f"{layer['var']} = input(" in result


def test_render_lvs_contains_port_layers():
    """Rendered deck must define all port layers."""
    result = render(write=False)
    for port in PORT_LAYERS:
        assert f"{port['var']} = input(" in result


def test_render_lvs_contains_marker_layers():
    """Rendered deck must define all marker layers."""
    result = render(write=False)
    for marker in MARKER_LAYERS:
        assert f"{marker['var']} = input(" in result


def test_render_lvs_contains_connectivity():
    """Rendered deck must include all connection statements."""
    result = render(write=False)
    for conn in CONNECTIONS:
        assert f"connect({conn['layer_a']}, {conn['via']}, {conn['layer_b']})" in result


def test_render_lvs_contains_port_connections():
    """Rendered deck must connect port layers to their metal."""
    result = render(write=False)
    for port in PORT_LAYERS:
        assert f"connect({port['metal_var']}, {port['var']})" in result


def test_render_lvs_contains_device_extraction():
    """Rendered deck must extract every registered device type."""
    result = render(write=False)
    for device in DEVICES:
        assert f"{device['var']}_region = " in result
        assert device["model_name"] in result


def test_render_lvs_contains_compare():
    """Rendered deck must end with a comparison step."""
    result = render(write=False)
    assert "schematic(spice_netlist)" in result
    assert "compare" in result


def test_rendered_file_matches_template():
    """The committed .lvs file must match what the renderer produces."""
    string_result = render(write=False)
    committed_lvs_deck_path = _render_lvs_mod._HERE / "qpdk.lvs"
    assert committed_lvs_deck_path.exists(), (
        "qpdk.lvs must be committed alongside the template"
    )
    file_result = committed_lvs_deck_path.read_text()
    assert string_result == file_result
