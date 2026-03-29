"""Render the QPDK KLayout LVS deck from a Jinja2 template.

The template ``qpdk.lvs.j2`` lives next to this script and is rendered into
``qpdk.lvs`` using the layer and device definitions from :mod:`qpdk.tech`.

Run directly to regenerate the deck::

    uv run python qpdk/klayout/lvs/render_lvs.py
"""

from __future__ import annotations

import pathlib
import re

from jinja2 import Environment, FileSystemLoader

_HERE = pathlib.Path(__file__).parent


def _layer_entry(
    var: str,
    gds_layer: int,
    gds_datatype: int,
    comment: str,
    **extra: object,
) -> dict:
    """Build a layer-definition dict for the Jinja2 template."""
    return {
        "var": var,
        "gds_layer": gds_layer,
        "gds_datatype": gds_datatype,
        "comment": comment,
        **extra,
    }


def _connection(
    layer_a: str,
    via: str,
    layer_b: str,
    comment: str,
) -> dict:
    """Build a connectivity-rule dict for the Jinja2 template."""
    return {"layer_a": layer_a, "via": via, "layer_b": layer_b, "comment": comment}


def _device(
    var: str,
    description: str,
    physical_layer: str,
    marker_var: str,
    class_name: str,
    model_name: str,
    terminals: list[dict],
    *,
    sheet_rho: float = 0.0,
) -> dict:
    """Build a device-extraction dict for the Jinja2 template."""
    return {
        "var": var,
        "description": description,
        "physical_layer": physical_layer,
        "marker_var": marker_var,
        "class_name": class_name,
        "model_name": model_name,
        "terminals": terminals,
        "sheet_rho": sheet_rho,
    }


# ── Layer tables ────────────────────────────────────────────────────────────

PHYSICAL_LAYERS = [
    _layer_entry("m1_draw", 1, 0, "M1 additive metal"),
    _layer_entry("m1_etch", 1, 1, "M1 subtractive etch"),
    _layer_entry("m2_draw", 2, 0, "M2 additive metal (flip-chip)"),
    _layer_entry("m2_etch", 2, 1, "M2 subtractive etch"),
    _layer_entry("ab_draw", 10, 0, "Airbridge metal"),
    _layer_entry("ab_via", 10, 1, "Airbridge landing pads"),
    _layer_entry("jj_area", 20, 0, "Josephson junction area"),
    _layer_entry("jj_patch", 20, 1, "Josephson junction patch"),
    _layer_entry("ind", 30, 0, "Indium bumps"),
    _layer_entry("tsv", 31, 0, "Through-silicon vias"),
]

PORT_LAYERS = [
    _layer_entry(
        "port_m1",
        1,
        10,
        "Port shapes on M1",
        metal_var="m1_draw",
        connect_comment="Ports connect to M1 metal",
    ),
    _layer_entry(
        "port_m2",
        2,
        10,
        "Port shapes on M2",
        metal_var="m2_draw",
        connect_comment="Ports connect to M2 metal",
    ),
]

MARKER_LAYERS = [
    _layer_entry("mk_transmon", 200, 0, "Transmon qubit marker"),
    _layer_entry("mk_resonator", 201, 0, "Resonator marker"),
    _layer_entry("mk_inductor", 202, 0, "Inductor marker"),
    _layer_entry("mk_capacitor", 203, 0, "Capacitor marker"),
    _layer_entry("mk_jj", 204, 0, "Josephson junction marker"),
]

# ── Connectivity ────────────────────────────────────────────────────────────

CONNECTIONS = [
    _connection("m1_draw", "tsv", "m2_draw", "M1 ↔ TSV ↔ M2"),
    _connection("m1_draw", "ind", "m2_draw", "M1 ↔ indium bump ↔ M2"),
    _connection("m1_draw", "ab_draw", "m1_draw", "M1 ↔ airbridge ↔ M1"),
    _connection("m2_draw", "ab_draw", "m2_draw", "M2 ↔ airbridge ↔ M2"),
]

# ── Devices ─────────────────────────────────────────────────────────────────

DEVICES = [
    _device(
        var="transmon",
        description="Transmon qubit",
        physical_layer="m1_draw",
        marker_var="mk_transmon",
        class_name="SubCircuit",
        model_name="TRANSMON",
        terminals=[
            {"name": "pad_a", "layer": "port_m1"},
            {"name": "pad_b", "layer": "port_m1"},
        ],
    ),
    _device(
        var="resonator",
        description="Coplanar waveguide resonator",
        physical_layer="m1_draw",
        marker_var="mk_resonator",
        class_name="SubCircuit",
        model_name="RESONATOR",
        terminals=[
            {"name": "input", "layer": "port_m1"},
            {"name": "output", "layer": "port_m1"},
        ],
    ),
    _device(
        var="inductor",
        description="Meander inductor",
        physical_layer="m1_draw",
        marker_var="mk_inductor",
        class_name="RBA::Resistor",
        model_name="INDUCTOR",
        terminals=[
            {"name": "tA", "layer": "port_m1"},
            {"name": "tB", "layer": "port_m1"},
        ],
        sheet_rho=0.0,
    ),
    _device(
        var="capacitor",
        description="Interdigital capacitor",
        physical_layer="m1_draw",
        marker_var="mk_capacitor",
        class_name="RBA::Capacitor",
        model_name="CAPACITOR",
        terminals=[],
    ),
    _device(
        var="jj",
        description="Josephson junction",
        physical_layer="jj_area",
        marker_var="mk_jj",
        class_name="SubCircuit",
        model_name="JJ",
        terminals=[
            {"name": "top", "layer": "port_m1"},
            {"name": "bottom", "layer": "port_m1"},
        ],
    ),
]


def render(*, write: bool = True) -> str:
    """Render the LVS deck from the Jinja2 template.

    Args:
        write: If ``True``, write the result to ``qpdk.lvs`` next to this script.

    Returns:
        The rendered LVS deck as a string.
    """
    env = Environment(
        loader=FileSystemLoader(_HERE),
        autoescape=False,  # noqa: S701 — LVS deck is not HTML; escaping would break Ruby syntax
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("qpdk.lvs.j2")
    rendered = template.render(
        physical_layers=PHYSICAL_LAYERS,
        port_layers=PORT_LAYERS,
        marker_layers=MARKER_LAYERS,
        connections=CONNECTIONS,
        devices=DEVICES,
    )

    # Collapse runs of 3+ blank lines to 2
    rendered = re.sub(r"\n{4,}", "\n\n\n", rendered)

    if write:
        out = _HERE / "qpdk.lvs"
        out.write_text(rendered)
        print(f"Wrote '{out}'")  # noqa: T201

    return rendered


if __name__ == "__main__":
    render()
