"""Reusable schematic factory for qpdk cells, linked to SAX models.

Mirrors gdsfactory's ``gpdk/_schematic.py`` port-pattern approach and
IHP's ``s.info["models"]`` SPICE-link pattern, but carries SAX model
references instead of SPICE.
"""

from __future__ import annotations

from kfactory.schematic import DSchematic

__all__ = [
    "bend_circular_schematic",
    "double_pad_transmon_schematic",
    "launcher_schematic",
    "lumped_element_resonator_schematic",
    "meander_inductor_schematic",
    "quarter_wave_resonator_coupled_schematic",
    "resonator_coupled_schematic",
    "resonator_half_wave_schematic",
    "resonator_quarter_wave_schematic",
    "resonator_schematic",
    "sax_model",
    "schematic",
    "straight_schematic",
]

# ---------------------------------------------------------------------------
# Port patterns
# ---------------------------------------------------------------------------

# 2-port horizontal (straight, taper, transition, resonator)
_2PORT = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
]

# 1-port (shorted/open resonator end, etc.)
_1PORT = [
    {"name": "o1", "side": "left", "type": "photonic"},
]

# 3-port (couplers, tees)
_3PORT = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "top", "type": "photonic"},
]

# Coupled resonator model ports. The resonator endpoint is intentionally
# retained in the SAX model boundary even though it is not a chip-level port.
_RESONATOR_COUPLED = [
    {"name": "coupling_o1", "side": "left", "type": "photonic"},
    {"name": "coupling_o2", "side": "right", "type": "photonic"},
    {"name": "resonator_o1", "side": "top", "type": "photonic"},
    {"name": "resonator_o2", "side": "bottom", "type": "photonic"},
]

_QUARTER_WAVE_RESONATOR_COUPLED = _RESONATOR_COUPLED[:3]

# Transmon qubit
_TRANSMON = [
    {"name": "left_pad", "side": "left", "type": "photonic"},
    {"name": "right_pad", "side": "right", "type": "photonic"},
]

# ---------------------------------------------------------------------------
# Schematic builder
# ---------------------------------------------------------------------------

_SIDE_XY = {
    "left": (-1, 0, 180),
    "right": (1, 0, 0),
    "top": (0, 1, 90),
    "bottom": (0, -1, 270),
}


def _make_schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict],
    models: list[dict] | None,
) -> DSchematic:
    """Build a DSchematic from port patterns.

    Args:
        symbol: Name of the symbol.
        tags: List of tags for the symbol.
        ports: List of port definitions.
        models: List of model definitions.

    Returns:
        DSchematic: The built schematic.

    Raises:
        ValueError: If a port has an unknown side.
    """
    # Deep-copy ports and models: both are lists of dicts shared across all
    # calls via module-level constants and closure variables.
    s = DSchematic()
    s.info["symbol"] = symbol
    s.info["tags"] = list(tags)
    s.info["ports"] = [dict(p) for p in ports]
    s.info["models"] = [dict(m) for m in models or []]

    side_counts: dict[str, int] = {}
    for port in ports:
        side_counts[port["side"]] = side_counts.get(port["side"], 0) + 1

    seen_sides: dict[str, int] = {}
    spacing = 0.5
    for port in ports:
        side = port["side"]
        try:
            bx, by, orientation = _SIDE_XY[side]
        except KeyError as exc:
            raise ValueError(
                f"schematic {symbol!r} port {port['name']!r}: unknown side "
                f"{side!r} (expected one of {sorted(_SIDE_XY)})"
            ) from exc
        idx = seen_sides.get(side, 0)
        seen_sides[side] = idx + 1
        total = side_counts[side]
        offset = (idx - (total - 1) / 2) * spacing
        if side in {"left", "right"}:
            x, y = bx, by + offset
        else:
            x, y = bx + offset, by

        port_type = port.get("type", "photonic")
        xs = "metal_routing" if port_type in {"electric", "electrical"} else "strip"
        s.create_port(
            name=port["name"],
            cross_section=xs,
            x=x,
            y=y,
            orientation=orientation,
        )

    return s


def schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict],
    models: list[dict] | None = None,
):
    """Return a ``schematic_function`` closure for use with ``@gf.cell``."""

    def _schematic_fn(**_kwargs) -> DSchematic:
        return _make_schematic(symbol, tags, ports, models)

    return _schematic_fn


def sax_model(
    name: str,
    module: str,
    port_order: list[str],
    qualname: str | None = None,
    params: dict[str, str] | None = None,
) -> dict:
    """Build a SAX model entry for ``s.info["models"]``."""
    return {
        "language": "sax",
        "name": name,
        "module": module,
        "qualname": qualname or name,
        "port_order": list(port_order),
        "params": dict(params) if params else {},
    }


# ---------------------------------------------------------------------------
# Pre-defined schematic functions
# ---------------------------------------------------------------------------

straight_schematic = schematic(
    symbol="straight",
    tags=["waveguides"],
    ports=_2PORT,
)

bend_circular_schematic = schematic(
    symbol="bend_circular",
    tags=["waveguides"],
    ports=_2PORT,
)

resonator_schematic = schematic(
    symbol="resonator",
    tags=["resonators"],
    ports=_2PORT,
    models=[
        sax_model(
            name="resonator",
            module="qpdk.models.resonator",
            port_order=["o1", "o2"],
        )
    ],
)

resonator_half_wave_schematic = schematic(
    symbol="resonator_half_wave",
    tags=["resonators"],
    ports=_2PORT,
    models=[
        sax_model(
            name="resonator_half_wave",
            module="qpdk.models.resonator",
            port_order=["o1", "o2"],
        )
    ],
)

resonator_quarter_wave_schematic = schematic(
    symbol="resonator_quarter_wave",
    tags=["resonators"],
    ports=_2PORT,
    models=[
        sax_model(
            name="resonator_quarter_wave",
            module="qpdk.models.resonator",
            port_order=["o1", "o2"],
        )
    ],
)

resonator_coupled_schematic = schematic(
    symbol="resonator_coupled",
    tags=["resonators", "couplers"],
    ports=_RESONATOR_COUPLED,
    models=[
        sax_model(
            name="resonator_coupled",
            module="qpdk.models.resonator",
            port_order=[
                "coupling_o1",
                "coupling_o2",
                "resonator_o1",
                "resonator_o2",
            ],
        )
    ],
)

quarter_wave_resonator_coupled_schematic = schematic(
    symbol="quarter_wave_resonator_coupled",
    tags=["resonators", "couplers"],
    ports=_QUARTER_WAVE_RESONATOR_COUPLED,
    models=[
        sax_model(
            name="quarter_wave_resonator_coupled",
            module="qpdk.models.resonator",
            port_order=["coupling_o1", "coupling_o2", "resonator_o1"],
        )
    ],
)

double_pad_transmon_schematic = schematic(
    symbol="double_pad_transmon",
    tags=["qubits", "transmons"],
    ports=_TRANSMON,
)

meander_inductor_schematic = schematic(
    symbol="meander_inductor",
    tags=["inductors"],
    ports=_2PORT,
)

lumped_element_resonator_schematic = schematic(
    symbol="lumped_element_resonator",
    tags=["resonators", "inductors", "capacitors"],
    ports=_2PORT,
    models=[
        sax_model(
            name="lumped_element_resonator",
            module="qpdk.models.inductor",
            port_order=["o1", "o2"],
        )
    ],
)

launcher_schematic = schematic(
    symbol="launcher",
    tags=["waveguides", "interconnects"],
    ports=[
        {"name": "waveport", "side": "left", "type": "photonic"},
        {"name": "o1", "side": "right", "type": "photonic"},
    ],
    models=[
        sax_model(
            name="launcher",
            module="qpdk.models.waveguides",
            port_order=["waveport", "o1"],
        )
    ],
)
