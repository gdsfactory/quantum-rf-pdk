"""Schematic metadata used mainly by gdsfactoryplus to link cells to SAX models."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Any

from kfactory.schematic import DSchematic

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

__all__ = [
    "airbridge_schematic",
    "bend_circular_all_angle_schematic",
    "bend_circular_schematic",
    "bend_euler_all_angle_schematic",
    "bend_euler_schematic",
    "bend_s_schematic",
    "bump_transition_schematic",
    "coupler_ring_schematic",
    "coupler_straight_schematic",
    "double_pad_transmon_schematic",
    "double_pad_transmon_with_bbox_schematic",
    "flipmon_schematic",
    "fluxonium_schematic",
    "fluxonium_with_bbox_schematic",
    "gatemon_coupler_schematic",
    "gatemon_schematic",
    "half_circle_coupler_schematic",
    "indium_bump_schematic",
    "interdigital_capacitor_schematic",
    "josephson_junction_schematic",
    "launcher_schematic",
    "lumped_element_resonator_schematic",
    "meander_inductor_schematic",
    "nxn_schematic",
    "open_2_port_schematic",
    "open_schematic",
    "plate_capacitor_schematic",
    "plate_capacitor_single_schematic",
    "quarter_wave_resonator_coupled_schematic",
    "resonator_coupled_schematic",
    "resonator_half_wave_schematic",
    "resonator_meander_schematic",
    "resonator_quarter_wave_schematic",
    "resonator_schematic",
    "sax_model",
    "schematic",
    "short_schematic",
    "snspd_schematic",
    "squid_junction_schematic",
    "straight_all_angle_schematic",
    "straight_open_schematic",
    "straight_schematic",
    "straight_shorted_schematic",
    "taper_cross_section_schematic",
    "tee_schematic",
    "tsv_schematic",
    "tsv_transition_double_sided_schematic",
    "unimon_arm_schematic",
    "unimon_coupled_schematic",
    "unimon_schematic",
    "xmon_transmon_schematic",
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

# Coupled resonator model ports. The resonator endpoint is intentionally
# retained in the SAX model boundary even though it is not a chip-level port.
# The coupling line runs above the resonator, matching the layout cell
# (``resonator_o1`` faces west under ``coupling_o1``, ``resonator_o2`` east).
_RESONATOR_COUPLED = [
    {"name": "coupling_o1", "side": "left", "type": "photonic"},
    {"name": "resonator_o1", "side": "left", "type": "photonic"},
    {"name": "coupling_o2", "side": "right", "type": "photonic"},
    {"name": "resonator_o2", "side": "right", "type": "photonic"},
]

_COUPLER_STRAIGHT = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "left", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# The resonator end shares the left side with the coupling input, below it,
# matching the layout cell (``resonator_o1`` faces west, under ``coupling_o1``).
_QUARTER_WAVE_RESONATOR_COUPLED = [
    {"name": "coupling_o1", "side": "left", "type": "photonic"},
    {"name": "resonator_o1", "side": "left", "type": "photonic"},
    {"name": "coupling_o2", "side": "right", "type": "photonic"},
]

# 90° bend: ``o1`` faces west, ``o2`` north, as in the layout cell.
_BEND = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
]

# Transmon qubit
_TRANSMON = [
    {"name": "left_pad", "side": "left", "type": "photonic"},
    {"name": "right_pad", "side": "right", "type": "photonic"},
]

# Josephson junction
_JUNCTION = [
    {"name": "left_wide", "side": "left", "type": "photonic"},
    {"name": "right_wide", "side": "right", "type": "photonic"},
]

# Unimon readout coupling port
_UNIMON_COUPLED = [
    {"name": "coupling_o3", "side": "left", "type": "photonic"},
]

# Ring coupler: the through line runs left to right, both ring ends point up.
_COUPLER_RING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
    {"name": "o3", "side": "top", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# Xmon transmon: one arm per side, matching the layout port orientations.
_XMON = [
    {"name": "top_arm", "side": "top", "type": "photonic"},
    {"name": "right_arm", "side": "right", "type": "photonic"},
    {"name": "bottom_arm", "side": "bottom", "type": "photonic"},
    {"name": "left_arm", "side": "left", "type": "photonic"},
]

# SQUID loop terminals. Each junction contributes one axial and one lateral
# terminal, so the four wide ends sit one per side as in the layout cell.
_SQUID = [
    {"name": "left_left_wide", "side": "bottom", "type": "photonic"},
    {"name": "left_right_wide", "side": "left", "type": "photonic"},
    {"name": "right_left_wide", "side": "top", "type": "photonic"},
    {"name": "right_right_wide", "side": "right", "type": "photonic"},
]

_SNSPD = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "right", "type": "electric"},
]


# ---------------------------------------------------------------------------
# Schematic builder
# ---------------------------------------------------------------------------

_SYMBOLS_PACKAGE = "qpdk.symbols"
_SYMBOLS = files(_SYMBOLS_PACKAGE)

_SIDE_XY: dict[str, tuple[int, int, Literal[0, 90, 180, 270]]] = {
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

    GDSFactory+ draws artwork only for the generic ``"ckt"`` device type, so
    every schematic declares it. Packaged artwork in ``qpdk/symbols`` named
    ``<symbol>.svg`` is attached through ``display.symbol``; without one,
    GDSFactory+ renders the layout geometry instead.

    Args:
        symbol: Name of the symbol, matching ``qpdk/symbols/<symbol>.svg``.
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
    s.info["symbol"] = "ckt"
    if (_SYMBOLS / f"{symbol}.svg").is_file():
        s.info["display"] = {"symbol": f"{_SYMBOLS_PACKAGE}:{symbol}.svg"}
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
        # Same-side ports run top-to-bottom and left-to-right in list order,
        # the order the GDSFactory+ schematic editor lays them out in.
        if side in {"left", "right"}:
            x, y = bx, by - offset
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
) -> Callable[..., DSchematic]:
    """Return a ``schematic_function`` closure for use with ``@gf.cell``."""

    def _schematic_fn(**_kwargs: Any) -> DSchematic:
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
    models=[
        sax_model(
            name="straight",
            module="qpdk.models.waveguides",
            port_order=["o1", "o2"],
        )
    ],
)

coupler_straight_schematic = schematic(
    symbol="coupler_straight",
    tags=["waveguides", "couplers"],
    ports=_COUPLER_STRAIGHT,
    models=[
        sax_model(
            name="coupler_straight",
            module="qpdk.models.couplers",
            port_order=["o1", "o2", "o3", "o4"],
        )
    ],
)

open_schematic = schematic(
    symbol="open",
    tags=["terminations"],
    ports=_1PORT,
    models=[
        sax_model(
            name="open",
            module="qpdk.models.generic",
            port_order=["o1"],
        )
    ],
)

open_2_port_schematic = schematic(
    symbol="open_2_port",
    tags=["terminations"],
    ports=[
        {"name": "o1", "side": "right", "type": "photonic"},
        {"name": "o2", "side": "left", "type": "photonic"},
    ],
    models=[
        sax_model(
            name="open_2_port",
            module="qpdk.models.generic",
            port_order=["o1", "o2"],
        )
    ],
)

short_schematic = schematic(
    symbol="short",
    tags=["terminations"],
    ports=_1PORT,
    models=[
        sax_model(
            name="short",
            module="qpdk.models.generic",
            port_order=["o1"],
        )
    ],
)

straight_open_schematic = schematic(
    symbol="straight_open",
    tags=["waveguides", "resonators", "terminations"],
    ports=_1PORT,
    models=[
        sax_model(
            name="straight_open",
            module="qpdk.models.waveguides",
            port_order=["o1"],
        )
    ],
)

straight_shorted_schematic = schematic(
    symbol="straight_shorted",
    tags=["waveguides", "resonators", "terminations"],
    ports=_1PORT,
    models=[
        sax_model(
            name="straight_shorted",
            module="qpdk.models.waveguides",
            port_order=["o1"],
        )
    ],
)

bend_circular_schematic = schematic(
    symbol="bend_circular",
    tags=["waveguides"],
    ports=_BEND,
    models=[sax_model("bend_circular", "qpdk.models.waveguides", ["o1", "o2"])],
)

bend_s_schematic = schematic(
    symbol="bend_s",
    tags=["waveguides"],
    ports=_2PORT,
    models=[
        sax_model(
            name="bend_s",
            module="qpdk.models.waveguides",
            port_order=["o1", "o2"],
        )
    ],
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

resonator_meander_schematic = schematic(
    symbol="resonator",
    tags=["resonators"],
    ports=_2PORT,
    models=[
        sax_model(
            name="resonator_meander",
            module="qpdk.models.resonator_sections",
            port_order=["o1", "o2"],
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

gatemon_schematic = schematic(
    symbol="gatemon",
    tags=["qubits", "gatemons"],
    ports=[
        {"name": "readout", "side": "top", "type": "photonic", "aliases": ["RF"]},
        {"name": "gate", "side": "left", "type": "photonic", "aliases": ["Vg"]},
    ],
    models=[sax_model("gatemon_island", "qpdk.models.gatemon", ["readout", "gate"])],
)

gatemon_coupler_schematic = schematic(
    symbol="gatemon_coupler",
    tags=["couplers", "gatemons"],
    ports=[*_2PORT, {"name": "qubit", "side": "bottom", "type": "photonic"}],
    models=[sax_model("gatemon_coupler", "qpdk.models.gatemon", ["o1", "o2", "qubit"])],
)

interdigital_capacitor_schematic = schematic(
    symbol="interdigital_capacitor",
    tags=["capacitors"],
    ports=_2PORT,
    models=[sax_model("interdigital_capacitor", "qpdk.models.capacitor", ["o1", "o2"])],
)

double_pad_transmon_schematic = schematic(
    symbol="double_pad_transmon",
    tags=["qubits", "transmons"],
    ports=_TRANSMON,
    models=[
        sax_model(
            name="double_pad_transmon",
            module="qpdk.models.qubit",
            port_order=["left_pad", "right_pad"],
        )
    ],
)

double_pad_transmon_with_bbox_schematic = schematic(
    symbol="double_pad_transmon_with_bbox",
    tags=["qubits", "transmons"],
    ports=_TRANSMON,
    models=[
        sax_model(
            name="double_pad_transmon_with_bbox",
            module="qpdk.models.qubit",
            port_order=["left_pad", "right_pad"],
        )
    ],
)

fluxonium_schematic = schematic(
    symbol="fluxonium",
    tags=["qubits", "inductors"],
    ports=_TRANSMON,
    models=[
        sax_model(
            name="fluxonium",
            module="qpdk.models.qubit",
            port_order=["left_pad", "right_pad"],
        )
    ],
)

fluxonium_with_bbox_schematic = schematic(
    symbol="fluxonium_with_bbox",
    tags=["qubits", "inductors"],
    ports=_TRANSMON,
    models=[
        sax_model(
            name="fluxonium_with_bbox",
            module="qpdk.models.qubit",
            port_order=["left_pad", "right_pad"],
        )
    ],
)

josephson_junction_schematic = schematic(
    symbol="josephson_junction",
    tags=["junctions"],
    ports=_JUNCTION,
    models=[
        sax_model(
            name="josephson_junction",
            module="qpdk.models.junction",
            port_order=["left_wide", "right_wide"],
        )
    ],
)

unimon_coupled_schematic = schematic(
    symbol="unimon_coupled",
    tags=["qubits", "couplers"],
    ports=_UNIMON_COUPLED,
    models=[
        sax_model(
            name="unimon_coupled",
            module="qpdk.models.unimon",
            port_order=["coupling_o3"],
        )
    ],
)

meander_inductor_schematic = schematic(
    symbol="meander_inductor",
    tags=["inductors"],
    ports=_2PORT,
    models=[sax_model("meander_inductor", "qpdk.models.inductor", ["o1", "o2"])],
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

plate_capacitor_single_schematic = schematic(
    symbol="plate_capacitor_single",
    tags=["capacitors"],
    ports=_1PORT,
)

half_circle_coupler_schematic = schematic(
    symbol="half_circle_coupler",
    tags=["capacitors", "couplers"],
    ports=[{"name": "o3", "side": "bottom", "type": "photonic"}],
)

snspd_schematic = schematic(
    symbol="snspd",
    tags=["detectors"],
    ports=_SNSPD,
)

squid_junction_schematic = schematic(
    symbol="squid_junction",
    tags=["junctions"],
    ports=_SQUID,
)

xmon_transmon_schematic = schematic(
    symbol="xmon_transmon",
    tags=["qubits", "transmons"],
    ports=_XMON,
)

flipmon_schematic = schematic(
    symbol="flipmon",
    tags=["qubits", "transmons"],
    ports=[{"name": "outer_ring_outside", "side": "right", "type": "photonic"}],
)

# The unimon's arm ends are placement markers, so the bare qubit has no circuit
# boundary of its own. Use ``unimon_coupled`` to connect it in a schematic.
unimon_schematic = schematic(
    symbol="unimon",
    tags=["qubits"],
    ports=[],
)

unimon_arm_schematic = schematic(
    symbol="resonator_quarter_wave",
    tags=["qubits", "resonators"],
    ports=_2PORT,
)

# The two pads are the terminals; the CPW passes underneath on the placement
# ports. Drawn horizontally to read as a two-terminal element, even though the
# layout cell stands the bridge across the CPW.
airbridge_schematic = schematic(
    symbol="airbridge",
    tags=["airbridges", "interconnects"],
    ports=[
        {"name": "e1", "side": "left", "type": "electric"},
        {"name": "e2", "side": "right", "type": "electric"},
    ],
)

# A via only carries a placement marker, so its symbol is artwork without ports.
tsv_schematic = schematic(
    symbol="tsv",
    tags=["interconnects"],
    ports=[],
)

indium_bump_schematic = schematic(
    symbol="indium_bump",
    tags=["interconnects"],
    ports=[],
)

# Through-chip and chip-to-chip CPW transitions: two-port interconnects whose
# ports sit on different metal layers.
tsv_transition_double_sided_schematic = schematic(
    symbol="tsv_transition_double_sided",
    tags=["interconnects", "waveguides"],
    ports=_2PORT,
)

bump_transition_schematic = schematic(
    symbol="bump_transition",
    tags=["interconnects", "waveguides"],
    ports=_2PORT,
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


bend_euler_schematic = schematic(
    symbol="bend_euler",
    tags=["waveguides"],
    ports=_BEND,
    models=[sax_model("bend_euler", "qpdk.models.waveguides", ["o1", "o2"])],
)


bend_circular_all_angle_schematic = schematic(
    symbol="bend_circular_all_angle",
    tags=["waveguides"],
    ports=_BEND,
    models=[
        sax_model("bend_circular_all_angle", "qpdk.models.waveguides", ["o1", "o2"])
    ],
)


bend_euler_all_angle_schematic = schematic(
    symbol="bend_euler_all_angle",
    tags=["waveguides"],
    ports=_BEND,
    models=[sax_model("bend_euler_all_angle", "qpdk.models.waveguides", ["o1", "o2"])],
)


straight_all_angle_schematic = schematic(
    symbol="straight_all_angle",
    tags=["waveguides"],
    ports=_2PORT,
    models=[sax_model("straight_all_angle", "qpdk.models.waveguides", ["o1", "o2"])],
)


taper_cross_section_schematic = schematic(
    symbol="taper_cross_section",
    tags=["waveguides"],
    ports=_2PORT,
    models=[sax_model("taper_cross_section", "qpdk.models.waveguides", ["o1", "o2"])],
)


coupler_ring_schematic = schematic(
    symbol="coupler_ring",
    tags=["waveguides", "couplers"],
    ports=_COUPLER_RING,
    models=[
        sax_model("coupler_ring", "qpdk.models.couplers", ["o1", "o2", "o3", "o4"])
    ],
)


plate_capacitor_schematic = schematic(
    symbol="plate_capacitor",
    tags=["capacitors", "couplers"],
    ports=_2PORT,
    models=[sax_model("plate_capacitor", "qpdk.models.capacitor", ["o1", "o2"])],
)


def nxn_schematic(
    west: int = 1,
    east: int = 1,
    north: int = 1,
    south: int = 1,
    **_kwargs: Any,
) -> DSchematic:
    """Describe the junction's variable port count without building geometry."""
    sides = ["left"] * west + ["top"] * north + ["right"] * east + ["bottom"] * south
    ports = [
        {"name": f"o{i}", "side": side, "type": "photonic"}
        for i, side in enumerate(sides, start=1)
    ]
    return _make_schematic(
        "nxn",
        ["waveguides"],
        ports,
        [sax_model("nxn", "qpdk.models.waveguides", [port["name"] for port in ports])],
    )


tee_schematic = schematic(
    symbol="tee",
    tags=["waveguides"],
    ports=[
        {"name": "o1", "side": "left", "type": "photonic"},
        {"name": "o2", "side": "top", "type": "photonic"},
        {"name": "o3", "side": "right", "type": "photonic"},
    ],
    models=[sax_model("tee", "qpdk.models.generic", ["o1", "o2", "o3"])],
)
