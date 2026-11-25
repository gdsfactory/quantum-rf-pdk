"""Small-signal SAX model of the probeline readout assembly.

The four-port :func:`~qpdk.models.couplers.coupler_straight` is the core. Ports
``o1`` and ``o4`` are the probeline ends ``coupling_o1`` and ``coupling_o2``,
``o2`` goes through the meander, and ``o3`` goes through the route to the qubit
coupling plate. The model uses uniform ``resonator_cross_section`` sections for
the route and meander. ``resonator_length`` includes the route plus coupling
span, so the meander is that length minus the span and the route is the span
minus ``probeline_coupling_length``. The meander far end is open when
``resonator_open_end`` is true and short otherwise. ``resonator_open_start``
normally keeps the meander connected. When true the model looks into an open at
``o2``.

The qubit load comes from ``transmon_readout_coupon_palace`` and
``flipmon_readout_coupon_palace`` at the nominal geometry of the shipped chips:
a 394 um by 20 um coupling plate offset 45 um for the double-pad, and a 58 um by
10 um plate offset 10 um for the flipmon. The coupon lookup always uses a 16 um
feedline gap and the nominal 800 um coupling length, even when the distributed
coupling length is 400 um, which keeps it inside the sampled 500 um to 800 um
domain. ``probeline_coupling_gap`` changes the distributed coupler only; over
16 um to 20 um the measured coupling changes by about 0.005 fF. The nominal
route plus coupling spans are 1742.5795478 um and
1900.0795478 um. The load keeps the island-to-island, island-to-readout and
island-to-ground capacitances. It drops the readout and feedline ground
capacitances and the readout-to-feedline mutual branch because the distributed
sections already contain them, and it drops the tiny feedline-to-island
branches. The flipmon top pad is bonded to ``island_a`` with an ideal wire, and
a 7 nH Josephson inductance joins the islands.

The model is linear and small-signal, with no driven FEM or measurement
calibration. It uses nominal qubit geometry at run time. Custom qubit geometry
needs another model, and a nonnominal route span needs an explicit
``route_and_coupler_length``. The optional ``resonator_meander_start``,
``coupler_offset``, ``coupler``, ``coupler_port`` and ``qubit_rotation``
settings are checked against the shipped layout, so a concrete nonnominal value
raises. An implicit span accepts only positive
coupling lengths up to 800 um; a longer coupling needs an explicit span. At
zero frequency the probeline is an ideal through line, with zero reflection and
unit transmission.
"""

from collections.abc import Sequence
from functools import cache, partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from jax.typing import ArrayLike
from kfactory.serialization import clean_value

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.transmon import flipmon_with_bbox
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.couplers import coupler_straight
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.datasets import Dataset, GridInterpolator, maxwell_to_mutual
from qpdk.models.jit import jit
from qpdk.models.waveguides import straight

type _QubitKind = Literal["double_pad", "flipmon"]
type _CouponCapacitances = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]
type _LayoutSetting = ArrayLike | Sequence[ArrayLike] | None

_NOMINAL_GAP_UM = 16.0
_NOMINAL_COUPLING_LENGTH_UM = 800.0
_FLIPMON_JUNCTION_SETTING = {"junction_position_rotation": 180}

# Dataset, nominal route plus coupling span, pad length and pad offset, in um.
_COUPON_GEOMETRY: dict[_QubitKind, tuple[str, float, float, float]] = {
    "double_pad": ("transmon_readout_coupon_palace", 1742.5795478, 394.0, 45.0),
    "flipmon": ("flipmon_readout_coupon_palace", 1900.0795478, 58.0, 10.0),
}

# Nominal layout settings of the shipped chips, in um.
_NOMINAL_MEANDER_START = (-900.0, -1200.0)
_NOMINAL_PLATE_OFFSET: dict[_QubitKind, tuple[float, float]] = {
    "double_pad": (-45.0, 0.0),
    "flipmon": (-10.0, 0.0),
}

_COUPON_GROUPS: dict[_QubitKind, tuple[tuple[str, ...], ...]] = {
    "double_pad": (("island_a",), ("island_b",), ("readout",), ("feedline",)),
    "flipmon": (
        ("island_a", "top_pad"),
        ("island_b",),
        ("readout",),
        ("feedline",),
    ),
}

_DOUBLE_PAD_NAMES = frozenset({"double_pad_transmon_with_bbox"})
_NOMINAL_COUPLER: dict[_QubitKind, partial[Any]] = {
    "double_pad": partial(
        plate_capacitor_single, width=20.0, length=_COUPON_GEOMETRY["double_pad"][2]
    ),
    "flipmon": partial(
        plate_capacitor_single, width=10.0, length=_COUPON_GEOMETRY["flipmon"][2]
    ),
}
_NOMINAL_COUPLER_STRING: dict[_QubitKind, str] = {
    "double_pad": clean_value(_NOMINAL_COUPLER["double_pad"]),
    "flipmon": clean_value(_NOMINAL_COUPLER["flipmon"]),
}
_NOMINAL_COUPLER_PORT: dict[_QubitKind, str] = {
    "double_pad": "left_pad",
    "flipmon": "outer_ring_outside",
}
_NOMINAL_QUBIT_ROTATION: dict[_QubitKind, float] = {
    "double_pad": 90.0,
    "flipmon": -90.0,
}
_FLIPMON_NAMES = frozenset({
    "flipmon_with_bbox",
    clean_value(partial(flipmon_with_bbox, **_FLIPMON_JUNCTION_SETTING)),
})


def _factory_name(spec: object) -> str:
    """Resolve a qubit component spec to a factory name."""
    if isinstance(spec, partial):
        if spec.args or spec.keywords not in ({}, _FLIPMON_JUNCTION_SETTING):
            raise ValueError(
                f"Unsupported qubit partial settings: {spec.args!r}, {spec.keywords!r}"
            )
        name = _factory_name(spec.func)
        if spec.keywords and name not in _FLIPMON_NAMES:
            raise ValueError(f"Unsupported qubit partial settings for {name!r}")
        return name
    if isinstance(spec, dict):
        name = spec.get("component") or spec.get("function")
        if name is None:
            raise TypeError("A qubit component dict needs a component or function name")
        settings = spec.get("settings") or {}
        if not isinstance(settings, dict):
            raise TypeError("A qubit component settings value must be a mapping")
        resolved = _factory_name(name)
        if settings and not (
            resolved in _FLIPMON_NAMES and settings == _FLIPMON_JUNCTION_SETTING
        ):
            raise ValueError(
                f"Unsupported qubit settings for {resolved!r}: {settings!r}"
            )
        return resolved
    if isinstance(spec, str):
        return spec
    if callable(spec):
        name = getattr(spec, "__name__", None)
        if name:
            return name
    raise TypeError(f"Unsupported qubit component spec {spec!r}")


def _qubit_kind(qubit: ComponentSpec) -> _QubitKind:
    """Map a known qubit factory or serialized identity to a nominal coupon."""
    name = _factory_name(qubit)
    if name in _DOUBLE_PAD_NAMES:
        return "double_pad"
    if name in _FLIPMON_NAMES:
        return "flipmon"
    raise ValueError(f"Qubit {name!r} has no nominal readout coupon")


def _layout_setting_matches(
    value: _LayoutSetting, nominal: tuple[float, float]
) -> jax.Array:
    """Return whether an optional (x, y) layout setting matches the nominal value."""
    if value is None:
        return jnp.asarray(True)
    actual = jnp.asarray(value, dtype=float)
    if actual.shape != (2,):
        raise ValueError(
            f"A layout setting must be a 2-tuple (x, y) in um, got shape {actual.shape}"
        )
    return jnp.all(actual == jnp.asarray(nominal, dtype=float))


def _plate_values_match(width: object, length: object, kind: _QubitKind) -> jax.Array:
    """Return whether a coupling plate has the nominal width and length."""
    nominal = _NOMINAL_COUPLER[kind].keywords
    try:
        width = jnp.asarray(width, dtype=float)
        length = jnp.asarray(length, dtype=float)
    except (TypeError, ValueError):
        return jnp.asarray(False)
    return jnp.all(width == nominal["width"]) & jnp.all(length == nominal["length"])


def _coupler_matches(value: ComponentSpec | None, kind: _QubitKind) -> jax.Array:
    """Return whether an optional coupling plate matches the nominal partial."""
    if value is None:
        return jnp.asarray(True)
    nominal = _NOMINAL_COUPLER[kind]
    if isinstance(value, partial):
        if (
            value.func is not nominal.func
            or value.args
            or set(value.keywords) != {"width", "length"}
        ):
            return jnp.asarray(False)
        return _plate_values_match(
            value.keywords["width"], value.keywords["length"], kind
        )
    if isinstance(value, dict):
        function = value.get("function") or value.get("component")
        module = value.get("module")
        settings = value.get("settings")
        if (
            function != nominal.func.__name__
            or (module is not None and module != nominal.func.__module__)
            or not isinstance(settings, dict)
            or set(settings) != {"width", "length"}
        ):
            return jnp.asarray(False)
        return _plate_values_match(settings["width"], settings["length"], kind)
    if isinstance(value, str):
        return jnp.asarray(value == _NOMINAL_COUPLER_STRING[kind])
    try:
        return jnp.asarray(clean_value(value) == _NOMINAL_COUPLER_STRING[kind])
    except (TypeError, ValueError):
        return jnp.asarray(False)


def _port_matches(value: str | None, nominal: str) -> jax.Array:
    """Return whether an optional coupler port matches the nominal port."""
    if value is None:
        return jnp.asarray(True)
    return jnp.asarray(isinstance(value, str) and value == nominal)


def _rotation_matches(value: ArrayLike | None, nominal: float) -> jax.Array:
    """Return whether an optional qubit rotation matches the nominal angle."""
    if value is None:
        return jnp.asarray(True)
    return jnp.all(jnp.asarray(value, dtype=float) == nominal)


@cache
def _invalid_message(kind: _QubitKind) -> str:
    """Return the shared eager and JIT diagnostic for a nonnominal assembly."""
    plate = _NOMINAL_COUPLER[kind].keywords
    return (
        "This model supports only the nominal shipped readout assembly: "
        f"resonator_meander_start {_NOMINAL_MEANDER_START}, "
        f"coupler_offset {_NOMINAL_PLATE_OFFSET[kind]}, "
        f"coupler_port {_NOMINAL_COUPLER_PORT[kind]!r}, "
        f"qubit_rotation {_NOMINAL_QUBIT_ROTATION[kind]:g}, "
        f"coupler plate width {plate['width']:g} um and length "
        f"{plate['length']:g} um, positive route and meander sections, and "
        "probeline_coupling_length at most 800 um unless "
        "route_and_coupler_length is explicit."
    )


def _check_valid(valid: jax.Array, message: str) -> jax.Array:
    """Return the reduced validity flag, raising on a nonnominal assembly."""
    valid = jnp.all(valid)
    if not isinstance(valid, jax.core.Tracer):
        if not bool(valid):
            raise ValueError(message)
        return valid

    def _raise(flag: ArrayLike) -> None:
        """Raise when the callback flag marks every batched element invalid."""
        flag = np.asarray(flag)
        if bool(np.all(flag)):
            raise ValueError(message)

    def _invalid() -> jax.Array:
        """Report a nonnominal assembly on the host from the invalid cond branch."""
        jax.debug.callback(_raise, ~valid)
        return jnp.asarray(0)

    jax.lax.cond(valid, lambda: jnp.asarray(0), _invalid)
    return valid


def _validate_layout(
    *,
    kind: _QubitKind,
    resonator_meander_start: _LayoutSetting,
    coupler_offset: _LayoutSetting,
    coupler: ComponentSpec | None,
    coupler_port: str | None,
    qubit_rotation: ArrayLike | None,
    resonator_length: jax.Array,
    coupling_length: jax.Array,
    span: jax.Array,
    explicit_span: bool,
) -> jax.Array:
    """Return the validity flag, raising for a nonnominal assembly."""
    valid = (
        _layout_setting_matches(resonator_meander_start, _NOMINAL_MEANDER_START)
        & _layout_setting_matches(coupler_offset, _NOMINAL_PLATE_OFFSET[kind])
        & _coupler_matches(coupler, kind)
        & _port_matches(coupler_port, _NOMINAL_COUPLER_PORT[kind])
        & _rotation_matches(qubit_rotation, _NOMINAL_QUBIT_ROTATION[kind])
        & (coupling_length > 0)
        & (span > coupling_length)
        & (resonator_length > span)
    )
    if not explicit_span:
        valid = valid & (coupling_length <= _NOMINAL_COUPLING_LENGTH_UM)
    return _check_valid(valid, _invalid_message(kind))


@cache
def _coupon_capacitances(kind: _QubitKind) -> _CouponCapacitances:
    """Reduce one nominal coupon to island ground and mutual capacitances in F."""
    dataset, _, pad_length, pad_offset = _COUPON_GEOMETRY[kind]
    grid = Dataset(dataset).grid("maxwell_capacitance", allow_missing=False)
    groups = _COUPON_GROUPS[kind]
    if set(grid.terminals) != {name for group in groups for name in group}:
        raise ValueError(f"Coupon terminals {grid.terminals} do not match {groups}")
    maxwell = GridInterpolator(grid)(
        pad_length=pad_length,
        pad_offset=pad_offset,
        feedline_gap=_NOMINAL_GAP_UM,
        coupling_length=_NOMINAL_COUPLING_LENGTH_UM,
    )
    connection = jnp.asarray([
        [float(name in group) for group in groups] for name in grid.terminals
    ])
    mutual = maxwell_to_mutual(connection.T @ maxwell @ connection)
    return mutual[0, 0], mutual[1, 1], mutual[0, 1], mutual[0, 2], mutual[1, 2]


@jit(inline=True)
def _qubit_reflection(
    f: ArrayLike,
    z0: ArrayLike,
    capacitances: _CouponCapacitances,
    *,
    inductance: ArrayLike,
    junction_capacitance: ArrayLike,
) -> jax.Array:
    """Return the reflection coefficient of the reduced qubit load at the readout node."""
    c_ag, c_bg, c_ab, c_ar, c_br = capacitances
    p = 1j * 2 * jnp.pi * jnp.asarray(f)
    a = c_ag + c_ab + c_ar
    b = c_bg + c_ab + c_br
    readout = c_ar + c_br
    ground_sum = a + b - 2 * c_ab
    determinant = a * b - c_ab**2
    coupling = c_ar**2 * b + 2 * c_ar * c_br * c_ab + c_br**2 * a
    denominator = ground_sum + p**2 * inductance * (
        determinant + junction_capacitance * ground_sum
    )
    numerator = p * readout * denominator - (
        p**3 * inductance * coupling
        + p * (1 + p**2 * inductance * junction_capacitance) * readout**2
    )
    # One fraction keeps the f=0 limit finite and avoids inf/inf at a load pole.
    return jnp.asarray((denominator - z0 * numerator) / (denominator + z0 * numerator))


def transmon_with_resonator_and_probeline(
    f: ArrayLike = DEFAULT_FREQUENCY,
    resonator_length: ArrayLike = 5000.0,
    probeline_coupling_gap: ArrayLike = _NOMINAL_GAP_UM,
    probeline_coupling_length: ArrayLike | None = None,
    resonator_cross_section: CrossSectionSpec = "cpw",
    resonator_open_start: bool | ArrayLike = False,
    resonator_open_end: bool | ArrayLike = True,
    qubit: ComponentSpec = "double_pad_transmon_with_bbox",
    route_and_coupler_length: ArrayLike | None = None,
    qubit_inductance: ArrayLike = 7e-9,
    junction_capacitance: ArrayLike = 0.0,
    resonator_meander_start: _LayoutSetting = None,
    coupler_offset: _LayoutSetting = None,
    coupler: ComponentSpec | None = None,
    coupler_port: str | None = None,
    qubit_rotation: ArrayLike | None = None,
) -> sax.SDict:
    """Return the two-port probeline response of the readout assembly.

    ``qubit`` selects the nominal coupon. The shipped flipmon artifacts pass the
    saved serialized ``flipmon_with_bbox`` partial or callable. The double-pad
    cell passes ``"double_pad_transmon_with_bbox"``. Other factories raise.

    Args:
        f: Frequency in Hz. Zero returns the exact through-line response.
        resonator_length: Total length in um, including the route and coupling span.
        probeline_coupling_gap: Coupler gap in um. It changes the distributed
            coupler only; the coupon lookup stays at 16 um.
        probeline_coupling_length: Coupling length in um. None uses 800 um. Without
            an explicit ``route_and_coupler_length`` it must be positive and at most
            800 um.
        resonator_cross_section: Cross-section of the coupling section, route and meander.
        resonator_open_start: Cut the meander at the coupling section when true.
        resonator_open_end: Open the meander far end when true, short it when false.
        qubit: Qubit component spec.
        route_and_coupler_length: Route plus coupling span in um. None uses the qubit
            default. An explicit span is required for a coupling length above 800 um.
            The nominal coupon geometry still applies.
        qubit_inductance: Josephson inductance in H.
        junction_capacitance: Additional junction capacitance in F.
        resonator_meander_start: Meander start (x, y) in um. None skips the check,
            for model-only calls. A concrete value must be ``(-900, -1200)``.
        coupler_offset: Coupling plate offset (x, y) in um. None skips the check,
            for model-only calls. A concrete value must be ``(-45, 0)`` for the
            double-pad or ``(-10, 0)`` for the flipmon.
        coupler: Coupling plate spec. None skips the check, for model-only calls.
            A concrete value must be the nominal plate partial, 20 um by 394 um for
            the double-pad or 10 um by 58 um for the flipmon.
        coupler_port: Qubit port the plate attaches to. None skips the check. A
            concrete value must be ``"left_pad"`` or ``"outer_ring_outside"``.
        qubit_rotation: Qubit rotation in degrees. None skips the check. A concrete
            value must be ``90`` for the double-pad or ``-90`` for the flipmon.

    Returns:
        Two-port S-parameters at ``coupling_o1`` and ``coupling_o2``.
    """
    kind = _qubit_kind(qubit)
    _, span_default, _, _ = _COUPON_GEOMETRY[kind]
    with jax.ensure_compile_time_eval():
        capacitances = _coupon_capacitances(kind)

    resonator_length = jnp.asarray(resonator_length, dtype=float)
    f = jnp.asarray(f, dtype=float)
    f = jnp.broadcast_to(f, jnp.broadcast_shapes(f.shape, resonator_length.shape))
    # The isolated resonator solves singular at DC, so evaluate there at the
    # default frequency and select the exact through-line response at the end.
    f_eval = jnp.where(f == 0, DEFAULT_FREQUENCY, f)
    coupling_length = jnp.asarray(
        _NOMINAL_COUPLING_LENGTH_UM
        if probeline_coupling_length is None
        else probeline_coupling_length,
        dtype=float,
    )
    explicit_span = route_and_coupler_length is not None
    span = jnp.asarray(
        span_default if route_and_coupler_length is None else route_and_coupler_length,
        dtype=float,
    )
    gap = jnp.asarray(probeline_coupling_gap, dtype=float)
    valid = _validate_layout(
        kind=kind,
        resonator_meander_start=resonator_meander_start,
        coupler_offset=coupler_offset,
        coupler=coupler,
        coupler_port=coupler_port,
        qubit_rotation=qubit_rotation,
        resonator_length=resonator_length,
        coupling_length=coupling_length,
        span=span,
        explicit_span=explicit_span,
    )

    z0 = cpw_z0_from_cross_section(resonator_cross_section, f_eval)
    load = {
        ("o1", "o1"): _qubit_reflection(
            f_eval,
            z0,
            capacitances,
            inductance=qubit_inductance,
            junction_capacitance=junction_capacitance,
        )
    }

    termination = jnp.where(resonator_open_end, 1.0, -1.0)
    meander = sax.evaluate_circuit_fg(
        ({"line,o2": "term,o1"}, {"o1": "line,o1"}),
        {
            "line": straight(
                f=f_eval,
                length=resonator_length - span,
                cross_section=resonator_cross_section,
            ),
            "term": {("o1", "o1"): termination},
        },
    )["o1", "o1"]

    instances: dict[str, sax.SType] = {
        "coupler": coupler_straight(
            f=f_eval,
            length=coupling_length,
            gap=gap,
            cross_section=resonator_cross_section,
        ),
        "route": straight(
            f=f_eval,
            length=span - coupling_length,
            cross_section=resonator_cross_section,
        ),
        "load": load,
        "meander": {("o1", "o1"): jnp.where(resonator_open_start, 1.0, meander)},
    }
    connections = {
        "coupler,o3": "route,o1",
        "route,o2": "load,o1",
        "coupler,o2": "meander,o1",
    }
    ports = {"coupling_o1": "coupler,o1", "coupling_o2": "coupler,o4"}
    s_params = sax.evaluate_circuit_fg((connections, ports), instances)
    at_dc = f == 0
    return {
        key: jnp.where(
            valid,
            jnp.where(at_dc, 1.0 if key[0] != key[1] else 0.0, value),
            jnp.full_like(value, jnp.nan),
        )
        for key, value in s_params.items()
    }


def flipmon_with_resonator_and_probeline(
    f: ArrayLike = DEFAULT_FREQUENCY,
    resonator_length: ArrayLike = 5000.0,
    probeline_coupling_gap: ArrayLike = _NOMINAL_GAP_UM,
    probeline_coupling_length: ArrayLike | None = None,
    resonator_cross_section: CrossSectionSpec = "cpw",
    resonator_open_start: bool | ArrayLike = False,
    resonator_open_end: bool | ArrayLike = True,
    qubit: ComponentSpec = "flipmon_with_bbox",
    route_and_coupler_length: ArrayLike | None = None,
    qubit_inductance: ArrayLike = 7e-9,
    junction_capacitance: ArrayLike = 0.0,
    resonator_meander_start: _LayoutSetting = None,
    coupler_offset: _LayoutSetting = None,
    coupler: ComponentSpec | None = None,
    coupler_port: str | None = None,
    qubit_rotation: ArrayLike | None = None,
) -> sax.SDict:
    """Return the flipmon readout assembly. See :func:`transmon_with_resonator_and_probeline`."""
    return transmon_with_resonator_and_probeline(
        f=f,
        resonator_length=resonator_length,
        probeline_coupling_gap=probeline_coupling_gap,
        probeline_coupling_length=probeline_coupling_length,
        resonator_cross_section=resonator_cross_section,
        resonator_open_start=resonator_open_start,
        resonator_open_end=resonator_open_end,
        qubit=qubit,
        route_and_coupler_length=route_and_coupler_length,
        qubit_inductance=qubit_inductance,
        junction_capacitance=junction_capacitance,
        resonator_meander_start=resonator_meander_start,
        coupler_offset=coupler_offset,
        coupler=coupler,
        coupler_port=coupler_port,
        qubit_rotation=qubit_rotation,
    )
