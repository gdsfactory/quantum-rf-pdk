"""Small-signal SAX model of the probeline readout assembly.

The four-port :func:`~qpdk.models.couplers.coupler_straight` is the core: two
ports are the probeline ends and the others go through the meander and the
route to the qubit coupling plate. Uniform nominal CPW sections model the route
and meander, with ``resonator_length`` covering the route plus coupling span
and the meander taking the remainder.

The qubit load comes from the nominal coupon dataset of the shipped chips. It
keeps the island-to-island, island-to-readout and island-to-ground
capacitances. The distributed sections already account for the readout and
feedline ground terms and their mutual capacitance. The load also omits the
tiny feedline-to-island branches. A Josephson inductance joins the islands and
defaults to 7 nH.

The model uses the shipped qubit geometry and accepts only the nominal CPW
cross-section. Its capacitances come from electrostatic FEM. The small-signal
resonances still need driven FEM or measurement validation. At zero frequency
the probeline is an ideal through line.
"""

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import cache, partial
from typing import Literal

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
_NOMINAL_MEANDER_START = (-900.0, -1200.0)
_NOMINAL_CROSS_SECTION = "cpw"
_FLIPMON_JUNCTION_SETTING = {"junction_position_rotation": 180}


def transmon_with_resonator_and_probeline(
    f: ArrayLike = DEFAULT_FREQUENCY,
    resonator_length: ArrayLike = 5000.0,
    probeline_coupling_gap: ArrayLike = _NOMINAL_GAP_UM,
    probeline_coupling_length: ArrayLike | None = None,
    resonator_cross_section: CrossSectionSpec = _NOMINAL_CROSS_SECTION,
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
        f: Frequency in Hz, scalar or 1D. Zero returns the exact through-line
            response.
        resonator_length: Total length in um, including the route and coupling span.
            Scalars and 1D arrays broadcast elementwise. Multidimensional arrays
            and incompatible vectors raise before evaluation. To sweep a frequency
            range at each of several lengths, use separate calls or ``vmap``.
        probeline_coupling_gap: Coupler gap in um. It changes the distributed
            coupler only; the coupon lookup stays at 16 um.
        probeline_coupling_length: Coupling length in um. None uses 800 um. Without
            an explicit ``route_and_coupler_length`` it must be positive and at most
            800 um.
        resonator_cross_section: Cross-section of the coupling section, route and
            meander. Only the nominal resolved CPW geometry (the default ``"cpw"``)
            is supported, in any equivalent GDSFactory+ spec form. Other geometry
            raises before the model is evaluated.
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
            A partial or mapping value carries only ``width`` and ``length``, since
            any extra plate setting would draw a different coupon. It must give a
            20 um by 394 um plate for the double-pad or 10 um by 58 um for the
            flipmon.
        coupler_port: Qubit port the plate attaches to. None skips the check. A
            concrete value must be ``"left_pad"`` or ``"outer_ring_outside"``.
        qubit_rotation: Qubit rotation in degrees. None skips the check. A concrete
            value must be ``90`` for the double-pad or ``-90`` for the flipmon.

    Returns:
        Two-port S-parameters at ``coupling_o1`` and ``coupling_o2``.
    """
    resonator_length = jnp.asarray(resonator_length, dtype=float)
    f = _broadcast_frequency(jnp.asarray(f, dtype=float), resonator_length)

    _validate_cross_section(resonator_cross_section)
    coupon = _coupon_for(qubit)
    with jax.ensure_compile_time_eval():
        capacitances = _coupon_capacitances(coupon)
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
        coupon.span if route_and_coupler_length is None else route_and_coupler_length,
        dtype=float,
    )
    gap = jnp.asarray(probeline_coupling_gap, dtype=float)
    valid = _validate_layout(
        coupon=coupon,
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
    resonator_cross_section: CrossSectionSpec = _NOMINAL_CROSS_SECTION,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _Coupon:
    """Nominal coupon of one qubit kind.

    Lengths are in um. ``pad_offset`` is the coupon dataset interpolation
    parameter and ``plate_offset`` the nominal layout (x, y) offset.
    """

    dataset: str
    names: frozenset[str]
    span: float
    plate_width: float
    plate_length: float
    pad_offset: float
    plate_offset: tuple[float, float]
    coupler_port: str
    rotation: float
    groups: tuple[tuple[str, ...], ...]


_DOUBLE_PAD_COUPON = _Coupon(
    dataset="transmon_readout_coupon_palace",
    names=frozenset({"double_pad_transmon_with_bbox"}),
    span=1742.5795478,
    plate_width=20.0,
    plate_length=394.0,
    pad_offset=45.0,
    plate_offset=(-45.0, 0.0),
    coupler_port="left_pad",
    rotation=90.0,
    groups=(("island_a",), ("island_b",), ("readout",), ("feedline",)),
)

_FLIPMON_COUPON = _Coupon(
    dataset="flipmon_readout_coupon_palace",
    names=frozenset({
        "flipmon_with_bbox",
        clean_value(partial(flipmon_with_bbox, **_FLIPMON_JUNCTION_SETTING)),
    }),
    span=1900.0795478,
    plate_width=10.0,
    plate_length=58.0,
    pad_offset=10.0,
    plate_offset=(-10.0, 0.0),
    coupler_port="outer_ring_outside",
    rotation=-90.0,
    # The top pad is bonded to island_a with an ideal wire.
    groups=(("island_a", "top_pad"), ("island_b",), ("readout",), ("feedline",)),
)

_COUPONS: dict[_QubitKind, _Coupon] = {
    "double_pad": _DOUBLE_PAD_COUPON,
    "flipmon": _FLIPMON_COUPON,
}


def _factory_name(spec: object) -> str:
    """Resolve a qubit component spec to a factory name."""
    if isinstance(spec, partial):
        if spec.args or spec.keywords not in ({}, _FLIPMON_JUNCTION_SETTING):
            raise ValueError(
                f"Unsupported qubit partial settings: {spec.args!r}, {spec.keywords!r}"
            )
        name = _factory_name(spec.func)
        if spec.keywords and name not in _FLIPMON_COUPON.names:
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
            resolved in _FLIPMON_COUPON.names and settings == _FLIPMON_JUNCTION_SETTING
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


def _coupon_for(qubit: ComponentSpec) -> _Coupon:
    """Find the nominal coupon for a qubit factory or serialized identity."""
    name = _factory_name(qubit)
    for coupon in _COUPONS.values():
        if name in coupon.names:
            return coupon
    raise ValueError(f"Qubit {name!r} has no nominal readout coupon")


@cache
def _nominal_plate_string(coupon: _Coupon) -> str:
    """Return the serialized nominal coupling plate component spec."""
    return clean_value(
        partial(
            plate_capacitor_single, width=coupon.plate_width, length=coupon.plate_length
        )
    )


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


def _coupler_matches(value: ComponentSpec | None, coupon: _Coupon) -> jax.Array:
    """Check the nominal coupling plate in the forms accepted by GDSFactory+."""
    if value is None:
        return jnp.asarray(True)
    factory = plate_capacitor_single
    if isinstance(value, partial):
        if value.func is not factory or value.args:
            return jnp.asarray(False)
        settings = value.keywords
    elif isinstance(value, dict):
        function = value.get("function") or value.get("component")
        module = value.get("module")
        if function != factory.__name__ or (
            module is not None and module != factory.__module__
        ):
            return jnp.asarray(False)
        settings = value.get("settings")
    else:
        try:
            serialized = value if isinstance(value, str) else clean_value(value)
            return jnp.asarray(serialized == _nominal_plate_string(coupon))
        except (TypeError, ValueError):
            return jnp.asarray(False)
    # Extra plate settings would draw geometry outside the coupon.
    if not isinstance(settings, dict) or set(settings) != {"width", "length"}:
        return jnp.asarray(False)
    try:
        width = jnp.asarray(settings["width"], dtype=float)
        length = jnp.asarray(settings["length"], dtype=float)
    except (TypeError, ValueError):
        return jnp.asarray(False)
    return jnp.all(width == coupon.plate_width) & jnp.all(length == coupon.plate_length)


@cache
def _invalid_message(coupon: _Coupon) -> str:
    """Return the shared eager and JIT diagnostic for a nonnominal assembly."""
    return (
        "This model supports only the nominal shipped readout assembly: "
        f"resonator_meander_start {_NOMINAL_MEANDER_START}, "
        f"coupler_offset {coupon.plate_offset}, "
        f"coupler_port {coupon.coupler_port!r}, "
        f"qubit_rotation {coupon.rotation:g}, "
        f"coupler plate width {coupon.plate_width:g} um and length "
        f"{coupon.plate_length:g} um, positive route and meander sections, and "
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


def _broadcast_frequency(f: jax.Array, resonator_length: jax.Array) -> jax.Array:
    """Broadcast frequency with length over one axis for SAX."""
    if f.ndim <= 1 and resonator_length.ndim <= 1:
        with suppress(ValueError):
            return jnp.broadcast_to(
                f, jnp.broadcast_shapes(f.shape, resonator_length.shape)
            )
    raise ValueError(
        "f and resonator_length must be broadcast-compatible scalars or 1D arrays. "
        f"Got f shape {f.shape} and resonator_length shape {resonator_length.shape}. "
        "Sweep length with separate calls or vmap."
    )


def _validate_cross_section(cross_section: CrossSectionSpec) -> None:
    """Require the drawn cross-section used to fix the coupling length and span."""
    # TODO(Python 3.15): `lazy from qpdk import PDK` (PEP 810) at module scope breaks the cycle.
    from qpdk import PDK  # ruff: ignore[import-outside-top-level]

    def geometry(spec: CrossSectionSpec) -> dict[str, object]:
        fields = PDK.get_cross_section(spec).model_dump()
        for section in fields["sections"]:
            if section["layer"] is not None:
                section["layer"] = int(PDK.get_layer(section["layer"]))
        return fields

    if geometry(cross_section) != geometry(_NOMINAL_CROSS_SECTION):
        raise ValueError(
            "This model supports only the nominal CPW cross-section "
            f"{_NOMINAL_CROSS_SECTION!r}. Got {cross_section!r}."
        )


def _validate_layout(
    *,
    coupon: _Coupon,
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
        & _layout_setting_matches(coupler_offset, coupon.plate_offset)
        & _coupler_matches(coupler, coupon)
        & jnp.asarray(
            coupler_port is None
            or (isinstance(coupler_port, str) and coupler_port == coupon.coupler_port)
        )
        & jnp.asarray(
            qubit_rotation is None
            or jnp.all(jnp.asarray(qubit_rotation, dtype=float) == coupon.rotation)
        )
        & (coupling_length > 0)
        & (span > coupling_length)
        & (resonator_length > span)
    )
    if not explicit_span:
        valid = valid & (coupling_length <= _NOMINAL_COUPLING_LENGTH_UM)
    return _check_valid(valid, _invalid_message(coupon))


@cache
def _coupon_capacitances(coupon: _Coupon) -> _CouponCapacitances:
    """Reduce one nominal coupon to island ground and mutual capacitances in F."""
    grid = Dataset(coupon.dataset).grid("maxwell_capacitance", allow_missing=False)
    groups = coupon.groups
    if set(grid.terminals) != {name for group in groups for name in group}:
        raise ValueError(f"Coupon terminals {grid.terminals} do not match {groups}")
    # A 400 um distributed coupler falls below the coupon's 500 um lower limit.
    maxwell = GridInterpolator(grid)(
        pad_length=coupon.plate_length,
        pad_offset=coupon.pad_offset,
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
