"""Model regressions for the qubit-resonator-probeline readout assemblies."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
import yaml
from gdsfactory.cross_section import CrossSection
from kfactory.serialization import clean_value

from qpdk import PDK
from qpdk.cells import plate_capacitor_single
from qpdk.models.readout_assembly import (
    flipmon_with_resonator_and_probeline,
    transmon_with_resonator_and_probeline,
)
from qpdk.tech import LAYER

_FREQUENCIES = jnp.linspace(3e9, 12e9, 301)
_NOTCH_FREQUENCIES = jnp.linspace(10e9, 14e9, 401)
_SAMPLE_DIR = Path(__file__).parents[2] / "qpdk/samples"

_READOUT_PORTS = ("coupling_o1", "coupling_o2")
# SAX sdict keys are (input, output), so this tuple is also the S21 key.
_READOUT_KEYS = {(a, b) for a in _READOUT_PORTS for b in _READOUT_PORTS}

_MODELS = [
    pytest.param(transmon_with_resonator_and_probeline, id="transmon"),
    pytest.param(flipmon_with_resonator_and_probeline, id="flipmon"),
]

_LAYOUT_SETTINGS = [
    pytest.param(
        transmon_with_resonator_and_probeline,
        {
            "resonator_meander_start": (-900.0, -1200.0),
            "coupler_offset": (-45.0, 0.0),
            "coupler": partial(plate_capacitor_single, width=20, length=394),
            "coupler_port": "left_pad",
            "qubit_rotation": 90.0,
        },
        id="transmon",
    ),
    pytest.param(
        flipmon_with_resonator_and_probeline,
        {
            "resonator_meander_start": (-900.0, -1200.0),
            "coupler_offset": (-10.0, 0.0),
            "coupler": partial(plate_capacitor_single, width=10, length=58),
            "coupler_port": "outer_ring_outside",
            "qubit_rotation": -90.0,
        },
        id="flipmon",
    ),
]

_NOMINAL_OFFSETS = [
    pytest.param(transmon_with_resonator_and_probeline, (-45.0, 0.0), id="transmon"),
    pytest.param(flipmon_with_resonator_and_probeline, (-10.0, 0.0), id="flipmon"),
]

_NONNOMINAL_LAYOUT_SETTINGS = [
    pytest.param(
        transmon_with_resonator_and_probeline,
        "coupler",
        clean_value(partial(plate_capacitor_single, width=20, length=400)),
        "coupler|nominal",
        id="transmon-coupler",
    ),
    pytest.param(
        transmon_with_resonator_and_probeline,
        "coupler_port",
        "right_pad",
        "coupler_port|nominal",
        id="transmon-coupler-port",
    ),
    pytest.param(
        transmon_with_resonator_and_probeline,
        "qubit_rotation",
        0.0,
        "qubit_rotation|nominal",
        id="transmon-qubit-rotation",
    ),
    pytest.param(
        flipmon_with_resonator_and_probeline,
        "coupler",
        clean_value(partial(plate_capacitor_single, width=10, length=60)),
        "coupler|nominal",
        id="flipmon-coupler",
    ),
    pytest.param(
        flipmon_with_resonator_and_probeline,
        "coupler_port",
        "inner_ring_near_junction",
        "coupler_port|nominal",
        id="flipmon-coupler-port",
    ),
    pytest.param(
        flipmon_with_resonator_and_probeline,
        "qubit_rotation",
        0.0,
        "qubit_rotation|nominal",
        id="flipmon-qubit-rotation",
    ),
]

_REGISTRATIONS = [
    pytest.param(
        "transmon_with_resonator_and_probeline",
        transmon_with_resonator_and_probeline,
        id="transmon",
    ),
    pytest.param(
        "double_pad_transmon_with_resonator_and_probeline",
        transmon_with_resonator_and_probeline,
        id="double-pad",
    ),
    pytest.param(
        "flipmon_with_resonator_and_probeline",
        flipmon_with_resonator_and_probeline,
        id="flipmon",
    ),
]


def _equivalent_cross_section_object() -> CrossSection:
    """Return the nominal CPW geometry with a non-default cross-section name."""
    return gf.cross_section.cross_section(
        width=10,
        layer=LAYER.M1_DRAW,
        radius=100,
        radius_min=11,
        sections=(
            gf.Section(width=6, offset=8, layer=LAYER.M1_ETCH, name="etch_offset_pos"),
            gf.Section(width=6, offset=-8, layer=LAYER.M1_ETCH, name="etch_offset_neg"),
            gf.Section(width=10, layer=LAYER.WG, name="waveguide"),
        ),
    )


_CROSS_SECTION_FORMS = [
    pytest.param("cpw", id="name"),
    pytest.param({"cross_section": "cpw"}, id="dict"),
    pytest.param(
        {"cross_section": "coplanar_waveguide", "settings": {}}, id="alias-settings"
    ),
    pytest.param(_equivalent_cross_section_object(), id="object"),
]

_BASELINE: dict[str, float | bool | str] = {
    "resonator_length": 5000.0,
    "resonator_cross_section": "cpw",
    "probeline_coupling_length": 200.0,
    "probeline_coupling_gap": 14.0,
    "qubit_inductance": 7e-9,
    "resonator_open_start": False,
    "resonator_open_end": True,
}
_SETTING_CHANGES: list[tuple[str, dict[str, float | bool], float]] = [
    ("coupling length", {"probeline_coupling_length": 400.0}, 1e-4),
    ("coupling gap", {"probeline_coupling_gap": 20.0}, 1e-4),
    ("qubit inductance", {"qubit_inductance": 1e-8}, 1e-8),
    ("open start", {"resonator_open_start": True}, 1e-4),
    ("open end", {"resonator_open_end": False}, 1e-4),
]


def _assert_close(
    first: sax.SDict,
    second: sax.SDict,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> None:
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], rtol=rtol, atol=atol)


def _max_difference(first: sax.SDict, second: sax.SDict) -> float:
    assert set(first) == set(second)
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(first[key]) - jnp.asarray(second[key]))))
        for key in first
    )


def _s_matrix(s_params: sax.SDict) -> jnp.ndarray:
    a, b = _READOUT_PORTS
    return jnp.stack(
        (
            jnp.stack((s_params[a, a], s_params[b, a]), axis=-1),
            jnp.stack((s_params[a, b], s_params[b, b]), axis=-1),
        ),
        axis=-2,
    )


@pytest.mark.parametrize(("name", "model"), _REGISTRATIONS)
def test_pdk_registers_readout_boundary(
    name: str, model: Callable[..., sax.SDict]
) -> None:
    """The named boundary, including the double-pad alias, is registered."""
    assert name in PDK.models
    _assert_close(PDK.models[name](f=_FREQUENCIES), model(f=_FREQUENCIES))


@pytest.mark.parametrize("model", _MODELS)
def test_readout_network_is_reciprocal_and_passive(
    model: Callable[..., sax.SDict],
) -> None:
    """Return the exact coupling_o1/coupling_o2 2x2 network with singular values <= 1."""
    s_params = model(f=_FREQUENCIES)
    assert set(s_params) == _READOUT_KEYS

    matrix = _s_matrix(s_params)
    assert bool(jnp.isfinite(matrix).all())
    np.testing.assert_allclose(
        matrix, jnp.swapaxes(matrix, -1, -2), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_array_less(jnp.linalg.svdvals(matrix), 1.0 + 1e-6)


@pytest.mark.parametrize("model", _MODELS)
def test_numeric_settings_are_jittable_scalars(
    model: Callable[..., sax.SDict],
) -> None:
    """SAX passes every numeric setting as a JAX scalar array."""

    @jax.jit
    def evaluate(
        length: jnp.ndarray,
        coupling_length: jnp.ndarray,
        coupling_gap: jnp.ndarray,
        inductance: jnp.ndarray,
    ) -> sax.SDict:
        return model(
            f=_FREQUENCIES,
            resonator_length=length,
            resonator_cross_section="cpw",
            probeline_coupling_length=coupling_length,
            probeline_coupling_gap=coupling_gap,
            qubit_inductance=inductance,
        )

    traced = evaluate(
        jnp.asarray(5000.0),
        jnp.asarray(400.0),
        jnp.asarray(16.0),
        jnp.asarray(7e-9),
    )
    eager = model(
        f=_FREQUENCIES,
        resonator_length=5000.0,
        resonator_cross_section="cpw",
        probeline_coupling_length=400.0,
        probeline_coupling_gap=16.0,
        qubit_inductance=7e-9,
    )
    _assert_close(traced, eager, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("model", _MODELS)
def test_physical_settings_change_the_response(
    model: Callable[..., sax.SDict],
) -> None:
    """Each forwarded setting changes the response rather than being dropped."""
    baseline = model(f=_FREQUENCIES, **_BASELINE)
    for label, change, floor in _SETTING_CHANGES:
        changed = model(f=_FREQUENCIES, **(_BASELINE | change))
        difference = _max_difference(baseline, changed)
        assert difference > floor, (
            f"{label} did not reach the model: max |dS| = {difference:.3g}"
        )


@pytest.mark.parametrize(("model", "settings"), _LAYOUT_SETTINGS)
def test_nominal_layout_settings_are_accepted(
    model: Callable[..., sax.SDict], settings: dict[str, object]
) -> None:
    """Explicit nominal layout settings match the default model call."""
    _assert_close(model(f=_FREQUENCIES), model(f=_FREQUENCIES, **settings))


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("cross_section", _CROSS_SECTION_FORMS)
def test_equivalent_cross_section_forms_match_nominal(
    model: Callable[..., sax.SDict], cross_section: object
) -> None:
    """Equivalent cross-section spec forms resolve to the same nominal CPW."""
    _assert_close(
        model(f=_FREQUENCIES),
        model(f=_FREQUENCIES, resonator_cross_section=cross_section),
    )


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(
    ("setting", "value"),
    [("radius", 200.0), ("width", 20.0)],
    ids=["radius", "width"],
)
def test_nonnominal_cross_section_is_rejected(
    model: Callable[..., sax.SDict], setting: str, value: float
) -> None:
    """A drawing-changing cross-section geometry raises before evaluation."""
    cross_section = {"cross_section": "cpw", "settings": {setting: value}}
    with pytest.raises(ValueError, match="nominal CPW"):
        model(f=_FREQUENCIES, resonator_cross_section=cross_section)


def test_rejected_cross_section_leaves_the_active_pdk() -> None:
    other_pdk = gf.Pdk(name="readout-validation")
    previous = gf.get_active_pdk()
    other_pdk.activate()
    try:
        with pytest.raises(ValueError, match="nominal CPW"):
            transmon_with_resonator_and_probeline(
                resonator_cross_section={
                    "cross_section": "cpw",
                    "settings": {"radius": 200.0},
                }
            )
        assert gf.get_active_pdk() is other_pdk
    finally:
        previous.activate()


@pytest.mark.parametrize("model", _MODELS)
def test_equivalent_cross_section_is_jittable(
    model: Callable[..., sax.SDict],
) -> None:
    """An equivalent dict spec resolves at trace time and compiles."""
    cross_section = {"cross_section": "cpw", "settings": {"width": 10}}

    @jax.jit
    def evaluate() -> sax.SDict:
        return model(f=_FREQUENCIES, resonator_cross_section=cross_section)

    _assert_close(jax.block_until_ready(evaluate()), model(f=_FREQUENCIES))


@pytest.mark.parametrize(
    ("model", "setting", "value", "match"), _NONNOMINAL_LAYOUT_SETTINGS
)
def test_nonnominal_layout_setting_is_rejected(
    model: Callable[..., sax.SDict],
    setting: str,
    value: object,
    match: str,
) -> None:
    """A changed coupling plate, plate port or rotation is rejected."""
    with pytest.raises(ValueError, match=match):
        model(f=_FREQUENCIES, **{setting: value})


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("resonator_meander_start", (-1100.0, -1200.0)),
        ("probeline_coupling_length", 1000.0),
    ],
    ids=["meander-start", "coupling-length"],
)
def test_nonnominal_setting_is_rejected(
    model: Callable[..., sax.SDict],
    setting: str,
    value: float | tuple[float, float],
) -> None:
    """Concrete nonnominal settings and implicit spans above 800 um raise."""
    with pytest.raises(ValueError, match="nominal"):
        model(f=_FREQUENCIES, **{setting: jnp.asarray(value)})


@pytest.mark.parametrize(("model", "coupler_offset"), _NOMINAL_OFFSETS)
def test_nonnominal_coupler_offset_is_rejected(
    model: Callable[..., sax.SDict], coupler_offset: tuple[float, float]
) -> None:
    """A shifted coupling plate is rejected for both qubits."""
    with pytest.raises(ValueError, match="nominal"):
        model(f=_FREQUENCIES, coupler_offset=(coupler_offset[0] - 10.0, 0.0))


@pytest.mark.parametrize("model", _MODELS)
def test_jit_reports_invalid_coupling_length(
    model: Callable[..., sax.SDict],
) -> None:
    """A traced invalid length raises from the host callback with a diagnostic."""

    @jax.jit
    def evaluate(coupling_length: jnp.ndarray) -> sax.SDict:
        return model(f=_FREQUENCIES, probeline_coupling_length=coupling_length)

    with pytest.raises(RuntimeError, match="probeline_coupling_length"):
        jax.block_until_ready(evaluate(jnp.asarray(1000.0)))

    valid = _s_matrix(jax.block_until_ready(evaluate(jnp.asarray(400.0))))
    assert bool(jnp.isfinite(valid).all())


@pytest.mark.parametrize("model", _MODELS)
def test_vector_of_valid_lengths_matches_scalar_calls(
    model: Callable[..., sax.SDict],
) -> None:
    """A vector of valid lengths is evaluated element by element."""
    frequencies = jnp.asarray(5e9)
    lengths = jnp.asarray([5000.0, 5200.0])
    vector = model(f=frequencies, resonator_length=lengths)

    for index, length in enumerate(lengths):
        scalar = model(f=frequencies, resonator_length=length)
        for key in _READOUT_KEYS:
            np.testing.assert_allclose(
                vector[key][index], scalar[key], rtol=1e-6, atol=1e-9
            )


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(
    ("frequency_values", "length_values"),
    [((4e9, 5e9), (5000.0, 5200.0)), ((4e9,), (5000.0, 5200.0))],
    ids=["paired", "singleton"],
)
def test_compatible_sweeps_broadcast_elementwise(
    model: Callable[..., sax.SDict],
    frequency_values: tuple[float, ...],
    length_values: tuple[float, ...],
) -> None:
    """Compatible frequency and length vectors pair elementwise."""
    frequencies = jnp.asarray(frequency_values)
    lengths = jnp.asarray(length_values)
    vector = model(f=frequencies, resonator_length=lengths)

    pairs = zip(*jnp.broadcast_arrays(frequencies, lengths), strict=True)
    for index, (frequency, length) in enumerate(pairs):
        scalar = model(f=frequency, resonator_length=length)
        for key in _READOUT_KEYS:
            np.testing.assert_allclose(
                vector[key][index], scalar[key], rtol=1e-6, atol=1e-9
            )


@pytest.mark.parametrize("model", _MODELS)
def test_mismatched_length_vectors_are_rejected(
    model: Callable[..., sax.SDict],
) -> None:
    """A frequency vector and a length vector must have equal length."""
    with pytest.raises(
        ValueError, match=r"f shape \(2,\) and resonator_length shape \(3,\)"
    ):
        model(
            f=jnp.asarray((4e9, 5e9)),
            resonator_length=jnp.asarray((5000.0, 5100.0, 5200.0)),
        )


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(
    ("setting", "value", "shape"),
    [
        ("f", jnp.asarray([[4e9, 5e9]]), "(1, 2)"),
        ("resonator_length", jnp.asarray([[5000.0], [5200.0]]), "(2, 1)"),
    ],
    ids=["frequency", "length"],
)
def test_multidimensional_sweeps_are_rejected(
    model: Callable[..., sax.SDict],
    setting: str,
    value: jnp.ndarray,
    shape: str,
) -> None:
    """A 2D sweep has no elementwise pairing and raises with its shape."""
    kwargs = {"f": jnp.asarray(5e9)}
    kwargs[setting] = value
    with pytest.raises(ValueError, match=rf"{setting} shape {re.escape(shape)}"):
        model(**kwargs)


@pytest.mark.parametrize("model", _MODELS)
def test_explicit_span_allows_coupling_above_implicit_limit(
    model: Callable[..., sax.SDict],
) -> None:
    """An explicit span is the escape hatch above the 800 um implicit limit."""
    s_params = model(
        f=_FREQUENCIES,
        probeline_coupling_length=1000.0,
        route_and_coupler_length=2200.0,
    )
    assert bool(jnp.isfinite(_s_matrix(s_params)).all())


@pytest.mark.parametrize("model", _MODELS)
def test_dc_is_exact_through_line_and_matches_5_ghz(
    model: Callable[..., sax.SDict],
) -> None:
    """A mixed sweep returns the exact DC through line and the same 5 GHz point."""
    mixed = model(f=jnp.asarray((0.0, 5e9)))
    single = model(f=jnp.asarray((5e9,)))

    for key in _READOUT_KEYS:
        expected = 1.0 if key[0] != key[1] else 0.0
        assert mixed[key][0].item() == expected
        np.testing.assert_allclose(mixed[key][1], single[key][0], rtol=1e-6, atol=1e-9)


def test_serialized_flipmon_matches_named_flipmon() -> None:
    """The live layout cell's serialized qubit selects the flipmon coupon."""
    live_qubit = PDK.cells["flipmon_with_resonator_and_probeline"]().settings["qubit"]
    serialized = transmon_with_resonator_and_probeline(f=_FREQUENCIES, qubit=live_qubit)
    named = flipmon_with_resonator_and_probeline(f=_FREQUENCIES)
    double_pad = transmon_with_resonator_and_probeline(f=_FREQUENCIES)

    _assert_close(serialized, named)
    difference = _max_difference(named, double_pad)
    assert difference > 1e-4, (
        f"flipmon and double-pad responses differ by only {difference:.3g}"
    )


@pytest.mark.parametrize("model", _MODELS)
def test_resonator_length_shifts_the_notch(
    model: Callable[..., sax.SDict],
) -> None:
    """A 200 um longer resonator moves the dominant notch down by the loaded scale."""
    nominal = model(f=_NOTCH_FREQUENCIES, resonator_length=5000.0)
    longer = model(f=_NOTCH_FREQUENCIES, resonator_length=5200.0)
    nominal_index = int(jnp.argmin(jnp.abs(nominal[_READOUT_PORTS])))
    longer_index = int(jnp.argmin(jnp.abs(longer[_READOUT_PORTS])))

    assert 0 < nominal_index < len(_NOTCH_FREQUENCIES) - 1, (
        f"notch at {float(_NOTCH_FREQUENCIES[nominal_index]) / 1e9:.3f} GHz "
        "is on the sweep edge"
    )
    assert longer_index < nominal_index, "longer resonator must lower the notch"
    shift = float(_NOTCH_FREQUENCIES[nominal_index] - _NOTCH_FREQUENCIES[longer_index])
    assert 0.1e9 < shift < 1.0e9, f"notch shifted by {shift / 1e9:.3f} GHz"


@pytest.mark.parametrize(
    ("stem", "model", "centers_ghz"),
    [
        pytest.param(
            "qubit_test_chip",
            transmon_with_resonator_and_probeline,
            (5.80, 6.54, 6.96, 6.00),
            id="transmon",
        ),
        pytest.param(
            "flipmon_test_chip",
            flipmon_with_resonator_and_probeline,
            (5.96, 6.22, 6.49, 6.78),
            id="flipmon",
        ),
    ],
)
@pytest.mark.parametrize("index", range(4))
def test_readout_chip_notches(
    stem: str,
    model: Callable[..., sax.SDict],
    centers_ghz: tuple[float, ...],
    index: int,
) -> None:
    document = yaml.safe_load((_SAMPLE_DIR / f"{stem}.pic.yml").read_text())
    settings = document["instances"][f"qubit{index + 1}"]["settings"]
    center_ghz = centers_ghz[index]
    frequencies = jnp.linspace(center_ghz - 0.05, center_ghz + 0.05, 1001) * 1e9
    response = model(
        f=frequencies,
        resonator_length=settings["resonator_length"],
        resonator_open_end=settings.get("resonator_open_end", True),
        probeline_coupling_length=settings.get("probeline_coupling_length"),
    )
    transmission = jnp.abs(response[_READOUT_PORTS])
    minimum = int(jnp.argmin(transmission))
    assert 0 < minimum < frequencies.size - 1
    assert float(transmission[minimum]) < 0.9
