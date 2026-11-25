"""Model regressions for the qubit-resonator-probeline readout assemblies."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from qpdk import PDK
from qpdk.models.readout_assembly import (
    flipmon_with_resonator_and_probeline,
    transmon_with_resonator_and_probeline,
)

_FREQUENCIES = jnp.linspace(3e9, 12e9, 301)
_NOTCH_FREQUENCIES = jnp.linspace(10e9, 14e9, 401)
_SERIALIZED_FLIPMON = "Fflipmon_with_bbox_Mqpdkpcellsptransmon_SJPR180"

_READOUT_PORTS = ("coupling_o1", "coupling_o2")
# SAX sdict keys are (input, output), so this tuple is also the S21 key.
_READOUT_KEYS = {(a, b) for a in _READOUT_PORTS for b in _READOUT_PORTS}

_MODELS = [
    pytest.param(transmon_with_resonator_and_probeline, id="transmon"),
    pytest.param(flipmon_with_resonator_and_probeline, id="flipmon"),
]

_LAYOUT_MODELS = [
    pytest.param(transmon_with_resonator_and_probeline, (-45.0, 0.0), id="transmon"),
    pytest.param(flipmon_with_resonator_and_probeline, (-10.0, 0.0), id="flipmon"),
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


@pytest.mark.parametrize(("model", "coupler_offset"), _LAYOUT_MODELS)
def test_nominal_layout_settings_are_accepted(
    model: Callable[..., sax.SDict], coupler_offset: tuple[float, float]
) -> None:
    """Explicit nominal layout settings match the default model call."""
    nominal = model(f=_FREQUENCIES)
    accepted = model(
        f=_FREQUENCIES,
        resonator_meander_start=jnp.asarray((-900.0, -1200.0)),
        coupler_offset=jnp.asarray(coupler_offset),
    )
    _assert_close(nominal, accepted)


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


@pytest.mark.parametrize(("model", "coupler_offset"), _LAYOUT_MODELS)
def test_nonnominal_coupler_offset_is_rejected(
    model: Callable[..., sax.SDict], coupler_offset: tuple[float, float]
) -> None:
    """A shifted coupling plate is rejected for both qubits."""
    with pytest.raises(ValueError, match="nominal"):
        model(f=_FREQUENCIES, coupler_offset=(coupler_offset[0] - 10.0, 0.0))


@pytest.mark.parametrize("model", _MODELS)
def test_jit_masks_invalid_coupling_length(
    model: Callable[..., sax.SDict],
) -> None:
    """A traced invalid length yields NaN rather than a plausible response."""

    @jax.jit
    def evaluate(coupling_length: jnp.ndarray) -> sax.SDict:
        return model(f=_FREQUENCIES, probeline_coupling_length=coupling_length)

    valid = _s_matrix(evaluate(jnp.asarray(400.0)))
    invalid = _s_matrix(evaluate(jnp.asarray(1000.0)))

    assert bool(jnp.isfinite(valid).all())
    assert bool(jnp.isnan(invalid).all())


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
    """The serialized cell setting selects the same qubit as the flipmon boundary."""
    serialized = transmon_with_resonator_and_probeline(
        f=_FREQUENCIES, qubit=_SERIALIZED_FLIPMON
    )
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
