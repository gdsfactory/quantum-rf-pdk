"""Tests for qpdk.models.waveguides module."""

import inspect
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from typing import Any, final

import gdsfactory as gf
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
import sax
from hypothesis import assume, given, settings
from numpy.testing import assert_allclose, assert_array_less

from qpdk.cells.waveguides import (
    bend_circular as bend_circular_cell,
    bend_circular_all_angle as bend_circular_all_angle_cell,
    bend_euler as bend_euler_cell,
    bend_euler_all_angle as bend_euler_all_angle_cell,
    bend_s as bend_s_cell,
)
from qpdk.models import models, waveguides as waveguide_models
from qpdk.models.waveguides import (
    airbridge,
    bend_circular,
    bend_circular_all_angle,
    bend_euler,
    bend_euler_all_angle,
    bend_s,
    indium_bump,
    nxn,
    rectangle,
    straight,
    straight_double_open,
    straight_open,
    straight_shorted,
    tsv,
)
from qpdk.tech import LAYER, coplanar_waveguide

from .base import OnePortModelTestSuite, TwoPortModelTestSuite

MAX_EXAMPLES = 20
_compiled_bend_s = jax.jit(bend_s, static_argnames=["npoints"])


@final
class TestStraightWaveguide(TwoPortModelTestSuite):
    """Unit and integration tests for straight waveguide model."""

    model_function = straight

    @staticmethod
    def get_model_kwargs() -> dict:
        """Get model-specific keyword arguments."""
        return {"length": 1000}

    @staticmethod
    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        length=st.floats(min_value=10, max_value=50000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_passivity_hypothesis(f_center: float, length: float) -> None:
        """Test that the waveguide satisfies passivity (energy conservation).

        For a passive two-port network: |S11|^2 + |S21|^2 <= 1

        Args:
            f_center: Center frequency in Hz
            length: Waveguide length in µm
        """
        f = jnp.linspace(f_center * 0.9, f_center * 1.1, 10)
        result = straight(f=f, length=length)

        s11 = result["o1", "o1"]
        s21 = result["o2", "o1"]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert_array_less(
            total_power,
            1.0 + 1e-6,
            err_msg=f"Passivity violated: max total power = {jnp.max(total_power)}",
        )

    @staticmethod
    @given(
        length1=st.floats(min_value=10, max_value=10000),
        length2=st.floats(min_value=10001, max_value=50000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_length_effect(length1: float, length2: float) -> None:
        """Test that longer waveguides have more attenuation.

        Args:
            length1: First waveguide length in µm
            length2: Second waveguide length in µm
        """
        assume(abs(length2 - length1) > 100)

        f = jnp.array([5e9])
        result1 = straight(f=f, length=length1)
        result2 = straight(f=f, length=length2)

        transmission1 = jnp.abs(result1["o2", "o1"])[0]
        transmission2 = jnp.abs(result2["o2", "o1"])[0]

        if length2 > length1:
            assert_array_less(
                transmission2,
                transmission1 + 1e-10,
                err_msg=(
                    f"Longer waveguide should have lower transmission: "
                    f"L1={length1}, |S21|1={transmission1}, "
                    f"L2={length2}, |S21|2={transmission2}"
                ),
            )

    @staticmethod
    def test_frequency_sweep() -> None:
        """Test straight waveguide across typical superconducting RF frequency range."""
        f = jnp.linspace(0.5e9, 10e9, 100)
        length = 5000  # 5 mm

        result = straight(f=f, length=length)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result["o2", "o1"]) == 100, "Should have 100 frequency points"

        s21 = result["o2", "o1"]
        assert jnp.all(jnp.isfinite(s21)), "All S21 values should be finite"
        assert_array_less(
            jnp.abs(s21),
            1.0 + 1e-10,
            err_msg="All |S21| values should be <= 1 (with numerical tolerance)",
        )

    @staticmethod
    def test_custom_cross_section() -> None:
        """Test straight waveguide with custom media parameters."""
        custom_cross_section = coplanar_waveguide(width=20, gap=10)

        f = jnp.array([5e9, 6e9])
        result = straight(f=f, length=1000, cross_section=custom_cross_section)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result["o2", "o1"]) == 2, "Should have 2 frequency points"

        s12 = result["o1", "o2"]
        s21 = result["o2", "o1"]
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    @staticmethod
    def test_zero_length() -> None:
        """Test straight waveguide with zero length (through connection)."""
        f = jnp.array([5e9])
        result = straight(f=f, length=0)

        s21 = result["o2", "o1"]
        transmission = jnp.abs(s21)[0]

        assert transmission > 0.99, (
            f"Zero length should have ~perfect transmission, got {transmission}"
        )


@final
class TestStraightOpen(OnePortModelTestSuite):
    """Tests for straight_open model."""

    model_function = straight_open

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 1000}


@final
class TestStraightShorted(OnePortModelTestSuite):
    """Tests for straight_shorted model."""

    model_function = straight_shorted

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 1000}


@final
class TestStraightDoubleOpen(TwoPortModelTestSuite):
    """Tests for straight_double_open model."""

    model_function = straight_double_open

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 1000}


@final
class TestTSV(TwoPortModelTestSuite):
    """Tests for TSV model."""

    model_function = tsv

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"via_height": 500.0}


@final
class TestIndiumBump(TwoPortModelTestSuite):
    """Tests for indium_bump model."""

    model_function = indium_bump

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"bump_height": 10.0}


@final
class TestBendCircular(TwoPortModelTestSuite):
    """Tests for bend_circular model."""

    model_function = bend_circular

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"angle": -180.0, "radius": 250.0}


@final
class TestBendEuler(TwoPortModelTestSuite):
    """Tests for bend_euler model."""

    model_function = bend_euler

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"angle": 180.0, "p": 0.25, "with_arc_floorplan": False}


@final
class TestBendS(TwoPortModelTestSuite):
    """Tests for bend_s model."""

    model_function = bend_s

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 500}


@given(
    dx=st.floats(min_value=100, max_value=2_000),
    bend_case=st.one_of(
        st.tuples(st.just(0.0), st.integers(min_value=0, max_value=2)),
        st.tuples(
            st.one_of(
                st.floats(min_value=-500, max_value=-1),
                st.floats(min_value=1, max_value=500),
            ),
            st.integers(min_value=3, max_value=199),
        ),
    ),
    frequency=st.floats(min_value=1e9, max_value=12e9),
)
@settings(max_examples=10, deadline=None)
def test_bend_s_serialized_settings_set_physical_length(
    dx: float,
    bend_case: tuple[float, int],
    frequency: float,
) -> None:
    """Consume the layout factory's serialized settings directly."""
    frequencies = jnp.array([frequency])
    dy, npoints = bend_case
    layout = bend_s_cell(
        size=(dx, dy),
        npoints=npoints,
        allow_min_radius_violation=True,
    )

    from_size = _compiled_bend_s(
        f=frequencies,
        size=(dx, dy),
        npoints=npoints,
    )
    from_length = bend_s(
        f=frequencies,
        length=layout.info["length"],
    )

    for key in from_size:
        assert_allclose(from_size[key], from_length[key], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("npoints", [0, 1, 2])
def test_bend_s_zero_offset_uses_straight_shortcut(npoints: int) -> None:
    """Match gdsfactory's zero-offset shortcut for every accepted point count."""
    frequencies = jnp.array([5e9])
    layout = bend_s_cell(size=(100, 0), npoints=npoints)
    from_size = _compiled_bend_s(
        f=frequencies,
        size=(100, 0),
        npoints=npoints,
    )
    from_length = bend_s(f=frequencies, length=layout.info["length"])

    for key in from_size:
        assert_allclose(from_size[key], from_length[key], rtol=1e-6, atol=1e-12)


def test_bend_s_defaults_match_layout_factory() -> None:
    """Keep geometry defaults aligned with the layout factory."""
    model_parameters = inspect.signature(bend_s).parameters
    cell_parameters = inspect.signature(bend_s_cell).parameters

    assert model_parameters["size"].default == cell_parameters["size"].default
    assert model_parameters["npoints"].default == cell_parameters["npoints"].default


def test_bend_s_accepts_serialized_npoints() -> None:
    """Accept the scalar array emitted by gdsfactoryplus model binding."""
    result = bend_s(f=jnp.array([5e9]), npoints=jnp.asarray(99.0))

    assert result


def test_bend_s_length_supports_reverse_mode_differentiation() -> None:
    """Keep S-bend geometry differentiable for circuit optimization."""
    gradient = jax.grad(
        lambda offset: waveguide_models._bend_s_length((20.0, offset), npoints=99)
    )(3.0)

    assert jnp.isfinite(gradient)


_CIRCULAR_BEND_SETTINGS: list[dict[str, Any]] = [
    {},
    {"angle": 45.0},
    {"angle": -90.0},
    {"angle": 180.0},
    {"angle": -135.0},
    # Below the cpw minimum radius (and zero), exercising the clamp and the fallback.
    {"radius": 5.0},
    {"radius": 0.0},
    {"radius": 250.0},
    {"npoints": 32},
    # radius (100) below radius_min (210): the regular cell clamps, the
    # all-angle factory does not.
    {"cross_section": "launcher_cross_section_big"},
]

_EULER_BEND_SETTINGS: list[dict[str, Any]] = [
    {},
    {"angle": 45.0},
    {"angle": -90.0},
    {"angle": 180.0},
    {"angle": -135.0},
    {"radius": 25.0},
    {"radius": 250.0},
    {"p": 0.25},
    {"p": 1.0},
    {"with_arc_floorplan": False},
    {"angle": 45.0, "radius": 30.0, "p": 0.75, "with_arc_floorplan": False},
    {"npoints": 32},
]

_BEND_PAIRS = [
    (
        bend_circular_cell,
        bend_circular,
        # Only the regular layout cells forward angular_step, through **kwargs.
        [
            *_CIRCULAR_BEND_SETTINGS,
            {"angular_step": 45.0},
            {"angle": 180.0, "angular_step": 30.0},
        ],
    ),
    (bend_circular_all_angle_cell, bend_circular_all_angle, _CIRCULAR_BEND_SETTINGS),
    (
        bend_euler_cell,
        bend_euler,
        # gdsfactory rejects angular_step together with npoints, and the regular
        # Euler cell defaults to npoints=720.
        [
            *_EULER_BEND_SETTINGS,
            {"angular_step": 45.0, "npoints": None},
            {"angle": 180.0, "angular_step": 30.0, "npoints": None},
        ],
    ),
    (bend_euler_all_angle_cell, bend_euler_all_angle, _EULER_BEND_SETTINGS),
]


def _case_id(model: Callable[..., Any], geometry: dict[str, Any]) -> str:
    """Render a case as a readable test id."""
    settings = ", ".join(f"{name}={value}" for name, value in geometry.items())
    return f"{model.__name__}-{settings or 'default'}"


# No minimum bend radius, unlike the PDK's coplanar waveguide cross-section.
_NO_RADIUS_MIN_CROSS_SECTION = gf.cross_section.cross_section(
    width=10,
    radius=100,
    radius_min=None,
    sections=(gf.Section(width=6, layer=LAYER.M1_ETCH, name="etch"),),
)

_BEND_CASES = [
    pytest.param(cell, model, geometry, id=_case_id(model, geometry))
    for cell, model, geometries in _BEND_PAIRS
    for geometry in geometries
]


def _assert_matches_layout(
    cell_factory: Callable[..., Any],
    model: Callable[..., Any],
    geometry: dict[str, Any],
) -> None:
    """Compare the model to straight propagation over the materialized cell's length."""
    component = cell_factory(**geometry)
    frequencies = jnp.array([2e9, 5e9, 8e9])
    expected = straight(
        f=frequencies,
        length=component.info["length"],
        cross_section=geometry.get("cross_section", "cpw"),
    )

    results = [
        model(f=frequencies, **geometry),
        jax.jit(partial(model, **geometry))(frequencies),
    ]
    for result in results:
        assert set(result) == set(expected)
        for key in expected:
            assert_allclose(result[key], expected[key], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(("cell_factory", "model", "geometry"), _BEND_CASES)
def test_bend_model_matches_layout_length(
    cell_factory: Callable[..., Any],
    model: Callable[..., Any],
    geometry: dict[str, Any],
) -> None:
    """Every bend model propagates like its cell, eagerly and jitted over frequency."""
    _assert_matches_layout(cell_factory, model, geometry)


@pytest.mark.parametrize(
    ("cell_factory", "model"),
    [(cell, model) for cell, model, _ in _BEND_PAIRS],
    ids=[model.__name__ for _, model, _ in _BEND_PAIRS],
)
def test_bend_model_circuit_overrides_geometry(
    cell_factory: Callable[..., Any], model: Callable[..., Any]
) -> None:
    """Geometry knobs stay live through ``Component.get_netlist()`` -> ``sax.circuit``.

    gdsfactory netlists carry the layout length in the instance ``info`` block,
    which sax merges into the model settings. A ``length`` model parameter would
    let that merged value shadow the geometry settings, so the bend models must
    not expose one.
    """
    frequencies = jnp.array([5e9])
    bend = cell_factory(angle=90.0, radius=250.0)
    component_class = (
        gf.ComponentAllAngle if isinstance(bend, gf.ComponentAllAngle) else gf.Component
    )
    component = component_class()
    reference = component << bend
    component.add_ports(reference.ports)
    netlist = component.get_netlist()
    (instance_name,) = netlist["instances"]
    assert instance_name == model.__name__
    circuit, _ = sax.circuit(netlist, models=models)

    default = circuit(f=frequencies)
    overridden = circuit(
        f=frequencies, **{instance_name: {"angle": 180.0, "radius": 100.0}}
    )

    # The override must move the result, or the check below is vacuous.
    assert not jnp.allclose(default["o2", "o1"], overridden["o2", "o1"])

    expected_settings = [
        (bend, default),
        (cell_factory(angle=180.0, radius=100.0), overridden),
    ]
    for cell, result in expected_settings:
        expected = straight(f=frequencies, length=cell.info["length"])
        for key in expected:
            assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("cell_factory", "model"),
    [
        (bend_circular_cell, bend_circular),
        (bend_circular_all_angle_cell, bend_circular_all_angle),
    ],
    ids=["bend_circular", "bend_circular_all_angle"],
)
@pytest.mark.parametrize("cross_section", ["cpw", "launcher_cross_section_big"])
def test_bend_circular_none_radius_matches_layout_default(
    cell_factory: Callable[..., Any],
    model: Callable[..., Any],
    cross_section: str,
) -> None:
    """A ``None`` radius falls back and then clamps like the cell's default radius.

    On ``launcher_cross_section_big`` the cross-section radius (100) is below
    ``radius_min`` (210), so the fallback result must be clamped too, exactly
    like the layout cell treats its own default radius.
    """
    frequencies = jnp.array([5e9])
    expected = straight(
        f=frequencies,
        length=cell_factory(cross_section=cross_section).info["length"],
        cross_section=cross_section,
    )
    result = model(f=frequencies, radius=None, cross_section=cross_section)

    for key in expected:
        assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)


def test_bend_circular_without_cross_section_radius_min() -> None:
    """Without a cross-section minimum, radii stay unclamped and None falls back."""
    frequencies = jnp.array([5e9])
    cross_section = _NO_RADIUS_MIN_CROSS_SECTION
    unclamped = bend_circular(f=frequencies, radius=5.0, cross_section=cross_section)
    expected = straight(
        f=frequencies,
        length=gf.path.arc(radius=5.0, angle=90.0).length(),
        cross_section=cross_section,
    )
    fallback = bend_circular(f=frequencies, radius=None, cross_section=cross_section)
    at_cross_section_radius = bend_circular(
        f=frequencies, radius=100.0, cross_section=cross_section
    )

    for key in expected:
        assert_allclose(unclamped[key], expected[key], rtol=1e-12, atol=1e-12)
        assert_allclose(
            fallback[key], at_cross_section_radius[key], rtol=1e-12, atol=1e-12
        )


@given(
    angle=st.one_of(
        st.floats(min_value=-180, max_value=-1),
        st.floats(min_value=1, max_value=180),
    ),
    radius=st.floats(min_value=20, max_value=500),
    p=st.sampled_from([0.25, 0.5, 0.75, 1.0]),
    with_arc_floorplan=st.booleans(),
)
@settings(max_examples=10, deadline=None)
def test_bend_euler_model_matches_layout_length_property(
    angle: float, radius: float, p: float, with_arc_floorplan: bool
) -> None:
    """Sweep continuous angles and radii against the Euler layout curve."""
    geometry = {
        "angle": angle,
        "radius": radius,
        "p": p,
        "with_arc_floorplan": with_arc_floorplan,
    }
    _assert_matches_layout(bend_euler_cell, bend_euler, geometry)


def test_bend_models_activate_pdk_themselves() -> None:
    """The bend models work in a fresh interpreter with no PDK activated.

    The suite cannot see this: ``tests/conftest.py`` activates the PDK at
    collection time, so the subprocess starts from a clean slate.
    """
    code = """\
import jax.numpy as jnp
from qpdk.models.waveguides import (
    bend_circular,
    bend_circular_all_angle,
    bend_euler,
    bend_euler_all_angle,
)
for model in (bend_circular, bend_circular_all_angle, bend_euler, bend_euler_all_angle):
    s21 = model(f=jnp.array([5e9]))['o2', 'o1']
    assert abs(s21) > 0.9, s21
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


@final
class TestRectangle(TwoPortModelTestSuite):
    """Tests for rectangle model."""

    model_function = rectangle

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 500}


@final
class TestNxN:
    """Tests for nxn model."""

    @staticmethod
    def test_n_ports_assignment() -> None:
        """Test that nxn model has the correct number of ports."""
        f = jnp.array([1e9])
        for n in range(1, 6):
            # Sum of ports = n
            result = nxn(f=f, west=n, east=0, north=0, south=0)
            assert isinstance(result, dict)
            # An N-port model has N*N S-parameters
            # Let's check the number of distinct port names in the keys
            ports = set()
            for p1, p2 in result:
                ports.add(p1)
                ports.add(p2)
            assert len(ports) == n, f"Expected {n} ports, got {len(ports)}"

    @staticmethod
    def test_passivity() -> None:
        """Test that nxn model is passive."""
        f = jnp.linspace(1e9, 10e9, 10)
        n = 4
        result = nxn(f=f, west=1, east=1, north=1, south=1)

        for j in range(1, n + 1):
            total_power = jnp.zeros_like(f)
            for i in range(1, n + 1):
                s_ij = result[f"o{i}", f"o{j}"]
                total_power += jnp.abs(s_ij) ** 2
            assert_array_less(
                total_power,
                1.0 + 1e-6,
                err_msg=f"Passivity violated for port o{j}: max power = {jnp.max(total_power)}",
            )

    @staticmethod
    def test_reciprocity() -> None:
        """Test that nxn model is reciprocal."""
        f = jnp.array([1e9, 5e9, 10e9])
        n = 3
        result = nxn(f=f, west=1, east=1, north=1, south=0)

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                s_ij = result[f"o{i}", f"o{j}"]
                s_ji = result[f"o{j}", f"o{i}"]
                assert_allclose(
                    s_ij,
                    s_ji,
                    atol=1e-10,
                    err_msg=f"Reciprocity violated between o{i} and o{j}",
                )

    @staticmethod
    @given(
        west=st.integers(min_value=0, max_value=2),
        east=st.integers(min_value=0, max_value=2),
        north=st.integers(min_value=0, max_value=2),
        south=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_with_hypothesis(west: int, east: int, north: int, south: int) -> None:
        """Test nxn model with random port counts using hypothesis."""
        n = west + east + north + south
        assume(n > 0)

        f = jnp.array([1e9, 10e9])
        result = nxn(f=f, west=west, east=east, north=north, south=south)

        # Check port count by looking at unique port names in S-parameter keys
        ports = set()
        for p1, p2 in result:
            ports.add(p1)
            ports.add(p2)
        assert len(ports) == n, f"Expected {n} ports, got {len(ports)} ({ports})"

        # Verify passivity for the first port (o1)
        total_power = jnp.zeros_like(f)
        for i in range(1, n + 1):
            s_i1 = result[f"o{i}", "o1"]
            total_power += jnp.abs(s_i1) ** 2
        assert_array_less(
            total_power,
            1.0 + 1e-6,
            err_msg=f"Passivity violated for port o1 with N={n}: max power = {jnp.max(total_power)}",
        )


class TestNxNEdgeCases:
    """Tests for nxn model edge cases."""

    @staticmethod
    def test_zero_ports_raises() -> None:
        """Test that nxn with 0 total ports raises ValueError."""
        f = jnp.array([5e9])
        with pytest.raises(ValueError, match="Total number of ports must be positive"):
            nxn(f=f, west=0, east=0, north=0, south=0)

    @staticmethod
    def test_single_port() -> None:
        """Test nxn with single port returns electrical_open."""
        f = jnp.array([5e9])
        result = nxn(f=f, west=1, east=0, north=0, south=0)
        assert isinstance(result, dict)
        ports = set()
        for p1, p2 in result:
            ports.add(p1)
            ports.add(p2)
        assert len(ports) == 1

    @staticmethod
    def test_two_ports() -> None:
        """Test nxn with two ports returns electrical_short."""
        f = jnp.array([5e9])
        result = nxn(f=f, west=1, east=1, north=0, south=0)
        assert isinstance(result, dict)
        ports = set()
        for p1, p2 in result:
            ports.add(p1)
            ports.add(p2)
        assert len(ports) == 2


@final
class TestAirbridge(TwoPortModelTestSuite):
    """Tests for airbridge model."""

    model_function = airbridge

    @staticmethod
    def get_model_kwargs() -> dict:
        """Get model-specific keyword arguments."""
        return {"cpw_width": 10.0, "bridge_width": 10.0, "airgap_height": 3.0}
