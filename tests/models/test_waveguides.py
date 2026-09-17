"""Tests for qpdk.models.waveguides module."""

import inspect
from collections.abc import Callable
from typing import final

import gdsfactory as gf
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import assume, given, settings
from numpy.testing import assert_allclose, assert_array_less

from qpdk.cells.waveguides import bend_s as bend_s_cell, straight as straight_cell
from qpdk.models import waveguides as waveguide_models
from qpdk.models.cpw import get_cpw_dimensions
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
    tsv,
)
from qpdk.tech import coplanar_waveguide

from .base import TwoPortModelTestSuite

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


@pytest.mark.parametrize("width", [5.0, 15.0])
def test_straight_width_override_uses_extruded_geometry(
    width: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass the layout's extruded CPW dimensions to the SAX model."""
    layout = straight_cell(width=width)
    settings = layout.settings.model_dump()
    cross_section = gf.get_cross_section(
        settings["cross_section"], width=settings["width"]
    )
    expected_width, expected_gap = get_cpw_dimensions(cross_section)
    model_settings: dict[str, object] = {}

    def capture_model_settings(**kwargs: object) -> dict[object, object]:
        model_settings.update(kwargs)
        return {}

    monkeypatch.setattr(
        waveguide_models, "_sax_coplanar_waveguide", capture_model_settings
    )
    straight(
        f=jnp.array([5e9]),
        length=settings["length"],
        cross_section=settings["cross_section"],
        width=settings["width"],
    )

    assert model_settings["width"] == expected_width
    assert model_settings["gap"] == expected_gap


@pytest.mark.parametrize(
    "model",
    [bend_circular, bend_circular_all_angle, bend_euler, bend_euler_all_angle],
)
def test_bend_width_override_is_forwarded(
    model: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward bend width settings to their straight model."""
    model_settings: dict[str, object] = {}

    def capture_model_settings(**kwargs: object) -> dict[object, object]:
        model_settings.update(kwargs)
        return {}

    monkeypatch.setattr(waveguide_models, "straight", capture_model_settings)
    model(f=jnp.array([5e9]), length=100, width=3.0)

    assert model_settings["width"] == pytest.approx(3.0)


@final
class TestStraightOpen(TwoPortModelTestSuite):
    """Tests for straight_open model."""

    model_function = straight_open

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
        return {"length": 500}


@final
class TestBendEuler(TwoPortModelTestSuite):
    """Tests for bend_euler model."""

    model_function = bend_euler

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 500}


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
    width_dbu=st.integers(min_value=1_000, max_value=15_000),
    frequency=st.floats(min_value=1e9, max_value=12e9),
)
@settings(max_examples=10, deadline=None)
def test_bend_s_serialized_settings_set_physical_length(
    dx: float,
    bend_case: tuple[float, int],
    width_dbu: int,
    frequency: float,
) -> None:
    """Consume the layout factory's serialized settings directly."""
    frequencies = jnp.array([frequency])
    dy, npoints = bend_case
    width = width_dbu * 0.002
    layout = bend_s_cell(
        size=(dx, dy),
        npoints=npoints,
        width=width,
        allow_min_radius_violation=True,
    )

    from_size = _compiled_bend_s(
        f=frequencies,
        size=(dx, dy),
        npoints=npoints,
        width=width,
    )
    from_length = bend_s(
        f=frequencies,
        length=layout.info["length"],
        width=width,
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
