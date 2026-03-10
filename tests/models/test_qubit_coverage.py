"""Additional tests for qpdk.models.qubit module - wrapper function coverage."""

from typing import final

import jax.numpy as jnp

from qpdk.models.qubit import (
    double_island_transmon_with_bbox,
    double_island_transmon_with_resonator,
    flipmon,
    flipmon_with_bbox,
    flipmon_with_resonator,
    transmon_with_resonator,
    xmon_transmon,
)

from .base import TwoPortModelTestSuite


@final
class TestDoubleIslandTransmonWithBbox(TwoPortModelTestSuite):
    """Tests for double_island_transmon_with_bbox wrapper (line 201)."""

    model_function = staticmethod(double_island_transmon_with_bbox)


@final
class TestFlipmon(TwoPortModelTestSuite):
    """Tests for flipmon wrapper (line 218)."""

    model_function = staticmethod(flipmon)


@final
class TestFlipmonWithBbox(TwoPortModelTestSuite):
    """Tests for flipmon_with_bbox wrapper (line 235)."""

    model_function = staticmethod(flipmon_with_bbox)


@final
class TestXmonTransmon(TwoPortModelTestSuite):
    """Tests for xmon_transmon wrapper (line 536)."""

    model_function = staticmethod(xmon_transmon)


class TestQubitWithResonatorWrappers:
    """Tests for qubit-resonator wrapper functions (lines 468, 491, 515)."""

    @staticmethod
    def test_flipmon_with_resonator() -> None:
        """Test flipmon_with_resonator returns valid S-params (line 468)."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = flipmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_double_island_transmon_with_resonator() -> None:
        """Test double_island_transmon_with_resonator returns valid S-params (line 491)."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = double_island_transmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_transmon_with_resonator_wrapper() -> None:
        """Test transmon_with_resonator wrapper returns valid S-params (line 515)."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = transmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_transmon_with_resonator_grounded() -> None:
        """Test transmon_with_resonator with grounded qubit."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = transmon_with_resonator(f=f, qubit_grounded=True)
        assert isinstance(result, dict)
        # Grounded produces 1 port (o1 only)
        expected_keys = {("o1", "o1")}
        assert set(result.keys()) == expected_keys
