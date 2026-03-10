"""Additional tests for qpdk.models.waveguides module - covering missing lines."""

from typing import final

import jax.numpy as jnp
import pytest

from qpdk.models.waveguides import (
    bend_circular,
    bend_euler,
    bend_s,
    indium_bump,
    nxn,
    rectangle,
    straight_double_open,
    straight_open,
    tsv,
)

from .base import TwoPortModelTestSuite


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


@final
class TestRectangle(TwoPortModelTestSuite):
    """Tests for rectangle model."""

    model_function = rectangle

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 500}


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
