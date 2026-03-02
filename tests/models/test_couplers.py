"""Tests for qpdk.models.couplers module."""

from typing import final

from qpdk.models.couplers import coupler_ring

from .base import FourPortModelTestSuite


@final
class TestCouplerRing(FourPortModelTestSuite):
    """Tests for coupler_ring model."""

    model_function = coupler_ring
