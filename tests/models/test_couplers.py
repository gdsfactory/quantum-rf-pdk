"""Tests for qpdk.models.couplers module."""

import jax.numpy as jnp

from qpdk.models.couplers import coupler_ring


class TestCouplerRing:
    """Tests for coupler_ring model."""

    def test_coupler_ring_default_parameters(self) -> None:
        """Test coupler_ring with default parameters."""
        result = coupler_ring()
        assert isinstance(result, dict)
        expected_ports = {"o1", "o2", "o3", "o4"}
        for p1 in expected_ports:
            for p2 in expected_ports:
                assert (p1, p2) in result

    def test_coupler_ring_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 5
        f = jnp.linspace(4e9, 6e9, n_freq)
        result = coupler_ring(f=f)
        for value in result.values():
            assert len(value) == n_freq

    def test_coupler_ring_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        f = jnp.linspace(4e9, 6e9, 10)
        result = coupler_ring(f=f)
        # Check a few reciprocal pairs
        assert jnp.allclose(result[("o1", "o4")], result[("o4", "o1")], atol=1e-10)
        assert jnp.allclose(result[("o2", "o3")], result[("o3", "o2")], atol=1e-10)
        assert jnp.allclose(result[("o1", "o2")], result[("o2", "o1")], atol=1e-10)

    def test_coupler_ring_passivity(self) -> None:
        """Test that the coupler satisfies passivity."""
        f = jnp.linspace(4e9, 6e9, 10)
        result = coupler_ring(f=f)
        # For a 4-port network, sum of power in each row of S-matrix <= 1
        for i in ["o1", "o2", "o3", "o4"]:
            power = sum(
                jnp.abs(result.get((i, j), 0)) ** 2 for j in ["o1", "o2", "o3", "o4"]
            )
            assert jnp.all(power <= 1.0 + 1e-6)
