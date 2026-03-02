"""Base test suite class for network model tests.

This module provides a common base class (`BaseModelTestSuite`) that implements
standard tests for S-parameter models like default parameters, output shape,
reciprocity, and passivity checks.
"""

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
import sax


class BaseModelTestSuite:
    """Base class for testing S-parameter network models.

    This class provides a common test interface for validating S-parameter models
    following the sax.SType interface. Subclasses should:

    1. Set the class attributes:
       - `model_function`: The model function to test (use staticmethod)
       - `expected_ports`: Set of expected port names (e.g., {"o1", "o2"})
       - Optionally `n_freq_default`: Number of frequency points for tests (default: 10)
       - Optionally `freq_range`: Tuple of (f_min, f_max) in Hz
       - Optionally `tolerance`: Numerical tolerance for reciprocity/passivity checks
       - Optionally `passivity_tolerance`: Additional tolerance for passivity checks

    2. Optionally override `get_model_kwargs()` to provide model-specific arguments

    Example:
        >>> @final
        ... class TestMyModel(BaseModelTestSuite):
        ...     model_function = staticmethod(my_model)
        ...     expected_ports = {"o1", "o2"}
        ...
        ...     def get_model_kwargs(self) -> dict:
        ...         return {"length": 1000}
    """

    # Class attributes to be set by subclasses
    model_function: ClassVar[Callable[..., sax.SType]]
    expected_ports: ClassVar[set[str]]
    n_freq_default: ClassVar[int] = 10
    freq_range: ClassVar[tuple[float, float]] = (4e9, 8e9)
    tolerance: ClassVar[float] = 1e-9
    passivity_tolerance: ClassVar[float] = 1e-6

    @classmethod
    def get_expected_keys(cls) -> set[tuple[str, str]]:
        """Generate expected S-parameter keys from port names.

        Returns:
            Set of tuples representing all port pair combinations.
        """
        return {(p1, p2) for p1 in cls.expected_ports for p2 in cls.expected_ports}

    def get_model_kwargs(self) -> dict:
        """Get additional keyword arguments to pass to the model function.

        Override this method to provide model-specific default arguments.

        Returns:
            Dictionary of keyword arguments for the model.
        """
        return {}

    def get_frequency_array(self, n_points: int | None = None) -> jnp.ndarray:
        """Generate a frequency array for testing.

        Subclasses can override to customize frequency range.

        Args:
            n_points: Number of frequency points. Uses n_freq_default if None.

        Returns:
            JAX array of frequency values in Hz.
        """
        n = n_points if n_points is not None else self.n_freq_default
        return jnp.linspace(*self.freq_range, n)

    def _call_model(self, **kwargs) -> sax.SType:
        """Call the model function with the given keyword arguments.

        This method ensures the model function is called correctly without
        passing `self` as the first argument.
        """
        return type(self).model_function(**kwargs)

    def test_default_parameters(self) -> None:
        """Test that the model returns valid S-parameters with default parameters."""
        result = self._call_model()

        assert isinstance(result, dict), "Result should be a dictionary"

        expected_keys = self.get_expected_keys()
        for key in expected_keys:
            assert key in result, f"Should have {key} parameter"

    def test_returns_stype(self) -> None:
        """Test that the model returns a valid sax.SType dictionary."""
        f = self.get_frequency_array(3)
        kwargs = self.get_model_kwargs()
        result = self._call_model(f=f, **kwargs)

        assert isinstance(result, dict), "Result should be a dictionary"

        expected_keys = self.get_expected_keys()
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

        for key, value in result.items():
            assert hasattr(value, "__len__"), f"Value for {key} should be array-like"

    def test_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = self.n_freq_default
        f = self.get_frequency_array(n_freq)
        kwargs = self.get_model_kwargs()
        result = self._call_model(f=f, **kwargs)

        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (Sij = Sji)."""
        f = self.get_frequency_array(50)
        kwargs = self.get_model_kwargs()
        result = self._call_model(f=f, **kwargs)

        ports = list(self.expected_ports)
        for i, port_i in enumerate(ports):
            for port_j in ports[i + 1 :]:
                sij = result.get((port_i, port_j))
                sji = result.get((port_j, port_i))

                if sij is not None and sji is not None:
                    max_diff = jnp.max(jnp.abs(sij - sji))
                    assert max_diff < self.tolerance, (
                        f"S[{port_i},{port_j}] and S[{port_j},{port_i}] should be equal, "
                        f"max diff: {max_diff}"
                    )

    def test_passivity(self) -> None:
        """Test that the model satisfies passivity (energy conservation).

        For a passive N-port network, for each column j of the S-matrix:
        sum_i |S_ij|^2 <= 1
        """
        f = self.get_frequency_array(100)
        kwargs = self.get_model_kwargs()
        result = self._call_model(f=f, **kwargs)

        ports = list(self.expected_ports)
        for port_j in ports:
            # Sum power from all rows for this column
            power_sum = sum(
                jnp.abs(result.get((port_i, port_j), 0)) ** 2 for port_i in ports
            )
            assert jnp.all(power_sum <= 1.0 + self.passivity_tolerance), (
                f"Passivity violated for column {port_j}: "
                f"max total power = {jnp.max(power_sum)}"
            )


class TwoPortModelTestSuite(BaseModelTestSuite):
    """Base test suite for 2-port network models.

    Provides default settings for models with ports "o1" and "o2".
    """

    expected_ports: ClassVar[set[str]] = {"o1", "o2"}


class FourPortModelTestSuite(BaseModelTestSuite):
    """Base test suite for 4-port network models.

    Provides default settings for models with ports "o1", "o2", "o3", "o4".
    """

    expected_ports: ClassVar[set[str]] = {"o1", "o2", "o3", "o4"}
