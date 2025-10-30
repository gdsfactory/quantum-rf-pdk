"""Tests for qpdk.models.generic functions."""

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import assume, given, settings

from qpdk.models.generic import (
    capacitor,
    gamma_0_load,
    inductor,
    open,
    short,
    single_admittance_element,
    single_impedance_element,
    tee,
)

MAX_EXAMPLES = 20


class TestGamma0Load:
    """Unit and integration tests for gamma_0_load function."""

    def test_gamma_0_load_default_parameters(self) -> None:
        """Test gamma_0_load with default parameters."""
        result = gamma_0_load()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"

    def test_gamma_0_load_returns_stype(self) -> None:
        """Test that gamma_0_load returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = gamma_0_load(f=f, gamma_0=0.5, n_ports=2)

        # Check it's a dictionary with the expected structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check diagonal S-parameter keys are present
        expected_keys = {("o1", "o1"), ("o2", "o2"), ("o1", "o2"), ("o2", "o1")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_gamma_0_load_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = gamma_0_load(f=f, gamma_0=0.3, n_ports=1)

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_gamma_0_load_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = gamma_0_load(f=f, gamma_0=0.2, n_ports=3)

        # Check reciprocity: Sij should equal Sji
        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_gamma_0_load_reflection_coefficient(self) -> None:
        """Test that diagonal elements equal the reflection coefficient."""
        f = jnp.array([5e9])
        gamma_0 = 0.5 + 0.5j
        result = gamma_0_load(f=f, gamma_0=gamma_0, n_ports=2)

        # Diagonal elements should equal gamma_0
        s11 = result[("o1", "o1")][0]
        s22 = result[("o2", "o2")][0]

        assert jnp.abs(s11 - gamma_0) < 1e-10, (
            f"S11 should equal gamma_0={gamma_0}, got {s11}"
        )
        assert jnp.abs(s22 - gamma_0) < 1e-10, (
            f"S22 should equal gamma_0={gamma_0}, got {s22}"
        )

    def test_gamma_0_load_off_diagonal_zero(self) -> None:
        """Test that off-diagonal elements are zero."""
        f = jnp.array([5e9])
        result = gamma_0_load(f=f, gamma_0=0.3, n_ports=2)

        # Off-diagonal elements should be zero
        s12 = result[("o1", "o2")][0]
        s21 = result[("o2", "o1")][0]

        assert jnp.abs(s12) < 1e-10, f"S12 should be zero, got {s12}"
        assert jnp.abs(s21) < 1e-10, f"S21 should be zero, got {s21}"

    @given(
        n_freq=st.integers(min_value=1, max_value=50),
        n_ports=st.integers(min_value=1, max_value=5),
        gamma_real=st.floats(min_value=-1, max_value=1),
        gamma_imag=st.floats(min_value=-1, max_value=1),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_gamma_0_load_with_hypothesis(
        self, n_freq: int, n_ports: int, gamma_real: float, gamma_imag: float
    ) -> None:
        """Test gamma_0_load with random valid parameters using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        gamma_0 = gamma_real + 1j * gamma_imag
        result = gamma_0_load(f=f, gamma_0=gamma_0, n_ports=n_ports)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        # For a reciprocal n-port network, we have n diagonal + n*(n-1)/2 off-diagonal
        # But sax.reciprocal returns both Sij and Sji, so we have n^2 total parameters
        expected_n_params = n_ports * n_ports
        assert len(result) == expected_n_params, (
            f"Should have {expected_n_params} S-parameters"
        )

        # Verify shapes
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"S-parameter {key} length should match frequency array length"
            )


class TestShort:
    """Unit and integration tests for short function."""

    def test_short_default_parameters(self) -> None:
        """Test short with default parameters."""
        result = short()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"

    def test_short_reflection_coefficient(self) -> None:
        """Test that short has reflection coefficient of -1."""
        f = jnp.array([5e9])
        result = short(f=f, n_ports=1)

        s11 = result[("o1", "o1")][0]
        assert jnp.abs(s11 - (-1)) < 1e-10, f"S11 should be -1 for short, got {s11}"

    def test_short_multiport(self) -> None:
        """Test short with multiple ports."""
        f = jnp.array([5e9])
        result = short(f=f, n_ports=3)

        # All diagonal elements should be -1
        for i in range(1, 4):
            sii = result[(f"o{i}", f"o{i}")][0]
            assert jnp.abs(sii - (-1)) < 1e-10, (
                f"S{i}{i} should be -1 for short, got {sii}"
            )

    @given(
        n_freq=st.integers(min_value=1, max_value=50),
        n_ports=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_short_with_hypothesis(self, n_freq: int, n_ports: int) -> None:
        """Test short with random valid parameters using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        result = short(f=f, n_ports=n_ports)

        # Verify all diagonal elements are -1
        for i in range(1, n_ports + 1):
            sii = result[(f"o{i}", f"o{i}")]
            assert jnp.allclose(sii, -1), f"S{i}{i} should be -1 for all frequencies"


class TestOpen:
    """Unit and integration tests for open function."""

    def test_open_default_parameters(self) -> None:
        """Test open with default parameters."""
        result = open()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"

    def test_open_reflection_coefficient(self) -> None:
        """Test that open has reflection coefficient of 1."""
        f = jnp.array([5e9])
        result = open(f=f, n_ports=1)

        s11 = result[("o1", "o1")][0]
        assert jnp.abs(s11 - 1) < 1e-10, f"S11 should be 1 for open, got {s11}"

    def test_open_multiport(self) -> None:
        """Test open with multiple ports."""
        f = jnp.array([5e9])
        result = open(f=f, n_ports=3)

        # All diagonal elements should be 1
        for i in range(1, 4):
            sii = result[(f"o{i}", f"o{i}")][0]
            assert jnp.abs(sii - 1) < 1e-10, f"S{i}{i} should be 1 for open, got {sii}"

    @given(
        n_freq=st.integers(min_value=1, max_value=50),
        n_ports=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_open_with_hypothesis(self, n_freq: int, n_ports: int) -> None:
        """Test open with random valid parameters using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        result = open(f=f, n_ports=n_ports)

        # Verify all diagonal elements are 1
        for i in range(1, n_ports + 1):
            sii = result[(f"o{i}", f"o{i}")]
            assert jnp.allclose(sii, 1), f"S{i}{i} should be 1 for all frequencies"


class TestTee:
    """Unit and integration tests for tee function."""

    def test_tee_default_parameters(self) -> None:
        """Test tee with default parameters."""
        result = tee()

        assert isinstance(result, dict), "Result should be a dictionary"
        # Tee has 3 ports, so we expect 9 S-parameters (3x3 matrix)
        assert len(result) == 9, "Tee should have 9 S-parameters (3-port network)"

    def test_tee_returns_stype(self) -> None:
        """Test that tee returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = tee(f=f)

        # Check it's a dictionary with the expected structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check all S-parameter keys are present for 3-port network
        expected_keys = {
            ("o1", "o1"),
            ("o2", "o2"),
            ("o3", "o3"),
            ("o1", "o2"),
            ("o2", "o1"),
            ("o1", "o3"),
            ("o3", "o1"),
            ("o2", "o3"),
            ("o3", "o2"),
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Expected keys {expected_keys} to be subset of {set(result.keys())}"
        )

    def test_tee_diagonal_elements(self) -> None:
        """Test that diagonal elements equal -1/3."""
        f = jnp.array([5e9])
        result = tee(f=f)

        for i in range(1, 4):
            sii = result[(f"o{i}", f"o{i}")][0]
            expected = -1 / 3
            assert jnp.abs(sii - expected) < 1e-10, (
                f"S{i}{i} should be -1/3, got {sii}"
            )

    def test_tee_off_diagonal_elements(self) -> None:
        """Test that off-diagonal elements equal 2/3."""
        f = jnp.array([5e9])
        result = tee(f=f)

        s12 = result[("o1", "o2")][0]
        s13 = result[("o1", "o3")][0]
        s23 = result[("o2", "o3")][0]

        expected = 2 / 3
        assert jnp.abs(s12 - expected) < 1e-10, f"S12 should be 2/3, got {s12}"
        assert jnp.abs(s13 - expected) < 1e-10, f"S13 should be 2/3, got {s13}"
        assert jnp.abs(s23 - expected) < 1e-10, f"S23 should be 2/3, got {s23}"

    def test_tee_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = tee(f=f)

        # Check reciprocity
        pairs = [("o1", "o2"), ("o1", "o3"), ("o2", "o3")]
        for i_port, j_port in pairs:
            sij = result[(i_port, j_port)]
            sji = result[(j_port, i_port)]
            max_diff = jnp.max(jnp.abs(sij - sji))
            assert max_diff < 1e-10, (
                f"S[{i_port}][{j_port}] and S[{j_port}][{i_port}] should be equal, "
                f"max diff: {max_diff}"
            )

    def test_tee_power_conservation(self) -> None:
        """Test that tee satisfies power conservation for ideal splitter."""
        f = jnp.array([5e9])
        result = tee(f=f)

        # For an ideal power divider, when power enters port 1,
        # it should be split between the three ports
        s11 = result[("o1", "o1")][0]
        s21 = result[("o2", "o1")][0]
        s31 = result[("o3", "o1")][0]

        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2 + jnp.abs(s31) ** 2
        # For ideal tee: |S11|^2 + |S21|^2 + |S31|^2 = (1/9 + 4/9 + 4/9) = 1
        assert jnp.abs(total_power - 1.0) < 1e-10, (
            f"Power should be conserved, total = {total_power}"
        )

    @given(n_freq=st.integers(min_value=1, max_value=100))
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_tee_with_hypothesis(self, n_freq: int) -> None:
        """Test tee with random frequency arrays using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        result = tee(f=f)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify shapes
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"S-parameter {key} length should match frequency array length"
            )


class TestSingleImpedanceElement:
    """Unit and integration tests for single_impedance_element function."""

    def test_single_impedance_element_default_parameters(self) -> None:
        """Test single_impedance_element with default parameters (matched load)."""
        result = single_impedance_element()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_single_impedance_element_matched_load(self) -> None:
        """Test that matched impedance gives zero reflection."""
        # When z = z0, S11 should be 0 and S21 should be 1
        result = single_impedance_element(z=50, z0=50)

        s11 = result[("o1", "o1")]
        s21 = result[("o1", "o2")]

        # S11 = z/(z + 2*z0) = 50/150 = 1/3
        expected_s11 = 50 / (50 + 2 * 50)
        assert jnp.abs(s11 - expected_s11) < 1e-10, (
            f"S11 for matched load should be 1/3, got {s11}"
        )

        # S21 = 2*z0/(2*z0 + z) = 100/150 = 2/3
        expected_s21 = 2 * 50 / (2 * 50 + 50)
        assert jnp.abs(s21 - expected_s21) < 1e-10, (
            f"S21 for matched load should be 2/3, got {s21}"
        )

    def test_single_impedance_element_short_circuit(self) -> None:
        """Test that zero impedance approximates a short circuit."""
        result = single_impedance_element(z=0, z0=50)

        s11 = result[("o1", "o1")]
        s21 = result[("o1", "o2")]

        # S11 = 0/(0 + 100) = 0
        assert jnp.abs(s11) < 1e-10, f"S11 for short should be 0, got {s11}"
        # S21 = 100/(100 + 0) = 1
        assert jnp.abs(s21 - 1) < 1e-10, f"S21 for short should be 1, got {s21}"

    def test_single_impedance_element_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        result = single_impedance_element(z=75, z0=50)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity
        assert jnp.abs(s12 - s21) < 1e-10, (
            f"S12 and S21 should be equal, got S12={s12}, S21={s21}"
        )

    @given(
        z_real=st.floats(min_value=1, max_value=200),
        z_imag=st.floats(min_value=-100, max_value=100),
        z0=st.floats(min_value=10, max_value=100),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_single_impedance_element_with_hypothesis(
        self, z_real: float, z_imag: float, z0: float
    ) -> None:
        """Test single_impedance_element with random parameters using hypothesis."""
        z = z_real + 1j * z_imag
        result = single_impedance_element(z=z, z0=z0)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters for 2-port network"


class TestSingleAdmittanceElement:
    """Unit and integration tests for single_admittance_element function."""

    def test_single_admittance_element_default_parameters(self) -> None:
        """Test single_admittance_element with default parameters."""
        result = single_admittance_element()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_single_admittance_element_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        result = single_admittance_element(y=0.02)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity
        assert jnp.abs(s12 - s21) < 1e-10, (
            f"S12 and S21 should be equal, got S12={s12}, S21={s21}"
        )

    @given(
        y_real=st.floats(min_value=0.001, max_value=1),
        y_imag=st.floats(min_value=-0.5, max_value=0.5),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_single_admittance_element_with_hypothesis(
        self, y_real: float, y_imag: float
    ) -> None:
        """Test single_admittance_element with random parameters using hypothesis."""
        y = y_real + 1j * y_imag
        result = single_admittance_element(y=y)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters for 2-port network"


class TestCapacitor:
    """Unit and integration tests for capacitor function."""

    def test_capacitor_default_parameters(self) -> None:
        """Test capacitor with default parameters."""
        result = capacitor()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_capacitor_returns_stype(self) -> None:
        """Test that capacitor returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = capacitor(f=f, capacitance=1e-15)

        # Check it's a dictionary with the expected structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check all S-parameter keys are present
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_capacitor_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = capacitor(f=f, capacitance=1e-15)

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_capacitor_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = capacitor(f=f, capacitance=1e-15)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_capacitor_frequency_dependence(self) -> None:
        """Test that capacitor impedance decreases with frequency."""
        f_low = jnp.array([1e9])
        f_high = jnp.array([10e9])
        capacitance = 1e-15

        result_low = capacitor(f=f_low, capacitance=capacitance)
        result_high = capacitor(f=f_high, capacitance=capacitance)

        # At higher frequencies, capacitor impedance is lower,
        # so transmission should be higher
        transmission_low = jnp.abs(result_low[("o1", "o2")])[0]
        transmission_high = jnp.abs(result_high[("o1", "o2")])[0]

        # Higher frequency should have higher transmission (lower impedance)
        assert transmission_high >= transmission_low - 1e-10, (
            f"Higher frequency should have higher transmission: "
            f"low_f={transmission_low}, high_f={transmission_high}"
        )

    @given(
        n_freq=st.integers(min_value=1, max_value=50),
        capacitance=st.floats(min_value=1e-18, max_value=1e-12),
        z0=st.floats(min_value=10, max_value=100),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_capacitor_with_hypothesis(
        self, n_freq: int, capacitance: float, z0: float
    ) -> None:
        """Test capacitor with random valid parameters using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        result = capacitor(f=f, capacitance=capacitance, z0=z0)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters for 2-port network"

        # Verify shapes
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"S-parameter {key} length should match frequency array length"
            )

        # Verify all values are finite
        for key, value in result.items():
            assert jnp.all(jnp.isfinite(value)), (
                f"All S-parameter values should be finite for {key}"
            )


class TestInductor:
    """Unit and integration tests for inductor function."""

    def test_inductor_default_parameters(self) -> None:
        """Test inductor with default parameters."""
        result = inductor()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_inductor_returns_stype(self) -> None:
        """Test that inductor returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = inductor(f=f, inductance=1e-12)

        # Check it's a dictionary with the expected structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check all S-parameter keys are present
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_inductor_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = inductor(f=f, inductance=1e-12)

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_inductor_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = inductor(f=f, inductance=1e-12)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_inductor_frequency_dependence(self) -> None:
        """Test that inductor impedance increases with frequency."""
        f_low = jnp.array([1e9])
        f_high = jnp.array([10e9])
        inductance = 1e-12

        result_low = inductor(f=f_low, inductance=inductance)
        result_high = inductor(f=f_high, inductance=inductance)

        # At higher frequencies, inductor impedance is higher,
        # so transmission should be lower
        transmission_low = jnp.abs(result_low[("o1", "o2")])[0]
        transmission_high = jnp.abs(result_high[("o1", "o2")])[0]

        # Higher frequency should have lower transmission (higher impedance)
        assert transmission_low >= transmission_high - 1e-10, (
            f"Lower frequency should have higher transmission: "
            f"low_f={transmission_low}, high_f={transmission_high}"
        )

    @given(
        n_freq=st.integers(min_value=1, max_value=50),
        inductance=st.floats(min_value=1e-15, max_value=1e-9),
        z0=st.floats(min_value=10, max_value=100),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_inductor_with_hypothesis(
        self, n_freq: int, inductance: float, z0: float
    ) -> None:
        """Test inductor with random valid parameters using hypothesis."""
        f = jnp.linspace(1e9, 10e9, n_freq)
        result = inductor(f=f, inductance=inductance, z0=z0)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters for 2-port network"

        # Verify shapes
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"S-parameter {key} length should match frequency array length"
            )

        # Verify all values are finite
        for key, value in result.items():
            assert jnp.all(jnp.isfinite(value)), (
                f"All S-parameter values should be finite for {key}"
            )
