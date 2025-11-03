"""Tests comparing S-parameter models results to Qucs-S results."""

from abc import ABC, abstractmethod
from functools import partial
from queue import Queue, SimpleQueue
from typing import ClassVar, final, override

import jax.numpy as jnp
import polars as pl
import pytest
import sax
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from qpdk.config import PATH
from qpdk.models.couplers import coupler_straight
from qpdk.models.generic import capacitor, inductor
from qpdk.models.media import cpw_media_skrf
from qpdk.models.waveguides import straight

TEST_DATA_PATH = PATH.tests / "models" / "data"

NUMERIC_TOLERANCES = {
    "rtol": 1 / 100,
    "atol": 0.001,
}


# Type alias for S-parameter comparison results
type SComparisonResults = tuple[
    float, jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]
]


class BaseCompareToQucs(ABC):
    """Base class for comparing S-parameter models to Qucs-S results."""

    # Subclasses should override these
    component_name: str = "Component"
    csv_filename: str = "component_qucs.csv"
    parameter_value: float = 0.0
    parameter_name: str = "parameter"
    parameter_unit: float = 1e-9
    skip_values: ClassVar[list[str]] = []

    @abstractmethod
    def get_model_function(self) -> sax.SType | partial[sax.SType]:
        """Return the model function to use. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement get_model_function")

    def get_results(self) -> SComparisonResults:
        """Helper method to compute S-parameters from qpdk and load Qucs-S reference data.

        Returns:
            Tuple containing:
            - parameter_value: The value of the component parameter used in the model.
            - f: Frequency array in Hz.
            - S_sax: Dictionary of S-parameters from the qpdk model (e.g., {"S11": ..., "S21": ...}).
            - S_qucs: Dictionary of S-parameters from Qucs-S reference data.
        """
        S_qucs = pl.read_csv(TEST_DATA_PATH / self.csv_filename)
        f = S_qucs["frequency"].to_jax()

        model_func = self.get_model_function()
        S_matrix = model_func(f=f, **{self.parameter_name: self.parameter_value})

        # Determine number of ports from the S-matrix
        # S-matrix keys are tuples like ("o1", "o1"), ("o2", "o1"), etc.
        port_numbers = set()
        for key in S_matrix:
            # Extract port numbers from keys like ("o1", "o2")
            for port in key:
                if isinstance(port, str) and port.startswith("o"):
                    port_numbers.add(int(port[1:]))

        num_ports = max(port_numbers, default=2)

        # Build dictionaries for S-parameters in Sij format
        # Only check first from diagonal and below diagonal (S11, S21, S31, S41, etc.)
        S_sax_dict = {}
        S_qucs_dict = {}

        for i in range(1, num_ports + 1):
            for j in range(1, i + 1):
                s_param_name = f"S{i}{j}"

                # Get from sax model (uses "oi" notation)
                sax_key = (f"o{i}", f"o{j}")
                if sax_key in S_matrix:
                    S_sax_dict[s_param_name] = S_matrix[sax_key]

                # Get from Qucs CSV (uses S[i,j] notation)
                qucs_real_col = f"r S[{i},{j}]"
                qucs_imag_col = f"i S[{i},{j}]"

                if qucs_real_col in S_qucs.columns and qucs_imag_col in S_qucs.columns:
                    S_qucs_dict[s_param_name] = (
                        S_qucs[qucs_real_col].to_jax()
                        + 1j * S_qucs[qucs_imag_col].to_jax()
                    )

        return (
            self.parameter_value,
            f,
            S_sax_dict,
            S_qucs_dict,
        )

    @pytest.fixture(scope="class")
    def results(self) -> SComparisonResults:
        return self.get_results()

    def test_compare_to_qucs(self, results: SComparisonResults) -> None:
        """Test that S-parameters match Qucs-S results within tolerance.

        Args:
            results: Tuple containing parameter_value, frequency array, S_sax_dict, S_qucs_dict

        Raises:
            AssertionError: If any S-parameter does not match within the specified tolerances.
        """
        _param_value, _f, S_sax_dict, S_qucs_dict = results

        for s_param_name in S_sax_dict:
            if s_param_name in self.skip_values:
                continue

            if s_param_name in S_qucs_dict:
                assert_allclose(
                    S_sax_dict[s_param_name],
                    S_qucs_dict[s_param_name],
                    **NUMERIC_TOLERANCES,
                    err_msg=f"{s_param_name} does not match",
                )

    def plot_comparison(self):
        """Generate comparison plots between qpdk (sax) and Qucs-S models."""
        param_value, f, S_sax_dict, S_qucs_dict = self.get_results()

        loosely_dashed = (1, 2)

        def _plot_s_parameter(s_qucs, s_sax, color, param_name):
            """Helper function to plot S-parameter data with consistent styling."""
            plt.plot(
                jnp.angle(s_qucs),
                abs(s_qucs),
                "-",
                linewidth=1,
                color=color,
                label=f"${param_name}$ Qucs-S",
            )
            plt.plot(
                jnp.angle(s_sax),
                abs(s_sax),
                "--",
                dashes=loosely_dashed,
                linewidth=2.5,
                color=color,
                label=f"${param_name}$ qpdk (sax)",
            )
            return color

        def _plot_magnitude(ax, f, s_qucs, s_sax, color, param_name):
            """Helper function to plot magnitude with consistent styling."""
            ax.plot(
                f / 1e9,
                20 * jnp.log10(abs(s_qucs)),
                "-",
                linewidth=1,
                color=color,
                label=rf"$\|{param_name}\|$ Qucs-S",
            )
            ax.plot(
                f / 1e9,
                20 * jnp.log10(abs(s_sax)),
                "--",
                dashes=loosely_dashed,
                linewidth=2.5,
                color=color,
                label=rf"$\|{param_name}\|$ qpdk (sax)",
            )

        def _plot_phase(ax, f, s_qucs, s_sax, color, param_name):
            """Helper function to plot phase with consistent styling."""
            ax.plot(
                f / 1e9,
                jnp.angle(s_qucs),
                "-",
                linewidth=1,
                color=color,
                label=f"∠${param_name}$ Qucs-S",
            )
            ax.plot(
                f / 1e9,
                jnp.angle(s_sax),
                "--",
                dashes=loosely_dashed,
                linewidth=2.5,
                color=color,
                label=f"∠${param_name}$ qpdk (sax)",
            )

        def _reset_queue(q: Queue) -> None:
            """Reset a queue with colors, effectively dictate when colors reset in plots."""
            while not q.empty():
                q.get()
            for color in plt.rcParams["axes.prop_cycle"].by_key()["color"]:
                q.put(color)

        # Default color cycler
        color_queue = SimpleQueue()
        _reset_queue(color_queue)

        # Get list of S-parameters to plot (common to both sax and qucs)
        s_params_to_plot = [s_param for s_param in S_sax_dict if s_param in S_qucs_dict]

        if not s_params_to_plot:
            raise ValueError(f"No S-parameters to plot for {self.component_name}")

        plt.figure(figsize=(12, 6))

        # Polar plot of S-parameters
        plt.subplot(121, projection="polar")

        for s_param in s_params_to_plot:
            latex_name = f"S_{{{s_param[1:]}}}"
            _plot_s_parameter(
                S_qucs_dict[s_param], S_sax_dict[s_param], color_queue.get(), latex_name
            )

        plt.title(f"{self.component_name} S-parameters")
        plt.legend()

        # Magnitude and phase vs frequency
        ax1 = plt.subplot(122)

        _reset_queue(color_queue)
        for s_param in s_params_to_plot:
            latex_name = f"S_{{{s_param[1:]}}}"
            _plot_magnitude(
                ax1,
                f,
                S_qucs_dict[s_param],
                S_sax_dict[s_param],
                color_queue.get(),
                latex_name,
            )

        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()

        # Use different colors from the cycler for phase plots
        for s_param in s_params_to_plot:
            latex_name = f"S_{{{s_param[1:]}}}"
            _plot_phase(
                ax2,
                f,
                S_qucs_dict[s_param],
                S_sax_dict[s_param],
                color_queue.get(),
                latex_name,
            )

        ax2.set_ylabel("Phase [rad]")
        ax2.legend(loc="upper right")

        plt.title(
            f"{self.component_name} $S$-parameters ({self.parameter_name}$={param_value / self.parameter_unit:.1f}\\cdot${self.parameter_unit})"
        )
        plt.show()

    def __str__(self) -> str:
        """String representation of the test suite instance."""
        return f"{self.__class__.__name__}(component_name={self.component_name}, csv_filename={self.csv_filename})"


@final
class TestCapacitorCompareToQucs(BaseCompareToQucs):
    """Test suite for comparing capacitor S-parameter models to Qucs-S results."""

    component_name = "Capacitor"
    csv_filename = "capacitor_qucs.csv"
    parameter_value = 60e-15
    parameter_name = "capacitance"
    parameter_unit = 1e-15  # femtofarads

    @override
    def get_model_function(self) -> sax.SType:
        return capacitor


@final
class TestInductorCompareToQucs(BaseCompareToQucs):
    """Test suite for comparing inductor S-parameter models to Qucs-S results."""

    component_name = "Inductor"
    csv_filename = "inductor_qucs.csv"
    parameter_value = 10e-9
    parameter_name = "inductance"
    parameter_unit = 1e-9  # nanohenries

    @override
    def get_model_function(self) -> sax.SType:
        return inductor


@pytest.mark.skip(reason="S11 does not match at all and S22 has some discrepancies")
@final
class TestCPWCompareToQucs(BaseCompareToQucs):
    """Test suite for comparing coplanar waveguide (CPW) S-parameter models to Qucs-S results."""

    component_name = "Coplanar waveguide"
    csv_filename = "cpw_w10_s_6_l10mm.csv"
    parameter_value = 10000.0  # length in micrometers
    parameter_name = "length"
    parameter_unit = 1e-6  # micrometers
    skip_values: ClassVar[list[str]] = ["S11"]

    @override
    def get_model_function(self) -> partial[sax.SType]:
        return partial(
            straight,
            media=cpw_media_skrf(),
        )


@final
class TestCouplerStraightCompareToQucs(BaseCompareToQucs):
    """Test suite for comparing coupled straight coplanar waveguide S-parameter model to Qucs-S results."""

    component_name = "Coupler Straight"
    csv_filename = "coupler_straight_qucs.csv"
    parameter_value = 10.0  # length in micrometers
    parameter_name = "length"
    parameter_unit = 1e-6  # micrometers
    # skip_values: ClassVar[list[Literal["S11", "S21"]]] = ["S11"]

    @override
    def get_model_function(self) -> partial[sax.SType]:
        return partial(
            coupler_straight,
            media=cpw_media_skrf(),
        )


if __name__ == "__main__":
    # Run the plotting comparison when executed directly
    for test_suite in (
        TestCapacitorCompareToQucs(),
        TestInductorCompareToQucs(),
        TestCPWCompareToQucs(),
        TestCouplerStraightCompareToQucs(),
    ):
        test_suite.plot_comparison()
