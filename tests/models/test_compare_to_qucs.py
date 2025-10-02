"""Tests comparing S-parameter models results to Qucs-S results."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias, final, override

import jax.numpy as jnp
import polars as pl
import pytest
import sax
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from qpdk.models.generic import capacitor, inductor

TEST_DATA_PATH = Path(__file__).parent / "data"

NUMERIC_TOLERANCES = {
    "rtol": 1e-2,
    "atol": 1e-3,
}


class BaseCompareToQucs(ABC):
    """Base class for comparing S-parameter models to Qucs-S results."""

    # Subclasses should override these
    component_name: str = "Component"
    csv_filename: str = "component_qucs.csv"
    parameter_value: float = 0.0
    parameter_name: str = "parameter"
    parameter_unit: float = 1e-9

    SComparisonResults: TypeAlias = tuple[
        float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]

    @abstractmethod
    def get_model_function(self) -> sax.SType:
        """Return the model function to use. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement get_model_function")

    def get_results(self) -> SComparisonResults:
        """Helper method to compute S-parameters from qpdk and load Qucs-S reference data.

        Returns:
            Tuple containing:
            - parameter_value: The value of the component parameter used in the model.
            - f: Frequency array in Hz.
            - S_21_sax: S21 parameter from the qpdk model.
            - S_11_sax: S11 parameter from the qpdk model.
            - S_21_qucs: S21 parameter from Qucs-S reference data.
            - S_11_qucs: S11 parameter from Qucs-S reference data.
        """
        S_qucs = pl.read_csv(TEST_DATA_PATH / self.csv_filename)
        f = S_qucs["frequency"].to_jax()

        model_func = self.get_model_function()
        S_sax = model_func(f=f, **{self.parameter_name: self.parameter_value})

        # Convert real and imaginary parts to JAX complex array
        S_21_qucs = S_qucs["r S[2,1]"].to_jax() + 1j * S_qucs["i S[2,1]"].to_jax()
        S_11_qucs = S_qucs["r S[1,1]"].to_jax() + 1j * S_qucs["i S[1,1]"].to_jax()

        return (
            self.parameter_value,
            f,
            S_sax["o2", "o1"],
            S_sax["o1", "o1"],
            S_21_qucs,
            S_11_qucs,
        )

    @pytest.fixture(scope="class")
    def results(self) -> SComparisonResults:
        return self.get_results()

    def test_compare_to_qucs(self, results: SComparisonResults):
        """Test that S-parameters match Qucs-S results within tolerance."""
        _param_value, _f, S_21_sax, S_11_sax, S_21_qucs, S_11_qucs = results
        assert_allclose(S_11_sax, S_11_qucs, **NUMERIC_TOLERANCES)
        assert_allclose(S_21_sax, S_21_qucs, **NUMERIC_TOLERANCES)

    def plot_comparison(self):
        """Generate comparison plots between qpdk (sax) and Qucs-S models."""
        param_value, f, S_21_sax, S_11_sax, S_21_qucs, S_11_qucs = self.get_results()

        def _plot_s_parameter(s_qucs, s_sax, color, param_name):
            """Helper function to plot S-parameter data with consistent styling."""
            plt.plot(
                jnp.angle(s_qucs),
                abs(s_qucs),
                "-",
                linewidth=1.5,
                color=color,
                label=f"${param_name}$ Qucs-S",
            )
            plt.plot(
                jnp.angle(s_sax),
                abs(s_sax),
                "--",
                linewidth=2.5,
                color=color,
                label=f"${param_name}$ qpdk (sax)",
            )
            return color

        def _plot_magnitude(ax, f, s_qucs, s_sax, color, param_name):
            """Helper function to plot magnitude with consistent styling."""
            ax.plot(
                f / 1e9,
                abs(s_qucs),
                "-",
                linewidth=1.5,
                color=color,
                label=rf"$\|{param_name}\|$ Qucs-S",
            )
            ax.plot(
                f / 1e9,
                abs(s_sax),
                "--",
                linewidth=2.5,
                color=color,
                label=rf"$\|{param_name}\|$ qpdk (sax)",
            )

        def _plot_phase(ax, f, s_qucs, s_sax, color, param_name):
            """Helper function to plot phase with consistent styling."""
            ax.plot(
                f / 1e9,
                jnp.angle(s_sax),
                "--",
                linewidth=2.5,
                color=color,
                label=f"∠${param_name}$ qpdk (sax)",
            )
            ax.plot(
                f / 1e9,
                jnp.angle(s_qucs),
                "-",
                linewidth=1.5,
                color=color,
                label=f"∠${param_name}$ Qucs-S",
            )

        # Default color cycler
        color_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Polar plot of S21 and S11
        plt.subplot(121, projection="polar")
        (line1,) = plt.plot(jnp.angle(S_21_qucs), abs(S_21_qucs), "-", linewidth=1.5)
        color1 = line1.get_color()
        _plot_s_parameter(S_21_qucs, S_21_sax, color1, "S_{21}")

        (line2,) = plt.plot(jnp.angle(S_11_qucs), abs(S_11_qucs), "-", linewidth=1.5)
        color2 = line2.get_color()
        _plot_s_parameter(S_11_qucs, S_11_sax, color2, "S_{11}")

        plt.title(f"{self.component_name} S-parameters")
        plt.legend()

        # Magnitude and phase vs frequency
        ax1 = plt.subplot(122)
        (line3,) = ax1.plot(f / 1e9, abs(S_21_qucs), "-", linewidth=1.5)
        color3 = line3.get_color()
        _plot_magnitude(ax1, f, S_21_qucs, S_21_sax, color3, "S_{21}")

        (line4,) = ax1.plot(f / 1e9, abs(S_11_qucs), "-", linewidth=1.5)
        color4 = line4.get_color()
        _plot_magnitude(ax1, f, S_11_qucs, S_11_sax, color4, "S_{11}")

        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("Magnitude [unitless]")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        _plot_phase(ax2, f, S_21_qucs, S_21_sax, color_cycler[2], "S_{21}")
        _plot_phase(ax2, f, S_11_qucs, S_11_sax, color_cycler[3], "S_{11}")

        ax2.set_ylabel("Phase [rad]")
        ax2.legend(loc="upper right")

        plt.title(
            f"{self.component_name} $S$-parameters ({self.parameter_name}$={param_value / self.parameter_unit:.1f}\\cdot${self.parameter_unit})"
        )
        plt.show()


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


if __name__ == "__main__":
    # Run the plotting comparison when executed directly
    for test_suite in (TestCapacitorCompareToQucs(), TestInductorCompareToQucs()):
        test_suite.plot_comparison()
