"""Tests for qpdk.helper.display_dataframe."""

from unittest.mock import patch

import pandas as pd
import pytest

from qpdk.helper import display_dataframe


class TestDisplayDataframe:
    """Tests for the display_dataframe() helper."""

    @staticmethod
    def test_dual_format_repr_html() -> None:
        """Test that the displayed object provides an HTML representation."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"A": ["x", "y"], "B": [1, 2]})
            display_dataframe(pdf)

        mock_display.assert_called_once()
        obj = mock_display.call_args[0][0]
        html = obj._repr_html_()
        assert "<table" in html
        assert "<style" in html

    @staticmethod
    def test_dual_format_repr_latex() -> None:
        """Test that the displayed object provides a LaTeX representation."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"A": ["x", "y"], "B": [1, 2]})
            display_dataframe(pdf)

        mock_display.assert_called_once()
        obj = mock_display.call_args[0][0]
        latex = obj._repr_latex_()
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    @staticmethod
    def test_latex_hides_index() -> None:
        """Test that the LaTeX output does not include a numeric index column."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"X": ["a"]})
            display_dataframe(pdf)

        obj = mock_display.call_args[0][0]
        latex = obj._repr_latex_()
        # pandas to_latex(index=False) should not include "0" as row label
        lines = latex.strip().split("\n")
        data_lines = [line for line in lines if "a" in line]
        for line in data_lines:
            stripped = line.strip()
            assert not stripped.startswith("0")

    @staticmethod
    def test_accepts_polars_dataframe() -> None:
        """Test that display_dataframe also accepts a polars DataFrame."""
        pl = pytest.importorskip("polars")
        pytest.importorskip("pyarrow")

        with patch("IPython.display.display") as mock_display:
            df = pl.DataFrame({"Col": ["a", "b"], "Val": [10, 20]})
            display_dataframe(df)

        mock_display.assert_called_once()
        obj = mock_display.call_args[0][0]
        html = obj._repr_html_()
        latex = obj._repr_latex_()
        assert "Col" in html
        assert "Col" in latex
