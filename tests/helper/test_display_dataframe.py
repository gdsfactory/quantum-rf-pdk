"""Tests for qpdk.helper.display_dataframe."""

from unittest.mock import patch

import pandas as pd
import pytest

from qpdk.helper import _latex_math_to_html, _latex_to_html, display_dataframe


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

    @staticmethod
    def test_html_converts_latex_subscripts() -> None:
        """Test that $...$ math in cells is converted to HTML subscripts."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"P": ["$E_J$", "$T_{\\mathrm{Purcell}}$"]})
            display_dataframe(pdf)

        obj = mock_display.call_args[0][0]
        html = obj._repr_html_()
        assert "E<sub>J</sub>" in html
        assert "T<sub>Purcell</sub>" in html
        assert "$" not in html

    @staticmethod
    def test_html_converts_greek_letters() -> None:
        """Test that LaTeX Greek letters in cells become Unicode in HTML."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"P": ["$\\kappa$", "$\\omega_r$"]})
            display_dataframe(pdf)

        obj = mock_display.call_args[0][0]
        html = obj._repr_html_()
        assert "κ" in html
        assert "ω<sub>r</sub>" in html

    @staticmethod
    def test_latex_preserves_math_delimiters() -> None:
        """Test that $...$ math passes through to LaTeX output unchanged."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"P": ["$E_J$", "$Q_{\\mathrm{ext}}$"]})
            display_dataframe(pdf)

        obj = mock_display.call_args[0][0]
        latex = obj._repr_latex_()
        assert "$E_J$" in latex
        assert "$Q_{\\mathrm{ext}}$" in latex

    @staticmethod
    def test_plain_text_unchanged() -> None:
        """Test that plain text without $...$ is not altered."""
        with patch("IPython.display.display") as mock_display:
            pdf = pd.DataFrame({"P": ["Resonator length", "10"]})
            display_dataframe(pdf)

        obj = mock_display.call_args[0][0]
        html = obj._repr_html_()
        latex = obj._repr_latex_()
        assert "Resonator length" in html
        assert "Resonator length" in latex


class TestLatexToHtml:
    """Tests for the _latex_to_html and _latex_math_to_html helpers."""

    @staticmethod
    def test_single_char_subscript() -> None:
        assert _latex_math_to_html("E_J") == "E<sub>J</sub>"

    @staticmethod
    def test_braced_subscript() -> None:
        assert _latex_math_to_html("Q_{ext}") == "Q<sub>ext</sub>"

    @staticmethod
    def test_mathrm_subscript() -> None:
        assert _latex_math_to_html("T_{\\mathrm{Purcell}}") == "T<sub>Purcell</sub>"

    @staticmethod
    def test_text_subscript() -> None:
        assert _latex_math_to_html("Q_{\\text{ext}}") == "Q<sub>ext</sub>"

    @staticmethod
    def test_greek_letter() -> None:
        assert _latex_math_to_html("\\kappa") == "κ"

    @staticmethod
    def test_greek_with_subscript() -> None:
        assert _latex_math_to_html("\\omega_t") == "ω<sub>t</sub>"

    @staticmethod
    def test_uppercase_greek() -> None:
        assert _latex_math_to_html("C_\\Sigma") == "C<sub>Σ</sub>"

    @staticmethod
    def test_compound_expression() -> None:
        assert _latex_math_to_html("|\\chi|/\\kappa") == "|χ|/κ"

    @staticmethod
    def test_superscript() -> None:
        assert _latex_math_to_html("x^2") == "x<sup>2</sup>"

    @staticmethod
    def test_braced_superscript() -> None:
        assert _latex_math_to_html("x^{10}") == "x<sup>10</sup>"

    @staticmethod
    def test_dollar_delimiters_stripped() -> None:
        assert _latex_to_html("$E_J$") == "E<sub>J</sub>"

    @staticmethod
    def test_mixed_math_and_text() -> None:
        assert _latex_to_html("$\\omega_q$ (NetKet)") == "ω<sub>q</sub> (NetKet)"

    @staticmethod
    def test_no_math_passthrough() -> None:
        assert _latex_to_html("Resonator length") == "Resonator length"

    @staticmethod
    def test_multiple_math_expressions() -> None:
        result = _latex_to_html("$E_J$ and $E_C$")
        assert result == "E<sub>J</sub> and E<sub>C</sub>"
