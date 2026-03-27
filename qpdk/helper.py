"""Helper functions for the qpdk package."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

from gdsfactory import Component, ComponentAllAngle, LayerEnum, get_component
from gdsfactory.technology import LayerViews
from gdsfactory.typings import ComponentAllAngleSpec, ComponentSpec, Layer


def deprecated(msg: str | Callable | None = None) -> Any:
    """Decorator to mark functions as deprecated.

    Can be used as @deprecated or @deprecated("custom message").
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            m = (
                msg
                if isinstance(msg, str)
                else f"{func.__name__} is deprecated and will be removed in a future version."
            )
            warnings.warn(m, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    if callable(msg):
        f = msg
        msg = None
        return decorator(f)
    return decorator


def denest_layerviews_to_layer_tuples(
    layer_views: LayerViews,
) -> dict[str, tuple[int, int]]:
    """De-nest LayerViews into a flat dictionary of layer names to layer tuples.

    Args:
        layer_views: LayerViews object containing the layer views.

    Returns:
        Dictionary mapping layer names to their corresponding (layer, datatype) tuples.
    """

    def denest_layer_dict_recursive(items: dict) -> dict:
        """Recursively denest layer views to any depth.

        Args:
            items: Dictionary of layer view items to process

        Returns:
            Dictionary mapping layer names to layer objects
        """
        layers = {}

        for key, value in items.items():
            if value.group_members:
                # Recursively process nested group members and merge results
                nested_layers = denest_layer_dict_recursive(value.group_members)
                layers.update(nested_layers)
            # Base case: add the layer to our dictionary
            elif hasattr(value, "layer"):
                layers[key] = value.layer

        return layers

    # Start the recursive denesting process and return the result
    return denest_layer_dict_recursive(layer_views.layer_views)


def show_components(
    *args: ComponentSpec | ComponentAllAngleSpec,
    spacing: int = 200,
) -> Sequence[Component]:
    """Show sequence of components in a single layout in a line.

    The components are spaced based on the maximum width and height of the components.

    Args:
        *args: Component specifications to show.
        spacing: Extra spacing between components.

    Returns:
        Components after :func:`gdsfactory.get_component`.
    """
    from qpdk import PDK  # noqa: PLC0415

    PDK.activate()

    components = [get_component(component_spec) for component_spec in args]
    any_all_angle = any(
        isinstance(component, ComponentAllAngle) for component in components
    )

    c = ComponentAllAngle() if any_all_angle else Component()

    max_component_width = max(component.size_info.width for component in components)
    max_component_height = max(component.size_info.height for component in components)
    if max_component_width > max_component_height:
        shift = (0, max_component_height + spacing)
    else:
        shift = (max_component_width + spacing, 0)

    for i, component in enumerate(components):
        (c << component).move((
            shift[0] * i,
            shift[1] * i,
        ))
        label_offset = (
            shift[0] * i + (component.size_info.width / 2),
            shift[1] * i + (component.size_info.height / 2),
        )
        label_text = component.name if hasattr(component, "name") else f"component_{i}"
        c.add_label(
            text=label_text,
            position=label_offset,
            layer=cast("LayerEnum", PDK.layers).TEXT,
        )
    c.show()

    return components


def layerenum_to_tuple(layerenum: LayerEnum) -> Layer:
    """Convert a LayerEnum object to a tuple containing layer and datatype values.

    Args:
        layerenum: The LayerEnum object to convert.
    """
    return layerenum.layer, layerenum.datatype


_GREEK_MAP: dict[str, str] = {
    r"\varepsilon": "ε",
    r"\varphi": "φ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "φ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
}


def _latex_math_to_html(expr: str) -> str:
    r"""Convert a single LaTeX math expression (without ``$`` delimiters) to HTML.

    Handles ``\mathrm``/``\text`` commands, Greek letters, and sub-/superscripts.
    """
    # Strip \mathrm{...} and \text{...} wrappers, keeping their content
    expr = re.sub(r"\\(?:mathrm|text|textrm)\{([^}]+)\}", r"\1", expr)

    # Replace Greek letter commands with Unicode (longest names first)
    for latex, char in sorted(_GREEK_MAP.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(latex, char)

    # Subscripts: _{...} then _X
    expr = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", expr)
    expr = re.sub(r"_(\w)", r"<sub>\1</sub>", expr)

    # Superscripts: ^{...} then ^X
    expr = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", expr)
    return re.sub(r"\^(\w)", r"<sup>\1</sup>", expr)


def _latex_to_html(text: str) -> str:
    r"""Convert ``$...$`` delimited LaTeX math in *text* to HTML.

    Non-math text is returned unchanged.
    """
    return re.sub(r"\$([^$]+)\$", lambda m: _latex_math_to_html(m.group(1)), text)


def display_dataframe(df: pd.DataFrame | pl.DataFrame) -> None:
    """Display a DataFrame with both HTML and LaTeX representations.

    Wraps a polars or pandas :class:`~pandas.DataFrame` in an object that
    provides both ``_repr_html_`` (styled, index-hidden) and
    ``_repr_latex_`` representations so that Jupyter Book renders a proper
    table in both HTML and PDF outputs.

    Cell values may contain ``$...$`` delimited LaTeX math.  The HTML
    representation converts these to Unicode/HTML (subscripts, Greek
    letters, etc.) while the LaTeX representation passes them through as
    native math.

    Args:
        df: A polars or pandas DataFrame to display.
    """
    import pandas as pd  # noqa: PLC0415
    from IPython.display import display  # noqa: PLC0415

    # Convert polars DataFrame to pandas if needed
    pdf: pd.DataFrame = df.to_pandas() if hasattr(df, "to_pandas") else df

    class _DualFormatTable:
        """Table object providing both HTML and LaTeX representations."""

        def _repr_html_(self) -> str:
            html_df = pdf.copy()
            for col in html_df.columns:
                if pd.api.types.is_string_dtype(html_df[col]):
                    html_df[col] = html_df[col].map(
                        lambda x: _latex_to_html(x) if isinstance(x, str) else x
                    )
            return html_df.style.hide(axis="index")._repr_html_()

        def _repr_latex_(self) -> str:
            return pdf.to_latex(index=False, escape=False)

    display(_DualFormatTable())
