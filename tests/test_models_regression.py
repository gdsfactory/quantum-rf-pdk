"""Regression tests for SAX model S-parameter frequency responses.

Tests all SAX models in the PDK with a frequency sweep and checks the results
against stored reference data using ``pytest_regressions``.  This ensures that
model outputs remain consistent across code changes, similar to the approach
used in `gdsfactory/cspdk <https://github.com/gdsfactory/cspdk>`_.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from qpdk.models import models

skip_test_models: set[str] = set()

model_names = sorted(
    name
    for name in models.keys() - skip_test_models
    if not name.startswith("_")
)


@pytest.mark.parametrize("model_name", model_names)
def test_models_with_frequency_sweep(
    model_name: str, ndarrays_regression: NDArraysRegressionFixture
) -> None:
    """Test models with different frequencies to avoid regressions in frequency response."""
    f = [5e9, 6e9, 7e9]
    model = models[model_name]
    s_params = model(f=f)

    arrays_to_check: dict[str, np.ndarray] = {}
    for key, value in sorted(s_params.items()):
        key_str = f"s_{key[0]}_{key[1]}"
        value_np = np.array(value)
        arrays_to_check[f"{key_str}_real"] = np.real(value_np)
        arrays_to_check[f"{key_str}_imag"] = np.imag(value_np)

    ndarrays_regression.check(
        arrays_to_check,
        default_tolerance={"atol": 1e-5, "rtol": 1e-5},
    )
