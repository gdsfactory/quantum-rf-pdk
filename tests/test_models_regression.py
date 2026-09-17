"""Regression tests for SAX model S-parameter frequency responses.

Tests all SAX models in the PDK with a frequency sweep and checks the results
against stored reference data using ``pytest_regressions``.  This ensures that
model outputs remain consistent across code changes, similar to the approach
used in `gdsfactory/cspdk <https://github.com/gdsfactory/cspdk>`_.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from qpdk.models import models

skip_test_models: set[str] = set()

model_names = sorted(
    name for name in models.keys() - skip_test_models if not name.startswith("_")
)

ATOL = 1e-5
RTOL = 1e-5
FREQUENCIES = [5e9, 6e9, 7e9]


@pytest.fixture(scope="session", autouse=True)
def _require_lfs_reference_files() -> None:
    """Fail with a clear message if the LFS-stored reference files are not pulled.

    The ``.npz`` reference files in ``tests/test_models_regression/`` are stored
    in Git LFS. In a fresh clone or worktree whose LFS objects have not been
    downloaded, they contain pointer text instead of binary data, which makes
    the regression comparison fail with a confusing npz parse error.
    """
    reference_dir = pathlib.Path(__file__).with_suffix("")
    for npz_path in sorted(reference_dir.glob("*.npz")):
        with npz_path.open("rb") as file:
            head = file.read(128)
        # Git LFS pointer text starts with "version " followed by the LFS API
        # URL and then an "oid sha256:..." line; real npz files start with the
        # zip magic bytes instead. (Split checks avoid a URL literal that link
        # checkers would try to resolve.)
        if head.startswith(b"version ") and b"oid sha256:" in head:
            pytest.fail(
                f"{npz_path.name} contains Git LFS pointer text instead of the "
                "actual reference data. Run `git lfs pull` to download the LFS "
                "files, then re-run the tests.",
                pytrace=False,
            )


@pytest.mark.parametrize("model_name", model_names)
def test_models_with_frequency_sweep(
    model_name: str, ndarrays_regression: NDArraysRegressionFixture
) -> None:
    """Test models with different frequencies to avoid regressions in frequency response."""
    f = FREQUENCIES
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
        # The compute environment is float64 (klujax, pulled in transitively
        # via sax, sets jax_enable_x64 at import), and the references are
        # stored as float64. Same-platform recompute still deviates up to
        # ~3.9e-6 (resonator_coupled / quarter_wave_resonator_coupled
        # s_coupling terms, f = 5/6/7 GHz, macOS arm64, jax 0.8.3) — roughly
        # 21% of the atol=1e-5 budget. That residual comes from reference
        # vintage (the stored values were produced by earlier model code and
        # JAX versions), amplified by sharp resonator features; disabling x64
        # moves it by under one percentage point.
        # tests/test_models_tolerance_matrix.py re-measures this margin on
        # every platform and warns as it shrinks.
        default_tolerance={"atol": ATOL, "rtol": RTOL},
    )
