"""Tolerance margin matrix for the model regression references.

Recomputes each model's S-parameters against the same stored ``.npz``
references used by ``test_models_regression.py`` and measures how much of
the tolerance budget (``|actual - ref| / (atol + rtol*|ref|)``) each platform
consumes. Compute and references are both float64 (klujax, imported via sax,
sets ``jax_enable_x64``), yet same-platform recompute typically consumes a
few percent of budget — up to ~20% on sharp coupled-resonator features
(measured macOS arm64, jax 0.8.3) — because the stored references come from
an earlier code/JAX vintage. Cross-platform (XLA/LAPACK) drift across the CI
OS matrix shows up as these margins growing in this test before the
regression test turns red.
"""

from __future__ import annotations

import operator
import pathlib
import warnings

import numpy as np
import pytest

from qpdk import logger
from qpdk.models import models

# Kept consistent with tests/test_models_regression.py, which owns the
# tolerance settings and frequency grid for the regression references.
# Duplicated here because importing constants across top-level test modules
# is fragile under pytest's importlib import mode.
ATOL = 1e-5
RTOL = 1e-5
FREQUENCIES = [5e9, 6e9, 7e9]
model_names = sorted(name for name in models if not name.startswith("_"))

REFERENCE_DIR = pathlib.Path(__file__).parent / "test_models_regression"
# Warn once a model consumes this fraction of its tolerance budget. Measured
# same-platform worst is ~21% of budget (macOS arm64, jax 0.8.3), so 50%
# leaves headroom while still catching cross-platform XLA/LAPACK drift
# before the hard gate in the test below trips.
EARLY_WARNING_BUDGET_FRACTION = 0.5


@pytest.fixture(scope="session", autouse=True)
def _require_lfs_reference_files() -> None:
    """Fail with a clear message if the LFS-stored reference files are not pulled."""
    for npz_path in sorted(REFERENCE_DIR.glob("*.npz")):
        with npz_path.open("rb") as file:
            head = file.read(128)
        # Git LFS pointer text starts with "version " followed by the LFS API
        # URL and then an "oid sha256:" line; real npz files start with the
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
def test_tolerance_margin(model_name: str) -> None:
    """Report each model's consumption of the regression tolerance budget.

    For every S-parameter array and frequency point, computes
    ``|actual - ref| / (atol + rtol*|ref|)`` against the stored references
    (same tolerance as :func:`test_models_regression.test_models_with_frequency_sweep`).
    Fails with a per-model breakdown when the budget is exceeded and warns when
    it is nearly exhausted.
    """
    reference_path = (
        REFERENCE_DIR / f"test_models_with_frequency_sweep_{model_name}_.npz"
    )
    if not reference_path.exists():
        pytest.fail(
            f"Missing reference file {reference_path.name} for tolerance matrix; "
            "the regression references must be regenerated consistently.",
            pytrace=False,
        )
    s_params = models[model_name](f=FREQUENCIES)

    worst_ratio, worst_key, worst_freq_label = 0.0, "", ""
    exceeded: list[tuple[float, str]] = []
    with np.load(reference_path) as reference:
        for key, value in sorted(s_params.items()):
            key_str = f"s_{key[0]}_{key[1]}"
            value_np = np.array(value)
            for suffix, component in (
                ("real", np.real(value_np)),
                ("imag", np.imag(value_np)),
            ):
                actual = np.atleast_1d(component)
                array_key = f"{key_str}_{suffix}"
                if array_key not in reference:
                    pytest.fail(
                        f"{array_key} missing in reference {reference_path.name}; "
                        "the model outputs changed since the references were "
                        "generated",
                        pytrace=False,
                    )
                ref = np.atleast_1d(reference[array_key])
                if actual.shape != ref.shape:
                    pytest.fail(
                        f"{array_key} shape {actual.shape} does not match "
                        f"reference shape {ref.shape} in {reference_path.name}; "
                        "the model outputs changed shape since the references "
                        "were generated",
                        pytrace=False,
                    )
                # Mirror np.isclose(..., equal_nan=True), which the regression
                # test uses: NaN == NaN passes, but NaN vs finite must fail (a
                # plain ratio would be NaN and silently compare False against
                # the 1.0 gate).
                nan_mismatch = np.isnan(actual) ^ np.isnan(ref)
                actual_f = np.where(np.isnan(actual), 0.0, actual)
                ref_f = np.where(np.isnan(ref), 0.0, ref)
                ratios = np.abs(actual_f - ref_f) / (ATOL + RTOL * np.abs(ref_f))
                ratios = np.where(nan_mismatch, np.inf, ratios)
                index = int(np.argmax(ratios))
                ratio = float(ratios[index])
                # Frequency-independent (0-d) references have no frequency
                # point to report.
                freq_label = (
                    "" if component.ndim == 0 else f" at f={FREQUENCIES[index]:.1e} Hz"
                )
                if ratio > 1.0:
                    if nan_mismatch[index]:
                        message = f"{array_key}: NaN mismatch vs reference{freq_label}"
                    else:
                        diff = abs(actual_f[index] - ref_f[index])
                        budget = ATOL + RTOL * np.abs(ref_f[index])
                        message = (
                            f"{array_key}: {ratio:.0%} of budget (|actual - ref| = "
                            f"{diff:.3e} vs allowed {budget:.3e}){freq_label}"
                        )
                    exceeded.append((ratio, message))
                if ratio > worst_ratio:
                    worst_ratio, worst_key, worst_freq_label = (
                        ratio,
                        array_key,
                        freq_label,
                    )

    if worst_key:
        worst_detail = f"({worst_key}{worst_freq_label})"
    else:
        worst_detail = "(all points matched the references exactly)"
    logger.info(
        f"Tolerance margin {model_name}: worst {worst_ratio:.1%} of budget "
        f"{worst_detail}"
    )
    if exceeded:
        exceeded.sort(key=operator.itemgetter(0), reverse=True)
        pytest.fail(
            f"{model_name} exceeded the tolerance budget (atol={ATOL}, "
            f"rtol={RTOL}); worst offenders: "
            + "; ".join(text for _, text in exceeded[:5]),
            pytrace=False,
        )
    if worst_ratio >= EARLY_WARNING_BUDGET_FRACTION:
        warnings.warn(
            f"{model_name} consumes {worst_ratio:.1%} of the tolerance budget "
            f"{worst_detail}; margin is "
            "shrinking, re-measure before loosening tolerances",
            stacklevel=1,
        )
